"""Cross-endpoint comparison for the MCP readability check.

The per-endpoint scorers score each endpoint in isolation. This module adds the
*comparison* layer: it groups the scored endpoint rows by product, treats every
endpoint of a product as an alternative *implementation* (e.g. its MCP Toolbox
tools vs its one-MCP endpoint), and computes **union-based completeness** -- each
implementation scored against the union of capabilities exposed by *any*
implementation of that product. No per-product ground-truth manifest is needed.

It writes the completeness columns back onto each endpoint row (so they ride the
normal EVALS output) and returns one comparison score row per product for the
scores file. It is intentionally generic over N implementations per product; a
product with a single implementation degrades to 100% coverage with an empty diff.

Capability alignment across implementations (whose tool names differ) is done
either deterministically from each endpoint's ``mcp_readability_capabilities``
signature (the default, offline-safe path) or, when a model is supplied, by an
LLM that assigns each tool a canonical capability label. The LLM path falls back
to the deterministic one on any failure so a run never aborts on alignment alone.
"""

import json
import logging
import re

# Base identity column used to group implementations of the same product.
PRODUCT_COLUMN = "mcp_readability_product_name"
IMPLEMENTATION_COLUMN = "mcp_readability_implementation"
# Deterministic per-endpoint capability signature (JSON list) emitted by
# ``McpToolMetricsScorer``; the deterministic alignment fallback reads it.
CAPABILITIES_COLUMN = "mcp_readability_capabilities"

# Completeness columns written back onto each endpoint row.
COMPLETENESS_COLUMNS = [
    "mcp_readability_completeness_percent",
    "mcp_readability_capabilities_covered",
    "mcp_readability_union_capability_count",
    "mcp_readability_unique_capabilities",
    "mcp_readability_missing_capabilities",
]

# Comparator name for the per-product completeness score row.
COMPLETENESS_COMPARATOR = "mcp_completeness"


def compare_products(endpoint_results, model=None) -> list[dict]:
    """Add completeness columns to rows and return per-product score rows.

    Args:
      endpoint_results: list of ``(row, tools)`` -- each scored endpoint row plus
        the ``mcp.types.Tool`` list it was scored from (used only for LLM
        alignment; the deterministic path reads the row's capability signature).
      model: optional generator for LLM-based capability alignment; ``None`` uses
        the deterministic signature.

    Returns:
      One score row per product (``comparator = mcp_completeness``).
    """
    groups: dict[str, list] = {}
    for row, tools in endpoint_results:
        product = row.get(PRODUCT_COLUMN, "") or ""
        groups.setdefault(product, []).append((row, tools))

    score_rows = []
    for product, members in groups.items():
        score_rows.append(_compare_group(product, members, model))
    return score_rows


def _compare_group(product: str, members: list, model=None) -> dict:
    """Compute union completeness for one product's implementations."""
    caps_by_member = _capabilities_for_group(members, model)

    union: set[str] = set()
    for caps in caps_by_member:
        union |= caps
    union_count = len(union)

    per_impl_pct = []
    for (row, _tools), caps in zip(members, caps_by_member):
        others: set[str] = set()
        for other in caps_by_member:
            if other is not caps:
                others |= other
        covered = len(caps & union)
        completeness = round(covered / union_count * 100, 2) if union_count else 100.0
        row["mcp_readability_completeness_percent"] = completeness
        row["mcp_readability_capabilities_covered"] = covered
        row["mcp_readability_union_capability_count"] = union_count
        # Capabilities only this implementation has (vs its peers), and those it
        # is missing relative to the union.
        row["mcp_readability_unique_capabilities"] = json.dumps(
            sorted(caps - others)
        )
        row["mcp_readability_missing_capabilities"] = json.dumps(
            sorted(union - caps)
        )
        per_impl_pct.append(
            (row.get(IMPLEMENTATION_COLUMN, "") or "?", completeness)
        )

    # Parity pass: every implementation covers the full union (none is missing a
    # capability a sibling exposes).
    at_parity = all(pct >= 100.0 for _, pct in per_impl_pct)
    logs = "; ".join(
        f"{impl}={pct}%" for impl, pct in sorted(per_impl_pct)
    )
    return {
        "id": product or "(unknown product)",
        "comparator": COMPLETENESS_COMPARATOR,
        "score": 100 if at_parity else 0,
        "comparison_logs": (
            f"union_capabilities={union_count}; completeness by "
            f"implementation: {logs}"
        ),
        "comparison_error": None,
    }


def _capabilities_for_group(members: list, model=None) -> list[set]:
    """Capability set per member, via the LLM aligner or the deterministic one."""
    if model is not None:
        try:
            return _llm_capabilities(members, model)
        except Exception as e:  # never abort a run on alignment alone
            logging.warning(
                "mcp_readability comparison: LLM alignment failed (%s); "
                "falling back to deterministic capability signatures.",
                e,
            )
    return [_deterministic_capabilities(row) for row, _tools in members]


def _deterministic_capabilities(row: dict) -> set:
    """Read the endpoint's capability signature written by the metrics scorer."""
    raw = row.get(CAPABILITIES_COLUMN)
    if not raw:
        return set()
    try:
        caps = json.loads(raw)
    except (TypeError, ValueError):
        return set()
    return {str(c) for c in caps if c}


# ----------------------------------------------------------------------
# LLM alignment
# ----------------------------------------------------------------------
_ALIGN_PROMPT = """You are aligning tools across alternative implementations of
the same product's API so equivalent operations map to one canonical capability.

For every tool below, output a short canonical capability label in the form
`<action>_<resource>` (e.g. `list_instances`, `get_backup`, `create_cluster`).
Tools that perform the SAME operation in different implementations MUST get the
SAME label, regardless of naming, prefixes, or casing.

### TOOLS
{tools}

### OUTPUT
Return ONLY a JSON object mapping each tool's id (the `id` field below) to its
canonical capability label, with no prose:
{{"<id>": "<action>_<resource>", ...}}"""


def _llm_capabilities(members: list, model) -> list[set]:
    """Assign each tool a canonical capability label via the model, then union.

    Every tool across all implementations is labeled in one call so equivalent
    operations collapse to the same label; each implementation's capability set is
    the set of labels of its tools.
    """
    entries = []  # (member_index, tool_id, name, description)
    for idx, (_row, tools) in enumerate(members):
        for tool in tools or []:
            name = _tool_name(tool)
            tool_id = f"{idx}::{name}"
            entries.append((idx, tool_id, name, _tool_description(tool)))

    if not entries:
        return [set() for _ in members]

    listing = "\n".join(
        f"- id={tid} | name={name} | {desc[:200]}"
        for _idx, tid, name, desc in entries
    )
    raw = model.generate(_ALIGN_PROMPT.format(tools=listing))
    mapping = _extract_json_object(raw)

    caps: list[set] = [set() for _ in members]
    for idx, tid, name, _desc in entries:
        label = mapping.get(tid) or mapping.get(name)
        if label:
            caps[idx].add(str(label).strip().lower())
    # Guard: if the model returned nothing usable, signal failure so the caller
    # falls back to the deterministic path rather than reporting empty coverage.
    if not any(caps):
        raise ValueError("LLM alignment produced no capability labels")
    return caps


def _extract_json_object(text: str) -> dict:
    """Pull a JSON object out of a model response (handles code fences)."""
    if not text:
        raise ValueError("empty model response")
    text = text.strip()
    fence = re.match(r"^```(?:json)?\s*(.*?)\s*```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    try:
        obj = json.loads(text)
    except json.JSONDecodeError:
        start, end = text.find("{"), text.rfind("}")
        if start != -1 and end > start:
            obj = json.loads(text[start:end + 1])
        else:
            raise ValueError("no JSON object found in model response")
    if not isinstance(obj, dict):
        raise ValueError("model response was not a JSON object")
    return obj


def _tool_name(tool) -> str:
    name = getattr(tool, "name", None)
    if name is None and isinstance(tool, dict):
        name = tool.get("name")
    return str(name or "")


def _tool_description(tool) -> str:
    desc = getattr(tool, "description", None)
    if desc is None and isinstance(tool, dict):
        desc = tool.get("description")
    return " ".join(str(desc or "").split())
