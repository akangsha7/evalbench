"""Deterministic schema-quality scorer for an MCP endpoint's tool listing.

The third static scorer in the MCP readability check (alongside
``mcp_tool_metrics`` and the LLM ``mcp_style_readability`` judge). It inspects the
JSON-Schema of every tool -- fully deterministic, model-independent -- and reports
signals a human reviewer would otherwise eyeball for LLM-consumption quality:

  - ``param_desc_coverage_percent``: parameters carrying a non-empty description
    (an agent should not need external docs to fill a parameter).
  - ``enum_param_count``: parameters that declare a closed set via ``enum``.
  - ``list_ops_with_pagination_percent``: of the list-style tools, how many expose
    pagination (``page_size``/``page_token``/``limit``/``offset``/...), so an agent
    can page instead of requesting an unbounded response.
  - ``annotation_coverage_percent``: tools declaring a behavior hint
    (``readOnlyHint``/``destructiveHint``), so an agent knows which calls mutate.
  - ``avg_required_params``: mean required-parameter count per tool (call burden).

It is a plug-and-play mcp_readability scorer (see
``scorers.mcp_readability_scoring``): the orchestrator calls :meth:`run` with the
per-endpoint context. Its binary summary metric is "parameter description
coverage meets the configured minimum".
"""

from collections.abc import Mapping, Sequence
from typing import Any

from scorers.mcp_readability_scoring import EndpointContext, ScoreContribution
from scorers.mcp_tool_metrics import normalize_capability

# Parameter names (normalized to lowercase, non-alnum stripped) that signal an
# operation exposes pagination.
_PAGINATION_PARAMS = frozenset(
    {
        "pagesize",
        "pagetoken",
        "limit",
        "offset",
        "cursor",
        "maxresults",
        "nexttoken",
        "nextpagetoken",
        "pageindex",
        "pagenumber",
    }
)

# Default minimum parameter-description coverage for the binary pass.
_DEFAULT_MIN_DESC_COVERAGE = 90.0


def _norm_param(name: str) -> str:
    return "".join(ch for ch in str(name).lower() if ch.isalnum())


def _input_schema(tool: Any) -> Mapping[str, Any]:
    schema = getattr(tool, "inputSchema", None)
    if schema is None and isinstance(tool, dict):
        schema = tool.get("inputSchema")
    return schema if isinstance(schema, Mapping) else {}


def _tool_annotations(tool: Any) -> Mapping[str, Any]:
    ann = getattr(tool, "annotations", None)
    if ann is None and isinstance(tool, dict):
        ann = tool.get("annotations")
    if ann is None:
        return {}
    if isinstance(ann, Mapping):
        return ann
    # pydantic ToolAnnotations model -> dict.
    dump = getattr(ann, "model_dump", None)
    if callable(dump):
        try:
            return dump(exclude_none=True)
        except TypeError:
            return dump()
    return {}


def _resolve_ref(
    schema: Mapping[str, Any], defs: Mapping[str, Any]
) -> tuple[Mapping[str, Any], str | None]:
    """Resolve a ``$ref`` into ``$defs`` (returns the schema unchanged if none)."""
    ref = schema.get("$ref") if isinstance(schema, Mapping) else None
    if ref and isinstance(ref, str) and ref.startswith("#/$defs/"):
        name = ref.split("/")[-1]
        if name in defs:
            return defs[name], name
    return schema, None


def _walk_params(
    schema: Mapping[str, Any],
    defs: Mapping[str, Any],
    visited: frozenset,
    stats: dict,
) -> None:
    """Recursively accumulate parameter counts into ``stats``.

    ``stats`` collects ``total`` (parameter count), ``described`` (params with a
    non-empty description), and ``enums`` (params with an ``enum``). Guards
    against self-referential schemas via ``visited`` ($def names already on the
    current path).
    """
    schema, _ = _resolve_ref(schema, defs)
    properties = schema.get("properties")
    if not isinstance(properties, Mapping):
        return

    for prop_schema in properties.values():
        if not isinstance(prop_schema, Mapping):
            stats["total"] += 1
            continue
        prop_schema, ref_name = _resolve_ref(prop_schema, defs)
        stats["total"] += 1
        if str(prop_schema.get("description", "")).strip():
            stats["described"] += 1
        if isinstance(prop_schema.get("enum"), (list, tuple)):
            stats["enums"] += 1

        if ref_name and ref_name in visited:
            continue  # recursive reference: do not descend
        next_visited = visited | {ref_name} if ref_name else visited

        prop_type = prop_schema.get("type")
        if prop_type == "object" and isinstance(
            prop_schema.get("properties"), Mapping
        ):
            _walk_params(prop_schema, defs, next_visited, stats)
        elif prop_type == "array" and isinstance(
            prop_schema.get("items"), Mapping
        ):
            items, item_ref = _resolve_ref(prop_schema["items"], defs)
            if item_ref and item_ref in visited:
                continue
            item_visited = next_visited | {item_ref} if item_ref else next_visited
            if items.get("type") == "object":
                _walk_params(items, defs, item_visited, stats)


class McpSchemaQualityScorer:
    """Computes deterministic schema-quality metrics for a list of MCP tools."""

    # Result-row columns this scorer contributes.
    COLUMNS = [
        "mcp_readability_param_desc_coverage_percent",
        "mcp_readability_enum_param_count",
        "mcp_readability_list_ops_with_pagination_percent",
        "mcp_readability_annotation_coverage_percent",
        "mcp_readability_avg_required_params",
    ]

    def __init__(self, config: dict | None = None, global_models=None):
        # Accept global_models for signature parity with the other scorers even
        # though this deterministic scorer needs no model.
        self.name = "mcp_schema_quality"
        config = config or {}
        self.min_desc_coverage = float(
            config.get("min_param_desc_coverage", _DEFAULT_MIN_DESC_COVERAGE)
        )

    def run(self, context: EndpointContext) -> ScoreContribution:
        """Evaluate one endpoint; pass iff description coverage meets the min."""
        metrics = self.score(context.tools)
        coverage = metrics["param_desc_coverage_percent"]
        passed = coverage >= self.min_desc_coverage
        return ScoreContribution(
            row_fields={
                "mcp_readability_param_desc_coverage_percent": coverage,
                "mcp_readability_enum_param_count": metrics["enum_param_count"],
                "mcp_readability_list_ops_with_pagination_percent": (
                    metrics["list_ops_with_pagination_percent"]
                ),
                "mcp_readability_annotation_coverage_percent": (
                    metrics["annotation_coverage_percent"]
                ),
                "mcp_readability_avg_required_params": (
                    metrics["avg_required_params"]
                ),
            },
            score=100 if passed else 0,
            logs=(
                f"param_desc_coverage_percent={coverage}, "
                f"enum_param_count={metrics['enum_param_count']}, "
                f"list_ops_with_pagination_percent="
                f"{metrics['list_ops_with_pagination_percent']}, "
                f"annotation_coverage_percent="
                f"{metrics['annotation_coverage_percent']}, "
                f"avg_required_params={metrics['avg_required_params']}"
            ),
        )

    def score(self, tools: Sequence[Any]) -> dict:
        """Compute schema-quality metrics over ``tools``."""
        total_params = 0
        described_params = 0
        enum_params = 0
        total_required = 0
        annotated_tools = 0
        list_ops = 0
        list_ops_paginated = 0

        for tool in tools:
            schema = _input_schema(tool)
            defs = schema.get("$defs") or schema.get("definitions") or {}
            if not isinstance(defs, Mapping):
                defs = {}

            stats = {"total": 0, "described": 0, "enums": 0}
            _walk_params(schema, defs, frozenset(), stats)
            total_params += stats["total"]
            described_params += stats["described"]
            enum_params += stats["enums"]

            required = schema.get("required")
            total_required += len(required) if isinstance(required, list) else 0

            if _tool_annotations(tool):
                annotated_tools += 1

            # Pagination coverage is only meaningful for list-style operations.
            if normalize_capability(_tool_name(tool)).startswith("list"):
                list_ops += 1
                top_props = schema.get("properties")
                names = (
                    {_norm_param(n) for n in top_props}
                    if isinstance(top_props, Mapping)
                    else set()
                )
                if names & _PAGINATION_PARAMS:
                    list_ops_paginated += 1

        total_tools = len(tools)
        return {
            "param_desc_coverage_percent": _pct(described_params, total_params),
            "enum_param_count": enum_params,
            "list_ops_with_pagination_percent": _pct(
                list_ops_paginated, list_ops
            ),
            "annotation_coverage_percent": _pct(annotated_tools, total_tools),
            "avg_required_params": (
                round(total_required / total_tools, 2) if total_tools else 0.0
            ),
        }


def _tool_name(tool: Any) -> str:
    name = getattr(tool, "name", None)
    if name is None and isinstance(tool, dict):
        name = tool.get("name")
    return str(name or "")


def _pct(numerator: int, denominator: int) -> float:
    """Percentage, rounded to 2 dp; 100.0 for an empty denominator (nothing to
    fail)."""
    if denominator <= 0:
        return 100.0
    return round(numerator / denominator * 100, 2)
