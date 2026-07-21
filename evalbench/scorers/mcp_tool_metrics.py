"""Deterministic metrics scorer for an MCP endpoint's tool listing.

This is the *metrics* half of the MCP readability check. Unlike the LLM style
judge, it computes purely deterministic, model-independent numbers over the tools
returned by ``McpToolsGenerator``:

  - ``total_tools``: the number of tools exposed by the endpoint.
  - ``estimated_tokens``: a rough token footprint of the tool definitions,
    approximated as ``len(JSON(tool)) / 4`` summed across tools.
  - ``token_budget_used_percent``: that estimate as a percentage of the
    configured ``token_budget`` (``None`` when no positive budget is set).
  - ``avg_tokens_per_tool``: ``estimated_tokens / total_tools`` (0 when empty).
  - ``efficiency_tools_per_1k_tokens``: capabilities delivered per 1k tokens of
    context, i.e. ``total_tools / (estimated_tokens / 1000)`` -- the headline
    "how much API per token" number for the Toolbox-vs-one-MCP comparison.
  - ``capabilities``: the endpoint's normalized capability signature -- a sorted,
    de-duplicated list of ``<action>_<resource>`` keys derived from tool names.
    The comparison phase unions these across implementations of the same product
    to compute completeness (and uses them as the deterministic alignment
    fallback when no LLM is available).

It is kept separate from the LLM judge so the metric logic stays deterministic
and independently testable. It is a plug-and-play mcp_readability scorer (see
``scorers.mcp_readability_scoring``): the orchestrator calls :meth:`run` with the
per-endpoint context. Its binary summary metric is "within token budget".
"""

from collections.abc import Sequence
import json
import re
from typing import Any

from scorers.mcp_readability_scoring import EndpointContext, ScoreContribution

# Rough chars-per-token heuristic for estimating a tool's token footprint.
_CHARS_PER_TOKEN = 4

# Action-verb synonym map: collapses equivalent verbs to a canonical action so
# the same capability matches across implementations that name it differently
# (e.g. ``describe_instance`` and ``get_instance`` are both a "get instance").
_ACTION_SYNONYMS = {
    "list": "list",
    "get": "get",
    "describe": "get",
    "fetch": "get",
    "read": "get",
    "show": "get",
    "create": "create",
    "add": "create",
    "insert": "create",
    "new": "create",
    "delete": "delete",
    "remove": "delete",
    "drop": "delete",
    "destroy": "delete",
    "update": "update",
    "patch": "update",
    "modify": "update",
    "edit": "update",
    "set": "update",
    "execute": "execute",
    "run": "execute",
    "query": "query",
}


def _split_tokens(name: str) -> list[str]:
    """Split a tool name into lowercase word tokens (snake_case + camelCase)."""
    if not name:
        return []
    # Break camelCase / PascalCase into separate words, then split on non-alnum.
    spaced = re.sub(r"(?<=[a-z0-9])(?=[A-Z])", " ", name)
    parts = re.split(r"[^a-zA-Z0-9]+", spaced)
    return [p.lower() for p in parts if p]


def normalize_capability(tool_name: str) -> str:
    """Normalize a tool name into an ``<action>_<resource>`` capability key.

    Collapses verb synonyms, singularizes a trailing-``s`` resource, and drops a
    leading action token that is a known verb so ``list_datasets`` and
    ``datasets_list`` (and camelCase variants) map to the same key. Names with no
    recognizable verb fall back to the joined, normalized tokens so distinct tools
    still produce distinct capabilities.
    """
    tokens = _split_tokens(tool_name)
    if not tokens:
        return ""

    action = None
    if tokens[0] in _ACTION_SYNONYMS:
        action = _ACTION_SYNONYMS[tokens[0]]
        resource_tokens = tokens[1:]
    elif tokens[-1] in _ACTION_SYNONYMS:
        action = _ACTION_SYNONYMS[tokens[-1]]
        resource_tokens = tokens[:-1]
    else:
        resource_tokens = tokens

    resource = "_".join(resource_tokens)
    if resource.endswith("s") and len(resource) > 1:
        resource = resource[:-1]  # naive singularize (datasets -> dataset)

    if action and resource:
        return f"{action}_{resource}"
    return action or resource


def capability_signature(tools: Sequence[Any]) -> list[str]:
    """Sorted, de-duplicated capability keys for a list of tools."""
    caps = {normalize_capability(_tool_name(t)) for t in tools}
    caps.discard("")
    return sorted(caps)


def _tool_name(tool: Any) -> str:
    """Best-effort tool name for both ``mcp.types.Tool`` and plain dicts."""
    name = getattr(tool, "name", None)
    if name is None and isinstance(tool, dict):
        name = tool.get("name")
    return str(name or "")


class McpToolMetricsScorer:
    """Computes deterministic size/cost metrics for a list of MCP tools."""

    # Result-row columns this scorer contributes.
    COLUMNS = [
        "mcp_readability_total_tools",
        "mcp_readability_estimated_tokens",
        "mcp_readability_token_budget_used_percent",
        "mcp_readability_avg_tokens_per_tool",
        "mcp_readability_efficiency_tools_per_1k_tokens",
        "mcp_readability_capabilities",
    ]

    def __init__(self, config: dict | None = None, global_models=None):
        # Accept global_models for signature parity with the other scorers even
        # though this deterministic scorer needs no model.
        self.name = "mcp_tool_metrics"
        config = config or {}
        # Token budget from this scorer's own config block; a per-call value
        # passed to score() still wins.
        self.token_budget = config.get("token_budget")

    def run(self, context: EndpointContext) -> ScoreContribution:
        """Evaluate one endpoint: compute metrics, pass iff within budget.

        An endpoint may override the configured budget with its own
        ``token_budget`` field.
        """
        budget = context.endpoint.get("token_budget", self.token_budget)
        metrics = self.score(context.tools, budget)
        used = metrics["token_budget_used_percent"]
        # No positive budget configured -> nothing to exceed -> pass.
        within_budget = used is None or used <= 100.0
        return ScoreContribution(
            row_fields={
                "mcp_readability_total_tools": metrics["total_tools"],
                "mcp_readability_estimated_tokens": metrics["estimated_tokens"],
                "mcp_readability_token_budget_used_percent": used or 0.0,
                "mcp_readability_avg_tokens_per_tool": (
                    metrics["avg_tokens_per_tool"]
                ),
                "mcp_readability_efficiency_tools_per_1k_tokens": (
                    metrics["efficiency_tools_per_1k_tokens"]
                ),
                "mcp_readability_capabilities": json.dumps(
                    metrics["capabilities"]
                ),
            },
            score=100 if within_budget else 0,
            logs=(
                f"total_tools={metrics['total_tools']}, "
                f"estimated_tokens={metrics['estimated_tokens']}, "
                f"token_budget_used_percent={used}, "
                f"efficiency_tools_per_1k_tokens="
                f"{metrics['efficiency_tools_per_1k_tokens']}"
            ),
        )

    def score(
        self, tools: Sequence[Any], token_budget: int | None = None
    ) -> dict:
        """Compute metrics over ``tools``.

        Args:
          tools: The endpoint's tools (``mcp.types.Tool`` objects or plain
            dicts in the raw ``tools/list`` shape).
          token_budget: Overrides the budget from config when provided.

        Returns:
          ``{"total_tools", "estimated_tokens", "token_budget_used_percent",
          "avg_tokens_per_tool", "efficiency_tools_per_1k_tokens",
          "capabilities"}``. ``token_budget_used_percent`` is ``None`` when no
          positive budget is configured.
        """
        budget = token_budget if token_budget is not None else self.token_budget

        total_tools = len(tools)
        total_chars = sum(len(self._to_json(tool)) for tool in tools)
        estimated_tokens = round(total_chars / _CHARS_PER_TOKEN)

        if budget and budget > 0:
            used_percent = round(estimated_tokens / budget * 100, 2)
        else:
            used_percent = None

        avg_tokens_per_tool = (
            round(estimated_tokens / total_tools, 2) if total_tools else 0.0
        )
        # Tools delivered per 1k tokens of context: higher is more efficient.
        efficiency = (
            round(total_tools / (estimated_tokens / 1000), 2)
            if estimated_tokens
            else 0.0
        )

        return {
            "total_tools": total_tools,
            "estimated_tokens": estimated_tokens,
            "token_budget_used_percent": used_percent,
            "avg_tokens_per_tool": avg_tokens_per_tool,
            "efficiency_tools_per_1k_tokens": efficiency,
            "capabilities": capability_signature(tools),
        }

    @staticmethod
    def _to_json(tool: Any) -> str:
        """Serialize a tool to JSON for the size estimate.

        Handles both ``mcp.types.Tool`` (a pydantic model) and plain dicts.
        ``by_alias`` keeps the wire field names (e.g. ``inputSchema``) and
        ``exclude_none`` drops unset optionals, so the estimate reflects what an
        endpoint actually puts on the wire.
        """
        model_dump_json = getattr(tool, "model_dump_json", None)
        if callable(model_dump_json):
            try:
                return model_dump_json(by_alias=True, exclude_none=True)
            except TypeError:  # older/non-standard pydantic signatures
                return model_dump_json()
        return json.dumps(tool, default=str, sort_keys=True)
