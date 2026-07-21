"""Reporter that renders a Toolbox-vs-one-MCP side-by-side comparison.

Consumes the per-endpoint EVALS rows produced by the ``mcp_readability``
orchestrator (each row already carries the token/efficiency, schema-quality,
style, and cross-endpoint completeness columns) and renders a single
human-readable report grouped by product:

  - an aggregate leaderboard across all products / implementations, and
  - a per-product section: a side-by-side metric table plus the capability diff
    (unique to each implementation and the shared/overlapping count).

It is pure rendering -- no model, no live calls -- and only acts on
``STORETYPE.EVALS``. Enable it via the ``reporting:`` block::

    reporting:
      comparison_report:
        output_directory: 'results'

Output (under ``<output_directory>/<job_id>/``):
  - ``comparison.md`` / ``comparison.html`` -- combined: every product plus the
    cross-product leaderboard.
  - ``comparison.<product-slug>.md`` / ``.html`` -- one standalone report per
    product (its own leaderboard + section), so a single run over a list of
    products yields one report each (e.g. ``comparison.alloydb.md``,
    ``comparison.cloud_sql.md``).
"""

import html
import json
import logging
import os
import re
import sys

import pandas as pd

from reporting.report import Reporter, STORETYPE
from scorers.mcp_readability_scoring import SEVERITY_BADGES, severity_tally

_PRODUCT = "mcp_readability_product_name"
_IMPL = "mcp_readability_implementation"

# (column, header, formatter) for the side-by-side + leaderboard metric tables.
_METRIC_COLUMNS = [
    ("mcp_readability_total_tools", "Tools", "int"),
    ("mcp_readability_estimated_tokens", "Est. tokens", "int"),
    ("mcp_readability_avg_tokens_per_tool", "Tokens/tool", "float"),
    ("mcp_readability_efficiency_tools_per_1k_tokens", "Tools/1k tok", "float"),
    ("mcp_readability_completeness_percent", "Completeness %", "float"),
    ("mcp_readability_cuj_coverage_percent", "CUJ coverage %", "float"),
    ("mcp_readability_p0_issues", "P0", "int"),
    ("mcp_readability_p1_issues", "P1", "int"),
    ("mcp_readability_p2_issues", "P2", "int"),
    ("mcp_readability_param_desc_coverage_percent", "Param desc %", "float"),
    ("mcp_readability_enum_param_count", "Enums", "int"),
    ("mcp_readability_avg_required_params", "Avg req. params", "float"),
]


class ComparisonReporter(Reporter):
    """Renders a per-product MCP implementation comparison report."""

    _DEFAULT_OUTPUT_DIR = "results"

    def __init__(self, reporting_config, job_id, run_time):
        super().__init__(reporting_config, job_id, run_time)
        if sys.argv and sys.argv[0].endswith("eval_server.py"):
            self.output_dir = "/tmp_session_files/results"
        else:
            self.output_dir = (reporting_config or {}).get(
                "output_directory", self._DEFAULT_OUTPUT_DIR
            )

    def store(self, results, type: STORETYPE) -> None:
        if type != STORETYPE.EVALS:
            return
        if not isinstance(results, pd.DataFrame) or results.empty:
            logging.warning("ComparisonReporter: no EVALS rows to render.")
            return
        if _PRODUCT not in results.columns:
            logging.warning(
                "ComparisonReporter: %s missing; not an mcp_readability run.",
                _PRODUCT,
            )
            return

        rows = results.to_dict(orient="records")
        directory = os.path.join(self.output_dir, str(self.job_id))
        os.makedirs(directory, exist_ok=True)

        written: list[str] = []
        # Combined report: every product plus the cross-product leaderboard.
        written += self._write("comparison", self._render_markdown(rows), directory)
        # One self-contained report per product (comparison.<slug>.md/.html).
        for product, product_rows in _group_by(rows, _PRODUCT).items():
            markdown = self._render_product_markdown(product, product_rows)
            written += self._write(
                f"comparison.{_slug(product)}", markdown, directory
            )

        logging.info("ComparisonReporter: wrote %s", ", ".join(written))

    def _write(self, stem: str, markdown: str, directory: str) -> list[str]:
        """Write ``<stem>.md`` and ``<stem>.html``; return the two paths."""
        md_path = os.path.join(directory, f"{stem}.md")
        html_path = os.path.join(directory, f"{stem}.html")
        with open(md_path, "w", encoding="utf-8") as f:
            f.write(markdown)
        with open(html_path, "w", encoding="utf-8") as f:
            f.write(_wrap_html(markdown))
        return [md_path, html_path]

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_markdown(self, rows: list[dict]) -> str:
        products = _group_by(rows, _PRODUCT)
        parts = ["# MCP Tool Comparison — Toolbox vs one-MCP", ""]
        parts.append(self._leaderboard(rows))
        parts.append(self._methodology_caveats())
        for product in sorted(products):
            parts.append(self._product_section(product, products[product]))
        parts.append(self._man_page_appendix(rows))
        return "\n".join(parts).rstrip() + "\n"

    def _render_product_markdown(self, product: str, rows: list[dict]) -> str:
        """A standalone report for a single product: its own leaderboard (that
        product's implementations only) plus the full product section."""
        title = product or "(unknown product)"
        parts = [f"# MCP Tool Comparison — {title}", ""]
        parts.append(self._leaderboard(rows))
        parts.append(self._methodology_caveats())
        parts.append(self._product_section(product, rows))
        parts.append(self._man_page_appendix(rows))
        return "\n".join(parts).rstrip() + "\n"

    def _methodology_caveats(self) -> str:
        """How to read the numbers, and where they are and aren't authoritative.

        Static, but essential for honest interpretation: the style-judge counts
        come from a single nondeterministic LLM pass, and both the completeness %
        and CUJ % rest on scoring conventions a reader cannot infer from the
        tables alone. Stating them here keeps a reader from treating every figure
        as an exact, reproducible measurement.
        """
        return "\n".join(
            [
                "## How to read this report",
                "",
                "- **Style findings (P0/P1/P2) come from a single LLM-judge "
                "pass and are non-deterministic.** Re-running can shift exact "
                "counts by a few; treat the *themes* as authoritative and the "
                "exact numbers as approximate. The deterministic metrics (tool "
                "count, tokens, completeness, schema signals) are exact.",
                "- **Each P0/P1/P2 is one finding on one tool.** A rule broken "
                "by several tools is listed under each of them below and "
                "counted once per tool, so a larger tool surface accumulates "
                "more findings for the same design mistake.",
                "- **Completeness % is a weak, name-based signal — prefer CUJ "
                "coverage.** Capabilities are matched by a normalized "
                "`<action>_<resource>` name and unioned across implementations. "
                "It counts *tools*, not real capability: a surface that splits "
                "one job across many fine-grained tools (e.g. 29 specialized "
                "`list_*`/SQL tools) scores far higher than one that exposes a "
                "single general-purpose `execute_sql`, even though the latter can "
                "do the same work. Semantically equivalent tools with different "
                "names (e.g. `get_operation` vs `wait_for_operation`) also count "
                "as distinct. Treat **CUJ coverage** below as the task-level "
                "measure of what each surface can actually accomplish.",
                "- **CUJ coverage % is weighted:** covered = 1, partial = 0.5, "
                "blocked = 0, averaged over the listed Critical User Journeys. It "
                "credits general-purpose tools (a single `execute_sql` can "
                "satisfy many journeys), so it reflects real capability better "
                "than the raw completeness %.",
                "- **Fetch caveat:** hosted endpoints fetched unauthenticated "
                "may expose a different tool set than an authenticated / "
                "instance-scoped deployment.",
                "",
            ]
        )

    def _leaderboard(self, rows: list[dict]) -> str:
        headers = ["Product", "Implementation"] + [
            h for _, h, _ in _METRIC_COLUMNS
        ]
        lines = ["## Leaderboard", "", _md_row(headers), _md_sep(len(headers))]
        for row in sorted(
            rows,
            key=lambda r: (
                str(r.get(_PRODUCT, "")),
                str(r.get(_IMPL, "")),
            ),
        ):
            cells = [
                str(row.get(_PRODUCT, "") or "?"),
                str(row.get(_IMPL, "") or "?"),
            ]
            cells += [_fmt(row.get(col), kind) for col, _, kind in _METRIC_COLUMNS]
            lines.append(_md_row(cells))
        lines.append("")
        return "\n".join(lines)

    def _product_section(self, product: str, impl_rows: list[dict]) -> str:
        lines = [f"## {product or '(unknown product)'}", ""]

        # Side-by-side metric table: one column per implementation.
        impls = [str(r.get(_IMPL, "") or "?") for r in impl_rows]
        header = ["Metric"] + impls
        lines += [_md_row(header), _md_sep(len(header))]
        for col, label, kind in _METRIC_COLUMNS:
            cells = [label] + [_fmt(r.get(col), kind) for r in impl_rows]
            lines.append(_md_row(cells))
        lines.append("")

        # Capability diff.
        lines += self._capability_diff(impl_rows)
        # CUJ coverage matrix (functionality achievable alone or by combining
        # tools) -- only rendered when CUJ data is present.
        lines += self._cuj_matrix(impl_rows)
        # Detailed style findings behind the P0/P1/P2 counts -- only rendered
        # when the LLM judge feedback is present on a row.
        lines += self._style_findings(impl_rows)
        return "\n".join(lines)

    def _style_findings(self, impl_rows: list[dict]) -> list[str]:
        """Render the actual style-judge findings behind the P0/P1/P2 counts.

        Reads each row's ``mcp_readability_llm_feedback_json`` (produced by the
        ``mcp_style_readability`` scorer) and renders, per implementation, the
        judge summary, findings grouped by tool, and any waived rules. This turns
        the headline counts into an auditable list -- every P0/P1/P2 in the
        metric table can be traced to a specific rule + tool. Renders nothing if
        no implementation carries judge feedback (scorer not configured).
        """
        by_impl: list[tuple[str, dict]] = []
        for row in impl_rows:
            impl = str(row.get(_IMPL, "") or "?")
            feedback = _as_dict(row.get("mcp_readability_llm_feedback_json"))
            if feedback:
                by_impl.append((impl, feedback))
        if not by_impl:
            return []

        lines = [
            "### Style findings (detail behind the P0/P1/P2 counts)",
            "",
            "Grouped per tool, as returned by the judge. A rule broken by "
            "several tools is listed under each of them and counts once per "
            "tool in the P0/P1/P2 table above.",
            "",
        ]
        for impl, feedback in by_impl:
            lines.append(f"#### {impl}")
            lines.append("")
            summary = str(feedback.get("summary", "")).strip()
            if summary:
                lines.append(f"_Summary:_ {summary}")
                lines.append("")

            by_tool = [
                e for e in _as_list(feedback.get("findings_by_tool"))
                if isinstance(e, dict) and isinstance(e.get("findings"), list)
            ]
            if not by_tool:
                lines.append("- _No findings_")
                lines.append("")
            for entry in by_tool:
                items = [f for f in entry["findings"] if isinstance(f, dict)]
                tool = str(entry.get("tool", "")).strip() or "?"
                lines.append(f"**{tool} — {severity_tally(items)}**")
                lines.append("")
                for finding in items:
                    lines.extend(_finding_lines(finding))
                lines.append("")

            waived = [
                w
                for w in _as_list(feedback.get("waived"))
                if isinstance(w, dict)
            ]
            lines.append(f"**Waived / allowed exceptions — {len(waived)}**")
            lines.append("")
            if not waived:
                lines.append("- _None_")
            else:
                for w in waived:
                    rule = str(w.get("rule_id", "")).strip() or "(rule)"
                    reason = str(w.get("reason", "")).strip() or "no reason given"
                    entry = f"- **{rule}** — {reason}"
                    if "would_have_violated" in w:
                        flag = "yes" if w.get("would_have_violated") else "no"
                        entry += f" (would have been flagged: {flag})"
                    lines.append(entry)
            lines.append("")
        return lines

    def _man_page_appendix(self, rows: list[dict]) -> str:
        """Render the tool man-page for each endpoint as a report appendix.

        The man-page is the exact tool surface the metrics and findings were
        computed from, so attaching it lets a reviewer verify the report against
        primary evidence instead of a hand-written summary. Renders nothing if no
        row carries a man-page (older runs predating persistence).
        """
        entries: list[tuple[str, str, str]] = []  # (product, impl, man_page)
        for row in rows:
            man_page = str(row.get("mcp_readability_man_page", "") or "").strip()
            if not man_page:
                continue
            product = str(row.get(_PRODUCT, "") or "(unknown product)")
            impl = str(row.get(_IMPL, "") or "?")
            entries.append((product, impl, man_page))
        if not entries:
            return ""

        entries.sort(key=lambda e: (e[0], e[1]))
        lines = [
            "## Appendix — tool man pages",
            "",
            "The exact tool surface each metric and finding above was computed "
            "from, for independent verification.",
            "",
        ]
        for product, impl, man_page in entries:
            lines.append(f"### {product} — {impl}")
            lines.append("")
            lines.append("```")
            lines.append(man_page)
            lines.append("```")
            lines.append("")
        return "\n".join(lines)

    def _cuj_matrix(self, impl_rows: list[dict]) -> list[str]:
        """Per-CUJ status matrix across a product's implementations.

        Reads each row's ``mcp_readability_cuj_results_json``. Renders nothing if
        no implementation carries CUJ results (scorer not configured / no CUJs).
        """
        impls = [str(r.get(_IMPL, "") or "?") for r in impl_rows]
        # cuj text -> {impl: status}
        per_cuj: dict[str, dict[str, str]] = {}
        order: list[str] = []
        any_data = False
        for impl, row in zip(impls, impl_rows):
            for entry in _as_list(row.get("mcp_readability_cuj_results_json")):
                if not isinstance(entry, dict):
                    continue
                any_data = True
                cuj = str(entry.get("cuj", "")).strip()
                if not cuj:
                    continue
                if cuj not in per_cuj:
                    per_cuj[cuj] = {}
                    order.append(cuj)
                per_cuj[cuj][impl] = str(entry.get("status", "")).strip().lower()
        if not any_data:
            return []

        glyph = {"covered": "✅", "partial": "🟡", "blocked": "❌"}
        lines = [
            "### CUJ coverage (achievable alone or by combining tools)",
            "",
            "✅ covered · 🟡 partial · ❌ blocked",
            "",
            _md_row(["Critical User Journey"] + impls),
            _md_sep(1 + len(impls)),
        ]
        for cuj in order:
            cells = [cuj]
            for impl in impls:
                status = per_cuj[cuj].get(impl, "")
                cells.append(glyph.get(status, "—"))
            lines.append(_md_row(cells))
        lines.append("")
        return lines

    def _capability_diff(self, impl_rows: list[dict]) -> list[str]:
        lines = ["### Capability coverage", ""]
        union_count = 0
        unique_total = 0
        for row in impl_rows:
            union_count = max(
                union_count,
                _as_int(row.get("mcp_readability_union_capability_count")),
            )
            impl = str(row.get(_IMPL, "") or "?")
            unique = _as_list(row.get("mcp_readability_unique_capabilities"))
            unique_total += len(unique)
            if unique:
                lines.append(
                    f"- **Only in {impl}** ({len(unique)}): "
                    + ", ".join(f"`{c}`" for c in unique)
                )
            else:
                lines.append(f"- **Only in {impl}**: none")
        shared = max(union_count - unique_total, 0)
        lines.append(
            f"- **Shared / overlapping**: {shared} of {union_count} "
            "union capabilities"
        )
        lines.append("")
        return lines


# ----------------------------------------------------------------------
# Formatting helpers
# ----------------------------------------------------------------------
def _finding_lines(finding: dict) -> list[str]:
    """Render one style-judge finding as a Markdown bullet with sub-detail.

    Rendered under its tool's heading, so the bullet carries the severity badge
    and rule rather than repeating the tool name.
    """
    rule = str(finding.get("rule_id", "")).strip() or "(rule)"
    sev = str(finding.get("severity", "")).upper()
    badge = SEVERITY_BADGES.get(sev, sev or "?")
    title = str(finding.get("title", "")).strip()
    header = f"- **{badge} · [{rule}]**"
    if title:
        header += f" — {title}"
    lines = [header]
    message = str(finding.get("message", "")).strip()
    if message:
        lines.append(f"  - _Issue:_ {message}")
    suggestion = str(finding.get("suggestion", "")).strip()
    if suggestion:
        lines.append(f"  - _Fix:_ {suggestion}")
    return lines


def _group_by(rows: list[dict], key: str) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = {}
    for row in rows:
        groups.setdefault(str(row.get(key, "") or ""), []).append(row)
    return groups


def _slug(product: str) -> str:
    """Filename-safe slug for a product name (``"Cloud SQL"`` -> ``cloud_sql``)."""
    slug = re.sub(r"[^a-z0-9]+", "_", str(product).strip().lower()).strip("_")
    return slug or "unknown"


def _md_row(cells: list[str]) -> str:
    return "| " + " | ".join(cells) + " |"


def _md_sep(n: int) -> str:
    return "| " + " | ".join(["---"] * n) + " |"


def _fmt(value, kind: str) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return "—"
    text = str(value).strip()
    if text in ("", "nan", "None", "<NA>"):
        return "—"
    if kind == "int":
        parsed = _as_int(value)
        return str(parsed) if parsed is not None else "—"
    if kind == "float":
        try:
            return f"{float(text):g}"
        except (TypeError, ValueError):
            return text
    return text


def _as_int(value):
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


def _as_list(value) -> list:
    if not value:
        return []
    if isinstance(value, list):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return []
    return parsed if isinstance(parsed, list) else []


def _as_dict(value) -> dict:
    if not value:
        return {}
    if isinstance(value, dict):
        return value
    try:
        parsed = json.loads(str(value))
    except (TypeError, ValueError):
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _wrap_html(markdown: str) -> str:
    """Minimal HTML wrapper (escaped markdown in a <pre>) for dashboard viewing.

    Kept dependency-free: the report is authored in Markdown; this only makes it
    openable in a browser without a Markdown renderer.
    """
    return (
        "<!doctype html><html><head><meta charset='utf-8'>"
        "<title>MCP Tool Comparison</title></head><body>"
        f"<pre>{html.escape(markdown)}</pre></body></html>"
    )
