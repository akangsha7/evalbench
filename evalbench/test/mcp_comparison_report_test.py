"""Unit tests for the ComparisonReporter (side-by-side report rendering)."""

import json
import os
import tempfile
import unittest

import pandas as pd

from reporting.comparison_report import ComparisonReporter
from reporting.report import STORETYPE


def _row(product, impl, **overrides):
    row = {
        "mcp_readability_product_name": product,
        "mcp_readability_implementation": impl,
        "mcp_readability_total_tools": 3,
        "mcp_readability_estimated_tokens": 300,
        "mcp_readability_avg_tokens_per_tool": 100.0,
        "mcp_readability_efficiency_tools_per_1k_tokens": 10.0,
        "mcp_readability_completeness_percent": 100.0,
        "mcp_readability_capabilities_covered": 3,
        "mcp_readability_union_capability_count": 3,
        "mcp_readability_unique_capabilities": json.dumps([]),
        "mcp_readability_missing_capabilities": json.dumps([]),
        "mcp_readability_p0_issues": 0,
        "mcp_readability_p1_issues": 0,
        "mcp_readability_p2_issues": 0,
        "mcp_readability_param_desc_coverage_percent": 100.0,
        "mcp_readability_enum_param_count": 0,
        "mcp_readability_list_ops_with_pagination_percent": 100.0,
        "mcp_readability_annotation_coverage_percent": 0.0,
        "mcp_readability_avg_required_params": 1.0,
    }
    row.update(overrides)
    return row


def _evals_df(rows):
    # The real pipeline builds the EVALS DataFrame with dtype="string"; mirror
    # that so the reporter is tested against stringified values.
    return pd.DataFrame.from_dict(rows, dtype="string")


class ComparisonReporterTest(unittest.TestCase):

    def _render(self, rows):
        with tempfile.TemporaryDirectory() as d:
            reporter = ComparisonReporter({"output_directory": d}, "job123", None)
            reporter.store(_evals_df(rows), STORETYPE.EVALS)
            md_path = os.path.join(d, "job123", "comparison.md")
            html_path = os.path.join(d, "job123", "comparison.html")
            self.assertTrue(os.path.exists(md_path))
            self.assertTrue(os.path.exists(html_path))
            with open(md_path) as f:
                return f.read()

    def test_renders_paired_product_with_diff(self):
        rows = [
            _row(
                "BigQuery",
                "one_mcp",
                mcp_readability_completeness_percent=50.0,
                mcp_readability_union_capability_count=4,
                mcp_readability_capabilities_covered=2,
                mcp_readability_unique_capabilities=json.dumps(["get_job_state"]),
                mcp_readability_missing_capabilities=json.dumps(
                    ["create_dataset", "delete_table"]
                ),
            ),
            _row(
                "BigQuery",
                "toolbox",
                mcp_readability_completeness_percent=75.0,
                mcp_readability_union_capability_count=4,
                mcp_readability_capabilities_covered=3,
                mcp_readability_unique_capabilities=json.dumps(
                    ["create_dataset", "delete_table"]
                ),
                mcp_readability_missing_capabilities=json.dumps(["get_job_state"]),
            ),
        ]
        md = self._render(rows)
        self.assertIn("# MCP Tool Comparison", md)
        self.assertIn("## Leaderboard", md)
        self.assertIn("## BigQuery", md)
        # Side-by-side table has both implementation columns.
        self.assertIn("one_mcp", md)
        self.assertIn("toolbox", md)
        # Capability diff lists the unique capabilities per implementation.
        self.assertIn("Only in one_mcp", md)
        self.assertIn("`get_job_state`", md)
        self.assertIn("`create_dataset`", md)
        # Shared count = union(4) - unique_total(3) = 1.
        self.assertIn("Shared / overlapping**: 1 of 4", md)

    def test_writes_one_report_per_product_plus_combined(self):
        rows = [
            _row("AlloyDB", "one_mcp"),
            _row("AlloyDB", "toolbox"),
            _row("Cloud SQL", "one_mcp"),
            _row("Cloud SQL", "toolbox"),
        ]
        with tempfile.TemporaryDirectory() as d:
            reporter = ComparisonReporter({"output_directory": d}, "job123", None)
            reporter.store(_evals_df(rows), STORETYPE.EVALS)
            out = os.path.join(d, "job123")
            files = set(os.listdir(out))
            # Combined report is still written.
            self.assertIn("comparison.md", files)
            self.assertIn("comparison.html", files)
            # One slugified report per product (md + html).
            for stem in ("comparison.alloydb", "comparison.cloud_sql"):
                self.assertIn(f"{stem}.md", files)
                self.assertIn(f"{stem}.html", files)

            # A per-product report is scoped to that product only.
            with open(os.path.join(out, "comparison.cloud_sql.md")) as f:
                cloud_sql = f.read()
            self.assertIn("# MCP Tool Comparison — Cloud SQL", cloud_sql)
            self.assertIn("## Cloud SQL", cloud_sql)
            self.assertNotIn("AlloyDB", cloud_sql)

            # The combined report still spans both products.
            with open(os.path.join(out, "comparison.md")) as f:
                combined = f.read()
            self.assertIn("## AlloyDB", combined)
            self.assertIn("## Cloud SQL", combined)

    def test_renders_detailed_findings_and_man_page_appendix(self):
        feedback = {
            "summary": "Leaks platform mechanics into the agent surface.",
            "findings_by_tool": [
                {
                    "tool": "general",
                    "findings": [
                        {
                            "severity": "P1",
                            "rule_id": "Use Enums",
                            "title": "type has no enum",
                        }
                    ],
                },
                {
                    "tool": "get_cluster",
                    "findings": [
                        {
                            "severity": "P0",
                            "rule_id": "Avoid Templated Strings",
                            "title": "Agent must build resource paths",
                            "message": "The name param requires projects/...",
                            "suggestion": "Split into project_id / cluster_id.",
                        }
                    ],
                },
            ],
            "waived": [
                {
                    "rule_id": "camelCase",
                    "reason": "OnePlatform convention, out of scope",
                    "would_have_violated": True,
                }
            ],
        }
        rows = [
            _row(
                "AlloyDB",
                "one_mcp",
                mcp_readability_llm_feedback_json=json.dumps(feedback),
                mcp_readability_man_page="TOOL: get_cluster\n  name (string)",
            ),
            _row("AlloyDB", "toolbox"),
        ]
        md = self._render(rows)
        # Detailed findings section is traceable to rule + tool.
        self.assertIn("### Style findings", md)
        self.assertIn("Leaks platform mechanics", md)
        # Each tool gets its own list, headed by its severity tally.
        self.assertIn("**get_cluster — 1 P0**", md)
        self.assertIn("[Avoid Templated Strings]", md)
        self.assertIn("_Issue:_", md)
        self.assertIn("_Fix:_", md)
        # The judge's "general" entry renders in the order it returned it.
        self.assertIn("**general — 1 P1**", md)
        self.assertLess(md.index("**general —"), md.index("**get_cluster —"))
        self.assertIn("[Use Enums]", md)
        # Waived rules are surfaced.
        self.assertIn("camelCase", md)
        self.assertIn("would have been flagged: yes", md)
        # Man-page appendix carries the raw tool surface inside a fenced block.
        self.assertIn("## Appendix — tool man pages", md)
        self.assertIn("### AlloyDB — one_mcp", md)
        self.assertIn("TOOL: get_cluster", md)

    def test_no_findings_or_man_page_omits_appendices(self):
        # Rows without feedback/man-page (e.g. offline metrics-only run) render
        # cleanly without the detail sections.
        md = self._render([_row("AlloyDB", "one_mcp"), _row("AlloyDB", "toolbox")])
        self.assertNotIn("### Style findings", md)
        self.assertNotIn("## Appendix — tool man pages", md)

    def test_renders_methodology_caveats(self):
        md = self._render([_row("AlloyDB", "one_mcp"), _row("AlloyDB", "toolbox")])
        self.assertIn("## How to read this report", md)
        # The interpretation caveats a reader can't infer from the tables.
        self.assertIn("non-deterministic", md)
        self.assertIn("one finding on one tool", md)
        self.assertIn("name-based", md)
        self.assertIn("covered = 1, partial = 0.5", md)

    def test_empty_dataframe_is_noop(self):
        with tempfile.TemporaryDirectory() as d:
            reporter = ComparisonReporter({"output_directory": d}, "job123", None)
            reporter.store(pd.DataFrame(), STORETYPE.EVALS)
            self.assertFalse(os.path.exists(os.path.join(d, "job123")))

    def test_ignores_non_evals_storetype(self):
        with tempfile.TemporaryDirectory() as d:
            reporter = ComparisonReporter({"output_directory": d}, "job123", None)
            reporter.store(_evals_df([_row("A", "toolbox")]), STORETYPE.SUMMARY)
            self.assertFalse(os.path.exists(os.path.join(d, "job123")))


if __name__ == "__main__":
    unittest.main()
