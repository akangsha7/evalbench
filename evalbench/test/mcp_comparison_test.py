"""Unit tests for the cross-endpoint (Toolbox vs one-MCP) comparison layer."""

import json
import unittest

from evaluator.mcp_readability import comparison as comparison_mod


def _row(product, implementation, capabilities):
    return {
        "mcp_readability_product_name": product,
        "mcp_readability_implementation": implementation,
        "mcp_readability_capabilities": json.dumps(capabilities),
    }


class DeterministicComparisonTest(unittest.TestCase):

    def test_union_completeness_and_diff(self):
        one_mcp = _row("BigQuery", "one_mcp", ["list_dataset", "get_job_state"])
        toolbox = _row(
            "BigQuery", "toolbox", ["list_dataset", "create_dataset", "delete_table"]
        )
        score_rows = comparison_mod.compare_products(
            [(one_mcp, []), (toolbox, [])]
        )

        # union = {list_dataset, get_job_state, create_dataset, delete_table}
        self.assertEqual(one_mcp["mcp_readability_union_capability_count"], 4)
        self.assertEqual(toolbox["mcp_readability_union_capability_count"], 4)

        self.assertEqual(one_mcp["mcp_readability_capabilities_covered"], 2)
        self.assertEqual(one_mcp["mcp_readability_completeness_percent"], 50.0)
        self.assertEqual(toolbox["mcp_readability_capabilities_covered"], 3)
        self.assertEqual(toolbox["mcp_readability_completeness_percent"], 75.0)

        self.assertEqual(
            json.loads(one_mcp["mcp_readability_unique_capabilities"]),
            ["get_job_state"],
        )
        self.assertEqual(
            json.loads(toolbox["mcp_readability_unique_capabilities"]),
            ["create_dataset", "delete_table"],
        )
        self.assertEqual(
            json.loads(one_mcp["mcp_readability_missing_capabilities"]),
            ["create_dataset", "delete_table"],
        )

        # Not at parity -> comparison score is 0.
        self.assertEqual(len(score_rows), 1)
        self.assertEqual(score_rows[0]["comparator"], "mcp_completeness")
        self.assertEqual(score_rows[0]["id"], "BigQuery")
        self.assertEqual(score_rows[0]["score"], 0)

    def test_solo_product_is_full_coverage(self):
        solo = _row("Spanner", "toolbox", ["list_instance", "get_instance"])
        score_rows = comparison_mod.compare_products([(solo, [])])
        self.assertEqual(solo["mcp_readability_completeness_percent"], 100.0)
        self.assertEqual(json.loads(solo["mcp_readability_missing_capabilities"]), [])
        self.assertEqual(score_rows[0]["score"], 100)

    def test_identical_implementations_are_at_parity(self):
        caps = ["list_instance", "get_instance"]
        a = _row("AlloyDB", "one_mcp", caps)
        b = _row("AlloyDB", "toolbox", caps)
        score_rows = comparison_mod.compare_products([(a, []), (b, [])])
        self.assertEqual(a["mcp_readability_completeness_percent"], 100.0)
        self.assertEqual(b["mcp_readability_completeness_percent"], 100.0)
        self.assertEqual(score_rows[0]["score"], 100)

    def test_products_grouped_independently(self):
        rows = [
            _row("A", "one_mcp", ["list_x"]),
            _row("A", "toolbox", ["list_x", "get_x"]),
            _row("B", "toolbox", ["list_y"]),
        ]
        score_rows = comparison_mod.compare_products([(r, []) for r in rows])
        by_id = {s["id"]: s for s in score_rows}
        self.assertEqual(set(by_id), {"A", "B"})
        self.assertEqual(by_id["B"]["score"], 100)  # solo product
        self.assertEqual(by_id["A"]["score"], 0)  # unequal implementations


class _FakeAligner:
    """Stand-in model that labels tools by a description-keyed canonical name."""

    def __init__(self, mapping):
        self._mapping = mapping
        self.calls = 0

    def generate(self, prompt):
        self.calls += 1
        return json.dumps(self._mapping)


class _Tool:
    def __init__(self, name, description=""):
        self.name = name
        self.description = description


class LlmAlignmentTest(unittest.TestCase):

    def test_llm_alignment_matches_differently_named_tools(self):
        # Deterministic signatures would NOT match these (product-prefixed names),
        # but the LLM aligner maps both to the same canonical capability.
        one_mcp = _row("CloudSQL", "one_mcp", ["bigquery_list_datasets"])
        toolbox = _row("CloudSQL", "toolbox", ["cloudsql_datasets_list"])
        members = [
            (one_mcp, [_Tool("bigquery_list_datasets")]),
            (toolbox, [_Tool("cloudsql_datasets_list")]),
        ]
        aligner = _FakeAligner(
            {
                "0::bigquery_list_datasets": "list_datasets",
                "1::cloudsql_datasets_list": "list_datasets",
            }
        )
        score_rows = comparison_mod.compare_products(members, model=aligner)
        self.assertEqual(aligner.calls, 1)
        self.assertEqual(one_mcp["mcp_readability_union_capability_count"], 1)
        self.assertEqual(one_mcp["mcp_readability_completeness_percent"], 100.0)
        self.assertEqual(score_rows[0]["score"], 100)

    def test_llm_failure_falls_back_to_deterministic(self):
        class _Boom:
            def generate(self, prompt):
                raise RuntimeError("model unavailable")

        one_mcp = _row("X", "one_mcp", ["list_x"])
        toolbox = _row("X", "toolbox", ["list_x", "get_x"])
        score_rows = comparison_mod.compare_products(
            [(one_mcp, [_Tool("list_x")]), (toolbox, [_Tool("list_x"), _Tool("get_x")])],
            model=_Boom(),
        )
        # Fell back to the deterministic capability signatures on the rows.
        self.assertEqual(one_mcp["mcp_readability_union_capability_count"], 2)
        self.assertEqual(one_mcp["mcp_readability_completeness_percent"], 50.0)
        self.assertEqual(score_rows[0]["score"], 0)


if __name__ == "__main__":
    unittest.main()
