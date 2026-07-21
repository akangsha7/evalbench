"""Unit tests for the deterministic MCP schema-quality scorer."""

import unittest

from mcp import types as mcp_types

from scorers.mcp_readability_scoring import EndpointContext
from scorers.mcp_schema_quality import McpSchemaQualityScorer


def _tool(name, schema, annotations=None):
    return mcp_types.Tool(
        name=name,
        description=f"{name} does a thing.",
        inputSchema=schema or {"type": "object"},
        annotations=annotations,
    )


def _ctx(tools):
    return EndpointContext(
        product_name="P",
        endpoint={},
        tools=tools,
        man_page="",
        exceptions=[],
    )


class McpSchemaQualityScorerTest(unittest.TestCase):

    def setUp(self):
        self.scorer = McpSchemaQualityScorer()

    def test_param_description_coverage(self):
        tools = [
            _tool(
                "create_dataset",
                {
                    "type": "object",
                    "properties": {
                        "project_id": {
                            "type": "string",
                            "description": "The project.",
                        },
                        "labels": {"type": "object"},  # no description
                    },
                    "required": ["project_id"],
                },
            )
        ]
        metrics = self.scorer.score(tools)
        self.assertEqual(metrics["param_desc_coverage_percent"], 50.0)

    def test_enum_param_count(self):
        tools = [
            _tool(
                "get_job",
                {
                    "type": "object",
                    "properties": {
                        "view": {
                            "type": "string",
                            "enum": ["BASIC", "FULL"],
                            "description": "d",
                        },
                        "id": {"type": "string", "description": "d"},
                    },
                },
            )
        ]
        self.assertEqual(self.scorer.score(tools)["enum_param_count"], 1)

    def test_pagination_only_counts_list_ops(self):
        paginated = _tool(
            "list_datasets",
            {
                "type": "object",
                "properties": {
                    "page_size": {"type": "integer", "description": "d"},
                    "page_token": {"type": "string", "description": "d"},
                },
            },
        )
        unpaginated = _tool(
            "list_tables",
            {
                "type": "object",
                "properties": {
                    "dataset_id": {"type": "string", "description": "d"},
                },
            },
        )
        # A non-list op with no pagination must not drag the percentage down.
        get_op = _tool(
            "get_dataset",
            {
                "type": "object",
                "properties": {"id": {"type": "string", "description": "d"}},
            },
        )
        metrics = self.scorer.score([paginated, unpaginated, get_op])
        # 1 of 2 list ops paginated.
        self.assertEqual(metrics["list_ops_with_pagination_percent"], 50.0)

    def test_annotation_coverage(self):
        annotated = _tool(
            "delete_table",
            {"type": "object"},
            annotations=mcp_types.ToolAnnotations(destructiveHint=True),
        )
        plain = _tool("get_table", {"type": "object"})
        metrics = self.scorer.score([annotated, plain])
        self.assertEqual(metrics["annotation_coverage_percent"], 50.0)

    def test_avg_required_params(self):
        a = _tool(
            "create_x",
            {
                "type": "object",
                "properties": {
                    "a": {"type": "string", "description": "d"},
                    "b": {"type": "string", "description": "d"},
                },
                "required": ["a", "b"],
            },
        )
        b = _tool(
            "get_x",
            {
                "type": "object",
                "properties": {"a": {"type": "string", "description": "d"}},
                "required": ["a"],
            },
        )
        self.assertEqual(self.scorer.score([a, b])["avg_required_params"], 1.5)

    def test_nested_object_params_counted(self):
        tools = [
            _tool(
                "create_instance",
                {
                    "type": "object",
                    "properties": {
                        "config": {
                            "type": "object",
                            "description": "The config.",
                            "properties": {
                                "tier": {"type": "string", "description": "d"},
                                "region": {"type": "string"},  # no description
                            },
                        }
                    },
                },
            )
        ]
        # 3 params total (config, config.tier, config.region); 2 described.
        metrics = self.scorer.score(tools)
        self.assertEqual(metrics["param_desc_coverage_percent"], round(2 / 3 * 100, 2))

    def test_empty_tools_are_neutral(self):
        metrics = self.scorer.score([])
        self.assertEqual(metrics["param_desc_coverage_percent"], 100.0)
        self.assertEqual(metrics["enum_param_count"], 0)
        self.assertEqual(metrics["avg_required_params"], 0.0)

    def test_run_pass_fail_on_coverage(self):
        good = _tool(
            "get_x",
            {
                "type": "object",
                "properties": {"a": {"type": "string", "description": "d"}},
            },
        )
        bad = _tool(
            "get_y",
            {"type": "object", "properties": {"a": {"type": "string"}}},
        )
        self.assertEqual(self.scorer.run(_ctx([good])).score, 100)
        self.assertEqual(self.scorer.run(_ctx([bad])).score, 0)

    def test_run_contributes_declared_columns(self):
        tools = [
            _tool(
                "get_x",
                {
                    "type": "object",
                    "properties": {"a": {"type": "string", "description": "d"}},
                },
            )
        ]
        contribution = self.scorer.run(_ctx(tools))
        self.assertEqual(
            set(contribution.row_fields), set(McpSchemaQualityScorer.COLUMNS)
        )


if __name__ == "__main__":
    unittest.main()
