"""Unit tests for the MCP Toolbox prebuilt capture helpers.

Runs entirely offline: the toolbox binary is never invoked. The environment
resolution loop is driven by a fake ``subprocess.run`` that replays the startup
errors a real server emits, and the MCP conversation is mocked out.
"""

import json
import os
import tempfile
import unittest
from unittest.mock import patch

from mcp import types as mcp_types

from generators.models.mcp_tools import McpToolsError, McpToolsGenerator
from generators.models import toolbox_prebuilt_capture as capture


def _missing(var: str):
    """A completed process that failed for want of ``var``."""
    class Result:
        returncode = 1
        stdout = ""
        stderr = (
            'ERROR "unable to parse prebuilt tool configuration for '
            f"'x': error parsing environment variables: environment "
            f'variable not found: \\"{var}\\" (line 18, column 10)"'
        )
    return Result()


def _started():
    class Result:
        returncode = 0
        stdout = ""
        stderr = 'INFO "Starting MCP Toolbox for Databases version 1.8.0"'
    return Result()


_TOOLS = [
    mcp_types.Tool(
        name="list_table_ids",
        description="List table IDs in a dataset.",
        inputSchema={
            "type": "object",
            "properties": {"dataset": {"type": "string"}},
            "required": ["dataset"],
        },
    ),
    mcp_types.Tool(
        name="execute_sql",
        description="Run a SQL query.",
        inputSchema={"type": "object", "properties": {}},
    ),
]


class PlaceholderEnvValueTest(unittest.TestCase):

    def test_shapes_value_by_variable_suffix(self):
        val = capture.placeholder_env_value
        self.assertEqual(val("BIGQUERY_PROJECT", "proj", "reg"), "proj")
        self.assertEqual(val("DATAPROC_REGION", "proj", "reg"), "reg")
        self.assertEqual(val("CLOUD_GDA_LOCATION", "proj", "reg"), "reg")
        self.assertEqual(val("MYSQL_USER", "proj", "reg"), "placeholder")

    def test_values_the_source_parses_are_well_formed(self):
        val = capture.placeholder_env_value
        # A port is parsed as a number and a URI needs a scheme, so a bare
        # "placeholder" would fail before the server could list its tools.
        self.assertTrue(val("CLICKHOUSE_PORT", "p", "r").isdigit())
        self.assertIn("://", val("NEO4J_URI", "p", "r"))
        self.assertIn("://", val("LOOKER_BASE_URL", "p", "r"))

    def test_is_deterministic(self):
        # Captures are diffed day over day, so the same variable must always
        # resolve to the same value.
        first = capture.placeholder_env_value("BIGQUERY_PROJECT", "p", "r")
        second = capture.placeholder_env_value("BIGQUERY_PROJECT", "p", "r")
        self.assertEqual(first, second)


class ResolveEnvTest(unittest.TestCase):

    def test_learns_one_variable_per_attempt(self):
        replies = [
            _missing("BIGQUERY_PROJECT"),
            _missing("BIGQUERY_LOCATION"),
            _started(),
        ]
        with patch.object(capture.subprocess, "run", side_effect=replies):
            env = capture._resolve_env("tb", "bigquery", "proj", "us-east1")
        self.assertEqual(env["BIGQUERY_PROJECT"], "proj")
        self.assertEqual(env["BIGQUERY_LOCATION"], "us-east1")

    def test_always_forces_looker_client_oauth(self):
        with patch.object(capture.subprocess, "run", return_value=_started()):
            env = capture._resolve_env("tb", "looker", "proj", "us-east1")
        self.assertEqual(env["LOOKER_USE_CLIENT_OAUTH"], "true")

    def test_gives_up_rather_than_looping_forever(self):
        forever = _missing("NEVER_SATISFIED")
        with patch.object(capture.subprocess, "run", return_value=forever):
            with self.assertRaises(capture.CaptureError):
                capture._resolve_env("tb", "x", "proj", "us-east1")


class CaptureLiveTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "nested", "bigquery.tools.json")

    def _capture(self, fetch_result=None, fetch_error=None):
        with patch.object(capture.subprocess, "run", return_value=_started()):
            with patch.object(
                McpToolsGenerator, "fetch_tools",
                side_effect=fetch_error,
                return_value=(fetch_result, ""),
            ) as fetch:
                count = capture.capture_live("tb", "bigquery", self.out)
        return count, fetch

    def test_writes_a_spec_the_generator_can_read_back(self):
        count, _ = self._capture(fetch_result=_TOOLS)
        self.assertEqual(count, 2)
        # The whole point of the capture: mcp_tools must consume it verbatim
        # through tools_source.type "file".
        tools, man_page = McpToolsGenerator({}).fetch_tools(
            {"tools_source": {"type": "file", "path": self.out}}
        )
        self.assertEqual([t.name for t in tools],
                         ["execute_sql", "list_table_ids"])
        self.assertIn("list_table_ids", man_page)

    def test_sorts_tools_so_captures_diff_cleanly(self):
        self._capture(fetch_result=_TOOLS)
        with open(self.out) as f:
            payload = json.load(f)
        names = [t["name"] for t in payload["tools"]]
        self.assertEqual(names, sorted(names))

    def test_passes_the_resolved_env_to_the_server(self):
        _, fetch = self._capture(fetch_result=_TOOLS)
        source = fetch.call_args[0][0]["tools_source"]
        self.assertEqual(source["args"], ["--prebuilt", "bigquery", "--stdio"])
        self.assertIn("LOOKER_USE_CLIENT_OAUTH", source["env"])

    def test_reports_a_source_that_dials_a_real_backend(self):
        # Expected for the likes of postgres, not a bug: the caller records it
        # as needing offline extraction instead of failing the run.
        with self.assertRaises(capture.CaptureError):
            self._capture(fetch_error=McpToolsError("connection refused"))

    def test_rejects_an_empty_tool_listing(self):
        with self.assertRaises(capture.CaptureError):
            self._capture(fetch_result=[])


_SKILL_MD = """---
name: postgres-data
description:
---

## Usage

Some preamble that is not a tool.

## Scripts


### execute_sql

Use this tool to execute a single SQL statement.

#### Parameters

| Name | Type | Description | Required | Default |
| :--- | :--- | :--- | :--- | :--- |
| sql | string | The sql to execute. | Yes |  |
| limit | integer | Max rows. | No | `50` |
| dry_run | boolean | Preview only. | No | `false` |
| recall | float | Target recall. | No | `0.95` |
| roles | array | Roles to grant. | No | `[]` |


---

### database_overview

Fetches the current state of the server.


---
"""


class ParseSkillMdTest(unittest.TestCase):

    def setUp(self):
        self.tools = {t["name"]: t
                      for t in capture._parse_skill_md(_SKILL_MD)}

    def test_reads_every_tool_and_ignores_prose_headings(self):
        # "## Usage" / "## Scripts" are not tools; only "### " headings are.
        self.assertEqual(set(self.tools), {"execute_sql", "database_overview"})

    def test_separates_required_from_optional(self):
        schema = self.tools["execute_sql"]["inputSchema"]
        self.assertEqual(schema["required"], ["sql"])

    def test_maps_toolbox_types_onto_json_schema(self):
        props = self.tools["execute_sql"]["inputSchema"]["properties"]
        self.assertEqual(props["limit"]["type"], "integer")
        # "float" is Toolbox's spelling; JSON Schema calls it "number".
        self.assertEqual(props["recall"]["type"], "number")

    def test_reads_defaults_as_typed_values(self):
        props = self.tools["execute_sql"]["inputSchema"]["properties"]
        self.assertEqual(props["limit"]["default"], 50)
        self.assertIs(props["dry_run"]["default"], False)
        self.assertEqual(props["roles"]["default"], [])
        # A blank cell is no default, not an empty string.
        self.assertNotIn("default", props["sql"])

    def test_keeps_the_section_rule_out_of_a_description(self):
        # A tool with no parameters runs to the "---" before the next heading.
        self.assertEqual(
            self.tools["database_overview"]["description"],
            "Fetches the current state of the server.",
        )

    def test_a_tool_without_parameters_still_has_a_schema(self):
        schema = self.tools["database_overview"]["inputSchema"]
        self.assertEqual(schema["properties"], {})
        self.assertEqual(schema["required"], [])


class CaptureOfflineTest(unittest.TestCase):

    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.out = os.path.join(self.tmp, "nested", "postgres.tools.json")

    def _run(self, returncode=0, groups=(("a", _SKILL_MD),)):
        def fake_run(cmd, **kwargs):
            out_dir = cmd[cmd.index("--output-dir") + 1]
            for name, text in groups:
                d = os.path.join(out_dir, name)
                os.makedirs(d, exist_ok=True)
                with open(os.path.join(d, "SKILL.md"), "w") as f:
                    f.write(text)

            class Result:
                pass
            Result.returncode = returncode
            Result.stdout = ""
            Result.stderr = "boom" if returncode else ""
            return Result

        with patch.object(capture.subprocess, "run", side_effect=fake_run):
            return capture.capture_offline("tb", "postgres", self.out)

    def test_writes_a_spec_the_generator_can_read_back(self):
        count = self._run()
        self.assertEqual(count, 2)
        tools, man_page = McpToolsGenerator({}).fetch_tools(
            {"tools_source": {"type": "file", "path": self.out}}
        )
        self.assertEqual([t.name for t in tools],
                         ["database_overview", "execute_sql"])
        self.assertIn("execute_sql", man_page)

    def test_collapses_a_tool_shared_by_several_skill_groups(self):
        # A prebuilt renders one skill per group and tools repeat across them.
        count = self._run(groups=(("a", _SKILL_MD), ("b", _SKILL_MD)))
        self.assertEqual(count, 2)

    def test_reports_a_failed_generate(self):
        with self.assertRaises(capture.CaptureError):
            self._run(returncode=1)

    def test_rejects_output_with_no_tools(self):
        with self.assertRaises(capture.CaptureError):
            self._run(groups=(("a", "# nothing here\n"),))


class WriteEndpointsYamlTest(unittest.TestCase):

    def test_emits_endpoints_the_orchestrator_can_consume(self):
        import yaml

        tmp = tempfile.mkdtemp()
        out = os.path.join(tmp, "endpoints.yaml")
        capture.write_endpoints_yaml(
            [
                {"prebuilt": "spanner", "path": "a/spanner.json",
                 "strategy": "live"},
                {"prebuilt": "bigquery", "path": "a/bigquery.json",
                 "strategy": "live"},
            ],
            out,
        )
        with open(out) as f:
            parsed = yaml.safe_load(f)

        names = [e["product_name"] for e in parsed["endpoints"]]
        self.assertEqual(
            names, ["MCP Toolbox (bigquery)", "MCP Toolbox (spanner)"]
        )
        first = parsed["endpoints"][0]
        self.assertEqual(first["tools_source"],
                         {"type": "file", "path": "a/bigquery.json"})
        self.assertEqual(first["endpoint_type"], "PREBUILT_TOOL")
        # Only fields the orchestrator reads; anything else is dead config.
        self.assertEqual(set(first),
                         {"product_name", "endpoint_type", "tools_source"})


class ListPrebuiltsTest(unittest.TestCase):

    def test_falls_back_when_github_is_unreachable(self):
        with patch.object(
            capture.urllib.request, "urlopen", side_effect=OSError("no net")
        ):
            names = capture.list_prebuilts("1.8.0")
        self.assertIn("bigquery", names)
        self.assertIn("postgres", names)

    def test_prefers_the_live_listing(self):
        payload = json.dumps([
            {"type": "file", "name": "bigquery.yaml"},
            {"type": "file", "name": "brand-new.yaml"},
            {"type": "file", "name": "README.md"},
        ]).encode()

        class Resp:
            def read(self):
                return payload

            def __enter__(self):
                return self

            def __exit__(self, *a):
                return False

        with patch.object(capture.urllib.request, "urlopen",
                          return_value=Resp()):
            names = capture.list_prebuilts("1.8.0")
        self.assertEqual(names, ["bigquery", "brand-new"])


if __name__ == "__main__":
    unittest.main()
