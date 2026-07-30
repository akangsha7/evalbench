"""Capture MCP tool listings for MCP Toolbox's prebuilt tool sets.

The readability check needs a man page for every prebuilt tool set shipped by
``googleapis/mcp-toolbox``, on a machine with no provisioned databases. This
module produces one raw ``tools/list`` JSON dump per prebuilt, which the
``mcp_tools`` generator then consumes via ``tools_source.type: file``.

Most prebuilts can be captured *live*: their sources only construct an API
client at startup rather than dialing a backend, so ``toolbox --prebuilt X
--stdio`` starts and answers ``tools/list`` against a project that does not
exist. Those that genuinely connect (plain SQL sources, the Cloud SQL / AlloyDB
data planes) fail here and are left to an offline extraction path.

Live capture is preferred where it works: tools that refine their schema against
the live source (injecting the configured project as a parameter default, for
instance) only do so on this path, so the captured schema matches what an agent
would really see.

Prebuilt configs reference their settings as ``${VAR}`` with no default, and the
server refuses to start until each one is set. Rather than track that list by
hand, ``capture_live`` reads the missing variable's name out of the startup
error, assigns a placeholder, and retries -- so new settings upstream do not
need a corresponding change here.
"""

import json
import logging
import os
import platform
import re
import stat
import subprocess
import tempfile
import urllib.request

import yaml

from .mcp_tools import McpToolsError, McpToolsGenerator

_RELEASE_BUCKET = "https://storage.googleapis.com/mcp-toolbox-for-databases"
_PREBUILT_DIR = "internal/prebuiltconfigs/tools"
_CONTENTS_API = (
    "https://api.github.com/repos/googleapis/mcp-toolbox/contents/"
    f"{_PREBUILT_DIR}?ref=v{{version}}"
)

# The server reports one missing variable per start, naming it in the error.
_MISSING_ENV = re.compile(
    r'environment variable not found: \\?"([A-Z0-9_]+)\\?"'
)

# Enough to walk the longest prebuilt's settings one at a time, with headroom.
_MAX_ENV_ATTEMPTS = 30

# Used when the GitHub API is unreachable. Only a starting point: a prebuilt
# missing from here is still captured if the API answers.
_FALLBACK_PREBUILTS = (
    "alloydb-omni", "alloydb-postgres", "alloydb-postgres-admin",
    "alloydb-postgres-observability", "bigquery", "clickhouse",
    "cloud-healthcare", "cloud-sql-mssql", "cloud-sql-mssql-admin",
    "cloud-sql-mssql-observability", "cloud-sql-mysql",
    "cloud-sql-mysql-admin", "cloud-sql-mysql-observability",
    "cloud-sql-postgres", "cloud-sql-postgres-admin",
    "cloud-sql-postgres-observability", "cloud-storage",
    "conversational-analytics-with-data-agent", "dataplex", "dataproc",
    "elasticsearch", "firestore", "looker", "looker-conversational-analytics",
    "looker-dev", "mindsdb", "mssql", "mysql", "neo4j", "oceanbase",
    "oracledb", "postgres", "serverless-spark", "singlestore", "snowflake",
    "spanner", "spanner-postgres", "sqlite",
)

# skills-generate renders Toolbox's own parameter vocabulary, which is not quite
# JSON Schema's. Anything absent here is already a JSON Schema type name.
_JSON_SCHEMA_TYPES = {"float": "number"}

# Forced regardless of what the config asks for: without it the Looker source
# performs a login round-trip against LOOKER_BASE_URL and cannot start.
_FORCED_ENV = {"LOOKER_USE_CLIENT_OAUTH": "true"}

logger = logging.getLogger(__name__)


class CaptureError(Exception):
    """Raised when a prebuilt's tools cannot be captured."""


def placeholder_env_value(var: str, project: str, region: str) -> str:
    """A stand-in value for ``var``, keyed off its name.

    Values are fixed so that captures differ day over day only when the tool
    definitions themselves change. Some are shaped by the variable's suffix
    because the source parses them (ports must be numeric, URIs need a scheme).
    """
    if var.endswith("_PORT"):
        return "9999"
    if var.endswith("_PROJECT"):
        return project
    if var.endswith(("_REGION", "_LOCATION")):
        return region
    if var.endswith("_HOST"):
        return "127.0.0.1"
    if var.endswith("_URI"):
        return "bolt://127.0.0.1:7687"
    if var.endswith("_URL"):
        return "https://placeholder.example.com"
    if var.endswith("_CONNECTION_STRING"):
        return "127.0.0.1:1521/placeholder"
    if var.endswith("_PROTOCOL"):
        return "http"
    if var == "SQLITE_DATABASE":
        return os.path.join(tempfile.gettempdir(), "toolbox-capture.db")
    return "placeholder"


def resolve_toolbox_binary(version: str, cache_dir: str | None = None) -> str:
    """Path to a ``toolbox`` binary at ``version``, downloading if needed.

    ``TOOLBOX_BIN`` wins if set, so a locally built binary (from an unreleased
    branch, say) can be used without touching the config.
    """
    override = os.environ.get("TOOLBOX_BIN")
    if override:
        if not os.path.exists(override):
            raise CaptureError(f"TOOLBOX_BIN does not exist: {override}")
        return override

    goos = {"darwin": "darwin", "linux": "linux"}.get(platform.system().lower())
    if not goos:
        raise CaptureError(f"Unsupported platform: {platform.system()}")
    goarch = {
        "x86_64": "amd64", "amd64": "amd64",
        "arm64": "arm64", "aarch64": "arm64",
    }.get(platform.machine().lower())
    if not goarch:
        raise CaptureError(f"Unsupported architecture: {platform.machine()}")

    cache_dir = cache_dir or os.path.join(
        tempfile.gettempdir(), "toolbox-binaries"
    )
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"toolbox-{version}-{goos}-{goarch}")
    if os.path.exists(path):
        return path

    url = f"{_RELEASE_BUCKET}/v{version}/{goos}/{goarch}/toolbox"
    logger.info("Downloading toolbox %s from %s", version, url)
    tmp = f"{path}.part"
    try:
        urllib.request.urlretrieve(url, tmp)
    except Exception as e:
        raise CaptureError(f"Could not download toolbox from {url}: {e}") from e
    os.chmod(tmp, os.stat(tmp).st_mode | stat.S_IEXEC | stat.S_IXGRP)
    os.replace(tmp, path)
    return path


def list_prebuilts(version: str) -> list[str]:
    """Names of the prebuilt tool sets shipped at ``version``."""
    url = _CONTENTS_API.format(version=version)
    try:
        with urllib.request.urlopen(url, timeout=30) as resp:
            entries = json.loads(resp.read())
        names = sorted(
            e["name"][:-5]
            for e in entries
            if e.get("type") == "file" and e.get("name", "").endswith(".yaml")
        )
        if names:
            return names
        raise CaptureError("no .yaml entries returned")
    except Exception as e:
        logger.warning(
            "Could not list prebuilts at v%s (%s); using the built-in list, "
            "which may be missing tool sets added upstream since.", version, e
        )
        return list(_FALLBACK_PREBUILTS)


def _resolve_env(
    binary: str, prebuilt: str, project: str, region: str
) -> dict[str, str]:
    """Placeholder values for the settings ``prebuilt`` requires.

    Starts the server with an empty stdin -- it reads EOF and exits -- once per
    missing variable, each time learning the next name from the startup error.
    Returns once the config parses; a source that then fails to connect is left
    for the caller to detect during the real capture.
    """
    env = dict(_FORCED_ENV)
    for _ in range(_MAX_ENV_ATTEMPTS):
        proc = subprocess.run(
            [binary, "--prebuilt", prebuilt, "--stdio"],
            input="", capture_output=True, text=True, timeout=120,
            env={**os.environ, **env},
        )
        match = _MISSING_ENV.search(proc.stderr)
        if not match:
            return env
        var = match.group(1)
        env[var] = placeholder_env_value(var, project, region)
    raise CaptureError(
        f"{prebuilt}: still missing environment variables after "
        f"{_MAX_ENV_ATTEMPTS} attempts"
    )


def capture_live(
    binary: str,
    prebuilt: str,
    out_path: str,
    project: str = "placeholder-project",
    region: str = "us-central1",
) -> int:
    """Write ``prebuilt``'s live ``tools/list`` to ``out_path``.

    Raises ``CaptureError`` if the server cannot start -- which for a source
    that dials a real backend is the expected outcome, not a bug.

    Returns the number of tools captured.
    """
    env = _resolve_env(binary, prebuilt, project, region)
    source = {
        "type": "stdio",
        "command": binary,
        "args": ["--prebuilt", prebuilt, "--stdio"],
        "env": env,
    }
    try:
        tools, _ = McpToolsGenerator({}).fetch_tools({"tools_source": source})
    except McpToolsError as e:
        raise CaptureError(f"{prebuilt}: live capture failed: {e}") from e
    if not tools:
        raise CaptureError(f"{prebuilt}: server returned no tools")

    payload = {
        "tools": [
            {
                "name": t.name,
                "description": t.description or "",
                "inputSchema": t.inputSchema or {},
            }
            for t in sorted(tools, key=lambda t: t.name)
        ]
    }
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return len(tools)


def _coerce_default(raw: str):
    """Turn a ``Default`` cell into a JSON value, or None if the cell is blank.

    Cells are rendered as `` `false` ``/`` `50` ``/`` `[]` ``. Anything that is
    not valid JSON is kept as the literal string.
    """
    text = raw.strip().strip("`").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text


def _parse_skill_md(text: str) -> list[dict]:
    """Tools described by one generated ``SKILL.md``.

    Each tool is a ``### <name>`` heading followed by its description and, when
    it takes any, a ``#### Parameters`` table of
    ``Name | Type | Description | Required | Default``.
    """
    tools = []
    sections = re.finditer(
        r"^### (\S+)\n(.*?)(?=^### |\Z)", text, re.S | re.M
    )
    for section in sections:
        name = section.group(1)
        body = section.group(2)
        # A tool with no parameters runs to the "---" rule before the next
        # heading, which would otherwise land in its description.
        description = body.split("#### Parameters")[0]
        description = re.sub(r"\n+-{3,}\s*$", "", description).strip()

        properties, required = {}, []
        for row in re.finditer(
            r"^\| (\S+) \| (\S+) \| (.*?) \| (Yes|No) \| (.*?) \|$",
            body, re.M,
        ):
            param, ptype, pdesc, is_required, pdefault = row.groups()
            # The table's vocabulary is Toolbox's, not JSON Schema's.
            schema = {"type": _JSON_SCHEMA_TYPES.get(ptype, ptype)}
            if pdesc.strip():
                schema["description"] = pdesc.strip()
            default = _coerce_default(pdefault)
            if default is not None:
                schema["default"] = default
            properties[param] = schema
            if is_required == "Yes":
                required.append(param)

        tools.append({
            "name": name,
            "description": description,
            "inputSchema": {
                "type": "object",
                "properties": properties,
                "required": required,
            },
        })
    return tools


def capture_offline(binary: str, prebuilt: str, out_path: str) -> int:
    """Write ``prebuilt``'s tools to ``out_path`` without starting a server.

    ``skills-generate`` renders each tool from its static config, so this works
    for sources that dial a real backend. Two things the live path would give
    are not recoverable from the rendered tables: the element type of an
    ``array`` parameter, and any schema refinement a tool makes against its
    source (an injected parameter default, say). Prefer ``capture_live``.

    Returns the number of tools captured.
    """
    with tempfile.TemporaryDirectory() as tmp:
        proc = subprocess.run(
            [binary, "skills-generate", "--prebuilt", prebuilt,
             "--output-dir", tmp],
            capture_output=True, text=True, timeout=300,
        )
        if proc.returncode != 0:
            raise CaptureError(
                f"{prebuilt}: skills-generate failed: {proc.stderr.strip()}"
            )

        # A prebuilt renders one skill per group, and a tool can belong to
        # several. Keyed by name, so the duplicates collapse.
        by_name: dict[str, dict] = {}
        for root, _, files in os.walk(tmp):
            for fname in files:
                if fname != "SKILL.md":
                    continue
                with open(os.path.join(root, fname)) as f:
                    for tool in _parse_skill_md(f.read()):
                        by_name.setdefault(tool["name"], tool)

    if not by_name:
        raise CaptureError(f"{prebuilt}: skills-generate produced no tools")

    payload = {"tools": [by_name[n] for n in sorted(by_name)]}
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2, sort_keys=True)
        f.write("\n")
    return len(by_name)


def write_endpoints_yaml(
    captures: list[dict], out_path: str, endpoint_type: str = "PREBUILT_TOOL"
) -> None:
    """Render an ``endpoints.yaml`` over successfully captured prebuilts.

    Each capture is ``{"prebuilt": ..., "path": ..., "strategy": ...}``. Only
    the fields the orchestrator reads are emitted; how a capture was obtained is
    recorded alongside it in the run's manifest instead.
    """
    endpoints = [
        {
            "product_name": f"MCP Toolbox ({c['prebuilt']})",
            "endpoint_type": endpoint_type,
            "tools_source": {"type": "file", "path": c["path"]},
        }
        for c in sorted(captures, key=lambda c: c["prebuilt"])
    ]
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w") as f:
        f.write(
            "# Generated by datasets/mcp_readability/"
            "refresh_toolbox_prebuilts.py -- do not edit.\n"
        )
        yaml.safe_dump(
            {"endpoints": endpoints}, f, sort_keys=False, default_flow_style=False
        )
