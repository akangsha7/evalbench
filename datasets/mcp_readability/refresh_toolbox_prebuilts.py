"""Capture the tool listing for every MCP Toolbox prebuilt tool set.

Run before the readability eval so each prebuilt has a man page to score:

    python datasets/mcp_readability/refresh_toolbox_prebuilts.py --version 1.8.0

Writes one ``tools/list`` JSON dump per prebuilt plus the endpoints file the run
config points at. Captures are build artifacts, not source -- they are
gitignored, and diffing yesterday's against today's is how you find which tool
description actually moved a score.

Prebuilts whose sources dial a real backend cannot be captured live; those fall
back to rendering the tools from static config, which needs no server. The
strategy used is recorded per prebuilt in the manifest. Pin ``--version`` and
bump it deliberately, so that a change in the captures means upstream changed a
tool and not that the version drifted.

Requires Application Default Credentials: most sources mint a token at startup
even though they never call the backend.
"""

import argparse
import json
import logging
import os
import sys

sys.path.insert(
    0, os.path.join(os.path.dirname(__file__), "..", "..", "evalbench")
)

from generators.models.toolbox_prebuilt_capture import (  # noqa: E402
    CaptureError,
    capture_live,
    capture_offline,
    list_prebuilts,
    resolve_toolbox_binary,
    write_endpoints_yaml,
)

_DEFAULT_OUT_DIR = "datasets/mcp_readability/captures/toolbox"
_DEFAULT_ENDPOINTS = (
    "datasets/mcp_readability/endpoints.toolbox_prebuilts.yaml"
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--version", required=True,
        help="MCP Toolbox release to capture, e.g. 1.8.0. Overridden by "
             "TOOLBOX_BIN if that points at a binary.",
    )
    parser.add_argument(
        "--only",
        help="Comma-separated prebuilt names, instead of all of them.",
    )
    parser.add_argument("--out-dir", default=_DEFAULT_OUT_DIR)
    parser.add_argument("--endpoints-out", default=_DEFAULT_ENDPOINTS)
    parser.add_argument(
        "--project", default="placeholder-project",
        help="Stands in for every *_PROJECT setting. A real project you can "
             "read raises coverage slightly -- a few sources check IAM at "
             "startup -- at the cost of tying captures to that project.",
    )
    parser.add_argument("--region", default="us-central1")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO, format="%(levelname)s %(message)s"
    )

    binary = resolve_toolbox_binary(args.version)
    logging.info("Using toolbox binary: %s", binary)

    if args.only:
        prebuilts = [p.strip() for p in args.only.split(",") if p.strip()]
    else:
        prebuilts = list_prebuilts(args.version)
    logging.info("Capturing %d prebuilt tool set(s)", len(prebuilts))

    captured, skipped = [], []
    for prebuilt in prebuilts:
        path = os.path.join(args.out_dir, f"{prebuilt}.tools.json")
        strategy = "live"
        try:
            count = capture_live(
                binary, prebuilt, path,
                project=args.project, region=args.region,
            )
        except CaptureError as live_error:
            # Expected for any source that dials a real backend. Fall back to
            # rendering the tools from static config, which needs no server.
            logging.info(
                "%s: no live capture (%s); falling back to offline",
                prebuilt, live_error,
            )
            strategy = "offline"
            try:
                count = capture_offline(binary, prebuilt, path)
            except CaptureError as offline_error:
                logging.warning("skip %s: %s", prebuilt, offline_error)
                skipped.append(prebuilt)
                continue
        logging.info("captured %s (%d tools, %s)", prebuilt, count, strategy)
        captured.append(
            {"prebuilt": prebuilt, "path": path, "strategy": strategy}
        )

    if not captured:
        logging.error("No prebuilt could be captured; not writing endpoints.")
        return 1

    write_endpoints_yaml(captured, args.endpoints_out)
    manifest = os.path.join(args.out_dir, "manifest.json")
    with open(manifest, "w") as f:
        json.dump(
            {
                "toolbox_version": args.version,
                "captured": captured,
                "skipped": skipped,
            },
            f, indent=2, sort_keys=True,
        )
        f.write("\n")

    live = sum(1 for c in captured if c["strategy"] == "live")
    logging.info(
        "Captured %d/%d prebuilts (%d live, %d offline) -> %s",
        len(captured), len(prebuilts), live, len(captured) - live,
        args.endpoints_out,
    )
    if skipped:
        logging.warning("Could not capture: %s", ", ".join(skipped))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
