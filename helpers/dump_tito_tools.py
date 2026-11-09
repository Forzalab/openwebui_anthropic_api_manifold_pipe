"""Deploy-time helper: write ``TITO_TOOLS_FILE`` from the live tool set.

``TITO_TOOLS_FILE`` is a JSON *list* in Anthropic tools format (C3):

    [{"name": ..., "description": ..., "input_schema": {...}}, ...]

``src/anthropic_pipe/tito.py`` loads this byte-stable array at request time
(``_tito_frozen_tools``) instead of the live ``__tools__``/``body.tools``, so a
tool-server outage or an upstream description edit does not change the tools the
Tito model sees. The file must include the file, memory, calendar and TinyFish
tools.

Run at deploy, inside the OpenWebUI image (or on a machine that can import it):

    python helpers/dump_tito_tools.py --output /path/to/tito_tools.json

The exact OpenWebUI tool API varies across versions; the loader below tries the
known entry points and fails loudly (exit 1) rather than writing an empty list.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]


def _to_anthropic_tool(name: str, spec: dict) -> dict | None:
    """Convert one OpenWebUI tool spec to Anthropic tools format, or None."""
    if not isinstance(spec, dict) or not name:
        return None
    # Names starting with "_" are private helpers, never callable tools.
    if name.startswith("_"):
        return None

    parameters = spec.get("parameters", {})
    if not isinstance(parameters, dict):
        parameters = {}
    input_schema: dict = {"type": "object", "properties": parameters.get("properties", {})}
    if "required" in parameters:
        input_schema["required"] = parameters["required"]

    return {
        "name": name,
        "description": spec.get("description") or f"Tool: {name}",
        "input_schema": input_schema,
    }


def _collect_tools() -> dict[str, dict]:
    """Return {name: spec} for every live tool, across the known entry points."""
    tools: dict[str, dict] = {}

    def _ingest(mapping):
        if not mapping:
            return
        for tool_id, tool_data in mapping.items():
            if not isinstance(tool_data, dict):
                continue
            spec = tool_data.get("spec") if isinstance(tool_data.get("spec"), dict) else tool_data
            name = spec.get("name") or (tool_id if isinstance(tool_id, str) else None)
            if name:
                tools[name] = spec

    # 1. OpenWebUI user/function tools (covers MCP tools: file, memory, calendar, TinyFish).
    try:
        from open_webui.utils.tools import get_tools

        _tools = get_tools()
        if hasattr(_tools, "__await__"):
            import asyncio

            _tools = asyncio.get_event_loop().run_until_complete(_tools)
        if isinstance(_tools, dict):
            _ingest(_tools)
    except Exception as e:
        print(f"warning: could not load open_webui.utils.tools.get_tools: {e}", file=sys.stderr)

    # 2. OpenWebUI builtin tools (skills/file-search/code helpers).
    try:
        from open_webui.utils.tools import get_builtin_tools

        _builtin = get_builtin_tools()
        if isinstance(_builtin, dict):
            _ingest(_builtin)
    except Exception as e:
        print(f"warning: could not load open_webui.utils.tools.get_builtin_tools: {e}", file=sys.stderr)

    # 3. An explicit JSON source (offline / test path): {"name": {"spec": ...}}.
    source = os.environ.get("TITO_TOOLS_SOURCE", "").strip()
    if source and os.path.exists(source):
        try:
            data = json.loads(Path(source).read_text(encoding="utf-8"))
            if isinstance(data, dict):
                _ingest(data)
            elif isinstance(data, list):
                for entry in data:
                    if isinstance(entry, dict) and entry.get("name"):
                        tools[entry["name"]] = entry
        except Exception as e:
            print(f"warning: could not read TITO_TOOLS_SOURCE: {e}", file=sys.stderr)

    return tools


def main() -> int:
    parser = argparse.ArgumentParser(description="Dump live tools to TITO_TOOLS_FILE")
    parser.add_argument(
        "--output",
        default=os.environ.get("TITO_TOOLS_FILE", str(REPO_ROOT / "tito_tools.json")),
        help="Output path (default: $TITO_TOOLS_FILE or ./tito_tools.json)",
    )
    args = parser.parse_args()

    raw = _collect_tools()
    anthropic_tools = []
    for name, spec in raw.items():
        converted = _to_anthropic_tool(name, spec)
        if converted is not None:
            anthropic_tools.append(converted)

    # Byte-stable order: the frozen array must not depend on dict insertion order.
    anthropic_tools.sort(key=lambda t: t["name"])

    if not anthropic_tools:
        print("error: no tools collected; refusing to write an empty TITO_TOOLS_FILE", file=sys.stderr)
        return 1

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(anthropic_tools, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    names = [t["name"] for t in anthropic_tools]
    print(f"wrote {len(anthropic_tools)} tools to {out}")
    print("tools: " + ", ".join(names))
    return 0


if __name__ == "__main__":
    sys.exit(main())
