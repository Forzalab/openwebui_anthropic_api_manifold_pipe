"""Self-check for Tito's per-turn VN reminder (valve TITO_VN_REMINDER, default "" = off).

The reminder is one text block appended to the last user message of the first
request of a turn, AFTER the cache breakpoints. Checks: an empty valve leaves the
payload unchanged; a set valve with Tito active appends the block without
cache_control and leaves the cached prefix bytes (tools, system, messages up to
and including the breakpoint block) identical to the run without a reminder; a
str last user message is converted to a list and the tools/system prefix stays
identical; Tito inactive changes nothing; removing the block restores the
payload exactly and a later tool-loop iteration carries no reminder; the
reminder text never reaches a log record (only its length does).

Runs without OpenWebUI, without the anthropic SDK and without network access.

Usage:
    python helpers/test_tito_vn_reminder.py
"""
from __future__ import annotations

import copy
import importlib.util
import json
import logging
import sys
import typing
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"
VN_TEXT = "SECRET-VN-REMINDER-TEXT: remember the family rules"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _load_mixin(path: Path) -> dict:
    # cache_control.py has no imports for typing names it uses in annotations;
    # the compiled artifact provides them at module scope, so do the same here.
    ns = {
        "__name__": "_cache_control_under_test",
        "Any": typing.Any,
        "Optional": typing.Optional,
        "Dict": typing.Dict,
        "List": typing.List,
        "Tuple": typing.Tuple,
        # Same text as MEMORY_CONTEXT_APPENDIX_HEADER in pipe_template.py.
        "MEMORY_CONTEXT_APPENDIX_HEADER": (
            "\n\n---\n**IMPORTANT:** The following is NOT part of the user's message, "
            "but context from a memory system to help answer the user's questions:\n\n"
        ),
    }
    exec(compile(path.read_text(encoding="utf-8"), str(path), "exec"), ns)
    return ns


TITO = _load("_tito", SRC / "tito.py")
CC = _load_mixin(SRC / "request" / "cache_control.py")


class FakePipe(CC["PipeCacheControlMethods"], TITO.PipeTitoMethods):
    pass


def base_payload(str_last: bool = False) -> dict:
    last = "second question" if str_last else [{"type": "text", "text": "second question"}]
    return {
        "model": "claude-test",
        "tools": [{"name": "t1", "description": "d", "input_schema": {"type": "object"}}],
        "system": [{"type": "text", "text": "SYSTEM PROMPT"}],
        "messages": [
            {"role": "user", "content": [{"type": "text", "text": "first question"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "first answer"}]},
            {"role": "user", "content": last},
        ],
    }


def make_pipe(valve_text: str) -> FakePipe:
    pipe = FakePipe()
    pipe.valves = SimpleNamespace(
        CACHE_CONTROL="cache tools array, system prompt and messages",
        CACHE_TTL="1 hour",
        CACHE_TTL_FOR_TOOLS_AND_SYSTEM_PROMT="same as CACHE_TTL",
        ENABLE_PROGRAMMATIC_TOOL_CALLING=False,
        TITO_VN_REMINDER=valve_text,
    )
    pipe.TOOL_LOOP_VOLATILE_CACHE_MIN_ITERATION = 2
    return pipe


def prepared(valve_text: str, active: bool, str_last: bool = False):
    """Mirror the orchestrator's iteration-1 order: cache breakpoints, then reminder."""
    payload = base_payload(str_last)
    pipe = make_pipe(valve_text)
    token = TITO.PipeTitoMethods.TITO_ACTIVE.set(active)
    try:
        pipe._apply_cache_control(payload, is_tool_loop=False, iteration=1)
        state = pipe._tito_vn_reminder_apply(payload) if active else None
    finally:
        TITO.PipeTitoMethods.TITO_ACTIVE.reset(token)
    return payload, state, pipe


def cached_prefix(payload: dict) -> str:
    """Bytes of tools, system and messages up to and including the last breakpoint."""
    msgs = payload["messages"]
    found = None
    for mi, msg in enumerate(msgs):
        if isinstance(msg["content"], list):
            for bi, block in enumerate(msg["content"]):
                if "cache_control" in block:
                    found = (mi, bi)
    assert found is not None, "no message breakpoint in payload"
    mi, bi = found
    cut = {
        "tools": payload["tools"],
        "system": payload["system"],
        "messages": msgs[:mi] + [{"role": msgs[mi]["role"], "content": msgs[mi]["content"][: bi + 1]}],
    }
    return json.dumps(cut, sort_keys=False, ensure_ascii=False)


def main() -> int:
    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    TITO.logger.addHandler(_Capture())
    TITO.logger.setLevel(logging.DEBUG)

    # 1. Valve empty: payload identical to the no-reminder baseline, no state.
    baseline, _, _ = prepared("", True)
    empty, state, _ = prepared("", True)
    assert state is None and empty == baseline, "empty valve changed the payload"

    # 2. Valve set, Tito active: reminder appended after the breakpoint.
    records.clear()
    p, state, _ = prepared(f"  {VN_TEXT}\n", True)
    assert state is not None, "reminder not applied"
    last_content = p["messages"][-1]["content"]
    assert last_content[-1] == {"type": "text", "text": VN_TEXT}, last_content[-1]
    assert "cache_control" not in last_content[-1], "reminder carries cache_control"
    assert last_content[:-1] == baseline["messages"][-1]["content"], "breakpoint block changed"
    assert p["messages"][:-1] == baseline["messages"][:-1], "earlier messages changed"
    assert p["tools"] == baseline["tools"] and p["system"] == baseline["system"]
    assert cached_prefix(p) == cached_prefix(baseline), "cached prefix bytes changed"
    assert "cache_control" in baseline["messages"][-1]["content"][-1], "baseline breakpoint missing"
    assert records and any(r == f"[TITO] vn_reminder chars={len(VN_TEXT)}" for r in records), records
    assert not any(VN_TEXT in r for r in records), "reminder text leaked into a log record"

    # 3. str last user message: converted to a list, reminder last, tools/system and
    #    earlier messages untouched. Only the tools/system breakpoints exist here.
    str_base, _, _ = prepared("", True, str_last=True)
    assert str_base["messages"][-1]["content"] == "second question"
    p_str, state, _ = prepared(VN_TEXT, True, str_last=True)
    assert state is not None
    assert p_str["messages"][-1]["content"] == [
        {"type": "text", "text": "second question"},
        {"type": "text", "text": VN_TEXT},
    ], p_str["messages"][-1]["content"]
    assert p_str["messages"][:-1] == str_base["messages"][:-1]
    assert p_str["tools"] == str_base["tools"] and p_str["system"] == str_base["system"]
    assert "cache_control" not in p_str["messages"][-1]["content"][-1]

    # 4. Tito inactive: unchanged even with the valve set.
    off_base, _, _ = prepared("", False)
    off, state, _ = prepared(VN_TEXT, False)
    assert state is None and off == off_base, "inactive Tito changed the payload"
    assert len(off["messages"][-1]["content"]) == 1, "inactive Tito appended a block"

    # 5. Removal before iteration 2 restores the payload exactly (list and str case),
    #    and a tool-loop request built afterwards carries no reminder text.
    p, state, pipe = prepared(VN_TEXT, True)
    pipe._tito_vn_reminder_remove(p, state)
    assert p == baseline, "removal did not restore the payload"
    p["messages"].append({"role": "assistant", "content": [{"type": "tool_use", "id": "u1", "name": "t1", "input": {}}]})
    p["messages"].append({"role": "user", "content": [{"type": "tool_result", "tool_use_id": "u1", "content": "ok"}]})
    token = TITO.PipeTitoMethods.TITO_ACTIVE.set(True)
    try:
        pipe._apply_cache_control(p, is_tool_loop=True, iteration=2)
    finally:
        TITO.PipeTitoMethods.TITO_ACTIVE.reset(token)
    assert VN_TEXT not in json.dumps(p, ensure_ascii=False), "reminder survived into iteration 2"

    p_str, state, pipe = prepared(VN_TEXT, True, str_last=True)
    pipe._tito_vn_reminder_remove(p_str, state)
    assert p_str == str_base, "str removal did not restore the payload"

    print("ok: tito vn reminder")
    return 0


if __name__ == "__main__":
    sys.exit(main())
