"""Red tests for the Tito pipe patches (SPEC.md in this repo's tito-spec/).

These tests are the SPEC's executable contract. They FAIL today (red) and pass
only once `src/anthropic_pipe/tito.py` exists and payload/tools/messages/
cache_control/runtime/orchestrator are wired per SPEC. Behavioural checks drive
the real functions; pure-function checks drive `tito.py`.

Runs without OpenWebUI and without the `anthropic` SDK (matches the repo's other
helpers/test_*.py). Usage:

    python tito-spec/tests/test_tito.py

Exit 0 = all checks pass (green). Exit 1 = red.
"""
from __future__ import annotations

import asyncio
import contextvars as _cv
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "anthropic_pipe"

# Expected values, each sourced from tito-spec/SPEC.md (not "because the spec says"):
#   effort "medium" / thinking adaptive                          -> SPEC §2.1
#   advisor model opus-5-5, max_uses 3, caching 5m               -> SPEC §2.2
#   cache ttl 1h / no tools bp / 2 system bp                     -> SPEC §2.3 (P6)
#   mid-conversation top-level key + beta header                 -> SPEC §3 (C1/D2)
#   hard cap 22 / soft 15 + HARD_CAP_TEXT / TITO_ERROR_TEXT      -> SPEC §6/§7 (P3/P5)
TITO_EXPECT = {
    "base_model": "claude-sonnet-5-5",          # body model (no remap; C2)
    "effort": "medium",
    "thinking": {"type": "adaptive"},
    "advisor_model": "claude-opus-5-5",
    "advisor_max_uses": 3,
    "advisor_caching": {"type": "ephemeral", "ttl": "5m"},
    "beta": "mid-conversation-system-clear-at-2026-08-21",
    "hard_cap": 22.0,
    "soft_cap": 15.0,
    "cache_ttl": "1h",
    "hard_cap_text": "Tito đang nghỉ tới ngày 1 tháng sau. Dori đã biết rồi, ông không cần làm gì.",
    "error_text": "Tito đang bận một chút, ông thử lại sau vài phút nhé.",
}

CHECKS = []  # (name, ok, detail)


def record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def _load_module(name: str, path: Path):
    """Import a src module by path, or return None if the file is absent."""
    if not path.exists():
        return None
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


# =============================================================================
# Pure tito.py helpers (red today: module does not exist yet)
# =============================================================================
_tito = _load_module("_tito", SRC / "tito.py")

if _tito is None:
    record("tito module exists", False, "src/anthropic_pipe/tito.py missing")
else:
    record("tito module exists", True)
    Tito = getattr(_tito, "PipeTitoMethods", None)
    if Tito is None:
        record("PipeTitoMethods defined", False, "no PipeTitoMethods class")
    else:
        record("PipeTitoMethods defined", True)
        cfg = getattr(Tito, "TITO_CONFIG", None)
        record("TITO_CONFIG defined", cfg is not None)
        if cfg:
            record("effort == medium", cfg.get("effort") == "medium", repr(cfg.get("effort")))
            record("thinking adaptive", cfg.get("thinking") == {"type": "adaptive"})
            record("advisor model", cfg.get("advisor", {}).get("model") == "claude-opus-5-5")
            record("advisor max_uses 3", cfg.get("advisor", {}).get("max_uses") == 3)
            record("advisor caching 5m", cfg.get("advisor", {}).get("caching") == "5m")
            record("enable_tool_search off", cfg.get("enable_tool_search") is False)
            record("web_search_max_uses 10", cfg.get("web_search_max_uses") == 10)
            record("user_location Fresno", cfg.get("user_location") == {"city": "Fresno"})
            record("hide_blocks", cfg.get("hide_blocks") == ["advisor", "web_search", "web_fetch"])
            record("cache P6", cfg.get("cache") == {"ttl": "1h", "tools_breakpoint": False, "system_breakpoints": 2},
                   repr(cfg.get("cache")))
            record("show_cost on", cfg.get("show_cost") is True)

        _is_tito = getattr(Tito, "_is_tito", None)
        if _is_tito:
            record("_is_tito metadata id tito", _is_tito({"model": {"id": "tito"}}) is True)
            record("_is_tito metadata id base", _is_tito({"model": {"id": "claude-sonnet-5-5"}}) is False)
            record("_is_tito None", _is_tito(None) is False)

        _is_mid = getattr(Tito, "_is_mid_conversation_system", None)
        if _is_mid:
            record("mid_conversation top-level True",
                   _is_mid({"role": "system", "content": "x", "mid_conversation": True}))
            record("mid_conversation key absent", not _is_mid({"role": "system", "content": "x"}))

        _decide = getattr(Tito, "_tito_budget_decision", None)
        if _decide:
            h, htxt = _decide(TITO_EXPECT["hard_cap"])
            record("budget >= hard -> hard", h == "hard", f"{h!r}")
            record("hard text exact", htxt == TITO_EXPECT["hard_cap_text"])
            record("budget just under hard -> soft", _decide(TITO_EXPECT["hard_cap"] - 0.01)[0] == "soft")
            record("budget == soft -> soft", _decide(TITO_EXPECT["soft_cap"])[0] == "soft")
            record("budget < soft -> ok", _decide(TITO_EXPECT["soft_cap"] - 0.01)[0] == "ok")

        _pmonth = getattr(Tito, "_tito_pacific_month", None)
        if _pmonth:
            # America/Los_Angeles = UTC-8 (PST) winter / UTC-7 (PDT) summer.
            # Mid-Jan: 2026-01-15 00:00 UTC = 2026-01-14 16:00 PST -> "2026-01".
            record("pacific month winter", _pmonth(datetime(2026, 1, 15, 0, 0, tzinfo=timezone.utc)) == "2026-01")
            # Mid-Jul: 2026-07-15 00:00 UTC = 2026-07-14 17:00 PDT -> "2026-07".
            record("pacific month summer", _pmonth(datetime(2026, 7, 15, 0, 0, tzinfo=timezone.utc)) == "2026-07")
            # Jan-1 boundary: 08:00 UTC = 00:00 PST -> "2026-01"; 07:59 UTC = Dec 31 -> "2025-12".
            record("pacific month Jan-1", _pmonth(datetime(2026, 1, 1, 8, 0, tzinfo=timezone.utc)) == "2026-01")
            record("pacific month Dec-31", _pmonth(datetime(2026, 1, 1, 7, 59, tzinfo=timezone.utc)) == "2025-12")

        _frozen = getattr(Tito, "_tito_frozen_tools", None)
        if _frozen:
            tmp = None
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                json.dump([{"name": "view_file", "description": "d", "input_schema": {"type": "object"}}], fh)
                tmp = fh.name
            os.environ["TITO_TOOLS_FILE"] = tmp
            try:
                got = _frozen()
                record("frozen tools returns list", isinstance(got, list), repr(type(got)))
            finally:
                os.environ.pop("TITO_TOOLS_FILE", None)
                Path(tmp).unlink(missing_ok=True)

        _fp = getattr(Tito, "_tito_fingerprint", None)
        if _fp:
            p1 = {"tools": [{"name": "a"}], "system": [{"type": "text", "text": "s"}],
                  "model": "claude-sonnet-5-5", "thinking": {"type": "adaptive"},
                  "effort": "medium", "betas": ["b"]}
            p2 = dict(p1, system=[{"type": "text", "text": "DIFFERENT"}])
            f1, f2 = _fp(p1), _fp(p2)
            record("fingerprint stable", isinstance(f1, str) and f1 == _fp(p1))
            record("fingerprint changes with system", f1 != f2)

        _err = getattr(Tito, "_tito_error_text", None)
        if _err:
            record("tito error text exact", _err() == TITO_EXPECT["error_text"])


# =============================================================================
# Behavioural: create_request_payload honours tito (effort/thinking, no remap)
# =============================================================================
def _load_payload():
    payload = _load_module("_tito_payload", SRC / "request" / "payload.py")
    if payload is None:
        return None
    status = _load_module("_tito_status", SRC / "response" / "status_events.py")
    if status is not None:
        payload.StatusEmitter = status.StatusEmitter
    payload.FILES_AVAILABLE = False
    return payload


class FakeValves:
    def __init__(self):
        self.DATA_RESIDENCY = "global"
        self.ENABLE_FAST_MODE = False
        self.CACHE_CONTROL = "cache tools array, system prompt and messages"
        self.ENABLE_PROGRAMMATIC_TOOL_CALLING = False
        self.ENABLE_INTERLEAVED_THINKING = False
        self.WEB_FETCH = False
        self.ANTHROPIC_API_KEY = "sk-admin"
        self.REFUSAL_FALLBACK = "off"
        self.ENABLE_CACHE_DIAGNOSTICS = False
        self.ANTHROPIC_BASE_URL = "https://api.anthropic.com"


class FakeUserValves:
    def __init__(self):
        self.ENABLE_THINKING = False
        self.EFFORT = "high"                      # valve says high; tito must force medium
        self.THINKING_BUDGET_TOKENS = 8192
        self.THINKING_DISPLAY = "omitted"
        self.SKILLS = []
        self.USE_FILES_API = False
        self.USE_PDF_NATIVE_UPLOAD = False
        self.ENABLE_ADVISOR_TOOL = False          # valve says off; tito must still add advisor
        self.ENABLE_COMPACTION = False
        self.COMPACTION_TRIGGER_TOKENS = 50000
        self.COMPACTION_INSTRUCTIONS = ""


class FakePipe:
    API_VERSION = "2026-01-01"
    _DEFAULT_API_BASE = "https://api.anthropic.com"

    def __init__(self, processed_messages=None):
        self.valves = FakeValves()
        self._processed_messages = processed_messages if processed_messages is not None else []

    def _is_tito(self, metadata):
        return (metadata or {}).get("model", {}).get("id") == "tito"

    def get_model_info(self, name):
        # Sonnet-5-5 capability surface (SPEC D1: body model already carries the base).
        return {
            "max_tokens": 128000,
            "supports_effort": True,
            "supports_effort_xhigh": True,
            "supports_effort_max": True,
            "supports_thinking": True,
            "supports_adaptive_thinking": True,
            "thinking_on_by_default": True,
            "thinking_off_type": "between_tools",
        }

    def _convert_messages_to_claude_format(self, raw_messages):
        return [], list(self._processed_messages), []

    async def _get_full_context_texts(self, *a, **k):
        return {}, [], []

    def _convert_tools_to_claude_format(self, *a, **k):
        return [], set()

    def _canonicalize_block(self, block):
        return block


async def _run_payload_checks(payload):
    if payload is None:
        record("payload module loads", False, "src/anthropic_pipe/request/payload.py missing")
        return
    record("payload module loads", True)

    async def call(pipe, body, metadata, user_valves=None):
        user = {"valves": user_valves or FakeUserValves()}
        return await payload.create_request_payload(
            pipe, body, metadata, user, None, _noop, None
        )

    TITO_META = {"model": {"id": "tito"}}
    BASE_BODY = {"model": TITO_EXPECT["base_model"], "messages": [], "reasoning_effort": "high"}

    # (a) tito: effort medium + thinking adaptive, ignoring valve/reasoning_effort. No remap.
    p = FakePipe()
    out, _h, _m, _t = await call(p, BASE_BODY, TITO_META)
    record("no model remap (body stays base)", out.get("model") == TITO_EXPECT["base_model"],
           f"model={out.get('model')!r}")
    record("tito effort medium (ignores reasoning_effort/valve)", out.get("output_config") == {"effort": "medium"},
           f"output_config={out.get('output_config')!r}")
    record("tito thinking adaptive", out.get("thinking") == TITO_EXPECT["thinking"],
           f"thinking={out.get('thinking')!r}")

    # (b) mid-conversation system present -> beta header added.
    mid = {"role": "system", "content": [{"type": "text", "text": "status"}],
           "clear_at": "next_user_message", "mid_conversation": True}
    p2 = FakePipe(processed_messages=[mid])
    out2, h2, _m2, _t2 = await call(p2, BASE_BODY, TITO_META)
    betas = (h2.get("anthropic-beta", "") or "").split(",")
    record("mid-conversation beta header added", TITO_EXPECT["beta"] in betas, f"betas={betas!r}")


async def _noop(event):
    pass


# =============================================================================
# Behavioural: _convert_tools_to_claude_format adds frozen advisor for tito
# =============================================================================
def _run_tools_checks():
    tools = _load_module("_tito_tools", SRC / "request" / "tools.py")
    models = _load_module("_tito_models", SRC / "shared" / "models.py")
    if tools is None:
        record("tools module loads", False, "missing")
        return
    record("tools module loads", True)
    tools.json = json
    tools.normalize_tool_name_list = lambda value: set()

    class TV:
        ENABLE_BASH_TOOL = False
        ENABLE_TEXT_EDITOR_TOOL = False
        WEB_SEARCH = False
        WEB_FETCH = False
        ENABLE_PROGRAMMATIC_TOOL_CALLING = False

    class TUV:
        ENABLE_ADVISOR_TOOL = False
        ADVISOR_MODEL = "claude-opus-5"
        ADVISOR_MAX_USES = 0
        ADVISOR_CACHING = "off"
        ENABLE_DYNAMIC_FILTERING = False
        ENABLE_TOOL_SEARCH = False
        TOOL_SEARCH_EXCLUDE_TOOLS = ""
        TOOL_SEARCH_MAX_DESCRIPTION_LENGTH = 1000
        TOOL_SEARCH_TYPE = "bm25"

    class Pipe(tools.PipeRequestToolsMethods, models.PipeModelSupportMethods):
        valves = TV()

        def get_model_info(self, name):
            return {}

        def _is_tito(self, metadata):
            return (metadata or {}).get("model", {}).get("id") == "tito"

        def _tito_config(self):
            return {"advisor": {"enabled": True, "model": "claude-opus-5-5", "max_uses": 3, "caching": "5m"},
                    "enable_tool_search": False}

    got, _ = Pipe()._convert_tools_to_claude_format(
        __tools__={}, body={}, actual_model_name=TITO_EXPECT["base_model"],
        __user__={"valves": TUV()}, __metadata__={"model": {"id": "tito"}},
    )
    advisor = next((t for t in got if t.get("name") == "advisor"), None)
    record("advisor present for tito (valve off)", advisor is not None)
    if advisor is not None:
        record("advisor model opus-5-5", advisor.get("model") == TITO_EXPECT["advisor_model"], repr(advisor.get("model")))
        record("advisor max_uses 3", advisor.get("max_uses") == TITO_EXPECT["advisor_max_uses"], repr(advisor.get("max_uses")))
        record("advisor caching 5m", advisor.get("caching") == TITO_EXPECT["advisor_caching"], repr(advisor.get("caching")))


# =============================================================================
# Behavioural: cache breakpoints (P6 layout + skip role:system + 1h ttl)
# =============================================================================
def _run_cache_checks():
    cc = _load_module("_tito_cache", SRC / "request" / "cache_control.py")
    if cc is None:
        record("cache_control module loads", False, "missing")
        return
    record("cache_control module loads", True)
    cc.Pipe = type("Pipe", (), {"_dump_sdk_obj": staticmethod(lambda o: o),
                                "_first_volatile_block_index": staticmethod(lambda m: None)})
    cc.MEMORY_CONTEXT_APPENDIX_HEADER = "\n\n---\nMEMORY"

    class CV:
        CACHE_TTL = "5 minutes"                # default; tito must override to 1h
        CACHE_TTL_FOR_TOOLS_AND_SYSTEM_PROMT = "same as CACHE_TTL"
        CACHE_CONTROL = "cache tools array, system prompt and messages"
        ENABLE_PROGRAMMATIC_TOOL_CALLING = False

    class Pipe(cc.PipeCacheControlMethods):
        valves = CV()
        TITO_ACTIVE = _cv.ContextVar("test_tito_active", default=False)

    # P6: full _apply_cache_control for tito.
    payload = {
        "tools": [{"name": "web_search", "type": "web_search_20250305"}],
        "system": [{"type": "text", "text": "block1"}, {"type": "text", "text": "block2"}],
        "messages": [{"role": "user", "content": [{"type": "text", "text": "hi"}]}],
    }
    p = Pipe()
    p.TITO_ACTIVE.set(True)
    p._apply_cache_control(payload, is_tool_loop=False, iteration=1)

    tools_cc = any("cache_control" in t for t in payload.get("tools", []))
    sys_cc = [("cache_control" in b) for b in payload["system"]]
    msg_cc = any("cache_control" in b for b in payload["messages"][-1]["content"])
    record("P6 no tools breakpoint for tito", not tools_cc)
    record("P6 both system blocks cached", sys_cc == [True, True], repr(sys_cc))
    record("P6 system breakpoint 1h ttl",
           payload["system"][0].get("cache_control") == {"type": "ephemeral", "ttl": TITO_EXPECT["cache_ttl"]},
           repr(payload["system"][0].get("cache_control")))
    record("P6 last-message breakpoint present", msg_cc)

    # PATCH D/H3: messages breakpoint must skip a mid_conversation system message.
    user_msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    status_msg = {"role": "system", "content": [{"type": "text", "text": "status"}],
                  "clear_at": "next_user_message", "mid_conversation": True}
    msgs = [dict(user_msg), dict(status_msg)]
    Pipe()._cache_last_stable_message(msgs)

    def has_cc(msg):
        return any("cache_control" in b for b in msg.get("content", []) if isinstance(b, dict))

    record("breakpoint NOT on status system msg", not has_cc(msgs[1]))
    record("breakpoint ON preceding user msg", has_cc(msgs[0]))


def main():
    _run_tools_checks()
    _run_cache_checks()
    asyncio.run(_run_payload_checks(_load_payload()))

    failed = [c for c in CHECKS if not c[1]]
    for name, ok, detail in CHECKS:
        print(f"{'PASS' if ok else 'FAIL'}: {name}" + (f"  [{detail}]" if detail else ""))
    print(f"\n{len(CHECKS) - len(failed)}/{len(CHECKS)} checks passed")
    if failed:
        print("\nRED: the following SPEC contract is not yet met:")
        for name, _ok, detail in failed:
            print(f"  - {name}" + (f"  [{detail}]" if detail else ""))
        sys.exit(1)
    print("All tito checks passed")


if __name__ == "__main__":
    main()
