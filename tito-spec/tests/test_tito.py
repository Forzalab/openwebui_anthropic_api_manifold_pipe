"""Red tests for the Tito pipe patches (SPEC.md in this repo's tito-spec/).

These tests are the SPEC's executable contract. They FAIL today (red) and pass
only once `src/anthropic_pipe/tito.py` exists and payload/tools/messages/
cache_control/orchestrator are wired per SPEC. Behavioural checks drive the real
functions with a `"tito"` model id; pure-function checks drive `tito.py`.

Runs without OpenWebUI and without the `anthropic` SDK (matches the repo's other
helpers/test_*.py). Usage:

    python tito-spec/tests/test_tito.py

Exit 0 = all checks pass (green). Exit 1 = red.
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "src" / "anthropic_pipe"

# Expected values, each sourced from tito-spec/SPEC.md (not "because the spec says"):
#   effort "medium" / thinking adaptive / model claude-sonnet-5-5   -> SPEC §2.1
#   advisor model opus-5-5, max_uses 3, caching 5m                  -> SPEC §2.2
#   mid-conversation beta header                                    -> SPEC §3
#   hard/soft caps 25/15 + HARD_CAP_TEXT                            -> SPEC §6
TITO_EXPECT = {
    "base_model": "claude-sonnet-5-5",
    "effort": "medium",
    "thinking": {"type": "adaptive"},
    "advisor_model": "claude-opus-5-5",
    "advisor_max_uses": 3,
    "advisor_caching": {"type": "ephemeral", "ttl": "5m"},
    "beta": "mid-conversation-system-clear-at-2026-08-21",
    "hard_cap": 25.0,
    "soft_cap": 15.0,
    "hard_cap_text": "Tito đang nghỉ tới ngày 1 tháng sau. Dori đã biết rồi, ông không cần làm gì.",
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
            record("base_model == claude-sonnet-5-5", cfg.get("base_model") == "claude-sonnet-5-5")
            record("advisor model", cfg.get("advisor", {}).get("model") == "claude-opus-5-5")
            record("advisor max_uses 3", cfg.get("advisor", {}).get("max_uses") == 3)
            record("advisor caching 5m", cfg.get("advisor", {}).get("caching") == "5m")
            record("web_search_max_uses 10", cfg.get("web_search_max_uses") == 10)
            record("user_location Fresno", cfg.get("user_location") == {"city": "Fresno"})
            record("cache TTL 1h both", cfg.get("cache") == {"tools_system_ttl": "1h", "messages_ttl": "1h"})

        _is_tito = getattr(Tito, "_is_tito", None)
        if _is_tito:
            record("_is_tito('tito')", _is_tito("tito") is True)
            record("_is_tito('claude-sonnet-5-5')", _is_tito("claude-sonnet-5-5") is False)
            record("_is_tito('anthropic/tito')", _is_tito("anthropic/tito") is True)

        _is_mid = getattr(Tito, "_is_mid_conversation_system", None)
        if _is_mid:
            record("mid_conversation marker True", _is_mid({"role": "system", "meta": {"mid_conversation": True}}))
            record("mid_conversation marker absent", not _is_mid({"role": "system", "content": "x"}))

        _decide = getattr(Tito, "_tito_budget_decision", None)
        if _decide:
            h, htxt = _decide(25.0)
            record("budget >=25 -> hard", h == "hard" and htxt == TITO_EXPECT["hard_cap_text"], f"{h!r} {htxt!r}")
            record("budget >=25 text exact", htxt == TITO_EXPECT["hard_cap_text"])
            s, stxt = _decide(15.0)
            record("budget 15 -> soft", s == "soft" and stxt is None, f"{s!r}")
            record("budget 14.99 -> ok", _decide(14.99)[0] == "ok")

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
            with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as fh:
                json.dump([{"name": "x", "description": "x", "input_schema": {"type": "object"}}], fh)
                tmp = fh.name
            import os
            os.environ["TITO_TOOLS_FILE"] = tmp
            try:
                record("frozen tools returns list", isinstance(_frozen(), list))
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


# =============================================================================
# Behavioural: create_request_payload honours tito (effort/thinking/model)
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

    def get_model_info(self, name):
        # Sonnet-5-5 capability surface (SPEC D2: base model drives capabilities).
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

    async def call(pipe, body, metadata=None, user_valves=None):
        user = {"valves": user_valves or FakeUserValves()}
        return await payload.create_request_payload(
            pipe, body, metadata or {}, user, None, _noop, None
        )

    # (a) tito: model remap + effort medium + thinking adaptive, ignoring valve/reasoning_effort.
    p = FakePipe()
    body = {"model": "anthropic/tito", "messages": [], "reasoning_effort": "high"}
    out, _h, _m, _t = await call(p, body)
    record("tito remaps model -> claude-sonnet-5-5", out.get("model") == TITO_EXPECT["base_model"],
           f"model={out.get('model')!r}")
    record("tito effort medium (ignores reasoning_effort/valve)", out.get("output_config") == {"effort": "medium"},
           f"output_config={out.get('output_config')!r}")
    record("tito thinking adaptive", out.get("thinking") == TITO_EXPECT["thinking"],
           f"thinking={out.get('thinking')!r}")

    # (b) mid-conversation system present -> beta header added.
    mid = {"role": "system", "content": [{"type": "text", "text": "status"}],
           "clear_at": "next_user_message", "meta": {"mid_conversation": True}}
    p2 = FakePipe(processed_messages=[mid])
    out2, h2, _m2, _t2 = await call(p2, {"model": "anthropic/tito", "messages": []})
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

        def _is_tito(self, name):
            return name == "tito"

        def _tito_config(self):
            return {
                "advisor": {"enabled": True, "model": "claude-opus-5-5", "max_uses": 3, "caching": "5m"},
                "enable_tool_search": False,
            }

    got, _ = Pipe()._convert_tools_to_claude_format(
        __tools__={}, body={}, actual_model_name="tito",
        __user__={"valves": TUV()}, __metadata__={},
    )
    advisor = next((t for t in got if t.get("name") == "advisor"), None)
    record("advisor present for tito (valve off)", advisor is not None)
    if advisor is not None:
        record("advisor model opus-5-5", advisor.get("model") == TITO_EXPECT["advisor_model"], repr(advisor.get("model")))
        record("advisor max_uses 3", advisor.get("max_uses") == TITO_EXPECT["advisor_max_uses"], repr(advisor.get("max_uses")))
        record("advisor caching 5m", advisor.get("caching") == TITO_EXPECT["advisor_caching"], repr(advisor.get("caching")))


# =============================================================================
# Behavioural: messages cache breakpoint skips role:system
# =============================================================================
def _run_cache_checks():
    cc = _load_module("_tito_cache", SRC / "request" / "cache_control.py")
    if cc is None:
        record("cache_control module loads", False, "missing")
        return
    record("cache_control module loads", True)
    # Inject names the module resolves from class Pipe / template scope at call time.
    cc.Pipe = type("Pipe", (), {"_dump_sdk_obj": staticmethod(lambda o: o),
                                "_first_volatile_block_index": staticmethod(lambda m: None)})
    cc.MEMORY_CONTEXT_APPENDIX_HEADER = "\n\n---\nMEMORY"

    class CV:
        CACHE_TTL = "1 hour"
        CACHE_TTL_FOR_TOOLS_AND_SYSTEM_PROMT = "same as CACHE_TTL"
        CACHE_CONTROL = "cache tools array, system prompt and messages"

    class Pipe(cc.PipeCacheControlMethods):
        valves = CV()

    user_msg = {"role": "user", "content": [{"type": "text", "text": "hi"}]}
    status_msg = {"role": "system", "content": [{"type": "text", "text": "status"}], "clear_at": "next_user_message"}
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
