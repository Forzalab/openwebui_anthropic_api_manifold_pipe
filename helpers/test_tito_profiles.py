"""Regression test for the per-model Tito profiles (``tito`` and ``mimi``).

Covers: profile lookup, guard and decide skipped for mimi but on for tito, mimi
caps from the env defaults (10 / 6), the spend filter (tito sums all messages,
mimi only its own), mimi texts on the error and hard-cap paths, the tools file
fallback, the VN reminder valve, the log tag and alert label, and that a model
with no profile gets no Tito behaviour.

Runs without OpenWebUI, without the anthropic SDK and without network access.

Usage:
    python helpers/test_tito_profiles.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TITO = _load("_tito_profiles", SRC / "tito.py")
PM = TITO.PipeTitoMethods

TITO_META = {"model": {"id": "tito"}}
MIMI_META = {"model": {"id": "mimi"}}
OTHER_META = {"model": {"id": "claude-opus-5-5"}}

MIMI_HARD = "Mimi đi nghỉ tới ngày 1 tháng sau nha bạn. Tùng biết rồi, bạn không cần làm gì hết."
MIMI_ERROR = "Mimi đang bận xíu, bạn thử lại sau vài phút nha."
MIMI_MISSING = "Công cụ này tạm thời chưa dùng được, bạn thử lại lát nữa nha."

CAP_ENVS = ("TITO_HARD_CAP_USD", "TITO_SOFT_CAP_USD", "MIMI_HARD_CAP_USD", "MIMI_SOFT_CAP_USD")


class FakeValves:
    TITO_OUTPUT_GUARD = True
    TITO_DECIDE = True
    TITO_DECIDE_KEY = ""
    TITO_DECIDE_TIMEOUT_MS = 1500
    TITO_DECIDE_INJECT = False
    TITO_VN_REMINDER = ""
    MIMI_VN_REMINDER = ""
    TITO_IMAGE_MAX_PX = 0


class FakePipe(PM):
    def __init__(self, **valve_overrides):
        self.valves = SimpleNamespace(**{**vars(FakeValves), **valve_overrides})


def use_request(metadata) -> None:
    """Mirror pipe_orchestrator: set TITO_ACTIVE and TITO_PROFILE for this request."""
    PM.TITO_ACTIVE.set(bool(PM._is_tito(metadata)))
    PM.TITO_PROFILE.set(PM._tito_profile(metadata))


def clean_env(**extra) -> None:
    for name in CAP_ENVS + ("TITO_TOOLS_FILE", "MIMI_TOOLS_FILE", "TITO_DECIDE_KEY",
                            "TITO_DISCORD_WEBHOOK"):
        os.environ.pop(name, None)
    os.environ.update(extra)


# ----------------------------------------------------------------- lookup
def test_profile_lookup():
    assert PM._tito_profile(TITO_META)["id"] == "tito"
    assert PM._tito_profile(MIMI_META)["id"] == "mimi"
    assert PM._tito_profile(OTHER_META) is None
    assert PM._tito_profile(None) is None
    assert PM._tito_profile({"model": None}) is None
    assert PM._is_tito(TITO_META) and PM._is_tito(MIMI_META)
    assert not PM._is_tito(OTHER_META) and not PM._is_tito({})


def test_profile_values_match_spec():
    tito, mimi = PM.TITO_PROFILES["tito"], PM.TITO_PROFILES["mimi"]
    assert tito["guard"] is True and tito["decide"] is True
    assert mimi["guard"] is False and mimi["decide"] is False
    assert tito["vn_reminder_valve"] == "TITO_VN_REMINDER"
    assert mimi["vn_reminder_valve"] == "MIMI_VN_REMINDER"
    assert tito["hard_cap_default"] == 22.0 and tito["soft_cap_default"] == 15.0
    assert mimi["hard_cap_default"] == 10.0 and mimi["soft_cap_default"] == 6.0
    assert tito["spend_model"] is None and mimi["spend_model"] == "mimi"
    assert tito["tools_env"] == "TITO_TOOLS_FILE" and mimi["tools_env"] == "MIMI_TOOLS_FILE"
    assert tito["label"] == "Tito" and mimi["label"] == "Mimi"
    assert tito["error_text"] == PM.TITO_ERROR_TEXT
    assert tito["hard_cap_text"] == PM.HARD_CAP_TEXT
    assert tito["missing_tool_text"] == PM.TITO_MISSING_TOOL_TEXT
    assert mimi["error_text"] == MIMI_ERROR
    assert mimi["hard_cap_text"] == MIMI_HARD
    assert mimi["missing_tool_text"] == MIMI_MISSING


def test_label_and_tag_follow_profile():
    use_request(MIMI_META)
    assert PM._tito_tag() == "[MIMI]" and PM._tito_label() == "Mimi"
    use_request(TITO_META)
    assert PM._tito_tag() == "[TITO]" and PM._tito_label() == "Tito"
    use_request(OTHER_META)
    assert PM._tito_active_profile()["id"] == "tito"  # no profile set -> tito's values


# ----------------------------------------------------------------- guard + decide
def test_guard_skipped_for_mimi_on_for_tito():
    calls = []

    class Ctx:
        timing: dict = {}

        def text(self):
            calls.append("text")
            raise RuntimeError("stop after the gate")

    async def run(meta):
        use_request(meta)
        return await FakePipe()._tito_output_guard(Ctx(), None, meta)

    assert asyncio.run(run(MIMI_META)) == "none"
    assert calls == [], "guard read the reply for mimi"
    asyncio.run(run(TITO_META))
    assert calls == ["text"], "guard must still run for tito"


def test_decide_skipped_for_mimi_on_for_tito():
    clean_env(TITO_DECIDE_KEY="test-key")
    try:
        use_request(MIMI_META)
        assert FakePipe()._tito_decide_enabled() is False
        assert FakePipe()._tito_decide_start({"messages": [{"role": "user", "content": "hi"}]}) is None
        assert FakePipe()._tito_decide_inject_on() is False

        use_request(TITO_META)
        assert FakePipe()._tito_decide_enabled() is True
    finally:
        os.environ.pop("TITO_DECIDE_KEY", None)


def test_decide_out_start_gated_by_profile_and_model():
    clean_env(TITO_DECIDE_KEY="test-key")
    try:
        class Ctx:
            def text(self):
                return "Một câu trả lời bằng tiếng Việt."

        use_request(MIMI_META)
        assert FakePipe()._tito_decide_out_start(Ctx(), MIMI_META) is None
        use_request(OTHER_META)
        assert FakePipe()._tito_decide_out_start(Ctx(), OTHER_META) is None
    finally:
        os.environ.pop("TITO_DECIDE_KEY", None)


# ----------------------------------------------------------------- caps
def test_budget_caps_tito_and_mimi_defaults():
    clean_env()
    use_request(MIMI_META)
    assert PM._tito_budget_decision(5.99) == ("ok", None)
    assert PM._tito_budget_decision(6.0) == ("soft", None)
    assert PM._tito_budget_decision(9.99) == ("soft", None)
    assert PM._tito_budget_decision(10.0) == ("hard", MIMI_HARD)

    use_request(TITO_META)
    assert PM._tito_budget_decision(14.99) == ("ok", None)
    assert PM._tito_budget_decision(15.0) == ("soft", None)
    assert PM._tito_budget_decision(21.99) == ("soft", None)
    assert PM._tito_budget_decision(22.0) == ("hard", PM.HARD_CAP_TEXT)


def test_mimi_caps_read_their_own_env():
    clean_env(MIMI_HARD_CAP_USD="4", TITO_HARD_CAP_USD="50")
    try:
        use_request(MIMI_META)
        assert PM._tito_budget_decision(4.0)[0] == "hard"
        use_request(TITO_META)
        assert PM._tito_budget_decision(4.0)[0] == "ok"
    finally:
        clean_env()


# ----------------------------------------------------------------- spend filter
class FakeChats:
    @staticmethod
    def get_chats():
        now = datetime.now(timezone.utc).timestamp()
        msgs = {
            "a": {"timestamp": now, "model": "tito", "usage": {"cost_usd": 3.0}},
            "b": {"timestamp": now, "model": "mimi", "usage": {"cost_usd": 2.0}},
            "c": {"timestamp": now, "info": {"model": "mimi"}, "usage": {"cost_usd": 1.0}},
            "d": {"timestamp": 0, "model": "mimi", "usage": {"cost_usd": 50.0}},  # old month
        }
        return [{"chat": {"history": {"messages": msgs}}}]


def test_spend_filter_per_profile():
    TITO.Chats = FakeChats
    TITO.CHATS_AVAILABLE = True
    try:
        now = datetime.now(timezone.utc)
        pipe = FakePipe()  # one instance: the spend cache lives on the instance
        use_request(TITO_META)
        tito_total = asyncio.run(pipe.month_spend_usd(now))
        assert abs(tito_total - 6.0) < 1e-9, tito_total  # all models this month

        use_request(MIMI_META)
        mimi_total = asyncio.run(pipe.month_spend_usd(now))
        assert abs(mimi_total - 3.0) < 1e-9, mimi_total  # mimi messages only (msg.model or info.model)

        # Each profile has its own cache: asking tito again must not return mimi's total.
        use_request(TITO_META)
        assert abs(asyncio.run(pipe.month_spend_usd(now)) - 6.0) < 1e-9
    finally:
        del TITO.Chats
        TITO.CHATS_AVAILABLE = False


# ----------------------------------------------------------------- texts
def test_mimi_texts_on_error_and_missing_tool_paths():
    use_request(MIMI_META)
    assert PM._tito_error_text() == MIMI_ERROR
    assert PM._tito_missing_tool_result("tu1", "x")["content"] == MIMI_MISSING
    use_request(TITO_META)
    assert PM._tito_error_text() == PM.TITO_ERROR_TEXT
    assert PM._tito_missing_tool_result("tu1", "x")["content"] == PM.TITO_MISSING_TOOL_TEXT


# ----------------------------------------------------------------- tools file
def test_tools_file_fallback_and_per_file_alert():
    clean_env()
    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8")
    json.dump([{"name": "lich", "description": "x", "input_schema": {"type": "object"}}], tmp)
    tmp.close()
    try:
        os.environ["TITO_TOOLS_FILE"] = tmp.name
        use_request(MIMI_META)  # MIMI_TOOLS_FILE unset -> falls back to TITO_TOOLS_FILE
        assert [t["name"] for t in PM._tito_frozen_tools()] == ["lich"]

        os.environ["MIMI_TOOLS_FILE"] = os.path.join(tempfile.gettempdir(), "no_such_mimi_tools.json")
        assert PM._tito_frozen_tools() == [], "a missing MIMI_TOOLS_FILE must not fall back"

        use_request(TITO_META)  # tito still reads TITO_TOOLS_FILE
        assert [t["name"] for t in PM._tito_frozen_tools()] == ["lich"]
    finally:
        clean_env()
        os.unlink(tmp.name)


def test_tools_alert_is_per_file():
    clean_env()
    alerts = []
    original = PM.__dict__["_tito_discord_alert"]
    PM._tito_discord_alert = classmethod(lambda cls, message: alerts.append(message))
    PM._tito_tools_alert_paths.clear()
    try:
        missing_a = os.path.join(tempfile.gettempdir(), "tito_profiles_missing_a.json")
        missing_b = os.path.join(tempfile.gettempdir(), "tito_profiles_missing_b.json")
        os.environ["TITO_TOOLS_FILE"] = missing_a
        use_request(TITO_META)
        PM._tito_frozen_tools()
        PM._tito_frozen_tools()  # same file: no second alert
        assert len(alerts) == 1 and alerts[0].startswith("Tito: frozen tools unavailable"), alerts
        os.environ["MIMI_TOOLS_FILE"] = missing_b
        use_request(MIMI_META)
        PM._tito_frozen_tools()  # different file: its own alert, labelled Mimi
        assert len(alerts) == 2 and alerts[1].startswith("Mimi: frozen tools unavailable"), alerts
    finally:
        PM._tito_discord_alert = original
        PM._tito_tools_alert_paths.clear()
        clean_env()


# ----------------------------------------------------------------- VN reminder
def test_vn_reminder_valve_per_profile():
    clean_env()

    def payload():
        return {"messages": [{"role": "user", "content": "chào"}]}

    use_request(MIMI_META)
    assert FakePipe(TITO_VN_REMINDER="tito reminder")._tito_vn_reminder_apply(payload()) is None, \
        "mimi must not read the tito reminder valve"
    assert FakePipe(MIMI_VN_REMINDER="mimi reminder")._tito_vn_reminder_apply(payload()) is not None

    use_request(TITO_META)
    assert FakePipe(TITO_VN_REMINDER="tito reminder")._tito_vn_reminder_apply(payload()) is not None
    assert FakePipe(MIMI_VN_REMINDER="mimi reminder")._tito_vn_reminder_apply(payload()) is None


# ----------------------------------------------------------------- other models
def test_other_model_has_no_tito_behaviour():
    clean_env(TITO_DECIDE_KEY="test-key")
    try:
        use_request(OTHER_META)
        pipe = FakePipe()
        assert not PM.TITO_ACTIVE.get() and PM.TITO_PROFILE.get() is None
        assert pipe._tito_vn_reminder_apply({"messages": [{"role": "user", "content": "x"}]}) is None
    finally:
        os.environ.pop("TITO_DECIDE_KEY", None)


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
