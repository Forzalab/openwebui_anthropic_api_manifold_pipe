"""Self-check for Mimi's one-turn injections (profile ``injects``).

Goose: first user turn of a chat, once per person (no earlier chat of that user
with a mimi message). Bear: about 1 in MIMI_BEAR_RATE chats, deterministic by
sha256(chat_id). Goose wins over bear. Checks: goose on the first-ever turn; no
goose when an earlier mimi chat exists; no goose on a Chats error; bear on the
chat id whose hash % 10 == 0 and not on one where it is not; rate 0 is off;
nothing on a second user turn; tito, an unknown model and the call-site gate
never reach the injection; the payload is not changed.

Runs without OpenWebUI, without the anthropic SDK and without network access.

Usage:
    python helpers/test_mimi_inject.py
"""
from __future__ import annotations

import asyncio
import copy
import hashlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TITO = _load("_mimi_inject", SRC / "tito.py")
PM = TITO.PipeTitoMethods

TITO_META = {"model": {"id": "tito"}, "chat_id": "c-tito"}
MIMI_META = {"model": {"id": "mimi"}, "chat_id": "c-now"}
OTHER_META = {"model": {"id": "claude-opus-5-5"}, "chat_id": "c-other"}
USER = {"id": "user-1"}
OTHER_USER = {"id": "user-2"}

GOOSE = (
    "First meeting with bạn. After the answer, Mimi mentions her new friend once, warmly and briefly: "
    "a white goose, soft and squishy, whose neck flops when hugged."
)
BEAR = (
    "If the mood is light and there is room, Mimi adds one short line of news from the bear forest, "
    "her neighbors in the Đà Lạt pines. In health, money or sad moments, the answer and bạn come first "
    "and the forest stays quiet."
)


class FakeValves:
    MIMI_GOOSE_INJECT = GOOSE
    MIMI_BEAR_INJECT = BEAR
    MIMI_BEAR_RATE = 10
    MIMI_VN_REMINDER = ""
    TITO_VN_REMINDER = ""


class FakePipe(PM):
    def __init__(self, **valve_overrides):
        self.valves = SimpleNamespace(**{**vars(FakeValves), **valve_overrides})


def use_request(metadata) -> None:
    """Mirror pipe_orchestrator: set TITO_ACTIVE and TITO_PROFILE for this request."""
    PM.TITO_ACTIVE.set(bool(PM._is_tito(metadata)))
    PM.TITO_PROFILE.set(PM._tito_profile(metadata))


def one_turn_payload(extra_user: bool = False) -> dict:
    messages = [{"role": "user", "content": "chào"}]
    if extra_user:
        messages = [
            {"role": "user", "content": "chào"},
            {"role": "assistant", "content": "chào bạn"},
            {"role": "user", "content": "hôm nay sao"},
        ]
    return {"model": "claude-opus-5-5", "messages": messages}


def mimi_chat(chat_id: str, user_id: str, model: str = "mimi") -> dict:
    return {
        "id": chat_id,
        "user_id": user_id,
        "chat": {"history": {"messages": {
            "m1": {"timestamp": 1, "role": "user", "content": "x"},
            "m2": {"timestamp": 2, "role": "assistant", "model": model, "content": "y"},
        }}},
    }


def tito_chat(chat_id: str, user_id: str) -> dict:
    return mimi_chat(chat_id, user_id, model="tito")


def set_chats(rows, raises: bool = False) -> None:
    class _Chats:
        @staticmethod
        def get_chats():
            if raises:
                raise RuntimeError("db down")
            return list(rows)

    TITO.Chats = _Chats
    TITO.CHATS_AVAILABLE = True


def clear_chats() -> None:
    if hasattr(TITO, "Chats"):
        del TITO.Chats
    TITO.CHATS_AVAILABLE = False


def run(pipe, payload, metadata, user) -> str:
    return asyncio.run(pipe._tito_mimi_inject(payload, metadata, user))


def ids_by_bucket(prefix: str):
    """A chat id with hash % 10 == 0 (bear hit) and one with hash % 10 != 0 (miss)."""
    hit = miss = None
    i = 0
    while hit is None or miss is None:
        cid = f"{prefix}-{i}"
        if int(hashlib.sha256(cid.encode("utf-8")).hexdigest(), 16) % 10 == 0:
            hit = hit or cid
        else:
            miss = miss or cid
        i += 1
    return hit, miss


# ----------------------------------------------------------------- goose
def test_goose_on_first_ever_turn():
    set_chats([])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=0)
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(), MIMI_META, USER) == GOOSE
    finally:
        clear_chats()


def test_no_goose_when_earlier_mimi_chat_exists():
    set_chats([mimi_chat("old-1", "user-1")])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=0)
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(), MIMI_META, USER) == ""
    finally:
        clear_chats()


def test_goose_ignores_other_users_and_tito_chats():
    set_chats([mimi_chat("theirs", "user-2"), tito_chat("tito-1", "user-1")])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=0)
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(), MIMI_META, USER) == GOOSE
    finally:
        clear_chats()


def test_current_chat_is_skipped():
    # The current chat already holds a mimi message (retry of the first turn).
    set_chats([mimi_chat("c-now", "user-1")])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=0)
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(), MIMI_META, USER) == GOOSE
    finally:
        clear_chats()


def test_no_goose_on_chats_error():
    set_chats([], raises=True)
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=0)
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(), MIMI_META, USER) == ""
    finally:
        clear_chats()


def test_no_goose_without_chats_or_user_id():
    clear_chats()
    pipe = FakePipe(MIMI_BEAR_RATE=0)
    use_request(MIMI_META)
    assert run(pipe, one_turn_payload(), MIMI_META, USER) == ""
    set_chats([])
    try:
        assert run(pipe, one_turn_payload(), MIMI_META, {}) == ""
    finally:
        clear_chats()


# ----------------------------------------------------------------- bear
def test_bear_fires_on_hit_and_not_on_miss():
    hit, miss = ids_by_bucket("bear-check")
    set_chats([mimi_chat("old-1", "user-1")])  # goose not due
    try:
        pipe = FakePipe()
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(), {"model": {"id": "mimi"}, "chat_id": hit}, USER) == BEAR
        assert run(pipe, one_turn_payload(), {"model": {"id": "mimi"}, "chat_id": miss}, USER) == ""
    finally:
        clear_chats()


def test_bear_rate_zero_is_off():
    hit, _ = ids_by_bucket("rate-zero")
    set_chats([mimi_chat("old-1", "user-1")])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=0)
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(), {"model": {"id": "mimi"}, "chat_id": hit}, USER) == ""
    finally:
        clear_chats()


def test_bear_is_deterministic_for_retries():
    hit, _ = ids_by_bucket("retry")
    set_chats([mimi_chat("old-1", "user-1")])
    try:
        pipe = FakePipe()
        use_request(MIMI_META)
        meta = {"model": {"id": "mimi"}, "chat_id": hit}
        assert run(pipe, one_turn_payload(), meta, USER) == run(pipe, one_turn_payload(), meta, USER) == BEAR
    finally:
        clear_chats()


# ----------------------------------------------------------------- gating
def test_nothing_on_second_user_turn():
    set_chats([])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=1)
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(extra_user=True), MIMI_META, USER) == ""
    finally:
        clear_chats()


def test_goose_beats_bear():
    hit, _ = ids_by_bucket("beats")
    set_chats([])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=1)  # every chat is a bear hit
        use_request(MIMI_META)
        assert run(pipe, one_turn_payload(), {"model": {"id": "mimi"}, "chat_id": hit}, USER) == GOOSE
    finally:
        clear_chats()


def test_tito_and_unknown_model_never_inject():
    set_chats([])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=1)
        for meta in (TITO_META, OTHER_META):
            use_request(meta)
            assert PM._tito_active_profile().get("injects") is False
        assert PM._tito_active_profile().get("injects") is False  # no profile at all
        assert PM.TITO_PROFILES["mimi"].get("injects") is True
        assert PM.TITO_PROFILES["tito"].get("injects") is False

        # The call-site gate is "TITO_ACTIVE and tool_loop_iteration == 1 and not is_internal
        # and profile.injects": for tito and unknown models it is False, so the method is
        # never called. If it were called anyway, it must not change the payload.
        for meta in (TITO_META, OTHER_META):
            use_request(meta)
            payload = one_turn_payload()
            before = copy.deepcopy(payload)
            text = run(pipe, payload, meta, USER)
            assert payload == before
            assert text in (GOOSE, BEAR, "")
    finally:
        clear_chats()


def test_inject_text_joins_vn_reminder_slot():
    set_chats([])
    try:
        pipe = FakePipe(MIMI_BEAR_RATE=0, MIMI_VN_REMINDER="reminder")
        use_request(MIMI_META)
        payload = one_turn_payload()
        state = pipe._tito_vn_reminder_apply(payload, GOOSE)
        assert state is not None
        text = state["block"]["text"]
        assert text == "reminder\n\n" + GOOSE, text
    finally:
        clear_chats()


def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
