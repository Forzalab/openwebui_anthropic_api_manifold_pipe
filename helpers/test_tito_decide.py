"""Self-check for Tito's decide shadow (valve TITO_DECIDE, default off).

Per Tito turn, three MSD-1 flags are scored over HTTP and LOGGED ONLY. Checks:
the valve off makes no call; a successful run parses probabilities.yes and logs
one ``[TITO] decide`` line in the agreed format; one failing flag falls back to
the Vietnamese keyword regex for that flag only; a timeout fails open with no
exception; the API key and the message text never reach a log record.

HTTP is mocked by replacing ``urllib.request.urlopen``. Runs without OpenWebUI,
without the anthropic SDK and without network access.

Usage:
    python helpers/test_tito_decide.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
import logging
import re
import sys
import time
import typing
import urllib.request
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"
KEY = "sk-or-TEST-KEY-DO-NOT-LOG-123456"
MESSAGE = "SECRET-MESSAGE-TEXT giá xe này bao nhiêu"
LOG_LINE = re.compile(
    r"^\[TITO\] decide sc=\d\.\d\d\((msd|regex)\) search=\d\.\d\d\((msd|regex)\) "
    r"story=\d\.\d\d\((msd|regex)\) ms=\d+$"
)


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TITO = _load("_tito_decide_under_test", SRC / "tito.py")
# decrypt_valve_secret lives in the template; the compiled pipe has it in the same
# namespace. Here the test key is plaintext, so identity is enough.
TITO.decrypt_valve_secret = lambda value: str(value)
TITO.logger = logging.getLogger("tito_decide_test")


class FakePipe(TITO.PipeTitoMethods):
    def __init__(self, **valves):
        self.valves = SimpleNamespace(
            TITO_DECIDE=valves.get("TITO_DECIDE", True),
            TITO_DECIDE_KEY=valves.get("TITO_DECIDE_KEY", KEY),
            TITO_DECIDE_TIMEOUT_MS=valves.get("TITO_DECIDE_TIMEOUT_MS", 1500),
        )


class Capture(logging.Handler):
    def __init__(self):
        super().__init__(level=logging.DEBUG)
        self.messages: list[str] = []

    def emit(self, record):
        self.messages.append(record.getMessage())


class FakeResponse:
    def __init__(self, body: bytes):
        self._body = body

    def read(self):
        return self._body

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def install_urlopen(scores: dict, fail: set = frozenset(), sleep_s: float = 0.0):
    """Replace urlopen. ``scores`` maps flag -> probability yes; ``fail`` flags raise."""
    calls: list[dict] = []

    def fake(req, timeout=None):
        body = json.loads(req.data)
        flag = next(iter(body["questions"]))
        calls.append({"flag": flag, "state": body["state"], "auth": req.get_header("Authorization"),
                      "model": body["model"], "url": req.full_url})
        if sleep_s:
            time.sleep(sleep_s)
        if flag in fail:
            # The message carries the key on purpose: the pipe must log only the type name.
            raise OSError(f"upstream refused Bearer {KEY}")
        payload = {"answers": {flag: {"probabilities": {"yes": scores[flag]}}}}
        return FakeResponse(json.dumps(payload).encode())

    urllib.request.urlopen = fake
    return calls


ELAPSED = {"collect": 0.0}


def run_turn(pipe, text: str, capture: Capture):
    """Start the decide task, then collect it, as the orchestrator does. Returns the state.

    ELAPSED["collect"] is the time spent in the collect step only; asyncio.run would
    otherwise wait for the default executor thread that is still sleeping in a timeout case.
    """
    async def go():
        state = pipe._tito_decide_start({"messages": [{"role": "user", "content": text}]})
        if state is not None:
            t0 = time.monotonic()
            await pipe._tito_decide_collect(state)
            ELAPSED["collect"] = time.monotonic() - t0
        return state

    capture.messages.clear()
    return asyncio.run(go())


def attach(capture: Capture):
    TITO.logger.handlers = [capture]
    TITO.logger.setLevel(logging.DEBUG)
    TITO.logger.propagate = False


def main() -> int:
    capture = Capture()
    attach(capture)
    real_urlopen = urllib.request.urlopen
    try:
        # 1. Valve off: no call, no log, start returns None.
        calls = install_urlopen({"smart_consumer": 0.9, "needs_search": 0.9, "ke_anh_nghe": 0.9})
        pipe = FakePipe(TITO_DECIDE=False)
        state = run_turn(pipe, MESSAGE, capture)
        assert state is None, "valve off must not start a task"
        assert calls == [], "valve off made an HTTP call"
        assert not any("[TITO] decide" in m for m in capture.messages), "valve off logged a decide line"

        # 1b. Valve on but key empty: also no call.
        pipe = FakePipe(TITO_DECIDE=True, TITO_DECIDE_KEY="")
        state = run_turn(pipe, MESSAGE, capture)
        assert state is None and calls == [], "empty key must not call"

        # 2. Success: parsed probabilities, one line in the agreed format, right request shape.
        calls = install_urlopen({"smart_consumer": 0.97, "needs_search": 0.41, "ke_anh_nghe": 0.12})
        pipe = FakePipe()
        run_turn(pipe, MESSAGE, capture)
        assert len(calls) == 3, f"expected 3 calls, got {len(calls)}"
        assert {c["flag"] for c in calls} == set(TITO.PipeTitoMethods.TITO_DECIDE_FLAGS)
        assert all(c["auth"] == "Bearer " + KEY for c in calls), "auth header wrong"
        assert all(c["url"] == "https://openrouter.ai/api/alpha/decisions" for c in calls)
        assert all(c["model"] == "microsoft/microsoft-decision-1" for c in calls)
        assert all(c["state"] == MESSAGE for c in calls), "state is not the last user message"
        decide = [m for m in capture.messages if m.startswith("[TITO] decide")]
        assert len(decide) == 1, f"expected one decide line, got {decide}"
        assert LOG_LINE.match(decide[0]), f"bad log format: {decide[0]}"
        assert "sc=0.97(msd)" in decide[0] and "search=0.41(msd)" in decide[0] and "story=0.12(msd)" in decide[0], decide[0]

        # 3. One flag fails: regex fallback for that flag only, marked regex.
        calls = install_urlopen({"smart_consumer": 0.97, "needs_search": 0.41, "ke_anh_nghe": 0.12},
                                fail={"needs_search"})
        pipe = FakePipe()
        result_state = run_turn(pipe, MESSAGE, capture)
        decide = [m for m in capture.messages if m.startswith("[TITO] decide")]
        assert len(decide) == 1 and LOG_LINE.match(decide[0]), decide
        assert "sc=0.97(msd)" in decide[0], decide[0]
        assert "search=1.00(regex)" in decide[0], decide[0]  # "bao nhiêu" hits the regex
        assert "story=0.12(msd)" in decide[0], decide[0]
        assert result_state is not None

        # 3b. Regex directly: keyword hits and misses.
        assert TITO.PipeTitoMethods._tito_decide_regex("Cái này giá bao nhiêu?", "smart_consumer") == 1.0
        assert TITO.PipeTitoMethods._tito_decide_regex("Chào buổi sáng", "smart_consumer") == 0.0
        assert TITO.PipeTitoMethods._tito_decide_regex("Thiệt không vậy?", "ke_anh_nghe") == 1.0
        assert TITO.PipeTitoMethods._tito_decide_regex("đôi giày mới", "smart_consumer") == 0.0  # "đô" inside "đôi"

        # 4. Timeout: the calls outlive TITO_DECIDE_TIMEOUT_MS. Fails open, no exception,
        #    a regex line is still logged, and the call returns in about the timeout.
        install_urlopen({"smart_consumer": 0.97, "needs_search": 0.41, "ke_anh_nghe": 0.12}, sleep_s=1.0)
        pipe = FakePipe(TITO_DECIDE_TIMEOUT_MS=150)
        try:
            run_turn(pipe, MESSAGE, capture)
        except Exception as exc:  # pragma: no cover - the assertion below is the real check
            raise AssertionError(f"timeout raised: {type(exc).__name__}") from exc
        elapsed = ELAPSED["collect"]
        assert elapsed < 0.6, f"timeout did not bound the wait: {elapsed:.2f}s"
        decide = [m for m in capture.messages if m.startswith("[TITO] decide")]
        assert len(decide) == 1, decide
        assert "(msd)" not in decide[0], f"timed-out flags must not be msd: {decide[0]}"

        # 5. Key and message text never reach a log record (including the failure path).
        install_urlopen({"smart_consumer": 0.97, "needs_search": 0.41, "ke_anh_nghe": 0.12},
                        fail={"smart_consumer", "needs_search", "ke_anh_nghe"})
        pipe = FakePipe()
        run_turn(pipe, MESSAGE, capture)
        joined = "\n".join(capture.messages)
        assert KEY not in joined, "API key leaked into logs"
        assert "SECRET-MESSAGE-TEXT" not in joined, "message text leaked into logs"
        assert "upstream refused" not in joined, "exception text leaked into logs"
        decide = [m for m in capture.messages if m.startswith("[TITO] decide")]
        assert len(decide) == 1 and "sc=-" not in decide[0] and "(regex)" in decide[0], decide
    finally:
        urllib.request.urlopen = real_urlopen

    print("ok: tito decide shadow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
