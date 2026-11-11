"""Regression test for the TITO-ERR codes on Tito's remaining failure sites.

Every real Tito-side failure logs exactly one grep-able ERROR line:
``TITO-ERR T{HHMM}-{4 hex} <component> <safe detail>``. Components covered
here: output_guard, tools_file, discord_alert, month_spend, image_shrink.
The detail never carries URLs, keys or request bodies.

Runs without OpenWebUI, without the anthropic SDK and without network access.

Usage:
    python helpers/test_tito_err_codes.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
import os
import re
import sys
import tempfile
import time
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TITO = _load("_tito_err_codes", SRC / "tito.py")
METHODS = TITO.PipeTitoMethods

CODE_RE = r"TITO-ERR T\d{4}-[0-9a-f]{4} "


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__(logging.DEBUG)
        self.records: list[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


def _capture():
    handler = _Capture()
    TITO.logger.addHandler(handler)
    TITO.logger.setLevel(logging.DEBUG)
    return handler


def _errors(handler: _Capture) -> list[str]:
    return [
        r.getMessage()
        for r in handler.records
        if r.levelno >= logging.ERROR and r.getMessage().startswith("TITO-ERR ")
    ]


def _single_tito_err(handler: _Capture, component: str) -> str:
    errs = _errors(handler)
    assert len(errs) == 1, errs
    assert re.fullmatch(CODE_RE + component + r" .*", errs[0]), errs[0]
    return errs[0]


def test_helper_code_format_and_redaction():
    handler = _capture()
    try:
        code = METHODS._tito_err(
            "probe",
            RuntimeError(
                "fail https://discord.com/api/webhooks/123/SECRETTOKEN "
                "x-api-key: sk-ant-api03-SECRETSECRET123"
            ),
        )
        assert re.fullmatch(r"T\d{4}-[0-9a-f]{4}", code), code
        msg = _single_tito_err(handler, "probe")
        assert "discord.com" not in msg and "SECRETTOKEN" not in msg, msg
        assert "[url]" in msg, msg
        assert "SECRETSECRET" not in msg, msg
        assert msg.startswith(f"TITO-ERR {code} probe RuntimeError: "), msg
    finally:
        TITO.logger.removeHandler(handler)


def test_url_in_exception_is_redacted_everywhere():
    text = METHODS._tito_safe_err(
        ValueError("see https://example.invalid/a?key=abc and http://10.0.0.1:8080/x")
    )
    assert "example.invalid" not in text and "10.0.0.1" not in text, text
    assert text.count("[url]") == 2, text


def test_tools_file_site():
    handler = _capture()
    saved = os.environ.get("TITO_TOOLS_FILE")
    missing = os.path.join(tempfile.gettempdir(), "tito_err_codes_missing_tools.json")
    os.environ["TITO_TOOLS_FILE"] = missing
    METHODS._tito_tools_alerted = True  # keep the Discord alert out of this test
    try:
        tools = METHODS._tito_frozen_tools()
        assert tools == [], tools
        _single_tito_err(handler, "tools_file")
    finally:
        TITO.logger.removeHandler(handler)
        if saved is None:
            os.environ.pop("TITO_TOOLS_FILE", None)
        else:
            os.environ["TITO_TOOLS_FILE"] = saved
        METHODS._tito_tools_alerted = False


def test_discord_alert_site():
    handler = _capture()
    saved = os.environ.get("TITO_DISCORD_WEBHOOK")
    # Port 9 on loopback refuses immediately; no external network is used.
    os.environ["TITO_DISCORD_WEBHOOK"] = "http://127.0.0.1:9/hook"
    try:
        METHODS._tito_discord_alert("probe alert")
        deadline = time.monotonic() + 5
        while not _errors(handler) and time.monotonic() < deadline:
            time.sleep(0.05)
        msg = _single_tito_err(handler, "discord_alert")
        assert "127.0.0.1" not in msg and "/hook" not in msg, msg
    finally:
        TITO.logger.removeHandler(handler)
        if saved is None:
            os.environ.pop("TITO_DISCORD_WEBHOOK", None)
        else:
            os.environ["TITO_DISCORD_WEBHOOK"] = saved


async def test_month_spend_site():
    handler = _capture()
    alerts: list[str] = []

    class _BoomChats:
        @staticmethod
        def get_chats():
            raise RuntimeError("db down at https://db.invalid/secret")

    old_chats = getattr(TITO, "Chats", None)
    old_flag = getattr(TITO, "CHATS_AVAILABLE", False)
    TITO.Chats = _BoomChats
    TITO.CHATS_AVAILABLE = True
    tito = METHODS()
    tito._tito_discord_alert = lambda message: alerts.append(message)
    try:
        total = await tito.month_spend_usd()
        assert total == 0.0, total
        msg = _single_tito_err(handler, "month_spend")
        assert "db.invalid" not in msg and "[url]" in msg, msg
        assert alerts and "db.invalid" not in alerts[0], alerts
    finally:
        TITO.logger.removeHandler(handler)
        TITO.Chats = old_chats
        TITO.CHATS_AVAILABLE = old_flag


def test_image_shrink_site():
    handler = _capture()
    tito = METHODS()
    tito.valves = SimpleNamespace(TITO_IMAGE_MAX_PX=100)
    bad = {
        "type": "image",
        "source": {"type": "base64", "media_type": "image/png", "data": "bm90IGFuIGltYWdl"},
    }
    payload = {"messages": [{"role": "user", "content": [bad]}]}
    try:
        out = tito._tito_shrink_images(payload)
        assert out["messages"][0]["content"][0] == bad, "corrupt image changed"
        _single_tito_err(handler, "image_shrink")
    finally:
        TITO.logger.removeHandler(handler)


async def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        if asyncio.iscoroutinefunction(test):
            await test()
        else:
            test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    asyncio.run(main())
