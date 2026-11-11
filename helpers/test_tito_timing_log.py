"""Self-check for the per-turn Tito timing line.

One INFO line per Tito turn, numbers only: ``[TITO] timing ttfb_ms=... ``.
The test feeds a fake request context with known marks and checks the exact
field order, the arithmetic, the image count and decoded byte size (computed
from base64 length, nested tool_result images included), and that no content
reaches the log. Runs without OpenWebUI and without network access.

Usage:
    python helpers/test_tito_timing_log.py
"""
from __future__ import annotations

import base64
import importlib.util
import logging
import re
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


TITO = _load("_tito", SRC / "tito.py")
LINE = re.compile(
    r"^\[TITO\] timing ttfb_ms=(\d+) first_text_ms=(\d+) stream_ms=(\d+) guard_ms=(\d+) "
    r"total_ms=(\d+) in=(\d+) out=(\d+) cache_read=(\d+) cache_write=(\d+) "
    r"images=(\d+) image_bytes=(\d+)$"
)


def main() -> int:
    png = base64.b64encode(b"\x89PNG" + b"x" * 96).decode()  # 100 raw bytes
    secret_text = "SECRET PROMPT TEXT"
    payload = {
        "messages": [
            {"role": "user", "content": [
                {"type": "text", "text": secret_text},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
                {"type": "image", "source": {"type": "url", "url": "https://example.invalid/a.png"}},
            ]},
            {"role": "user", "content": [
                {"type": "tool_result", "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": png}},
                ]},
            ]},
        ]
    }
    ctx = SimpleNamespace(timing={"t0": 100.0, "ttfb": 100.25, "first_text": 101.0,
                                  "stream": 1.5, "guard": 0.75})
    tito = TITO.PipeTitoMethods()
    tito._tito_count_images(ctx, payload)
    assert ctx.timing["images"] == 3, ctx.timing
    assert ctx.timing["image_bytes"] == 200, ctx.timing  # two base64 images x 100 bytes; URL image adds 0

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    log = TITO.logger
    handler = _Capture()
    log.addHandler(handler)
    log.setLevel(logging.INFO)
    try:
        ctx.timing["t0"] = 100.0
        import time as _t
        real = _t.monotonic
        TITO.time.monotonic = lambda: 102.0
        try:
            tito._tito_log_timing(ctx, {"input_tokens": 5, "output_tokens": 9,
                                        "cache_read_input_tokens": 40, "cache_creation_input_tokens": 3})
        finally:
            TITO.time.monotonic = real
    finally:
        log.removeHandler(handler)

    assert len(records) == 1, records
    m = LINE.match(records[0])
    assert m, records[0]
    assert m.groups() == ("250", "1000", "1500", "750", "2000", "5", "9", "40", "3", "3", "200"), m.groups()
    assert secret_text not in records[0] and "example.invalid" not in records[0]
    print("ok:", records[0])
    return 0


if __name__ == "__main__":
    sys.exit(main())
