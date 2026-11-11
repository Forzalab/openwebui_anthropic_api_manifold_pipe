"""Self-check for Tito's image shrink valve (TITO_IMAGE_MAX_PX, default 0 = off).

Checks: an oversized opaque PNG becomes a JPEG whose longest side is the cap,
an image with transparency stays PNG, an image within the cap is untouched, a
corrupt image is left as it was, nested tool_result images are handled, the
input payload is not mutated, and a missing Pillow leaves every image alone.
Runs without OpenWebUI and without network access.

Usage:
    python helpers/test_tito_image_shrink.py
"""
from __future__ import annotations

import base64
import copy
import importlib.util
import io
import logging
import sys
from pathlib import Path
from types import SimpleNamespace

from PIL import Image

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TITO = _load("_tito", SRC / "tito.py")


def _b64(img: Image.Image, fmt: str) -> str:
    buf = io.BytesIO()
    img.save(buf, format=fmt)
    return base64.b64encode(buf.getvalue()).decode("ascii")


def _block(data: str, media: str) -> dict:
    return {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}}


def _open(data: str) -> Image.Image:
    return Image.open(io.BytesIO(base64.b64decode(data)))


def main() -> int:
    opaque = _b64(Image.new("RGB", (400, 200), (200, 30, 30)), "PNG")
    alpha = _b64(Image.new("RGBA", (300, 300), (0, 0, 255, 128)), "PNG")
    small = _b64(Image.new("RGB", (40, 20), (1, 2, 3)), "PNG")
    corrupt = base64.b64encode(b"not an image").decode("ascii")
    payload = {"model": "claude", "messages": [
        {"role": "user", "content": [
            {"type": "text", "text": "hi", "cache_control": {"type": "ephemeral"}},
            _block(opaque, "image/png"),
            _block(alpha, "image/png"),
            _block(small, "image/png"),
            _block(corrupt, "image/png"),
            {"type": "image", "source": {"type": "url", "url": "https://example.invalid/x.png"}},
        ]},
        {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": "t1", "content": [_block(opaque, "image/png")]},
        ]},
    ]}
    before = copy.deepcopy(payload)
    tito = TITO.PipeTitoMethods()
    tito.valves = SimpleNamespace(TITO_IMAGE_MAX_PX=100)

    records: list[str] = []

    class _Capture(logging.Handler):
        def emit(self, record):
            records.append(record.getMessage())

    TITO.logger.addHandler(_Capture())
    TITO.logger.setLevel(logging.WARNING)
    out = tito._tito_shrink_images(payload)

    assert payload == before, "input payload was mutated"
    blocks = out["messages"][0]["content"]
    assert blocks[0]["cache_control"] == {"type": "ephemeral"}
    shrunk = blocks[1]["source"]
    assert shrunk["media_type"] == "image/jpeg", shrunk["media_type"]
    im = _open(shrunk["data"])
    assert im.size == (100, 50) and im.format == "JPEG", im.size
    kept = blocks[2]["source"]
    assert kept["media_type"] == "image/png" and _open(kept["data"]).format == "PNG"
    assert _open(kept["data"]).size == (100, 100)
    assert blocks[3] == payload["messages"][0]["content"][3], "small image changed"
    assert blocks[4] == payload["messages"][0]["content"][4], "corrupt image changed"
    assert blocks[5] == payload["messages"][0]["content"][5], "url image changed"
    nested = out["messages"][1]["content"][0]["content"][0]["source"]
    assert _open(nested["data"]).size == (100, 50)
    assert len(records) == 1 and "image_shrink " in records[0] and "TITO-ERR T" in records[0], records
    assert "not an image" not in records[0] and "example" not in records[0]

    # Pillow missing: every image is left alone, one WARNING.
    records.clear()
    saved = sys.modules.get("PIL")
    sys.modules["PIL"] = None  # makes `from PIL import Image` raise ImportError
    try:
        same = tito._tito_shrink_images(payload)
    finally:
        if saved is not None:
            sys.modules["PIL"] = saved
        else:
            sys.modules.pop("PIL", None)
    assert same is payload and len(records) == 1 and "image_shrink " in records[0] and "TITO-ERR T" in records[0], records
    print("ok: tito image shrink")
    return 0


if __name__ == "__main__":
    sys.exit(main())
