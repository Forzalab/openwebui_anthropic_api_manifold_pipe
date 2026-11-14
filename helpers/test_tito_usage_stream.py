"""Self-check for Tito's usage in the streamed reply.

OpenWebUI's middleware persists ``message.usage`` from the ``usage`` key of the
SSE chunks it reads from the pipe. A Tito turn must therefore end with a chunk
carrying the final usage, after the text chunk, or the middleware overwrites the
pipe's direct DB write with ``{"incomplete": true}``. The test checks the chunk
shape and that a middleware-style reader recovers the usage. Non-Tito turns
never reach this path (gate in ``pipe_orchestrator``).

Runs without OpenWebUI and without network access.

Usage:
    python helpers/test_tito_usage_stream.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TITO = _load("_tito", SRC / "tito.py")
USAGE = {"input_tokens": 12, "output_tokens": 7, "total_tokens": 19, "prompt_tokens": 12, "completion_tokens": 7}


async def _collect(gen) -> list:
    return [chunk async for chunk in gen]


def main() -> int:
    stream = TITO.PipeTitoMethods._tito_usage_stream
    chunks = asyncio.run(_collect(stream("hello", "tito", dict(USAGE))))
    assert len(chunks) == 2, chunks
    assert chunks[0]["choices"][0]["delta"] == {"content": "hello"}, chunks[0]
    assert "usage" not in chunks[0], chunks[0]
    assert chunks[1]["usage"] == USAGE, chunks[1]
    assert chunks[1]["choices"] == [], chunks[1]

    # Middleware-style read: the last usage-bearing chunk wins.
    seen = {}
    for chunk in chunks:
        if chunk.get("usage"):
            seen = chunk["usage"]
    assert seen == USAGE

    empty = asyncio.run(_collect(stream("", "tito", dict(USAGE))))
    assert len(empty) == 1 and empty[0]["usage"] == USAGE, empty
    print("ok: tito usage stream chunks")
    return 0


if __name__ == "__main__":
    sys.exit(main())
