"""Self-check regression test for Tito's quiet UI.

When ``TITO_QUIET_UI`` is on and the request is for the Tito model, the
frontend must receive only real answer content: no ``status`` lines, no
``source`` chips and no info/warning toasts. Error notifications still pass,
and every non-Tito model (``claude-*``) keeps today's behaviour exactly.

Runs without OpenWebUI and without network access: the status emitter and the
central ``Pipe.emit_event`` filter are exercised against a recording fake
emitter, and the Tito gate is evaluated with the real ``_is_tito`` check.

Usage:
    python helpers/test_tito_quiet_ui.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


STATUS = _load("_status_events", SRC / "response" / "status_events.py")
RUNTIME = _load("_runtime", SRC / "response" / "runtime.py")
TITO = _load("_tito", SRC / "tito.py")
# In the compiled artifact these live in the same module scope.
RUNTIME._should_drop_event = STATUS._should_drop_event


class RecordingEmitter:
    def __init__(self) -> None:
        self.events: List[Dict[str, Any]] = []

    async def __call__(self, event: Dict[str, Any]) -> None:
        self.events.append(event)


class FakeTitoPipe(TITO.PipeTitoMethods):
    """Only what the quiet-UI gate reads: the valve and the model metadata."""

    def __init__(self, quiet_valve: bool = True) -> None:
        self.valves = SimpleNamespace(TITO_QUIET_UI=quiet_valve)


def _set_quiet(pipe: FakeTitoPipe, metadata: Dict[str, Any]) -> bool:
    quiet = pipe._tito_quiet_ui(metadata)
    STATUS.QUIET_UI.set(quiet)
    return quiet


def _runtime() -> Any:
    return RUNTIME.PipeStreamRuntimeSupportMethods()


async def test_tito_quiet_drops_statuses_and_sources():
    """Tito + valve on: StatusEmitter emits no status, source is dropped."""
    pipe = FakeTitoPipe(True)
    assert _set_quiet(pipe, {"model": {"id": "tito"}}) is True
    try:
        emitter = RecordingEmitter()
        status = STATUS.StatusEmitter(emitter)
        await status.waiting()
        await status.thinking()
        await status.responding()
        await status.complete("✅ Response Complete")
        assert emitter.events == [], f"status events leaked: {emitter.events}"

        runtime = _runtime()
        await runtime.emit_event(
            {"type": "source", "data": {"source": {"name": "x", "url": "https://x"}}},
            emitter,
        )
        await runtime.emit_event(
            {"type": "status", "data": {"description": "❌ Response with Errors", "done": True}},
            emitter,
        )
        assert emitter.events == [], f"source/status events leaked: {emitter.events}"
    finally:
        STATUS.QUIET_UI.set(False)


async def test_tito_quiet_drops_non_error_notifications_only():
    """Info/warning toasts are dropped; an error toast still reaches the user."""
    pipe = FakeTitoPipe(True)
    assert _set_quiet(pipe, {"model": {"id": "tito"}}) is True
    try:
        emitter = RecordingEmitter()
        status = STATUS.StatusEmitter(emitter)
        await status.notification("☁️ Uploading file", type="info")
        await status.notification("Web fetch is off", type="warning")
        assert emitter.events == [], f"non-error notifications leaked: {emitter.events}"

        await status.notification("Tito đang bận một chút", type="error")
        assert len(emitter.events) == 1, emitter.events
        assert emitter.events[0]["type"] == "notification"
        assert emitter.events[0]["data"]["type"] == "error"

        runtime = _runtime()
        await runtime.emit_event(
            {"type": "notification", "data": {"type": "error", "content": "boom"}},
            emitter,
        )
        assert len(emitter.events) == 2, emitter.events
    finally:
        STATUS.QUIET_UI.set(False)


async def test_tito_quiet_keeps_real_content():
    """Message/replace/chat:completion are real content and always pass."""
    pipe = FakeTitoPipe(True)
    assert _set_quiet(pipe, {"model": {"id": "tito"}}) is True
    try:
        emitter = RecordingEmitter()
        runtime = _runtime()
        for event in (
            {"type": "message", "data": {"content": "Chào ông"}},
            {"type": "replace", "data": {"content": "Chào ông"}},
            {"type": "chat:completion", "data": {"done": True}},
        ):
            await runtime.emit_event(event, emitter)
        assert [e["type"] for e in emitter.events] == [
            "message",
            "replace",
            "chat:completion",
        ], emitter.events
    finally:
        STATUS.QUIET_UI.set(False)


async def test_non_tito_model_keeps_status_events():
    """claude-* requests are byte-for-byte unchanged: statuses still show."""
    pipe = FakeTitoPipe(True)
    assert _set_quiet(pipe, {"model": {"id": "claude-opus-5-5"}}) is False
    try:
        emitter = RecordingEmitter()
        status = STATUS.StatusEmitter(emitter)
        await status.waiting()
        assert emitter.events == [
            {"type": "status", "data": {"description": "Waiting for response...", "done": False, "hidden": False}}
        ], emitter.events

        # The central runtime filter must be a no-op for claude-* too.
        runtime = _runtime()
        await runtime.emit_event(
            {"type": "source", "data": {"source": {"name": "x", "url": "https://x"}}},
            emitter,
        )
        await runtime.emit_event(
            {"type": "notification", "data": {"type": "info", "content": "hi"}},
            emitter,
        )
        assert [e["type"] for e in emitter.events] == [
            "status",
            "source",
            "notification",
        ], emitter.events
    finally:
        STATUS.QUIET_UI.set(False)


async def test_valve_off_keeps_tito_status_events():
    """Tito + valve off: the quiet UI is disabled and statuses flow again."""
    pipe = FakeTitoPipe(False)
    assert _set_quiet(pipe, {"model": {"id": "tito"}}) is False
    try:
        emitter = RecordingEmitter()
        status = STATUS.StatusEmitter(emitter)
        await status.waiting()
        assert len(emitter.events) == 1, emitter.events
        assert emitter.events[0]["type"] == "status"
    finally:
        STATUS.QUIET_UI.set(False)


async def test_header_model_does_not_trigger_quiet_ui():
    """The gate is the OWUI model id in metadata, never body['model']."""
    pipe = FakeTitoPipe(True)
    # Base-model headers are what OWUI rewrites the body to; they must not count.
    assert pipe._tito_quiet_ui({"model": {"id": "tito"}, "body_model": "claude"}) is True
    assert pipe._tito_quiet_ui({}) is False
    STATUS.QUIET_UI.set(False)


async def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        await test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    asyncio.run(main())
