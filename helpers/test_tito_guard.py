"""Self-check regression test for Tito's output guard.

After a Tito reply finishes, the pipe may ask ``claude-haiku-5-5`` for a
minimal edit of the VISIBLE text only, and then replace the message once.
Hidden parts (``<details>`` blocks, ``[anthropic-hidden-...]`` carriers and
``[](anthropic:...)`` metadata markers) must survive byte-for-byte, an edit
that loses a number must be rejected, and any error must leave the streamed
reply untouched. Non-Tito models never reach the guard.

Runs without OpenWebUI, without the anthropic SDK and without network access:
the Anthropic client is a recording fake and the Tito gate is the real
``_is_tito`` check.

Usage:
    python helpers/test_tito_guard.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import json
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


GUARD = _load("_tito_guard", SRC / "response" / "tito_guard.py")
TITO = _load("_tito", SRC / "tito.py")
# In the compiled artifact these live in the same module scope.
for _name in (
    "split_visible",
    "visible_text",
    "tito_guard_layout",
    "tito_guard_rebuild",
    "tito_guard_trigger",
    "tito_guard_safety_ok",
    "guard",
):
    setattr(TITO, _name, getattr(GUARD, _name))

TITO_META = {"model": {"id": "tito"}}
OTHER_META = {"model": {"id": "claude-opus-5-5"}}


class FakeMessages:
    def __init__(self, client: "FakeClient") -> None:
        self.client = client

    async def create(self, **kwargs: Any):
        self.client.calls.append(kwargs)
        if self.client.error is not None:
            raise self.client.error
        user_text = kwargs["messages"][0]["content"]
        reply = self.client.reply(user_text) if callable(self.client.reply) else self.client.reply
        return SimpleNamespace(content=[SimpleNamespace(type="text", text=reply)])


class FakeClient:
    def __init__(self, reply: Any = '{"ok": true}', error: Any = None) -> None:
        self.reply = reply
        self.error = error
        self.calls: List[Dict[str, Any]] = []
        self.messages = FakeMessages(self)


class FakeCtx:
    """Just the surface the guard touches: accumulated text and full replace."""

    def __init__(self, content: str) -> None:
        self.content = content
        self.replaces: List[str] = []

    def text(self) -> str:
        return self.content

    async def emit_replace(self, content: str) -> None:
        self.replaces.append(content)
        self.content = content


class FakePipe(TITO.PipeTitoMethods):
    def __init__(self, guard_valve: bool = True) -> None:
        self.valves = SimpleNamespace(TITO_OUTPUT_GUARD=guard_valve)


async def test_clean_text_skips_haiku():
    """A reply that passes every local pre-check costs no Haiku call."""
    clean = "Hôm nay trời đẹp ghê.\n\nAnh ăn cơm chưa?"
    assert GUARD.tito_guard_trigger(clean) is None, GUARD.tito_guard_trigger(clean)

    client = FakeClient('{"ok": true}')
    ctx = FakeCtx(clean)
    await FakePipe()._tito_output_guard(ctx, client, TITO_META)
    assert client.calls == [], client.calls
    assert ctx.replaces == [], ctx.replaces


async def test_two_questions_replaced_once_hidden_bytes_kept():
    """Two questions -> Haiku called; one replace; hidden blocks untouched."""
    content = (
        "Chào anh.\n\n"
        "Anh khỏe không? Anh ăn gì chưa?\n\n"
        '<details type="reasoning">\n<summary>Đang nghĩ</summary>\n> giấu kín\n</details>\n'
        "Dạ Tito đây."
    )
    # split_visible must reproduce the input byte-for-byte when joined.
    segments = GUARD.split_visible(content)
    assert "".join(text for _, text in segments) == content
    hidden = [text for is_visible, text in segments if not is_visible]
    assert len(hidden) == 1, hidden

    def fix(user_text: str) -> str:
        return json.dumps(
            {"ok": False, "edited": user_text.replace("Chào anh.", "Chào anh ạ.")},
            ensure_ascii=False,
        )

    client = FakeClient(fix)
    ctx = FakeCtx(content)
    await FakePipe()._tito_output_guard(ctx, client, TITO_META)

    assert len(client.calls) == 1, client.calls
    call = client.calls[0]
    assert call["model"] == "claude-haiku-5-5", call
    assert call["temperature"] == 0, call
    assert "Return JSON only" in call["system"], call["system"][-80:]
    assert "[[TITO-KEEP-1]]" in call["messages"][0]["content"], call["messages"]

    assert len(ctx.replaces) == 1, ctx.replaces
    replaced = ctx.replaces[0]
    assert hidden[0] in replaced, "hidden block drifted"
    assert "Chào anh ạ." in replaced, replaced
    assert ctx.content == replaced


async def test_dropped_number_rejects_edit():
    """An edit that loses a number must be rejected; the reply stays as streamed."""
    content = "Tito mua 2 kg gạo giúp anh nhé. Anh ăn gì chưa? Anh ngủ chưa?"
    assert GUARD.tito_guard_trigger(content) == "multiple_questions"

    def drop_number(user_text: str) -> str:
        return json.dumps(
            {"ok": False, "edited": user_text.replace("2 kg", "ít")},
            ensure_ascii=False,
        )

    client = FakeClient(drop_number)
    ctx = FakeCtx(content)
    await FakePipe()._tito_output_guard(ctx, client, TITO_META)

    assert len(client.calls) == 1, client.calls
    assert ctx.replaces == [], ctx.replaces
    assert ctx.content == content


async def test_dropped_hidden_marker_rejects_edit():
    """An edit that drops the keep-marker cannot be rebuilt; keep the original."""
    content = (
        "Chào anh.\n\n"
        "Anh khỏe không? Anh ăn gì chưa?\n\n"
        '<details type="tool_calls" id="x">\nbody\n</details>\n'
        "Dạ Tito đây."
    )
    client = FakeClient(
        lambda user_text: json.dumps(
            {"ok": False, "edited": user_text.replace("[[TITO-KEEP-1]]", "")},
            ensure_ascii=False,
        )
    )
    ctx = FakeCtx(content)
    await FakePipe()._tito_output_guard(ctx, client, TITO_META)

    assert len(client.calls) == 1, client.calls
    assert ctx.replaces == [], ctx.replaces
    assert ctx.content == content


async def test_rebuild_rejects_out_of_order_markers():
    """Markers present but swapped cannot be put back in place; reject."""
    segments = [
        (True, "a"),
        (False, "H"),
        (True, "b"),
        (False, "H2"),
        (True, "c"),
    ]
    edited = "a\n\n[[TITO-KEEP-2]]\n\nb\n\n[[TITO-KEEP-1]]\n\nc"
    assert GUARD.tito_guard_rebuild(segments, edited) is None


async def test_haiku_error_keeps_reply():
    """A failing Haiku call must not block or alter the reply."""
    content = "Tito chào anh. Anh ăn gì chưa? Anh ngủ chưa?"
    client = FakeClient(error=RuntimeError("boom"))
    ctx = FakeCtx(content)
    await FakePipe()._tito_output_guard(ctx, client, TITO_META)

    assert len(client.calls) == 1, client.calls
    assert ctx.replaces == [], ctx.replaces
    assert ctx.content == content


async def test_non_tito_model_never_runs_guard():
    """The gate is the OWUI model id in metadata, not the reply content."""
    content = "Anh ăn gì chưa? Anh ngủ chưa?"
    client = FakeClient('{"ok": false, "edited": "x?"}')
    ctx = FakeCtx(content)
    await FakePipe()._tito_output_guard(ctx, client, OTHER_META)
    assert client.calls == [], client.calls
    assert ctx.replaces == [], ctx.replaces

    # Tito with the valve off is the same: no call, no replace.
    client = FakeClient('{"ok": false, "edited": "x?"}')
    ctx = FakeCtx(content)
    await FakePipe(guard_valve=False)._tito_output_guard(ctx, client, TITO_META)
    assert client.calls == [], client.calls
    assert ctx.replaces == [], ctx.replaces


async def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        await test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    asyncio.run(main())
