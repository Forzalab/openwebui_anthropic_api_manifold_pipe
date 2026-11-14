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
import logging
import re
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
    "guard_verdict",
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
    assert "temperature" not in call, call
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


class PipeLikeCtx:
    """Mirrors PipeRequestContext accumulation: one list, emit_replace resets it."""

    def __init__(self, content: str) -> None:
        self.final_message: List[str] = [content]
        self.events: List[Dict[str, Any]] = []

    def text(self) -> str:
        return "".join(self.final_message)

    async def emit_event(self, event: Dict[str, Any]) -> None:
        self.events.append(event)

    async def emit_replace(self, content: str) -> None:
        await self.emit_event({"type": "replace", "data": {"content": content}})
        self.final_message.clear()
        self.final_message.append(content)


async def test_final_replace_carries_guarded_text():
    """Guard replaces, then the orchestrator's final replace (pipe_orchestrator.py:
    consolidated = final_text(); emit replace) must emit the guarded text last."""
    content = "Chào anh.\n\nAnh khỏe không? Anh ăn gì chưa?"

    def fix(user_text: str) -> str:
        return json.dumps(
            {"ok": False, "edited": user_text.replace("Chào anh.", "Chào anh ạ.")},
            ensure_ascii=False,
        )

    ctx = PipeLikeCtx(content)
    await FakePipe()._tito_output_guard(ctx, FakeClient(fix), TITO_META)

    # Orchestrator tail after the guard.
    consolidated = ctx.text()
    if consolidated:
        await ctx.emit_event({"type": "replace", "data": {"content": consolidated}})

    replaces = [e["data"]["content"] for e in ctx.events if e.get("type") == "replace"]
    assert len(replaces) == 2, replaces
    assert "Chào anh ạ." in replaces[0], replaces[0]
    assert replaces[-1] == replaces[0], (replaces[0], replaces[-1])


async def test_guard_disables_thinking_and_sends_no_temperature():
    """The guard call sends thinking disabled (Haiku 5.5 thinks by default) and no temperature."""
    client = FakeClient('{"ok": true}')
    await GUARD.guard(client, "Tito chào anh. Anh ăn gì chưa? Anh ngủ chưa?")
    assert len(client.calls) == 1, client.calls
    call = client.calls[0]
    assert call.get("thinking") == {"type": "disabled"}, call
    assert "temperature" not in call, call


async def test_guard_never_sends_temperature():
    """No request kwarg named temperature, on success, on an edit, or on an SDK TypeError."""
    text = "Tito chào anh. Anh ăn gì chưa? Anh ngủ chưa?"
    client = FakeClient('{"ok": true}')
    assert await GUARD.guard(client, text) is None
    client = FakeClient(
        lambda t: json.dumps({"ok": False, "edited": t.replace("Anh ngủ chưa?", "Anh ngủ chưa ạ?")})
    )
    assert await GUARD.guard(client, text) is not None
    for call in client.calls:
        assert "temperature" not in call, call

    # A TypeError from the SDK propagates: no retry branch exists any more.
    client = FakeClient(error=TypeError("unexpected keyword argument 'temperature'"))
    try:
        await GUARD.guard(client, text)
    except TypeError:
        pass
    else:
        raise AssertionError("TypeError was swallowed")
    assert len(client.calls) == 1, client.calls
    assert "temperature" not in client.calls[0], client.calls[0]


class _Capture(logging.Handler):
    def __init__(self) -> None:
        super().__init__()
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


async def _run_guard_logged(ctx, client, meta=TITO_META):
    logger = logging.getLogger("_tito")
    handler = _Capture()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        await FakePipe()._tito_output_guard(ctx, client, meta)
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)
    return [r for r in handler.records if r.name == "_tito"]


async def test_outcome_line_once_per_run_without_reply_text():
    content = "Chào anh.\n\nAnh khỏe không? Anh ăn gì chưa?"

    def fix(user_text: str) -> str:
        return json.dumps(
            {"ok": False, "edited": user_text.replace("Chào anh.", "Chào anh ạ.")},
            ensure_ascii=False,
        )

    records = await _run_guard_logged(FakeCtx(content), FakeClient(fix))
    infos = [r for r in records if r.levelno == logging.INFO]
    assert len(infos) == 1, [r.getMessage() for r in records]
    msg = infos[0].getMessage()
    assert msg.startswith("[TITO] output guard outcome=replaced "), msg
    assert "trigger=multiple_questions" in msg, msg
    assert "Chào" not in msg and "Anh" not in msg, msg
    assert not [r for r in records if r.levelno >= logging.ERROR], records


async def test_outcome_labels():
    content = "Tito mua 2 kg gạo giúp anh nhé. Anh ăn gì chưa? Anh ngủ chưa?"
    cases = [
        (FakeClient('{"ok": true}'), "kept_ok"),
        (FakeClient("không phải JSON"), "kept_parse"),
        (
            FakeClient(lambda t: json.dumps({"ok": False, "edited": t.replace("2 kg", "ít")})),
            "kept_safety",
        ),
    ]
    for client, expected in cases:
        records = await _run_guard_logged(FakeCtx(content), client)
        infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
        assert len(infos) == 1, infos
        assert infos[0].startswith(f"[TITO] output guard outcome={expected} "), infos


async def test_error_logs_tito_err_line():
    content = "Tito chào anh. Anh ăn gì chưa? Anh ngủ chưa?"
    client = FakeClient(error=RuntimeError("boom sk-abcdefghijklmnop"))
    records = await _run_guard_logged(FakeCtx(content), client)

    infos = [r.getMessage() for r in records if r.levelno == logging.INFO]
    assert len(infos) == 1 and infos[0].startswith("[TITO] output guard outcome=error "), infos
    errs = [r.getMessage() for r in records if r.levelno == logging.ERROR]
    assert len(errs) == 1, errs
    assert re.fullmatch(
        r"TITO-ERR T\d{4}-[0-9a-f]{4} output_guard RuntimeError: boom \[redacted\]", errs[0]
    ), errs[0]


async def test_rebuild_keeps_blank_line_before_hidden_carrier():
    # 2026-11-11 leak: the edit path ate the blank line before a server-tool
    # carrier, so Conduit showed "[anthropic-hidden-...]: #<base64>" as text.
    split_visible, tito_guard_layout, tito_guard_rebuild = GUARD.split_visible, GUARD.tito_guard_layout, GUARD.tito_guard_rebuild

    content = "Chờ chút nha.\n\n[anthropic-hidden-srvtoolu_X]: #QUJD\n\nSố là 559.\n\n**Anh gọi không?**"
    segments = split_visible(content)
    layout, _ = tito_guard_layout(segments)
    assert tito_guard_rebuild(segments, layout) == content
    for squeezed in ("nha.\n\n",):  # model collapses blank lines around the sentinel
        out = tito_guard_rebuild(segments, layout.replace("nha.\n\n\n\n", squeezed))
        assert "nha.\n\n[anthropic-hidden-srvtoolu_X]: #QUJD" in out, repr(out)
        assert "\n\n\n[anthropic" not in out, repr(out)


async def test_english_paragraph_triggers_english_sentence():
    """A run of 5+ English words outside quotes, links and URLs is a rule break."""
    text = "The car has a clean title and no smog issues at all. Anh thấy sao?"
    assert GUARD.tito_guard_trigger(text) == "english_sentence", GUARD.tito_guard_trigger(text)

    # Longer than 12 words in quotes: the quote exemption does not apply.
    long_quote = (
        'Người bán nói: "I will be honest with you about this car and its history '
        'and its condition today." Anh thấy sao?'
    )
    assert GUARD.tito_guard_trigger(long_quote) == "english_sentence", GUARD.tito_guard_trigger(long_quote)


async def test_short_quoted_english_line_is_not_a_second_question():
    """A "?" inside a short quoted English line is anh's line, not Tito's question."""
    text = 'Anh cứ nói với họ: "Would you take $45?" Họ sẽ đồng ý thôi. Anh thấy sao?'
    assert GUARD.tito_guard_trigger(text) is None, GUARD.tito_guard_trigger(text)

    with_meaning = (
        'Anh cứ nói với họ: "Would you take $45?" (anh lấy 45 đô nha.) '
        "Họ sẽ đồng ý thôi. Anh thấy sao?"
    )
    assert GUARD.tito_guard_trigger(with_meaning) is None, GUARD.tito_guard_trigger(with_meaning)

    # A second question outside the quote still counts.
    two = 'Anh nói: "Would you take $45?" Họ hỏi nữa không? Anh thấy sao?'
    assert GUARD.tito_guard_trigger(two) == "multiple_questions", GUARD.tito_guard_trigger(two)


async def test_brand_and_model_names_do_not_trigger():
    """Brand, model and listing names inside Vietnamese prose are not English sentences."""
    text = "Anh xem giúp Tito cái Toyota Camry 2009, KBB ghi 300 triệu, bản M65 nha. Anh thấy sao?"
    assert GUARD.tito_guard_trigger(text) is None, GUARD.tito_guard_trigger(text)


async def test_url_does_not_trigger():
    """Words inside a URL, a markdown link or a code span are not checked."""
    text = "Anh xem link này nha: https://www.kbb.com/toyota/camry/2009/values giá tốt không?"
    assert GUARD.tito_guard_trigger(text) is None, GUARD.tito_guard_trigger(text)

    linked = "Anh xem [Toyota Camry Value Listing Page](https://example.com/x) nha, được không?"
    assert GUARD.tito_guard_trigger(linked) is None, GUARD.tito_guard_trigger(linked)


async def test_list_item_with_four_sentences_is_long_paragraph():
    """Each list item is its own unit: four sentences in one item is too long."""
    text = (
        "Anh nên làm mấy việc này:\n"
        "- Kiểm tra máy. Thay dầu nha. Rửa xe luôn. Hẹn thợ sáng mai.\n"
        "- Mua thêm nước. Anh thấy sao?"
    )
    assert GUARD.tito_guard_trigger(text) == "long_paragraph", GUARD.tito_guard_trigger(text)


async def test_short_list_items_are_fine():
    """A list of short items passes: the limit is per item, not per block.

    Edited for v0.12.3: the old items had three sentences each (old limit 3);
    the limit is now two, so each item is trimmed to two sentences.
    """
    text = (
        "Anh nên làm mấy việc này:\n"
        "- Kiểm tra máy. Thay dầu nha.\n"
        "- Mua nước. Hẹn thợ nha.\n"
        "- Lấy thuốc. Anh thấy sao?"
    )
    assert GUARD.tito_guard_trigger(text) is None, GUARD.tito_guard_trigger(text)


async def test_unaccented_vietnamese_runs_do_not_trigger():
    """Real false positives: unaccented Vietnamese words are not English."""
    for fragment in (
        "Tito ghi ra cho anh nha",
        "n qua Facebook hay Craigslist",
        "xe Toyota, Honda hay Mazda",
        "theo Kelley Blue Book, chi",
    ):
        text = f"Anh coi thử: {fragment}. Anh thấy sao?"
        assert GUARD.tito_guard_trigger(text) is None, (fragment, GUARD.tito_guard_trigger(text))


async def test_brand_run_without_function_words_does_not_trigger():
    """Five or more ASCII words with no English function word are a name list, not a sentence."""
    text = "Anh coi giá: Kelley Blue Book Honda Civic Mazda Toyota nha. Anh thấy sao?"
    assert GUARD.tito_guard_trigger(text) is None, GUARD.tito_guard_trigger(text)


async def test_real_english_sentence_triggers():
    """A real English sentence with function words still triggers english_sentence."""
    text = (
        "Transfers are free with a paid fare and allow unlimited transfers for 90 minutes. "
        "Anh thấy sao?"
    )
    assert GUARD.tito_guard_trigger(text) == "english_sentence", GUARD.tito_guard_trigger(text)


def _sentence(n_words: int, end: str = ".") -> str:
    """A Vietnamese-looking sentence of exactly n_words whitespace tokens."""
    return " ".join(["nhớ"] * (n_words - 1) + [f"anh{end}"])


async def test_two_sentences_pass_three_trigger():
    """Limit is 2 sentences per paragraph (v0.12.3): two pass, three are long."""
    two = "Hôm nay trời đẹp ghê. Anh ăn cơm chưa?"
    assert GUARD.tito_guard_trigger(two) is None, GUARD.tito_guard_trigger(two)

    three = "Hôm nay trời đẹp ghê. Tito nấu cơm nha. Anh ăn cơm chưa?"
    assert GUARD.tito_guard_trigger(three) == "long_paragraph", GUARD.tito_guard_trigger(three)


async def test_closing_quote_ends_a_sentence():
    """A quote that closes after . ends the sentence: three sentences are long."""
    text = 'Bác nói: "Được rồi nha." Tito cũng vậy. Anh ăn cơm chưa?'
    assert GUARD.tito_guard_trigger(text) == "long_paragraph", GUARD.tito_guard_trigger(text)


async def test_three_sentence_list_item_is_long_paragraph():
    """Each list item is judged alone: three sentences in one item trigger."""
    text = (
        "Anh nên làm mấy việc này:\n"
        "- Kiểm tra máy. Thay dầu nha. Rửa xe luôn.\n"
        "- Mua nước nha. Anh thấy sao?"
    )
    assert GUARD.tito_guard_trigger(text) == "long_paragraph", GUARD.tito_guard_trigger(text)


async def test_sentence_word_cap_is_twenty_five():
    """A sentence of 25 words passes; 26 words is long_paragraph."""
    assert GUARD.MAX_SENTENCE_WORDS == 25 and GUARD.MAX_SENTENCES == 2
    assert GUARD.MAX_UNIT_WORDS == 45
    ok = f"{_sentence(25)} Anh ăn cơm chưa?"
    assert GUARD.tito_guard_trigger(ok) is None, GUARD.tito_guard_trigger(ok)

    long_one = f"{_sentence(26)} Anh ăn cơm chưa?"
    assert GUARD.tito_guard_trigger(long_one) == "long_paragraph", GUARD.tito_guard_trigger(long_one)


async def test_unit_word_cap_is_forty_five():
    """Two sentences of 22 + 23 words pass; 23 + 23 (46 words) is long_paragraph."""
    ok = f"{_sentence(22)} {_sentence(23, '?')}"
    assert GUARD.tito_guard_trigger(ok) is None, GUARD.tito_guard_trigger(ok)

    over = f"{_sentence(23)} {_sentence(23, '?')}"
    assert GUARD.tito_guard_trigger(over) == "long_paragraph", GUARD.tito_guard_trigger(over)


async def test_list_item_word_cap_is_forty_five():
    """The 45-word cap applies to each list item, not only to plain paragraphs."""
    ok = (
        "Anh nên làm mấy việc này:\n"
        f"- {_sentence(22)} {_sentence(23)}\n"
        "- Mua nước nha. Anh thấy sao?"
    )
    assert GUARD.tito_guard_trigger(ok) is None, GUARD.tito_guard_trigger(ok)

    over = (
        "Anh nên làm mấy việc này:\n"
        f"- {_sentence(23)} {_sentence(23)}\n"
        "- Mua nước nha. Anh thấy sao?"
    )
    assert GUARD.tito_guard_trigger(over) == "long_paragraph", GUARD.tito_guard_trigger(over)


async def test_url_is_not_counted_for_words_or_sentences():
    """A URL token is not a word: real words plus a URL still pass."""
    with_url = (
        " ".join(["nhớ"] * 19)
        + " https://www.kbb.com/toyota/camry/2009/values anh. Anh ăn cơm chưa?"
    )
    assert GUARD.tito_guard_trigger(with_url) is None, GUARD.tito_guard_trigger(with_url)

    # A URL that ends a sentence still ends it: three sentences stay long.
    url_end = "Xem https://www.kbb.com/values. Hôm nay trời đẹp ghê. Anh ăn cơm chưa?"
    assert GUARD.tito_guard_trigger(url_end) == "long_paragraph", GUARD.tito_guard_trigger(url_end)


async def test_short_quoted_english_is_not_counted_for_words():
    """A quoted English line of 12 words or fewer adds no words to the sentence."""
    # 17 Vietnamese words + a 12-word English quote + "nhớ." would be 30 words
    # if the quote were counted, so the pass proves the quote is not counted.
    quote12 = "Would you take forty five dollars for this car today sir thanks"
    assert len(quote12.split()) == 12
    prefix = " ".join(["nhớ"] * 17)
    text = f'{prefix} "{quote12}" nhớ. Anh thấy sao?'
    assert GUARD.tito_guard_trigger(text) is None, GUARD.tito_guard_trigger(text)

    # A 13-word quote is not exempt: 17 + 13 + 1 words is over 25.
    quote13 = quote12 + " now"
    assert len(quote13.split()) == 13
    text = f'{prefix} "{quote13}" nhớ. Anh thấy sao?'
    assert GUARD.tito_guard_trigger(text) == "long_paragraph", GUARD.tito_guard_trigger(text)


async def main():
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        await test()
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    asyncio.run(main())
