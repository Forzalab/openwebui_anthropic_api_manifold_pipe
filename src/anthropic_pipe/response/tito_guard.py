"""Tito output guard: one Haiku check of the visible reply text.

Tito's answer is streamed exactly as before. Only when a cheap local pre-check
finds something that looks like a rule break is ``claude-haiku-5-5`` asked for
a minimal fix; if the edit passes the safety diff, the pipe replaces the
message once. Everything that is not visible prose — ``<details>``
collapsibles, ``[anthropic-hidden-...]`` carriers and ``[](anthropic:...)``
metadata markers — is kept byte-for-byte; only prose ever reaches Haiku.

Compiled into ``anthropic_pipe.py`` as a module-level generated section
(``helpers/build_anthropic_pipe.py``).
"""
from __future__ import annotations

import json
import logging
import re

logger = logging.getLogger(__name__)

TITO_GUARD_MODEL = "claude-haiku-5-5"

TITO_GUARD_RULES = """You fix a reply from "Tito", an old Vietnamese friend talking to "anh Đính" on a phone. Make the SMALLEST edits so all rules hold. NEVER change facts, numbers, dates, names, places, prices, links, or meaning. NEVER add new information.

1. Exactly ONE question, as the very last sentence, ending with "?". Remove or turn other questions into statements. If there is no question, add ONE short warm question that fits the reply.
2. Tito calls himself "Tito", never "tôi"/"em"/"con". Dad = "anh" / "anh Đính". Mom = "chị Ngân". Son = "thằng Tùng". Never "Tony", never "ông" for Dad.
3. No talk about tools, data, systems, records, AI, or "bịa/nói bậy". If the reply says it cannot look something up, replace that with a light friend line ("trí nhớ Tito dạo này như cá vàng").
4. No offer of things Tito can't do: calendar entries, ringing reminders, phone calls. Replace with "để Tito ghi ra cho anh nhớ nha".
5. No telling him to sleep, rest, stop, or "để mai" unless he said he is tired.
6. No flattery opener about his question ("câu hỏi hay", "đúng ý").
7. No lecture or moralizing sentence; keep facts.
8. Paragraphs: 1 to 3 short sentences, blank line between. No long dashes (— –); use commas or periods.
9. Do not quote poems.

If the text contains a marker like [[TITO-KEEP-1]], keep every marker exactly where it is, exactly once, and do not change it.
Return JSON only: {"ok": true} if nothing breaks, else {"ok": false, "edited": "<the full fixed text>"}."""

# Invisible parts of the persisted message. Matched spans are never sent to
# Haiku and are re-inserted from the original string, so they cannot drift.
_HIDDEN_PATTERNS = (
    # Any collapsible: tool calls, reasoning, code interpreter, cache trace,
    # cache diagnostics, compaction.
    re.compile(r"\n?<details\b[^>]*>.*?</details>\n?", re.DOTALL),
    # Invisible markdown link-reference carriers (encrypted server-tool blocks).
    re.compile(r"^[ ]{0,3}\[anthropic-hidden[^\]]*\]:[^\n]*(?:\n|$)", re.MULTILINE),
    # Inline metadata markers (file anchors, container id, cache-diagnostics id).
    re.compile(r"[ ]?\[\]\(anthropic:[^)]*\)[ ]?"),
)

# Marker the model must keep in place between prose fragments. It is checked
# before the edit is accepted, so a dropped or altered marker rejects the edit.
_TITO_GUARD_SENTINEL = "\n\n[[TITO-KEEP-{n}]]\n\n"

# Local pre-check: these are the cheap signals that a rule may be broken.
_FORBIDDEN_WORDS_RE = re.compile(
    r"công cụ|dữ liệu|hệ thống|bịa|nói bậy|trí tuệ nhân tạo|đi ngủ|để mai",
    re.IGNORECASE,
)
# "AI" is only the acronym: a case-insensitive match would hit the everyday
# Vietnamese word "ai" (who) in almost every reply.
_AI_ACRONYM_RE = re.compile(r"\bAI\b")
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s)")

_NUMBER_RE = re.compile(r"\d+(?:[.,:/-]\d+)*")
_URL_RE = re.compile(r"https?://[^\s<>\"']+")
_DIGIT_TOKEN_RE = re.compile(r"\S*\d\S*")


def _hidden_spans(content: str) -> list:
    """Merged, ordered ``(start, end)`` spans of everything that is not prose."""
    spans = []
    for pattern in _HIDDEN_PATTERNS:
        spans.extend((match.start(), match.end()) for match in pattern.finditer(content))
    spans.sort()
    merged = []
    for start, end in spans:
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def split_visible(content: str) -> list:
    """Split content into ``(is_visible, text)`` segments.

    Hidden segments are exact slices of the input, so concatenating every
    segment reproduces ``content`` byte-for-byte.
    """
    if not content:
        return []
    segments = []
    cursor = 0
    for start, end in _hidden_spans(content):
        if start > cursor:
            segments.append((True, content[cursor:start]))
        segments.append((False, content[start:end]))
        cursor = end
    if cursor < len(content):
        segments.append((True, content[cursor:]))
    return segments


def visible_text(segments: list) -> str:
    """Concatenated prose only: every hidden segment dropped."""
    return "".join(text for is_visible, text in segments if is_visible)


def tito_guard_layout(segments: list) -> tuple:
    """Return ``(text_for_haiku, sentinels)`` for the visible segments only."""
    visible = [text for is_visible, text in segments if is_visible]
    if not visible:
        return "", []
    sentinels = [
        _TITO_GUARD_SENTINEL.format(n=index)
        for index in range(1, len(visible))
    ]
    parts = [visible[0]]
    for sentinel, text in zip(sentinels, visible[1:]):
        parts.append(sentinel)
        parts.append(text)
    return "".join(parts), sentinels


def tito_guard_rebuild(segments: list, edited: str) -> "str | None":
    """Put edited prose back around the original hidden segments.

    Returns None when a sentinel is missing, duplicated or out of order: the
    safer outcome is then to keep the streamed reply untouched.
    """
    expected = sum(1 for is_visible, _ in segments if is_visible)
    if not expected:
        return None
    sentinels = [
        _TITO_GUARD_SENTINEL.format(n=index) for index in range(1, expected)
    ]
    edited_visible = []
    rest = edited
    for sentinel in sentinels:
        if rest.count(sentinel) != 1:
            return None
        head, _, rest = rest.partition(sentinel)
        edited_visible.append(head)
    edited_visible.append(rest)
    if len(edited_visible) != expected:
        return None

    parts = []
    visible_index = 0
    for is_visible, text in segments:
        if is_visible:
            parts.append(edited_visible[visible_index])
            visible_index += 1
        else:
            parts.append(text)
    return _CARRIER_NEEDS_BLANK_RE.sub("\\1\n\n", "".join(parts))


# A link-reference carrier only stays invisible after a BLANK line (see
# formatting._format_hidden_block). The edit path can eat that blank line
# (sentinel partition, model trimming), which renders the base64 as text.
_CARRIER_NEEDS_BLANK_RE = re.compile(r"([^\n])\n?(?=[ ]{0,3}\[anthropic-hidden[^\]]*\]:)")


def tito_guard_trigger(text: str) -> "str | None":
    """Cheap local pre-check: name of the first rule signal, or None when clean."""
    if not text or not text.strip():
        return None
    stripped = text.rstrip(" *\t\r\n")
    if not stripped.endswith("?"):
        return "no_final_question"
    if text.count("?") > 1:
        return "multiple_questions"
    if re.search(r"\btôi\b", text, re.IGNORECASE):
        return "first_person"
    if "\u2014" in text or "\u2013" in text:
        return "long_dash"
    if _FORBIDDEN_WORDS_RE.search(text) or _AI_ACRONYM_RE.search(text):
        return "forbidden_word"
    for paragraph in re.split(r"\n\s*\n", text):
        paragraph = paragraph.strip()
        if not paragraph:
            continue
        if len(_SENTENCE_END_RE.split(paragraph)) > 3:
            return "long_paragraph"
    return None


def _safety_tokens(original: str) -> set:
    """Numbers, URLs and digit-bearing tokens that an edit must not lose."""
    tokens = set(_NUMBER_RE.findall(original))
    tokens.update(
        url.rstrip(".,;:!?)]}'\"") for url in _URL_RE.findall(original)
    )
    for token in _DIGIT_TOKEN_RE.findall(original):
        tokens.add(token.strip(".,;:!?)]}'\"("))
    return {token for token in tokens if token}


def tito_guard_safety_ok(original: str, edited: str) -> bool:
    """True when every number, URL and digit-bearing token survived the edit."""
    return all(token in edited for token in _safety_tokens(original))


def _parse_guard_json(text: str):
    """Extract the JSON object from a Haiku answer, tolerating a markdown fence."""
    candidate = (text or "").strip()
    if candidate.startswith("```"):
        candidate = re.sub(r"^```[A-Za-z]*\s*", "", candidate)
        candidate = re.sub(r"\s*```$", "", candidate)
    try:
        data = json.loads(candidate)
        return data if isinstance(data, dict) else None
    except (TypeError, ValueError):
        pass
    start = candidate.find("{")
    end = candidate.rfind("}")
    if start == -1 or end <= start:
        return None
    try:
        data = json.loads(candidate[start : end + 1])
    except (TypeError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def _guard_response_text(response) -> str:
    parts = []
    for block in getattr(response, "content", None) or []:
        if isinstance(block, dict):
            if block.get("type") == "text":
                parts.append(block.get("text", "") or "")
        elif getattr(block, "type", None) == "text":
            parts.append(getattr(block, "text", "") or "")
    return "".join(parts)


async def guard_verdict(client, visible_text: str) -> tuple:
    """One Haiku call. Returns ``(outcome, edited)``.

    ``outcome`` is ``"kept_ok"`` (nothing breaks), ``"kept_parse"`` (no usable
    JSON answer), ``"kept_safety"`` (the edit lost a number, URL or digit token)
    or ``"replace"`` with the edited text. Client exceptions propagate.
    No ``temperature`` is sent: current models reject it.
    """
    if not visible_text or not visible_text.strip():
        return ("kept_ok", None)

    response = await client.messages.create(
        model=TITO_GUARD_MODEL,
        max_tokens=min(8192, max(256, len(visible_text) * 2)),
        system=TITO_GUARD_RULES,
        messages=[{"role": "user", "content": visible_text}],
    )

    data = _parse_guard_json(_guard_response_text(response))
    if data is None:
        return ("kept_parse", None)
    if data.get("ok") is not False:
        return ("kept_ok", None)
    edited = data.get("edited")
    if not isinstance(edited, str) or not edited.strip():
        return ("kept_parse", None)
    if not tito_guard_safety_ok(visible_text, edited):
        return ("kept_safety", None)
    return ("replace", edited)


async def guard(client, visible_text: str):
    """One Haiku call. Returns the edited text, or None to keep the original.

    Raises when the client fails; the caller treats every failure as "keep".
    """
    outcome, edited = await guard_verdict(client, visible_text)
    return edited if outcome == "replace" else None
