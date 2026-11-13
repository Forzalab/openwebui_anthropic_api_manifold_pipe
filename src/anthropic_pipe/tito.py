"""Tito model patches: fixed config, budget gate, mid-conversation system
messages, frozen tool array and error mapping. Compiled into ``class Pipe`` as a
method group (``helpers/build_anthropic_pipe.py``). Self-contained: no
``open_webui`` / ``anthropic`` imports at module level so it can be imported in
isolation by ``tito-spec/tests/test_tito.py``.
"""
from __future__ import annotations

import contextvars
import hashlib
import json
import logging
import os
import time
from datetime import datetime

logger = logging.getLogger(__name__)


class PipeTitoMethods:
    """Everything gated on the OWUI model id ``tito``.

    All state that must survive across turns is kept lazily on the instance
    (``_tito_spend_cache``, ``_tito_soft_warned``, ``_tito_fingerprint_state``)
    rather than in ``__init__``, so the mixin does not need to touch the
    template's constructor.
    """

    TITO_MODEL_ID = "tito"

    # One Pipe object serves concurrent requests; the active flag is a ContextVar
    # for the same reason as HIDDEN_BLOCKS / SLIM_OUTPUT in pipe_template.py.
    TITO_ACTIVE: contextvars.ContextVar[bool] = contextvars.ContextVar(
        "anthropic_pipe_tito_active", default=False
    )

    # Frozen config for "tito" (SPEC section 2). Keys not present here fall
    # through to the existing valve behaviour; keys present here override it.
    TITO_CONFIG: dict = {
        "thinking": {"type": "adaptive"},
        "effort": "medium",
        "advisor": {"enabled": True, "model": "claude-opus-5-5", "max_uses": 3, "caching": "5m"},
        # Executor model-id prefix -> advisor model; wins over advisor.model when
        # the executor starts with the prefix (any claude-haiku-* gets Sonnet).
        "advisor_by_executor": {"claude-haiku-": "claude-sonnet-5-5"},
        "enable_tool_search": False,
        "hide_blocks": ["advisor", "web_search", "web_fetch"],
        "web_search_max_uses": 3,
        "web_fetch_max_uses": 2,
        "web_fetch_max_content_tokens": 8000,
        "user_location": {"city": "Fresno"},
        "cache": {"ttl": "1h", "messages_ttl": "5m", "tools_breakpoint": False, "system_breakpoints": 2},
        "show_cost": True,
        "strip_reasoning_effort": True,
        "skip_tool_search_nudge": True,
        "skip_terminal_hint": True,
        "always_strip_tool_manifest": True,
    }

    # Copied (not imported) from open-webui backend/open_webui/utils/tito.py.
    TITO_SNAPSHOT_HEADING = "NHỮNG ĐIỀU TITO NHỚ VỀ ANH"
    _TITO_ALERT_MAX = 500
    _tito_tools_alerted = False  # once per process

    HARD_CAP_TEXT = "Tito đang nghỉ tới ngày 1 tháng sau. Dori đã biết rồi, ông không cần làm gì."
    TITO_ERROR_TEXT = "Tito đang bận một chút, ông thử lại sau vài phút nhé."
    TITO_MISSING_TOOL_TEXT = "Lịch tạm thời không dùng được, ông thử lại lát nữa nhé."

    @staticmethod
    def _is_tito(metadata) -> bool:
        """True when the request is for the Tito model.

        OWUI rewrites ``body["model"]`` to the base model id before the pipe sees
        it, so the gate is the OWUI model id in metadata, never ``body["model"]``.
        """
        return (metadata or {}).get("model", {}).get("id") == "tito"

    def _tito_quiet_ui(self, metadata) -> bool:
        """True when Tito's quiet UI applies: Tito request AND the valve is on."""
        return bool(
            getattr(self.valves, "TITO_QUIET_UI", True) and self._is_tito(metadata)
        )

    async def _tito_output_guard(
        self, request_ctx, client, metadata, total_usage=None, pricing=None
    ) -> str:
        """Check Tito's visible reply with Haiku and replace it once if needed.

        Only visible prose is sent; ``<details>`` blocks, ``[anthropic-hidden-]``
        carriers and ``[](anthropic:)`` markers are re-inserted byte-for-byte.
        Never raises: any failure leaves the streamed reply exactly as it is.

        Logs exactly one INFO line per Tito run (lengths only, never reply text):
        ``outcome`` is clean | kept_ok | kept_parse | kept_safety | kept_markers |
        replaced | error. On error a ``TITO-ERR`` line with a support id follows.

        With ``total_usage`` and ``pricing`` the Haiku call's own cost is added
        to ``cost_usd`` and ``cost_breakdown_usd['guard']`` (before the usage is
        persisted); a missing usage or rate card adds nothing.
        """
        if not getattr(self.valves, "TITO_OUTPUT_GUARD", True):
            return "none"
        if not self._is_tito(metadata):
            return "none"
        timing = getattr(request_ctx, "timing", None)
        if timing is not None:
            timing["guard_t0"] = time.monotonic()

        usage_sink: list = []
        outcome = "error"
        trigger = "none"
        content = ""
        visible = ""
        rebuilt_len = 0
        try:
            content = request_ctx.text() or ""
            segments = split_visible(content)
            visible = visible_text(segments)
            found = tito_guard_trigger(visible) if visible.strip() else None
            if not found:
                outcome = "clean"
            else:
                trigger = found
                request_text, _sentinels = tito_guard_layout(segments)
                verdict, edited = await guard_verdict(client, request_text, usage_sink)
                outcome = verdict
                if verdict == "replace":
                    rebuilt = tito_guard_rebuild(segments, edited)
                    if rebuilt is None:
                        outcome = "kept_markers"
                    elif rebuilt == content:
                        outcome = "kept_ok"
                    else:
                        await request_ctx.emit_replace(rebuilt)
                        rebuilt_len = len(rebuilt)
                        outcome = "replaced"
        except Exception as exc:
            outcome = "error"
            self._tito_err("output_guard", exc)
        if total_usage is not None and pricing is not None and usage_sink:
            try:
                guard_model, guard_usage = usage_sink[0]
                guard_usd = pricing.price_call(guard_model, guard_usage)
                if guard_usd:
                    total_usage["cost_usd"] = round(total_usage.get("cost_usd", 0.0) + guard_usd, 6)
                    breakdown = dict(total_usage.get("cost_breakdown_usd") or {})
                    breakdown["guard"] = round(breakdown.get("guard", 0.0) + guard_usd, 6)
                    total_usage["cost_breakdown_usd"] = breakdown
            except Exception as exc:
                logger.debug("[TITO] guard cost capture failed: %s", exc)
        logger.info(
            "[TITO] output guard outcome=%s trigger=%s chars=%d visible=%d rebuilt=%d",
            outcome,
            trigger,
            len(content),
            len(visible),
            rebuilt_len,
        )
        return trigger

    def _tito_config(self) -> dict:
        """Return the frozen Tito config (a shallow copy to avoid in-place edits)."""
        return dict(self.TITO_CONFIG)

    @staticmethod
    def _is_mid_conversation_system(msg) -> bool:
        """True for a mid-conversation system message.

        Accepts both the wire form produced by OWUI (top-level
        ``mid_conversation: true``) and the pipe's own emitted form
        (``role: "system"`` + ``clear_at: "next_user_message"``).
        """
        if not isinstance(msg, dict):
            return False
        return msg.get("mid_conversation") is True or (
            msg.get("role") == "system" and msg.get("clear_at") == "next_user_message"
        )

    @staticmethod
    def _tito_cap(env_name: str, default: float) -> float:
        """Read a USD cap from the environment with a default, tolerating garbage."""
        raw = (os.environ.get(env_name) or "").strip()
        if not raw:
            return default
        try:
            return float(raw)
        except (TypeError, ValueError):
            return default

    @classmethod
    def _tito_budget_decision(cls, month_spend_usd: float):
        """Pure decision function: (status, message) for a month's spend."""
        hard = cls._tito_cap("TITO_HARD_CAP_USD", 22.0)
        soft = cls._tito_cap("TITO_SOFT_CAP_USD", 15.0)
        if month_spend_usd >= hard:
            return ("hard", cls.HARD_CAP_TEXT)
        if month_spend_usd >= soft:
            return ("soft", None)
        return ("ok", None)

    @staticmethod
    def _tito_pacific_utc_offset(dt) -> int:
        """UTC offset (hours) of America/Los_Angeles at ``dt``: -8 PST / -7 PDT.

        Implemented by hand (US DST rules) rather than ``zoneinfo`` so the
        compiled artifact does not depend on a system tzdata database.
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz

        def _nth_sunday(year: int, month: int, n: int):
            first = _dt(year, month, 1, tzinfo=_tz.utc)
            days_to_sunday = (6 - first.weekday()) % 7
            return first + _td(days=days_to_sunday + (n - 1) * 7)

        dst_start = _nth_sunday(dt.year, 3, 2) + _td(hours=10)  # 02:00 PST
        dst_end = _nth_sunday(dt.year, 11, 1) + _td(hours=9)    # 02:00 PDT
        return -7 if dst_start <= dt < dst_end else -8

    @classmethod
    def _tito_pacific_month(cls, dt) -> str:
        """Calendar month key (``YYYY-MM``) for ``dt`` in America/Los_Angeles."""
        from datetime import timedelta as _td

        offset = cls._tito_pacific_utc_offset(dt)
        return (dt + _td(hours=offset)).strftime("%Y-%m")

    @staticmethod
    def _tito_safe_err(exc, limit: int = 500) -> str:
        """Exception text for alerts: type + message truncated to ``limit``, with
        anything that looks like an API key / auth header value redacted."""
        import re as _re

        raw = f"{type(exc).__name__}: {exc}"
        raw = _re.sub(r"https?://\S+", "[url]", raw)
        raw = _re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[redacted]", raw)
        raw = _re.sub(
            r"(?i)(x-api-key|authorization|api[_-]?key|bearer)(['\"]?\s*[:=]?\s*)(?:Bearer\s+)?[^\s,'\"}]+",
            r"\1\2[redacted]",
            raw,
        )
        return raw[:limit]

    @classmethod
    def _tito_err(cls, component: str, exc) -> str:
        """Log one grep-able ``TITO-ERR <code> <component> <detail>`` line; return the code.

        Code is ``T{HHMM}-{4 hex}`` with HHMM in America/Los_Angeles. The detail
        goes through ``_tito_safe_err`` (type + message, URLs and keys redacted).
        """
        from datetime import datetime as _dt, timedelta as _td, timezone as _tz

        now = _dt.now(_tz.utc)
        local = now + _td(hours=cls._tito_pacific_utc_offset(now))
        code = f"T{local:%H%M}-{os.urandom(2).hex()}"
        logger.error("TITO-ERR %s %s %s", code, component, cls._tito_safe_err(exc, limit=200))
        return code

    @classmethod
    def _tito_split_system(cls, system):
        """N1: one system text block holding ``"\n\n" + heading`` becomes two blocks
        so each gets its own cache breakpoint. Already-split input is unchanged."""
        if not isinstance(system, list) or len(system) != 1:
            return system
        blk = system[0]
        if not isinstance(blk, dict) or blk.get("type", "text") != "text":
            return system
        text = blk.get("text")
        marker = "\n\n" + cls.TITO_SNAPSHOT_HEADING
        if not isinstance(text, str) or marker not in text:
            return system
        head, _, tail = text.partition(marker)
        if not head.strip():
            return system
        first = dict(blk, text=head)
        second = dict(blk, text=cls.TITO_SNAPSHOT_HEADING + tail)
        return [first, second]

    @classmethod
    def _tito_frozen_tools(cls) -> list:
        """Load the frozen tools array from ``TITO_TOOLS_FILE`` (JSON list).

        Missing / unreadable / empty is loud: ERROR log + one Discord alert per
        process (the tools array is part of the cache prefix).
        """
        path = (os.environ.get("TITO_TOOLS_FILE") or "").strip()
        problem = None
        err = None
        tools: list = []
        if not path:
            problem = "TITO_TOOLS_FILE is not set"
        else:
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    data = json.load(fh)
                if isinstance(data, list):
                    tools = [t for t in data if isinstance(t, dict)]
                if not tools:
                    problem = f"TITO_TOOLS_FILE {path} has no tools"
            except Exception as e:
                err = e
                problem = f"TITO_TOOLS_FILE {path} unreadable: {cls._tito_safe_err(e)}"
        if problem:
            # One TITO-ERR line per failure (the ERROR record for this turn).
            cls._tito_err("tools_file", err if err is not None else RuntimeError(problem))
            if not cls._tito_tools_alerted:
                cls._tito_tools_alerted = True
                cls._tito_discord_alert(f"Tito: frozen tools unavailable: {problem}")
        return tools

    @classmethod
    def _strip_cache_control(cls, node):
        """Recursively drop ``cache_control`` markers (placement hints, not content)."""
        if isinstance(node, dict):
            return {
                k: cls._strip_cache_control(v)
                for k, v in node.items()
                if k != "cache_control"
            }
        if isinstance(node, list):
            return [cls._strip_cache_control(v) for v in node]
        return node

    @classmethod
    def _tito_fingerprint_shas(cls, payload: dict):
        """Return ``(sha256_tools, sha256_system)`` over the canonicalized payload."""
        def _ser(x):
            if not x:
                return ""
            return json.dumps(
                x, sort_keys=True, ensure_ascii=False, separators=(",", ":"), default=str
            )

        tools = cls._strip_cache_control(payload.get("tools") or [])
        system = cls._strip_cache_control(payload.get("system") or [])
        tools_sha = hashlib.sha256(_ser(tools).encode("utf-8")).hexdigest()
        system_sha = hashlib.sha256(_ser(system).encode("utf-8")).hexdigest()
        return tools_sha, system_sha

    @classmethod
    def _tito_fingerprint(cls, payload: dict) -> str:
        """One log line summarising the outgoing payload's stable identifiers."""
        tools_sha, system_sha = cls._tito_fingerprint_shas(payload)
        model = payload.get("model", "")
        thinking = payload.get("thinking") or {}
        thinking_s = thinking.get("type", "") if isinstance(thinking, dict) else str(thinking)
        effort = payload.get("effort", "")
        if not effort:
            oc = payload.get("output_config")
            if isinstance(oc, dict):
                effort = oc.get("effort", "")
        betas = payload.get("betas") or []
        betas_csv = ",".join(betas) if isinstance(betas, list) else str(betas)
        return (
            f"sha256_tools={tools_sha} sha256_system={system_sha} "
            f"model={model} thinking={thinking_s} effort={effort} betas={betas_csv}"
        )

    @classmethod
    def _tito_missing_tool_result(cls, tool_use_id: str = "", tool_name: str = "") -> dict:
        """The ``is_error`` tool_result returned when a frozen tool has no live callable."""
        return {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": cls.TITO_MISSING_TOOL_TEXT,
            "is_error": True,
        }

    @classmethod
    def _tito_error_text(cls) -> str:
        """The fixed Vietnamese line shown to the user on any Tito API error."""
        return cls.TITO_ERROR_TEXT

    @classmethod
    def _tito_discord_alert(cls, message: str) -> None:
        """Best-effort Discord webhook alert (fire-and-forget, never blocks/fails)."""
        webhook = (os.environ.get("TITO_DISCORD_WEBHOOK") or "").strip()
        if not webhook:
            return

        def _send():
            try:
                import json as _json
                import urllib.request as _ur

                data = _json.dumps({"content": message}).encode("utf-8")
                req = _ur.Request(
                    webhook,
                    data=data,
                    headers={"Content-Type": "application/json"},
                    method="POST",
                )
                _ur.urlopen(req, timeout=5)
            except Exception as e:
                cls._tito_err("discord_alert", e)

        try:
            import threading as _th

            _th.Thread(target=_send, daemon=True).start()
        except Exception:
            pass

    @staticmethod
    def _tito_chat_messages(chat) -> list:
        """Messages of a ChatModel (``.chat["history"]["messages"]``, dict keyed by id)."""
        if isinstance(chat, dict):
            body = chat.get("chat") if isinstance(chat.get("chat"), dict) else chat
        else:
            body = getattr(chat, "chat", None)
        if not isinstance(body, dict):
            return []
        msgs = (body.get("history") or {}).get("messages")
        if isinstance(msgs, dict):
            return [m for m in msgs.values() if isinstance(m, dict)]
        return []

    @classmethod
    def _tito_message_month(cls, msg, dt) -> "str | None":
        """Pacific month key for a message's timestamp, or None when absent/garbage."""
        if not isinstance(msg, dict):
            return None
        ts = msg.get("timestamp") or msg.get("created_at") or msg.get("updated_at")
        if ts is None:
            return None
        from datetime import datetime as _dt, timezone as _tz

        try:
            if isinstance(ts, datetime):
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=_tz.utc)
                return cls._tito_pacific_month(ts.astimezone(_tz.utc))
            return cls._tito_pacific_month(
                _dt.fromtimestamp(float(ts), tz=_tz.utc)
            )
        except (TypeError, ValueError, OSError, OverflowError):
            return None

    @staticmethod
    def _tito_usage_cost(msg) -> float:
        """``usage.cost_usd`` for a message, 0.0 when absent or malformed."""
        if not isinstance(msg, dict):
            return 0.0
        usage = msg.get("usage")
        if not isinstance(usage, dict):
            return 0.0
        try:
            return float(usage.get("cost_usd") or 0.0)
        except (TypeError, ValueError):
            return 0.0

    async def month_spend_usd(self, now=None) -> float:
        """Sum ``usage.cost_usd`` across all users' chats for the Pacific month of ``now``.

        Fail-open: any DB error returns 0.0 and fires a Discord alert, so the
        budget gate never blocks on a broken query. Cached for 60 seconds.
        The exact ORM/query against ``open_webui.models.chats`` is
        VERIFY-AT-RUNTIME (OWUI is not in this repo); the store point is the
        ``Chats.upsert_message_to_chat_by_id_and_message_id`` call in
        pipe_orchestrator.
        """
        if now is None:
            from datetime import datetime as _dt, timezone as _tz

            now = _dt.now(_tz.utc)

        cached = getattr(self, "_tito_spend_cache", None)
        if cached is not None:
            value, ts = cached
            if time.time() - ts < 60:
                return value

        total = 0.0
        try:
            chats_cls = globals().get("Chats")
            if not globals().get("CHATS_AVAILABLE", False) or chats_cls is None:
                return 0.0

            month_key = self._tito_pacific_month(now)
            chats = chats_cls.get_chats()
            if hasattr(chats, "__await__"):
                chats = await chats
            for chat in chats or []:
                for msg in self._tito_chat_messages(chat):
                    if self._tito_message_month(msg, now) == month_key:
                        total += self._tito_usage_cost(msg)
        except Exception as e:
            total = 0.0
            self._tito_err("month_spend", e)
            self._tito_discord_alert(
                f"Tito: month_spend_usd DB error (fail-open): {self._tito_safe_err(e)}"
            )

        setattr(self, "_tito_spend_cache", (total, time.time()))
        return total

    def _tito_log_fingerprint(self, payload: dict, metadata) -> None:
        """Log one fingerprint line per turn; warn + Discord on tools/system drift."""
        tools_sha, system_sha = self._tito_fingerprint_shas(payload)
        logger.info("[TITO] %s", self._tito_fingerprint(payload))

        chat_id = (metadata or {}).get("chat_id")
        if not chat_id:
            return
        state = getattr(self, "_tito_fingerprint_state", None)
        if state is None:
            state = {}
            self._tito_fingerprint_state = state
        prev = state.get(chat_id)
        if prev is not None and (prev[0] != tools_sha or prev[1] != system_sha):
            logger.warning(
                "[TITO] tools/system changed within chat_id=%s "
                "(tools=%s system=%s)",
                chat_id,
                "changed" if prev[0] != tools_sha else "same",
                "changed" if prev[1] != system_sha else "same",
            )
            self._tito_discord_alert(
                f"Tito: tools/system fingerprint changed for chat_id={chat_id}"
            )
        state[chat_id] = (tools_sha, system_sha)

    def _tito_vn_reminder_apply(self, payload: dict, extra_text: str = ""):
        """Append the Tito per-turn VN reminder after the cache breakpoints.

        Called only for Tito, tool-loop iteration 1, after ``_apply_cache_control``.
        The reminder is one text block (no cache_control) appended to the last
        user message, so every breakpoint still ends before it and the cached
        prefix bytes do not change. A str content becomes a one-block list, the
        only byte change to that message; it is safe only because no breakpoint
        sits on a later message, so the case is skipped otherwise. Returns a state
        for ``_tito_vn_reminder_remove``, or None. Never logs the text.
        """
        _tito_cv = getattr(self, "TITO_ACTIVE", None)
        if _tito_cv is None or not _tito_cv.get():
            return None
        text = str(getattr(self.valves, "TITO_VN_REMINDER", "") or "").strip()
        extra = (extra_text or "").strip()
        text = "\n\n".join(part for part in (text, extra) if part)
        if not text:
            return None
        try:
            messages = payload.get("messages") or []
            target = None
            for idx in range(len(messages) - 1, -1, -1):
                msg = messages[idx]
                if isinstance(msg, dict) and msg.get("role") == "user":
                    target = idx
                    break
            if target is None:
                return None
            for later in messages[target + 1:]:
                content = later.get("content") if isinstance(later, dict) else None
                if isinstance(content, list) and any(
                    isinstance(b, dict) and "cache_control" in b for b in content
                ):
                    logger.info("[TITO] vn_reminder skipped: breakpoint after last user message")
                    return None
            msg = messages[target]
            original = msg.get("content")
            if isinstance(original, str):
                content = [{"type": "text", "text": original}]
                msg["content"] = content
            elif isinstance(original, list):
                content = original
            else:
                return None
            block = {"type": "text", "text": text}
            content.append(block)
            logger.info("[TITO] vn_reminder chars=%d", len(text))
            return {"block": block, "original": original}
        except Exception as exc:
            self._tito_err("vn_reminder", exc)
            return None

    @staticmethod
    def _tito_vn_reminder_remove(payload: dict, state) -> None:
        """Drop the reminder block (found by identity) and restore a str content."""
        if not state:
            return
        for msg in payload.get("messages") or []:
            content = msg.get("content") if isinstance(msg, dict) else None
            if not isinstance(content, list):
                continue
            for i, block in enumerate(content):
                if block is state["block"]:
                    del content[i]
                    if isinstance(state["original"], str):
                        msg["content"] = state["original"]
                    return

    # TITO_DECIDE (shadow mode): MSD-1 yes/no flags per turn, logged only.
    # TITO_DECIDE_FLAGS are input flags (scored in parallel on the last user message);
    # TITO_DECIDE_OUT_FLAGS is the output check on Tito's visible reply.
    TITO_DECIDE_URL = "https://openrouter.ai/api/alpha/decisions"
    # Main model first; the fallback is asked only when the main call fails, within the same budget.
    TITO_DECIDE_MODEL = "~typesafe/jev-latest"
    TITO_DECIDE_FALLBACK_MODEL = "microsoft/microsoft-decision-1"
    # Every question carries an "instructions" field (some decision models reject questions without it).
    TITO_DECIDE_IN_INSTRUCTIONS = "The state is one chat message, in Vietnamese, from an elderly Vietnamese immigrant dad in Fresno to a friendly chat companion. Judge the message as written."
    TITO_DECIDE_OUT_INSTRUCTIONS = "The state is one reply, in Vietnamese, from a friendly chat companion to an elderly Vietnamese immigrant dad. Judge the reply as written."
    TITO_DECIDE_FLAGS = {
        "smart_consumer": {
            "question": "Is the dad weighing a purchase, a sale, a price, a seller, a used item, or money paperwork (an official letter, a title, a bill), where a fair price range, the one or two things worth checking, or facts a seller might leave out would help him?",
            "criteria": {"yes": "A buying, selling, price or money-paperwork situation",
                         "no": "No purchase, price, seller or money paperwork involved"},
        },
        "needs_search": {
            "question": "Does a good reply need anything LOCAL (Fresno, his bus, his job, nearby places or events), RECENT (prices, news, rules that change), or beyond common knowledge, so a quick web search would make it better?",
            "criteria": {"yes": "A web search would improve the reply",
                         "no": "The reply works fine from general knowledge or feelings alone"},
        },
        "casual_chat": {
            "question": "Is the dad just chatting about his day or his feelings, rather than asking for information, advice, or a decision? Chatting examples: 'Nay khách sạn đông quá', 'Hôm qua tôi nấu canh chua', 'Mấy bữa nay trời lạnh'. Asking examples: 'Thuốc này uống mấy viên vây!?', 'Chỗ nào bán phở ngon gần nhà?', 'Cái này tiếng Anh nói sao?'",
            "criteria": {"yes": "Casual chat about his day or feelings",
                         "no": "Asking for information, advice, or a decision"},
        },
    }
    # Output check: is Tito's visible reply fully natural Vietnamese? No regex fallback.
    TITO_DECIDE_OUT_FLAGS = {
        "pure_vn": {
            "question": "Is this reply written fully in natural everyday Vietnamese? English is fine only for names, brands, model names, short terms like title or smog, and URLs. Whole English sentences or English quotes make it NOT pure.",
            "criteria": {"yes": "Fully natural Vietnamese apart from names, brands and short terms",
                         "no": "Contains English sentences or quotes, or reads like a translation"},
        },
    }
    # Regex fallback per flag (Vietnamese keywords, case-insensitive, NFC text).
    # casual_chat has no regex fallback: on MSD failure it gets no score (src=none).
    TITO_DECIDE_REGEX = {
        "smart_consumer": r"\b(giá|đô|bán|mua|title|tiền|thuế|IRS|marketplace)\b",
        "needs_search": r"\b(bao nhiêu|ở đâu|mới|bây giờ|Fresno|tin)\b",
    }
    # INJECT mode (valve TITO_DECIDE_INJECT): MSD-scored flags at or above these
    # thresholds add one line to the per-turn block. casual_chat always adds one
    # of two lines (chat or question), split at its threshold.
    TITO_DECIDE_INJECT_MIN = {"smart_consumer": 0.5, "needs_search": 0.25}
    TITO_DECIDE_INJECT_LINES = {
        "smart_consumer": "anh is weighing a purchase. Tito is the friend who knows prices: a fair range first, the one or two things worth checking, and an English line anh can say to bargain, laid out so anh can scan it on his phone, in everyday words. The choice stays anh's.",
        "needs_search": "anh's question touches something local, recent, or past what everyone already knows. Tito looks it up first, then talks like someone who just checked.",
        "casual_chat":"anh is sharing about his day. A good friend hears the specific thing anh said and echoes it back warmly, sometimes with a gentle guess at how it felt (\"chắc anh cũng…\"), brings in what anh shared before, stays curious about one detail (a place, a smell, a meal, a person), and asks one easy question about it. Half-answers are fine. When anh brings up the old days, the friend walks there with him, then back to today.",
    }
    TITO_DECIDE_CASUAL_NO_LINE = "anh is asking about something he wants to know."
    # Log line order and short names: sc=smart_consumer, search=needs_search, casual=casual_chat.
    TITO_DECIDE_LOG_NAMES = (
        ("smart_consumer", "sc"),
        ("needs_search", "search"),
        ("casual_chat", "casual"),
    )

    def _tito_decide_key(self) -> str:
        """Plaintext TITO_DECIDE_KEY, or "" when unset or unreadable. Never logged."""
        try:
            raw = getattr(self.valves, "TITO_DECIDE_KEY", "") or ""
            return decrypt_valve_secret(raw).strip()
        except Exception:
            return ""

    def _tito_decide_enabled(self) -> bool:
        """True when the valve is on and a key is configured."""
        return bool(getattr(self.valves, "TITO_DECIDE", False)) and bool(self._tito_decide_key())

    def _tito_decide_models(self) -> tuple:
        """(main, fallback) decide model ids from the valves; empty valve = class default."""
        main = (getattr(self.valves, "TITO_DECIDE_MODEL", "") or "").strip() or self.TITO_DECIDE_MODEL
        fallback = (
            (getattr(self.valves, "TITO_DECIDE_FALLBACK_MODEL", "") or "").strip()
            or self.TITO_DECIDE_FALLBACK_MODEL
        )
        return main, fallback

    @classmethod
    def _tito_decide_post(cls, api_key: str, text: str, flag: str, timeout_s: float, model: str) -> float:
        """Blocking decide call for one flag on ``model``; returns probabilities.yes. Runs in a thread."""
        import json as _json
        import urllib.request as _ur

        spec = {**cls.TITO_DECIDE_FLAGS, **cls.TITO_DECIDE_OUT_FLAGS}
        instructions = (
            cls.TITO_DECIDE_OUT_INSTRUCTIONS if flag in cls.TITO_DECIDE_OUT_FLAGS
            else cls.TITO_DECIDE_IN_INSTRUCTIONS
        )
        body = {
            "model": model,
            "state": text,
            "questions": {flag: {"type": "choice", "instructions": instructions, **spec[flag]}},
        }
        req = _ur.Request(
            cls.TITO_DECIDE_URL,
            data=_json.dumps(body).encode("utf-8"),
            method="POST",
            headers={
                "Authorization": "Bearer " + api_key,
                "Content-Type": "application/json",
            },
        )
        with _ur.urlopen(req, timeout=timeout_s) as resp:
            data = _json.loads(resp.read())
        prob = float(data["answers"][flag]["probabilities"]["yes"])
        if not 0.0 <= prob <= 1.0:
            raise ValueError("probability out of range")
        return prob

    @classmethod
    def _tito_decide_regex(cls, text: str, flag: str) -> float:
        """Regex fallback score for one flag: 1.0 on a keyword hit, else 0.0."""
        import re as _re
        import unicodedata as _ud

        norm = _ud.normalize("NFC", text or "")
        return 1.0 if _re.search(cls.TITO_DECIDE_REGEX[flag], norm, _re.IGNORECASE) else 0.0

    @classmethod
    def _tito_decide_fallback(cls, text: str, flag: str) -> dict:
        """Score for a flag whose MSD calls all failed. casual_chat gets no score (src=none)."""
        if flag == "casual_chat":
            return {"p": 0.0, "src": "none"}
        return {"p": cls._tito_decide_regex(text, flag), "src": "regex"}

    async def _tito_decide_ask(self, key: str, text: str, flag: str, timeout_s: float):
        """Ask Jev (main), then MSD-1 (fallback) if Jev failed, within one ``timeout_s`` budget.

        Returns ``{"p": float, "src": "jev" | "msd"}`` or None when both fail or time runs out.
        Never raises.
        """
        import asyncio as _aio

        try:
            main, fallback = self._tito_decide_models()
            t0 = time.monotonic()
            for model, src in ((main, "jev"), (fallback, "msd")):
                remaining = timeout_s - (time.monotonic() - t0)
                if remaining <= 0:
                    break
                try:
                    prob = await _aio.wait_for(
                        _aio.to_thread(self._tito_decide_post, key, text, flag, remaining, model),
                        remaining,
                    )
                    return {"p": prob, "src": src}
                except Exception:
                    # Silent on purpose: the decide line shows the source that answered, or regex/none.
                    continue
            return None
        except Exception:
            return None

    async def _tito_decide_shadow(self, text: str):
        """Score the input flags for ``text``. Never raises.

        Each flag asks Jev, then MSD-1, concurrently with the per-flag budget
        TITO_DECIDE_TIMEOUT_MS; a flag that fails or times out falls back to the
        regex (casual_chat: none). Returns ``{flag: {"p": float, "src": "jev" | "msd" | "regex" | "none"}}``
        or None on an unexpected failure.
        """
        import asyncio as _aio

        try:
            timeout_s = max(0.1, int(getattr(self.valves, "TITO_DECIDE_TIMEOUT_MS", 1500) or 1500) / 1000)
            key = self._tito_decide_key()
            results: dict = {}

            async def one(flag: str) -> None:
                got = await self._tito_decide_ask(key, text, flag, timeout_s) if key else None
                results[flag] = got or self._tito_decide_fallback(text, flag)

            tasks = [_aio.ensure_future(one(flag)) for flag in self.TITO_DECIDE_FLAGS]
            try:
                await _aio.wait(tasks, timeout=timeout_s)
            finally:
                for task in tasks:
                    if not task.done():
                        task.cancel()
            out = {}
            for flag in self.TITO_DECIDE_FLAGS:
                out[flag] = results.get(flag) or self._tito_decide_fallback(text, flag)
            return out
        except Exception as exc:
            logger.info("[TITO] decide failed err=%s", type(exc).__name__)
            return None

    @staticmethod
    def _tito_decide_last_user_text(payload: dict) -> str:
        """Plain text of the last user message in an outgoing payload, or ""."""
        messages = (payload or {}).get("messages") or []
        for msg in reversed(messages):
            if not isinstance(msg, dict) or msg.get("role") != "user":
                continue
            content = msg.get("content")
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "\n".join(
                    str(b.get("text", ""))
                    for b in content
                    if isinstance(b, dict) and b.get("type") == "text"
                )
            return ""
        return ""

    def _tito_decide_start(self, payload: dict):
        """Start the shadow decide task for this turn. Returns (task, t0) or None.

        Called before the main stream starts so the calls run in parallel with it.
        Never raises.
        """
        import asyncio as _aio

        try:
            if not self._tito_decide_enabled():
                return None
            text = self._tito_decide_last_user_text(payload)
            if not text.strip():
                return None
            return (_aio.ensure_future(self._tito_decide_shadow(text)), time.monotonic())
        except Exception as exc:
            logger.info("[TITO] decide start failed err=%s", type(exc).__name__)
            return None

    async def _tito_decide_collect(self, state) -> None:
        """Await the decide task (bounded) and log the single [TITO] decide line.

        Logs scores and sources only, never text or the key. Never raises.
        """
        import asyncio as _aio

        try:
            task, t0 = state
            timeout_s = max(0.1, int(getattr(self.valves, "TITO_DECIDE_TIMEOUT_MS", 1500) or 1500) / 1000)
            try:
                result = await _aio.wait_for(task, timeout_s)
            except _aio.TimeoutError:
                task.cancel()
                ms = int((time.monotonic() - t0) * 1000)
                logger.info("[TITO] decide outcome=timeout ms=%d", ms)
                return
            ms = int((time.monotonic() - t0) * 1000)
            self._tito_decide_log(result, ms)
        except Exception as exc:
            logger.info("[TITO] decide collect failed err=%s", type(exc).__name__)

    def _tito_decide_log(self, result, ms: int) -> None:
        """The one [TITO] decide line (input flags). Scores and sources only. Never raises."""
        if not result:
            logger.info("[TITO] decide outcome=none ms=%d", ms)
            return
        parts = []
        for flag, short in self.TITO_DECIDE_LOG_NAMES:
            item = result.get(flag) or {}
            parts.append(f"{short}={float(item.get('p', 0.0)):.2f}({item.get('src', 'regex')})")
        logger.info("[TITO] decide %s ms=%d", " ".join(parts), ms)

    def _tito_decide_timeout_s(self) -> float:
        return max(0.1, int(getattr(self.valves, "TITO_DECIDE_TIMEOUT_MS", 1500) or 1500) / 1000)

    def _tito_decide_inject_on(self) -> bool:
        """True when TITO_DECIDE_INJECT is on and the decide shadow is enabled."""
        return bool(getattr(self.valves, "TITO_DECIDE_INJECT", False)) and self._tito_decide_enabled()

    async def _tito_decide_inject(self, state) -> str:
        """INJECT mode: await the input decide before the main call; return the block text or "".

        Only model scores (src jev or msd) count: a regex fallback or a timeout gives
        no lines, so the reply proceeds unchanged. Logs flag names only. Never raises.
        """
        import asyncio as _aio

        try:
            task, t0 = state
            try:
                result = await _aio.wait_for(task, self._tito_decide_timeout_s())
            except _aio.TimeoutError:
                task.cancel()
                logger.info("[TITO] decide_inject outcome=timeout")
                return ""
            self._tito_decide_log(result, int((time.monotonic() - t0) * 1000))
            if not result:
                return ""
            lines, names = [], []
            for flag in self.TITO_DECIDE_FLAGS:
                item = result.get(flag) or {}
                if item.get("src") not in ("jev", "msd"):
                    continue
                prob = float(item.get("p", 0.0))
                if flag == "casual_chat":
                    line = (self.TITO_DECIDE_INJECT_LINES[flag] if prob >= 0.5
                            else self.TITO_DECIDE_CASUAL_NO_LINE)
                elif prob >= self.TITO_DECIDE_INJECT_MIN[flag]:
                    line = self.TITO_DECIDE_INJECT_LINES[flag]
                else:
                    continue
                lines.append(line)
                names.append(flag)
            logger.info("[TITO] decide_inject flags=%s", ",".join(names) or "none")
            return "\n\n".join(lines)
        except Exception as exc:
            logger.info("[TITO] decide_inject failed err=%s", type(exc).__name__)
            return ""

    def _tito_decide_out_start(self, request_ctx, metadata):
        """Start the pure_vn check on Tito's visible reply. Returns (task, t0) or None.

        Called right before the output guard, so both see the same pre-edit reply.
        Never raises.
        """
        import asyncio as _aio

        try:
            if not self._tito_decide_enabled() or not self._is_tito(metadata):
                return None
            visible = visible_text(split_visible(request_ctx.text() or ""))
            if not visible.strip():
                return None
            return (_aio.ensure_future(self._tito_decide_out_call(visible)), time.monotonic())
        except Exception as exc:
            logger.info("[TITO] decide_out start failed err=%s", type(exc).__name__)
            return None

    async def _tito_decide_out_call(self, text: str):
        """pure_vn check (Jev, then MSD-1); returns {"p", "src"} or None. Never raises."""
        try:
            key = self._tito_decide_key()
            if not key:
                return None
            return await self._tito_decide_ask(key, text, "pure_vn", self._tito_decide_timeout_s())
        except Exception:
            return None

    async def _tito_decide_out_collect(self, state, guard_trigger: str) -> None:
        """Await the pure_vn check, bounded from its start, and log one decide_out line.

        Logs the score, source and guard trigger only (never text or the key).
        A failure or timeout logs src=none. Never raises.
        """
        import asyncio as _aio

        try:
            task, t0 = state
            remaining = max(0.001, self._tito_decide_timeout_s() - (time.monotonic() - t0))
            got = None
            try:
                got = await _aio.wait_for(task, remaining)
            except _aio.TimeoutError:
                task.cancel()
            except Exception:
                got = None
            ms = int((time.monotonic() - t0) * 1000)
            if got:
                part = f"pure_vn={float(got['p']):.2f}({got['src']})"
            else:
                part = "pure_vn=-(none)"
            logger.info("[TITO] decide_out %s guard_trigger=%s ms=%d", part, guard_trigger, ms)
        except Exception as exc:
            logger.info("[TITO] decide_out collect failed err=%s", type(exc).__name__)

    def _tito_shrink_images(self, payload: dict) -> dict:
        """Return a payload whose oversized base64 images are downscaled.

        Called only for Tito with ``TITO_IMAGE_MAX_PX`` > 0. Blocks are replaced
        in new lists and dicts, so the caller's payload is not mutated. An image
        that cannot be processed is kept as it was; the WARNING names only the
        exception type.
        """
        max_px = int(getattr(self.valves, "TITO_IMAGE_MAX_PX", 0) or 0)
        if max_px <= 0:
            return payload
        try:
            import base64
            import io

            from PIL import Image
        except Exception as exc:
            self._tito_err("image_shrink", exc)
            return payload

        def shrink(block: dict) -> dict:
            source = block.get("source")
            if not isinstance(source, dict) or source.get("type") != "base64":
                return block
            data = source.get("data")
            if not isinstance(data, str):
                return block
            try:
                with Image.open(io.BytesIO(base64.b64decode(data))) as im:
                    width, height = im.size
                    if max(width, height) <= max_px:
                        return block
                    im.load()
                    has_alpha = im.mode in ("RGBA", "LA") or (
                        im.mode == "P" and "transparency" in im.info
                    )
                    scale = max_px / max(width, height)
                    resized = im.resize(
                        (max(1, round(width * scale)), max(1, round(height * scale))),
                        Image.LANCZOS,
                    )
                out = io.BytesIO()
                if has_alpha:
                    resized.save(out, format="PNG", optimize=True)
                    media_type = "image/png"
                else:
                    if resized.mode not in ("RGB", "L"):
                        resized = resized.convert("RGB")
                    resized.save(out, format="JPEG", quality=85)
                    media_type = "image/jpeg"
                new_source = {
                    **source,
                    "media_type": media_type,
                    "data": base64.b64encode(out.getvalue()).decode("ascii"),
                }
                return {**block, "source": new_source}
            except Exception as exc:
                self._tito_err("image_shrink", exc)
                return block

        def walk(blocks: list) -> list:
            result = []
            for block in blocks:
                if isinstance(block, dict):
                    if block.get("type") == "image":
                        block = shrink(block)
                    elif isinstance(block.get("content"), list):
                        block = {**block, "content": walk(block["content"])}
                result.append(block)
            return result

        messages = []
        for message in payload.get("messages") or []:
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                message = {**message, "content": walk(message["content"])}
            messages.append(message)
        return {**payload, "messages": messages}

    @staticmethod
    def _tito_count_images(request_ctx, payload: dict) -> None:
        """Record image block count and decoded bytes of the outgoing request.

        Numbers only. Base64 size is computed from the string length, so no
        image is decoded. URL-sourced images count as blocks with 0 bytes.
        """
        count = 0
        total = 0

        def walk(blocks) -> None:
            nonlocal count, total
            for block in blocks or []:
                if not isinstance(block, dict):
                    continue
                if block.get("type") == "image":
                    count += 1
                    source = block.get("source") or {}
                    data = source.get("data") if isinstance(source, dict) else None
                    if isinstance(data, str):
                        pad = len(data) - len(data.rstrip("="))
                        total += len(data) * 3 // 4 - pad
                elif isinstance(block.get("content"), list):
                    walk(block["content"])

        for message in (payload or {}).get("messages") or []:
            if isinstance(message, dict) and isinstance(message.get("content"), list):
                walk(message["content"])
        request_ctx.timing["images"] = count
        request_ctx.timing["image_bytes"] = total

    def _tito_log_timing(self, request_ctx, usage: dict) -> None:
        """One INFO line per Tito turn: timings and token counts, no content."""
        timing = request_ctx.timing
        t0 = timing.get("t0")
        if t0 is None:
            return
        now = time.monotonic()

        def ms(key: str, end: float | None = None) -> int:
            value = timing.get(key)
            return 0 if value is None else round(((end if end is not None else value) - t0) * 1000)

        logger.info(
            "[TITO] timing ttfb_ms=%d first_text_ms=%d stream_ms=%d guard_ms=%d total_ms=%d "
            "in=%d out=%d cache_read=%d cache_write=%d images=%d image_bytes=%d",
            ms("ttfb"),
            ms("first_text"),
            round(timing.get("stream", 0.0) * 1000),
            round(timing.get("guard", 0.0) * 1000),
            ms("t0", now),
            usage.get("input_tokens", 0),
            usage.get("output_tokens", 0),
            usage.get("cache_read_input_tokens", 0),
            usage.get("cache_creation_input_tokens", 0),
            timing.get("images", 0),
            timing.get("image_bytes", 0),
        )

    @staticmethod
    def _tito_usage_stream(text: str, model: str, usage: dict):
        """Stream a finished Tito reply so OpenWebUI's middleware sees its usage.

        The middleware persists ``message.usage`` from the ``usage`` key of the
        SSE chunks it reads from the pipe (``utils/middleware.py``, the chunk loop
        that merges ``data['usage']`` into ``current_usage``). A plain ``str``
        return carries no usage, so the middleware's final save wrote
        ``{"incomplete": true}`` over the direct DB write made by the pipe. This
        yields the same content chunk a ``str`` return would, then one empty-
        choices chunk carrying the final usage; OpenWebUI appends its own finish
        chunk after it.
        """

        async def _chunks():
            if text:
                yield {
                    "object": "chat.completion.chunk",
                    "model": model,
                    "choices": [{"index": 0, "delta": {"content": text}, "finish_reason": None}],
                }
            yield {"object": "chat.completion.chunk", "model": model, "choices": [], "usage": usage}

        return _chunks()
