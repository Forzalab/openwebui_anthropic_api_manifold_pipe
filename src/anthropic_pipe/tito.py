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
        "enable_tool_search": False,
        "hide_blocks": ["advisor", "web_search", "web_fetch"],
        "web_search_max_uses": 10,
        "web_fetch_max_uses": 5,
        "user_location": {"city": "Fresno"},
        "cache": {"ttl": "1h", "tools_breakpoint": False, "system_breakpoints": 2},
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

    async def _tito_output_guard(self, request_ctx, client, metadata) -> None:
        """Check Tito's visible reply with Haiku and replace it once if needed.

        Only visible prose is sent; ``<details>`` blocks, ``[anthropic-hidden-]``
        carriers and ``[](anthropic:)`` markers are re-inserted byte-for-byte.
        Never raises: any failure leaves the streamed reply exactly as it is.

        Logs exactly one INFO line per Tito run (lengths only, never reply text):
        ``outcome`` is clean | kept_ok | kept_parse | kept_safety | kept_markers |
        replaced | error. On error a ``TITO-ERR`` line with a support id follows.
        """
        if not getattr(self.valves, "TITO_OUTPUT_GUARD", True):
            return
        if not self._is_tito(metadata):
            return
        timing = getattr(request_ctx, "timing", None)
        if timing is not None:
            timing["guard_t0"] = time.monotonic()

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
                verdict, edited = await guard_verdict(client, request_text)
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
            from datetime import datetime as _dt, timedelta as _td, timezone as _tz

            now = _dt.now(_tz.utc)
            local = now + _td(hours=self._tito_pacific_utc_offset(now))
            err_id = f"T{local:%H%M}-{os.urandom(2).hex()}"
            logger.error(
                "TITO-ERR %s output_guard %s", err_id, self._tito_safe_err(exc, limit=200)
            )
        logger.info(
            "[TITO] output guard outcome=%s trigger=%s chars=%d visible=%d rebuilt=%d",
            outcome,
            trigger,
            len(content),
            len(visible),
            rebuilt_len,
        )

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
        raw = _re.sub(r"sk-[A-Za-z0-9_\-]{8,}", "[redacted]", raw)
        raw = _re.sub(
            r"(?i)(x-api-key|authorization|api[_-]?key|bearer)(['\"]?\s*[:=]?\s*)(?:Bearer\s+)?[^\s,'\"}]+",
            r"\1\2[redacted]",
            raw,
        )
        return raw[:limit]

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
                problem = f"TITO_TOOLS_FILE {path} unreadable: {cls._tito_safe_err(e)}"
        if problem:
            logger.error("[TITO] frozen tools unavailable: %s", problem)
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

    @staticmethod
    def _tito_discord_alert(message: str) -> None:
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
                logger.debug("Tito Discord alert failed: %s", e)

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
            logger.warning("Tito month_spend_usd DB error (fail-open): %s", e)
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

        def ms(key: str, end: float = None) -> int:
            value = timing.get(key)
            return 0 if value is None else int(round(((end if end is not None else value) - t0) * 1000))

        logger.info(
            "[TITO] timing ttfb_ms=%d first_text_ms=%d stream_ms=%d guard_ms=%d total_ms=%d "
            "in=%d out=%d cache_read=%d cache_write=%d images=%d image_bytes=%d",
            ms("ttfb"),
            ms("first_text"),
            int(round(timing.get("stream", 0.0) * 1000)),
            int(round(timing.get("guard", 0.0) * 1000)),
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
