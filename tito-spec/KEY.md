# KEY — answers, expected values, open questions

## Exact strings / values (do not invent)

- Model id: `tito` (OWUI model id; gate = `__metadata__["model"]["id"] == "tito"`). Base model in
  `body["model"]` is already `claude-sonnet-5-5` (OWUI rewrites it — no pipe remap).
- Status marker (C1): TOP-LEVEL key `"mid_conversation": true` on the system message dict.
- TITO_CONFIG effort: `"medium"`. thinking: `{"type":"adaptive"}`.
- Advisor: model `claude-opus-5-5`, `max_uses: 3`, `caching: "5m"` → wire `{"type":"ephemeral","ttl":"5m"}`.
- `hide_blocks`: `["advisor", "web_search", "web_fetch"]`.
- `web_search_max_uses`: `10`. `web_fetch_max_uses`: `5`.
- `user_location`: `{"city": "Fresno"}` (city only; region/country/timezone empty).
- Cache (P6): ttl `"1h"` everywhere; `tools_breakpoint: False`; `system_breakpoints: 2`.
- `show_cost`: `True`. Token display OFF (SHOW_TOKEN_COUNT stays "Off" for the user).
- Caps (P3): `TITO_HARD_CAP_USD` default **22** (cost_usd undercounts advisor + cache; console cap 25
  is the backstop). `TITO_SOFT_CAP_USD` default **15**.
- `HARD_CAP_TEXT` (exact, byte-for-byte):
  `Tito đang nghỉ tới ngày 1 tháng sau. Dori đã biết rồi, ông không cần làm gì.`
- `TITO_ERROR_TEXT` (P5, exact):
  `Tito đang bận một chút, ông thử lại sau vài phút nhé.`
- Missing-tool result text (placeholder — confirm with Tony; "lịch tạm thời không dùng được" style):
  `Lịch tạm thời không dùng được, ông thử lại lát nữa nhé.`
- Beta headers to add for tito: `mid-conversation-system-clear-at-2026-08-21` (PATCH D),
  `advisor-tool-2026-03-01` (advisor, already emitted when ENABLE_ADVISOR_TOOL).
- Env vars: `TITO_TOOLS_FILE` (path to frozen tools JSON), `TITO_DISCORD_WEBHOOK` (URL),
  `TITO_HARD_CAP_USD` (default 22), `TITO_SOFT_CAP_USD` (default 15).

## Where SHOW_COST stores cost_usd (P2) — found

`total_usage["cost_usd"]` set at `pipe_orchestrator.py:1266`; projected by `_public_usage`
(`runtime.py:131`, keeps all non-underscore keys) and persisted at `pipe_orchestrator.py:1483` via
`Chats.upsert_message_to_chat_by_id_and_message_id(chat_id, message_id, {"usage": _public_usage(total_usage)})`.
So `message.usage.cost_usd` in the OWUI `chat_message` table is the source. `month_spend_usd(now)` sums
that column over all users' chats, bucketed by Pacific month (`America/Los_Angeles`). The exact ORM/query
against `open_webui.models.chats` is **VERIFY-AT-RUNTIME** (OWUI not in this repo).

## Open questions (confirm before build — each has a working default in SPEC)

1. **C2 gate (VERIFY-AT-RUNTIME).** `__metadata__["model"]["id"] == "tito"` is assumed populated on every
   Tito request. Confirm `main.py:1343` actually puts the OWUI model id there before build.
2. **Missing-tool Vietnamese text** — placeholder above; get final wording from Tony.
3. **`self.id = "anthropic"` vs docstring `id: anthropic_new`** — pre-existing inconsistency; irrelevant
   here but worth a separate cleanup ticket.

## Verification results (doc file:line → real code)

Full table in SPEC §9. Headline: every cited line is exact or drifted ≤ a few lines; the `mw:` /
`utils/tools.py:` / `functions.py:` / `main.py:` cites point into the OWUI repo (not in this workdir)
and are marked EXT. No doc cite is behaviorally wrong. Real locations used throughout.

## Test-runner decision (conflicting instruction)

The brief says "Unit tests: pytest". This repo has **no** pytest usage: the four existing tests are
standalone scripts in `helpers/test_*.py` (`async def test_*(module)` + `main()`), run via
`python helpers/test_X.py` (that is how AGENTS.md documents them). Resolution: the red test follows the
repo's real convention (`python tito-spec/tests/test_tito.py`, exit 0/1) — this is the rule that *runs*
today; pytest would not collect these async tests without a `pytest-asyncio` config that does not exist
here. Noted, not blocked.

## Notes the coder must respect

- `anthropic` SDK and `open_webui` are NOT installed in this environment; `helpers/test_model_cache.py`
  fails here on `from anthropic import AsyncAnthropic`. Keep `tito.py` free of those imports at module level.
- No `httpx/requests/aiohttp` import exists in `src/` today; the Discord alert needs a lazy import inside
  the method (use `httpx` — a transitive dep of `anthropic` — or `urllib`).
- `cost_usd` is only computed when `include_usage` is true (pipe_orchestrator.py:243-246) — the tito
  include_usage force (AC10) is a hard prerequisite of the budget gate, not optional.
