# KEY — answers, expected values, open questions

## Exact strings / values (do not invent)

- Model id: `tito`. Base model: `claude-sonnet-5-5`.
- TITO_CONFIG effort: `"medium"`. thinking: `{"type":"adaptive"}`.
- Advisor: model `claude-opus-5-5`, `max_uses: 3`, `caching: "5m"` → wire `{"type":"ephemeral","ttl":"5m"}`.
- `hide_blocks`: `["advisor", "web_search", "web_fetch"]`.
- `web_search_max_uses`: `10`. `web_fetch_max_uses`: `5`.
- `user_location`: `{"city": "Fresno"}` (city only; country/region/timezone empty).
- Cache TTL: `"1h"` tools_system AND `"1h"` messages.
- `show_cost`: `True`. Token display: OFF (SHOW_TOKEN_COUNT stays "Off" for the user).
- HARD CAP: `TITO_HARD_CAP_USD = 25`. SOFT CAP: `TITO_SOFT_CAP_USD = 15`.
- HARD_CAP_TEXT (exact, byte-for-byte):
  `Tito đang nghỉ tới ngày 1 tháng sau. Dori đã biết rồi, ông không cần làm gì.`
- Missing-tool result text (placeholder — confirm with Tony; "lịch tạm thời không dùng được" style):
  `Lịch tạm thời không dùng được, ông thử lại lát nữa nhé.`
- Beta headers to add for tito: `mid-conversation-system-clear-at-2026-08-21` (PATCH D),
  `advisor-tool-2026-03-01` (advisor, already emitted when ENABLE_ADVISOR_TOOL).
- Env vars: `TITO_TOOLS_FILE` (path to frozen tools JSON), `TITO_DISCORD_WEBHOOK` (URL),
  `TITO_HARD_CAP_USD` (default 25), `TITO_SOFT_CAP_USD` (default 15).

## Open questions (confirm before build — all have a working default chosen in SPEC)

1. **D1/D2 tito→base mapping.** No mapping exists in the pipe today; `body["model"]="tito"` reaches
   Anthropic as `"tito"`. SPEC chooses `.split("/")[-1] == "tito"` + remap to `claude-sonnet-5-5`.
   If OWUI actually sends the base model id in `body["model"]` and exposes `"tito"` only via
   `__metadata__["model_id"]`, the gate must key on `__metadata__["model_id"]` instead.
2. **D4 marker key.** Pipe contract pinned as `msg["meta"]["mid_conversation"]` truthy. Confirm OWUI
   PATCH C (out of repo) writes exactly this key.
3. **Missing-tool Vietnamese text** — placeholder above; get the final wording from Tony.
4. **`self.id = "anthropic"` vs docstring `id: anthropic_new`** — pre-existing inconsistency; irrelevant
   to these patches but worth a separate cleanup ticket.

## Verification results (doc file:line → real code)

Full table in SPEC §9. Headline: every cited line is either exact or drifted ≤ a few lines; the
`mw:` / `utils/tools.py:` cites point into the OWUI repo (not in this workdir) and are marked EXT.
No doc cite is behaviorally wrong. Real locations used throughout the SPEC.

## Test-runner decision (conflicting instruction)

The brief says "Unit tests: pytest". This repo has **no** pytest usage: the four existing tests are
standalone scripts in `helpers/test_*.py` (`async def test_*(module)` + `main()`), run via
`python helpers/test_X.py` (that is also how AGENTS.md documents them). Resolution: the red test
follows the repo's real convention (standalone, `python tito-spec/tests/test_tito.py`, exit 0/1) —
this is the rule that *runs* today; pytest would not collect these async tests without an added
`pytest-asyncio` config that does not exist here. Noted, not blocked.

## Notes the coder must respect

- `anthropic` SDK and `open_webui` are NOT installed in this environment; `helpers/test_model_cache.py`
  fails here on `from anthropic import AsyncAnthropic`. Keep `tito.py` free of those imports at module level.
- No `httpx/requests/aiohttp` import exists in `src/` today; the Discord alert needs a lazy import inside
  the method (use `httpx` — a transitive dep of `anthropic` — or `urllib`).
- `_public_usage` keeps every non-underscore key (runtime.py:156), so `cost_usd`/`cost_breakdown_usd`
  already persist to `message.usage`. The budget gate reads the sum of `usage.cost_usd`.
- `cost_usd` is only computed when `include_usage` is true (pipe_orchestrator.py:243-246) — the tito
  include_usage force (AC10) is a hard prerequisite of the budget gate, not optional.
