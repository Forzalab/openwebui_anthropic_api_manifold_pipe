# SPEC — Tito pipe patches (B, D, B4 tool-freeze, budget gate)

Scope: the **pipe fork** (this repo, `src/anthropic_pipe/`). The four patches below all live
in the pipe, gated on model id `"tito"`; every other model path is byte-for-byte unchanged.
The OWUI-fork patches (A/C/E/G from `tito-selfhost-PLAN-LOCKED.md`) are a *different repo* and
out of scope here — but PATCH D defines the wire contract PATCH C (OWUI) must produce, so that
contract is pinned below.

Authoritative source docs (wins, in order): `tito-selfhost-PLAN-LOCKED.md`,
`tito-plan-vet-fable.md`, `tito-system-prompt-audit.md`. Each `file:line` cited in those docs was
checked against this repo's real code; the reconciliation table is at the bottom. Where a doc's
line drifted or pointed into the OWUI repo (not present here), the real location is used and noted.

---

## 0. Non-goals / explicitly out of scope

- No OWUI-fork changes (PATCH A/C/E/G, `apply_tito_lock`, `merge_system_messages`, Conduit).
- No keepalive, no "still here" gesture, no mid-chat Haiku swap, no rollover/compaction.
- No change to non-`tito` requests. Every branch below is `if _is_tito(...):` guarded.
- No pricing-table changes for advisor iterations / Sonnet-5.5 cache-read rate (vet H6.2/H6.3 are
  *accuracy* notes on `cost_usd`; the budget gate consumes whatever `cost_usd` records. Accepted risk.)

## 1. Decisions (proposed; confirm before build)

D1 — **"tito" is the last path segment of the requested model id.** `is_tito = model_name.split("/")[-1] == "tito"`.
    Robust whether OWUI sends `tito`, `anthropic/tito` or `anthropic_new/tito`. (There is currently
    **no** tito→base mapping anywhere in the pipe; `body["model"]="tito"` today reaches Anthropic as
    model `"tito"` and 400s.)

D2 — **The pipe remaps tito → `TITO_CONFIG["base_model"] = "claude-sonnet-5-5"`** for the API call and
    for `get_model_info`, *before* any capability/effort/thinking logic. `get_model_info("tito")`
    currently returns the generic defaults (no effort, no adaptive thinking); capabilities must come
    from the base model.

D3 — **`TITO_CONFIG` is a single frozen dict, not UserValves.** For `tito` the pipe ignores
    `__user__["valves"]` *and* `pipe.valves` for every key the dict covers. Keys not in the dict fall
    through to existing valve behavior (minimizes churn).

D4 — **Mid-conversation status marker = `msg["meta"]["mid_conversation"]` truthy** (set by OWUI PATCH C,
    out of scope here). The pipe-side contract is: a `role:"system"` message whose
    `meta.mid_conversation` is truthy is a *status* message, passed through as a top-level
    `role:"system"` message in `messages[]` (never folded into `system`), with
    `clear_at:"next_user_message"`.

D5 — **New module `src/anthropic_pipe/tito.py`**, a self-contained method-group mixin
    (`class PipeTitoMethods`) with no `open_webui`/`anthropic` imports at module level (so it loads
    standalone for tests). Added to `METHOD_GROUP_SOURCES` in `helpers/build_anthropic_pipe.py`.

---

## 2. PATCH B — fixed config for "tito" (vet B1 + PLAN "PATCH B")

New module constant:

```python
TITO_CONFIG = {
    "base_model": "claude-sonnet-5-5",
    "thinking": {"type": "adaptive"},          # fixed; display key never added for tito
    "effort": "medium",                        # fixed; body.reasoning_effort is ignored
    "advisor": {"enabled": True, "model": "claude-opus-5-5", "max_uses": 3, "caching": "5m"},
    "enable_tool_search": False,               # no defer_loading, no tool_search_tool_*
    "hide_blocks": ["advisor", "web_search", "web_fetch"],
    "web_search_max_uses": 10,
    "web_fetch_max_uses": 5,                   # unchanged from default; pinned for stability
    "user_location": {"city": "Fresno"},       # only non-empty fields are sent (tools.py:214-227)
    "cache": {"tools_system_ttl": "1h", "messages_ttl": "1h"},
    "show_cost": True,
    "strip_reasoning_effort": True,
    "skip_tool_search_nudge": True,            # payload.py:704-731
    "skip_terminal_hint": True,                # payload.py:737-750
    "always_strip_tool_manifest": True,        # pipe_orchestrator.py:215-222
}
```

### 2.1 Model remap + effort + thinking (payload.py)
- In `create_request_payload` (payload.py), immediately after `actual_model_name` is computed
  (payload.py:30), add: if `pipe._is_tito(actual_model_name)` → `actual_model_name = TITO_CONFIG["base_model"]`.
- Effort (payload.py:78-84): when tito, skip `body.get("reasoning_effort")` entirely and set
  `effective_effort = "medium"`; `output_config = {"effort": "medium"}`.
- Thinking (payload.py:88-143): when tito, force `enable_thinking = True` and emit
  `thinking = {"type": "adaptive"}` (sonnet-5-5 supports adaptive thinking; no budget_tokens, no display key).
- No terminal hint, no tool-search nudge for tito (guard both blocks).

### 2.2 Tools (tools.py)
- `_convert_tools_to_claude_format` (tools.py): when tito —
  - advisor tool built from `TITO_CONFIG["advisor"]` regardless of `ENABLE_ADVISOR_TOOL`;
    `{"type":"advisor_20260301","name":"advisor","model":"claude-opus-5-5","max_uses":3,"caching":{"type":"ephemeral","ttl":"5m"}}`.
    **Do not** run the executor↔advisor downgrade guard for tito (the pair Sonnet-5-5→Opus-5-5 is valid; vet tu:437).
  - `enable_tool_search=False` → no `defer_loading`, no `tool_search_tool_*` (tools.py:408-465).
  - web_search `max_uses = 10`, `user_location = {"city":"Fresno"}` (only non-empty fields).
  - no `ENABLE_DYNAMIC_FILTERING` → `web_search_20250305` (per vet M1; NOT `_20260209`).
- Beta header `advisor-tool-2026-03-01` still added (payload.py:465) when advisor enabled.

### 2.3 Cache TTL (cache_control.py)
- `_cache_control_marker` (cache_control.py:13-41): when tito, return `{"type":"ephemeral","ttl":"1h"}`
  for **both** scopes (tools_system and messages). This overrides `CACHE_TTL`/`CACHE_TTL_FOR_TOOLS_AND_SYSTEM_PROMT`.
- `CACHE_CONTROL` for tito is forced to `"cache tools array, system prompt and messages"`
  (already the default at pipe_template.py:1062 — keep it, just don't let a valve downgrade tito).

### 2.4 OWUI tool-manifest strip (pipe_orchestrator.py)
- pipe_orchestrator.py:215-222 currently strips the manifest only when `deferred_tools` present AND
  native tool search. For tito, strip `OWUI_TOOL_MANIFEST_RE` from every `role:"system"` string message
  unconditionally.

### 2.5 Per-turn fingerprint + change alert (NEW)
- After the payload is built (and after `_apply_cache_control`), for tito log exactly one line:
  `[TITO] sha256_tools=<hex> sha256_system=<hex> model=<base> thinking=<type> effort=<effort> betas=<csv>`.
  `sha256_tools`/`sha256_system` over the *canonicalized, wire-order* serialization (mirror
  `_canonicalize_block`), empty→empty-string hash.
- Keep a per-`chat_id` record of the previous tuple. If the tuple changes within the same `chat_id`,
  log at `WARNING` (prefix drift) — else `INFO`. Reuse the existing `self._cache_diff_state` keying
  pattern or a dedicated `self._tito_fingerprint_state: dict[str, tuple]`.

## 3. PATCH D — mid-conversation system messages (vet H1/H3 + PLAN "PATCH D")

Current: `_convert_messages_to_claude_format` (messages.py:81-106) folds **every** `role:"system"`
message into the top-level `system` array. Change (tito only):

- A `role:"system"` message with `meta.mid_conversation` truthy is **not** folded into `system`;
  instead it is appended to `processed_messages` as `{"role":"system","content":[<text blocks>],
  "clear_at":"next_user_message"}` at its replay position (after the user message that produced it).
- Orphan status: a mid-conversation system message is **dropped** when the *next* replayed message
  is not an assistant turn and it is not the last message (vet H1: replaying `user, system, user, ...`
  is a permanent 400). Implemented as a post-pass over `processed_messages`.
- Beta header `mid-conversation-system-clear-at-2026-08-21` is added (payload.py beta assembly) when
  any mid-conversation system message is present. (Live-verified on sonnet-5-5 + haiku-5-5 per BUILD LOG.)
- Never put `cache_control` on a mid-conversation system message (see §4).
- Non-tito behavior unchanged: all `role:"system"` messages still fold into `system`.

## 4. PATCH D cache breakpoint (vet H3)

- `_apply_cache_control` step 4 / `_cache_last_stable_message` (cache_control.py:663-695): when the
  newest message is a `role:"system"` mid-conversation message, skip it and place the messages
  breakpoint on the preceding non-system cacheable block (the user message). Same rule in the tool-loop
  branch. `_place_cache_on_last_cacheable_block` already skips thinking/redacted; extend the callers to
  also skip `role:"system"` messages.

## 5. B4 — frozen tools array + missing-tool handling

- **Deploy-time dump:** a helper writes the final, canonical `tools` array to a JSON file whose path is
  env `TITO_TOOLS_FILE`. (Dump helper = small standalone script in `helpers/`, run at deploy, committed
  or produced in the Space image; the pipe only ever *reads* it.)
- **Request-time (tito):** `_convert_tools_to_claude_format` returns the frozen array loaded from
  `TITO_TOOLS_FILE` (byte-stable across outages and upstream description edits) instead of live
  `__tools__`/`body.tools`. Tools stay byte-identical; order fixed.
- **Missing MCP tool at call time:** when a tool_use names a tool not present among the live callables,
  return a `tool_result` with `is_error: True` and Vietnamese text in the "lịch tạm thời không dùng được"
  family (exact string pinned as a constant, see KEY.md), and fire a Discord alert
  (env `TITO_DISCORD_WEBHOOK`). **Never** remove the tool from `tools[]` (the prefix must not change).
- Frozen tools are *always* sent even when no live socket/tool-server is reachable (vet B2/B4).

## 6. BUDGET GATE (PLAN "DECISIONS 2026-11-09" + vet H5/H6/H7)

- Month = calendar month in `America/Los_Angeles` (Pacific). Month key = `"YYYY-MM"`.
- Spend source = `cost_usd` persisted by SHOW_COST → `total_usage["cost_usd"]`
  (pipe_orchestrator.py:1266) → `_public_usage` (runtime.py:131, keeps non-underscore keys) →
  `Chats.upsert_message_to_chat_by_id_and_message_id(... {"usage": ...})` (pipe_orchestrator.py:1483).
  `cost_usd` survives into `message.usage.cost_usd` in the DB.
- **Prerequisite fix (vet H6.1):** for tito, force `include_usage = True` (pipe_orchestrator.py:243-246)
  so `cost_usd` is always computed and persisted, while token display stays off (`SHOW_TOKEN_COUNT` not
  shown to Dad). Without this the gate reads $0 forever.
- Pure decision function (testable, no DB):
  `_tito_budget_decision(month_spend_usd: float) -> tuple[str, str|None]`:
  - `month_spend_usd >= TITO_HARD_CAP_USD` (25) → `("hard", HARD_CAP_TEXT)`.
  - `TITO_SOFT_CAP_USD` (15) `<= month_spend_usd < 25` → `("soft", None)` (fires one Discord warning per month).
  - else → `("ok", None)`.
- `HARD_CAP_TEXT` (exact): `"Tito đang nghỉ tới ngày 1 tháng sau. Dori đã biết rồi, ông không cần làm gì."`
- In `pipe()`, **before** any client build or API call (right after the task-model branch and auth),
  for tito: read `month_spend_usd` (DB read, cached a few minutes — vet M11), call
  `_tito_budget_decision`. On `"hard"`: emit the text as a normal assistant message and return —
  **no Anthropic API call**. On `"soft"`: if no warning sent for this month key, send one Discord alert.
- Soft-warning dedupe key = month key; state dict `self._tito_soft_warned: set[str]` (in-memory; a Space
  restart may re-warn once — accepted).

## 7. Error mapping (vet H7) — in scope for budget gate only

The four patches do **not** include the full "map every pipe error to Vietnamese + Discord" change
(vet H7). Only the budget-gate hard-cap text and the B4 missing-tool text are in scope here.
H7 is listed in FLASH-BRIEF as a follow-up, not implemented now.

## 8. Build / regeneration

- Add `SRC_DIR / "tito.py"` to `METHOD_GROUP_SOURCES` in `helpers/build_anthropic_pipe.py`.
- `python helpers/build_anthropic_pipe.py` (writes `anthropic_pipe.py` + `anthropic_pipe.min.py`).
- Bump `version:` + changelog in `src/anthropic_pipe/pipe_template.py`; mirror to `README.md`.
- Commit `src/` **and** both regenerated artifacts (AGENTS.md release checklist).

## 9. File:line reconciliation (doc vs real code)

Legend: ✓ exact/near-exact · ≈ drifted a few lines (real loc shown) · EXT = points into OWUI repo (not here).

| Doc cite | Claim | Real location | Verdict |
|---|---|---|---|
| vet B1 `pipe_template.py:1186-1330` | UserValves block | 1180-1361 | ≈ (block 1180-1361) |
| vet B1 `ENABLE_THINKING default False` | — | 1186-1189 | ✓ |
| vet B1 `EFFORT default high` | — | 1200-1203 | ✓ |
| vet B1 `ENABLE_ADVISOR_TOOL default False` | — | 1314-1317 | ✓ |
| vet B1 `WEB_SEARCH_MAX_USES (5)` | — | 1257-1262 | ✓ |
| vet B1 `ENABLE_TOOL_SEARCH default True` | — | 1289-1292 | ✓ |
| vet B1 `payload.py:83,88,461` | EFFORT / ENABLE_THINKING / advisor-beta | 82, 88, 465 | ≈ |
| vet B1 `tools.py:270-304` | advisor valve reads | 264-322 | ≈ |
| vet B1 `payload.py:77-81 reasoning_effort beats valve` | — | 78-82 | ✓ |
| vet B1 `tools.py:429-437 deferral` | — | 410-434 | ≈ |
| vet B1 `payload.py:711-735 nudge` | — | 704-731 | ≈ |
| vet B1 `pipe_orchestrator.py:1139-1145 drop_block skip between_tools` | — | 1137-1145 | ✓ |
| vet B1 `pipe_template.py:908-910 between_tools` | sonnet-5-5 off_type | 907-910 | ✓ |
| vet B2/B3/B4/B5 `mw:…` / `utils/tools.py:808` | OWUI internals | — | EXT |
| vet B4 `tools.py:383-399 tool copy` | — | 393-400 | ≈ |
| audit `messages.py:81-106 system→blocks` | — | 81-106 | ✓ |
| audit `payload.py:711-735` / `740-751` | nudge / terminal hint | 704-731 / 737-750 | ≈ |
| audit `pipe_orchestrator.py:215-222 manifest strip` | — | 215-222 | ✓ |
| vet H3 `cache_control.py:667-676 last=messages[-1]` | — | 663-695 (last at 671) | ≈ |
| vet H3 `cache_control.py:563-566` | in-turn breakpoint budget | 562-565 | ✓ |
| vet H3 `cache_control.py:654-659 volatile detect` | — | 654-660 | ≈ |
| vet H6 `pipe_orchestrator.py:243-246 include_usage` | — | 243-246 | ✓ |
| vet H6 `pipe_orchestrator.py:1264-1267 cost` | — | 1263-1267 | ≈ |
| vet H6 `pricing.py:43 sonnet-5-5 cache_read 0.10` | — | 43 | ✓ |
| audit `cache_control on last system block` | — | cache_control.py:508-516 | ✓ |
| vet M1 `tools.py:196-208 web_search version` | — | 196-212 | ≈ |
| vet M1 `tools.py:175-190 user_location` | — | 175-227 | ≈ |
| vet H5 `pipe_orchestrator.py:332-335 hash diff` | — | 332-335 | ✓ |
| vet H7 `pipe_orchestrator.py:1087-1115` / `1177-1181` | retry/error text | 1086-1116 / 1166-1182 | ≈ |
| vet `payload.py:32-34 max_tokens` | — | 32-34 | ✓ |
| vet `payload.py:546-570 cache diag` | — | 540-563 | ≈ |

Only material mismatches are line drift (a few lines) and OWUI-repo cites; no cited file:line is
*factually* wrong about behavior. All specs above use the real locations.
