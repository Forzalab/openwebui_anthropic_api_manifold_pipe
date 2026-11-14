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

Reviewer fixes (orc decisions C1–C3, P1–P6) are folded in here; they supersede the prior wording.

---

## 0. Non-goals / explicitly out of scope

- No OWUI-fork changes (PATCH A/C/E/G, `apply_tito_lock`, `merge_system_messages`, Conduit).
- No keepalive, no "still here" gesture, no mid-chat Haiku swap, no rollover/compaction.
- No change to non-`tito` requests. Every branch below is `if _is_tito(...):` guarded.
- No pricing-table changes for advisor iterations / Sonnet-5.5 cache-read rate (vet H6.2/H6.3 are
  *accuracy* notes on `cost_usd`; the budget gate consumes whatever `cost_usd` records. Accepted risk.)

## 1. Decisions (final, from orc)

D1 — **Gate = `__metadata__["model"]["id"] == "tito"`.** OWUI `functions.py:265-266` rewrites the
    body model to the base model id, so the pipe **never** sees `"tito"` in `body["model"]`.
    There is **no remap** in the pipe: `body["model"]` is already `claude-sonnet-5-5`.
    **VERIFY-AT-RUNTIME** (the `functions.py:265-266` / `main.py:1343` cites are OWUI-repo, not in
    this workdir): confirm before build that `__metadata__["model"]["id"] == "tito"` is populated on
    every Tito request. `_is_tito(metadata) = (metadata or {}).get("model", {}).get("id") == "tito"`.

D2 — **Status marker = TOP-LEVEL key on the message dict**, shared by pipe and OWUI:
    `{"role":"system","content":[…],"mid_conversation":true}`. Not `meta.mid_conversation`.

D3 — **`TITO_CONFIG` is a single frozen dict, not UserValves.** For `tito` the pipe ignores
    `__user__["valves"]` *and* `pipe.valves` for every key the dict covers. Keys not in the dict fall
    through to existing valve behavior.

D4 — **`TITO_ACTIVE` ContextVar** (set once in `pipe()` from `_is_tito(__metadata__)`), mirroring the
    existing `SLIM_OUTPUT`/`HIDDEN_BLOCKS` pattern. Deep methods without metadata
    (`_convert_messages_to_claude_format`, `_cache_control_marker`, `_apply_cache_control`,
    `_cache_last_stable_message`) read `TITO_ACTIVE.get()`.

D5 — **New module `src/anthropic_pipe/tito.py`**, a self-contained method-group mixin
    (`class PipeTitoMethods`) with `contextvars` imported at its top (pipe_template already imports it;
    the standalone file needs its own import for tests). Added to `METHOD_GROUP_SOURCES` in
    `helpers/build_anthropic_pipe.py`.

---

## 2. PATCH B — fixed config for "tito" (vet B1 + PLAN "PATCH B")

```python
TITO_CONFIG = {
    "thinking": {"type": "adaptive"},          # fixed; no display key, no budget_tokens
    "effort": "medium",                        # fixed; body.reasoning_effort is ignored
    "advisor": {"enabled": True, "model": "claude-opus-5-5", "max_uses": 3, "caching": "5m"},
    "enable_tool_search": False,               # no defer_loading, no tool_search_tool_*
    "hide_blocks": ["advisor", "web_search", "web_fetch"],
    "web_search_max_uses": 10,
    "web_fetch_max_uses": 5,                   # unchanged from default; pinned for stability
    "user_location": {"city": "Fresno"},       # only non-empty fields sent (tools.py:214-227)
    "cache": {"ttl": "1h", "tools_breakpoint": False, "system_breakpoints": 2},  # see P6
    "show_cost": True,
    "strip_reasoning_effort": True,
    "skip_tool_search_nudge": True,            # payload.py:704-731
    "skip_terminal_hint": True,                # payload.py:737-750
    "always_strip_tool_manifest": True,        # pipe_orchestrator.py:215-222
}
```

### 2.1 Effort + thinking (payload.py)
- Effort (payload.py:78-84): when tito, skip `body.get("reasoning_effort")` entirely and set
  `effective_effort = "medium"`; `output_config = {"effort": "medium"}`.
- Thinking (payload.py:88-143): when tito, force `enable_thinking = True` and emit
  `thinking = {"type": "adaptive"}` (sonnet-5-5 supports adaptive thinking).
- No terminal hint, no tool-search nudge for tito (guard both blocks).

### 2.2 Tools (tools.py)
- `_convert_tools_to_claude_format` (tools.py): when tito —
  - advisor tool built from `TITO_CONFIG["advisor"]` regardless of `ENABLE_ADVISOR_TOOL`;
    `{"type":"advisor_20260301","name":"advisor","model":"claude-opus-5-5","max_uses":3,"caching":{"type":"ephemeral","ttl":"5m"}}`.
    **Do not** run the executor↔advisor downgrade guard for tito (pair Sonnet-5-5→Opus-5-5 is valid).
  - `enable_tool_search=False` → no `defer_loading`, no `tool_search_tool_*` (tools.py:408-465).
  - web_search `max_uses = 10`, `user_location = {"city":"Fresno"}` (only non-empty fields).
  - no `ENABLE_DYNAMIC_FILTERING` → `web_search_20250305` (vet M1; NOT `_20260209`).
- Beta header `advisor-tool-2026-03-01` still added (payload.py:465) when advisor enabled.

### 2.3 Cache TTL + breakpoints (cache_control.py) — P6
- For tito, every cache marker is `{"type":"ephemeral","ttl":"1h"}` (tools_system **and** messages).
- **No separate tools breakpoint** (vet M4): skip `_apply_cache_control` step 2 (tools) for tito.
- **Two system breakpoints**: system arrives as 2 text blocks (block1 = raw prompt, block2 = memory
  snapshot, produced by OWUI PATCH A) → place `cache_control` after block1 **and** after block2.
- **Last-message breakpoint** at 1h; never on a `mid_conversation` message (see §4).
- Max 4 `cache_control` markers total on any tito request (Anthropic hard limit); tito uses ≤3.
- `CACHE_CONTROL` for tito is forced to `"cache tools array, system prompt and messages"`
  (already the default, pipe_template.py:1062 — just don't let a valve downgrade tito).

### 2.4 OWUI tool-manifest strip (pipe_orchestrator.py)
- pipe_orchestrator.py:215-222 strips the manifest only when `deferred_tools` present AND native
  tool search. For tito, strip `OWUI_TOOL_MANIFEST_RE` from every `role:"system"` string message
  unconditionally.

### 2.5 Per-turn fingerprint + change alert (P4)
- After the payload is built (post `_apply_cache_control`), for tito log exactly one line:
  `[TITO] sha256_tools=<hex> sha256_system=<hex> model=<base> thinking=<type> effort=<effort> betas=<csv>`.
  `sha256_tools`/`sha256_system` over the *canonicalized, wire-order* serialization (mirror
  `_canonicalize_block`), empty→empty-string hash.
- Keep a per-`chat_id` record of the previous tuple. If `sha256_tools` **or** `sha256_system` changed
  within the same `chat_id`, log `WARNING` **and** send a Discord alert (env `TITO_DISCORD_WEBHOOK`) —
  not just a log line.

## 3. PATCH D — mid-conversation system messages (vet H1/H3 + PLAN "PATCH D")

Current: `_convert_messages_to_claude_format` (messages.py:81-106) folds **every** `role:"system"`
message into the top-level `system` array. Change (tito only):

- A message with top-level `mid_conversation: true` (D2) is **not** folded into `system`; it is
  appended to `processed_messages` as `{"role":"system","content":[<text blocks>],
  "clear_at":"next_user_message"}` at its replay position (after the user message that produced it).
- Orphan status: a mid-conversation system message is **dropped** when the *next* replayed message
  is not an assistant turn and it is not the last message (vet H1: replaying `user, system, user, ...`
  is a permanent 400). Post-pass over `processed_messages`.
- Beta header `mid-conversation-system-clear-at-2026-08-21` added (payload.py beta assembly) when any
  mid-conversation system message is present. (Live-verified on sonnet-5-5 + haiku-5-5 per BUILD LOG.)
- Never put `cache_control` on a mid-conversation system message (see §4).
- Non-tito behavior unchanged.

## 4. PATCH D cache breakpoint (vet H3 + P6)

- `_apply_cache_control` step 4 / `_cache_last_stable_message` (cache_control.py:663-695): when the
  newest message is a `role:"system"` mid-conversation message, skip it and place the messages
  breakpoint on the preceding non-system cacheable block (the user message). Same in the tool-loop
  branch. `_place_cache_on_last_cacheable_block` already skips thinking/redacted; extend the callers
  to also skip `role:"system"` messages.

## 5. B4 — frozen tools array + missing-tool handling (C3)

- **File format (C3):** `TITO_TOOLS_FILE` is a JSON **list** in Anthropic tools format —
  `[{"name","description","input_schema"}, …]` — exactly as sent to the API. Must include the file
  tools (`view_file`, `query_chat_files`, …), memory tools (`add_memory`, `search_memories`, …),
  calendar tools and the TinyFish tools. Written by a deploy-time dump helper (see §8).
- **Request-time (tito):** `_convert_tools_to_claude_format` returns the frozen array loaded from
  `TITO_TOOLS_FILE` (byte-stable across outages and upstream description edits) instead of live
  `__tools__`/`body.tools`. Order fixed.
- **Missing MCP tool at call time:** when a tool_use names a tool absent from the live callables,
  return a `tool_result` with `is_error: True` and the Vietnamese text constant (KEY.md), and fire a
  Discord alert (env `TITO_DISCORD_WEBHOOK`). **Never** remove the tool from `tools[]`.
- Frozen tools are *always* sent even when no live socket/tool-server is reachable (vet B2/B4).

## 6. BUDGET GATE (PLAN "DECISIONS 2026-11-09" + vet H5/H6/H7 + P2/P3)

- Month = calendar month in `America/Los_Angeles` (Pacific). Month key = `"YYYY-MM"`.
- **Spend source (P2):** `cost_usd` persisted by SHOW_COST → `total_usage["cost_usd"]`
  (pipe_orchestrator.py:1266) → `_public_usage` (runtime.py:131, keeps non-underscore keys) →
  `Chats.upsert_message_to_chat_by_id_and_message_id(... {"usage": ...})` (pipe_orchestrator.py:1483).
  `cost_usd` lands in `message.usage.cost_usd`.
- **Helper (P2):** `month_spend_usd(now) -> float` = SUM of `usage.cost_usd` across **all users'
  chats** in the OWUI DB whose message timestamp falls in the Pacific month containing `now`.
  Exact ORM/query against `open_webui.models.chats` is **VERIFY-AT-RUNTIME** (OWUI not in this repo);
  the store point is the `upsert_message_to_chat_by_id_and_message_id` call above. Cache the sum for
  **60s**. **Fail-OPEN**: on any DB error return `0.0` **and** send a Discord alert (never block Dad).
- **Prerequisite fix (vet H6.1):** for tito force `include_usage = True` (pipe_orchestrator.py:243-246)
  so `cost_usd` is always computed/persisted, while token display stays off.
- Pure decision function (testable, no DB):
  `_tito_budget_decision(month_spend_usd: float) -> tuple[str, str|None]`:
  - `month_spend_usd >= TITO_HARD_CAP_USD` (**default 22**, P3) → `("hard", HARD_CAP_TEXT)`.
  - `TITO_SOFT_CAP_USD` (**15**) `<= month_spend_usd < hard` → `("soft", None)` (one Discord warning/month).
  - else → `("ok", None)`.
  Caps read from env (`TITO_HARD_CAP_USD`/`TITO_SOFT_CAP_USD`) with the defaults above.
- `HARD_CAP_TEXT` (exact): `"Tito đang nghỉ tới ngày 1 tháng sau. Dori đã biết rồi, ông không cần làm gì."`
- In `pipe()`, **before** any client build or API call (right after the task-model branch and auth),
  for tito: `spend = month_spend_usd(now)`; `_tito_budget_decision(spend)`. On `"hard"`: emit
  `HARD_CAP_TEXT` as a normal assistant message and return — **no Anthropic API call**. On `"soft"`:
  if no warning sent for this month key, one Discord alert. Soft dedupe key = month key;
  `self._tito_soft_warned: set[str]` (in-memory; a Space restart may re-warn once — accepted).

## 7. P5 — minimal error mapping for tito (vet H7, now in scope)

- For tito, any Anthropic API error (rate-limit, overload, billing/hard-cap, 400/401/500, connection)
  that would otherwise surface raw English/German text → emit a fixed Vietnamese assistant line
  `TITO_ERROR_TEXT` (exact, KEY.md) **and** send the raw error to Discord (`TITO_DISCORD_WEBHOOK`).
  Never show the raw error to the user. Implemented as a tito branch in `handle_errors` /
  `_handle_stream_exception` (runtime.py) and the truncated-stream retry notices
  (pipe_orchestrator.py:1086-1116). Non-tito error handling unchanged.

## 8. Build / regeneration

- Add `SRC_DIR / "tito.py"` to `METHOD_GROUP_SOURCES` in `helpers/build_anthropic_pipe.py`.
- Deploy dump helper `helpers/dump_tito_tools.py` writes `TITO_TOOLS_FILE` (C3 format) from the live
  tool set; run at deploy, committed or produced in the Space image.
- `python helpers/build_anthropic_pipe.py` (writes `anthropic_pipe.py` + `anthropic_pipe.min.py`).
- Bump `version:` + changelog in `src/anthropic_pipe/pipe_template.py`; mirror to `README.md`.
- Commit `src/` **and** both regenerated artifacts (AGENTS.md release checklist).

## 9. File:line reconciliation (doc vs real code)

Legend: ✓ exact/near-exact · ≈ drifted a few lines (real loc shown) · EXT = points into OWUI repo (not here).

| Doc cite | Claim | Real location | Verdict |
|---|---|---|---|
| vet B1 `pipe_template.py:1186-1330` | UserValves block | 1180-1361 | ≈ |
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
| vet M1 `tools.py:196-208 web_search version` | — | 196-212 | ≈ |
| vet M1 `tools.py:175-190 user_location` | — | 175-227 | ≈ |
| vet H5 `pipe_orchestrator.py:332-335 hash diff` | — | 332-335 | ✓ |
| vet H7 `pipe_orchestrator.py:1087-1115` / `1177-1181` | retry/error text | 1086-1116 / 1166-1182 | ≈ |
| vet `payload.py:32-34 max_tokens` | — | 32-34 | ✓ |
| vet `payload.py:546-570 cache diag` | — | 540-563 | ≈ |
| orc C2 `functions.py:265-266` / `main.py:1343` | tito gate / body rewrite | — | EXT (VERIFY-AT-RUNTIME) |

No cited file:line is *factually* wrong about behavior; only line drift and OWUI-repo cites. Real
locations used throughout.
