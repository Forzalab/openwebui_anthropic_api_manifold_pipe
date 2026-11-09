# FLASH-BRIEF — Tito pipe patches (B, D, B4 freeze, budget gate, P5 error mapping)

Coder brief. Read `tito-spec/SPEC.md` (wins) then `tito-spec/KEY.md`. Build the SPEC, nothing more.
Work only in `/home/user/openwebui_anthropic_api_manifold_pipe`.

## Allowed files (edit ONLY these)
- `src/anthropic_pipe/tito.py` (new; `class PipeTitoMethods` mixin, `contextvars` imported at top, no open_webui/anthropic imports)
- `src/anthropic_pipe/request/payload.py` (tito guard: effort, thinking, nudge/hint, beta header)
- `src/anthropic_pipe/request/tools.py` (tito guard: advisor, frozen tools, web max_uses/user_location)
- `src/anthropic_pipe/request/messages.py` (tito guard: mid-conversation passthrough + orphan drop)
- `src/anthropic_pipe/request/cache_control.py` (tito TTL 1h + P6 breakpoint layout + skip role:system)
- `src/anthropic_pipe/response/runtime.py` (P5 error mapping for tito)
- `src/anthropic_pipe/pipe_orchestrator.py` (TITO_ACTIVE set, manifest strip, include_usage force, fingerprint+Discord, budget gate, missing-tool result)
- `helpers/build_anthropic_pipe.py` (add `SRC_DIR / "tito.py"` to `METHOD_GROUP_SOURCES`)
- `helpers/dump_tito_tools.py` (new; deploy dump helper)
- `src/anthropic_pipe/pipe_template.py` + `README.md` (version bump + changelog only)

Regenerate with `python helpers/build_anthropic_pipe.py` (writes both artifacts). Commit `src/` + artifacts.

## Acceptance criteria (numbered)
1. `tito.py` defines `TITO_MODEL_ID = "tito"`, `TITO_ACTIVE` ContextVar, `TITO_CONFIG` (SPEC §2), and
   methods `_is_tito`, `_tito_config`, `_is_mid_conversation_system`, `_tito_budget_decision`,
   `month_spend_usd`, `_tito_pacific_month`, `_tito_frozen_tools`, `_tito_fingerprint`,
   `_tito_missing_tool_result`, `_tito_error_text`.
2. `_is_tito(metadata) = (metadata or {}).get("model",{}).get("id") == "tito"`. **No model remap** —
   `body["model"]` is already `claude-sonnet-5-5`. Set `TITO_ACTIVE` in `pipe()`. Non-tito path untouched.
3. For tito: `output_config={"effort":"medium"}`, `thinking={"type":"adaptive"}`; `body.reasoning_effort`
   ignored; no tool-search nudge; no terminal hint.
4. For tito: advisor `{type advisor_20260301, model claude-opus-5-5, max_uses 3, caching 5m}` (no
   downgrade guard); `enable_tool_search` off; web_search max_uses 10, user_location `{"city":"Fresno"}`
   (only non-empty); `web_search_20250305` (not 20260209).
5. P6 cache: for tito every marker `ttl "1h"`; **no tools breakpoint**; cache_control after system
   block1 AND block2; last-message breakpoint 1h; never on `mid_conversation` msgs; ≤4 markers total.
6. PATCH D: msg with top-level `mid_conversation:true` → emitted as `role:"system"` in `messages[]` with
   `clear_at:"next_user_message"` (never folded into `system`); orphan status dropped (next msg not
   assistant and not last); beta `mid-conversation-system-clear-at-2026-08-21` added.
7. Messages breakpoint skips `role:"system"` msgs; if last is mid-conversation system, breakpoint on
   preceding user message.
8. B4/C3: tito tools come from `TITO_TOOLS_FILE` (JSON list in Anthropic format: name/description/
   input_schema; must include file + memory + calendar + TinyFish tools); never dropped from `tools[]`.
9. Missing live tool at call time → `tool_result` `is_error:True` with Vietnamese constant + Discord
   alert; never removed from `tools[]`.
10. Budget (P2/P3): tito forces `include_usage=True` (cost recorded, display off). Before any API call,
    `month_spend_usd(now)` (sum of all users' `usage.cost_usd`, Pacific month, fail-open + Discord on DB
    error, 60s cache) → `_tito_budget_decision`: `>= TITO_HARD_CAP_USD` (default 22) → return
    `HARD_CAP_TEXT`, no API call; `>= 15` → one Discord warning/Pacific month.
11. P5: for tito, any Anthropic API error → `TITO_ERROR_TEXT` + Discord(raw error); never raw English.
12. For tito: strip `OWUI_TOOL_MANIFEST_RE` from system messages unconditionally; fingerprint line per
    turn (`sha256_tools sha256_system model thinking effort betas`); on tools/system sha change within a
    `chat_id`, WARN **and** Discord.
13. All `if _is_tito(...)`/`TITO_ACTIVE` guarded; non-tito output identical to HEAD (3208fa5).

## Done proof (run these; all must PASS)
- `python tito-spec/tests/test_tito.py`  → prints `All tito checks passed` (RED test must go green)
- `python helpers/test_fast_mode_header.py`  → `4 checks passed`
- `python helpers/test_advisor_pairing.py`   → `All advisor-pairing checks passed`
- `python helpers/test_valve_encryption.py`  → `11 checks passed`
- `python helpers/build_anthropic_pipe.py`   → writes both artifacts, exit 0
- (skip `helpers/test_model_cache.py` — needs `anthropic` SDK not installed here)

## Stop-and-ask rule
STOP (write the exact question, end) when: a SPEC file:line doesn't match the code; the C2 gate
(`__metadata__["model"]["id"]`) is wrong at runtime; `TITO_TOOLS_FILE` schema is unclear; or a test in
`tito-spec/tests/` can't go green without changing non-tito behavior.

## Key answers / expected values
See `tito-spec/KEY.md` (exact strings, env vars, caps 22/15, open questions). Do NOT invent your own
Vietnamese text or cap values.
