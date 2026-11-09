# FLASH-BRIEF — Tito pipe patches (B, D, B4 freeze, budget gate)

Coder brief. Read `tito-spec/SPEC.md` (wins) then `tito-spec/KEY.md`. Build the SPEC, nothing more.
Work only in `/home/user/openwebui_anthropic_api_manifold_pipe`.

## Allowed files (edit ONLY these)
- `src/anthropic_pipe/tito.py` (new; self-contained `class PipeTitoMethods` mixin, no open_webui/anthropic imports at module level)
- `src/anthropic_pipe/request/payload.py` (tito guard: model remap, effort, thinking, nudge/hint, beta header)
- `src/anthropic_pipe/request/tools.py` (tito guard: advisor, frozen tools, web max_uses/user_location)
- `src/anthropic_pipe/request/messages.py` (tito guard: mid-conversation system passthrough + orphan drop)
- `src/anthropic_pipe/request/cache_control.py` (tito TTL 1h + skip role:system on messages breakpoint)
- `src/anthropic_pipe/pipe_orchestrator.py` (manifest strip, include_usage force, fingerprint log, budget gate, missing-tool error result)
- `helpers/build_anthropic_pipe.py` (add `SRC_DIR / "tito.py"` to `METHOD_GROUP_SOURCES`)
- `helpers/dump_tito_tools.py` (new; deploy-time dump helper)
- `src/anthropic_pipe/pipe_template.py` + `README.md` (version bump + changelog only)

Regenerate with `python helpers/build_anthropic_pipe.py` (writes both artifacts). Commit `src/` + artifacts.

## Acceptance criteria (numbered)
1. `tito.py` defines `TITO_CONFIG` exactly as SPEC §2, `TITO_MODEL_ID = "tito"`, and methods
   `_is_tito`, `_tito_config`, `_is_mid_conversation_system`, `_tito_budget_decision`, `_tito_pacific_month`,
   `_tito_frozen_tools`, `_tito_fingerprint`, `_tito_missing_tool_result`.
2. `is_tito = model_name.split("/")[-1] == "tito"`; for tito `actual_model_name` becomes `claude-sonnet-5-5`
   before capability/effort/thinking logic. Non-tito path untouched.
3. For tito: `output_config = {"effort":"medium"}`, `thinking = {"type":"adaptive"}`; `body.reasoning_effort`
   ignored; no tool-search nudge; no terminal hint.
4. For tito: advisor tool `{type advisor_20260301, model claude-opus-5-5, max_uses 3, caching 5m}` (no downgrade guard);
   `enable_tool_search` off (no defer_loading, no `tool_search_tool_*`); web_search max_uses 10,
   user_location `{"city":"Fresno"}` (only non-empty fields); `web_search_20250305` (not 20260209).
5. For tito: cache TTL `1h` for tools_system **and** messages; `CACHE_CONTROL` full.
6. PATCH D: a `role:"system"` msg with `meta.mid_conversation` truthy → emitted as `role:"system"` in
   `messages[]` with `clear_at:"next_user_message"` (never folded into `system`); orphan status dropped
   (next msg not assistant and not last); beta `mid-conversation-system-clear-at-2026-08-21` added.
7. Messages cache breakpoint skips `role:"system"` messages; if last msg is a mid-conversation system,
   breakpoint lands on the preceding user message.
8. For tito: tools come from `TITO_TOOLS_FILE` (frozen, byte-stable); never dropped from `tools[]`.
9. Missing live tool at call time → `tool_result` `is_error:True` with the Vietnamese text constant
   (KEY.md) + Discord alert (`TITO_DISCORD_WEBHOOK`), never removed from `tools[]`.
10. Budget gate: for tito force `include_usage=True` (cost recorded, display off). Before any API call,
    `_tito_budget_decision(month_spend)` — `>=25` → return `HARD_CAP_TEXT` (no API call); `>=15` → one
    Discord warning per Pacific month. Month = `America/Los_Angeles`, key `YYYY-MM`.
11. For tito: strip `OWUI_TOOL_MANIFEST_RE` from system messages unconditionally.
12. For tito: log one fingerprint line per turn
    (`sha256_tools sha256_system model thinking effort betas`); WARN on change within same `chat_id`.
13. All of the above `if _is_tito(...)` guarded; non-tito output identical to HEAD (3208fa5).

## Done proof (run these; all must PASS)
- `python tito-spec/tests/test_tito.py`  → prints `All tito checks passed` (this is the RED test; it must be green when done)
- `python helpers/test_fast_mode_header.py`  → `4 checks passed` (no regression)
- `python helpers/test_advisor_pairing.py`   → `All advisor-pairing checks passed`
- `python helpers/test_valve_encryption.py`  → `11 checks passed`
- `python helpers/build_anthropic_pipe.py`   → writes both artifacts, exits 0
- (skip `helpers/test_model_cache.py` — needs `anthropic` SDK not installed here)

## Stop-and-ask rule
STOP (write the exact question, end) when: a SPEC file:line doesn't match the code; the tito→base model
decision (SPEC D1/D2) is wrong; `TITO_TOOLS_FILE` schema is unclear; or a test in `tito-spec/tests/`
can't be made to pass without changing non-tito behavior.

## Key answers / expected values
See `tito-spec/KEY.md` (exact strings, env vars, open questions). Do NOT invent your own Vietnamese text
or cap values.
