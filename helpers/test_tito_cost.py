"""Self-check for the Tito cost settings.

Covers: web_search / web_fetch caps (3 / 2), the web_fetch ``max_content_tokens``
cap (8000, only when the key is set and > 0), the Tito message-breakpoint TTL
split (tools/system stay 1h, messages 5m when ``messages_ttl`` is "5m"), that
non-Tito cache markers are unchanged, and that the ADVISOR_MODEL valve accepts
claude-sonnet-5-5 without changing its default.

Runs without OpenWebUI and without network access.

Usage:
    python helpers/test_tito_cost.py
"""
from __future__ import annotations

import ast
import contextvars
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"
CHECKS: list = []


def record(name: str, ok: bool, detail: str = "") -> None:
    CHECKS.append((name, bool(ok), detail))


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


TITO = _load("_tito_cost", SRC / "tito.py")
TOOLS = _load("_tito_cost_tools", SRC / "request" / "tools.py")
MODELS = _load("_tito_cost_models", SRC / "shared" / "models.py")
CC = _load("_tito_cost_cc", SRC / "request" / "cache_control.py")
TITO_CFG = TITO.PipeTitoMethods.TITO_CONFIG


# ---------------------------------------------------------------- config
record("tito web_search_max_uses == 3", TITO_CFG["web_search_max_uses"] == 3)
record("tito web_fetch_max_uses == 2", TITO_CFG["web_fetch_max_uses"] == 2)
record("tito web_fetch_max_content_tokens == 8000",
       TITO_CFG.get("web_fetch_max_content_tokens") == 8000)
record("tito cache keeps 1h tools/system ttl", TITO_CFG["cache"].get("ttl") == "1h")
record("tito cache messages_ttl == 5m", TITO_CFG["cache"].get("messages_ttl") == "5m")
record("tito advisor unchanged (opus-5-5, max_uses 3)",
       TITO_CFG["advisor"]["model"] == "claude-opus-5-5" and TITO_CFG["advisor"]["max_uses"] == 3)


# ---------------------------------------------------------------- web tools
class TV:
    ENABLE_BASH_TOOL = False
    ENABLE_TEXT_EDITOR_TOOL = False
    WEB_SEARCH = True
    WEB_FETCH = True
    ENABLE_PROGRAMMATIC_TOOL_CALLING = False


class TUV:
    ENABLE_ADVISOR_TOOL = False
    ADVISOR_MODEL = "claude-opus-5"
    ADVISOR_MAX_USES = 0
    ADVISOR_CACHING = "off"
    ENABLE_DYNAMIC_FILTERING = False
    ENABLE_TOOL_SEARCH = False
    TOOL_SEARCH_EXCLUDE_TOOLS = ""
    TOOL_SEARCH_MAX_DESCRIPTION_LENGTH = 1000
    TOOL_SEARCH_TYPE = "bm25"


TOOLS.json = json
TOOLS.normalize_tool_name_list = lambda value: set()


def _tools_for(cfg: dict):
    class Pipe(TOOLS.PipeRequestToolsMethods, MODELS.PipeModelSupportMethods):
        valves = TV()

        def get_model_info(self, name):
            return {}

        def _is_tito(self, metadata):
            return (metadata or {}).get("model", {}).get("id") == "tito"

        def _tito_config(self):
            return dict(cfg)

    got, _ = Pipe()._convert_tools_to_claude_format(
        __tools__={}, body={}, actual_model_name="claude-sonnet-5-5",
        __user__={"valves": TUV()}, __metadata__={"model": {"id": "tito"}},
    )
    return {t.get("name"): t for t in got}


tito_tools = _tools_for(TITO_CFG)
ws = tito_tools.get("web_search", {})
wf = tito_tools.get("web_fetch", {})
record("tito web_search max_uses 3", ws.get("max_uses") == 3, repr(ws.get("max_uses")))
record("tito web_fetch max_uses 2", wf.get("max_uses") == 2, repr(wf.get("max_uses")))
record("tito web_fetch max_content_tokens 8000",
       wf.get("max_content_tokens") == 8000, repr(wf.get("max_content_tokens")))

no_cap = _tools_for({**TITO_CFG, "web_fetch_max_content_tokens": 0})
record("web_fetch omits max_content_tokens when 0",
       "max_content_tokens" not in no_cap.get("web_fetch", {}))
absent_cfg = {k: v for k, v in TITO_CFG.items() if k != "web_fetch_max_content_tokens"}
record("web_fetch omits max_content_tokens when key absent",
       "max_content_tokens" not in _tools_for(absent_cfg).get("web_fetch", {}))


# ---------------------------------------------------------------- cache TTL
class CV:
    CACHE_TTL = "5 minutes"
    CACHE_TTL_FOR_TOOLS_AND_SYSTEM_PROMT = "same as CACHE_TTL"
    CACHE_CONTROL = "cache tools array, system prompt and messages"
    ENABLE_PROGRAMMATIC_TOOL_CALLING = False


def _pipe(cfg, valves=None, active=False):
    class Pipe(CC.PipeCacheControlMethods):
        TITO_ACTIVE = contextvars.ContextVar("cost_tito_active", default=False)

        def _tito_config(self):
            return dict(cfg)

    p = Pipe()
    p.valves = valves or CV()
    p.TITO_ACTIVE.set(active)
    return p


tito_p = _pipe(TITO_CFG, active=True)
record("tito message marker has no ttl",
       tito_p._cache_control_marker("messages") == {"type": "ephemeral"},
       repr(tito_p._cache_control_marker("messages")))
record("tito tools/system marker is 1h",
       tito_p._cache_control_marker("tools_system") == {"type": "ephemeral", "ttl": "1h"},
       repr(tito_p._cache_control_marker("tools_system")))
no_split = _pipe({**TITO_CFG, "cache": {"ttl": "1h"}}, active=True)
record("tito messages stay 1h when messages_ttl absent",
       no_split._cache_control_marker("messages") == {"type": "ephemeral", "ttl": "1h"},
       repr(no_split._cache_control_marker("messages")))
record("tito 1h tools/system precede 5m messages (order)",
       tito_p._cache_control_marker("tools_system").get("ttl") == "1h"
       and "ttl" not in tito_p._cache_control_marker("messages"))

# Non-Tito: markers follow the valves exactly as before.
plain = _pipe(TITO_CFG, active=False)
record("non-tito messages default 5m (no ttl)",
       plain._cache_control_marker("messages") == {"type": "ephemeral"})
record("non-tito tools/system default 5m (no ttl)",
       plain._cache_control_marker("tools_system") == {"type": "ephemeral"})
one_hour = SimpleNamespace(CACHE_TTL="1 hour", CACHE_TTL_FOR_TOOLS_AND_SYSTEM_PROMT="same as CACHE_TTL",
                           CACHE_CONTROL="x", ENABLE_PROGRAMMATIC_TOOL_CALLING=False)
p1h = _pipe(TITO_CFG, valves=one_hour, active=False)
record("non-tito CACHE_TTL 1 hour -> 1h on messages",
       p1h._cache_control_marker("messages") == {"type": "ephemeral", "ttl": "1h"})


# ---------------------------------------------------------------- advisor valve
tree = ast.parse((SRC / "pipe_template.py").read_text(encoding="utf-8"))
literal_opts = None
default_val = None
for node in ast.walk(tree):
    if isinstance(node, ast.AnnAssign) and getattr(node.target, "id", None) == "ADVISOR_MODEL":
        literal_opts = [e.value for e in node.annotation.slice.elts]
        default_val = ast.literal_eval(node.value.keywords[0].value) if node.value.keywords else None
        break
record("ADVISOR_MODEL accepts claude-sonnet-5-5", literal_opts is not None and "claude-sonnet-5-5" in literal_opts)
record("ADVISOR_MODEL default still claude-opus-5", default_val == "claude-opus-5", repr(default_val))


# ---------------------------------------------------------------- report
failed = [c for c in CHECKS if not c[1]]
for name, ok, detail in CHECKS:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
print(f"\n{len(CHECKS) - len(failed)} checks passed" if not failed else f"\n{len(failed)} FAILED")
sys.exit(1 if failed else 0)
