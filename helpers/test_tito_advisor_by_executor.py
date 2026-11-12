"""Self-check for the Tito advisor model chosen by executor.

Tito's advisor is ``TITO_CONFIG["advisor"]["model"]`` (claude-opus-5-5) unless the
executor matches a ``TITO_CONFIG["advisor_by_executor"]`` prefix, in which case
the advisor is that model. A Haiku executor (claude-haiku-*) gets
claude-sonnet-5-5. Every other executor keeps claude-opus-5-5, and non-Tito
requests keep the ADVISOR_MODEL valve path untouched.

Runs without OpenWebUI and without network access.

Usage:
    python3 helpers/test_tito_advisor_by_executor.py
"""
from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

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


TITO = _load("_tito_adv_exec", SRC / "tito.py")
TOOLS = _load("_tito_adv_exec_tools", SRC / "request" / "tools.py")
MODELS = _load("_tito_adv_exec_models", SRC / "shared" / "models.py")
TITO_CFG = TITO.PipeTitoMethods.TITO_CONFIG

TOOLS.json = json
TOOLS.normalize_tool_name_list = lambda value: set()


class TV:
    ENABLE_BASH_TOOL = False
    ENABLE_TEXT_EDITOR_TOOL = False
    WEB_SEARCH = False
    WEB_FETCH = False
    ENABLE_PROGRAMMATIC_TOOL_CALLING = False


class TUV:
    ENABLE_ADVISOR_TOOL = True
    ADVISOR_MODEL = "claude-opus-5"
    ADVISOR_MAX_USES = 0
    ADVISOR_CACHING = "off"
    ENABLE_DYNAMIC_FILTERING = False
    ENABLE_TOOL_SEARCH = False
    TOOL_SEARCH_EXCLUDE_TOOLS = ""
    TOOL_SEARCH_MAX_DESCRIPTION_LENGTH = 1000
    TOOL_SEARCH_TYPE = "bm25"


def _advisor_for(executor: str, *, tito: bool = True, cfg: dict = TITO_CFG):
    """Advisor model in the built tools for ``executor`` (None when absent)."""

    class Pipe(TOOLS.PipeRequestToolsMethods, MODELS.PipeModelSupportMethods):
        valves = TV()

        def get_model_info(self, name):
            return {}

        def _is_tito(self, metadata):
            return (metadata or {}).get("model", {}).get("id") == "tito"

        def _tito_config(self):
            return dict(cfg)

        def _tito_frozen_tools(self):
            return []

    metadata = {"model": {"id": "tito"}} if tito else {"model": {"id": "other"}}
    tools, _ = Pipe()._convert_tools_to_claude_format(
        __tools__={}, body={}, actual_model_name=executor,
        __user__={"valves": TUV()}, __metadata__=metadata,
    )
    for tool in tools:
        if tool.get("name") == "advisor":
            return tool.get("model")
    return None


# ---------------------------------------------------------------- config
record("advisor_by_executor maps claude-haiku- to claude-sonnet-5-5",
       TITO_CFG.get("advisor_by_executor") == {"claude-haiku-": "claude-sonnet-5-5"},
       repr(TITO_CFG.get("advisor_by_executor")))
record("advisor.model stays claude-opus-5-5",
       TITO_CFG["advisor"]["model"] == "claude-opus-5-5")

# ---------------------------------------------------------------- tito executors
record("tito haiku-5-5 executor -> sonnet-5-5 advisor",
       _advisor_for("claude-haiku-5-5") == "claude-sonnet-5-5",
       repr(_advisor_for("claude-haiku-5-5")))
record("tito haiku dated id -> sonnet-5-5 advisor",
       _advisor_for("claude-haiku-5-5-20260215") == "claude-sonnet-5-5",
       repr(_advisor_for("claude-haiku-5-5-20260215")))
record("tito haiku with provider prefix -> sonnet-5-5 advisor",
       _advisor_for("anthropic/claude-haiku-5-5") == "claude-sonnet-5-5",
       repr(_advisor_for("anthropic/claude-haiku-5-5")))
record("tito sonnet-5-5 executor -> opus-5-5 advisor",
       _advisor_for("claude-sonnet-5-5") == "claude-opus-5-5",
       repr(_advisor_for("claude-sonnet-5-5")))
record("tito opus-5-5 executor -> opus-5-5 advisor",
       _advisor_for("claude-opus-5-5") == "claude-opus-5-5",
       repr(_advisor_for("claude-opus-5-5")))
record("tito sonnet-4-6 executor -> opus-5-5 advisor",
       _advisor_for("claude-sonnet-4-6") == "claude-opus-5-5",
       repr(_advisor_for("claude-sonnet-4-6")))

no_map = {k: v for k, v in TITO_CFG.items() if k != "advisor_by_executor"}
record("tito haiku without advisor_by_executor -> opus-5-5 (fallback to model)",
       _advisor_for("claude-haiku-5-5", cfg=no_map) == "claude-opus-5-5",
       repr(_advisor_for("claude-haiku-5-5", cfg=no_map)))

# ---------------------------------------------------------------- non-tito unchanged
record("non-tito haiku executor keeps ADVISOR_MODEL valve (opus-5)",
       _advisor_for("claude-haiku-5-5", tito=False) == "claude-opus-5",
       repr(_advisor_for("claude-haiku-5-5", tito=False)))
record("non-tito sonnet executor keeps ADVISOR_MODEL valve (opus-5)",
       _advisor_for("claude-sonnet-5-5", tito=False) == "claude-opus-5",
       repr(_advisor_for("claude-sonnet-5-5", tito=False)))


# ---------------------------------------------------------------- report
failed = [c for c in CHECKS if not c[1]]
for name, ok, detail in CHECKS:
    print(f"  {'ok  ' if ok else 'FAIL'} {name}" + (f"  [{detail}]" if detail and not ok else ""))
print(f"\n{len(CHECKS) - len(failed)} checks passed" if not failed else f"\n{len(failed)} FAILED")
sys.exit(1 if failed else 0)
