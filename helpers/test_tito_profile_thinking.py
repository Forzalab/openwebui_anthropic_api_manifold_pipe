"""Regression test: per-profile thinking switch (tito on, mimi off, others untouched).

Usage:
    python helpers/test_tito_profile_thinking.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"
spec = importlib.util.spec_from_file_location("_tito_thinking", SRC / "tito.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["_tito_thinking"] = mod
spec.loader.exec_module(mod)
PM = mod.PipeTitoMethods

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


check("tito thinking on", PM._tito_profile({"model": {"id": "tito"}})["thinking"] is True)
check("mimi thinking off", PM._tito_profile({"model": {"id": "mimi"}})["thinking"] is False)
check("other model has no profile", PM._tito_profile({"model": {"id": "claude-opus-5-5"}}) is None)
payload_src = (SRC / "request" / "payload.py").read_text()
check("payload reads profile thinking", 'bool(_tito_prof.get("thinking", True))' in payload_src)
sys.exit(1 if fails else 0)
