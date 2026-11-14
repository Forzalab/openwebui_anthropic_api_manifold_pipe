"""Regression test: mimi hide_thinking sends thinking only as the invisible carrier.

Usage:
    python helpers/test_mimi_hide_thinking.py
"""
from __future__ import annotations

import importlib.util
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("_pipe_full", ROOT / "anthropic_pipe.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["_pipe_full"] = mod
spec.loader.exec_module(mod)
pipe = mod.Pipe()

fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


def with_profile(model_id, fn):
    tok = pipe.TITO_PROFILE.set(pipe._tito_profile({"model": {"id": model_id}}) if model_id else None)
    try:
        return fn()
    finally:
        pipe.TITO_PROFILE.reset(tok)


# tito: visible details, unchanged
out = with_profile("tito", lambda: pipe._format_thinking_block("hmm", 2.0, signature="SIG"))
check("tito keeps <details>", out.startswith('<details type="reasoning"'))
# mimi: streaming partial is empty
out = with_profile("mimi", lambda: pipe._format_thinking_block("hmm", None))
check("mimi partial empty", out == "")
# mimi: finished block is a hidden carrier that decodes to a thinking block
out = with_profile("mimi", lambda: pipe._format_thinking_block("hmm", 2.0, signature="SIG"))
check("mimi no <details>", "<details" not in out)
m = mod.PATTERN_HIDDEN_BLOCK.search(out)
check("mimi carrier present", m is not None)
dec = pipe._decode_block_payload(m.group(1)) if m else None
check("carrier decodes to thinking block", dec == [{"type": "thinking", "thinking": "hmm", "signature": "SIG"}])
# other model: unchanged
out = with_profile(None, lambda: pipe._format_thinking_block("hmm", 2.0, signature="SIG"))
check("other model keeps <details>", out.startswith('<details type="reasoning"'))
sys.exit(1 if fails else 0)
