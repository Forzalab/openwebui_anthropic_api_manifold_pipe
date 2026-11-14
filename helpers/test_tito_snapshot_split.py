"""Regression test: N1 system split per profile memory heading (tito unchanged, mimi heading).

Usage:
    python helpers/test_tito_snapshot_split.py
"""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"
spec = importlib.util.spec_from_file_location("_tito_split", SRC / "tito.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["_tito_split"] = mod
spec.loader.exec_module(mod)
PM = mod.PipeTitoMethods

TITO_H = "NHỮNG ĐIỀU TITO NHỚ VỀ ANH"
MIMI_H = "NHỮNG ĐIỀU MIMI NHỚ VỀ BẠN"
fails = 0


def check(name, cond):
    global fails
    print(("ok   " if cond else "FAIL ") + name)
    fails += 0 if cond else 1


def blk(text):
    return [{"type": "text", "text": text}]


check("tito heading constant unchanged", PM.TITO_SNAPSHOT_HEADING == TITO_H)
check("mimi profile heading", PM.TITO_PROFILES["mimi"]["snapshot_heading"] == MIMI_H)

out = PM._tito_split_system(blk("prompt\n\n" + TITO_H + "\n- a"))
check("tito splits", out == [{"type": "text", "text": "prompt"}, {"type": "text", "text": TITO_H + "\n- a"}])

out = PM._tito_split_system(blk("Mimi prompt\n\n" + MIMI_H + "\n- b"))
check("mimi splits", out == [{"type": "text", "text": "Mimi prompt"}, {"type": "text", "text": MIMI_H + "\n- b"}])

src = blk("no heading here")
check("no heading unchanged", PM._tito_split_system(src) is src)

src = blk("\n\n" + MIMI_H + "\n- c")
check("empty head unchanged", PM._tito_split_system(src) is src)

both = "p\n\n" + MIMI_H + "\nx\n\n" + TITO_H + "\ny"
out = PM._tito_split_system(blk(both))
check("tito heading wins when both present", out[1]["text"].startswith(TITO_H))

sys.exit(1 if fails else 0)
