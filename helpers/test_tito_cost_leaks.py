"""Self-check for the two budget-gate leaks: advisor iterations and the guard call.

(a) advisor `usage.iterations` are priced at the ADVISOR model's rates under the
    `advisor` breakdown key, without touching the executor's own components;
(b) the guard's Haiku usage adds `guard` to the breakdown and raises `cost_usd`;
(c) a guard exception leaves the cost unchanged.

Runs without OpenWebUI and without network access.

Usage:
    python helpers/test_tito_cost_leaks.py
"""
from __future__ import annotations

import asyncio
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

SRC = Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe"


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


PRICING = _load("_leaks_pricing", SRC / "shared" / "pricing.py")
GUARD = _load("_leaks_guard", SRC / "response" / "tito_guard.py")
TITO = _load("_leaks_tito", SRC / "tito.py")
for _name in ("split_visible", "visible_text", "tito_guard_layout", "tito_guard_rebuild",
              "tito_guard_trigger", "tito_guard_safety_ok", "guard", "guard_verdict"):
    setattr(TITO, _name, getattr(GUARD, _name))

MP = PRICING.ModelPricing
EXECUTOR = "claude-haiku-4-5"  # $1 / $5
ADVISOR = "claude-opus-5-5"  # $4 / $20, cache_read $0.20


def _tally(**top) -> dict:
    usage = {"input_tokens": 0, "output_tokens": 0, "cache_creation_input_tokens": 0,
             "cache_read_input_tokens": 0}
    usage.update(top)
    return usage


def test_advisor_iterations_priced_at_advisor_rates():
    iterations = [
        SimpleNamespace(type="message", input_tokens=1000, output_tokens=100),
        SimpleNamespace(type="advisor_message", model=ADVISOR, input_tokens=2000,
                        output_tokens=500, cache_read_input_tokens=10000),
        {"type": "message", "input_tokens": 1500, "output_tokens": 200},  # dict form
        {"type": "mystery", "model": ADVISOR, "input_tokens": 999999},  # unknown: skipped
    ]
    usage = SimpleNamespace(iterations=iterations)
    total = _tally(input_tokens=1000, output_tokens=300)
    MP.record_advisor_iterations(usage, total, {})

    pricing = MP()
    breakdown = pricing.breakdown(EXECUTOR, total)
    # executor tokens only at executor rates: no double count of advisor tokens
    assert breakdown["input"] == round(1000 * 1.0 / 1e6, 6), breakdown
    assert breakdown["output"] == round(300 * 5.0 / 1e6, 6), breakdown
    expected_advisor = round((2000 * 4.0 + 500 * 20.0 + 10000 * 0.20) / 1e6, 6)
    assert breakdown["advisor"] == expected_advisor, breakdown
    assert pricing.estimate(EXECUTOR, total) == round(sum(breakdown.values()), 6)


def test_advisor_iterations_cumulative_within_one_call():
    total = _tally()
    first = SimpleNamespace(iterations=[
        {"type": "advisor_message", "model": ADVISOR, "input_tokens": 100, "output_tokens": 10}])
    snap = MP.record_advisor_iterations(first, total, {})
    # same call, later message_delta: cumulative list grew; only growth is added
    second = SimpleNamespace(iterations=[
        {"type": "advisor_message", "model": ADVISOR, "input_tokens": 100, "output_tokens": 10},
        {"type": "advisor_message", "model": ADVISOR, "input_tokens": 50, "output_tokens": 5}])
    MP.record_advisor_iterations(second, total, snap)
    assert total["_advisor"][ADVISOR]["input"] == 150
    assert total["_advisor"][ADVISOR]["output"] == 15
    # no advisor and malformed usage: nothing recorded, no raise
    clean = _tally()
    MP.record_advisor_iterations(SimpleNamespace(iterations=None), clean, {})
    MP.record_advisor_iterations(None, clean, {})
    assert "advisor" not in (MP().breakdown(EXECUTOR, clean) or {})


class _Messages:
    def __init__(self, client):
        self.client = client

    async def create(self, **kwargs):
        if self.client.error is not None:
            raise self.client.error
        return SimpleNamespace(
            content=[SimpleNamespace(type="text", text='{"ok": true}')],
            usage=SimpleNamespace(input_tokens=2000, output_tokens=100,
                                  cache_creation_input_tokens=0, cache_read_input_tokens=0),
        )


class _Client:
    def __init__(self, error=None):
        self.error = error
        self.messages = _Messages(self)


class _Ctx:
    def __init__(self, content):
        self.content = content

    def text(self):
        return self.content

    async def emit_replace(self, content):
        self.content = content


class _Pipe(TITO.PipeTitoMethods):
    def __init__(self):
        self.valves = SimpleNamespace(TITO_OUTPUT_GUARD=True)


META = {"model": {"id": "tito"}}
BREAKING = "Tito chào anh. Anh ăn gì chưa? Anh ngủ chưa?"  # trips the pre-check


def _base_total() -> dict:
    return {"cost_usd": 0.01, "cost_breakdown_usd": {"input": 0.01}}


async def test_guard_usage_adds_guard_cost():
    total = _base_total()
    await _Pipe()._tito_output_guard(_Ctx(BREAKING), _Client(), META, total_usage=total, pricing=MP())
    guard_usd = round((2000 * 0.10 + 100 * 0.50) / 1e6, 6)  # claude-haiku-5-5
    assert total["cost_breakdown_usd"]["guard"] == guard_usd, total
    assert total["cost_usd"] == round(0.01 + guard_usd, 6), total
    assert total["cost_usd"] > 0.01


async def test_guard_exception_leaves_cost_unchanged():
    total = _base_total()
    await _Pipe()._tito_output_guard(
        _Ctx(BREAKING), _Client(error=RuntimeError("boom")), META, total_usage=total, pricing=MP())
    assert total == _base_total(), total


async def test_clean_reply_and_default_args_add_nothing():
    total = _base_total()
    await _Pipe()._tito_output_guard(_Ctx("Tito chào anh. Anh khỏe không?"), _Client(), META,
                                     total_usage=total, pricing=MP())
    assert total == _base_total(), total
    await _Pipe()._tito_output_guard(_Ctx(BREAKING), _Client(), META)  # old signature still fine


async def main():
    test_advisor_iterations_priced_at_advisor_rates()
    test_advisor_iterations_cumulative_within_one_call()
    await test_guard_usage_adds_guard_cost()
    await test_guard_exception_leaves_cost_unchanged()
    await test_clean_reply_and_default_args_add_nothing()
    print("5 checks passed")


if __name__ == "__main__":
    asyncio.run(main())
