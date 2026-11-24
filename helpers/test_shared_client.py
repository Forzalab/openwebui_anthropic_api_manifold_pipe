"""Self-check for the shared AsyncAnthropic client cache.

Every request path builds its client through ``_build_anthropic_client``. A
fresh client per call leaks an httpx connection pool per turn, so the factory
must reuse one client per distinct connection config. The cache key must also
never contain the plaintext API key.

Runs without OpenWebUI and without network access: ``AsyncAnthropic`` is
replaced by a fake that records constructor arguments.

Usage:
    python helpers/test_shared_client.py
"""
from __future__ import annotations

import hashlib
import importlib.util
import sys
from pathlib import Path

MODULE_PATH = (
    Path(__file__).resolve().parents[1] / "src" / "anthropic_pipe" / "shared" / "models.py"
)


def _load_module():
    spec = importlib.util.spec_from_file_location("_shared_models", MODULE_PATH)
    module = importlib.util.module_from_spec(spec)
    sys.modules["_shared_models"] = module
    spec.loader.exec_module(module)
    # The compiled artifact resolves these from module scope inside class Pipe.
    module.hashlib = hashlib
    module.decrypt_valve_secret = lambda value: value
    return module


class FakeValves:
    def __init__(self, **kwargs):
        self.ANTHROPIC_API_KEY = "sk-test"
        self.ANTHROPIC_BASE_URL = "https://api.anthropic.com"
        self.ANTHROPIC_WORKSPACE_ID = ""
        self.__dict__.update(kwargs)


class FakeAnthropic:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.closed = False
        self.close_calls = 0
        FakeAnthropic.instances.append(self)

    def close(self):
        self.closed = True
        self.close_calls += 1


def build_pipe_class(module):
    class TestPipe(module.PipeModelSupportMethods):
        _DEFAULT_API_BASE = "https://api.anthropic.com"
        _anthropic_clients: dict = {}

        def __init__(self, pipe_valves):
            self.valves = pipe_valves

    module.AsyncAnthropic = FakeAnthropic
    FakeAnthropic.instances = []
    return TestPipe


def build_pipe(module, valves=None):
    return build_pipe_class(module)(valves or FakeValves())


def test_same_config_reuses_client_across_instances(module):
    pipe_class = build_pipe_class(module)
    first_pipe = pipe_class(FakeValves())
    second_pipe = pipe_class(FakeValves())

    first = first_pipe._build_anthropic_client(
        "sk-test", default_headers={"b": "2", "a": "1"}, timeout=30
    )
    second = second_pipe._build_anthropic_client(
        "sk-test", default_headers={"a": "1", "b": "2"}, timeout=30
    )

    assert first is second, "same config must return the identical client object"
    assert len(FakeAnthropic.instances) == 1, "same config must construct one client"
    assert first.kwargs["api_key"] == "sk-test"
    assert first.close_calls == 0, "a shared client must not be closed per request"


def test_different_api_key_gets_different_client(module):
    pipe = build_pipe(module)

    first = pipe._build_anthropic_client("sk-test")
    second = pipe._build_anthropic_client("sk-other")

    assert first is not second, "a different key must not reuse the previous client"
    assert len(FakeAnthropic.instances) == 2, "a different key must build a new client"


def test_cache_key_does_not_contain_plaintext_api_key(module):
    pipe = build_pipe(module, FakeValves(ANTHROPIC_BASE_URL="https://proxy.internal/v1"))
    pipe._build_anthropic_client("sk-secret-value")

    cache_keys = list(pipe._anthropic_clients)
    assert len(cache_keys) == 1
    assert "sk-secret-value" not in repr(cache_keys), (
        "the plaintext API key must never be a cache key"
    )
    assert len(cache_keys[0][0]) == 64, "the key fingerprint must be a sha256 hex digest"
    assert cache_keys[0][0] != "sk-secret-value"


def main():
    module = _load_module()
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for test in tests:
        test(module)
        print(f"  ok  {test.__name__}")
    print(f"\n{len(tests)} checks passed")


if __name__ == "__main__":
    main()
