#!/usr/bin/env python
"""Standalone data-generation test harness.

Run:  python tests/test_datagen.py
Optional live Ollama smoke test:
      OLLAMA_MODEL=qwen2.5:3b python tests/test_datagen.py --live

Exercises the REAL parsing/repair pipeline against a battery of malformed
LLM outputs so we can see exactly which cases still fail. No GPU needed.
Paste the full output back so fixes can target the actual failures.
"""
from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

# Make `touster` importable when run from repo root or tests/
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Windows consoles default to cp1252 and crash on non-ASCII output
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:  # noqa: BLE001
    pass

from touster.dataset import generate as gen
from touster.dataset import structure as struct
from touster.dataset.schema import from_list

# ── tiny test harness ──────────────────────────────────────────────────────
_PASS = 0
_FAIL = 0
_RESULTS: list[str] = []


def expect(name: str, fn, *, want_samples: int | None = None) -> None:
    """Run fn(), expect it to return a list of dicts that from_list accepts.

    want_samples: if set, assert exactly this many valid samples come out.
    """
    global _PASS, _FAIL
    try:
        out = fn()
        ds = from_list(out)
        n = len(ds)
        if want_samples is not None and n != want_samples:
            _FAIL += 1
            _RESULTS.append(f"FAIL  {name}: got {n} samples, wanted {want_samples}")
        else:
            _PASS += 1
            _RESULTS.append(f"pass  {name}: {n} sample(s)")
    except Exception as exc:  # noqa: BLE001
        _FAIL += 1
        _RESULTS.append(f"FAIL  {name}: {type(exc).__name__}: {exc}")


def expect_raises(name: str, fn) -> None:
    global _PASS, _FAIL
    try:
        fn()
        _FAIL += 1
        _RESULTS.append(f"FAIL  {name}: expected an exception, none raised")
    except Exception:  # noqa: BLE001
        _PASS += 1
        _RESULTS.append(f"pass  {name}: raised as expected")


# ── battery of malformed mode-0 generator outputs ──────────────────────────
GOOD = '[{"messages":[{"role":"user","content":"Q?"},{"role":"assistant","content":"A."}]}]'
TRAILING_COMMA_OBJ = (
    '[{"messages":[{"role":"user","content":"Q?"},'
    '{"role":"assistant","content":"A."}]},]'
)
TRAILING_COMMA_FIELD = (
    '[{"messages":[{"role":"user","content":"Q?",},'
    '{"role":"assistant","content":"A.",}]}]'
)
FENCED = "```json\n" + GOOD + "\n```"
FLAT = '[{"role":"user","content":"Q?"},{"role":"assistant","content":"A."}]'
LEADING_PROSE = "Sure! Here is your data:\n" + GOOD
NESTED = '[{"messages":[[{"role":"user","content":"Q?"},{"role":"assistant","content":"A."}]]}]'
BAD_ESCAPE = '[{"messages":[{"role":"user","content":"path C:\\Users\\x"},{"role":"assistant","content":"ok"}]}]'
DOUBLE_ESCAPE = '[{"messages":[{"role":"user","content":"a\\\\p b"},{"role":"assistant","content":"ok"}]}]'
LINE_COMMENT = "[\n// here you go\n" + GOOD[1:]
SINGLE_OBJECT = '{"messages":[{"role":"user","content":"Q?"},{"role":"assistant","content":"A."}]}'
TRUNCATED = '[{"messages":[{"role":"user","content":"Q?"},{"role":"assistant","content":"A."}]},{"messages":[{"role":"user","content":"trunc'


def run_parser_battery() -> None:
    print("\n== mode-0 generate._parse_llm_json battery ==")
    expect("clean array", lambda: gen._parse_llm_json(GOOD), want_samples=1)
    expect("trailing comma after object", lambda: gen._parse_llm_json(TRAILING_COMMA_OBJ), want_samples=1)
    expect("trailing comma after field", lambda: gen._parse_llm_json(TRAILING_COMMA_FIELD), want_samples=1)
    expect("markdown fenced", lambda: gen._parse_llm_json(FENCED), want_samples=1)
    expect("flat [{role,content}] auto-wrap", lambda: gen._parse_llm_json(FLAT), want_samples=1)
    expect("leading prose", lambda: gen._parse_llm_json(LEADING_PROSE), want_samples=1)
    expect("double-nested messages", lambda: gen._parse_llm_json(NESTED), want_samples=1)
    expect("invalid escape \\U \\x in path", lambda: gen._parse_llm_json(BAD_ESCAPE), want_samples=1)
    expect("already-valid \\\\p pair not corrupted", lambda: gen._parse_llm_json(DOUBLE_ESCAPE), want_samples=1)
    expect("// line comment", lambda: gen._parse_llm_json(LINE_COMMENT), want_samples=1)
    expect("single object (salvage)", lambda: gen._parse_llm_json(SINGLE_OBJECT), want_samples=1)
    expect("truncated 2nd item (salvage 1)", lambda: gen._parse_llm_json(TRUNCATED), want_samples=1)


def run_structure_battery() -> None:
    print("\n== mode-1 structure._parse_llm_json battery (same inputs) ==")
    expect("clean array", lambda: struct._parse_llm_json(GOOD), want_samples=1)
    expect("trailing comma after object", lambda: struct._parse_llm_json(TRAILING_COMMA_OBJ), want_samples=1)
    expect("trailing comma after field", lambda: struct._parse_llm_json(TRAILING_COMMA_FIELD), want_samples=1)
    expect("markdown fenced", lambda: struct._parse_llm_json(FENCED), want_samples=1)
    expect("flat [{role,content}] auto-wrap", lambda: struct._parse_llm_json(FLAT), want_samples=1)
    expect("// line comment", lambda: struct._parse_llm_json(LINE_COMMENT), want_samples=1)


# ── mock-client retry path ─────────────────────────────────────────────────
class MockClient:
    """Returns a scripted sequence of replies, ignoring the prompt."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies
        self._i = 0
        self.calls = 0

    def chat(self, messages, model="", temperature=0.7, max_tokens=2048, format="") -> str:
        self.calls += 1
        reply = self._replies[min(self._i, len(self._replies) - 1)]
        self._i += 1
        return reply

    def list_models(self):
        return ["mock"]


def run_mock_client() -> None:
    print("\n== _generate_batch retry / shape handling (mock client) ==")

    def bad_then_good():
        c = MockClient(["not json at all", GOOD])
        return gen._generate_batch(c, "topic", 1, "mock", "")

    expect("retry recovers after 1 bad reply", bad_then_good, want_samples=1)

    def flat_reply():
        c = MockClient([FLAT])
        return gen._generate_batch(c, "topic", 1, "mock", "")

    expect("batch returns flat->wrapped", flat_reply, want_samples=1)

    def always_bad():
        c = MockClient(["garbage", "still garbage"])
        return gen._generate_batch(c, "topic", 1, "mock", "")

    expect_raises("two bad replies -> RuntimeError", always_bad)


# ── optional live Ollama smoke test ────────────────────────────────────────
def run_live() -> None:
    model = os.environ.get("OLLAMA_MODEL", "")
    print(f"\n== LIVE Ollama smoke test (model={model or 'auto'}) ==")
    try:
        from touster.llm.factory import build_client

        client = build_client(ollama_port=int(os.environ.get("OLLAMA_PORT", "11434")), model=model)
        avail = client.list_models()
        print(f"  available models: {avail}")
        if not model:
            # auto-pick a chat-capable model — embedding models (e.g. nomic-embed)
            # return HTTP 400 "does not support chat"
            chat_models = [m for m in avail if "embed" not in m.lower()]
            model = chat_models[0] if chat_models else (avail[0] if avail else "")
            print(f"  auto-picked chat model: {model}")
        ds = gen.generate_dataset(client, "Python list tips", num_samples=4, model=model, batch_size=4)
        print(f"  generated {len(ds)} samples; first:")
        first = ds.to_list()[0]
        for m in first["messages"]:
            print(f"    {m['role']}: {m['content'][:80]}")
        print("  LIVE: pass")
    except Exception as exc:  # noqa: BLE001
        print(f"  LIVE: FAIL — {type(exc).__name__}: {exc}")
        traceback.print_exc()


def main() -> int:
    run_parser_battery()
    run_structure_battery()
    run_mock_client()
    if "--live" in sys.argv:
        run_live()

    print("\n" + "=" * 60)
    for line in _RESULTS:
        print(line)
    print("=" * 60)
    print(f"TOTAL: {_PASS} passed, {_FAIL} failed")
    return 1 if _FAIL else 0


if __name__ == "__main__":
    raise SystemExit(main())
