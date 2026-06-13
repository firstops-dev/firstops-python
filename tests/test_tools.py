"""Tests for the @firstops.tool base decorator."""

import asyncio
import base64
import inspect
import json
from types import SimpleNamespace

import pytest

import firstops._runtime as rt_mod
from firstops.events import (
    CHANNEL_SYSTEM_TOOLS,
    EVENT_POST_TOOL_USE,
    EVENT_PRE_TOOL_USE,
    Decision,
)
from firstops.tools import FirstOpsPolicyError, tool


class _FakeEnforcement:
    def __init__(self, decision: Decision):
        self.decision = decision
        self.events = []

    def evaluate(self, event):
        self.events.append(event)
        # Post events are audit-only; always allow them regardless of script.
        if event.event_type == EVENT_POST_TOOL_USE:
            return Decision(action="allow")
        return self.decision


@pytest.fixture
def govern(monkeypatch):
    """Install a fake runtime returning a scripted decision; yield the fake."""

    def _install(decision: Decision) -> _FakeEnforcement:
        fake = _FakeEnforcement(decision)
        monkeypatch.setattr(
            rt_mod, "runtime", lambda: SimpleNamespace(enforcement=fake)
        )
        return fake

    return _install


def test_tool_allows_and_emits_pre_event(govern):
    fake = govern(Decision(action="allow"))

    @tool
    def add(a, b):
        return a + b

    assert add(2, 3) == 5
    pre = [e for e in fake.events if e.event_type == EVENT_PRE_TOOL_USE]
    assert len(pre) == 1
    assert pre[0].tool_name == "add"
    assert pre[0].channel == CHANNEL_SYSTEM_TOOLS
    assert pre[0].tool_input == {"a": 2, "b": 3}


def test_tool_emits_post_audit(govern):
    fake = govern(Decision(action="allow"))

    @tool
    def echo(x):
        return x

    echo("hi")
    post = [e for e in fake.events if e.event_type == EVENT_POST_TOOL_USE]
    assert len(post) == 1
    assert post[0].tool_output == {"result": "hi"}


def test_tool_deny_raises_and_does_not_run(govern):
    govern(Decision(action="deny", reason="not allowed", policy_id="p1"))
    ran = {"called": False}

    @tool
    def danger():
        ran["called"] = True
        return "ran"

    with pytest.raises(FirstOpsPolicyError) as exc:
        danger()
    assert exc.value.tool_name == "danger"
    assert exc.value.policy_id == "p1"
    assert ran["called"] is False  # body never executed


def test_tool_modify_applies_scrubbed_args(govern):
    payload = base64.b64encode(json.dumps({"x": 99}).encode()).decode()
    govern(Decision.from_wire({"decision": "modify", "modified_payload": payload}))

    @tool
    def keep(x):
        return x

    # Called with x=1, but the modify replaces it with x=99.
    assert keep(1) == 99


def test_tool_async(govern):
    fake = govern(Decision(action="allow"))

    @tool
    async def aadd(a, b):
        return a + b

    assert asyncio.run(aadd(4, 5)) == 9
    assert any(e.event_type == EVENT_PRE_TOOL_USE for e in fake.events)


def test_tool_async_deny_raises(govern):
    govern(Decision(action="deny", reason="no"))

    @tool
    async def atool():
        return "ran"

    with pytest.raises(FirstOpsPolicyError):
        asyncio.run(atool())


def test_tool_no_runtime_runs_ungoverned(monkeypatch):
    monkeypatch.setattr(rt_mod, "runtime", lambda: None)

    @tool
    def add(a, b):
        return a + b

    assert add(1, 1) == 2  # no governance, just runs


def test_tool_preserves_signature_and_name():
    @tool
    def documented(a: int, b: str = "x") -> str:
        """my docstring"""
        return f"{a}{b}"

    assert documented.__name__ == "documented"
    assert documented.__doc__ == "my docstring"
    sig = inspect.signature(documented)
    assert list(sig.parameters) == ["a", "b"]


def test_tool_refuses_built_framework_tool_object():
    class FakeStructuredTool:
        def invoke(self, *a, **k):
            return None

        def __call__(self, *a, **k):
            return None

    with pytest.raises(TypeError, match="plain function"):
        tool(FakeStructuredTool())


def test_tool_custom_name(govern):
    fake = govern(Decision(action="allow"))

    @tool(name="renamed")
    def original():
        return 1

    original()
    assert fake.events[0].tool_name == "renamed"
