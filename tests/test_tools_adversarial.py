"""Adversarial tests for @firstops.tool — designed to FIND BUGS, not to pass.

These probe the decorator's two riskiest surfaces:

  1. ``_apply_pre`` reconstructs a *modify*'d call as pure keyword args
     (``return (), new_input``). That is wrong for any function whose
     parameters are not all bind-as-keyword: positional-only params,
     ``*args``, and bound methods (``self``) all explode into a TypeError
     that propagates INTO THE CALLER. Per invariant #1 (fail-open everywhere
     except a real DENY), a benign ``modify`` must never crash the tool.

  2. ``post_tool_use`` audit is skipped when the body raises, and for
     generator/async-generator tools it fires on the *un-iterated* generator
     object — so the audited output is a generator repr, never the yielded
     values. Scrub-on-output is structurally impossible for streaming tools,
     and failed calls are invisible to governance.

Bugs are encoded as ``xfail(strict=True)``: they fail today (documenting the
hole) and will turn into a hard failure the moment the bug is fixed, so they
double as regression guards. Non-bug behaviors are plain asserting tests.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

import firstops._runtime as rt_mod
from firstops.events import (
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
        if event.event_type == EVENT_POST_TOOL_USE:
            return Decision(action="allow")
        return self.decision


@pytest.fixture
def govern(monkeypatch):
    def _install(decision: Decision) -> _FakeEnforcement:
        fake = _FakeEnforcement(decision)
        monkeypatch.setattr(
            rt_mod, "runtime", lambda: SimpleNamespace(enforcement=fake)
        )
        return fake

    return _install


def _modify(payload: dict) -> Decision:
    raw = base64.b64encode(json.dumps(payload).encode()).decode()
    return Decision.from_wire({"decision": "modify", "modified_payload": raw})


# ---------------------------------------------------------------------------
# BUG CLASS A: modify reconstructs the call as pure-kwargs and crashes the
# caller for any non-keyword-bindable signature. Root cause: tools._apply_pre
# does ``return (), new_input`` unconditionally. Severity: BLOCKING — turns a
# benign policy decision into a hard crash, violating fail-open invariant #1.
# ---------------------------------------------------------------------------


def test_modify_positional_only_param_does_not_crash_caller(govern):
    govern(_modify({"x": 99}))

    @tool
    def posonly(x, /):
        return x

    # A modify rewriting x->99 must apply (or, if unapplicable, fail open to
    # the original call). It must NOT raise TypeError into the caller.
    assert posonly(1) == 99


def test_modify_varargs_tool_does_not_crash_caller(govern):
    govern(_modify({"args": [1, 2, 3]}))

    @tool
    def variadic(*args):
        return sum(args)

    # Whatever the scrub semantics, the call must not explode.
    variadic(10)


def test_modify_method_does_not_crash_caller(govern):
    govern(_modify({"x": 7}))

    class C:
        @tool
        def m(self, x):
            return x

    assert C().m(1) == 7


def test_modify_mismatched_keys_falls_open_not_crash(govern):
    govern(_modify({"totally_unrelated_key": 1}))

    @tool
    def strict(a):
        return a

    # The scrub key doesn't fit the signature; we should fall back to the
    # original args (fail-open), not raise into the caller.
    assert strict(5) == 5


def test_modify_matching_keyword_args_still_works(govern):
    """Sanity guard: the happy path (kwarg-bindable signature) DOES apply."""
    govern(_modify({"a": 1, "b": 2}))

    @tool
    def add(a, b):
        return a + b

    assert add(9, 9) == 3  # 1 + 2, scrubbed


# ---------------------------------------------------------------------------
# BUG CLASS B: generators. A @tool on a (sync) generator returns the generator
# immediately, so post_tool_use audit fires BEFORE the body runs and audits a
# generator repr, never the yielded values. Root cause: tools.wrapper has no
# generator branch; inspect.iscoroutinefunction is False for gen funcs.
# Severity: CONCERN — output governance/scrub is silently absent for streaming
# tools, and the post event is misleading.
# ---------------------------------------------------------------------------


def test_generator_post_audit_sees_yielded_output_not_generator_repr(govern):
    fake = govern(Decision(action="allow"))

    @tool
    def stream():
        yield "secret-1"
        yield "secret-2"

    list(stream())  # drive the generator to completion
    post = [e for e in fake.events if e.event_type == EVENT_POST_TOOL_USE]
    assert len(post) == 1
    # Output governance can only scrub what it can see. The audited output must
    # reflect the yielded values, not "<generator object ...>".
    assert "generator object" not in json.dumps(post[0].tool_output)


def test_generator_governs_eagerly_and_audits_after_iteration(govern):
    """FIXED: governance (pre-event) runs EAGERLY at call time, but post-audit
    is deferred until the generator is exhausted — so the audit reflects a
    completed stream, not an un-iterated generator object."""
    fake = govern(Decision(action="allow"))
    side: list[str] = []

    @tool
    def stream():
        side.append("body-ran")
        yield 1

    gen = stream()
    # Pre fired at call; post NOT yet (body not iterated).
    assert any(e.event_type == EVENT_PRE_TOOL_USE for e in fake.events)
    assert not any(e.event_type == EVENT_POST_TOOL_USE for e in fake.events)
    assert side == []
    list(gen)
    assert side == ["body-ran"]
    post = [e for e in fake.events if e.event_type == EVENT_POST_TOOL_USE]
    assert len(post) == 1


def test_generator_deny_blocks_before_iteration(govern):
    """Positive guard: a DENY on a generator tool still blocks (the pre-event
    runs in the wrapper before the generator object is returned)."""
    govern(Decision(action="deny", reason="no"))

    @tool
    def stream():
        yield 1

    with pytest.raises(FirstOpsPolicyError):
        stream()


def test_async_generator_deny_blocks(govern):
    """Positive guard: async-generator tools route through the sync wrapper
    (iscoroutinefunction is False), but the pre-event still fires before the
    async_generator object is returned, so DENY blocks."""
    govern(Decision(action="deny", reason="no"))

    @tool
    async def astream():
        yield 1

    async def drive():
        async for _ in astream():
            pass

    with pytest.raises(FirstOpsPolicyError):
        asyncio.run(drive())


# ---------------------------------------------------------------------------
# BUG CLASS C: a tool that RAISES skips post-audit entirely. Root cause:
# tools.wrapper calls func() then _audit_post() sequentially with no
# try/finally; an exception short-circuits the audit. Severity: CONCERN —
# failed tool calls leave no post_tool_use audit trail.
# ---------------------------------------------------------------------------


def test_raising_tool_still_emits_post_audit(govern):
    fake = govern(Decision(action="allow"))

    @tool
    def boom():
        raise ValueError("kaboom")

    with pytest.raises(ValueError):
        boom()

    post = [e for e in fake.events if e.event_type == EVENT_POST_TOOL_USE]
    assert len(post) == 1, "failed tool call produced no audit event"


def test_each_call_costs_exactly_one_pre_and_one_post_roundtrip(govern):
    """Latency guard: post-audit is a second synchronous sentinel round-trip
    per call. This is by design but must be asserted so a future change that
    adds a third round-trip (or makes post async/batched) is a conscious one.
    """
    fake = govern(Decision(action="allow"))

    @tool
    def add(a, b):
        return a + b

    add(1, 2)
    assert [e.event_type for e in fake.events] == [
        EVENT_PRE_TOOL_USE,
        EVENT_POST_TOOL_USE,
    ]


# ---------------------------------------------------------------------------
# Non-bug behaviors worth pinning so a regression is caught.
# ---------------------------------------------------------------------------


def test_non_jsonable_arg_is_reprd_not_garbage(govern):
    """A non-JSON-able argument (an arbitrary object) must reach enforcement as
    a repr string, never raise and never leak a non-serializable value into the
    wire body."""
    fake = govern(Decision(action="allow"))

    class Obj:
        def __repr__(self):
            return "<Obj repr>"

    @tool
    def takes(o):
        return "ok"

    assert takes(Obj()) == "ok"
    pre = [e for e in fake.events if e.event_type == EVENT_PRE_TOOL_USE][0]
    assert pre.tool_input == {"o": "<Obj repr>"}
    # The wire body must be JSON-serializable.
    json.dumps(pre.tool_input)


def test_keyword_only_and_default_args_bind_correctly(govern):
    fake = govern(Decision(action="allow"))

    @tool
    def f(a, b=10, *, c=20):
        return a + b + c

    assert f(1) == 31
    pre = [e for e in fake.events if e.event_type == EVENT_PRE_TOOL_USE][0]
    assert pre.tool_input == {"a": 1, "b": 10, "c": 20}
