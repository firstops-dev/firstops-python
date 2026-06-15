"""M3 bug-hunt — adversarial tests for the request-path scrub capability flag
and coverage honesty.

Two kinds of tests live here:

1. Regression locks (passing): behavior that is correct today and must stay
   correct — the flag survives every decorator wrapper kind and every decision,
   and the OpenAI path never claims a scrub it cannot apply.

2. Bug pins (``xfail(strict=True)``): real holes found during the hunt. Each is
   a *failing* test that documents the defect; when the bug is fixed the test
   flips to a hard failure (strict xfail) so the fix can't land without
   un-xfailing it. Root cause + severity are in each docstring.

Reuses the ``_FakeEnforcement`` / ``_rt`` pattern from ``test_m3_scrub_coverage``.
"""

from __future__ import annotations

import asyncio
import base64
import json
from types import SimpleNamespace

import pytest

import firstops.tools as tools_mod
from firstops import coverage
from firstops.events import EVENT_PRE_TOOL_USE, ActionEvent, Decision
from firstops.integrations import openai_agents
from firstops.integrations._common import decide, govern_tool
from firstops.tools import FirstOpsPolicyError, _GOVERNED_TOOLS, governed_tool_names, tool


class _FakeEnforcement:
    def __init__(self, decision: Decision):
        self.decision = decision
        self.events = []

    def evaluate(self, event):
        self.events.append(event)
        if event.event_type != EVENT_PRE_TOOL_USE:
            return Decision(action="allow")
        return self.decision


def _rt(decision: Decision):
    return SimpleNamespace(enforcement=_FakeEnforcement(decision))


def _modify(new_input: dict) -> Decision:
    raw = base64.b64encode(json.dumps(new_input).encode()).decode()
    return Decision(action="modify", modified_payload=base64.b64decode(raw))


def _pre_events(rt):
    return [e for e in rt.enforcement.events if e.event_type == EVENT_PRE_TOOL_USE]


def _run_decorated_tool(kind: str, decision: Decision):
    """Decorate a tool of the given wrapper kind, drive it once under ``rt``,
    swallow a policy block, and return the runtime so the caller can inspect
    the emitted pre-event. ``kind`` ∈ {sync, async, gen, agen}."""
    rt = _rt(decision)
    orig = tools_mod._runtime.runtime
    tools_mod._runtime.runtime = lambda: rt
    try:
        if kind == "sync":

            @tool
            def f(x):
                return x

            try:
                f(1)
            except FirstOpsPolicyError:
                pass
        elif kind == "async":

            @tool
            async def f(x):
                return x

            try:
                asyncio.run(f(1))
            except FirstOpsPolicyError:
                pass
        elif kind == "gen":

            @tool
            def f(x):
                yield x

            try:
                list(f(1))
            except FirstOpsPolicyError:
                pass
        elif kind == "agen":

            @tool
            async def f(x):
                yield x

            async def drive():
                out = []
                async for i in f(1):
                    out.append(i)
                return out

            try:
                asyncio.run(drive())
            except FirstOpsPolicyError:
                pass
        else:  # pragma: no cover - test programming error
            raise AssertionError(f"unknown kind {kind!r}")
    finally:
        tools_mod._runtime.runtime = orig
    return rt


# ==========================================================================
# Regression locks — flag plumbing through every decorator wrapper (Invariant 1)
# ==========================================================================


@pytest.mark.parametrize("kind", ["sync", "async", "gen", "agen"])
@pytest.mark.parametrize("action", ["allow", "deny", "modify"])
def test_decorator_sets_flag_true_for_every_wrapper_and_decision(kind, action):
    """The decorator can always rebind args from a modify payload, so it MUST
    emit producer_can_apply_modify=True on the pre-event for *every* wrapper
    kind (sync/async/generator/async-generator) and regardless of the verdict
    sentinel returns. If a future refactor routes one wrapper around
    ``_evaluate_pre``, sentinel would escalate a scrub to deny for that surface
    only — a silent, surface-specific honesty regression. Pin all eight cells."""
    decision = _modify({"x": 9}) if action == "modify" else Decision(action=action, reason="r")
    rt = _run_decorated_tool(kind, decision)
    pre = _pre_events(rt)
    assert len(pre) == 1, f"{kind} emitted {len(pre)} pre-events"
    assert pre[0].producer_can_apply_modify is True


def test_govern_tool_flag_survives_modify_and_deny():
    """The forwarded flag is independent of the decision: a producer that CAN
    apply a modify advertises that up front, before sentinel has decided. Lock
    that govern_tool keeps the flag set on deny/modify, not just allow."""
    for decision in (Decision(action="deny", reason="r"), _modify({"a": 1})):
        rt = _rt(decision)
        govern_tool(rt, "t", {"a": 0}, can_apply_modify=True)
        assert _pre_events(rt)[0].producer_can_apply_modify is True


# ==========================================================================
# Regression lock — OpenAI honesty invariant (Invariant 2, the central one)
# ==========================================================================


@pytest.mark.parametrize(
    "decision",
    [
        Decision(action="allow"),
        Decision(action="deny", reason="r"),
        pytest.param(None, id="modify"),  # built below to avoid shared mutable default
    ],
)
def test_openai_never_claims_modify_capability(decision):
    """OpenAI guardrails are read-only. If the OpenAI path ever set the flag
    True, sentinel would ship a `modify` the guardrail silently drops →
    unscrubbed data proceeds. This must hold across allow/deny/modify verdicts,
    because the flag is set BEFORE the verdict is known — a verdict-dependent
    leak would be the worst kind (only triggers in production on a real scrub)."""
    if decision is None:
        decision = _modify({"x": 1})
    rt = _rt(decision)
    openai_agents._decide_tool(rt, "send_email", {"to": "a@b.com"})
    assert _pre_events(rt)[0].producer_can_apply_modify is False


def test_common_decide_default_leaves_flag_false():
    """The _common.decide default (can_apply_modify=False) is what OpenAI relies
    on. A change to that default would silently flip OpenAI's honesty."""
    rt = _rt(Decision(action="allow"))
    decide(rt, "t", {})
    assert _pre_events(rt)[0].producer_can_apply_modify is False


# ==========================================================================
# Regression lock — flag emitted on the wire ONLY when True (omitempty parity)
# ==========================================================================


def test_wire_omits_flag_when_false_present_when_true():
    on = ActionEvent(event_type=EVENT_PRE_TOOL_USE, tool_name="t", producer_can_apply_modify=True)
    off = ActionEvent(event_type=EVENT_PRE_TOOL_USE, tool_name="t")
    assert on.to_wire()["producer_can_apply_modify"] is True
    assert "producer_can_apply_modify" not in off.to_wire()


def test_capability_returns_defensive_copy():
    """capability() must hand back a copy — a caller mutating the returned dict
    must not corrupt the shared CAPABILITY_MATRIX (which feeds every other
    coverage answer in the process)."""
    d = coverage.capability("openai_agents")
    d["scrub"] = True
    assert coverage.capability("openai_agents")["scrub"] is False


# ==========================================================================
# BUG PINS — coverage honesty holes (Invariant 3: must not overclaim or crash)
# ==========================================================================


def test_coverage_does_not_crash_on_non_string_declared_entries():
    """ROOT CAUSE: ungoverned_tools/coverage_report do
    ``sorted(set(declared) - governed)``. If ``declared`` contains mixed types
    (e.g. a tool registry that yields ints/None for malformed entries), the
    set-difference keeps the non-strings and ``sorted`` raises TypeError
    comparing int<None. A coverage call that raises is itself a silent gap —
    the caller gets no coverage answer at all.

    SEVERITY: MEDIUM. Coverage runs at agent-setup time, not inside the agent
    loop, so it won't deny a live call — but Invariant 3 explicitly forbids a
    crash on odd input, and a tool registry feeding non-string names is a
    realistic source. Expected fix: coerce to str (or skip/str-key the sort)
    so the call returns a well-defined report instead of raising."""
    out = coverage.coverage_report(["real_tool", 123, None])
    # Once fixed, the string entry should still be reported and no crash occur.
    assert "real_tool" in out["ungoverned"]


@pytest.mark.xfail(
    strict=True,
    reason="BUG: _GOVERNED_TOOLS is a process-global set with no reset; a tool "
    "decorated anywhere in the process makes coverage OVERCLAIM it as governed "
    "for an unrelated agent's declared list. Invariant 3: must not overclaim. "
    "Severity: HIGH.",
)
def test_governed_set_does_not_overclaim_across_agents():
    """ROOT CAUSE: ``@firstops.tool`` does ``_GOVERNED_TOOLS.add(name)`` into a
    module-global set and never clears it. ``governed_tool_names()`` is the only
    source of truth for coverage, and it has no notion of *which* agent/graph a
    tool belongs to. So if agent A (a different module, a prior import, or an
    earlier run in the same process) decorated a tool whose name collides with a
    name in agent B's declared list, coverage reports it GOVERNED for B even
    though B never wired any FirstOps governance for it.

    This is the precise failure the coverage module exists to prevent: a SILENT
    coverage gap dressed up as coverage. A security reviewer auditing agent B
    would see ``charge_card`` in the 'governed' column and move on, while in
    B's process that tool runs ungoverned.

    SEVERITY: HIGH. Coverage honesty is the whole point of the module; an
    overclaim is worse than a missing claim. The fix needs agent/registration
    scoping (e.g. coverage takes the *governed names this agent registered*,
    not a process-global), or a documented reset + per-agent snapshot. Until
    then this test pins the contamination.
    """
    _GOVERNED_TOOLS.clear()

    # Agent A — somewhere else entirely — governs a tool named "charge_card".
    @tool
    def charge_card():  # noqa: D401 - belongs to a *different* agent
        return "A"

    # Agent B declares charge_card but NEVER decorated it in B's scope.
    # B's true coverage of charge_card is zero.
    report = coverage.coverage_report(["charge_card", "send_refund"])

    # Honest answer: B governs neither, so charge_card must NOT be claimed.
    assert report["governed"] == [], (
        "charge_card claimed governed for an agent that never governed it "
        f"(cross-contamination via process-global set): {report}"
    )
    assert coverage.ungoverned_tools(["charge_card", "send_refund"]) == [
        "charge_card",
        "send_refund",
    ]


def test_custom_named_tool_not_reported_as_false_gap():
    """ROOT CAUSE: ``@tool(name="alias")`` adds ``"alias"`` to _GOVERNED_TOOLS,
    not the function's __name__. A developer enumerating their tools by function
    name (the natural ``declared`` list) will pass ``real_fn`` while the
    governed set only knows ``alias`` — so ``real_fn`` is reported ungoverned
    even though it IS governed.

    This is the inverse honesty failure: coverage cries wolf. Less dangerous
    than an overclaim (it under-claims, which is fail-safe for security), but it
    erodes trust in the coverage report — and a report nobody trusts gets
    ignored, which reintroduces the silent-gap risk the module fights.

    SEVERITY: LOW-MEDIUM. Expected fix: register both the __name__ and the
    custom name, or have coverage reconcile on the same key the developer
    declares. This test pins the false gap until then."""
    _GOVERNED_TOOLS.clear()

    @tool(name="custom_alias")
    def real_fn():
        return 1

    # Declared by the natural python name; it IS governed, so no gap expected.
    assert coverage.ungoverned_tools(["real_fn"]) == []


# ==========================================================================
# Coverage edge cases that ARE handled correctly — lock them so a "fix" for the
# bugs above doesn't regress these.
# ==========================================================================


def test_coverage_empty_declared_is_empty_report():
    _GOVERNED_TOOLS.clear()
    assert coverage.coverage_report([]) == {"governed": [], "ungoverned": []}
    assert coverage.ungoverned_tools([]) == []


def test_coverage_deduplicates_declared():
    _GOVERNED_TOOLS.clear()
    assert coverage.coverage_report(["b", "b", "b"]) == {
        "governed": [],
        "ungoverned": ["b"],
    }


def test_capability_unknown_and_case_sensitive():
    assert coverage.capability("does_not_exist") == {}
    # Case-sensitive by design (matches the surface keys exactly).
    assert coverage.capability("Claude") == {}
    assert coverage.capability("claude")["scrub"] is True
