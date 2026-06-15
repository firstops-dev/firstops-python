"""M3 tests — request-path scrub capability flag + coverage honesty."""

from types import SimpleNamespace

import firstops._runtime as rt_mod
from firstops import coverage
from firstops.events import EVENT_PRE_TOOL_USE, ActionEvent, Decision
from firstops.integrations import claude, langgraph, openai_agents
from firstops.integrations._common import decide, govern_tool
from firstops.tools import _GOVERNED_TOOLS, tool


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


# ---- the capability flag on the wire --------------------------------------


def test_event_emits_flag_only_when_set():
    on = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE,
        tool_name="t",
        producer_can_apply_modify=True,
    )
    assert on.to_wire()["producer_can_apply_modify"] is True
    off = ActionEvent(event_type=EVENT_PRE_TOOL_USE, tool_name="t")
    assert "producer_can_apply_modify" not in off.to_wire()


def test_govern_tool_forwards_flag():
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, "t", {}, can_apply_modify=True)
    assert rt.enforcement.events[0].producer_can_apply_modify is True
    rt2 = _rt(Decision(action="allow"))
    govern_tool(rt2, "t", {})
    assert rt2.enforcement.events[0].producer_can_apply_modify is False


def test_decorator_sets_flag_true():
    rt = _rt(Decision(action="allow"))
    import firstops.tools as tools_mod

    orig = tools_mod._runtime.runtime
    tools_mod._runtime.runtime = lambda: rt
    try:

        @tool
        def f(x):
            return x

        f(1)
    finally:
        tools_mod._runtime.runtime = orig
    pre = [e for e in rt.enforcement.events if e.event_type == EVENT_PRE_TOOL_USE][0]
    assert pre.producer_can_apply_modify is True


def test_claude_adapter_sets_flag_true():
    rt = _rt(Decision(action="allow"))
    claude._govern_pre_tool_use(rt, {"tool_name": "Bash", "tool_input": {"c": "ls"}})
    assert rt.enforcement.events[0].producer_can_apply_modify is True


def test_openai_adapter_leaves_flag_false():
    # Guardrails can't mutate args → must NOT claim it can apply a modify.
    rt = _rt(Decision(action="allow"))
    openai_agents._decide_tool(rt, "t", {})
    assert rt.enforcement.events[0].producer_can_apply_modify is False


# ---- coverage honesty ------------------------------------------------------


def test_capability_matrix_reflects_jaggedness():
    assert coverage.capability("openai_agents")["scrub"] is False
    assert coverage.capability("claude")["scrub"] is True
    assert coverage.capability("base_decorator") == {
        "block": True,
        "scrub": True,
        "audit": True,
    }
    assert coverage.capability("unknown") == {}


def test_ungoverned_tools_detects_gap():
    _GOVERNED_TOOLS.clear()

    @tool
    def governed_one():
        return 1

    gaps = coverage.ungoverned_tools(["governed_one", "forgot_to_decorate"])
    assert gaps == ["forgot_to_decorate"]


def test_coverage_report_split():
    _GOVERNED_TOOLS.clear()

    @tool
    def a():
        return 1

    report = coverage.coverage_report(["a", "b", "c"])
    assert report["governed"] == ["a"]
    assert report["ungoverned"] == ["b", "c"]
