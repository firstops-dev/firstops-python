"""Tests for the harness adapters' framework-free governance core.

The frameworks themselves aren't installed here, so we test the decision logic
and the exact native outputs the adapters emit, plus that the lazily-imported
shells fail cleanly when the framework is absent.
"""

import base64
import json
from types import SimpleNamespace

import pytest

from firstops.events import (
    CHANNEL_MCP,
    CHANNEL_SYSTEM_TOOLS,
    EVENT_PRE_TOOL_USE,
    Decision,
)
from firstops.integrations import _common  # noqa: F401  (kept for import-sanity)
from firstops.integrations._common import (
    coerce_input,
    decide,
    govern_tool,
    modified_input,
)


class _FakeEnforcement:
    def __init__(self, decision: Decision):
        self.decision = decision
        self.events = []

    def evaluate(self, event):
        self.events.append(event)
        return self.decision


def _rt(decision: Decision):
    return SimpleNamespace(enforcement=_FakeEnforcement(decision))


def _modify(payload: dict) -> Decision:
    raw = base64.b64encode(json.dumps(payload).encode()).decode()
    return Decision.from_wire({"decision": "modify", "modified_payload": raw})


# ---- _common ---------------------------------------------------------------


def test_govern_tool_builds_system_tools_event():
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, "send_email", {"to": "a@b.com"})
    ev = rt.enforcement.events[0]
    assert ev.event_type == EVENT_PRE_TOOL_USE
    assert ev.tool_name == "send_email"
    assert ev.channel == CHANNEL_SYSTEM_TOOLS
    assert ev.tool_input == {"to": "a@b.com"}
    assert ev.mcp is None


def test_govern_tool_builds_mcp_event_with_metadata():
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, "mcp__notion__search", {"q": "x"})
    ev = rt.enforcement.events[0]
    assert ev.channel == CHANNEL_MCP
    assert ev.mcp is not None
    assert ev.mcp.server == "notion"
    assert ev.mcp.tool == "search"


def test_govern_tool_allows_when_no_runtime():
    d = govern_tool(None, "anything", {})
    assert d.action == "allow"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ({"a": 1}, {"a": 1}),
        ('{"a": 1}', {"a": 1}),
        ("plain", {"input": "plain"}),
        (None, {}),
        (123, {"input": 123}),
    ],
)
def test_coerce_input(raw, expected):
    assert coerce_input(raw) == expected


def test_modified_input_roundtrip():
    assert modified_input(_modify({"x": 9})) == {"x": 9}


def test_modified_input_none_when_not_modify():
    assert modified_input(Decision(action="allow")) is None


def test_decide_allow_deny_modify():
    assert decide(_rt(Decision(action="allow")), "t", {}) == ("allow", None)
    assert decide(_rt(Decision(action="deny", reason="no")), "t", {}) == ("deny", "no")
    assert decide(_rt(_modify({"x": 1})), "t", {}) == ("modify", {"x": 1})


# ---- Claude adapter --------------------------------------------------------

from firstops.integrations import claude


def test_claude_allow_returns_empty():
    rt = _rt(Decision(action="allow"))
    assert (
        claude._govern_pre_tool_use(
            rt, {"tool_name": "Bash", "tool_input": {"command": "ls"}}
        )
        == {}
    )


def test_claude_deny_envelope():
    rt = _rt(Decision(action="deny", reason="destructive command rule"))
    out = claude._govern_pre_tool_use(
        rt, {"tool_name": "Bash", "tool_input": {"command": "danger"}}
    )
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "deny"
    assert "destructive" in hso["permissionDecisionReason"]


def test_claude_modify_updates_input():
    rt = _rt(_modify({"command": "ls -la"}))
    out = claude._govern_pre_tool_use(
        rt, {"tool_name": "Bash", "tool_input": {"command": "ls"}}
    )
    hso = out["hookSpecificOutput"]
    assert hso["permissionDecision"] == "allow"
    assert hso["updatedInput"] == {"command": "ls -la"}


def test_claude_hooks_requires_sdk():
    with pytest.raises(RuntimeError, match="claude-agent-sdk"):
        claude.firstops_hooks(_rt(Decision(action="allow")))


# ---- OpenAI Agents adapter -------------------------------------------------

from firstops.integrations import openai_agents


def test_openai_decide_block():
    assert openai_agents._decide_tool(
        _rt(Decision(action="deny", reason="x")), "t", {}
    ) == (True, "x")


def test_openai_decide_allow():
    assert openai_agents._decide_tool(_rt(Decision(action="allow")), "t", {}) == (
        False,
        "",
    )


def test_openai_modify_blocks_fail_closed():
    # Guardrails can't mutate args, so a scrub-required (modify) decision fails
    # CLOSED (block) rather than letting unscrubbed args through.
    blocked, reason = openai_agents._decide_tool(_rt(_modify({"x": 1})), "t", {})
    assert blocked is True
    assert "scrub" in reason


def test_openai_guardrail_requires_sdk():
    with pytest.raises(RuntimeError, match="openai-agents"):
        openai_agents.firstops_tool_input_guardrail(_rt(Decision(action="allow")))


# ---- LangGraph adapter -----------------------------------------------------

from firstops.integrations import langgraph


def test_langgraph_middleware_requires_langchain():
    with pytest.raises(RuntimeError, match="langchain"):
        langgraph.FirstOpsMiddleware(_rt(Decision(action="allow")))
