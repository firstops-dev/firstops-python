"""Tests for the action-event model and hookwire (de)serialization."""

import base64

from firstops.events import (
    CHANNEL_MCP,
    CHANNEL_SYSTEM_TOOLS,
    DECISION_ALLOW,
    DECISION_DENY,
    DECISION_MODIFY,
    EVENT_PRE_TOOL_USE,
    ActionEvent,
    Decision,
    MCPInfo,
)


def test_action_event_wire_shape_minimal():
    ev = ActionEvent(event_type=EVENT_PRE_TOOL_USE, tool_name="send_email")
    wire = ev.to_wire()
    # Always-present keys.
    assert wire["event_type"] == "pre_tool_use"
    assert wire["tool_name"] == "send_email"
    assert wire["agent"] == "firstops-sdk"
    # Optional keys omitted when unset.
    assert "tool_input" not in wire
    assert "channel" not in wire
    assert "mcp" not in wire


def test_action_event_preserves_nested_structure():
    # The whole point of sending tool_input as a JSON object (not a
    # stringified map) is that nested arrays/objects survive the wire.
    nested = {"to": "a@b.com", "body": {"lines": [1, 2, 3], "meta": {"x": True}}}
    ev = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE,
        tool_name="send_email",
        channel=CHANNEL_SYSTEM_TOOLS,
        tool_input=nested,
    )
    wire = ev.to_wire()
    assert wire["tool_input"] == nested
    assert wire["channel"] == "system_tools"


def test_action_event_mcp_info():
    ev = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE,
        tool_name="search",
        channel=CHANNEL_MCP,
        mcp=MCPInfo(server="notion", tool="search", url="https://mcp.notion.com/sse"),
    )
    assert ev.to_wire()["mcp"] == {
        "server": "notion",
        "tool": "search",
        "url": "https://mcp.notion.com/sse",
    }


def test_action_event_empty_mcp_omitted():
    ev = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE, tool_name="x", channel=CHANNEL_MCP, mcp=MCPInfo()
    )
    assert "mcp" not in ev.to_wire()


def test_decision_from_wire_allow():
    d = Decision.from_wire({"decision": "allow"})
    assert d.action == DECISION_ALLOW
    assert d.modified_payload is None
    assert not d.blocked
    assert not d.failed_open


def test_decision_from_wire_deny():
    d = Decision.from_wire({"decision": "deny", "reason": "blocked", "policy_id": "p1"})
    assert d.action == DECISION_DENY
    assert d.blocked
    assert d.reason == "blocked"
    assert d.policy_id == "p1"


def test_decision_from_wire_modify_base64():
    payload = b'{"to":"[REDACTED]"}'
    d = Decision.from_wire(
        {
            "decision": "modify",
            "modified_payload": base64.b64encode(payload).decode(),
            "policy_id": "pii-1",
        }
    )
    assert d.action == DECISION_MODIFY
    assert d.modified_payload == payload
    assert d.modified is True


def test_decision_from_wire_empty_defaults_to_allow():
    # A backend that returns {} (or a missing decision) must not crash and
    # must default to allow — never silently deny.
    d = Decision.from_wire({})
    assert d.action == DECISION_ALLOW


def test_decision_fail_open():
    d = Decision.fail_open("sentinel down")
    assert d.action == DECISION_ALLOW
    assert d.failed_open is True
    assert "sentinel down" in d.reason
