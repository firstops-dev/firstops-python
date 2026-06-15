"""Bug-hunting tests for the M2 harness-adapter governance core.

These probe the framework-FREE logic the adapters share (``_common``) and the
pure decision functions in each adapter, against the four stated invariants:

  1. Adapters fail open: no runtime / enforcement weirdness must not raise or
     block into the agent loop (except a real DENY).
  2. ``decide`` returns exactly ("allow"|"deny"|"modify", payload). A malformed
     modify payload must degrade to allow, never a broken modify.
  3. The Claude hook output must be a valid shape for allow ({}), deny, and
     modify — and must NEVER emit a modify envelope with null/empty updatedInput.
  4. OpenAI ``_decide_tool``: modify is not representable → degrade to
     (False, "") allow, never block.

Tests that currently FAIL mark a real defect; each is annotated with
``xfail(strict=True)`` and a BUG-ID so the suite stays green until the code is
fixed, at which point the xfail flips to an unexpected-pass and forces removal
of the marker. Search "BUG-" to find them.
"""

import base64
import json
from types import SimpleNamespace

import pytest

from firstops.events import (
    CHANNEL_MCP,
    CHANNEL_SYSTEM_TOOLS,
    ActionEvent,
    Decision,
)
from firstops.integrations import claude, openai_agents
from firstops.integrations._common import (
    coerce_input,
    decide,
    govern_tool,
    modified_input,
)


# --------------------------------------------------------------------------
# Test doubles (mirror the baseline fixtures in test_integrations.py)
# --------------------------------------------------------------------------


class _FakeEnforcement:
    def __init__(self, decision: Decision):
        self.decision = decision
        self.events = []

    def evaluate(self, event):
        self.events.append(event)
        return self.decision


def _rt(decision: Decision):
    return SimpleNamespace(enforcement=_FakeEnforcement(decision))


def _modify(payload) -> Decision:
    """A modify Decision whose base64 payload encodes ``payload`` (any JSON type)."""
    raw = base64.b64encode(json.dumps(payload).encode()).decode()
    return Decision.from_wire({"decision": "modify", "modified_payload": raw})


# ==========================================================================
# coerce_input — adversarial inputs (Invariant 1: must never raise)
# ==========================================================================


def test_coerce_input_utf8_bytes_roundtrips():
    assert coerce_input(b"hello") == {"input": "hello"}


def test_coerce_input_json_bytes_parses_to_dict():
    assert coerce_input(b'{"a": 1}') == {"a": 1}


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("[1,2,3]", {"input": "[1,2,3]"}),  # JSON list string -> not a dict, wrap raw
        ("42", {"input": "42"}),  # JSON scalar string -> wrap raw
        ("null", {"input": "null"}),
        ("true", {"input": "true"}),
        ("", {"input": ""}),  # empty string
        ({}, {}),  # empty dict
        ("unicode: café ☃", {"input": "unicode: café ☃"}),
    ],
)
def test_coerce_input_string_scalars_and_unicode(raw, expected):
    assert coerce_input(raw) == expected


def test_coerce_input_nested_dict_returned_as_is():
    nested = {"a": {"b": [1, 2, {"c": 3}]}}
    assert coerce_input(nested) == nested


def test_coerce_input_large_input_does_not_choke():
    big = "x" * 1_000_000
    assert coerce_input(big) == {"input": big}


def test_coerce_input_non_utf8_bytes_must_not_raise():
    # Should degrade gracefully, not raise.
    out = coerce_input(b"\xff\xfe\x00")
    assert isinstance(out, dict)
    assert "input" in out


def test_coerce_input_produces_json_serializable_value():
    class Weird:
        def __str__(self):
            return "weird"

    coerced = coerce_input(Weird())
    # The body must be serializable, because to_wire() will json.dumps it.
    json.dumps(coerced)  # must not raise


# ==========================================================================
# modified_input — malformed modify payloads degrade to None (Invariant 2)
# ==========================================================================


def test_modified_input_dict_payload_returned():
    assert modified_input(_modify({"x": 9})) == {"x": 9}


def test_modified_input_list_payload_is_none():
    # A modify whose payload is a JSON list is not a usable input dict.
    d = _modify([1, 2, 3])
    assert modified_input(d) is None


@pytest.mark.parametrize("scalar", [42, "string", 3.14, True, None])
def test_modified_input_scalar_payload_is_none(scalar):
    assert modified_input(_modify(scalar)) is None


def test_modified_input_empty_dict_is_a_dict():
    # PIN THE CHOICE: an empty-dict {} payload IS a dict, so modified_input
    # returns it, and decide() therefore reports ("modify", {}). Whether {} is
    # a *meaningful* scrub is the adapter's problem (see BUG-2 for Claude).
    assert modified_input(_modify({})) == {}


def test_modified_input_none_when_not_modify():
    assert modified_input(Decision(action="allow")) is None
    assert modified_input(Decision(action="deny", reason="x")) is None


# ==========================================================================
# decide — exact (action, payload) contract (Invariant 2)
# ==========================================================================


def test_decide_allow():
    assert decide(_rt(Decision(action="allow")), "t", {}) == ("allow", None)


def test_decide_deny_with_reason():
    assert decide(_rt(Decision(action="deny", reason="no")), "t", {}) == ("deny", "no")


def test_decide_deny_empty_reason_gets_fallback():
    action, reason = decide(_rt(Decision(action="deny", reason="")), "t", {})
    assert action == "deny"
    assert reason == "blocked by FirstOps policy"  # never an empty deny reason


def test_decide_modify_dict():
    assert decide(_rt(_modify({"x": 1})), "t", {}) == ("modify", {"x": 1})


def test_decide_modify_list_degrades_to_allow():
    # Invariant 2: a malformed modify must NOT become a broken modify.
    assert decide(_rt(_modify([1, 2, 3])), "t", {}) == ("allow", None)


def test_decide_modify_scalar_degrades_to_allow():
    assert decide(_rt(_modify("just a string")), "t", {}) == ("allow", None)


def test_decide_no_runtime_allows():
    assert decide(None, "t", {}) == ("allow", None)


# ==========================================================================
# govern_tool — event construction, channel, tool_name edge cases
# ==========================================================================


def test_govern_tool_system_tools_channel():
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, "send_email", {"to": "a@b.com"})
    ev = rt.enforcement.events[0]
    assert ev.channel == CHANNEL_SYSTEM_TOOLS
    assert ev.mcp is None


def test_govern_tool_mcp_channel_and_metadata():
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, "mcp__notion__search", {"q": "x"})
    ev = rt.enforcement.events[0]
    assert ev.channel == CHANNEL_MCP
    assert ev.mcp.server == "notion"
    assert ev.mcp.tool == "search"


def test_govern_tool_malformed_mcp_name_is_system_tools():
    # classify() and mcp_info() must agree: a malformed mcp__ name is NOT mcp.
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, "mcp__foo", {})
    ev = rt.enforcement.events[0]
    assert ev.channel == CHANNEL_SYSTEM_TOOLS
    assert ev.mcp is None


def test_govern_tool_empty_tool_name_does_not_raise():
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, "", {})
    assert rt.enforcement.events[0].channel == CHANNEL_SYSTEM_TOOLS


def test_govern_tool_unicode_tool_name():
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, "send_\U0001f4e7", {})
    assert rt.enforcement.events[0].tool_name == "send_\U0001f4e7"


def test_govern_tool_failed_open_allow_flows_through_as_allow():
    # An enforcement that fails open (allow + failed_open) must read as allow.
    rt = _rt(Decision.fail_open("sentinel unreachable"))
    assert decide(rt, "t", {}) == ("allow", None)


def test_govern_tool_none_tool_name_must_not_raise():
    rt = _rt(Decision(action="allow"))
    govern_tool(rt, None, {})  # must not raise
    assert rt.enforcement.events[0].channel == CHANNEL_SYSTEM_TOOLS


def test_govern_tool_does_not_alias_caller_input_dict():
    rt = _rt(Decision(action="allow"))
    caller_input = {"to": "a@b.com"}
    govern_tool(rt, "send_email", caller_input)
    ev = rt.enforcement.events[0]
    # Mutating the caller dict after the fact must not change the audited event.
    caller_input["to"] = "attacker@evil.com"
    assert ev.tool_input == {"to": "a@b.com"}


# ==========================================================================
# Claude adapter — exact hook envelope shape (Invariant 3)
# ==========================================================================


def test_claude_allow_returns_empty_dict():
    rt = _rt(Decision(action="allow"))
    out = claude._govern_pre_tool_use(rt, {"tool_name": "Bash", "tool_input": {"command": "ls"}})
    assert out == {}


def test_claude_deny_envelope_exact_shape():
    rt = _rt(Decision(action="deny", reason="destructive command rule"))
    out = claude._govern_pre_tool_use(rt, {"tool_name": "Bash", "tool_input": {"command": "wipe"}})
    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "deny",
            "permissionDecisionReason": "destructive command rule",
        }
    }


def test_claude_deny_empty_reason_never_emits_blank_reason():
    rt = _rt(Decision(action="deny", reason=""))
    out = claude._govern_pre_tool_use(rt, {"tool_name": "Bash", "tool_input": {}})
    reason = out["hookSpecificOutput"]["permissionDecisionReason"]
    assert reason  # not "" and not None
    assert reason == "blocked by FirstOps policy"


def test_claude_modify_envelope_exact_shape():
    rt = _rt(_modify({"command": "ls -la"}))
    out = claude._govern_pre_tool_use(rt, {"tool_name": "Bash", "tool_input": {"command": "ls"}})
    assert out == {
        "hookSpecificOutput": {
            "hookEventName": "PreToolUse",
            "permissionDecision": "allow",
            "updatedInput": {"command": "ls -la"},
        }
    }


def test_claude_modify_list_payload_degrades_to_allow_empty():
    # A modify whose payload is a list is unusable -> allow ({}), not a broken
    # modify envelope.
    rt = _rt(_modify([1, 2, 3]))
    out = claude._govern_pre_tool_use(rt, {"tool_name": "Bash", "tool_input": {}})
    assert out == {}


def test_claude_missing_tool_name_allows():
    rt = _rt(Decision(action="allow"))
    assert claude._govern_pre_tool_use(rt, {"tool_input": {}}) == {}


def test_claude_missing_tool_input_still_governs():
    rt = _rt(Decision(action="deny", reason="r"))
    out = claude._govern_pre_tool_use(rt, {"tool_name": "Bash"})
    assert out["hookSpecificOutput"]["permissionDecision"] == "deny"


def test_claude_tool_input_json_string_is_parsed_for_event():
    rt = _rt(Decision(action="allow"))
    claude._govern_pre_tool_use(rt, {"tool_name": "Bash", "tool_input": '{"command": "ls"}'})
    # The event body should carry the parsed object, not the raw string.
    assert rt.enforcement.events[0].tool_input == {"command": "ls"}


def test_claude_empty_dict_modify_must_not_emit_empty_updated_input():
    rt = _rt(_modify({}))
    out = claude._govern_pre_tool_use(rt, {"tool_name": "Bash", "tool_input": {"command": "ls"}})
    hso = out.get("hookSpecificOutput")
    if hso is not None:
        # If anything is emitted, it must not be a modify with empty input.
        assert not (hso.get("permissionDecision") == "allow" and hso.get("updatedInput") == {})


# ==========================================================================
# OpenAI Agents adapter — modify is not representable (Invariant 4)
# ==========================================================================


def test_openai_deny_blocks_with_reason():
    assert openai_agents._decide_tool(_rt(Decision(action="deny", reason="x")), "t", {}) == (True, "x")


def test_openai_allow():
    assert openai_agents._decide_tool(_rt(Decision(action="allow")), "t", {}) == (False, "")


def test_openai_modify_blocks_fail_closed():
    # Guardrails can't mutate args, so a scrub-required (modify) decision must
    # FAIL CLOSED (block) rather than let unscrubbed args through.
    blocked, reason = openai_agents._decide_tool(_rt(_modify({"x": 1})), "t", {})
    assert blocked is True
    assert "scrub" in reason


def test_openai_modify_empty_dict_also_blocks():
    blocked, _ = openai_agents._decide_tool(_rt(_modify({})), "t", {})
    assert blocked is True


def test_openai_deny_reason_always_string():
    # Even if the upstream reason is a non-string, _decide_tool must return a str.
    d = Decision(action="deny")
    d.reason = 12345  # type: ignore[assignment]
    blocked, reason = openai_agents._decide_tool(_rt(d), "t", {})
    assert blocked is True
    assert isinstance(reason, str)


def test_openai_deny_empty_reason_gets_fallback_string():
    blocked, reason = openai_agents._decide_tool(_rt(Decision(action="deny", reason="")), "t", {})
    assert blocked is True
    assert reason == "blocked by FirstOps policy"


# ==========================================================================
# Cross-adapter consistency: the SAME decision must be honored consistently
# ==========================================================================


def test_deny_is_honored_by_both_adapters():
    d = Decision(action="deny", reason="pii rule")
    claude_out = claude._govern_pre_tool_use(_rt(d), {"tool_name": "t", "tool_input": {}})
    openai_out = openai_agents._decide_tool(_rt(d), "t", {})
    assert claude_out["hookSpecificOutput"]["permissionDecision"] == "deny"
    assert openai_out[0] is True


def test_modify_is_applied_by_claude_but_blocked_by_openai():
    # Same scrub decision: Claude rewrites the args (updatedInput); OpenAI can't
    # mutate, so it fails closed (block) — never a silent unscrubbed allow.
    d = _modify({"x": 1})
    claude_out = claude._govern_pre_tool_use(_rt(d), {"tool_name": "t", "tool_input": {}})
    openai_out = openai_agents._decide_tool(_rt(d), "t", {})
    assert claude_out["hookSpecificOutput"].get("updatedInput") == {"x": 1}
    assert openai_out[0] is True  # OpenAI can't mutate -> block


# ==========================================================================
# Double governance — documentation test (decorator + adapter both fire)
# ==========================================================================


def test_double_governance_emits_pre_event_twice(monkeypatch):
    """A tool that is BOTH @firstops.tool-decorated AND governed by an adapter
    is evaluated TWICE: once by the adapter at the harness execution boundary,
    and once by the decorator inside the call. This is the documented behavior
    (the adapter and the base decorator are independent governance layers).

    Pinning it here so a future de-dup effort is a deliberate, test-visible
    change rather than an accidental one.
    """
    import firstops
    import firstops._runtime as runtime_mod

    shared = _rt(Decision(action="allow"))
    monkeypatch.setattr(runtime_mod, "runtime", lambda: shared)

    @firstops.tool
    def send_email(to):
        return "sent"

    # Adapter governs at the boundary (one pre_tool_use):
    decide(shared, "send_email", {"to": "a@b.com"})
    # Then the decorated tool actually runs (decorator governs inside):
    send_email("a@b.com")

    pre = [e for e in shared.enforcement.events if e.event_type == "pre_tool_use"]
    # Two pre_tool_use evaluations for one logical call: adapter + decorator.
    assert len(pre) == 2
