"""SDK ↔ Go hookwire contract parity (M0.T0.3 / design §I.1, the blocking DoR).

This is the SDK half of the golden-fixture parity pinned on the Go side in
backend/shared/lib/hookwire/sdk_contract_test.go. The two halves together
guarantee the wire contract cannot drift undetected:

  - Go test:  decodes the SDK's request JSON into EvaluateHookRequestJSON and
              asserts ToProto preserves nested structure.
  - This test: decodes the EXACT bytes Go's encoding/json produces for an
              EvaluateHookResponseJSON into Decision.from_wire and asserts the
              base64 modified_payload round-trips.

GO_RESPONSE_JSON below is captured VERBATIM from:
  json.Marshal(hookwire.EvaluateHookResponseJSON{
      Decision:"modify", Reason:"PII detected",
      ModifiedPayload:[]byte(`{"to":"[REDACTED]"}`), PolicyID:"pii-1"})

If the Go response shape changes, regenerate this constant and the Go fixture
in lockstep — a divergence between them is the contract break we're guarding.
"""

from __future__ import annotations

import json

from firstops.events import (
    DECISION_ALLOW,
    DECISION_MODIFY,
    ActionEvent,
    Decision,
    MCPInfo,
)

# Verbatim output of Go's json.Marshal on the response struct.
GO_RESPONSE_JSON = (
    '{"decision":"modify","reason":"PII detected",'
    '"modified_payload":"eyJ0byI6IltSRURBQ1RFRF0ifQ==","policy_id":"pii-1"}'
)


def test_sdk_decodes_go_response_bytes_exactly():
    """The SDK must decode the real Go-marshaled response: base64 string →
    raw bytes, all four fields populated.
    """
    d = Decision.from_wire(json.loads(GO_RESPONSE_JSON))
    assert d.action == DECISION_MODIFY
    assert d.reason == "PII detected"
    assert d.policy_id == "pii-1"
    assert d.modified_payload == b'{"to":"[REDACTED]"}'
    assert d.modified is True


def test_go_allow_response_decodes_clean():
    """A minimal Go allow response (omitempty drops everything but decision)."""
    d = Decision.from_wire(json.loads('{"decision":"allow"}'))
    assert d.action == DECISION_ALLOW
    assert d.modified_payload is None
    assert not d.failed_open


def test_sdk_request_field_names_match_go_struct_tags():
    """Pin the request JSON keys the SDK emits against the Go struct's json
    tags. If the SDK renames a field, sentinel's ShouldBindJSON silently drops
    it (omitempty) and governance data is lost. Hard-code the contract.
    """
    ev = ActionEvent(
        event_type="pre_tool_use",
        tool_name="send_email",
        channel="system_tools",
        tool_input={"x": 1},
        tool_output={"y": 2},
        session_id="s",
        cwd="/c",
        cli_version="1.0",
        mcp=MCPInfo(server="srv", tool="t", url="u"),
    )
    wire = ev.to_wire()
    # These are the exact json tags on hookwire.EvaluateHookRequestJSON.
    go_tags = {
        "event_type",
        "agent",
        "tool_name",
        "tool_input",
        "tool_output",
        "session_id",
        "cwd",
        "channel",
        "mcp",
        "cli_version",
    }
    unknown = set(wire.keys()) - go_tags
    assert not unknown, f"SDK emits keys the Go struct ignores: {unknown}"
    # And the mcp sub-object keys match hookwire.MCPInfo tags.
    assert set(wire["mcp"].keys()) <= {"server", "tool", "url"}


def test_agent_label_matches_normalizer_trigger():
    """The SDK's default agent label MUST equal the string the Go normalizer
    matches to assign SourceSDK (normalizer.go: req.Agent == "firstops-sdk").
    If these drift, every SDK action mis-attributes to hook_claude_code.
    """
    ev = ActionEvent(event_type="pre_tool_use", tool_name="x")
    assert ev.to_wire()["agent"] == "firstops-sdk"
