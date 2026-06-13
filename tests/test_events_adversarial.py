"""Adversarial tests for events.py — wire fidelity + Decision parsing (M0.T0.3).

Probes the gaps between what the SDK emits/parses and what the backend
hookwire contract expects. The backend reference shapes:

  request : backend/shared/lib/hookwire/types.go EvaluateHookRequestJSON
            (tool_input/tool_output are `map[string]any` with `omitempty`)
  response: EvaluateHookResponseJSON {decision, reason, modified_payload([]byte),
            policy_id}; Go marshals []byte as a base64 string.

Designed to FIND BUGS, not to pass.
"""

from __future__ import annotations

import base64

import pytest

from firstops.events import (
    DECISION_ALLOW,
    DECISION_MODIFY,
    EVENT_PRE_TOOL_USE,
    ActionEvent,
    Decision,
)


# ---------------------------------------------------------------------------
# tool_input / tool_output fidelity
# ---------------------------------------------------------------------------


def test_tool_output_round_trips_nested():
    """tool_output must survive as a JSON object (symmetric with tool_input)."""
    out = {"messages": [{"role": "assistant", "content": "hi"}], "n": 3, "ok": True}
    ev = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE, tool_name="x", tool_output=out
    )
    assert ev.to_wire()["tool_output"] == out


def test_empty_dict_tool_input_is_emitted_not_omitted():
    """An explicitly-empty tool_input ({}) is DISTINCT from absent (None).

    The SDK emits `tool_input: {}` when the value is {} (only None omits it).
    The backend struct has `omitempty` on the map, so Go would *omit* an empty
    map on its own marshal — but the SDK is the *producer* here, and sentinel's
    ShouldBindJSON accepts an explicit empty object. This test pins the SDK's
    behavior so a future 'optimize away empty dicts' change is caught: {} and
    None must remain distinguishable on the wire.
    """
    ev_empty = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE, tool_name="x", tool_input={}
    )
    ev_none = ActionEvent(event_type=EVENT_PRE_TOOL_USE, tool_name="x")
    assert "tool_input" in ev_empty.to_wire(), "{} must be emitted, not dropped"
    assert ev_empty.to_wire()["tool_input"] == {}
    assert "tool_input" not in ev_none.to_wire(), "None must be omitted"


def test_falsy_string_fields_are_omitted_consistently():
    """session_id='', cwd='', channel='' are omitted (truthiness gate).

    Documents the current behavior: empty strings are omitted via `if self.x:`.
    This matches the backend's `omitempty` on those string fields, so it is
    correct — but it means you cannot send an *explicit* empty channel to mean
    'match all' distinctly from 'unset'. Pin it so the behavior is intentional.
    """
    ev = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE,
        tool_name="x",
        session_id="",
        cwd="",
        channel="",
    )
    wire = ev.to_wire()
    assert "session_id" not in wire
    assert "cwd" not in wire
    assert "channel" not in wire


def test_unicode_and_mixed_types_in_tool_input_preserved():
    payload = {
        "emoji": "🔥",
        "unicode": "café—dash",
        "nested_list": [1, 2, [3, {"deep": True}]],
        "int": 42,
        "float": 3.14,
        "bool": False,
        "null": None,
    }
    ev = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE, tool_name="x", tool_input=payload
    )
    # The SDK must NOT stringify or coerce — it passes the object through.
    assert ev.to_wire()["tool_input"] == payload


def test_none_value_inside_tool_input_is_kept():
    """A None *inside* the dict is a value (kept), distinct from a None dict."""
    ev = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE, tool_name="x", tool_input={"k": None}
    )
    wire = ev.to_wire()
    assert "tool_input" in wire
    assert wire["tool_input"] == {"k": None}


# ---------------------------------------------------------------------------
# Decision.from_wire — fail-open & base64 edge cases
# ---------------------------------------------------------------------------


def test_from_wire_invalid_base64_payload_fails_open_not_raises():
    """FIXED (was: raised binascii.Error). A `modify` verdict whose
    modified_payload is not valid base64 must NOT raise out of from_wire — it
    fails open visibly so direct callers (adapters, contract tests) never see
    an exception. Invariant #1: never raise into the caller.
    """
    d = Decision.from_wire(
        {"decision": "modify", "modified_payload": "!!!not-base64!!!"}
    )
    assert d.action == DECISION_ALLOW
    assert d.failed_open is True
    assert d.modified_payload is None


def test_from_wire_garbage_base64_is_rejected_not_silently_decoded():
    """FIXED (was: silent coercion to wrong bytes). With validate=True a
    corrupted modified_payload is rejected and the decision fails open, rather
    than producing silently-wrong scrub bytes applied to a live call.
    """
    d = Decision.from_wire({"decision": "modify", "modified_payload": "!!!!"})
    assert d.action == DECISION_ALLOW
    assert d.failed_open is True
    assert d.modified_payload is None


def test_from_wire_modify_with_no_payload_fails_open():
    """FIXED (was: action='modify' with no payload — signals disagreed). A
    `modify` verdict with nothing to apply is malformed: fail open visibly
    rather than report a modify the caller can't act on.
    """
    d = Decision.from_wire({"decision": "modify"})
    assert d.action == DECISION_ALLOW
    assert d.modified is False
    assert d.modified_payload is None
    assert d.failed_open is True


def test_from_wire_decision_null_defaults_to_allow():
    """decision: null → allow (not silent deny). Per invariant #4, null≈missing.

    Documents that a null verdict is indistinguishable from a clean allow
    (failed_open stays False). If the team wants null flagged as failed_open,
    this test must change.
    """
    d = Decision.from_wire({"decision": None})
    assert d.action == DECISION_ALLOW
    assert d.failed_open is False


def test_from_wire_empty_object_is_allow():
    d = Decision.from_wire({})
    assert d.action == DECISION_ALLOW
    assert d.failed_open is False


def test_from_wire_unknown_verdict_fails_open_visibly():
    d = Decision.from_wire({"decision": "quarantine", "reason": "weird"})
    assert d.action == DECISION_ALLOW
    assert d.failed_open is True
    assert "quarantine" in d.reason


def test_from_wire_empty_string_decision_defaults_to_allow():
    """decision: '' (empty string) → allow via the `or` short-circuit."""
    d = Decision.from_wire({"decision": ""})
    assert d.action == DECISION_ALLOW
    assert d.failed_open is False


def test_from_wire_large_binary_payload_round_trips():
    """A gigantic / binary modified_payload must decode intact."""
    blob = bytes(range(256)) * 4096  # 1 MiB of all byte values
    d = Decision.from_wire(
        {"decision": "modify", "modified_payload": base64.b64encode(blob).decode()}
    )
    assert d.modified_payload == blob


def test_from_wire_bytes_payload_passthrough():
    """If modified_payload arrives as raw bytes (not str), it is used as-is."""
    raw = b"already-bytes"
    d = Decision.from_wire({"decision": "modify", "modified_payload": raw})
    assert d.modified_payload == raw
