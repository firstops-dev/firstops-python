"""Adversarial fail-open matrix for enforcement.py (M0.T0.4).

Invariant #1 (the security spine): evaluate() must NEVER raise into the caller
and NEVER block. Every failure path — transport error, connect vs read timeout,
non-200, 3xx redirect, malformed/empty/non-JSON body, decision:null, unknown
verdict, an exception from event.to_wire(), and any unexpected exception — must
resolve to a fail-open ALLOW (audited). Auth (DPoP) is the only fail-closed
surface and even it surfaces here as a fail-open allow.

These tests try to make evaluate() block, raise, or silently deny.
"""

from __future__ import annotations

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from firstops._identity import build_identity
from firstops.enforcement import EnforcementClient
from firstops.events import ActionEvent

_GATEWAY = "https://gw.example.com"


def _pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


def _client(handler) -> EnforcementClient:
    ident = build_identity("a", _pem(), _GATEWAY)
    return EnforcementClient(ident, http=httpx.Client(transport=httpx.MockTransport(handler)))


def _event() -> ActionEvent:
    return ActionEvent(event_type="pre_tool_use", tool_name="t", channel="system_tools")


def _assert_fail_open(d):
    assert d.action == "allow", f"expected fail-open allow, got {d.action!r}"
    assert d.failed_open is True
    assert not d.blocked


# ---------------------------------------------------------------------------
# transport / timeout
# ---------------------------------------------------------------------------


def test_connect_timeout_fails_open():
    def handler(req):
        raise httpx.ConnectTimeout("connect timed out")

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_read_timeout_fails_open():
    def handler(req):
        raise httpx.ReadTimeout("read timed out")

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_pool_timeout_fails_open():
    def handler(req):
        raise httpx.PoolTimeout("pool exhausted")

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_connect_error_fails_open():
    def handler(req):
        raise httpx.ConnectError("connection refused")

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_unexpected_non_httpx_exception_fails_open():
    """A bug-class exception (not an httpx error) must still be swallowed."""
    def handler(req):
        raise RuntimeError("totally unexpected")

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_keyboard_interrupt_is_NOT_swallowed():
    """Control-flow exceptions (KeyboardInterrupt/SystemExit) must propagate.

    `except Exception` does not catch BaseException — verify the agent can
    still be Ctrl-C'd mid-evaluate. If someone widens the except to
    `BaseException`, this catches the regression.
    """
    def handler(req):
        raise KeyboardInterrupt()

    with pytest.raises(KeyboardInterrupt):
        _client(handler).evaluate(_event())


# ---------------------------------------------------------------------------
# HTTP status codes
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("code", [301, 302, 307, 308])
def test_redirect_fails_open_and_is_not_followed(code):
    """A 3xx must fail open, NOT be followed.

    follow_redirects defaults to False, so a redirect is a non-200 → fail-open.
    Following it would risk taking a 'decision' from an attacker-controlled
    Location, or SSRF. This pins both: fail-open AND no follow.
    """
    hits = {"n": 0}

    def handler(req):
        hits["n"] += 1
        return httpx.Response(code, headers={"Location": "https://evil.example.com/x"})

    _assert_fail_open(_client(handler).evaluate(_event()))
    assert hits["n"] == 1, "redirect was followed — must not be"


@pytest.mark.parametrize("code", [400, 401, 403, 404, 429, 500, 502, 503, 504])
def test_error_statuses_fail_open(code):
    def handler(req):
        return httpx.Response(code, json={"error": "x"})

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_204_no_content_fails_open():
    """A 200 is required; any other 2xx (e.g. 204 with no body) fails open."""
    def handler(req):
        return httpx.Response(204)

    _assert_fail_open(_client(handler).evaluate(_event()))


# ---------------------------------------------------------------------------
# 200 with pathological bodies
# ---------------------------------------------------------------------------


def test_200_empty_object_is_clean_allow():
    """200 {} → allow, but NOT failed_open (missing decision == allow, #4)."""
    def handler(req):
        return httpx.Response(200, json={})

    d = _client(handler).evaluate(_event())
    assert d.action == "allow"
    assert d.failed_open is False


def test_200_non_json_body_fails_open():
    def handler(req):
        return httpx.Response(200, content=b"<html>not json</html>")

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_200_json_array_not_object_fails_open():
    """A JSON array (not an object) → .get() raises AttributeError → fail-open."""
    def handler(req):
        return httpx.Response(200, json=[1, 2, 3])

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_200_decision_null_is_allow():
    def handler(req):
        return httpx.Response(200, json={"decision": None})

    d = _client(handler).evaluate(_event())
    assert d.action == "allow"


def test_200_unknown_verdict_fails_open_visibly():
    def handler(req):
        return httpx.Response(200, json={"decision": "quarantine"})

    _assert_fail_open(_client(handler).evaluate(_event()))


def test_200_modify_with_invalid_base64_fails_open_not_raises():
    """HOLE GUARD: Decision.from_wire raises on bad base64. Inside evaluate()
    that raise MUST be caught and converted to fail-open — it must never reach
    the agent. (Decision.from_wire itself still raises for direct callers; see
    test_events_adversarial.test_from_wire_invalid_base64_payload_raises.)
    """
    def handler(req):
        return httpx.Response(
            200, json={"decision": "modify", "modified_payload": "!!!bad!!!"}
        )

    # Must not raise; must fail open.
    d = _client(handler).evaluate(_event())
    _assert_fail_open(d)


def test_200_giant_binary_modified_payload_decodes():
    import base64

    blob = bytes(range(256)) * 8192  # 2 MiB

    def handler(req):
        return httpx.Response(
            200,
            json={"decision": "modify", "modified_payload": base64.b64encode(blob).decode()},
        )

    d = _client(handler).evaluate(_event())
    assert d.action == "modify"
    assert d.modified_payload == blob


# ---------------------------------------------------------------------------
# event-side failures
# ---------------------------------------------------------------------------


def test_exception_in_to_wire_fails_open():
    """If event.to_wire() raises (a malformed event), evaluate must fail open,
    not propagate the error into the agent's tool-call path.
    """
    class BadEvent(ActionEvent):
        def to_wire(self):  # type: ignore[override]
            raise ValueError("boom in to_wire")

    bad = BadEvent(event_type="pre_tool_use", tool_name="t")

    def handler(req):
        return httpx.Response(200, json={"decision": "deny"})

    _assert_fail_open(_client(handler).evaluate(bad))


# ---------------------------------------------------------------------------
# happy-path decisions still work
# ---------------------------------------------------------------------------


def test_deny_is_not_failed_open():
    def handler(req):
        return httpx.Response(200, json={"decision": "deny", "reason": "policy"})

    d = _client(handler).evaluate(_event())
    assert d.blocked
    assert d.failed_open is False


def test_evaluate_truly_never_raises_across_matrix():
    """Belt-and-suspenders: sweep a battery of broken handlers; none may raise."""
    import base64

    # '!!!bad!!!' is genuinely invalid base64 (9 chars) → from_wire raises →
    # evaluate must catch it and fail open. Use that, not a length-valid blob.
    handlers = [
        lambda r: (_ for _ in ()).throw(httpx.ConnectError("x")),
        lambda r: httpx.Response(500),
        lambda r: httpx.Response(200, content=b"garbage"),
        lambda r: httpx.Response(200, json=[1, 2]),
        lambda r: httpx.Response(200, json={"decision": "modify", "modified_payload": "!!!bad!!!"}),
        lambda r: httpx.Response(302, headers={"Location": "/x"}),
        lambda r: httpx.Response(200, json={"decision": None}),
    ]
    for h in handlers:
        try:
            d = _client(h).evaluate(_event())
        except Exception as e:  # pragma: no cover
            raise AssertionError(f"evaluate() raised for handler {h}: {e!r}")
        # Every one of these is a failure or a no-decision → must allow.
        assert d.action == "allow", f"handler {h} returned {d.action!r}, not allow"
