"""Tests for the enforcement client — decision handling + fail-open."""

import base64

import httpx
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


def _generate_pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


def _client(handler) -> EnforcementClient:
    identity = build_identity("agent-1", _generate_pem(), _GATEWAY)
    http = httpx.Client(transport=httpx.MockTransport(handler))
    return EnforcementClient(identity, http=http)


def _event() -> ActionEvent:
    return ActionEvent(
        event_type="pre_tool_use", tool_name="send_email", channel="system_tools"
    )


def _decode_dpop_claims(proof: str) -> dict:
    import base64
    import json

    claims_b64 = proof.split(".")[1]
    claims_b64 += "=" * (-len(claims_b64) % 4)  # restore padding
    return json.loads(base64.urlsafe_b64decode(claims_b64))


def test_evaluate_allow_and_headers():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["dpop"] = request.headers.get("DPoP")
        captured["auth"] = request.headers.get("Authorization")
        captured["path"] = request.url.path
        return httpx.Response(200, json={"decision": "allow"})

    d = _client(handler).evaluate(_event())
    assert d.action == "allow"
    assert not d.failed_open
    # DPoP-signed, agent bearer present, correct path.
    assert captured["dpop"] and len(captured["dpop"].split(".")) == 3
    assert captured["auth"] == "Bearer fo_agent_agent-1"
    assert captured["path"] == "/api/v1/daemon/evaluate-hook"


def test_dpop_htu_and_htm_are_byte_exact():
    # Regression guard for the gateway-host / htu class of bug: sentinel
    # validates htu by exact string match, so the signed htu MUST equal
    # gateway + path with no query and no host drift. A wrong default host
    # silently fails open — this test makes htu visible.
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["claims"] = _decode_dpop_claims(request.headers["DPoP"])
        return httpx.Response(200, json={"decision": "allow"})

    _client(handler).evaluate(_event())
    claims = captured["claims"]
    assert claims["htu"] == f"{_GATEWAY}/api/v1/daemon/evaluate-hook"
    assert claims["htm"] == "POST"


def test_unknown_decision_fails_open_visibly():
    # A verdict the SDK can't act on must allow (never block) but be flagged.
    def handler(request):
        return httpx.Response(200, json={"decision": "quarantine"})

    d = _client(handler).evaluate(_event())
    assert d.action == "allow"
    assert d.failed_open is True


def test_evaluate_deny():
    def handler(request):
        return httpx.Response(200, json={"decision": "deny", "reason": "policy"})

    d = _client(handler).evaluate(_event())
    assert d.blocked
    assert not d.failed_open


def test_evaluate_modify():
    payload = b'{"to":"[REDACTED]"}'

    def handler(request):
        return httpx.Response(
            200,
            json={
                "decision": "modify",
                "modified_payload": base64.b64encode(payload).decode(),
            },
        )

    d = _client(handler).evaluate(_event())
    assert d.action == "modify"
    assert d.modified_payload == payload


def test_fail_open_on_transport_error():
    def handler(request):
        raise httpx.ConnectError("connection refused")

    d = _client(handler).evaluate(_event())
    assert d.action == "allow"
    assert d.failed_open is True


def test_fail_open_on_5xx():
    def handler(request):
        return httpx.Response(500, text="boom")

    d = _client(handler).evaluate(_event())
    assert d.action == "allow"
    assert d.failed_open is True


def test_fail_open_on_401():
    # Auth failure must NOT block the agent — it surfaces as a fail-open allow.
    def handler(request):
        return httpx.Response(401, json={"error": "unregistered key"})

    d = _client(handler).evaluate(_event())
    assert d.action == "allow"
    assert d.failed_open is True


def test_fail_open_on_malformed_body():
    def handler(request):
        return httpx.Response(200, content=b"this is not json")

    d = _client(handler).evaluate(_event())
    assert d.action == "allow"
    assert d.failed_open is True


def test_evaluate_never_raises():
    # Defensive: even a totally broken transport must not raise into the agent.
    def handler(request):
        raise RuntimeError("unexpected")

    # RuntimeError is not an httpx.HTTPError — confirm it still doesn't escape.
    try:
        d = _client(handler).evaluate(_event())
    except Exception as e:  # pragma: no cover - this is the assertion
        raise AssertionError(f"evaluate() raised into caller: {e!r}")
    assert d.failed_open is True
