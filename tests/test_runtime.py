"""Tests for the process runtime wiring (init / shutdown / idempotency)."""

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from firstops import _runtime
from firstops.enforcement import EnforcementClient

# Unreachable gateway — init() must not connect anywhere; the sidecar only
# binds a local listener. A high port avoids collisions with other suites.
_GATEWAY = "http://127.0.0.1:1"
_PORT = 19911


def _generate_pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


def test_init_builds_runtime_and_shutdown_clears_it():
    pem = _generate_pem()
    rt = _runtime.init("agent-1", pem, port=_PORT, gateway_url=_GATEWAY)
    try:
        assert rt.identity.agent_id == "agent-1"
        assert rt.identity.gateway_url == _GATEWAY  # normalized, no trailing slash
        assert isinstance(rt.enforcement, EnforcementClient)
        assert _runtime.runtime() is rt
    finally:
        _runtime.shutdown()
    assert _runtime.runtime() is None


def test_init_is_idempotent_for_same_identity():
    pem = _generate_pem()
    rt1 = _runtime.init("agent-1", pem, port=_PORT, gateway_url=_GATEWAY)
    try:
        rt2 = _runtime.init("agent-1", pem, port=_PORT, gateway_url=_GATEWAY)
        assert rt1 is rt2
    finally:
        _runtime.shutdown()


def test_shutdown_is_idempotent():
    # No init() — shutdown() must be a safe no-op.
    _runtime.shutdown()
    assert _runtime.runtime() is None
