"""Adversarial lifecycle tests for _runtime.py (M0.T0.5).

The runtime keeps process-global state (_runtime) and delegates idempotency +
the agent-mismatch guard to proxy.init. These tests probe the seams:
double-init with different params, init→shutdown→init, shutdown-without-init,
port-already-bound, re-key, and module-global leakage between tests.

A fixture force-resets the global between tests so a leak in one test can't
mask a bug in another (and so these tests stay Isolated).
"""

from __future__ import annotations

import socket

import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from firstops import _runtime, proxy
from firstops.dpop import jwk_thumbprint

# Unreachable gateway — init binds only a local listener, never connects out.
_GATEWAY = "http://127.0.0.1:1"


def _pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


@pytest.fixture(autouse=True)
def _clean_runtime():
    """Guarantee a clean global before AND after each test (Isolated)."""
    _runtime.shutdown()
    yield
    _runtime.shutdown()


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return port


# ---------------------------------------------------------------------------
# basic lifecycle
# ---------------------------------------------------------------------------


def test_shutdown_without_init_is_noop():
    # autouse fixture already called shutdown(); call again explicitly.
    _runtime.shutdown()
    assert _runtime.runtime() is None
    assert proxy.is_running() is False


def test_init_then_shutdown_then_init_again_is_fresh_runtime():
    port = _free_port()
    pem = _pem()
    rt1 = _runtime.init("agent-1", pem, port=port, gateway_url=_GATEWAY)
    _runtime.shutdown()
    assert _runtime.runtime() is None
    rt2 = _runtime.init("agent-1", pem, port=port, gateway_url=_GATEWAY)
    assert rt2 is not rt1, "re-init after shutdown must build a fresh Runtime"
    assert _runtime.runtime() is rt2


# ---------------------------------------------------------------------------
# mismatch guard — different identity while running
# ---------------------------------------------------------------------------


def test_double_init_different_agent_raises_and_leaves_runtime_intact():
    port = _free_port()
    pem = _pem()
    rt = _runtime.init("agent-1", pem, port=port, gateway_url=_GATEWAY)
    with pytest.raises(RuntimeError):
        _runtime.init("agent-2", pem, port=port, gateway_url=_GATEWAY)
    # The original runtime + proxy identity must be untouched by the failed call.
    assert _runtime.runtime() is rt
    assert proxy.current_agent_id() == "agent-1"


def test_double_init_different_gateway_raises():
    port = _free_port()
    pem = _pem()
    _runtime.init("agent-1", pem, port=port, gateway_url=_GATEWAY)
    with pytest.raises(RuntimeError):
        _runtime.init("agent-1", pem, port=port, gateway_url="http://127.0.0.1:2")
    assert proxy.current_agent_id() == "agent-1"


def test_double_init_different_port_raises():
    port = _free_port()
    pem = _pem()
    _runtime.init("agent-1", pem, port=port, gateway_url=_GATEWAY)
    with pytest.raises(RuntimeError):
        _runtime.init("agent-1", pem, port=_free_port(), gateway_url=_GATEWAY)


def test_failed_mismatch_init_does_not_leak_a_second_runtime():
    """After a rejected mismatching init, exactly one runtime exists and it is
    the original. Guards against 'partial init left _runtime pointing at a new
    object whose proxy never started'.
    """
    port = _free_port()
    pem = _pem()
    rt = _runtime.init("agent-1", pem, port=port, gateway_url=_GATEWAY)
    for bad in [("agent-x", port, _GATEWAY), ("agent-1", port, "http://127.0.0.1:9")]:
        with pytest.raises(RuntimeError):
            _runtime.init(bad[0], pem, port=bad[1], gateway_url=bad[2])
    assert _runtime.runtime() is rt


# ---------------------------------------------------------------------------
# idempotency edge cases
# ---------------------------------------------------------------------------


def test_reinit_same_params_is_idempotent_same_object():
    port = _free_port()
    pem = _pem()
    rt1 = _runtime.init("agent-1", pem, port=port, gateway_url=_GATEWAY)
    rt2 = _runtime.init("agent-1", pem, port=port, gateway_url=_GATEWAY)
    assert rt1 is rt2


def test_reinit_same_params_different_key_raises():
    """FIXED: re-init with same agent/port/gateway but a DIFFERENT private key
    now RAISES rather than silently keeping the stale key (which would 401 and
    fail open invisibly). A re-key must be explicit: shutdown() then init().
    """
    port = _free_port()
    pem1, pem2 = _pem(), _pem()
    rt1 = _runtime.init("agent-1", pem1, port=port, gateway_url=_GATEWAY)
    first_jkt = jwk_thumbprint(rt1.identity._key)
    with pytest.raises(RuntimeError, match="different key"):
        _runtime.init("agent-1", pem2, port=port, gateway_url=_GATEWAY)
    # Original runtime intact, still the first key.
    assert _runtime.runtime() is rt1
    assert jwk_thumbprint(_runtime.runtime().identity._key) == first_jkt


def test_runtime_gateway_url_is_normalized_no_trailing_slash():
    port = _free_port()
    rt = _runtime.init("agent-1", _pem(), port=port, gateway_url=_GATEWAY + "/")
    assert rt.identity.gateway_url == _GATEWAY  # trailing slash stripped


# ---------------------------------------------------------------------------
# resource / binding failures
# ---------------------------------------------------------------------------


def test_init_when_port_already_bound_propagates_and_does_not_set_runtime():
    """If the listener port is already taken by something else, init() must
    fail (OSError from bind) and must NOT leave a half-built runtime behind.
    """
    port = _free_port()
    blocker = socket.socket()
    blocker.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    blocker.bind(("127.0.0.1", port))
    blocker.listen(1)
    try:
        with pytest.raises(OSError):
            _runtime.init("agent-1", _pem(), port=port, gateway_url=_GATEWAY)
        # The failed bind must not have set the global runtime.
        assert _runtime.runtime() is None
        assert proxy.is_running() is False
    finally:
        blocker.close()


def test_invalid_gateway_url_raises_value_error_and_no_runtime():
    with pytest.raises(ValueError):
        _runtime.init("agent-1", _pem(), port=_free_port(), gateway_url="not-a-url")
    assert _runtime.runtime() is None


def test_invalid_pem_raises_and_no_runtime():
    with pytest.raises(Exception):
        _runtime.init("agent-1", "-----BEGIN nonsense-----", port=_free_port(), gateway_url=_GATEWAY)
    assert _runtime.runtime() is None
    assert proxy.is_running() is False


# ---------------------------------------------------------------------------
# global leakage guard
# ---------------------------------------------------------------------------


def test_runtime_global_is_isolated_between_tests():
    """If a previous test leaked the global, this fails — proving the fixture
    actually isolates. (Also a canary if someone removes the fixture.)
    """
    assert _runtime.runtime() is None
    assert proxy.is_running() is False
