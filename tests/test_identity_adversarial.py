"""Adversarial tests for Identity / DPoP htu canonicalization (M0.T0.2).

The sentinel validates htu by **exact byte-for-byte string compare** (no
normalization):

    backend/shared/lib/dpop/validate.go:132  ->  if htu != expectedURL { ... }
    expectedURL = h.config.BaseURL + "/api/v1/daemon/evaluate-hook"

So whatever the SDK signs as htu MUST equal the operator's configured BaseURL
+ path, verbatim. A divergence does NOT error loudly — it 401s, and the
enforcement client converts a 401 into a fail-open *allow*. The agent then runs
completely ungoverned while the dashboard says "protected." That is the
existential silent-fail-open failure mode the design doc calls out, so these
tests probe the canonicalization byte-for-byte.

These are designed to FAIL where Identity.proof normalizes inconsistently.
"""

from __future__ import annotations

import base64
import json

import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from firstops._identity import build_identity

_HOOK_PATH = "/api/v1/daemon/evaluate-hook"


def _pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


def _htu(proof: str) -> str:
    claims_b64 = proof.split(".")[1]
    claims_b64 += "=" * (-len(claims_b64) % 4)
    return json.loads(base64.urlsafe_b64decode(claims_b64))["htu"]


def _htm(proof: str) -> str:
    claims_b64 = proof.split(".")[1]
    claims_b64 += "=" * (-len(claims_b64) % 4)
    return json.loads(base64.urlsafe_b64decode(claims_b64))["htm"]


# What sentinel would compute given the same gateway BaseURL the operator set.
def _sentinel_expected_htu(base_url: str) -> str:
    # Sentinel does a plain concat: config.BaseURL + path. No normalization.
    return base_url.rstrip("/") + _HOOK_PATH


def test_htu_strips_query_and_fragment():
    """Query + fragment MUST be stripped (sentinel's BaseURL+path has neither)."""
    ident = build_identity("a", _pem(), "https://h.example.com")
    url = ident.gateway_url + _HOOK_PATH + "?nonce=abc#frag"
    proof = ident.proof("POST", url)
    assert _htu(proof) == "https://h.example.com" + _HOOK_PATH
    assert "?" not in _htu(proof)
    assert "#" not in _htu(proof)


def test_htm_is_post_uppercase():
    ident = build_identity("a", _pem(), "https://h.example.com")
    proof = ident.proof("POST", ident.gateway_url + _HOOK_PATH)
    assert _htm(proof) == "POST"


def test_htu_with_trailing_slash_gateway_matches_sentinel():
    """A trailing slash on gateway_url must not produce a double slash in htu."""
    base = "https://h.example.com/"
    ident = build_identity("a", _pem(), base)
    proof = ident.proof("POST", ident.gateway_url + _HOOK_PATH)
    assert _htu(proof) == _sentinel_expected_htu(base)


def test_htu_with_port_matches_sentinel():
    base = "https://h.example.com:8443"
    ident = build_identity("a", _pem(), base)
    proof = ident.proof("POST", ident.gateway_url + _HOOK_PATH)
    assert _htu(proof) == _sentinel_expected_htu(base)


def test_htu_does_not_alter_host_casing():
    """HOLE PROBE: urlunparse lowercases the *scheme* but NOT the *host*.

    If an operator's SENTINEL_PROXY_BASE_URL is 'https://H.Example.com' (mixed
    case — uncommon but legal), sentinel computes
    'https://H.Example.com/api/v1/daemon/evaluate-hook' and compares byte-for-
    byte. The SDK must sign the SAME bytes. Since proof() must NOT introduce
    normalization sentinel doesn't do, the signed htu host must equal whatever
    the caller put in gateway_url, character for character.

    This asserts the *contract*: htu == (operator's base_url, verbatim) + path.
    It fails today because urlunparse lowercases the scheme, so for a base_url
    with an uppercase scheme the SDK's htu diverges from a plain concat.
    """
    base = "HTTPS://H.Example.com"
    ident = build_identity("a", _pem(), base)
    # The enforcement client signs over identity.gateway_url + path.
    signed = _htu(ident.proof("POST", ident.gateway_url + _HOOK_PATH))
    # Sentinel compares against operator BaseURL + path, byte-exact.
    expected = _sentinel_expected_htu(base)
    assert signed == expected, (
        f"htu divergence => silent 401 => fail-open allow.\n"
        f"  SDK signed : {signed!r}\n"
        f"  sentinel   : {expected!r}\n"
        f"  Identity.proof must not normalize what sentinel does not."
    )


def test_path_prefixed_gateway_does_not_double_the_prefix():
    """HOLE PROBE: gateway_url with a path prefix (e.g. https://h/api).

    EnforcementClient builds its URL as gateway_url + '/api/v1/daemon/...'.
    If gateway_url already ends in '/api', the POST target AND the htu become
    '.../api/api/v1/daemon/evaluate-hook' (doubled). Whether that is a bug
    depends on the deployment, but the htu the SDK signs must equal the URL it
    actually POSTs to AND equal sentinel's expected. This test documents the
    behavior so a path-prefixed gateway can't silently 404/401.
    """
    base = "https://h.example.com/api"
    ident = build_identity("a", _pem(), base)
    post_target = ident.gateway_url + _HOOK_PATH
    signed = _htu(ident.proof("POST", post_target))
    # The htu must at least match the URL we actually POST to (self-consistency).
    assert signed == post_target, (
        f"signed htu {signed!r} != POST target {post_target!r}; "
        f"htu/URL self-inconsistency is an automatic 401."
    )


def test_proof_htu_matches_enforcement_client_post_url_exactly():
    """End-to-end self-consistency: the htu the client signs equals the URL it
    POSTs to. This is the invariant that protects against *any* normalization
    drift between EnforcementClient._url and Identity.proof, regardless of how
    weird the gateway_url is.
    """
    from firstops.enforcement import EnforcementClient

    for base in [
        "https://h.example.com",
        "https://h.example.com/",
        "https://h.example.com:8443",
        "https://h.example.com/api",
    ]:
        ident = build_identity("a", _pem(), base)
        client = EnforcementClient(ident)
        # The URL the client will POST to (private, but load-bearing).
        post_url = client._url
        signed = _htu(ident.proof("POST", post_url))
        assert signed == post_url, (
            f"base={base!r}: signed htu {signed!r} != POST url {post_url!r}"
        )
        client.close()
