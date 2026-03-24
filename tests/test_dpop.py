"""Tests for DPoP proof generation — validates interoperability with the Go backend."""

import base64
import hashlib
import json

from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from firstops.dpop import (
    create_proof,
    jwk_thumbprint,
    load_private_key,
    public_key_jwk,
)


def _generate_pem() -> str:
    """Generate a test P-256 key and return PEM string."""
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()


def test_load_private_key():
    pem = _generate_pem()
    key = load_private_key(pem)
    assert isinstance(key, ec.EllipticCurvePrivateKey)
    assert isinstance(key.curve, ec.SECP256R1)


def test_public_key_jwk_fields():
    pem = _generate_pem()
    key = load_private_key(pem)
    jwk = public_key_jwk(key)
    assert jwk["kty"] == "EC"
    assert jwk["crv"] == "P-256"
    assert "x" in jwk
    assert "y" in jwk
    # x and y should be base64url-encoded 32-byte values
    x_bytes = base64.urlsafe_b64decode(jwk["x"] + "==")
    y_bytes = base64.urlsafe_b64decode(jwk["y"] + "==")
    assert len(x_bytes) == 32
    assert len(y_bytes) == 32


def test_thumbprint_determinism():
    pem = _generate_pem()
    key = load_private_key(pem)
    t1 = jwk_thumbprint(key)
    t2 = jwk_thumbprint(key)
    assert t1 == t2
    # Thumbprint is base64url-encoded SHA-256 (43 chars without padding)
    assert len(t1) == 43


def test_thumbprint_matches_go_algorithm():
    """Verify our thumbprint matches the canonical construction used by the Go backend."""
    pem = _generate_pem()
    key = load_private_key(pem)
    jwk = public_key_jwk(key)

    # Manually compute the same way as Go: {"crv":"...","kty":"EC","x":"...","y":"..."}
    canonical = f'{{"crv":"{jwk["crv"]}","kty":"{jwk["kty"]}","x":"{jwk["x"]}","y":"{jwk["y"]}"}}'
    digest = hashlib.sha256(canonical.encode()).digest()
    expected = base64.urlsafe_b64encode(digest).rstrip(b"=").decode()

    assert jwk_thumbprint(key) == expected


def test_create_proof_structure():
    pem = _generate_pem()
    key = load_private_key(pem)
    proof = create_proof(key, "POST", "https://api.firstops.ai/mcp/proxy/conn123")

    parts = proof.split(".")
    assert len(parts) == 3

    # Decode header
    header = json.loads(base64.urlsafe_b64decode(parts[0] + "=="))
    assert header["typ"] == "dpop+jwt"
    assert header["alg"] == "ES256"
    assert "jwk" in header
    assert header["jwk"]["kty"] == "EC"

    # Decode claims
    claims = json.loads(base64.urlsafe_b64decode(parts[1] + "=="))
    assert claims["htm"] == "POST"
    assert claims["htu"] == "https://api.firstops.ai/mcp/proxy/conn123"
    assert "jti" in claims
    assert "iat" in claims


def test_create_proof_signature_verifies():
    """Verify the proof's signature is valid — critical for backend acceptance."""
    pem = _generate_pem()
    key = load_private_key(pem)
    proof = create_proof(key, "GET", "https://example.com/test")

    parts = proof.split(".")
    signed_content = f"{parts[0]}.{parts[1]}"

    # Decode signature from IEEE P1363 to (r, s)
    sig_bytes = base64.urlsafe_b64decode(parts[2] + "==")
    assert len(sig_bytes) == 64
    r = int.from_bytes(sig_bytes[:32], "big")
    s = int.from_bytes(sig_bytes[32:], "big")

    # Verify with public key
    der_sig = utils.encode_dss_signature(r, s)
    pub = key.public_key()
    # This will raise InvalidSignature if verification fails
    pub.verify(der_sig, signed_content.encode(), ec.ECDSA(SHA256()))


def test_different_keys_different_thumbprints():
    key1 = load_private_key(_generate_pem())
    key2 = load_private_key(_generate_pem())
    assert jwk_thumbprint(key1) != jwk_thumbprint(key2)
