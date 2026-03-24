"""DPoP proof generation (RFC 9449) using ES256 (P-256)."""

import base64
import hashlib
import json
import os
import time

from cryptography.hazmat.primitives.asymmetric import ec, utils
from cryptography.hazmat.primitives.hashes import SHA256
from cryptography.hazmat.primitives.serialization import load_pem_private_key


def load_private_key(pem_data: str) -> ec.EllipticCurvePrivateKey:
    """Load an EC P-256 private key from PEM string."""
    key = load_pem_private_key(pem_data.encode(), password=None)
    if not isinstance(key, ec.EllipticCurvePrivateKey):
        raise ValueError("expected EC private key")
    if not isinstance(key.curve, ec.SECP256R1):
        raise ValueError("expected P-256 curve")
    return key


def public_key_jwk(key: ec.EllipticCurvePrivateKey) -> dict[str, str]:
    """Return the JWK representation of the public key."""
    pub = key.public_key()
    numbers = pub.public_numbers()
    return {
        "kty": "EC",
        "crv": "P-256",
        "x": _b64url_int(numbers.x, 32),
        "y": _b64url_int(numbers.y, 32),
    }


def jwk_thumbprint(key: ec.EllipticCurvePrivateKey) -> str:
    """Compute RFC 7638 JWK thumbprint (base64url SHA-256)."""
    jwk = public_key_jwk(key)
    # RFC 7638: members in lexicographic order for EC: crv, kty, x, y
    canonical = f'{{"crv":"{jwk["crv"]}","kty":"{jwk["kty"]}","x":"{jwk["x"]}","y":"{jwk["y"]}"}}'
    digest = hashlib.sha256(canonical.encode()).digest()
    return base64.urlsafe_b64encode(digest).rstrip(b"=").decode()


def create_proof(key: ec.EllipticCurvePrivateKey, method: str, url: str) -> str:
    """Create a DPoP proof JWT for the given HTTP method and URL."""
    jwk = public_key_jwk(key)

    header = {"typ": "dpop+jwt", "alg": "ES256", "jwk": jwk}
    claims = {
        "jti": base64.urlsafe_b64encode(os.urandom(16)).rstrip(b"=").decode(),
        "htm": method,
        "htu": url,
        "iat": int(time.time()),
    }

    header_b64 = _b64url_json(header)
    claims_b64 = _b64url_json(claims)
    signed_content = f"{header_b64}.{claims_b64}"

    # Sign with ES256 — cryptography hashes internally
    der_sig = key.sign(signed_content.encode(), ec.ECDSA(SHA256()))
    # Convert DER to IEEE P1363 (fixed 64 bytes)
    r, s = utils.decode_dss_signature(der_sig)
    sig_bytes = r.to_bytes(32, "big") + s.to_bytes(32, "big")
    sig_b64 = base64.urlsafe_b64encode(sig_bytes).rstrip(b"=").decode()

    return f"{signed_content}.{sig_b64}"


def _b64url_json(obj: dict) -> str:
    data = json.dumps(obj, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(data).rstrip(b"=").decode()


def _b64url_int(n: int, size: int) -> str:
    b = n.to_bytes(size, "big")
    return base64.urlsafe_b64encode(b).rstrip(b"=").decode()
