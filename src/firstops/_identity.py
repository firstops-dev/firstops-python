"""Shared agent identity — the loaded key, DPoP signer, and gateway URL.

Established once per process and shared by the MCP sidecar proxy and the
enforcement (EvaluateHook) client, so a single agent identity backs every
signed request the SDK makes. Key material never leaves this object.
"""

from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import urlparse

from cryptography.hazmat.primitives.asymmetric import ec

from firstops.dpop import create_proof, jwk_thumbprint, load_private_key


@dataclass
class Identity:
    """An agent's signing identity. Construct via :func:`build_identity`."""

    agent_id: str
    gateway_url: str  # normalized, no trailing slash
    _key: ec.EllipticCurvePrivateKey

    @property
    def bearer_token(self) -> str:
        return f"fo_agent_{self.agent_id}"

    @property
    def jkt(self) -> str:
        """RFC 7638 JWK thumbprint of this identity's key."""
        return jwk_thumbprint(self._key)

    def proof(self, method: str, url: str) -> str:
        """Create a DPoP proof (RFC 9449) for an HTTP method + target URL.

        The htu claim is the URL with query and fragment stripped, matching the
        gateway's **byte-exact** htu binding. We strip via string slicing (not a
        urlparse round-trip) so we do NOT alter scheme/host casing — any
        normalization the SDK applies that sentinel does not would 401 every
        proof, which `evaluate()` silently converts to a fail-open allow.
        """
        htu = url.split("#", 1)[0].split("?", 1)[0]
        return create_proof(self._key, method, htu)


def build_identity(agent_id: str, private_key_pem: str, gateway_url: str) -> Identity:
    """Load the key and validate the gateway URL, returning an Identity.

    Raises:
        ValueError: if ``gateway_url`` is malformed or the key is not P-256.
    """
    gateway = gateway_url.rstrip("/")
    parsed = urlparse(gateway)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid gateway_url: {gateway_url}")
    key = load_private_key(private_key_pem)
    return Identity(agent_id=agent_id, gateway_url=gateway, _key=key)
