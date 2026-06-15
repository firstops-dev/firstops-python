"""The enforcement spine — an in-process client of sentinel's EvaluateHook.

Every governed action (tool call, LLM call) is forwarded here; sentinel
decides. The SDK performs **no local policy evaluation**.

Failure semantics (invariant): enforcement **fails open** — on any transport
error, timeout, or non-200, the action is allowed and the failure is audited
locally. Authentication (DPoP) is the only fail-closed surface, and even an
auth failure never blocks the agent: it surfaces as a fail-open allow here.
"""

from __future__ import annotations

import logging

import httpx

from firstops._identity import Identity
from firstops.events import ActionEvent, Decision

logger = logging.getLogger("firstops")

_HOOK_PATH = "/api/v1/daemon/evaluate-hook"
_DEFAULT_TIMEOUT = 5.0


class EnforcementClient:
    """Forwards action events to sentinel and returns the Decision."""

    def __init__(
        self,
        identity: Identity,
        *,
        timeout: float = _DEFAULT_TIMEOUT,
        http: httpx.Client | None = None,
    ):
        self._identity = identity
        self._url = identity.gateway_url + _HOOK_PATH
        self._http = http or httpx.Client(timeout=timeout)

    def evaluate(self, event: ActionEvent) -> Decision:
        """Evaluate one action. **Never raises** — fails open on any error.

        Enforcement is fail-open by invariant: a transport error, timeout,
        non-200, malformed body, or any unexpected exception must allow the
        action (and audit the failure), never block the agent. The only
        fail-closed surface is DPoP auth, and even an auth rejection surfaces
        here as a fail-open allow rather than a raised exception.
        """
        try:
            # htu binds to the path only (no query); _url has no query string.
            proof = self._identity.proof("POST", self._url)
            resp = self._http.post(
                self._url,
                json=event.to_wire(),
                headers={
                    "Authorization": f"Bearer {self._identity.bearer_token}",
                    "DPoP": proof,
                    "Content-Type": "application/json",
                },
            )
            if resp.status_code != 200:
                logger.warning(
                    "firstops enforcement: status %d, failing open", resp.status_code
                )
                return Decision.fail_open(f"status {resp.status_code}")
            return Decision.from_wire(resp.json())
        except Exception as e:  # noqa: BLE001 - fail-open is the whole point
            logger.warning("firstops enforcement: failing open: %s", e)
            return Decision.fail_open(f"error: {e}")

    def close(self) -> None:
        self._http.close()
