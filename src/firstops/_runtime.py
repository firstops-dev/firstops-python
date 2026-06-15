"""Process-wide FirstOps runtime — the single entrypoint an agent calls.

``firstops.init()`` establishes the agent identity, builds the enforcement
client (the EvaluateHook spine), and starts the dual-mode sidecar (MCP terminal
+ LLM chain-link). The returned :class:`Runtime` handle is what tool decorators
and harness adapters use to forward action events; it is also stored
process-globally so ``firstops.shutdown()`` works without a handle.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass

from firstops import proxy
from firstops._identity import Identity, build_identity
from firstops.enforcement import EnforcementClient

_DEFAULT_PORT = 9322
_DEFAULT_GATEWAY = "https://api.firstops.dev"

# Default LLM upstreams. Override per-provider via ``init(llm_upstreams=...)``
# to point the chain-link at a customer's existing gateway instead.
_DEFAULT_LLM_UPSTREAMS = {
    "openai": "https://api.openai.com",
    "anthropic": "https://api.anthropic.com",
}

_lock = threading.Lock()
_runtime: Runtime | None = None


@dataclass
class Runtime:
    """Handle to the live FirstOps runtime for this process."""

    identity: Identity
    enforcement: EnforcementClient
    port: int
    llm_upstreams: dict[str, str]


def init(
    agent_id: str,
    private_key_pem: str,
    *,
    port: int = _DEFAULT_PORT,
    gateway_url: str = _DEFAULT_GATEWAY,
    llm_upstreams: dict[str, str] | None = None,
) -> Runtime:
    """Establish identity, the enforcement client, and the dual-mode sidecar.

    Idempotent for the same identity: the sidecar refuses to switch agents
    mid-flight (raises ``RuntimeError``); re-initializing with the same
    parameters returns the existing runtime.

    Args:
        llm_upstreams: per-provider upstream base URLs for the LLM chain-link.
            Merged over the defaults (openai/anthropic public endpoints). Point
            a provider at your own gateway to chain in front of it.

    Returns:
        The process :class:`Runtime` handle.
    """
    global _runtime

    gateway = gateway_url.rstrip("/")
    upstreams = dict(_DEFAULT_LLM_UPSTREAMS)
    if llm_upstreams:
        upstreams.update(llm_upstreams)

    with _lock:
        if _runtime is not None:
            # Idempotent re-confirm — proxy.init owns the mismatch guard.
            proxy.init(
                agent_id=agent_id,
                private_key_pem=private_key_pem,
                port=port,
                gateway_url=gateway,
                enforcement=_runtime.enforcement,
                llm_upstreams=_runtime.llm_upstreams,
            )
            return _runtime

        identity = build_identity(agent_id, private_key_pem, gateway)
        enforcement = EnforcementClient(identity)
        proxy.init(
            agent_id=agent_id,
            private_key_pem=private_key_pem,
            port=port,
            gateway_url=gateway,
            enforcement=enforcement,
            llm_upstreams=upstreams,
        )
        _runtime = Runtime(
            identity=identity,
            enforcement=enforcement,
            port=port,
            llm_upstreams=upstreams,
        )
        return _runtime


def shutdown() -> None:
    """Stop the sidecar and tear down the runtime. Idempotent.

    Lifecycle: call at process/agent teardown, **not** concurrently with
    in-flight governed calls. ``EnforcementClient.evaluate`` is fully
    fail-open, so a racing ``evaluate()`` during shutdown degrades to an
    allow (never a crash or hang) — but the supported pattern is init-once,
    run, shutdown-once.
    """
    global _runtime
    proxy.shutdown()
    with _lock:
        if _runtime is not None:
            _runtime.enforcement.close()
            _runtime = None


def runtime() -> Runtime | None:
    """Return the live runtime, or None if init() has not been called."""
    with _lock:
        return _runtime


def llm_base_url(provider: str = "openai") -> str:
    """Return the local sidecar base URL to point an LLM client at.

    Point your client's ``base_url`` (or ``OPENAI_BASE_URL`` /
    ``ANTHROPIC_BASE_URL``) here; the sidecar governs the request and forwards
    to the configured upstream.
    """
    rt = runtime()
    if rt is None:
        raise RuntimeError("firstops.init() must be called before llm_base_url()")
    base = f"http://127.0.0.1:{rt.port}/llm/{provider}"
    return base + "/v1" if provider == "openai" else base


def mcp_url(connection_id: str) -> str:
    """Return the local sidecar URL to point an MCP client at for a connection.

    The sidecar DPoP-signs each request and forwards it to the FirstOps gateway,
    which brokers the upstream credentials — the agent never holds them.
    """
    rt = runtime()
    if rt is None:
        raise RuntimeError("firstops.init() must be called before mcp_url()")
    return f"http://127.0.0.1:{rt.port}/mcp/proxy/{connection_id}"
