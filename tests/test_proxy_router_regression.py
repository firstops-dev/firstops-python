"""Router-refactor regression: /mcp/ DPoP-signs, /llm/ never does.

T1.3 split proxy.py into a router (/mcp/ terminal vs /llm/<provider>/
chain-link). The one invariant the refactor must never break: the MCP route
still attaches the agent's Bearer + DPoP proof, and the LLM route NEVER
DPoP-signs an upstream (the agent's own Authorization passes through verbatim,
and no `DPoP` header is added). A single mock server backs BOTH an MCP gateway
path and an LLM upstream path so the two routes are compared apples-to-apples
against the same listener.
"""

from __future__ import annotations

import threading
import time
from http.server import BaseHTTPRequestHandler, HTTPServer

import httpx
import pytest
from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

from firstops import proxy
from firstops.events import Decision


def _pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


class _Recorder(BaseHTTPRequestHandler):
    captured: list = []

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        self.rfile.read(n) if n else None
        _Recorder.captured.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "dpop": self.headers.get("DPoP"),
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def log_message(self, *a):
        pass


class _StubEnforcement:
    def evaluate(self, event):
        return Decision(action="allow")


@pytest.fixture
def router():
    _Recorder.captured = []
    server = HTTPServer(("127.0.0.1", 0), _Recorder)
    base = f"http://127.0.0.1:{server.server_address[1]}"
    threading.Thread(target=server.serve_forever, daemon=True).start()

    local = _free_port()
    # The SAME server is both the MCP gateway and the openai LLM upstream.
    proxy.init(
        agent_id="ag",
        private_key_pem=_pem(),
        port=local,
        gateway_url=base,
        enforcement=_StubEnforcement(),
        llm_upstreams={"openai": base},
    )
    time.sleep(0.1)
    try:
        yield local
    finally:
        proxy.shutdown()
        server.shutdown()


def _free_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_mcp_route_dpop_signs(router):
    httpx.post(
        f"http://127.0.0.1:{router}/mcp/proxy/conn-1",
        content=b'{"x":1}',
        headers={"Content-Type": "application/json"},
        timeout=5.0,
    )
    cap = _Recorder.captured[-1]
    assert cap["auth"] == "Bearer fo_agent_ag"  # FirstOps brokered credential
    assert cap["dpop"] is not None  # DPoP proof attached


def test_llm_route_never_dpop_signs_and_passes_agent_auth(router):
    httpx.post(
        f"http://127.0.0.1:{router}/llm/openai/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        headers={"Authorization": "Bearer sk-agent-own-key"},
        timeout=5.0,
    )
    cap = _Recorder.captured[-1]
    # The agent's OWN key is forwarded verbatim — NOT replaced with the
    # FirstOps brokered token.
    assert cap["auth"] == "Bearer sk-agent-own-key"
    # And crucially: no DPoP proof is ever attached to an LLM upstream.
    assert cap["dpop"] is None


def test_llm_route_does_not_inject_firstops_bearer(router):
    """Even when the agent sends NO Authorization, the LLM route must not
    fall back to injecting the FirstOps brokered Bearer (that would leak the
    brokered credential to an arbitrary upstream)."""
    httpx.post(
        f"http://127.0.0.1:{router}/llm/openai/v1/models",
        json={"model": "gpt-4o"},
        timeout=5.0,
    )
    cap = _Recorder.captured[-1]
    assert cap["auth"] != "Bearer fo_agent_ag"
    assert cap["dpop"] is None
