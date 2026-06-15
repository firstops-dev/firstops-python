"""Regression guards for the proxy after the _identity.py extraction (M0.T0.2).

T0.2 lifted key-loading + DPoP signing out of proxy.py into Identity. The
acceptance bar is 'no MCP behavior change'. The existing test_proxy.py covers
POST header injection + double-init. These add the cases the refactor is most
likely to have silently broken:

  - DELETE verb is proxied (the daemon issues DELETE for MCP session teardown)
  - htu on the MCP path strips the query string (Identity.proof owns this now;
    a regression here re-introduces the htu-mismatch class of 401s)
  - the DPoP proof is bound to the request's METHOD (htm matches the verb)
  - SSE responses stream through with the gateway→localhost URL rewrite intact
  - all three forwarded headers (Content-Type, Accept, Mcp-Session-Id) pass
"""

from __future__ import annotations

import base64
import json
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

from firstops.proxy import init, shutdown


def _pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


def _decode_claims(dpop: str) -> dict:
    c = dpop.split(".")[1]
    c += "=" * (-len(c) % 4)
    return json.loads(base64.urlsafe_b64decode(c))


class _MockGateway(BaseHTTPRequestHandler):
    captured: list = []
    # The gateway base the proxy is pointed at, used for SSE URL-rewrite checks.
    base: str = ""

    def _record(self, method, body=b""):
        _MockGateway.captured.append(
            {
                "method": method,
                "path": self.path,
                "headers": {k.lower(): v for k, v in self.headers.items()},
                "body": body,
            }
        )

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        self._record("POST", body)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"ok":true}')

    def do_DELETE(self):
        self._record("DELETE")
        self.send_response(204)
        self.end_headers()

    def do_GET(self):
        self._record("GET")
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        # Emit an endpoint event referencing the gateway's own message URL so
        # the proxy's gateway→localhost rewrite has something to rewrite.
        msg_url = _MockGateway.base + "/mcp/sse/message"
        self.wfile.write(f"event: endpoint\ndata: {msg_url}\n\n".encode())
        self.wfile.flush()

    def log_message(self, *a):
        pass


@pytest.fixture
def gateway():
    _MockGateway.captured = []
    srv = HTTPServer(("127.0.0.1", 0), _MockGateway)
    _MockGateway.base = f"http://127.0.0.1:{srv.server_address[1]}"
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    try:
        yield _MockGateway.base
    finally:
        srv.shutdown()


def _free_local_port() -> int:
    import socket

    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_delete_is_proxied_with_dpop(gateway):
    """DELETE (MCP session teardown) must be forwarded with auth headers."""
    local = _free_local_port()
    init(agent_id="ag", private_key_pem=_pem(), port=local, gateway_url=gateway)
    try:
        time.sleep(0.1)
        resp = httpx.delete(f"http://127.0.0.1:{local}/mcp/proxy/conn-1")
        assert resp.status_code == 204
        assert len(_MockGateway.captured) == 1
        req = _MockGateway.captured[0]
        assert req["method"] == "DELETE"
        assert req["headers"]["authorization"] == "Bearer fo_agent_ag"
        assert "dpop" in req["headers"]
        # htm in the proof must match the verb.
        claims = _decode_claims(req["headers"]["dpop"])
        assert claims["htm"] == "DELETE"
    finally:
        shutdown()


def test_mcp_htu_strips_query_string(gateway):
    """A request path with a query (e.g. /mcp/sse/message?sessionId=x) must
    sign an htu WITHOUT the query — Identity.proof owns this canonicalization
    after the refactor. A regression here re-introduces htu-mismatch 401s on
    every MCP message POST.
    """
    local = _free_local_port()
    init(agent_id="ag", private_key_pem=_pem(), port=local, gateway_url=gateway)
    try:
        time.sleep(0.1)
        httpx.post(
            f"http://127.0.0.1:{local}/mcp/sse/message?sessionId=abc123&seq=2",
            json={"jsonrpc": "2.0", "method": "ping", "id": 1},
            headers={"Content-Type": "application/json"},
        )
        req = _MockGateway.captured[-1]
        # The upstream path still carries the query (proxy forwards it)...
        assert "sessionId=abc123" in req["path"]
        # ...but the SIGNED htu must NOT.
        claims = _decode_claims(req["headers"]["dpop"])
        assert "?" not in claims["htu"], f"htu leaked query: {claims['htu']!r}"
        assert claims["htu"].endswith("/mcp/sse/message")
        assert claims["htu"] == gateway + "/mcp/sse/message"
    finally:
        shutdown()


def test_forwards_all_three_headers(gateway):
    local = _free_local_port()
    init(agent_id="ag", private_key_pem=_pem(), port=local, gateway_url=gateway)
    try:
        time.sleep(0.1)
        httpx.post(
            f"http://127.0.0.1:{local}/mcp/proxy/conn-1",
            content=b'{"x":1}',
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "Mcp-Session-Id": "sess-42",
            },
        )
        h = _MockGateway.captured[-1]["headers"]
        assert h["content-type"] == "application/json"
        assert h["accept"] == "application/json"
        assert h["mcp-session-id"] == "sess-42"
    finally:
        shutdown()


def test_sse_streams_and_rewrites_gateway_url_to_localhost(gateway):
    """An SSE GET must stream through and the gateway message URL must be
    rewritten to the localhost sidecar URL (so the MCP client posts back
    through the proxy, not directly to the gateway).
    """
    local = _free_local_port()
    init(agent_id="ag", private_key_pem=_pem(), port=local, gateway_url=gateway)
    try:
        time.sleep(0.1)
        with httpx.stream(
            "GET",
            f"http://127.0.0.1:{local}/mcp/sse/connect",
            headers={"Accept": "text/event-stream"},
            timeout=5.0,
        ) as resp:
            assert resp.status_code == 200
            assert "text/event-stream" in resp.headers.get("content-type", "")
            body = ""
            for chunk in resp.iter_text():
                body += chunk
                if "data:" in body:
                    break
        # The gateway base must be rewritten to the local sidecar.
        local_msg = f"http://127.0.0.1:{local}/mcp/sse/message"
        assert local_msg in body, f"SSE URL not rewritten to localhost; got {body!r}"
        assert gateway + "/mcp/sse/message" not in body
    finally:
        shutdown()
