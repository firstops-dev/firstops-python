"""Tests for the sidecar proxy — verifies header injection and URL rewriting."""

import json
import threading
import time
from http.server import HTTPServer, BaseHTTPRequestHandler

from cryptography.hazmat.primitives.asymmetric import ec
from cryptography.hazmat.primitives.serialization import (
    Encoding,
    NoEncryption,
    PrivateFormat,
)

import httpx

from firstops.proxy import init, shutdown


def _generate_pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()).decode()


class _MockGateway(BaseHTTPRequestHandler):
    """Captures requests for assertion."""

    captured: list = []

    def do_POST(self):
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length > 0 else b""
        _MockGateway.captured.append({
            "method": "POST",
            "path": self.path,
            "headers": dict(self.headers),
            "body": body,
        })
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"result": "ok"}).encode())

    def do_GET(self):
        _MockGateway.captured.append({
            "method": "GET",
            "path": self.path,
            "headers": dict(self.headers),
        })
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        self.wfile.write(b"data: hello\n\n")

    def log_message(self, fmt, *args):
        pass  # suppress logs


def test_proxy_adds_auth_headers():
    """Verify the proxy injects Authorization and DPoP headers."""
    _MockGateway.captured = []

    # Start mock gateway
    mock = HTTPServer(("127.0.0.1", 19876), _MockGateway)
    mock_thread = threading.Thread(target=mock.serve_forever, daemon=True)
    mock_thread.start()

    try:
        pem = _generate_pem()
        init(
            agent_id="test-agent-123",
            private_key_pem=pem,
            port=19877,
            gateway_url="http://127.0.0.1:19876",
        )

        # Give proxy a moment to start
        time.sleep(0.1)

        # Send a request through the proxy
        resp = httpx.post(
            "http://127.0.0.1:19877/mcp/proxy/conn-abc",
            json={"jsonrpc": "2.0", "method": "tools/list", "id": 1},
            headers={"Content-Type": "application/json"},
        )
        assert resp.status_code == 200

        # Check the mock gateway received the right headers
        assert len(_MockGateway.captured) == 1
        req = _MockGateway.captured[0]
        assert req["path"] == "/mcp/proxy/conn-abc"
        # Headers object is case-insensitive; use .get() style via original headers
        hdrs = {k.lower(): v for k, v in req["headers"].items()}
        assert hdrs["authorization"] == "Bearer fo_agent_test-agent-123"
        assert "dpop" in hdrs
        assert hdrs["content-type"] == "application/json"

        # Verify DPoP proof is a valid JWT
        dpop = hdrs["dpop"]
        parts = dpop.split(".")
        assert len(parts) == 3

    finally:
        shutdown()
        mock.shutdown()


def test_proxy_rejects_double_init():
    """Verify calling init twice raises."""
    pem = _generate_pem()
    init(agent_id="a", private_key_pem=pem, port=19878, gateway_url="http://127.0.0.1:1")
    try:
        raised = False
        try:
            init(agent_id="b", private_key_pem=pem, port=19879, gateway_url="http://127.0.0.1:1")
        except RuntimeError:
            raised = True
        assert raised
    finally:
        shutdown()
