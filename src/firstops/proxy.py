"""Sidecar proxy that adds DPoP auth headers to MCP requests."""

import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, urlunparse

import httpx

from firstops.dpop import create_proof, load_private_key

logger = logging.getLogger("firstops")

_DEFAULT_PORT = 9322
_DEFAULT_GATEWAY = "https://api.firstops.ai"

_server: HTTPServer | None = None
_server_thread: threading.Thread | None = None


def init(
    agent_id: str,
    private_key_pem: str,
    port: int = _DEFAULT_PORT,
    gateway_url: str = _DEFAULT_GATEWAY,
) -> None:
    """Start the sidecar proxy on localhost.

    Args:
        agent_id: The agent principal ID (without fo_agent_ prefix).
        private_key_pem: EC P-256 private key in PEM format.
        port: Local port for the proxy (default 9322).
        gateway_url: FirstOps gateway base URL.
    """
    global _server, _server_thread

    if _server is not None:
        raise RuntimeError("firstops proxy already running")

    key = load_private_key(private_key_pem)
    bearer_token = f"fo_agent_{agent_id}"
    gateway = gateway_url.rstrip("/")

    # Validate gateway URL
    parsed = urlparse(gateway)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"invalid gateway_url: {gateway_url}")

    handler_class = _make_handler(key, bearer_token, gateway, port)
    _server = HTTPServer(("127.0.0.1", port), handler_class)
    _server_thread = threading.Thread(target=_server.serve_forever, daemon=True)
    _server_thread.start()
    logger.info("firstops proxy listening on 127.0.0.1:%d", port)


def shutdown() -> None:
    """Stop the sidecar proxy."""
    global _server, _server_thread
    if _server is not None:
        _server.shutdown()
        _server = None
        _server_thread = None
        logger.info("firstops proxy stopped")


def _make_handler(key, bearer_token: str, gateway: str, local_port: int):
    """Create a request handler class bound to the given config."""

    # Pre-create a client for non-streaming requests
    client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self._proxy("POST")

        def do_GET(self):
            self._proxy("GET")

        def do_DELETE(self):
            self._proxy("DELETE")

        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def _proxy(self, method: str):
            path = self.path  # e.g. /mcp/proxy/<connID> or /mcp/sse/message?...
            gateway_url = gateway + path

            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

            # Create DPoP proof against the gateway URL (path only, no query for htu)
            parsed = urlparse(gateway_url)
            htu = urlunparse((parsed.scheme, parsed.netloc, parsed.path, "", "", ""))
            proof = create_proof(key, method, htu)

            # Build upstream headers
            upstream_headers = {
                "Authorization": f"Bearer {bearer_token}",
                "DPoP": proof,
            }
            # Forward relevant headers
            for h in ("Content-Type", "Accept", "Mcp-Session-Id"):
                v = self.headers.get(h)
                if v:
                    upstream_headers[h] = v

            # Check if this is an SSE request
            is_sse = (
                method == "GET"
                and self.headers.get("Accept", "").startswith("text/event-stream")
            )

            if is_sse:
                self._stream_sse(gateway_url, upstream_headers)
            else:
                self._forward(method, gateway_url, upstream_headers, body)

        def _forward(self, method: str, url: str, headers: dict, body: bytes | None):
            """Forward a non-streaming request."""
            try:
                resp = client.request(method, url, headers=headers, content=body)
            except httpx.HTTPError as e:
                logger.error("upstream request failed: %s", e)
                self.send_error(502, "upstream request failed")
                return

            self.send_response(resp.status_code)
            for k, v in resp.headers.items():
                if k.lower() not in ("transfer-encoding", "connection"):
                    self.send_header(k, v)
            self.end_headers()
            self.wfile.write(resp.content)

        def _stream_sse(self, url: str, headers: dict):
            """Stream an SSE response, rewriting gateway URLs to localhost."""
            try:
                with httpx.stream("GET", url, headers=headers, timeout=None) as resp:
                    self.send_response(resp.status_code)
                    for k, v in resp.headers.items():
                        if k.lower() not in ("transfer-encoding", "connection"):
                            self.send_header(k, v)
                    self.end_headers()

                    gateway_msg_url = gateway + "/mcp/sse/message"
                    local_msg_url = f"http://127.0.0.1:{local_port}/mcp/sse/message"

                    for chunk in resp.iter_bytes():
                        # Rewrite gateway message URL to localhost
                        text = chunk.decode("utf-8", errors="replace")
                        text = text.replace(gateway_msg_url, local_msg_url)
                        self.wfile.write(text.encode())
                        self.wfile.flush()
            except httpx.HTTPError as e:
                logger.error("upstream SSE request failed: %s", e)
                self.send_error(502, "upstream SSE request failed")

    return ProxyHandler
