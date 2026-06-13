"""Sidecar proxy — a dual-mode local forward proxy.

Two routes share one local listener:

  ``/mcp/...``            terminal to the FirstOps gateway (DPoP-signed,
                          credential-brokered) — the original behavior.
  ``/llm/<provider>/...`` inline chain-link: governs the request via
                          EvaluateHook(channel=llm), then forwards to the
                          customer-configured upstream (their gateway or the
                          provider). FirstOps is NOT in the LLM data path; the
                          agent's own Authorization is passed through verbatim.
"""

import json
import logging
import threading
from http.server import HTTPServer, BaseHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse

import httpx

from firstops._identity import Identity, build_identity
from firstops.events import CHANNEL_LLM, EVENT_PRE_TOOL_USE, ActionEvent

logger = logging.getLogger("firstops")

# Headers we must not forward verbatim to an LLM upstream: the RFC 7230
# hop-by-hop set plus the few httpx sets itself. The agent's Authorization
# (its model key) is intentionally NOT here — it passes through verbatim.
_LLM_STRIP_HEADERS = frozenset(
    {
        "host",
        "content-length",
        "accept-encoding",
        "connection",
        "keep-alive",
        "transfer-encoding",
        "te",
        "trailer",
        "upgrade",
        "proxy-authorization",
        "proxy-authenticate",
    }
)

# Hard cap on a forwarded request body (defensive against a huge Content-Length).
_MAX_BODY_BYTES = 100 * 1024 * 1024

_DEFAULT_PORT = 9322
_DEFAULT_GATEWAY = "https://api.firstops.dev"

# Module-global proxy state. Guarded by _lock for thread safety.
_lock = threading.Lock()
_server: HTTPServer | None = None
_server_thread: threading.Thread | None = None
_current_agent_id: str | None = None
_current_port: int | None = None
_current_gateway: str | None = None
_current_jkt: str | None = None


def init(
    agent_id: str,
    private_key_pem: str,
    port: int = _DEFAULT_PORT,
    gateway_url: str = _DEFAULT_GATEWAY,
    enforcement=None,
    llm_upstreams: dict[str, str] | None = None,
) -> None:
    """Start the sidecar proxy on localhost.

    This is **idempotent** for the same agent: calling init() twice with the
    same ``agent_id``, ``port``, and ``gateway_url`` is a no-op on the second
    call. This lets library code call init() defensively without worrying
    about coordinating with other callers in the same process.

    Calling init() with a **different** agent, port, or gateway while the
    proxy is already running raises ``RuntimeError`` — the sidecar is tied to
    one agent's private key and cannot serve multiple identities from a single
    instance. Call ``shutdown()`` first if you need to switch agents.

    Args:
        agent_id: The agent principal ID (without ``fo_agent_`` prefix).
        private_key_pem: EC P-256 private key in PEM format.
        port: Local port for the proxy (default 9322).
        gateway_url: FirstOps gateway base URL.

    Raises:
        RuntimeError: If a proxy is already running for a different agent,
            port, or gateway.
        ValueError: If ``gateway_url`` is malformed.
    """
    global _server, _server_thread, _current_agent_id, _current_port, _current_gateway
    global _current_jkt

    gateway = gateway_url.rstrip("/")

    # Build identity up front (validates gateway URL + key) so bad input fails
    # fast and so we can compare the key thumbprint on the idempotent path.
    identity = build_identity(agent_id, private_key_pem, gateway)
    jkt = identity.jkt

    with _lock:
        if _server is not None:
            same_target = (
                _current_agent_id == agent_id
                and _current_port == port
                and _current_gateway == gateway
            )
            if same_target and _current_jkt == jkt:
                logger.debug(
                    "firstops proxy already running for agent %s on port %d — init() is a no-op",
                    agent_id,
                    port,
                )
                return
            if same_target:
                # Same agent/port/gateway but a DIFFERENT key — refuse to
                # silently keep signing with the stale key (that would 401 and
                # fail open invisibly). Force an explicit re-key.
                raise RuntimeError(
                    f"firstops proxy is already running for agent {agent_id!r} "
                    f"with a different key; call shutdown() before re-keying."
                )

            # Mismatch — refuse to silently replace the running sidecar.
            raise RuntimeError(
                f"firstops proxy is already running for agent "
                f"{_current_agent_id!r} on port {_current_port} "
                f"(gateway={_current_gateway!r}); cannot start a second "
                f"instance for agent {agent_id!r}. Call shutdown() first "
                f"if you want to switch identities."
            )

        handler_class = _make_handler(identity, port, enforcement, llm_upstreams)

        class ThreadingHTTPServer(ThreadingMixIn, HTTPServer):
            daemon_threads = True

        _server = ThreadingHTTPServer(("127.0.0.1", port), handler_class)
        _server_thread = threading.Thread(
            target=_server.serve_forever, daemon=True
        )
        _server_thread.start()
        _current_agent_id = agent_id
        _current_port = port
        _current_gateway = gateway
        _current_jkt = jkt
        logger.info(
            "firstops proxy listening on 127.0.0.1:%d for agent %s",
            port,
            agent_id,
        )


def shutdown() -> None:
    """Stop the sidecar proxy. Idempotent — no-op if not running."""
    global _server, _server_thread, _current_agent_id, _current_port, _current_gateway
    global _current_jkt
    with _lock:
        if _server is not None:
            _server.shutdown()
            _server.server_close()  # release the listening socket, not just the loop
            _server = None
            _server_thread = None
            _current_agent_id = None
            _current_port = None
            _current_gateway = None
            _current_jkt = None
            logger.info("firstops proxy stopped")


def is_running() -> bool:
    """Return True if the sidecar proxy is currently running in this process."""
    with _lock:
        return _server is not None


def current_agent_id() -> str | None:
    """Return the agent ID the running proxy is serving, or None if not running."""
    with _lock:
        return _current_agent_id


def _denial_body(provider: str, decision) -> bytes:
    """Build a provider-shaped error envelope for a denied LLM request."""
    msg = f"blocked by FirstOps policy: {decision.reason}"
    if provider == "anthropic":
        env = {
            "type": "error",
            "error": {"type": "firstops_policy_violation", "message": msg},
        }
    else:  # openai-shaped default
        env = {
            "error": {
                "message": msg,
                "type": "firstops_policy_violation",
                "policy_id": decision.policy_id,
            }
        }
    return json.dumps(env).encode()


def _make_handler(identity: Identity, local_port: int, enforcement, llm_upstreams):
    """Create a request handler class bound to the given identity + config."""

    gateway = identity.gateway_url
    # Pre-create a client for non-streaming requests
    client = httpx.Client(timeout=httpx.Timeout(120.0, connect=10.0))

    class ProxyHandler(BaseHTTPRequestHandler):
        def do_POST(self):
            self._dispatch("POST")

        def do_GET(self):
            self._dispatch("GET")

        def do_DELETE(self):
            self._dispatch("DELETE")

        def log_message(self, fmt, *args):
            logger.debug(fmt, *args)

        def _dispatch(self, method: str):
            # Route by path prefix: LLM chain-link vs MCP terminal.
            if self.path.startswith("/llm/"):
                self._handle_llm(method)
            else:
                self._proxy_mcp(method)

        # ---- LLM chain-link route -------------------------------------
        def _handle_llm(self, method: str):
            if enforcement is None or llm_upstreams is None:
                self.send_error(503, "LLM governance not configured")
                return

            # /llm/<provider>/<rest...>
            rest = self.path[len("/llm/"):]
            provider, _, tail = rest.partition("/")
            upstream_base = llm_upstreams.get(provider)
            if not upstream_base:
                self.send_error(502, f"unknown LLM provider: {provider!r}")
                return

            path_only, sep, query = tail.partition("?")
            # Reject path traversal: the path we govern MUST equal the path we
            # forward. Letting httpx normalize `..` would desync the two.
            if ".." in path_only.split("/"):
                self.send_error(400, "invalid path")
                return

            body = self._read_request_body()
            if body is None:
                self.send_error(400, "invalid or oversized request body")
                return

            target = upstream_base.rstrip("/") + "/" + path_only
            if sep:
                target += "?" + query

            # Pre-request governance (channel=llm). Returns (forward_body, denial).
            body, denial = self._govern_llm(provider, path_only, body)
            if denial is not None:
                self.send_response(403)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(denial)
                return

            # Forward everything, strip the few headers we must replace. The
            # agent's own Authorization (its model key) passes through verbatim;
            # we never DPoP-sign an upstream that isn't the FirstOps gateway.
            fwd_headers = {
                k: v
                for k, v in self.headers.items()
                if k.lower() not in _LLM_STRIP_HEADERS
            }
            self._forward(method, target, fwd_headers, body or None)

        def _read_request_body(self) -> bytes | None:
            """Read the request body safely. Returns None on malformed/oversized
            framing (caller replies 400). No chunked-request support — a missing
            Content-Length is treated as an empty body."""
            raw = self.headers.get("Content-Length")
            if raw is None:
                return b""
            try:
                n = int(raw)
            except ValueError:
                return None
            if n < 0 or n > _MAX_BODY_BYTES:
                return None
            return self.rfile.read(n) if n > 0 else b""

        def _govern_llm(self, provider: str, path_only: str, body: bytes):
            """Evaluate an LLM request. Returns (body_to_forward, denial_or_None)."""
            if not body:
                return body, None
            try:
                parsed = json.loads(body)
            except (ValueError, TypeError):
                return body, None  # non-JSON — can't govern, forward as-is
            if not isinstance(parsed, dict):
                return body, None

            tool_name = f"{provider}." + path_only.strip("/").replace("/", ".")
            event = ActionEvent(
                event_type=EVENT_PRE_TOOL_USE,
                tool_name=tool_name,
                channel=CHANNEL_LLM,
                tool_input=parsed,
                # The sidecar rewrites the request body before forwarding, so a
                # prompt scrub can ship as modify rather than escalate to deny.
                producer_can_apply_modify=True,
            )
            decision = enforcement.evaluate(event)
            if decision.blocked:
                return body, _denial_body(provider, decision)
            if decision.modified and decision.modified_payload:
                return decision.modified_payload, None
            return body, None

        # ---- MCP terminal route (original behavior) -------------------
        def _proxy_mcp(self, method: str):
            path = self.path  # e.g. /mcp/proxy/<connID> or /mcp/sse/message?...
            gateway_url = gateway + path

            # Read request body
            content_length = int(self.headers.get("Content-Length", 0))
            body = self.rfile.read(content_length) if content_length > 0 else None

            # Create DPoP proof; identity.proof canonicalizes htu (strips query).
            proof = identity.proof(method, gateway_url)

            # Build upstream headers
            upstream_headers = {
                "Authorization": f"Bearer {identity.bearer_token}",
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
            """Forward a request, streaming if the response is SSE."""
            try:
                # Use a streaming request so we can detect SSE responses
                # before reading the full body.
                with httpx.stream(
                    method, url, headers=headers, content=body,
                    timeout=httpx.Timeout(120.0, connect=10.0),
                ) as resp:
                    self.send_response(resp.status_code)
                    for k, v in resp.headers.items():
                        if k.lower() not in ("transfer-encoding", "connection"):
                            self.send_header(k, v)
                    self.end_headers()

                    # Forward the body RAW: httpx auto-decompresses iter_bytes()/
                    # read(), but we keep the upstream Content-Encoding/Length
                    # headers, so we must pass the original (possibly gzipped)
                    # bytes through untouched or the client's decode fails.
                    # Flush per chunk so SSE/streaming responses arrive live.
                    for chunk in resp.iter_raw():
                        self.wfile.write(chunk)
                        self.wfile.flush()
            except httpx.HTTPError as e:
                logger.error("upstream request failed: %s", e)
                self.send_error(502, "upstream request failed")

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
