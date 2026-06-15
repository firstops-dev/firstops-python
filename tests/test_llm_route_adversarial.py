"""Adversarial tests for the LLM chain-link route — fresh network code.

The /llm/<provider>/... route is new and sits in the request data path. These
tests probe it the way an attacker or a malformed client would:

  - path traversal in the request line (governance sees one path, upstream
    receives another) — a path-confusion bug
  - non-dict / non-JSON / empty bodies (governed? forwarded? visible?)
  - query-string preservation, GET vs POST, missing Authorization
  - SSE streaming passthrough
  - the load-bearing invariant #2: the agent's model key (Authorization) is
    forwarded verbatim and NEVER appears in any log output

Reuses the _StubEnforcement + mock-upstream pattern from test_llm_route.py.
Bugs are xfail(strict=True); everything else is a positive regression guard.
"""

from __future__ import annotations

import io
import json
import logging
import socket
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
from firstops.events import CHANNEL_LLM, Decision

_PROXY_PORT = 19942
_UPSTREAM_PORT = 19943
_SECRET_KEY = "Bearer sk-SUPERSECRETMODELKEY-do-not-log"


def _pem() -> str:
    key = ec.generate_private_key(ec.SECP256R1())
    return key.private_bytes(
        Encoding.PEM, PrivateFormat.TraditionalOpenSSL, NoEncryption()
    ).decode()


class _StubEnforcement:
    def __init__(self, decision: Decision):
        self.decision = decision
        self.events = []

    def evaluate(self, event):
        self.events.append(event)
        return self.decision


class _MockUpstream(BaseHTTPRequestHandler):
    captured: list = []
    sse: bool = False

    def _handle(self, method: str):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        _MockUpstream.captured.append(
            {
                "method": method,
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": body,
            }
        )
        if _MockUpstream.sse:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for i in range(3):
                self.wfile.write(f"data: chunk{i}\n\n".encode())
                self.wfile.flush()
                time.sleep(0.02)
            return
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"ok": True}).encode())

    def do_POST(self):
        self._handle("POST")

    def do_GET(self):
        self._handle("GET")

    def log_message(self, *a):
        pass


@pytest.fixture
def sidecar():
    """Start a mock upstream + sidecar; also capture all 'firstops' log output
    so tests can assert the model key never appears in logs."""
    _MockUpstream.captured = []
    _MockUpstream.sse = False
    upstream = HTTPServer(("127.0.0.1", _UPSTREAM_PORT), _MockUpstream)
    t = threading.Thread(target=upstream.serve_forever, daemon=True)
    t.start()

    log_buf = io.StringIO()
    handler = logging.StreamHandler(log_buf)
    handler.setLevel(logging.DEBUG)
    fo_logger = logging.getLogger("firstops")
    prior_level = fo_logger.level
    fo_logger.setLevel(logging.DEBUG)
    fo_logger.addHandler(handler)

    state = {"logs": log_buf}

    def start(decision: Decision) -> _StubEnforcement:
        stub = _StubEnforcement(decision)
        state["stub"] = stub
        proxy.init(
            agent_id="agent-1",
            private_key_pem=_pem(),
            port=_PROXY_PORT,
            gateway_url="http://127.0.0.1:1",
            enforcement=stub,
            llm_upstreams={"openai": f"http://127.0.0.1:{_UPSTREAM_PORT}"},
        )
        return stub

    state["start"] = start
    yield state

    fo_logger.removeHandler(handler)
    fo_logger.setLevel(prior_level)
    proxy.shutdown()
    upstream.shutdown()
    upstream.server_close()


def _post(body, auth: str = _SECRET_KEY, path="/llm/openai/v1/chat/completions", **kw):
    return httpx.post(
        f"http://127.0.0.1:{_PROXY_PORT}{path}",
        headers={"Authorization": auth},
        timeout=5.0,
        **({"json": body} if isinstance(body, (dict, list)) else {"content": body}),
        **kw,
    )


def _raw_request(path: str, body: bytes = b'{"a":1}') -> str:
    """Send a request with a literal, un-normalized path (httpx would collapse
    `..` client-side, hiding the traversal). Returns the HTTP status line."""
    s = socket.create_connection(("127.0.0.1", _PROXY_PORT), timeout=5)
    req = (
        f"POST {path} HTTP/1.1\r\nHost: x\r\n"
        f"Content-Length: {len(body)}\r\n"
        f"Content-Type: application/json\r\n"
        f"Authorization: {_SECRET_KEY}\r\nConnection: close\r\n\r\n"
    ).encode() + body
    s.sendall(req)
    data = b""
    while True:
        chunk = s.recv(4096)
        if not chunk:
            break
        data += chunk
    s.close()
    return data.split(b"\r\n", 1)[0].decode()


# ---------------------------------------------------------------------------
# INVARIANT #2: the model key is never logged.
# ---------------------------------------------------------------------------


def test_model_key_never_appears_in_logs(sidecar):
    stub = sidecar["start"](Decision(action="allow"))
    _post({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})
    assert _MockUpstream.captured[-1]["auth"] == _SECRET_KEY  # forwarded verbatim
    logs = sidecar["logs"].getvalue()
    assert "SUPERSECRETMODELKEY" not in logs, "model key leaked into logs"
    # And it must not be in any captured pre-event tool_input either.
    assert "SUPERSECRETMODELKEY" not in json.dumps(stub.events[0].tool_input)


def test_model_key_not_logged_even_on_upstream_failure(sidecar):
    """Point the provider at a dead upstream so _forward hits the error path,
    which logs. The key must still never appear."""
    sidecar["start"](Decision(action="allow"))
    # Re-init with a dead upstream port.
    proxy.shutdown()
    stub = _StubEnforcement(Decision(action="allow"))
    proxy.init(
        agent_id="agent-1",
        private_key_pem=_pem(),
        port=_PROXY_PORT,
        gateway_url="http://127.0.0.1:1",
        enforcement=stub,
        llm_upstreams={"openai": "http://127.0.0.1:1"},  # nothing listening
    )
    resp = _post({"model": "gpt-4o", "messages": []})
    assert resp.status_code == 502
    assert "SUPERSECRETMODELKEY" not in sidecar["logs"].getvalue()


# ---------------------------------------------------------------------------
# Path confusion / traversal. governance sees /llm/openai/... but the upstream
# receives a different, traversed path.
# ---------------------------------------------------------------------------


def test_path_traversal_does_not_desync_governed_and_forwarded_path(sidecar):
    """FIXED: a `..` in the LLM path is rejected with 400 before governance or
    forwarding, so the governed path can never desync from the forwarded path."""
    stub = sidecar["start"](Decision(action="allow"))
    status = _raw_request("/llm/openai/../secret/admin")
    # Rejected outright — never governed, never forwarded.
    assert "400" in status
    assert _MockUpstream.captured == []
    assert stub.events == []


# ---------------------------------------------------------------------------
# Body shape handling: array, non-JSON, empty.
# ---------------------------------------------------------------------------


def test_json_array_body_is_forwarded_ungoverned(sidecar):
    """A JSON array (not a dict) cannot be governed by the current parser
    (`isinstance(parsed, dict)` gate). Documents that it is forwarded WITHOUT a
    pre-event — a governance hole. Fail-open is acceptable, silence is not: this
    test makes the gap visible so a future change can flag/audit it."""
    stub = sidecar["start"](Decision(action="deny", reason="should-not-apply"))
    resp = httpx.post(
        f"http://127.0.0.1:{_PROXY_PORT}/llm/openai/v1/embeddings",
        content=b"[1, 2, 3]",
        headers={"Authorization": _SECRET_KEY, "Content-Type": "application/json"},
        timeout=5.0,
    )
    # Even though the stub would DENY, an array body bypasses governance and is
    # forwarded as-is.
    assert resp.status_code == 200
    assert stub.events == []
    assert _MockUpstream.captured[-1]["body"] == b"[1, 2, 3]"


def test_non_json_body_forwarded_ungoverned(sidecar):
    stub = sidecar["start"](Decision(action="deny", reason="should-not-apply"))
    resp = httpx.post(
        f"http://127.0.0.1:{_PROXY_PORT}/llm/openai/v1/audio",
        content=b"\x00\x01binary-not-json",
        headers={"Authorization": _SECRET_KEY},
        timeout=5.0,
    )
    assert resp.status_code == 200
    assert stub.events == []  # un-governed
    assert _MockUpstream.captured[-1]["body"] == b"\x00\x01binary-not-json"


def test_empty_body_is_forwarded_and_ungoverned(sidecar):
    stub = sidecar["start"](Decision(action="deny", reason="should-not-apply"))
    resp = httpx.post(
        f"http://127.0.0.1:{_PROXY_PORT}/llm/openai/v1/models",
        content=b"",
        headers={"Authorization": _SECRET_KEY},
        timeout=5.0,
    )
    assert resp.status_code == 200
    assert stub.events == []
    assert len(_MockUpstream.captured) == 1


# ---------------------------------------------------------------------------
# Request mechanics: query preservation, GET, missing auth, large body.
# ---------------------------------------------------------------------------


def test_query_string_is_preserved_to_upstream(sidecar):
    sidecar["start"](Decision(action="allow"))
    _post(
        {"model": "gpt-4o", "messages": []},
        path="/llm/openai/v1/chat/completions?api-version=2024&beta=true",
    )
    assert (
        _MockUpstream.captured[-1]["path"]
        == "/v1/chat/completions?api-version=2024&beta=true"
    )


def test_get_to_llm_route_forwards_as_get(sidecar):
    sidecar["start"](Decision(action="allow"))
    resp = httpx.get(
        f"http://127.0.0.1:{_PROXY_PORT}/llm/openai/v1/models",
        headers={"Authorization": _SECRET_KEY},
        timeout=5.0,
    )
    assert resp.status_code == 200
    assert _MockUpstream.captured[-1]["method"] == "GET"


def test_missing_authorization_still_forwards(sidecar):
    """No Authorization header: the route must still forward (fail-open), it
    just forwards no auth. The upstream decides."""
    sidecar["start"](Decision(action="allow"))
    resp = httpx.post(
        f"http://127.0.0.1:{_PROXY_PORT}/llm/openai/v1/chat/completions",
        json={"model": "gpt-4o", "messages": []},
        timeout=5.0,
    )
    assert resp.status_code == 200
    assert _MockUpstream.captured[-1]["auth"] is None


def test_large_body_round_trips(sidecar):
    sidecar["start"](Decision(action="allow"))
    big = {"model": "gpt-4o", "messages": [{"role": "user", "content": "x" * 200_000}]}
    resp = _post(big)
    assert resp.status_code == 200
    assert len(_MockUpstream.captured[-1]["body"]) > 200_000


def test_sse_response_streams_back(sidecar):
    sidecar["start"](Decision(action="allow"))
    _MockUpstream.sse = True
    chunks = []
    with httpx.stream(
        "POST",
        f"http://127.0.0.1:{_PROXY_PORT}/llm/openai/v1/chat/completions",
        json={"model": "gpt-4o", "stream": True, "messages": []},
        headers={"Authorization": _SECRET_KEY},
        timeout=5.0,
    ) as resp:
        assert resp.status_code == 200
        assert "text/event-stream" in resp.headers.get("content-type", "")
        for c in resp.iter_text():
            chunks.append(c)
    assert "data: chunk0" in "".join(chunks)
    assert "data: chunk2" in "".join(chunks)


def test_deny_response_shape_is_provider_error_envelope(sidecar):
    sidecar["start"](Decision(action="deny", reason="pii", policy_id="p-9"))
    resp = _post({"model": "gpt-4o", "messages": []})
    assert resp.status_code == 403
    err = resp.json()["error"]
    assert err["type"] == "firstops_policy_violation"
    assert err["policy_id"] == "p-9"
    assert _MockUpstream.captured == []  # never forwarded


def test_governed_pre_event_is_llm_channel(sidecar):
    stub = sidecar["start"](Decision(action="allow"))
    _post({"model": "gpt-4o", "messages": []})
    assert stub.events[-1].channel == CHANNEL_LLM
