"""Tests for the LLM chain-link route on the sidecar.

Spins up the real sidecar (proxy.init) with a stub enforcement and a local
mock 'upstream' so we exercise the actual routing/forwarding code.
"""

import json
import threading
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
from firstops.events import CHANNEL_LLM, EVENT_PRE_TOOL_USE, Decision

_PROXY_PORT = 19940
_UPSTREAM_PORT = 19941


def _generate_pem() -> str:
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

    def do_POST(self):
        n = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(n) if n else b""
        _MockUpstream.captured.append(
            {
                "path": self.path,
                "auth": self.headers.get("Authorization"),
                "body": body,
            }
        )
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(json.dumps({"id": "resp-1", "ok": True}).encode())

    def log_message(self, *a):
        pass


@pytest.fixture
def sidecar():
    """Start a mock upstream + the sidecar with a controllable stub enforcement."""
    _MockUpstream.captured = []
    upstream = HTTPServer(("127.0.0.1", _UPSTREAM_PORT), _MockUpstream)
    t = threading.Thread(target=upstream.serve_forever, daemon=True)
    t.start()

    state = {}

    def start(decision: Decision) -> _StubEnforcement:
        stub = _StubEnforcement(decision)
        state["stub"] = stub
        proxy.init(
            agent_id="agent-1",
            private_key_pem=_generate_pem(),
            port=_PROXY_PORT,
            gateway_url="http://127.0.0.1:1",
            enforcement=stub,
            llm_upstreams={"openai": f"http://127.0.0.1:{_UPSTREAM_PORT}"},
        )
        return stub

    yield start

    proxy.shutdown()
    upstream.shutdown()
    upstream.server_close()


def _post(body: dict, auth: str = "Bearer sk-user-key"):
    return httpx.post(
        f"http://127.0.0.1:{_PROXY_PORT}/llm/openai/v1/chat/completions",
        json=body,
        headers={"Authorization": auth},
        timeout=5.0,
    )


def test_llm_allow_forwards_and_passes_auth_through(sidecar):
    stub = sidecar(Decision(action="allow"))
    resp = _post({"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]})

    assert resp.status_code == 200
    assert resp.json()["ok"] is True
    # Reached the upstream at the right path with the user's key passed through.
    assert len(_MockUpstream.captured) == 1
    cap = _MockUpstream.captured[0]
    assert cap["path"] == "/v1/chat/completions"
    assert cap["auth"] == "Bearer sk-user-key"
    # Governed as an LLM-channel pre event.
    assert stub.events[0].channel == CHANNEL_LLM
    assert stub.events[0].event_type == EVENT_PRE_TOOL_USE
    assert stub.events[0].tool_input["model"] == "gpt-4o"


def test_llm_deny_returns_403_and_does_not_forward(sidecar):
    sidecar(Decision(action="deny", reason="prompt injection", policy_id="pi-1"))
    resp = _post({"model": "gpt-4o", "messages": [{"role": "user", "content": "x"}]})

    assert resp.status_code == 403
    assert "firstops" in resp.json()["error"]["type"]
    assert _MockUpstream.captured == []  # never hit the upstream


def test_llm_modify_rewrites_body_before_forward(sidecar):
    import base64

    scrubbed = json.dumps(
        {"model": "gpt-4o", "messages": [{"role": "user", "content": "[REDACTED]"}]}
    ).encode()
    dec = Decision.from_wire(
        {"decision": "modify", "modified_payload": base64.b64encode(scrubbed).decode()}
    )
    sidecar(dec)
    _post({"model": "gpt-4o", "messages": [{"role": "user", "content": "my SSN is 123"}]})

    assert len(_MockUpstream.captured) == 1
    forwarded = json.loads(_MockUpstream.captured[0]["body"])
    assert forwarded["messages"][0]["content"] == "[REDACTED]"


def test_llm_unknown_provider_502(sidecar):
    sidecar(Decision(action="allow"))
    resp = httpx.post(
        f"http://127.0.0.1:{_PROXY_PORT}/llm/cohere/v1/chat",
        json={"x": 1},
        timeout=5.0,
    )
    assert resp.status_code == 502
