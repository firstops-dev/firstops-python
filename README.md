# FirstOps Python SDK

Secure MCP proxy sidecar with [DPoP](https://datatracker.ietf.org/doc/html/rfc9449) authentication for AI agents.

FirstOps secures agent-to-tool connections. This SDK runs a lightweight local proxy that transparently adds DPoP-signed authentication headers to every MCP request your agent makes — no changes to your agent code required.

## Install

```bash
pip install firstops
```

## Quick Start

```python
import firstops

# Start the proxy sidecar (runs in background thread)
firstops.init(
    agent_id="your-agent-id",
    private_key_pem=open("agent-key.pem").read(),
)

# Point your MCP client at localhost:9322 instead of the remote server.
# The proxy handles auth transparently — DPoP proofs, bearer tokens, SSE streaming.

# When done:
firstops.shutdown()
```

### What happens

1. `firstops.init()` starts a local HTTP proxy on `127.0.0.1:9322`
2. Your MCP client sends requests to `localhost:9322/mcp/...`
3. The proxy signs each request with a DPoP proof (RFC 9449, ES256)
4. The signed request is forwarded to the FirstOps gateway
5. SSE streaming responses are proxied back with URLs rewritten to localhost

### Configuration

| Parameter | Default | Description |
|-----------|---------|-------------|
| `agent_id` | *required* | Your agent's principal ID |
| `private_key_pem` | *required* | EC P-256 private key (PEM format) |
| `port` | `9322` | Local proxy port |
| `gateway_url` | `https://api.firstops.ai` | FirstOps gateway URL |

## Requirements

- Python 3.10+
- Dependencies: `cryptography`, `httpx`

## Development

```bash
git clone https://github.com/firstops-dev/firstops-python.git
cd firstops-python
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## License

MIT
