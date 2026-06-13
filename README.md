# FirstOps Python SDK

The FirstOps SDK has two halves:

1. **Management client** (`FirstOps`) — programmatically create agents, register MCP connections, and manage their lifecycle from your backend. Authenticates with a tenant-scoped API key.
2. **Runtime proxy** (`firstops.init`) — a lightweight in-process sidecar that transparently signs every MCP request with a [DPoP](https://datatracker.ietf.org/doc/html/rfc9449) proof. Runs inside the agent process.

The two halves are used at different points in an agent's lifecycle. The management client runs in your **platform code** (the backend that provisions agents). The runtime proxy runs inside the **agent itself** (the process that calls MCP tools).

## Install

```bash
pip install firstops
```

## Requirements

- Python 3.10+
- Dependencies: `cryptography`, `httpx`

---

## 1. Management Client — Provisioning Agents

Use this in your platform's backend code to create agents and wire up their MCP connections on demand.

### Get an API key

1. Log in to the FirstOps dashboard as an admin.
2. Go to **Settings → API Keys** and create a key with the scopes you need:
   - `agents:write` — create and delete agent principals
   - `agents:read` — list and get agents
   - `connections:write` — register and delete MCP connections
   - `connections:read` — list connections
3. Copy the raw key (starts with `fo_key_`). It is shown **once** — store it in your secrets manager.

### Quick Start

```python
from firstops import FirstOps

# Initialize the management client
client = FirstOps(api_key="fo_key_...")

# 1. Create an agent (returns principal ID, token, and private key)
agent = client.agents.create(name="research-bot-for-alice")
print(f"Agent ID:    {agent.id}")
print(f"Agent token: {agent.token}")       # fo_agent_<id> — used in Authorization header
print(f"Private key: {agent.private_key}") # PEM — shown once, save it securely

# 2. Register MCP connections for the agent
slack_conn = client.connections.register(
    principal_id=agent.id,
    name="slack",
    upstream_url="https://mcp.slack.com/sse",
)

gdrive_conn = client.connections.register(
    principal_id=agent.id,
    name="google-drive",
    upstream_url="https://mcp.google.com/drive/sse",
    auth_type="oauth2",
)

# 3. List the agent's connections
for conn in client.connections.list(principal_id=agent.id):
    print(f"{conn.name} — {conn.status}")

# 4. Remove a connection when the user no longer needs it
client.connections.delete(slack_conn.id)

# 5. Delete the agent when the user deletes their instance
client.agents.delete(agent.id)
```

### Context Manager

`FirstOps` is also usable as a context manager for automatic connection cleanup:

```python
with FirstOps(api_key="fo_key_...") as client:
    agents = client.agents.list()
```

### API Reference

#### `FirstOps(api_key, base_url="https://api.firstops.ai", timeout=30.0)`

The top-level management client.

| Parameter | Default | Description |
|-----------|---------|-------------|
| `api_key` | *required* | Tenant-scoped API key (must start with `fo_key_`) |
| `base_url` | `https://api.firstops.ai` | FirstOps API base URL |
| `timeout` | `30.0` | HTTP timeout in seconds |

#### `client.agents`

| Method | Required Scope | Returns |
|--------|----------------|---------|
| `create(name: str)` | `agents:write` | `Agent` (with `private_key`) |
| `list()` | `agents:read` | `list[Agent]` |
| `get(agent_id: str)` | `agents:read` | `Agent` |
| `delete(agent_id: str)` | `agents:write` | `None` |

**Note:** `agent.private_key` is only populated on `create()`. It is never returned again — store it alongside your agent record at creation time.

#### `client.connections`

| Method | Required Scope | Returns |
|--------|----------------|---------|
| `register(principal_id, name, upstream_url, ...)` | `connections:write` | `Connection` |
| `list(principal_id=None)` | `connections:read` | `list[Connection]` |
| `delete(connection_id: str)` | `connections:write` | `None` |

Full signature for `register`:

```python
client.connections.register(
    principal_id="pr_...",                  # required — the agent's principal ID
    name="slack",                           # required — display name
    upstream_url="https://mcp.slack.com/sse", # required — remote MCP server URL
    auth_type="",                           # optional — "oauth2", "bearer", etc.
    transport_type="",                      # optional — "sse" or empty for auto-detect
    upstream_headers=None,                  # optional — dict of headers to forward
    upstream_query_params=None,             # optional — dict of query params
    source="sdk",                           # optional — audit label
)
```

### Error Handling

All API errors raise `FirstOpsError`:

```python
from firstops import FirstOps, FirstOpsError

try:
    client.agents.delete("pr_does_not_exist")
except FirstOpsError as e:
    print(f"Error {e.status_code}: {e.message}")
```

---

## 2. Runtime Proxy — Securing MCP Calls Inside an Agent

Use this inside the agent process itself. It starts a local HTTP proxy that transparently adds DPoP-signed authentication headers to every MCP request your agent makes — no changes to your agent code required.

### Quick Start

```python
import firstops

# Start the proxy sidecar (runs in background thread)
firstops.init(
    agent_id="your-agent-id",                     # from client.agents.create(...).id
    private_key_pem=open("agent-key.pem").read(), # from client.agents.create(...).private_key
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
| `agent_id` | *required* | Agent's principal ID (without `fo_agent_` prefix) |
| `private_key_pem` | *required* | EC P-256 private key in PEM format |
| `port` | `9322` | Local proxy port |
| `gateway_url` | `https://api.firstops.ai` | FirstOps gateway URL |

---

## End-to-End Example: Dynamic Agent Platform

Here's how the two halves fit together in a typical "SaaS that offers AI agents" platform:

```python
# ─── Platform backend (your FastAPI/Django service) ─────────────
from firstops import FirstOps

firstops = FirstOps(api_key=os.environ["FIRSTOPS_API_KEY"])

@app.post("/users/{user_id}/agents")
def create_user_agent(user_id: str, config: dict):
    # Create a FirstOps agent identity for this end-user's instance
    agent = firstops.agents.create(name=f"research-bot-{user_id}")

    # Store the agent credentials alongside the user's record
    db.save_agent(
        user_id=user_id,
        agent_id=agent.id,
        private_key=agent.private_key,  # encrypt this at rest
    )

    # Wire up the tools the user selected
    for tool in config["selected_tools"]:
        firstops.connections.register(
            principal_id=agent.id,
            name=tool["name"],
            upstream_url=tool["url"],
        )

    return {"agent_id": agent.id}

@app.delete("/users/{user_id}/agents/{agent_id}")
def delete_user_agent(user_id: str, agent_id: str):
    firstops.agents.delete(agent_id)  # cascades to connections
    db.delete_agent(agent_id)


# ─── Agent runtime (the worker process that actually runs the agent) ───
import firstops as fo_runtime

def run_agent_task(agent_id: str, private_key: str, task: str):
    fo_runtime.init(agent_id=agent_id, private_key_pem=private_key)
    try:
        # Your MCP-using agent logic — point MCP clients at 127.0.0.1:9322
        mcp_client = MCPClient(base_url="http://127.0.0.1:9322")
        return mcp_client.run(task)
    finally:
        fo_runtime.shutdown()
```

---

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
