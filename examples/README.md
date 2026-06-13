# FirstOps SDK — Examples

Runnable agents that govern every LLM call, tool call, and MCP call through
FirstOps.

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ..                      # the FirstOps SDK (this repo)
pip install "langchain>=1.0" langgraph langchain-openai langchain-mcp-adapters openai
```

## Config (env vars)

| Var | Meaning |
|-----|---------|
| `FO_AGENT_ID` | Agent principal ID (UUID) from `client.agents.create(...)` |
| `FO_PRIVATE_KEY_PATH` | Path to the agent's EC P-256 private-key PEM |
| `FO_GATEWAY` | FirstOps gateway base URL (default `https://api.firstops.dev`) |
| `FO_PORT` | Local sidecar port (default `9322`) |
| `OPENAI_API_KEY` | OpenAI key — passes through the sidecar to OpenAI, never stored |
| `FO_MCP_CONNECTION_ID` | (MCP example) a registered MCP connection ID for the agent |

## Examples

- **`langgraph_basic.py`** — a LangGraph agent with two local tools and the LLM
  routed through the sidecar chain-link. Exercises tool governance + LLM
  governance.
- **`langgraph_notion_mcp.py`** — adds a Notion MCP server (via the sidecar's
  MCP proxy) and a local `write_to_file` tool, then asks the agent to fetch
  customer info from Notion and write it to a local file. Exercises **MCP +
  local tool** governance together.

```bash
FO_AGENT_ID=... FO_PRIVATE_KEY_PATH=... OPENAI_API_KEY=... \
  python langgraph_basic.py

FO_AGENT_ID=... FO_PRIVATE_KEY_PATH=... OPENAI_API_KEY=... \
  FO_MCP_CONNECTION_ID=... python langgraph_notion_mcp.py
```

Each run prints a `[GOVERN]` line for every governed action (channel, tool,
decision), so you can see exactly what FirstOps evaluated.
