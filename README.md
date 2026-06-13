# FirstOps Python SDK

Govern what your AI agents do. FirstOps applies identity, policy enforcement, credential brokering, and audit to every **LLM call**, **tool call**, and **MCP call** your agent makes — across LangGraph, the Claude Agent SDK, and the OpenAI Agents SDK, or any custom loop.

```bash
pip install "firstops[langgraph]"   # or [claude], [openai], [all]
```

- **Python 3.10+**
- Core deps: `cryptography`, `httpx`. Your agent framework comes in via the extra you pick.

---

## What FirstOps governs

| Surface | How it's wired | What you get |
|---|---|---|
| **LLM calls** | point the model `base_url` at the local sidecar | inspect prompts/responses, scrub PII, block, audit |
| **Tool calls** | one adapter (or `@firstops.tool`) | block / rewrite args / audit — including framework built-ins |
| **MCP servers** | point the MCP client at the local proxy | server-side policy + **credential brokering** (the agent never holds the upstream token) |

Every action is evaluated by FirstOps and returns `allow` / `deny` / `modify` — your agent logic doesn't change.

## Quick start (LangGraph)

```python
import firstops
from firstops.integrations.langgraph import FirstOpsMiddleware
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

fo = firstops.init(
    agent_id="<agent-uuid>",                      # from the FirstOps dashboard
    private_key_pem=open("agent-key.pem").read(),
)

# Route the LLM through FirstOps; wire one middleware to govern every tool call.
llm = ChatOpenAI(model="gpt-4o-mini", base_url=firstops.llm_base_url("openai"), api_key="sk-...")
agent = create_agent(model=llm, tools=[...], middleware=[FirstOpsMiddleware(fo)])

agent.invoke({"messages": [{"role": "user", "content": "..."}]})
```

The whole integration is `init()` + a `base_url` swap + one middleware. See [`examples/`](examples/) for runnable agents, including MCP.

## Other harnesses

**Claude Agent SDK** — one `PreToolUse` hook governs every tool (built-ins, MCP, custom):

```python
from claude_agent_sdk import query, ClaudeAgentOptions
from firstops.integrations.claude import firstops_hooks

options = ClaudeAgentOptions(hooks=firstops_hooks(fo), permission_mode="bypassPermissions")
async for _ in query(prompt="...", options=options):
    pass
```

**OpenAI Agents SDK** — a guardrail per tool + the model routed through the sidecar:

```python
from agents import Agent, function_tool, set_default_openai_client
from firstops.integrations.openai_agents import firstops_tool_input_guardrail
from openai import AsyncOpenAI

set_default_openai_client(AsyncOpenAI(base_url=firstops.llm_base_url("openai"), api_key="sk-..."))
guard = firstops_tool_input_guardrail(fo)

@function_tool(tool_input_guardrails=[guard])
def send_email(to: str, body: str) -> str: ...
```

**Any framework / custom loop** — the base API:

```python
@firstops.tool          # govern any callable: block / scrub args / audit
def send_email(to: str, body: str): ...
```

## MCP servers

Point your MCP client at the local proxy; FirstOps brokers the upstream credentials.

```python
from langchain_mcp_adapters.client import MultiServerMCPClient

mcp = MultiServerMCPClient({"notion": {"url": firstops.mcp_url("<connection-id>"), "transport": "streamable_http"}})
tools = await mcp.get_tools()
```

## Management client

Provision agents and connections from your backend:

```python
from firstops import FirstOps

admin = FirstOps(api_key="fo_key_...")
agent = admin.agents.create(name="research-bot")      # -> id + private_key (shown once)
admin.connections.register(principal_id=agent.id, name="slack", upstream_url="https://mcp.slack.com/sse")
```

## How it works

`firstops.init()` starts a local sidecar and establishes the agent's identity (a DPoP-bound principal — RFC 9449). Tool and LLM actions are forwarded to the FirstOps gateway, which evaluates them against your policies and returns the verdict; MCP and LLM traffic flow through the sidecar with credentials brokered. Enforcement fails open on infrastructure errors; authentication fails closed.

## Documentation

- Guides: <https://firstops.dev/docs>
- Repository: <https://github.com/firstops-dev/firstops-python>

## License

MIT
