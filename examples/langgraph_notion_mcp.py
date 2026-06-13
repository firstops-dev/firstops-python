"""LangGraph agent governed by FirstOps — MCP + local tool, end to end.

Wires a Notion MCP server (through the sidecar's MCP proxy) alongside a local
`write_to_file` tool, then asks the agent to fetch customer info from Notion and
write it to a local file. Exercises BOTH governance paths in one run:

  - MCP tool calls  -> sidecar /mcp/proxy/<connID> -> gateway (server-side
    enforcement + credential brokering) AND FirstOpsMiddleware (tool channel)
  - the local tool  -> FirstOpsMiddleware (system_tools channel)
  - the LLM         -> sidecar /llm chain-link

Run with the env vars in README.md (incl. FO_MCP_CONNECTION_ID).
"""

import asyncio
import os
from pathlib import Path

import firstops
from firstops.integrations.langgraph import FirstOpsMiddleware
from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_openai import ChatOpenAI

from _shared import load_config, trace

OUT_DIR = Path(__file__).parent / "out"


@lc_tool
def write_to_file(filename: str, content: str) -> str:
    """Write text content to a local file in the output directory."""
    OUT_DIR.mkdir(exist_ok=True)
    path = OUT_DIR / Path(filename).name  # no path traversal
    path.write_text(content)
    print(f"   [TOOL write_to_file] wrote {len(content)} bytes to {path}")
    return f"wrote {len(content)} bytes to {path}"


async def main():
    cfg = load_config()
    conn_id = os.environ["FO_MCP_CONNECTION_ID"].strip()

    fo = firstops.init(
        cfg["agent_id"], cfg["key_pem"], gateway_url=cfg["gateway"], port=cfg["port"]
    )
    trace(fo)
    try:
        # Point the MCP client at the sidecar's MCP proxy for this connection.
        # The sidecar DPoP-signs and forwards to the gateway, which brokers the
        # Notion credentials — the agent never sees them.
        mcp_client = MultiServerMCPClient(
            {"notion": {"url": firstops.mcp_url(conn_id), "transport": "streamable_http"}}
        )
        mcp_tools = await mcp_client.get_tools()
        print(f">>> Notion MCP exposed {len(mcp_tools)} tools: "
              f"{[t.name for t in mcp_tools][:8]}{' ...' if len(mcp_tools) > 8 else ''}")

        llm = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=firstops.llm_base_url("openai"),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        agent = create_agent(
            model=llm,
            tools=mcp_tools + [write_to_file],
            middleware=[FirstOpsMiddleware(fo)],
        )

        print("\n>>> invoking agent (Notion MCP + local write_to_file)\n")
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            "Find the customer database in Notion and fetch the "
                            "customer records. Print the customer info, then write "
                            "it to a local file called customers.txt."
                        ),
                    }
                ]
            }
        )
        print(f"\n>>> final answer:\n{result['messages'][-1].content}\n")
    finally:
        firstops.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
