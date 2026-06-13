"""OpenAI Agents SDK agent with a Notion MCP server, governed by FirstOps.

Governance per surface:
  - LLM        -> sidecar chain-link (set_default_openai_client at the sidecar)
  - local tool -> per-`@function_tool` FirstOps guardrail (block-only)
  - MCP server -> sidecar MCP proxy -> gateway (credential brokering +
                  server-side enforcement; the agent never holds Notion's token)

Run with the env vars in README.md, incl. FO_MCP_CONNECTION_ID.
"""

import asyncio
import os

import firstops
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_tracing_disabled,
)
from agents.mcp import MCPServerStreamableHttp
from firstops.integrations.openai_agents import firstops_tool_input_guardrail
from openai import AsyncOpenAI

from _shared import load_config, trace


async def main():
    cfg = load_config()
    conn_id = os.environ["FO_MCP_CONNECTION_ID"].strip()

    fo = firstops.init(
        cfg["agent_id"], cfg["key_pem"], gateway_url=cfg["gateway"], port=cfg["port"]
    )
    trace(fo)
    try:
        set_default_openai_client(
            AsyncOpenAI(
                base_url=firstops.llm_base_url("openai"),
                api_key=os.environ["OPENAI_API_KEY"],
            )
        )
        set_tracing_disabled(True)

        guard = firstops_tool_input_guardrail(fo)

        @function_tool(tool_input_guardrails=[guard])
        def write_to_file(filename: str, content: str) -> str:
            """Write text content to a local file."""
            with open(filename, "w") as f:
                f.write(content)
            print(f"   [TOOL write_to_file] wrote {len(content)} bytes to {filename}")
            return f"wrote {len(content)} bytes to {filename}"

        async with MCPServerStreamableHttp(
            name="notion", params={"url": firstops.mcp_url(conn_id)}
        ) as notion:
            agent = Agent(
                name="assistant",
                instructions="Use Notion to find data, and write files when asked.",
                mcp_servers=[notion],
                tools=[write_to_file],
            )
            print("\n>>> running OpenAI Agents SDK agent (Notion MCP + local tool)\n")
            result = await Runner.run(
                agent,
                "Fetch the customer records from the Notion customer database and "
                "write them to customers_openai.txt.",
            )
            print(f"\n>>> final answer:\n{result.final_output}\n")
    finally:
        firstops.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
