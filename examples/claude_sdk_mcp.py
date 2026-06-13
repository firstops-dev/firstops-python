"""Claude Agent SDK agent with a Notion MCP server, governed by FirstOps.

Same daemon-model hook as claude_sdk_basic.py, but now the agent also talks to
a Notion MCP server (through the FirstOps proxy). The single PreToolUse hook
governs BOTH the MCP tool calls (mcp__notion__*) and the built-in Write tool.

Run with the env vars in README.md, incl. FO_MCP_CONNECTION_ID. Uses your local
Claude Code auth (no OpenAI key needed).
"""

import asyncio
import os
from pathlib import Path

import firstops
from claude_agent_sdk import (
    AssistantMessage,
    ClaudeAgentOptions,
    ResultMessage,
    TextBlock,
    ToolUseBlock,
    query,
)
from firstops.integrations.claude import firstops_hooks

from _shared import load_config, trace

WORKDIR = Path(__file__).parent / "claude_work"


async def main():
    cfg = load_config()
    conn_id = os.environ["FO_MCP_CONNECTION_ID"].strip()

    fo = firstops.init(
        cfg["agent_id"], cfg["key_pem"], gateway_url=cfg["gateway"], port=cfg["port"]
    )
    trace(fo)
    WORKDIR.mkdir(exist_ok=True)
    try:
        options = ClaudeAgentOptions(
            hooks=firstops_hooks(fo),  # governs both MCP and built-in tools
            mcp_servers={
                "notion": {"type": "http", "url": firstops.mcp_url(conn_id)}
            },
            allowed_tools=["Write", "Read", "mcp__notion"],
            permission_mode="bypassPermissions",
            cwd=str(WORKDIR),
        )
        prompt = (
            "Use the Notion tools to find the customer database and fetch the "
            "customer records. Then write the customer info to a file named "
            "customers_claude.txt."
        )
        print("\n>>> running Claude agent (Notion MCP + Write, governed)\n")
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        print(f"   [CLAUDE] {block.text.strip()[:140]}")
                    elif isinstance(block, ToolUseBlock):
                        print(f"   [TOOL-USE] {block.name}")
            elif isinstance(message, ResultMessage):
                print(f"\n>>> result:\n{getattr(message, 'result', message)}")
    finally:
        firstops.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
