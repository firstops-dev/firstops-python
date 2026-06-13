"""Claude Agent SDK agent governed by FirstOps.

The Claude Agent SDK runs tools (Bash, Write, Read, MCP) inside the Claude Code
subprocess. FirstOps governs each one via a single PreToolUse hook — block,
rewrite args (updatedInput), or allow — the daemon model, in-process.

`permission_mode="bypassPermissions"` makes the FirstOps hook the sole gate:
a hook `deny` still blocks; everything else flows. Run with the env vars in
README.md (no OpenAI key needed — Claude uses your local Claude Code auth).
"""

import asyncio
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
    fo = firstops.init(
        cfg["agent_id"], cfg["key_pem"], gateway_url=cfg["gateway"], port=cfg["port"]
    )
    trace(fo)
    WORKDIR.mkdir(exist_ok=True)
    try:
        options = ClaudeAgentOptions(
            hooks=firstops_hooks(fo),  # ← every tool call governed by FirstOps
            allowed_tools=["Bash", "Write", "Read"],
            permission_mode="bypassPermissions",
            cwd=str(WORKDIR),
        )
        prompt = (
            "Create a file named greeting.txt containing exactly "
            "'Hello from a FirstOps-governed Claude agent'. "
            "Then run a bash command to print today's date. "
            "Finally, read greeting.txt back and report its contents."
        )
        print("\n>>> running Claude agent (tools governed via PreToolUse hook)\n")
        async for message in query(prompt=prompt, options=options):
            if isinstance(message, AssistantMessage):
                for block in message.content:
                    if isinstance(block, TextBlock) and block.text.strip():
                        print(f"   [CLAUDE] {block.text.strip()[:160]}")
                    elif isinstance(block, ToolUseBlock):
                        print(f"   [TOOL-USE] {block.name}  {block.input}")
            elif isinstance(message, ResultMessage):
                print(f"\n>>> result:\n{getattr(message, 'result', message)}")
    finally:
        firstops.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
