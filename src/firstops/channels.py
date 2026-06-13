"""Channel classification for tool calls.

Mirrors the daemon's tool_name → channel mapping so sentinel sees SDK tool
calls the same way it sees coding-agent hook events. MCP tools (named
``mcp__<server>__<tool>``) classify as the MCP channel; everything else a
decorated tool does is ``system_tools``. LLM traffic is stamped ``llm``
directly by the sidecar's LLM route, not by this classifier.
"""

from __future__ import annotations

from firstops.events import CHANNEL_MCP, CHANNEL_SYSTEM_TOOLS, MCPInfo

_MCP_PREFIX = "mcp__"


def classify(tool_name: str) -> str:
    """Return the channel a tool call belongs to.

    A name is MCP only if it's a *well-formed* ``mcp__<server>__<tool>`` (so
    classify and :func:`mcp_info` always agree — we never emit a ``channel=mcp``
    event with no mcp metadata). A malformed ``mcp__`` name (e.g. ``mcp__foo``)
    is treated as ``system_tools``.
    """
    if mcp_info(tool_name) is not None:
        return CHANNEL_MCP
    return CHANNEL_SYSTEM_TOOLS


def mcp_info(tool_name: str) -> MCPInfo | None:
    """Parse ``mcp__<server>__<tool>`` into MCPInfo, or None if not a well-formed MCP name."""
    if not tool_name.startswith(_MCP_PREFIX):
        return None
    parts = tool_name.split("__")
    # Require non-empty server and tool segments.
    if len(parts) >= 3 and parts[1] and "__".join(parts[2:]):
        return MCPInfo(server=parts[1], tool="__".join(parts[2:]))
    return None
