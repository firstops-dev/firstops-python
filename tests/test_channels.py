"""Channel classification table — classify() and mcp_info().

The classifier mirrors the daemon's tool_name -> channel mapping. These are
table-driven and probe the malformed-name boundary: too-few parts, empty tool,
the literal prefix, casing, and unicode.
"""

from __future__ import annotations

import pytest

from firstops.channels import classify, mcp_info
from firstops.events import CHANNEL_MCP, CHANNEL_SYSTEM_TOOLS


@pytest.mark.parametrize(
    "name,expected",
    [
        ("mcp__github__create_issue", CHANNEL_MCP),
        ("mcp__s__t", CHANNEL_MCP),
        ("mcp__", CHANNEL_SYSTEM_TOOLS),  # malformed -> system_tools (agrees w/ mcp_info)
        ("mcp__server__", CHANNEL_SYSTEM_TOOLS),  # empty tool -> system_tools
        ("mcp", CHANNEL_SYSTEM_TOOLS),  # not the prefix
        ("MCP__x__y", CHANNEL_SYSTEM_TOOLS),  # case-sensitive
        ("bash", CHANNEL_SYSTEM_TOOLS),
        ("read_file", CHANNEL_SYSTEM_TOOLS),
        ("", CHANNEL_SYSTEM_TOOLS),
        ("mcp__café__tool", CHANNEL_MCP),  # unicode server name
    ],
)
def test_classify_table(name, expected):
    assert classify(name) == expected


@pytest.mark.parametrize(
    "name,server,tool",
    [
        ("mcp__github__create_issue", "github", "create_issue"),
        ("mcp__s__a__b", "s", "a__b"),  # tool may itself contain __
        ("mcp__café__tool", "café", "tool"),  # unicode preserved
    ],
)
def test_mcp_info_parses_server_and_tool(name, server, tool):
    info = mcp_info(name)
    assert info is not None
    assert info.server == server
    assert info.tool == tool


@pytest.mark.parametrize("name", ["mcp__", "mcp", "bash", "", "mcp_single"])
def test_mcp_info_returns_none_for_unparseable(name):
    assert mcp_info(name) is None


def test_mcp_info_empty_tool_is_rejected_and_classified_system():
    """FIXED: `mcp__server__` (empty tool) is malformed — mcp_info returns None
    and classify falls to system_tools, so the two never disagree (we never emit
    a channel=mcp event with no mcp metadata)."""
    assert classify("mcp__server__") == CHANNEL_SYSTEM_TOOLS
    assert mcp_info("mcp__server__") is None


def test_classify_and_mcp_info_fully_agree():
    """FIXED: classify == MCP iff mcp_info parses. No exceptions — the degenerate
    literal `mcp__` is system_tools and yields no info."""
    for name in ("mcp__", "mcp__server__", "mcp__github__create_issue", "bash"):
        is_mcp = mcp_info(name) is not None
        assert (classify(name) == CHANNEL_MCP) == is_mcp
