"""Claude Agent SDK adapter — the daemon model, in-process.

A single ``PreToolUse`` hook governs every tool the agent calls: built-ins
(``Bash``, ``Write``, ``WebFetch``), MCP tools (``mcp__*``), and in-process
SDK-MCP tools — with both block (deny) and argument rewrite (updatedInput).

Usage::

    from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions
    from firstops.integrations.claude import firstops_hooks

    options = ClaudeAgentOptions(hooks=firstops_hooks(fo))
    async with ClaudeSDKClient(options=options) as client:
        ...

Targets the Claude Agent SDK PreToolUse hook contract (hookSpecificOutput /
permissionDecision). The governance logic (`_govern_pre_tool_use`) is
framework-free and unit-tested; `firstops_hooks` is the thin lazily-imported
shell.
"""

from __future__ import annotations

from typing import Any

from firstops import _runtime
from firstops.integrations._common import (
    ACTION_DENY,
    ACTION_MODIFY,
    HARNESS_CLAUDE,
    decide,
)


def _govern_pre_tool_use(rt, input_data: dict[str, Any]) -> dict[str, Any]:
    """Map a PreToolUse hook input to a hook response. Pure / testable.

    Returns ``{}`` (allow) when there's nothing to do, a deny envelope, or an
    allow-with-updatedInput envelope for a scrub.
    """
    tool_name = input_data.get("tool_name", "") or ""
    tool_input = input_data.get("tool_input")
    # Claude PreToolUse updatedInput can rewrite args → request-path scrub applies.
    action, payload = decide(
        rt, tool_name, tool_input, can_apply_modify=True, harness=HARNESS_CLAUDE
    )
    if action == ACTION_DENY:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "deny",
                "permissionDecisionReason": payload,
            }
        }
    # Only emit an updatedInput envelope when there's a non-empty replacement —
    # Claude's updatedInput is a FULL replacement of tool_input, so an empty
    # dict would wipe every argument. Empty/odd payload → allow unchanged.
    if action == ACTION_MODIFY and isinstance(payload, dict) and payload:
        return {
            "hookSpecificOutput": {
                "hookEventName": "PreToolUse",
                "permissionDecision": "allow",
                "updatedInput": payload,
            }
        }
    return {}


def firstops_hooks(fo=None) -> dict[str, Any]:
    """Return a Claude Agent SDK ``hooks`` config that governs all tool calls."""
    rt = fo if fo is not None else _runtime.runtime()
    try:
        from claude_agent_sdk import HookMatcher
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "claude-agent-sdk is not installed: pip install claude-agent-sdk"
        ) from e

    async def _pre_tool_use(input_data, tool_use_id, context):
        return _govern_pre_tool_use(rt, input_data)

    # matcher=None is the match-ALL contract; "*" would match only a tool
    # literally named "*" (i.e. nothing) and silently disable governance.
    return {"PreToolUse": [HookMatcher(matcher=None, hooks=[_pre_tool_use])]}
