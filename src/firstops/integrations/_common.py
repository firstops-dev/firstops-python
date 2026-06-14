"""Shared governance core for harness adapters.

The adapters all reduce to: build a pre_tool_use event, ask sentinel, translate
the Decision into the framework's native verb (deny / mutate / allow). That
reduction lives here so every adapter shares one tested implementation.
"""

from __future__ import annotations

import json
from typing import Any

from firstops.channels import classify, mcp_info
from firstops.events import CHANNEL_MCP, EVENT_PRE_TOOL_USE, ActionEvent, Decision

# A neutral decision vocabulary the adapters translate into framework types.
ACTION_ALLOW = "allow"
ACTION_DENY = "deny"
ACTION_MODIFY = "modify"

# Harness identifiers stamped into event metadata (audit/filtering).
HARNESS_LANGGRAPH = "langgraph"
HARNESS_CLAUDE = "claude-agent-sdk"
HARNESS_OPENAI_AGENTS = "openai-agents"
HARNESS_GOOGLE_ADK = "google-adk"


def _json_safe(value: Any) -> Any:
    """Return a JSON-serializable view of ``value`` (str fallback)."""
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return str(value)


def coerce_input(tool_input: Any) -> dict[str, Any]:
    """Normalize a framework's tool input into a JSON-able dict for the event.

    Never raises and always returns a JSON-serializable dict (bytes decode with
    ``errors="replace"``; non-serializable objects are stringified) so a tool
    passing binary/exotic args can't crash the agent loop.
    """
    if isinstance(tool_input, dict):
        return {k: _json_safe(v) for k, v in tool_input.items()}
    if isinstance(tool_input, bytes):
        tool_input = tool_input.decode("utf-8", errors="replace")
    if isinstance(tool_input, str):
        try:
            parsed = json.loads(tool_input)
            if isinstance(parsed, dict):
                return {k: _json_safe(v) for k, v in parsed.items()}
        except (ValueError, TypeError):
            pass
        return {"input": tool_input}
    if tool_input is None:
        return {}
    return {"input": _json_safe(tool_input)}


def govern_tool(
    rt,
    tool_name: str,
    tool_input: Any,
    can_apply_modify: bool = False,
    harness: str = "",
) -> Decision:
    """Evaluate a tool call via the enforcement spine. Allows if no runtime.

    ``can_apply_modify``: True when the adapter can rewrite the tool args before
    execution (Claude updatedInput, LangGraph arg rewrite) — lets sentinel ship
    a request-path scrub as ``modify`` instead of escalating to deny. Adapters
    that can't mutate (OpenAI guardrails) leave it False.

    ``harness``: the producing framework, stamped into event metadata when known.

    Hardened to never raise into the agent loop: a None/odd tool_name and
    non-serializable inputs are coerced rather than propagated.
    """
    if rt is None:
        return Decision(action="allow")
    name = tool_name or ""
    channel = classify(name)
    event = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE,
        tool_name=name,
        channel=channel,
        # coerce_input already returns a fresh dict — no aliasing of the
        # caller's live args (which frameworks may mutate after the call).
        tool_input=coerce_input(tool_input),
        mcp=mcp_info(name) if channel == CHANNEL_MCP else None,
        producer_can_apply_modify=can_apply_modify,
        metadata={"harness": harness} if harness else None,
    )
    return rt.enforcement.evaluate(event)


def modified_input(decision: Decision) -> dict[str, Any] | None:
    """Return the scrubbed input dict from a modify decision, or None."""
    if decision.modified and decision.modified_payload:
        try:
            value = json.loads(decision.modified_payload)
            if isinstance(value, dict):
                return value
        except (ValueError, TypeError):
            pass
    return None


def decide(
    rt,
    tool_name: str,
    tool_input: Any,
    can_apply_modify: bool = False,
    harness: str = "",
) -> tuple[str, Any]:
    """Reduce a tool call to ``(action, payload)``:

    - ``("deny", reason)`` — block the call
    - ``("modify", new_input_dict)`` — proceed with scrubbed input
    - ``("allow", None)`` — proceed unchanged

    ``can_apply_modify`` and ``harness`` are forwarded to the event (see govern_tool).
    """
    decision = govern_tool(
        rt, tool_name, tool_input, can_apply_modify=can_apply_modify, harness=harness
    )
    if decision.blocked:
        return ACTION_DENY, decision.reason or "blocked by FirstOps policy"
    new_input = modified_input(decision)
    if new_input is not None:
        return ACTION_MODIFY, new_input
    return ACTION_ALLOW, None
