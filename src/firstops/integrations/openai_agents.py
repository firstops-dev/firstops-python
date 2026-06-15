"""OpenAI Agents SDK adapter — tool input guardrails.

Uses the SDK's tool input guardrail to block a tool call before it runs.

**Attachment is per-tool, not per-agent.** OpenAI Agents has no agent-level
tool-input guardrail — `tool_input_guardrails` is a field on `@function_tool` /
`FunctionTool`. So wire the FirstOps guardrail onto each function tool::

    from agents import function_tool
    from firstops.integrations.openai_agents import firstops_tool_input_guardrail

    guard = firstops_tool_input_guardrail(fo)

    @function_tool(tool_input_guardrails=[guard])
    def send_email(to: str, body: str): ...

Capability note (honest): OpenAI Agents tool guardrails are **read-only** — they
can block but cannot mutate tool arguments. So argument *scrub* is not available
through this adapter; a ``modify`` decision degrades to allow here. To scrub tool
args on OpenAI Agents, wrap the underlying function with ``@firstops.tool``
instead (the base-API decorator).

The decision logic (`_decide_tool`) is framework-free and tested; the guardrail
wiring is the thin lazily-imported shell.
"""

from __future__ import annotations

from typing import Any

from firstops import _runtime
from firstops.integrations._common import (
    ACTION_DENY,
    ACTION_MODIFY,
    HARNESS_OPENAI_AGENTS,
    decide,
)


def _decide_tool(rt, tool_name: str, tool_input: Any) -> tuple[bool, str]:
    """Return ``(blocked, reason)`` for a tool call.

    Guardrails can't mutate args (read-only). We leave ``can_apply_modify``
    False, so sentinel escalates a request-path scrub to deny — but defensively,
    if a ``modify`` ever reaches here we **block** (fail closed) rather than let
    unscrubbed arguments through. To scrub on OpenAI Agents, use ``@firstops.tool``.
    """
    action, payload = decide(rt, tool_name, tool_input, harness=HARNESS_OPENAI_AGENTS)
    if action == ACTION_DENY:
        return True, str(payload)
    if action == ACTION_MODIFY:
        return True, (
            "scrub required but tool-arg rewrite is unsupported on OpenAI Agents "
            "(use @firstops.tool to scrub)"
        )
    return False, ""


def firstops_tool_input_guardrail(fo=None):
    """Return a single reusable tool-input guardrail to attach per function tool.

    Pass it in each tool's ``tool_input_guardrails=[...]`` list.
    """
    rt = fo if fo is not None else _runtime.runtime()
    try:
        from agents import tool_input_guardrail
        from agents.tool_guardrails import ToolGuardrailFunctionOutput
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "openai-agents not installed: pip install openai-agents"
        ) from e

    @tool_input_guardrail
    async def _fo_tool_input_guardrail(data):
        # ToolInputGuardrailData carries .context (ToolContext) and .agent.
        # tool_arguments is a raw JSON string; coerce_input handles that.
        ctx = getattr(data, "context", None)
        tool_name = getattr(ctx, "tool_name", "") or ""
        tool_args = getattr(ctx, "tool_arguments", None)
        blocked, reason = _decide_tool(rt, tool_name, tool_args)
        if blocked:
            return ToolGuardrailFunctionOutput.reject_content(
                message=f"blocked by FirstOps policy: {reason}"
            )
        return ToolGuardrailFunctionOutput.allow()

    return _fo_tool_input_guardrail
