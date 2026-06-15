"""LangGraph adapter — agent middleware that governs every tool call.

Built on LangChain v1 agent middleware (`wrap_tool_call`), which can block
(return a ToolMessage instead of running the tool) and mutate (rewrite the
tool args). One middleware governs all tools the agent calls, including
framework built-ins (`ShellTool`, `SQLDatabaseToolkit`, …) the developer never
authored.

Usage::

    from langchain.agents import create_agent
    from firstops.integrations.langgraph import FirstOpsMiddleware

    agent = create_agent(model, tools=[...], middleware=[FirstOpsMiddleware(fo)])

Coverage honesty: middleware attaches per compiled graph and does NOT
auto-propagate into subgraphs. A subgraph built without FirstOps middleware is
ungoverned — wire one per graph. (Detect-and-warn for subgraphs is tracked for
a later milestone.)

The decision logic is `firstops.integrations._common.decide` (framework-free,
tested); `FirstOpsMiddleware` is the thin lazily-imported shell targeting the
LangChain v1 middleware API.
"""

from __future__ import annotations

from firstops import _runtime
from firstops.integrations._common import (
    ACTION_DENY,
    ACTION_MODIFY,
    HARNESS_LANGGRAPH,
    decide,
)


def FirstOpsMiddleware(fo=None):
    """Return a LangChain agent middleware instance that governs tool calls."""
    rt = fo if fo is not None else _runtime.runtime()
    try:
        from langchain.agents.middleware import AgentMiddleware
        from langchain_core.messages import ToolMessage
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError(
            "langchain/langgraph not installed: pip install langchain langgraph"
        ) from e

    class _FirstOpsMiddleware(AgentMiddleware):
        def wrap_tool_call(self, request, handler):
            call = getattr(request, "tool_call", None) or {}
            name = call.get("name", "") or ""
            args = call.get("args", {}) or {}
            action, payload = decide(
                rt, name, args, can_apply_modify=True, harness=HARNESS_LANGGRAPH
            )
            if action == ACTION_DENY:
                return ToolMessage(
                    content=f"blocked by FirstOps policy: {payload}",
                    tool_call_id=call.get("id", ""),
                    status="error",
                )
            if action == ACTION_MODIFY:
                request = request.override(
                    tool_call={**request.tool_call, "args": payload}
                )
            return handler(request)

        async def awrap_tool_call(self, request, handler):
            call = getattr(request, "tool_call", None) or {}
            name = call.get("name", "") or ""
            args = call.get("args", {}) or {}
            action, payload = decide(
                rt, name, args, can_apply_modify=True, harness=HARNESS_LANGGRAPH
            )
            if action == ACTION_DENY:
                return ToolMessage(
                    content=f"blocked by FirstOps policy: {payload}",
                    tool_call_id=call.get("id", ""),
                    status="error",
                )
            if action == ACTION_MODIFY:
                request = request.override(
                    tool_call={**request.tool_call, "args": payload}
                )
            return await handler(request)

    return _FirstOpsMiddleware()
