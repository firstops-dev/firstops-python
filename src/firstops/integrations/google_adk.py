"""Google ADK adapter — `before_tool_callback` governs every tool call.

ADK's agent-level `before_tool_callback` can both **block** a tool (return a
result, which short-circuits the call) and **rewrite its args** (mutate the args
dict in place) — so FirstOps gets block and scrub on Google ADK.

Usage::

    from google.adk.agents import LlmAgent
    from firstops.integrations.google_adk import firstops_before_tool_callback

    agent = LlmAgent(
        name="assistant",
        model=...,
        tools=[...],
        before_tool_callback=firstops_before_tool_callback(fo),
    )

The callback is a plain function (ADK invokes it as
``callback(tool=, args=, tool_context=)``), so this adapter needs no ADK import.
"""

from __future__ import annotations

from typing import Any

from firstops import _runtime
from firstops.integrations._common import (
    ACTION_DENY,
    ACTION_MODIFY,
    HARNESS_GOOGLE_ADK,
    decide,
)


def firstops_before_tool_callback(fo=None):
    """Return a ``before_tool_callback`` that governs every tool call."""
    rt = fo if fo is not None else _runtime.runtime()

    def _before_tool(tool, args, tool_context) -> dict[str, Any] | None:
        tool_name = getattr(tool, "name", "") or ""
        action, payload = decide(
            rt, tool_name, args, can_apply_modify=True, harness=HARNESS_GOOGLE_ADK
        )
        if action == ACTION_DENY:
            # A non-None return short-circuits the tool; this becomes the result
            # the model sees.
            return {"status": "denied", "error": f"blocked by FirstOps policy: {payload}"}
        if action == ACTION_MODIFY and isinstance(payload, dict) and isinstance(args, dict):
            # Rewrite the call's args in place (full replacement with scrubbed input).
            args.clear()
            args.update(payload)
        return None

    return _before_tool
