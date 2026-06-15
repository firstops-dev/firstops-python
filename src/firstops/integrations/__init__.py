"""Harness adapters — one-touch governance for supported agent frameworks.

Each adapter wires the FirstOps enforcement spine into a framework's official,
intervention-capable extension point:

  - ``firstops.integrations.claude``        — Claude Agent SDK ``PreToolUse`` hook
  - ``firstops.integrations.langgraph``     — LangGraph agent middleware (wrap_tool_call)
  - ``firstops.integrations.openai_agents`` — OpenAI Agents SDK tool guardrails

Adapters import their framework lazily, so importing this package never
requires the frameworks to be installed.
"""
