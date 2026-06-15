"""OpenAI Agents SDK agent governed by FirstOps.

Two wiring points (different from LangGraph/Claude):
  - LLM: route the Agents SDK's default OpenAI client at the sidecar chain-link
    (chat-completions API; tracing disabled so traces don't bypass governance).
  - Tools: attach the FirstOps guardrail PER `@function_tool`
    (OpenAI Agents has no agent-level tool guardrail). The guardrail is
    block-only — it can deny a tool, but can't rewrite its args.

Run with the env vars in README.md (needs OPENAI_API_KEY).
"""

import asyncio
import os

import firstops
from agents import (
    Agent,
    Runner,
    function_tool,
    set_default_openai_client,
    set_tracing_disabled,
)
from firstops.integrations.openai_agents import firstops_tool_input_guardrail
from openai import AsyncOpenAI

from _shared import load_config, trace


async def main():
    cfg = load_config()
    fo = firstops.init(
        cfg["agent_id"], cfg["key_pem"], gateway_url=cfg["gateway"], port=cfg["port"]
    )
    trace(fo)
    try:
        # LLM calls -> sidecar chain-link (-> OpenAI). Key passes through.
        client = AsyncOpenAI(
            base_url=firstops.llm_base_url("openai"),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        set_default_openai_client(client)
        # Use the Agents SDK's native Responses API — the sidecar forwards it to
        # OpenAI unchanged (the default model is a reasoning model that needs it).
        set_tracing_disabled(True)

        guard = firstops_tool_input_guardrail(fo)  # one guardrail, attached per tool

        @function_tool(tool_input_guardrails=[guard])
        def get_weather(city: str) -> str:
            """Get the current weather for a city."""
            print(f"   [TOOL get_weather] city={city}")
            return f"It is 21C and sunny in {city}."

        @function_tool(tool_input_guardrails=[guard])
        def send_email(to: str, body: str) -> str:
            """Send an email to a recipient."""
            print(f"   [TOOL send_email] to={to} body={body!r}")
            return "email sent"

        agent = Agent(name="assistant", tools=[get_weather, send_email])

        print("\n>>> running OpenAI Agents SDK agent\n")
        result = await Runner.run(
            agent,
            "Check the weather in Paris, then email it to alice@example.com.",
        )
        print(f"\n>>> final answer:\n{result.final_output}\n")
    finally:
        firstops.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
