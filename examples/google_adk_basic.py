"""Google ADK agent governed by FirstOps.

One `before_tool_callback` governs every tool call (block + rewrite args). The
LLM runs through the sidecar chain-link via LiteLLM pointed at OpenAI.

Run with the env vars in README.md (needs OPENAI_API_KEY).
"""

import asyncio
import os

import firstops
from firstops.integrations.google_adk import firstops_before_tool_callback
from google.adk.agents import LlmAgent
from google.adk.models.lite_llm import LiteLlm
from google.adk.runners import InMemoryRunner
from google.genai import types

from _shared import load_config, trace

APP = "firstops-demo"


def get_weather(city: str) -> dict:
    """Get the current weather for a city."""
    print(f"   [TOOL get_weather] city={city}")
    return {"weather": f"21C and sunny in {city}"}


def send_email(to: str, body: str) -> dict:
    """Send an email to a recipient."""
    print(f"   [TOOL send_email] to={to} body={body!r}")
    return {"status": "sent"}


async def main():
    cfg = load_config()
    fo = firstops.init(
        cfg["agent_id"], cfg["key_pem"], gateway_url=cfg["gateway"], port=cfg["port"]
    )
    trace(fo)
    try:
        agent = LlmAgent(
            name="assistant",
            model=LiteLlm(
                model="openai/gpt-4o-mini",
                api_base=firstops.llm_base_url("openai"),  # -> sidecar chain-link
                api_key=os.environ["OPENAI_API_KEY"],
            ),
            instruction="You are a helpful assistant.",
            tools=[get_weather, send_email],
            before_tool_callback=firstops_before_tool_callback(fo),  # governs tools
        )
        runner = InMemoryRunner(agent=agent, app_name=APP)
        session = await runner.session_service.create_session(app_name=APP, user_id="u1")
        msg = types.Content(
            role="user",
            parts=[types.Part(text="Check the weather in Paris, then email it to alice@example.com.")],
        )
        print("\n>>> running Google ADK agent\n")
        async for event in runner.run_async(
            user_id="u1", session_id=session.id, new_message=msg
        ):
            if event.content and event.content.parts:
                for part in event.content.parts:
                    if getattr(part, "text", None) and part.text.strip():
                        print(f"   [ADK] {part.text.strip()[:160]}")
    finally:
        firstops.shutdown()


if __name__ == "__main__":
    asyncio.run(main())
