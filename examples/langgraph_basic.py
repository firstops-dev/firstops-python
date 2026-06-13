"""LangGraph agent governed by FirstOps — tool calls + LLM, no MCP.

Every LLM call routes through the sidecar chain-link; every tool call is
intercepted by FirstOpsMiddleware. Run with the env vars in README.md.
"""

import os

import firstops
from firstops.integrations.langgraph import FirstOpsMiddleware
from langchain.agents import create_agent
from langchain_core.tools import tool as lc_tool
from langchain_openai import ChatOpenAI

from _shared import load_config, trace


@lc_tool
def get_weather(city: str) -> str:
    """Get the current weather for a city."""
    print(f"   [TOOL get_weather] city={city}")
    return f"It is 21C and sunny in {city}."


@lc_tool
def send_email(to: str, body: str) -> str:
    """Send an email to a recipient."""
    print(f"   [TOOL send_email] to={to} body={body!r}")
    return "email sent"


def main():
    cfg = load_config()
    fo = firstops.init(
        cfg["agent_id"], cfg["key_pem"], gateway_url=cfg["gateway"], port=cfg["port"]
    )
    trace(fo)
    try:
        llm = ChatOpenAI(
            model="gpt-4o-mini",
            base_url=firstops.llm_base_url("openai"),
            api_key=os.environ["OPENAI_API_KEY"],
        )
        agent = create_agent(
            model=llm,
            tools=[get_weather, send_email],
            middleware=[FirstOpsMiddleware(fo)],
        )
        print("\n>>> invoking agent\n")
        result = agent.invoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": "Check the weather in Paris, then email it to alice@example.com.",
                    }
                ]
            }
        )
        print(f"\n>>> final answer:\n{result['messages'][-1].content}\n")
    finally:
        firstops.shutdown()


if __name__ == "__main__":
    main()
