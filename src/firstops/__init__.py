"""FirstOps SDK — secure MCP proxy sidecar with DPoP authentication and management API."""

from firstops._runtime import (
    Runtime,
    init,
    llm_base_url,
    mcp_url,
    runtime,
    shutdown,
)
from firstops.client import (
    Agent,
    Connection,
    FirstOps,
    FirstOpsError,
    ParamDefinition,
    ServerTemplate,
)
from firstops.coverage import capability, coverage_report, ungoverned_tools
from firstops.enforcement import EnforcementClient
from firstops.events import ActionEvent, Decision
from firstops.llm import anthropic_client, configure_llm_env, openai_client
from firstops.proxy import current_agent_id, is_running
from firstops.tools import FirstOpsPolicyError, tool

__all__ = [
    # management client
    "FirstOps",
    "FirstOpsError",
    "Agent",
    "Connection",
    "ServerTemplate",
    "ParamDefinition",
    # runtime
    "init",
    "shutdown",
    "runtime",
    "Runtime",
    "is_running",
    "current_agent_id",
    # base API — tool governance
    "tool",
    "FirstOpsPolicyError",
    # base API — LLM chain-link + MCP
    "llm_base_url",
    "mcp_url",
    "openai_client",
    "anthropic_client",
    "configure_llm_env",
    # enforcement surface
    "EnforcementClient",
    "ActionEvent",
    "Decision",
    # coverage honesty
    "capability",
    "coverage_report",
    "ungoverned_tools",
]
