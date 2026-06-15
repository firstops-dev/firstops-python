"""Coverage honesty — surface what is and isn't governed.

For a security product, a *silent* coverage gap is the worst failure. This
module makes the gaps loud: reconcile the tools an agent declares against the
tools FirstOps actually governs, and expose the per-adapter capability matrix
so no surface claims more than it delivers.
"""

from __future__ import annotations

from collections.abc import Iterable

from firstops.tools import governed_tool_names

# Per-surface capability: what each integration can actually enforce.
# block = can stop the call; scrub = can rewrite args/prompt before it runs;
# audit = emits the action to the audit trail.
CAPABILITY_MATRIX: dict[str, dict[str, bool]] = {
    "base_decorator": {"block": True, "scrub": True, "audit": True},
    "claude": {"block": True, "scrub": True, "audit": True},
    "langgraph": {"block": True, "scrub": True, "audit": True},
    # OpenAI Agents tool guardrails are read-only: block + audit, but no
    # argument scrub (use the base decorator to scrub on OpenAI Agents).
    "openai_agents": {"block": True, "scrub": False, "audit": True},
    "llm_chain_link": {"block": True, "scrub": True, "audit": True},
    "mcp_proxy": {"block": True, "scrub": True, "audit": True},
}


def capability(surface: str) -> dict[str, bool]:
    """Return the capability dict for a surface, or {} if unknown."""
    return dict(CAPABILITY_MATRIX.get(surface, {}))


def ungoverned_tools(declared: Iterable[str]) -> list[str]:
    """Return declared tool names NOT governed by ``@firstops.tool`` (a gap).

    Honesty caveats (read before trusting the result):
    - This reconciles ONLY tools governed via the base decorator. A tool
      governed by a harness adapter (at the framework's execution boundary)
      will appear here as "ungoverned" even though it IS governed — adapter
      coverage is per-registration, not per-name.
    - The governed set is **process-wide**, not per-agent. With one agent per
      process (the common case) that equals per-agent; with multiple agents in
      one process it's the union, which can under-report a gap.

    Non-string entries are coerced to ``str`` so a malformed registry can't
    crash the check.
    """
    return sorted({str(d) for d in declared} - governed_tool_names())


def coverage_report(declared: Iterable[str]) -> dict[str, list[str]]:
    """Return a ``{governed, ungoverned}`` split of the declared tool names.

    Same caveats as :func:`ungoverned_tools` — ``ungoverned`` here means
    "not decorator-governed" and may include adapter-governed tools; the
    governed set is process-wide, not per-agent.
    """
    declared_set = {str(d) for d in declared}
    governed = governed_tool_names()
    return {
        "governed": sorted(declared_set & governed),
        "ungoverned": sorted(declared_set - governed),
    }
