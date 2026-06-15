"""Action-event model and wire (de)serialization for hook evaluation.

Mirrors the backend ``hookwire`` JSON contract exactly (see
``backend/shared/lib/hookwire/types.go``):

  request : event_type, agent, tool_name, tool_input(obj), tool_output(obj),
            session_id, cwd, channel, mcp{server,tool,url}, cli_version
  response: decision, reason, modified_payload(base64 bytes), policy_id

Principal and tenant are resolved server-side from the DPoP JWK thumbprint and
are never sent in the body.
"""

from __future__ import annotations

import base64
import binascii
from dataclasses import dataclass
from typing import Any

# --- channel constants (match enforcement/types.go Channel) ---
CHANNEL_MCP = "mcp"
CHANNEL_SYSTEM_TOOLS = "system_tools"
CHANNEL_LLM = "llm"

# --- event types (match enforcement EventType) ---
EVENT_PRE_TOOL_USE = "pre_tool_use"
EVENT_POST_TOOL_USE = "post_tool_use"

# --- decisions (match hookwire response decision) ---
DECISION_ALLOW = "allow"
DECISION_DENY = "deny"
DECISION_ASK = "ask"
DECISION_MODIFY = "modify"
_KNOWN_DECISIONS = frozenset(
    {DECISION_ALLOW, DECISION_DENY, DECISION_ASK, DECISION_MODIFY}
)

# Producer label for the `agent` field. The backend normalizer
# (FromHookRequest) maps this to audit Source `SourceSDK` ("sdk") per
# design-doc §7, so SDK actions are attributed correctly in audit. Keep this
# value in sync with the normalizer's `req.Agent == "firstops-sdk"` case.
SDK_AGENT_LABEL = "firstops-sdk"


@dataclass
class MCPInfo:
    """MCP-specific metadata; populated only when ``channel == "mcp"``."""

    server: str = ""
    tool: str = ""
    url: str = ""

    def to_wire(self) -> dict[str, str]:
        out: dict[str, str] = {}
        if self.server:
            out["server"] = self.server
        if self.tool:
            out["tool"] = self.tool
        if self.url:
            out["url"] = self.url
        return out


@dataclass
class ActionEvent:
    """A single governable action, serializable to the hookwire request shape."""

    event_type: str
    tool_name: str
    channel: str = ""
    tool_input: dict[str, Any] | None = None
    tool_output: dict[str, Any] | None = None
    agent: str = SDK_AGENT_LABEL
    session_id: str = ""
    cwd: str = ""
    mcp: MCPInfo | None = None
    cli_version: str = ""
    # The producer can apply a request-path modification before the action runs
    # (decorator rebind / Claude updatedInput / LangGraph arg rewrite / LLM body
    # rewrite). When True, sentinel ships `modify` for outbound scrub instead of
    # escalating to deny. Adapters that CAN'T mutate (OpenAI guardrails) leave
    # this False.
    producer_can_apply_modify: bool = False
    # Free-form producer metadata (e.g. {"harness": "langgraph"}). Merged into
    # the event metadata server-side and surfaced in audit.
    metadata: dict[str, str] | None = None

    def to_wire(self) -> dict[str, Any]:
        body: dict[str, Any] = {
            "event_type": self.event_type,
            "agent": self.agent,
            "tool_name": self.tool_name,
        }
        # tool_input/output are JSON objects (not stringified) so nested
        # structure is preserved end-to-end.
        if self.tool_input is not None:
            body["tool_input"] = self.tool_input
        if self.tool_output is not None:
            body["tool_output"] = self.tool_output
        if self.session_id:
            body["session_id"] = self.session_id
        if self.cwd:
            body["cwd"] = self.cwd
        if self.channel:
            body["channel"] = self.channel
        if self.mcp is not None:
            mcp_wire = self.mcp.to_wire()
            if mcp_wire:
                body["mcp"] = mcp_wire
        if self.cli_version:
            body["cli_version"] = self.cli_version
        if self.producer_can_apply_modify:
            body["producer_can_apply_modify"] = True
        if self.metadata:
            body["metadata"] = self.metadata
        return body


@dataclass
class Decision:
    """The enforcement verdict for an action."""

    action: str = DECISION_ALLOW
    reason: str = ""
    modified_payload: bytes | None = None
    policy_id: str = ""
    failed_open: bool = False  # True when we allowed due to an infra failure

    @property
    def blocked(self) -> bool:
        return self.action == DECISION_DENY

    @property
    def modified(self) -> bool:
        return self.action == DECISION_MODIFY and self.modified_payload is not None

    @classmethod
    def from_wire(cls, data: dict[str, Any]) -> Decision:
        action = data.get("decision") or DECISION_ALLOW
        reason = data.get("reason", "") or ""
        policy_id = data.get("policy_id", "") or ""

        # An unrecognized verdict must not silently behave like a clean allow:
        # fail open (never block on a verdict we can't act on) but set the
        # failed_open flag so it is visible downstream, not indistinguishable
        # from a real allow.
        if action not in _KNOWN_DECISIONS:
            return cls(
                action=DECISION_ALLOW,
                reason=f"unknown decision {action!r}: {reason}".rstrip(": "),
                policy_id=policy_id,
                failed_open=True,
            )

        raw = data.get("modified_payload")
        payload: bytes | None = None
        if raw:
            try:
                # Go marshals []byte as a base64 string. validate=True so a
                # corrupted payload raises instead of silently decoding to
                # wrong bytes that would then be applied to a live call.
                payload = (
                    base64.b64decode(raw, validate=True)
                    if isinstance(raw, str)
                    else bytes(raw)
                )
            except (binascii.Error, ValueError):
                return cls(
                    action=DECISION_ALLOW,
                    reason=f"invalid modified_payload: {reason}".rstrip(": "),
                    policy_id=policy_id,
                    failed_open=True,
                )

        # A modify verdict with no usable payload is malformed: fail open
        # visibly rather than report a "modify" the caller can't apply.
        if action == DECISION_MODIFY and payload is None:
            return cls(
                action=DECISION_ALLOW,
                reason=f"modify without payload: {reason}".rstrip(": "),
                policy_id=policy_id,
                failed_open=True,
            )

        return cls(
            action=action,
            reason=reason,
            modified_payload=payload,
            policy_id=policy_id,
        )

    @classmethod
    def fail_open(cls, reason: str) -> Decision:
        return cls(action=DECISION_ALLOW, reason=reason, failed_open=True)
