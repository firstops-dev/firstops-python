"""Shared boilerplate for the example agents: config + a governance tracer."""

import os


def load_config() -> dict:
    agent_id = os.environ["FO_AGENT_ID"].strip()
    key_path = os.environ["FO_PRIVATE_KEY_PATH"]
    with open(key_path) as f:
        key_pem = f.read()
    return {
        "agent_id": agent_id,
        "key_pem": key_pem,
        "gateway": os.environ.get("FO_GATEWAY", "https://api.firstops.dev"),
        "port": int(os.environ.get("FO_PORT", "9322")),
    }


def trace(fo) -> None:
    """Wrap the enforcement client so every governed action prints."""
    orig = fo.enforcement.evaluate

    def traced(event):
        d = orig(event)
        print(
            f"   [GOVERN] {event.channel:<13} {event.event_type:<14} "
            f"{event.tool_name:<32} -> {d.action}"
            + (f"  (FAILED_OPEN: {d.reason})" if d.failed_open else "")
        )
        return d

    fo.enforcement.evaluate = traced
