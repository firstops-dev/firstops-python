"""Convenience helpers for pointing LLM clients at the sidecar chain-link.

These are optional sugar over :func:`firstops.llm_base_url`. The SDK does not
depend on ``openai`` / ``anthropic`` — the client factories import them lazily
and raise a clear error if the package is absent.
"""

from __future__ import annotations

import os
from typing import Any

from firstops._runtime import llm_base_url


def openai_client(**kwargs: Any):
    """Return an ``openai.OpenAI`` client pointed at the sidecar.

    Pass your real ``api_key`` as usual (or set ``OPENAI_API_KEY``) — it is
    forwarded verbatim to the upstream; FirstOps never stores it.
    """
    try:
        import openai
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError("openai is not installed: pip install openai") from e
    kwargs.setdefault("base_url", llm_base_url("openai"))
    return openai.OpenAI(**kwargs)


def anthropic_client(**kwargs: Any):
    """Return an ``anthropic.Anthropic`` client pointed at the sidecar."""
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover - env-dependent
        raise RuntimeError("anthropic is not installed: pip install anthropic") from e
    kwargs.setdefault("base_url", llm_base_url("anthropic"))
    return anthropic.Anthropic(**kwargs)


def configure_llm_env() -> dict[str, str]:
    """Set ``OPENAI_BASE_URL`` / ``ANTHROPIC_BASE_URL`` to the sidecar.

    For frameworks that construct their own LLM client internally and only
    honor the env vars. Returns the variables it set.
    """
    env = {
        "OPENAI_BASE_URL": llm_base_url("openai"),
        "ANTHROPIC_BASE_URL": llm_base_url("anthropic"),
    }
    os.environ.update(env)
    return env
