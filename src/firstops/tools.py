"""The base-API tool decorator — govern any callable.

``@firstops.tool`` wraps a function so each call is forwarded to sentinel as a
``pre_tool_use`` event (block / modify args) and a ``post_tool_use`` event
(audit). It is the harness-agnostic floor: it audits everywhere and blocks
where the surrounding harness propagates a raised exception (per the design's
Discovery A, raising is the block mechanism for the base layer).

Coverage is honest: a decorated tool is governed; an un-decorated one is not.
For framework built-ins the developer never authored, use the harness adapter
instead (it hooks the execution boundary).
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
from typing import Any, Callable

from firstops import _runtime
from firstops.channels import classify, mcp_info
from firstops.events import (
    CHANNEL_MCP,
    EVENT_POST_TOOL_USE,
    EVENT_PRE_TOOL_USE,
    ActionEvent,
    Decision,
)

logger = logging.getLogger("firstops")

# Attributes that mark an already-built framework tool object (LangChain
# StructuredTool, LlamaIndex FunctionTool, etc.). We refuse to wrap these —
# the user must decorate the underlying function before the framework wraps it.
_TOOL_OBJECT_MARKERS = ("invoke", "ainvoke", "args_schema", "_run")


# Names of tools governed by @firstops.tool this process — used by
# firstops.coverage to reconcile declared-vs-governed and surface gaps.
_GOVERNED_TOOLS: set[str] = set()


def governed_tool_names() -> set[str]:
    """Return the set of tool names governed by @firstops.tool."""
    return set(_GOVERNED_TOOLS)


class FirstOpsPolicyError(Exception):
    """Raised when sentinel denies a governed tool call."""

    def __init__(self, tool_name: str, reason: str, policy_id: str = ""):
        self.tool_name = tool_name
        self.reason = reason
        self.policy_id = policy_id
        super().__init__(f"FirstOps blocked tool {tool_name!r}: {reason}")


def tool(fn: Callable | None = None, *, name: str | None = None):
    """Decorator that governs a tool function. Usable as ``@tool`` or ``@tool(name=...)``."""

    def decorator(func: Callable) -> Callable:
        _check_wrappable(func)
        tool_name = name or getattr(func, "__name__", "tool")
        # Register BOTH the governance name and the function's __name__ so a
        # coverage check that enumerates tools by either key sees it governed.
        _GOVERNED_TOOLS.add(tool_name)
        fn_name = getattr(func, "__name__", None)
        if fn_name:
            _GOVERNED_TOOLS.add(fn_name)

        # Order matters: async-gen and generator are NOT coroutine functions,
        # so they must be detected first or they'd fall into the sync wrapper
        # and return an un-iterated (async)generator with the body unexecuted.
        if inspect.isasyncgenfunction(func):
            # Govern EAGERLY at call time (the wrapper is a plain function that
            # returns the async generator), so a DENY blocks before any
            # iteration rather than on the first __anext__.
            @functools.wraps(func)
            def agwrapper(*args: Any, **kwargs: Any):
                args, kwargs = _govern(func, tool_name, args, kwargs)
                return _agen_iterate(func, tool_name, args, kwargs)

            return agwrapper

        if inspect.isgeneratorfunction(func):

            @functools.wraps(func)
            def gwrapper(*args: Any, **kwargs: Any):
                args, kwargs = _govern(func, tool_name, args, kwargs)
                return _gen_iterate(func, tool_name, args, kwargs)

            return gwrapper

        if inspect.iscoroutinefunction(func):

            @functools.wraps(func)
            async def awrapper(*args: Any, **kwargs: Any) -> Any:
                args, kwargs = _govern(func, tool_name, args, kwargs)
                result = None
                error: Exception | None = None
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    error = e
                    raise
                finally:
                    _audit_post(tool_name, result, error)

            return awrapper

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            args, kwargs = _govern(func, tool_name, args, kwargs)
            result = None
            error: Exception | None = None
            try:
                result = func(*args, **kwargs)
                return result
            except Exception as e:
                error = e
                raise
            finally:
                _audit_post(tool_name, result, error)

        return wrapper

    if fn is not None:
        return decorator(fn)
    return decorator


# Sentinel marker for "the result is a stream we didn't materialize".
_STREAM_SENTINEL = object()


def _gen_iterate(func, tool_name, args, kwargs):
    """Drive a generator tool, auditing once after it's exhausted/aborted."""
    error: Exception | None = None
    try:
        yield from func(*args, **kwargs)
    except Exception as e:
        error = e
        raise
    finally:
        _audit_post(tool_name, _STREAM_SENTINEL, error)


async def _agen_iterate(func, tool_name, args, kwargs):
    """Async analogue of :func:`_gen_iterate`."""
    error: Exception | None = None
    try:
        async for item in func(*args, **kwargs):
            yield item
    except Exception as e:
        error = e
        raise
    finally:
        _audit_post(tool_name, _STREAM_SENTINEL, error)


def _check_wrappable(func: Callable) -> None:
    if (
        inspect.isfunction(func)
        or inspect.ismethod(func)
        or inspect.iscoroutinefunction(func)
        or inspect.isgeneratorfunction(func)
        or inspect.isasyncgenfunction(func)
    ):
        return
    if callable(func) and any(hasattr(func, m) for m in _TOOL_OBJECT_MARKERS):
        raise TypeError(
            f"@firstops.tool expects a plain function, not a built framework tool "
            f"object ({type(func).__name__}). Decorate the underlying function "
            f"before the framework wraps it (put @firstops.tool innermost)."
        )
    if not callable(func):
        raise TypeError(f"@firstops.tool expects a callable, got {type(func).__name__}")
    # Other callables (functools.partial, lambdas) are allowed.


def _govern(
    func: Callable, tool_name: str, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    """Run the pre_tool_use evaluation and apply the decision to the call args."""
    tool_input = _bind_inputs(func, args, kwargs)
    decision = _evaluate_pre(tool_name, tool_input)
    return _apply_pre(func, decision, tool_name, args, kwargs)


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except (TypeError, ValueError):
        return repr(value)


def _bind_inputs(func: Callable, args: tuple, kwargs: dict) -> dict[str, Any]:
    """Map a call's args/kwargs to a JSON-able {param: value} dict."""
    try:
        bound = inspect.signature(func).bind_partial(*args, **kwargs)
        bound.apply_defaults()
        return {k: _jsonable(v) for k, v in bound.arguments.items()}
    except (TypeError, ValueError):
        out: dict[str, Any] = {f"arg{i}": _jsonable(a) for i, a in enumerate(args)}
        out.update({k: _jsonable(v) for k, v in kwargs.items()})
        return out


def _evaluate_pre(tool_name: str, tool_input: dict[str, Any]) -> Decision | None:
    rt = _runtime.runtime()
    if rt is None:
        return None  # not initialized — no governance, run normally
    channel = classify(tool_name)
    event = ActionEvent(
        event_type=EVENT_PRE_TOOL_USE,
        tool_name=tool_name,
        channel=channel,
        tool_input=tool_input,
        mcp=mcp_info(tool_name) if channel == CHANNEL_MCP else None,
        # The decorator rebinds args from a modify payload, so request-path
        # scrub is applicable — let sentinel ship modify, not escalate to deny.
        producer_can_apply_modify=True,
    )
    return rt.enforcement.evaluate(event)


def _apply_pre(
    func: Callable, decision: Decision | None, tool_name: str, args: tuple, kwargs: dict
) -> tuple[tuple, dict]:
    if decision is None:
        return args, kwargs
    if decision.blocked:
        raise FirstOpsPolicyError(tool_name, decision.reason, decision.policy_id)
    if decision.modified and decision.modified_payload:
        rebound = _rebind(func, decision.modified_payload, args, kwargs)
        if rebound is not None:
            return rebound
        # We signalled producer_can_apply_modify, so sentinel shipped a scrub
        # instead of a deny. If it doesn't fit the signature, fail CLOSED —
        # running the tool with unscrubbed args is worse than blocking.
        logger.warning(
            "firstops: could not apply modify to %s; blocking (fail closed)", tool_name
        )
        raise FirstOpsPolicyError(
            tool_name,
            "policy required a modification this tool could not apply",
            decision.policy_id,
        )
    return args, kwargs


def _rebind(
    func: Callable, payload: bytes, args: tuple, kwargs: dict
) -> tuple[tuple, dict] | None:
    """Overlay sentinel's scrubbed inputs onto the call, respecting param kinds.

    Returns (args, kwargs) honoring positional-only / *args / **kwargs, or None
    if the payload can't be applied (caller then falls open to original args).
    """
    try:
        new_input = json.loads(payload)
    except (ValueError, TypeError):
        return None
    if not isinstance(new_input, dict):
        return None
    try:
        sig = inspect.signature(func)
        bound = sig.bind_partial(*args, **kwargs)
        bound.apply_defaults()
        merged = dict(bound.arguments)
        merged.update(new_input)

        out_args: list[Any] = []
        out_kwargs: dict[str, Any] = {}
        consumed = set()
        for pname, param in sig.parameters.items():
            if param.kind == inspect.Parameter.VAR_POSITIONAL:
                out_args.extend(merged.get(pname, ()) or ())
            elif param.kind == inspect.Parameter.VAR_KEYWORD:
                out_kwargs.update(merged.get(pname, {}) or {})
            elif pname in merged:
                if param.kind == inspect.Parameter.POSITIONAL_ONLY:
                    out_args.append(merged[pname])
                else:
                    out_kwargs[pname] = merged[pname]
            consumed.add(pname)
        # Validate the reconstruction actually binds before returning it.
        sig.bind(*out_args, **out_kwargs)
        return tuple(out_args), out_kwargs
    except (TypeError, ValueError):
        return None


def _audit_post(tool_name: str, result: Any, error: Exception | None = None) -> None:
    """Emit a post_tool_use audit event. Best-effort: never affects the call."""
    rt = _runtime.runtime()
    if rt is None:
        return
    try:
        if error is not None:
            output: dict[str, Any] = {"error": repr(error)}
        elif result is _STREAM_SENTINEL:
            output = {"result": "<stream>"}
        else:
            output = {"result": _jsonable(result)}
        event = ActionEvent(
            event_type=EVENT_POST_TOOL_USE,
            tool_name=tool_name,
            channel=classify(tool_name),
            tool_output=output,
        )
        rt.enforcement.evaluate(event)
    except Exception as e:  # pragma: no cover - defensive
        logger.debug("firstops post-eval failed for %s: %s", tool_name, e)
