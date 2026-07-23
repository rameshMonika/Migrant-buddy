"""Langfuse tracing -- opt-in observability for the retrieval/generation
pipeline. Decorating a function with `observe()` (or `observe(as_type=
"generation")` for the actual LLM calls) makes it show up as a timed span
in the Langfuse UI, with its arguments/return value captured automatically,
nested under whichever decorated function is above it in the call stack.

Disabled by default (see migrantbuddy.config.LANGFUSE_ENABLED) -- when
disabled, `observe` is a true no-op: it returns the original function
completely unwrapped, not wrapped-but-inactive. That means every existing
test keeps working unchanged with tracing off (the default in dev/CI),
since decorating a function with this no-op has zero effect on its
identity or behavior.
"""

from typing import Callable, TypeVar

from migrantbuddy.config import LANGFUSE_ENABLED

F = TypeVar("F", bound=Callable)


def _noop_observe(func: F | None = None, **kwargs):
    def decorator(f: F) -> F:
        return f

    if func is not None:
        return func
    return decorator


if LANGFUSE_ENABLED:
    from langfuse import observe
else:
    observe = _noop_observe
