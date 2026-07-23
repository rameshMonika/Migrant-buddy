from migrantbuddy.config import LANGFUSE_ENABLED
from migrantbuddy.observability import _noop_observe, observe


def test_noop_observe_bare_returns_the_same_function_object():
    def my_function():
        return "result"

    decorated = _noop_observe(my_function)

    assert decorated is my_function


def test_noop_observe_with_kwargs_returns_a_decorator_that_is_transparent():
    def my_function(x, y):
        return x + y

    decorator = _noop_observe(as_type="generation")
    decorated = decorator(my_function)

    assert decorated is my_function
    assert decorated(2, 3) == 5


def test_observe_matches_langfuse_enabled_config():
    # Whether `observe` is the no-op or the real Langfuse decorator depends
    # on whether LANGFUSE_* credentials are configured (via .env or real
    # env vars) wherever this test happens to run -- assert the wiring is
    # consistent with that, rather than hardcoding one or the other.
    if LANGFUSE_ENABLED:
        assert observe is not _noop_observe
    else:
        assert observe is _noop_observe


def test_noop_observe_preserves_function_behavior_through_class_methods():
    class Example:
        @_noop_observe()
        def method(self, value):
            return value * 2

    assert Example().method(21) == 42
