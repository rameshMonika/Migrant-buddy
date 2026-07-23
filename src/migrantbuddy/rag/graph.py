"""Builds the LangGraph state graph for a single conversation turn:
summarize (no-op until history grows past the threshold) -> rewrite_query
-> retrieve -> generate.

Stage 2: the checkpointer is now swappable via
migrantbuddy.config.CHECKPOINTER_BACKEND -- "memory" (default, LangGraph's
built-in in-memory saver, no external service, conversations lost on
restart) or "redis" (persists across restarts, thread_id-keyed, TTL so old
conversations expire). See CLAUDE.md's Conversational memory decision.

Caveat: the Redis branch is built against my best understanding of
langgraph-checkpoint-redis's API, not verified against a running instance
yet -- most likely spot for a real mismatch (exact TTL parameter shape
especially) is here, to be fixed against the actual error once Redis
(Memurai) is running locally.

Also wires up retrieval caching (migrantbuddy.config.RETRIEVAL_CACHE_ENABLED,
off by default) -- a separate toggle from CHECKPOINTER_BACKEND, since you may
want one without the other. See retrieval/cache.py.
"""

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph

from migrantbuddy.config import (
    CHECKPOINTER_BACKEND,
    CONVERSATION_TTL_SECONDS,
    REDIS_URL,
    RETRIEVAL_CACHE_ENABLED,
)
from migrantbuddy.rag.nodes import generate_node, make_retrieve_node, rewrite_query_node, summarize_node
from migrantbuddy.rag.state import ConversationState
from migrantbuddy.retrieval import Retriever
from migrantbuddy.retrieval.cache import RetrievalCache


def _build_checkpointer():
    if CHECKPOINTER_BACKEND == "memory":
        return MemorySaver()
    elif CHECKPOINTER_BACKEND == "redis":
        from langgraph.checkpoint.redis import RedisSaver

        checkpointer = RedisSaver.from_conn_string(
            REDIS_URL, ttl={"default_ttl": CONVERSATION_TTL_SECONDS // 60, "refresh_on_read": True}
        )
        checkpointer.setup()
        return checkpointer
    raise ValueError(f"Unknown checkpointer backend: {CHECKPOINTER_BACKEND!r} (expected 'memory' or 'redis')")


def build_graph(retriever: Retriever) -> CompiledStateGraph:
    graph = StateGraph(ConversationState)

    cache = RetrievalCache() if RETRIEVAL_CACHE_ENABLED else None

    graph.add_node("summarize", summarize_node)
    graph.add_node("rewrite_query", rewrite_query_node)
    graph.add_node("retrieve", make_retrieve_node(retriever, cache=cache))
    graph.add_node("generate", generate_node)

    graph.add_edge(START, "summarize")
    graph.add_edge("summarize", "rewrite_query")
    graph.add_edge("rewrite_query", "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile(checkpointer=_build_checkpointer())
