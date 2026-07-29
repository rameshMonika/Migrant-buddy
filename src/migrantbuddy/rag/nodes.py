"""Graph node functions for a single conversation turn. Each node reads
what it needs from ConversationState and returns a partial state update --
LangGraph merges the returned dict into state (`messages` appended via
MessagesState's built-in reducer, other fields overwritten).

Thin by design: no prompt text lives here (see generation/prompts.py), no
retrieval logic lives here (see retrieval/service.py) -- these functions
only adapt existing modules to the graph's state-in/state-out node shape.

retrieve_node needs a Retriever instance (loaded models/index, expensive to
construct) that must be built once and reused -- not something a node
function can own itself, since LangGraph calls nodes as plain functions.
make_retrieve_node() is a small factory that closes over an
already-constructed Retriever, matching how generate_node/rewrite_query_node
close over nothing (they call the module-level generation functions, which
already default their own config).
"""

from typing import Sequence

from langchain_core.messages import AIMessage, BaseMessage, RemoveMessage

from migrantbuddy.config import (
    GENERATION_BACKEND,
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL_NAME,
    MESSAGES_KEPT_VERBATIM,
    QUERY_REWRITE_MAX_TOKENS,
    SUMMARY_MAX_TOKENS,
    SUMMARY_TRIGGER_MESSAGE_COUNT,
)
from migrantbuddy.generation.ollama_client import generate as ollama_generate
from migrantbuddy.generation.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    build_prompt,
    build_query_rewrite_prompt,
    build_summary_prompt,
)
from migrantbuddy.generation.vllm_client import generate as vllm_generate
from migrantbuddy.observability import observe
from migrantbuddy.rag.state import ConversationState
from migrantbuddy.retrieval import Retriever
from migrantbuddy.retrieval.cache import RetrievalCache


def _call_backend(
    system_prompt: str,
    user_prompt: str,
    *,
    backend: str = GENERATION_BACKEND,
    model_name: str = GENERATION_MODEL_NAME,
    max_tokens: int = GENERATION_MAX_TOKENS,
) -> str:
    if backend == "ollama":
        return ollama_generate(
            system_prompt, user_prompt, model_name=model_name, max_tokens=max_tokens
        )
    elif backend == "vllm":
        return vllm_generate(
            system_prompt, user_prompt, model_name=model_name, max_tokens=max_tokens
        )
    raise ValueError(f"Unknown generation backend: {backend!r} (expected 'ollama' or 'vllm')")


def _messages_to_turns(messages: Sequence[BaseMessage]) -> list[dict]:
    return [
        {"role": "assistant" if isinstance(m, AIMessage) else "user", "content": m.content}
        for m in messages
    ]


@observe()
def summarize_node(state: ConversationState) -> dict:
    messages = state["messages"]
    if len(messages) <= SUMMARY_TRIGGER_MESSAGE_COUNT:
        return {}

    to_fold_in = messages[:-MESSAGES_KEPT_VERBATIM]
    existing_summary = state.get("summary", "")
    prompt = build_summary_prompt(existing_summary, _messages_to_turns(to_fold_in))
    summary = _call_backend(SUMMARY_SYSTEM_PROMPT, prompt, max_tokens=SUMMARY_MAX_TOKENS)

    # RemoveMessage trims the folded-in messages from state (LangGraph's
    # message reducer honors these specially) so the prompt stops growing --
    # `summary` carries their content forward instead of dropping it.
    removed = [RemoveMessage(id=m.id) for m in to_fold_in]
    return {"summary": summary, "messages": removed}


@observe()
def rewrite_query_node(state: ConversationState) -> dict:
    messages = state["messages"]
    latest = messages[-1].content
    history = _messages_to_turns(messages[:-1])
    summary = state.get("summary", "")

    if not history and not summary:
        # First turn -- nothing to rewrite against yet.
        return {"standalone_query": latest}

    prompt = build_query_rewrite_prompt(summary, history, latest)
    standalone_query = _call_backend(
        QUERY_REWRITE_SYSTEM_PROMPT, prompt, max_tokens=QUERY_REWRITE_MAX_TOKENS
    )
    return {"standalone_query": standalone_query}


def make_retrieve_node(
    retriever: Retriever, *, top_k: int = 5, cache: RetrievalCache | None = None
):
    @observe()
    def retrieve_node(state: ConversationState) -> dict:
        query = state.get("standalone_query") or state["messages"][-1].content

        if cache is not None:
            cached_results = cache.get(query, top_k)
            if cached_results is not None:
                try:
                    context_chunks = [retriever.chunks_by_id[r.chunk_id] for r in cached_results]
                    return {"context_chunks": context_chunks}
                except KeyError:
                    # Stale entry from before a re-ingestion -- the cached
                    # chunk_id no longer exists in the current corpus. Fall
                    # through to a fresh lookup rather than crashing; the
                    # cache.set() below will overwrite it with a current one.
                    pass

        results = retriever.hybrid_rerank(query, top_k=top_k)
        if cache is not None:
            cache.set(query, top_k, results)

        context_chunks = [retriever.chunks_by_id[result.chunk_id] for result in results]
        return {"context_chunks": context_chunks}

    return retrieve_node


@observe()
def generate_node(state: ConversationState) -> dict:
    query = state["messages"][-1].content
    context_chunks = state["context_chunks"]
    prompt = build_prompt(query, context_chunks)
    answer = _call_backend(SYSTEM_PROMPT, prompt)
    return {
        "messages": [AIMessage(content=answer)],
        "sources": [chunk.url for chunk in context_chunks],
    }
