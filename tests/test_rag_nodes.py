import pytest
from langchain_core.messages import AIMessage, HumanMessage, RemoveMessage

from migrantbuddy.indexing import Chunk
from migrantbuddy.rag.nodes import (
    generate_node,
    make_retrieve_node,
    rewrite_query_node,
    summarize_node,
)
from migrantbuddy.retrieval import RetrievalResult


def make_chunk(chunk_id: str, url: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=url,
        url=url,
        heading_path="Section",
        text=text,
        token_count=len(text) // 4,
    )


def make_messages(n: int) -> list:
    messages = []
    for i in range(n):
        cls = HumanMessage if i % 2 == 0 else AIMessage
        messages.append(cls(content=f"message {i}", id=str(i)))
    return messages


# --- summarize_node ---


def test_summarize_node_is_noop_below_threshold(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.rag.nodes.ollama_generate",
        lambda *a, **kw: pytest.fail("should not be called"),
    )
    state = {"messages": make_messages(12), "summary": ""}

    result = summarize_node(state)

    assert result == {}


def test_summarize_node_folds_older_messages_above_threshold(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_ollama_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        captured["user_prompt"] = user_prompt
        return "condensed summary"

    monkeypatch.setattr("migrantbuddy.rag.nodes.ollama_generate", fake_ollama_generate)
    messages = make_messages(13)
    state = {"messages": messages, "summary": ""}

    result = summarize_node(state)

    assert result["summary"] == "condensed summary"
    # MESSAGES_KEPT_VERBATIM=6 -- the first 13-6=7 messages get removed.
    assert len(result["messages"]) == 7
    assert all(isinstance(m, RemoveMessage) for m in result["messages"])
    assert [m.id for m in result["messages"]] == [str(i) for i in range(7)]
    assert "message 0" in captured["user_prompt"]


def test_summarize_node_uses_the_tighter_summary_token_cap(monkeypatch: pytest.MonkeyPatch):
    # Regression check: summarize_node used to fall through to
    # GENERATION_MAX_TOKENS (512, sized for a full answer) with no cap of
    # its own, producing summaries far longer than "a few sentences."
    from migrantbuddy.config import SUMMARY_MAX_TOKENS

    captured = {}

    def fake_ollama_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        captured["max_tokens"] = max_tokens
        return "condensed summary"

    monkeypatch.setattr("migrantbuddy.rag.nodes.ollama_generate", fake_ollama_generate)
    state = {"messages": make_messages(13), "summary": ""}

    summarize_node(state)

    assert captured["max_tokens"] == SUMMARY_MAX_TOKENS


def test_rewrite_query_node_uses_the_tighter_rewrite_token_cap(monkeypatch: pytest.MonkeyPatch):
    from migrantbuddy.config import QUERY_REWRITE_MAX_TOKENS

    captured = {}

    def fake_ollama_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        captured["max_tokens"] = max_tokens
        return "standalone rewritten query"

    monkeypatch.setattr("migrantbuddy.rag.nodes.ollama_generate", fake_ollama_generate)
    state = {
        "messages": [
            HumanMessage(content="How much overtime pay if my salary is $3000?", id="1"),
            AIMessage(content="You'd get 1.5x your hourly rate.", id="2"),
            HumanMessage(content="What about daily-rated workers?", id="3"),
        ],
        "summary": "",
    }

    rewrite_query_node(state)

    assert captured["max_tokens"] == QUERY_REWRITE_MAX_TOKENS


def test_summarize_node_folds_existing_summary_into_prompt(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_ollama_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        captured["user_prompt"] = user_prompt
        return "new summary"

    monkeypatch.setattr("migrantbuddy.rag.nodes.ollama_generate", fake_ollama_generate)
    state = {"messages": make_messages(13), "summary": "prior summary text"}

    summarize_node(state)

    assert "prior summary text" in captured["user_prompt"]


# --- rewrite_query_node ---


def test_rewrite_query_node_skips_rewrite_on_first_turn(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.rag.nodes.ollama_generate",
        lambda *a, **kw: pytest.fail("should not be called"),
    )
    state = {"messages": [HumanMessage(content="How much overtime pay?", id="1")], "summary": ""}

    result = rewrite_query_node(state)

    assert result == {"standalone_query": "How much overtime pay?"}


def test_rewrite_query_node_calls_backend_when_history_exists(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.rag.nodes.ollama_generate", lambda *a, **kw: "standalone rewritten query"
    )
    state = {
        "messages": [
            HumanMessage(content="How much overtime pay if my salary is $3000?", id="1"),
            AIMessage(content="You'd get 1.5x your hourly rate.", id="2"),
            HumanMessage(content="What about daily-rated workers?", id="3"),
        ],
        "summary": "",
    }

    result = rewrite_query_node(state)

    assert result == {"standalone_query": "standalone rewritten query"}


def test_rewrite_query_node_calls_backend_when_only_summary_exists(monkeypatch: pytest.MonkeyPatch):
    # Edge case: everything except the latest message has already been
    # folded into a summary -- history is empty but summary isn't, so
    # rewriting should still happen (not treated as a first turn).
    monkeypatch.setattr(
        "migrantbuddy.rag.nodes.ollama_generate", lambda *a, **kw: "rewritten from summary"
    )
    state = {
        "messages": [HumanMessage(content="What about daily-rated workers?", id="1")],
        "summary": "Earlier the user asked about overtime pay for a $3000 salary.",
    }

    result = rewrite_query_node(state)

    assert result == {"standalone_query": "rewritten from summary"}


# --- retrieve_node (via make_retrieve_node) ---


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}
        self.last_query = None
        self.last_top_k = None

    def hybrid_rerank(self, query: str, top_k: int = 5):
        self.last_query = query
        self.last_top_k = top_k
        return [
            RetrievalResult(chunk_id=chunk_id, score=1.0)
            for chunk_id in list(self.chunks_by_id)[:top_k]
        ]


def test_retrieve_node_uses_standalone_query_when_present():
    chunks = [make_chunk("c1", "https://example.com/a", "text a")]
    retriever = FakeRetriever(chunks)
    node = make_retrieve_node(retriever, top_k=1)
    state = {
        "messages": [HumanMessage(content="raw message", id="1")],
        "standalone_query": "rewritten query",
    }

    result = node(state)

    assert retriever.last_query == "rewritten query"
    assert [c.chunk_id for c in result["context_chunks"]] == ["c1"]


def test_retrieve_node_falls_back_to_latest_message_without_standalone_query():
    chunks = [make_chunk("c1", "https://example.com/a", "text a")]
    retriever = FakeRetriever(chunks)
    node = make_retrieve_node(retriever, top_k=1)
    state = {"messages": [HumanMessage(content="raw message", id="1")], "standalone_query": ""}

    node(state)

    assert retriever.last_query == "raw message"


# --- retrieve_node caching ---


class FakeCache:
    def __init__(self, hit: list[RetrievalResult] | None = None):
        self.hit = hit
        self.set_calls: list[tuple[str, int, list[RetrievalResult]]] = []

    def get(self, query: str, top_k: int):
        return self.hit

    def set(self, query: str, top_k: int, results: list[RetrievalResult]):
        self.set_calls.append((query, top_k, results))


def test_retrieve_node_skips_retriever_on_cache_hit():
    chunks = [make_chunk("c1", "https://example.com/a", "text a")]
    retriever = FakeRetriever(chunks)
    cache = FakeCache(hit=[RetrievalResult(chunk_id="c1", score=0.9)])
    node = make_retrieve_node(retriever, top_k=1, cache=cache)
    state = {"messages": [HumanMessage(content="query", id="1")], "standalone_query": ""}

    result = node(state)

    assert retriever.last_query is None  # never called
    assert [c.chunk_id for c in result["context_chunks"]] == ["c1"]


def test_retrieve_node_populates_cache_on_miss():
    chunks = [make_chunk("c1", "https://example.com/a", "text a")]
    retriever = FakeRetriever(chunks)
    cache = FakeCache(hit=None)
    node = make_retrieve_node(retriever, top_k=1, cache=cache)
    state = {
        "messages": [HumanMessage(content="query", id="1")],
        "standalone_query": "rewritten query",
    }

    node(state)

    assert retriever.last_query == "rewritten query"  # cache miss -- retriever was called
    assert len(cache.set_calls) == 1
    assert cache.set_calls[0][0] == "rewritten query"
    assert cache.set_calls[0][1] == 1


def test_retrieve_node_falls_through_on_stale_cache_entry():
    # Cache hit references a chunk_id that no longer exists in the current
    # corpus (e.g. after a re-ingestion) -- must not crash, must fall back
    # to a fresh retrieval instead.
    chunks = [make_chunk("c1", "https://example.com/a", "text a")]
    retriever = FakeRetriever(chunks)
    cache = FakeCache(hit=[RetrievalResult(chunk_id="stale-chunk-id", score=0.9)])
    node = make_retrieve_node(retriever, top_k=1, cache=cache)
    state = {"messages": [HumanMessage(content="query", id="1")], "standalone_query": "query"}

    result = node(state)

    assert retriever.last_query == "query"  # fell through to a real retrieval
    assert [c.chunk_id for c in result["context_chunks"]] == ["c1"]


# --- generate_node ---


def test_generate_node_returns_ai_message_and_sources(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.rag.nodes.ollama_generate", lambda *a, **kw: "the generated answer"
    )
    chunks = [make_chunk("c1", "https://example.com/a", "context text")]
    state = {"messages": [HumanMessage(content="query", id="1")], "context_chunks": chunks}

    result = generate_node(state)

    assert len(result["messages"]) == 1
    assert isinstance(result["messages"][0], AIMessage)
    assert result["messages"][0].content == "the generated answer"
    assert result["sources"] == ["https://example.com/a"]
