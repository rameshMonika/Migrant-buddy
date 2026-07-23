import pytest
from langchain_core.messages import HumanMessage
from langgraph.checkpoint.memory import MemorySaver

from migrantbuddy.indexing import Chunk
from migrantbuddy.rag.graph import _build_checkpointer, build_graph
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


class FakeRetriever:
    def __init__(self, chunks: list[Chunk]):
        self.chunks_by_id = {chunk.chunk_id: chunk for chunk in chunks}

    def hybrid_rerank(self, query: str, top_k: int = 5):
        return [RetrievalResult(chunk_id=chunk_id, score=1.0) for chunk_id in list(self.chunks_by_id)[:top_k]]


@pytest.fixture
def graph(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("migrantbuddy.rag.nodes.ollama_generate", lambda *a, **kw: "a generated answer")
    chunks = [make_chunk("c1", "https://example.com/a", "context text about salary")]
    return build_graph(FakeRetriever(chunks))


def invoke(graph, thread_id: str, message: str):
    config = {"configurable": {"thread_id": thread_id}}
    return graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)


def test_single_turn_produces_answer_and_sources(graph):
    result = invoke(graph, "thread-1", "How much overtime pay am I entitled to?")

    assert result["messages"][-1].content == "a generated answer"
    assert result["sources"] == ["https://example.com/a"]


def test_same_thread_id_accumulates_message_history(graph):
    invoke(graph, "thread-1", "How much overtime pay am I entitled to?")
    result = invoke(graph, "thread-1", "What about daily-rated workers?")

    # 2 human + 2 AI messages from the two turns.
    assert len(result["messages"]) == 4
    assert result["messages"][0].content == "How much overtime pay am I entitled to?"
    assert result["messages"][2].content == "What about daily-rated workers?"


def test_different_thread_ids_have_isolated_history(graph):
    invoke(graph, "thread-1", "How much overtime pay am I entitled to?")
    result = invoke(graph, "thread-2", "What about daily-rated workers?")

    # thread-2 has never been invoked before -- only this turn's messages.
    assert len(result["messages"]) == 2
    assert result["messages"][0].content == "What about daily-rated workers?"


def test_conversation_past_threshold_gets_summarized(monkeypatch: pytest.MonkeyPatch):
    # Every turn after the first calls ollama_generate twice (rewrite +
    # generate), plus a third time on whichever turn triggers
    # summarization -- so this can't be a short, fixed-length response
    # list. Distinguish by system prompt instead, since that's the one
    # call whose *content* this test actually needs to check.
    from migrantbuddy.generation.prompts import SUMMARY_SYSTEM_PROMPT

    def fake_ollama_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        if system_prompt == SUMMARY_SYSTEM_PROMPT:
            return "condensed summary"
        return "a response"

    monkeypatch.setattr("migrantbuddy.rag.nodes.ollama_generate", fake_ollama_generate)
    chunks = [make_chunk("c1", "https://example.com/a", "context text")]
    graph = build_graph(FakeRetriever(chunks))

    result = None
    for i in range(7):
        result = invoke(graph, "thread-1", f"question {i}")

    # SUMMARY_TRIGGER_MESSAGE_COUNT=12, MESSAGES_KEPT_VERBATIM=6 -- by the 7th
    # turn (13 messages: 7 human + 6 AI before this turn's own additions
    # land) summarization should have kicked in at least once, leaving a
    # non-empty summary and fewer than the full unbounded message count.
    assert result["summary"] != ""
    assert len(result["messages"]) < 14


# --- checkpointer selection ---


def test_build_checkpointer_returns_memory_saver_by_default(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("migrantbuddy.rag.graph.CHECKPOINTER_BACKEND", "memory")

    checkpointer = _build_checkpointer()

    assert isinstance(checkpointer, MemorySaver)


def test_build_checkpointer_rejects_unknown_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("migrantbuddy.rag.graph.CHECKPOINTER_BACKEND", "something-else")

    with pytest.raises(ValueError, match="Unknown checkpointer backend"):
        _build_checkpointer()


def test_build_checkpointer_configures_redis_when_selected(monkeypatch: pytest.MonkeyPatch):
    # Doesn't need a real Redis running -- fakes langgraph-checkpoint-redis's
    # RedisSaver so this only checks that _build_checkpointer wires the
    # right URL/TTL through and calls .setup(), not real Redis connectivity.
    from migrantbuddy.config import CONVERSATION_TTL_SECONDS, REDIS_URL

    monkeypatch.setattr("migrantbuddy.rag.graph.CHECKPOINTER_BACKEND", "redis")
    created = {}

    class FakeRedisSaver:
        def __init__(self):
            self.setup_called = False

        @classmethod
        def from_conn_string(cls, url, ttl):
            created["url"] = url
            created["ttl"] = ttl
            return cls()

        def setup(self):
            self.setup_called = True

    monkeypatch.setattr("langgraph.checkpoint.redis.RedisSaver", FakeRedisSaver)

    checkpointer = _build_checkpointer()

    assert isinstance(checkpointer, FakeRedisSaver)
    assert checkpointer.setup_called
    assert created["url"] == REDIS_URL
    assert created["ttl"] == {"default_ttl": CONVERSATION_TTL_SECONDS // 60, "refresh_on_read": True}
