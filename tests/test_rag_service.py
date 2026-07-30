import pytest
from langchain_core.messages import HumanMessage

from migrantbuddy.indexing import Chunk
from migrantbuddy.rag.service import ConversationService


def make_chunk(chunk_id: str, url: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        document_id=url,
        url=url,
        heading_path="Section",
        text=text,
        token_count=len(text) // 4,
    )


class FakeGraphState:
    def __init__(self, values: dict):
        self.values = values


class FakeGraph:
    """Stands in for the compiled LangGraph -- graph behavior itself
    (summarization, query rewriting, retrieval, generation) is covered by
    test_rag_graph.py/test_rag_nodes.py. This file only tests what
    ConversationService is responsible for: wrapping the message, passing
    thread_id through config, building the prompt from the paused state
    (prepare_turn), and writing the finished answer back + resuming
    (finalize_turn) -- matching the interrupt_before=["generate"] contract
    rag/graph.py compiles with.
    """

    def __init__(self, context_chunks: list[Chunk] | None = None):
        self.context_chunks = context_chunks if context_chunks is not None else []
        self.last_invoke_inputs = None
        self.last_invoke_config = None
        self.last_update_state_call = None
        self.resumed = False
        self._messages: list = []

    def invoke(self, inputs, config):
        self.last_invoke_config = config
        if inputs is None:
            self.resumed = True
            return {}
        self.last_invoke_inputs = inputs
        self._messages = list(inputs["messages"])
        return {}

    def get_state(self, config):
        return FakeGraphState({"messages": self._messages, "context_chunks": self.context_chunks})

    def update_state(self, config, values, as_node):
        self.last_update_state_call = {"config": config, "values": values, "as_node": as_node}


def test_prepare_turn_wraps_message_as_human_message(monkeypatch: pytest.MonkeyPatch):
    fake_graph = FakeGraph()
    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: fake_graph)

    service = ConversationService(retriever=object())
    service.prepare_turn("thread-1", "How much overtime pay?")

    assert len(fake_graph.last_invoke_inputs["messages"]) == 1
    assert isinstance(fake_graph.last_invoke_inputs["messages"][0], HumanMessage)
    assert fake_graph.last_invoke_inputs["messages"][0].content == "How much overtime pay?"


def test_prepare_turn_passes_thread_id_via_config(monkeypatch: pytest.MonkeyPatch):
    fake_graph = FakeGraph()
    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: fake_graph)

    service = ConversationService(retriever=object())
    prepared = service.prepare_turn("thread-abc", "query")

    assert prepared.config == {"configurable": {"thread_id": "thread-abc"}}


def test_prepare_turn_builds_prompt_from_paused_state_and_derives_sources(
    monkeypatch: pytest.MonkeyPatch,
):
    chunk = make_chunk("c1", "https://example.com/a", "context text about salary")
    fake_graph = FakeGraph(context_chunks=[chunk])
    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: fake_graph)

    service = ConversationService(retriever=object())
    prepared = service.prepare_turn("thread-1", "How much overtime pay?")

    assert "How much overtime pay?" in prepared.user_prompt
    assert "context text about salary" in prepared.user_prompt
    assert prepared.sources == ["https://example.com/a"]


def test_finalize_turn_writes_answer_and_sources_as_the_generate_node(
    monkeypatch: pytest.MonkeyPatch,
):
    fake_graph = FakeGraph()
    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: fake_graph)

    service = ConversationService(retriever=object())
    prepared = service.prepare_turn("thread-1", "query")
    service.finalize_turn(prepared, "the final answer")

    call = fake_graph.last_update_state_call
    assert call["config"] == prepared.config
    assert call["as_node"] == "generate"
    assert call["values"]["messages"][0].content == "the final answer"
    assert call["values"]["sources"] == prepared.sources


def test_finalize_turn_resumes_the_graph_past_the_interrupt(monkeypatch: pytest.MonkeyPatch):
    fake_graph = FakeGraph()
    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: fake_graph)

    service = ConversationService(retriever=object())
    prepared = service.prepare_turn("thread-1", "query")
    service.finalize_turn(prepared, "the final answer")

    assert fake_graph.resumed
