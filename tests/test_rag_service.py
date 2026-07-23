import pytest
from langchain_core.messages import AIMessage, HumanMessage

from migrantbuddy.rag.service import ConversationService


class FakeGraph:
    """Stands in for the compiled LangGraph -- graph behavior itself
    (summarization, query rewriting, retrieval, generation) is covered by
    test_rag_graph.py/test_rag_nodes.py. This file only tests what
    ConversationService itself is responsible for: wrapping the message,
    passing thread_id through config, and shaping the result.
    """

    def __init__(self, response_content: str = "an answer", sources: list[str] | None = None):
        self.response_content = response_content
        self.sources = sources if sources is not None else []
        self.last_inputs = None
        self.last_config = None

    def invoke(self, inputs, config):
        self.last_inputs = inputs
        self.last_config = config
        return {
            "messages": inputs["messages"] + [AIMessage(content=self.response_content)],
            "sources": self.sources,
        }


def test_answer_wraps_message_as_human_message(monkeypatch: pytest.MonkeyPatch):
    fake_graph = FakeGraph()
    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: fake_graph)

    service = ConversationService(retriever=object())
    service.answer("thread-1", "How much overtime pay?")

    assert len(fake_graph.last_inputs["messages"]) == 1
    assert isinstance(fake_graph.last_inputs["messages"][0], HumanMessage)
    assert fake_graph.last_inputs["messages"][0].content == "How much overtime pay?"


def test_answer_passes_thread_id_via_config(monkeypatch: pytest.MonkeyPatch):
    fake_graph = FakeGraph()
    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: fake_graph)

    service = ConversationService(retriever=object())
    service.answer("thread-abc", "query")

    assert fake_graph.last_config == {"configurable": {"thread_id": "thread-abc"}}


def test_answer_returns_conversation_result_with_expected_fields(monkeypatch: pytest.MonkeyPatch):
    fake_graph = FakeGraph(response_content="the final answer", sources=["https://example.com/a"])
    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: fake_graph)

    service = ConversationService(retriever=object())
    result = service.answer("thread-1", "query")

    assert result.thread_id == "thread-1"
    assert result.message == "query"
    assert result.answer == "the final answer"
    assert result.sources == ["https://example.com/a"]


def test_answer_defaults_sources_to_empty_list_when_missing(monkeypatch: pytest.MonkeyPatch):
    class GraphWithoutSources:
        def invoke(self, inputs, config):
            return {"messages": inputs["messages"] + [AIMessage(content="answer")]}

    monkeypatch.setattr("migrantbuddy.rag.service.build_graph", lambda retriever: GraphWithoutSources())

    service = ConversationService(retriever=object())
    result = service.answer("thread-1", "query")

    assert result.sources == []
