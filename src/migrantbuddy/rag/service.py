"""Runs a single conversation turn through the LangGraph graph -- the
entry point api/routes.py calls. Replaces the old single-turn
RagService(query) -> answer: state (message history, running summary) now
persists server-side in the graph's checkpointer, keyed by thread_id,
instead of being recomputed from a single stateless query each time.
"""

from dataclasses import dataclass

from langchain_core.messages import HumanMessage

from migrantbuddy.observability import observe
from migrantbuddy.rag.graph import build_graph
from migrantbuddy.retrieval import Retriever


@dataclass
class ConversationResult:
    thread_id: str
    message: str
    answer: str
    sources: list[str]


class ConversationService:
    def __init__(self, retriever: Retriever):
        self.graph = build_graph(retriever)

    @observe()
    def answer(self, thread_id: str, message: str) -> ConversationResult:
        config = {"configurable": {"thread_id": thread_id}}
        result = self.graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)
        return ConversationResult(
            thread_id=thread_id,
            message=message,
            answer=result["messages"][-1].content,
            sources=result.get("sources", []),
        )
