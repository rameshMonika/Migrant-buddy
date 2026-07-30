"""Runs a single conversation turn through the LangGraph graph -- the
entry point api/routes.py calls. State (message history, running summary)
persists server-side in the graph's checkpointer, keyed by thread_id,
instead of being recomputed from a single stateless query each time.

Split into prepare_turn()/finalize_turn() (rather than one blocking
answer()) so /chat can stream the final answer token-by-token:
prepare_turn() runs the graph up to its interrupt_before=["generate"] pause
(rag/graph.py) and returns everything needed to build + stream the answer
from outside the graph; finalize_turn() injects the finished text back into
checkpointed state once streaming ends. See rag/graph.py's module docstring
for why this two-phase shape is necessary now, not just an added
convenience -- a single plain invoke() no longer produces a complete turn.
"""

from dataclasses import dataclass

from langchain_core.messages import AIMessage, HumanMessage

from migrantbuddy.generation.prompts import SYSTEM_PROMPT, build_prompt
from migrantbuddy.observability import observe
from migrantbuddy.rag.graph import build_graph
from migrantbuddy.retrieval import Retriever


@dataclass
class PreparedTurn:
    """Everything needed to generate and stream the final answer, once the
    graph has paused before generate. `config` is threaded back into
    finalize_turn() so the answer lands in the right thread's checkpointed
    state.
    """

    config: dict
    system_prompt: str
    user_prompt: str
    sources: list[str]


class ConversationService:
    def __init__(self, retriever: Retriever):
        self.graph = build_graph(retriever)

    @observe()
    def prepare_turn(self, thread_id: str, message: str) -> PreparedTurn:
        config = {"configurable": {"thread_id": thread_id}}
        self.graph.invoke({"messages": [HumanMessage(content=message)]}, config=config)

        state = self.graph.get_state(config).values
        query = state["messages"][-1].content
        context_chunks = state["context_chunks"]

        return PreparedTurn(
            config=config,
            system_prompt=SYSTEM_PROMPT,
            user_prompt=build_prompt(query, context_chunks),
            sources=[chunk.url for chunk in context_chunks],
        )

    @observe()
    def finalize_turn(self, prepared: PreparedTurn, answer: str) -> None:
        self.graph.update_state(
            prepared.config,
            {"messages": [AIMessage(content=answer)], "sources": prepared.sources},
            as_node="generate",
        )
        # Resumes past the interrupt to END -- a no-op completion, since
        # generate -> END is the only outgoing edge and END has no work of
        # its own.
        self.graph.invoke(None, config=prepared.config)
