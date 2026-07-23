"""Conversation state for the LangGraph turn graph. Extends LangGraph's
MessagesState (which already provides `messages`, appended via its
built-in reducer) with the fields it doesn't provide out of the box.

thread_id is deliberately NOT a state field -- LangGraph identifies which
conversation's state to load/save via `config["configurable"]["thread_id"]`
at invoke time, handled by the checkpointer; it isn't data that flows
between nodes the way these fields are.
"""

from langgraph.graph import MessagesState

from migrantbuddy.indexing import Chunk


class ConversationState(MessagesState):
    # Condensed older turns, once history grows past the summarization
    # threshold (see rag/nodes.py). Empty string until that first happens.
    summary: str
    # Set by rewrite_query_node; read by retrieve_node. Falls back to the
    # latest raw message if rewriting was skipped (e.g. first turn).
    standalone_query: str
    # Set by retrieve_node; read by generate_node.
    context_chunks: list[Chunk]
    # Set by generate_node; surfaced in the final result for the frontend
    # to render as source links.
    sources: list[str]
