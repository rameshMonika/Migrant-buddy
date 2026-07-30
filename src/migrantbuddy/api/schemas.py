"""Request/response shapes for the API. Pydantic (via FastAPI) so invalid
requests are rejected before they reach ConversationService.

thread_id identifies the conversation -- the frontend generates one per
conversation and reuses it across turns; ConversationService's checkpointer
uses it to load/save that conversation's state server-side, so the client
only ever needs to send the latest message, not the full history.
"""

from pydantic import BaseModel


class ChatRequest(BaseModel):
    message: str
    thread_id: str
