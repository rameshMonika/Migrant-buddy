"""POST /chat -- the single endpoint the frontend calls to send a message.
ConversationService is resolved from app.state (set up in main.py at
startup) rather than constructed here, so tests can swap in a fake
ConversationService via FastAPI's dependency_overrides without touching
real models/Chroma/Ollama.

Rate limiting (get_rate_limiter) is optional -- returns None unless main.py
set up app.state.rate_limiter (RATE_LIMIT_ENABLED), in which case /chat is
skipped entirely and behaves exactly as before. Only /chat is
rate-limited, not /health -- a liveness check should always succeed.

Speech-to-text lives in its own standalone service (migrantbuddy.speech.main),
not here -- see that module's routes.py for /transcribe/ws.

GET /health is a plain liveness check -- confirms the server process is up
and routing works, independent of ConversationService/Chroma/Ollama being
reachable. Deliberately shallow (no dependency checks) so it stays fast and
doesn't flap if e.g. Ollama is briefly unavailable; a deeper readiness check
would be a separate endpoint if that's ever needed.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from migrantbuddy.api.schemas import ChatRequest, ChatResponse
from migrantbuddy.rag import ConversationService
from migrantbuddy.rate_limit import RateLimiter

router = APIRouter()


def get_conversation_service(request: Request) -> ConversationService:
    return request.app.state.conversation_service


def get_rate_limiter(request: Request) -> RateLimiter | None:
    return getattr(request.app.state, "rate_limiter", None)


def _check_rate_limit(request: Request, rate_limiter: RateLimiter | None) -> None:
    if rate_limiter is not None:
        client_key = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_key):
            raise HTTPException(status_code=429, detail="Too many requests -- please wait a moment and try again.")


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat", response_model=ChatResponse)
def chat(
    request: Request,
    chat_request: ChatRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
    rate_limiter: RateLimiter | None = Depends(get_rate_limiter),
) -> ChatResponse:
    _check_rate_limit(request, rate_limiter)

    result = conversation_service.answer(chat_request.thread_id, chat_request.message)
    return ChatResponse(
        thread_id=result.thread_id,
        message=result.message,
        answer=result.answer,
        sources=result.sources,
    )
