"""POST /chat -- the single endpoint the frontend calls to send a message.
Streams the answer back as Server-Sent Events instead of one JSON blob, so
text appears as it's generated (time-to-first-token) instead of only after
the full answer completes -- see rag/service.py and rag/graph.py for how
this interacts with conversation memory/checkpointing.

Event sequence: `sources` (once retrieval completes, before any generation)
-> repeated `token` (one per text delta) -> `done` (stream end). Hand-rolled
SSE formatting, no new dependency -- this is simple enough not to need one.

ConversationService is resolved from app.state (set up in main.py at
startup) rather than constructed here, so tests can swap in a fake
ConversationService via FastAPI's dependency_overrides without touching
real models/Chroma/Ollama.

Rate limiting (get_rate_limiter) is optional -- returns None unless main.py
set up app.state.rate_limiter (RATE_LIMIT_ENABLED), in which case /chat is
skipped entirely and behaves exactly as before. Checked before the stream
starts -- a denied request still gets an immediate 429, not a stream that
opens and then errors. Only /chat is rate-limited, not /health -- a
liveness check should always succeed.

Speech-to-text lives in its own standalone service (migrantbuddy.speech.main),
not here -- see that module's routes.py for /transcribe/ws.

GET /health is a plain liveness check -- confirms the server process is up
and routing works, independent of ConversationService/Chroma/Ollama being
reachable. Deliberately shallow (no dependency checks) so it stays fast and
doesn't flap if e.g. Ollama is briefly unavailable; a deeper readiness check
would be a separate endpoint if that's ever needed.
"""

import json

from fastapi import APIRouter, Depends, HTTPException, Request
from fastapi.responses import StreamingResponse

from migrantbuddy.api.schemas import ChatRequest
from migrantbuddy.generation.service import stream_answer
from migrantbuddy.generation.truncation import trim_to_last_complete_sentence
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
            raise HTTPException(
                status_code=429, detail="Too many requests -- please wait a moment and try again."
            )


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/chat")
def chat(
    request: Request,
    chat_request: ChatRequest,
    conversation_service: ConversationService = Depends(get_conversation_service),
    rate_limiter: RateLimiter | None = Depends(get_rate_limiter),
) -> StreamingResponse:
    _check_rate_limit(request, rate_limiter)

    # Runs summarize/rewrite_query/retrieve synchronously (fast, non-LLM
    # for the last one) before the stream opens -- only the final answer
    # itself streams.
    prepared = conversation_service.prepare_turn(chat_request.thread_id, chat_request.message)

    def event_stream():
        yield _sse("sources", {"sources": prepared.sources})

        result_info: dict = {}
        full_answer = ""
        for delta in stream_answer(
            prepared.system_prompt, prepared.user_prompt, result_info=result_info
        ):
            full_answer += delta
            yield _sse("token", {"text": delta})

        # Trims only the *saved* copy -- tokens already streamed live can't
        # be un-shown, see generation/service.py's stream_answer() docstring.
        if result_info.get("truncated"):
            full_answer = trim_to_last_complete_sentence(full_answer)

        conversation_service.finalize_turn(prepared, full_answer)
        yield _sse("done", {})

    return StreamingResponse(event_stream(), media_type="text/event-stream")
