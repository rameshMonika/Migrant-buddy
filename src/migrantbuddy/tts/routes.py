"""Two plain POST endpoints backing the frontend's LiveAvatar integration --
no websocket relay, no local viseme analysis, both replaced by LiveAvatar's
own server-side lip-sync (see config.py's LiveAvatar comment).

POST /session/start mints a LiveAvatar LITE-mode session token (the only
step that needs our secret LIVEAVATAR_API_KEY -- everything after this
happens client-side with the short-lived token this returns). POST /speak
generates one sentence's ElevenLabs audio for the frontend to feed into
that session via `repeatAudio()`.

Rate limiting follows the same `get_rate_limiter(request)` +
`_check_rate_limit` pattern as api/routes.py's /chat, not the
websocket-flavored dependency this file used when it exposed /speak/ws.

GET /health is a plain liveness check, same rationale as the other
services' -- confirms this process is up, independent of ElevenLabs/
LiveAvatar being reachable.
"""

from fastapi import APIRouter, Depends, HTTPException, Request

from migrantbuddy.rate_limit import RateLimiter
from migrantbuddy.tts.elevenlabs_client import generate_speech
from migrantbuddy.tts.liveavatar_client import create_session_token
from migrantbuddy.tts.schemas import SessionStartResponse, SpeakRequest, SpeakResponse

router = APIRouter()


def get_rate_limiter(request: Request) -> RateLimiter | None:
    return getattr(request.app.state, "rate_limiter", None)


def _check_rate_limit(request: Request, rate_limiter: RateLimiter | None) -> None:
    if rate_limiter is not None:
        client_key = request.client.host if request.client else "unknown"
        if not rate_limiter.is_allowed(client_key):
            raise HTTPException(
                status_code=429, detail="Too many requests -- please wait a moment and try again."
            )


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.post("/session/start")
def start_session(
    request: Request, rate_limiter: RateLimiter | None = Depends(get_rate_limiter)
) -> SessionStartResponse:
    _check_rate_limit(request, rate_limiter)
    return SessionStartResponse(**create_session_token())


@router.post("/speak")
def speak(
    request: Request,
    speak_request: SpeakRequest,
    rate_limiter: RateLimiter | None = Depends(get_rate_limiter),
) -> SpeakResponse:
    _check_rate_limit(request, rate_limiter)
    audio_base64 = generate_speech(speak_request.text, previous_text=speak_request.previous_text)
    return SpeakResponse(audio_base64=audio_base64)
