"""FastAPI app factory for the standalone TTS (ElevenLabs) service -- its
own process, its own port, run independently of the RAG service and the
Whisper service. Owns nothing but its own rate limiter -- no local model to
load (ElevenLabs is a remote API), so restarting or scaling this service
never touches the others. See README for how to run all three together.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from migrantbuddy.config import (
    FRONTEND_ORIGIN,
    TTS_RATE_LIMIT_ENABLED,
    TTS_RATE_LIMIT_MAX_REQUESTS,
    TTS_RATE_LIMIT_WINDOW_SECONDS,
)
from migrantbuddy.rate_limit import RateLimiter
from migrantbuddy.tts.routes import router


def create_app() -> FastAPI:
    app = FastAPI(title="migrantBuddy TTS")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.rate_limiter = (
        RateLimiter(
            max_requests=TTS_RATE_LIMIT_MAX_REQUESTS,
            window_seconds=TTS_RATE_LIMIT_WINDOW_SECONDS,
        )
        if TTS_RATE_LIMIT_ENABLED
        else None
    )

    app.include_router(router)
    return app


app = create_app()
