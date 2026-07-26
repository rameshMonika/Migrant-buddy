"""FastAPI app factory for the standalone Whisper/speech-to-text service --
its own process, its own port, run independently of the RAG service
(migrantbuddy.api.main). Owns nothing but a Transcriber (faster-whisper
model, loaded once at startup) and its own rate limiter -- no
Chroma/retriever/LLM, so restarting or scaling one service never touches
the other. See README for how to run both.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from migrantbuddy.config import (
    FRONTEND_ORIGIN,
    WHISPER_RATE_LIMIT_ENABLED,
    WHISPER_RATE_LIMIT_MAX_REQUESTS,
    WHISPER_RATE_LIMIT_WINDOW_SECONDS,
)
from migrantbuddy.rate_limit import RateLimiter
from migrantbuddy.speech.routes import router
from migrantbuddy.speech.transcription import Transcriber


def create_app() -> FastAPI:
    app = FastAPI(title="migrantBuddy Speech")

    app.add_middleware(
        CORSMiddleware,
        allow_origins=[FRONTEND_ORIGIN],
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.state.transcriber = Transcriber()
    app.state.rate_limiter = (
        RateLimiter(
            max_requests=WHISPER_RATE_LIMIT_MAX_REQUESTS,
            window_seconds=WHISPER_RATE_LIMIT_WINDOW_SECONDS,
        )
        if WHISPER_RATE_LIMIT_ENABLED
        else None
    )

    app.include_router(router)
    return app


app = create_app()
