"""WS /transcribe/ws -- near-real-time speech-to-text for the frontend's mic
input. Streaming-only, no batch POST /transcribe: a short recording is just
a WS session that happens to finalize one segment when it ends, so a
separate batch endpoint would be redundant.

Protocol: the client sends binary audio chunks as they're recorded (each a
`websocket.send_bytes` call), and a text message `"stop"` once recording
ends (no more audio coming). The server replies with a JSON
TranscriptMessage (`{"text": ..., "final": true}`) each time a segment is
confirmed finished, plus a final flush of whatever's left in the buffer in
response to `"stop"`, before the connection closes.

Transcriber/RateLimiter are resolved from app.state (set up in
speech/main.py at startup), same dependency-override pattern as
migrantbuddy.api.routes, so tests can swap in fakes without touching real
models/Redis.

GET /health is a plain liveness check, same rationale as the RAG service's
-- confirms this process is up independent of the Whisper model being
loaded correctly.
"""

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect

from migrantbuddy.rate_limit import RateLimiter
from migrantbuddy.speech.schemas import TranscriptMessage
from migrantbuddy.speech.streaming import MIN_RETRANSCRIBE_INTERVAL_SECONDS, StreamingTranscriber
from migrantbuddy.speech.transcription import Transcriber

router = APIRouter()


def get_transcriber(websocket: WebSocket) -> Transcriber:
    return websocket.app.state.transcriber


def get_rate_limiter(websocket: WebSocket) -> RateLimiter | None:
    return getattr(websocket.app.state, "rate_limiter", None)


def get_streaming_interval_seconds(websocket: WebSocket) -> float:
    # Overridable via app.state (tests set this to 0 so scripted sends
    # transcribe deterministically instead of waiting on wall-clock time).
    return getattr(websocket.app.state, "streaming_interval_seconds", MIN_RETRANSCRIBE_INTERVAL_SECONDS)


@router.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}


@router.websocket("/transcribe/ws")
async def transcribe_ws(
    websocket: WebSocket,
    transcriber: Transcriber = Depends(get_transcriber),
    rate_limiter: RateLimiter | None = Depends(get_rate_limiter),
    streaming_interval_seconds: float = Depends(get_streaming_interval_seconds),
) -> None:
    if rate_limiter is not None:
        client_key = websocket.client.host if websocket.client else "unknown"
        if not rate_limiter.is_allowed(client_key):
            # Reject during the handshake -- close() before accept() sends a
            # WS close/reject rather than requiring an open connection.
            await websocket.close(code=1013, reason="Too many requests -- please wait a moment and try again.")
            return

    try:
        # The client can disconnect between uvicorn parsing the handshake
        # and this accept() actually running -- that race surfaces as a
        # RuntimeError (not WebSocketDisconnect) from the ASGI layer, since
        # there's no live connection left to send 'websocket.accept' to.
        await websocket.accept()
    except RuntimeError:
        return

    streaming = StreamingTranscriber(transcriber, min_retranscribe_interval_seconds=streaming_interval_seconds)

    try:
        while True:
            message = await websocket.receive()
            if message["type"] == "websocket.disconnect":
                return

            audio_chunk = message.get("bytes")
            if audio_chunk is not None:
                for segment in streaming.push_chunk(audio_chunk):
                    await websocket.send_json(
                        TranscriptMessage(text=segment.text, final=segment.is_final).model_dump()
                    )
            elif message.get("text") == "stop":
                for segment in streaming.flush():
                    await websocket.send_json(
                        TranscriptMessage(text=segment.text, final=segment.is_final).model_dump()
                    )
                # Explicit close -- returning here without one leaves the
                # ASGI connection open with nothing left reading it, since
                # falling off the end of the handler doesn't send a close
                # frame on its own.
                await websocket.close()
                return
    except (WebSocketDisconnect, RuntimeError):
        # RuntimeError covers the same race as above, but on send() instead
        # of accept() -- the client can vanish mid-stream too.
        return
