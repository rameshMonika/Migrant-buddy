import pytest
from fastapi import FastAPI, WebSocketDisconnect
from fastapi.testclient import TestClient

from migrantbuddy.speech.routes import get_rate_limiter, get_streaming_interval_seconds, get_transcriber, router


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class ScriptedWhisperModel:
    """Stands in for faster-whisper's WhisperModel -- each call to
    transcribe() returns the next scripted batch of segment texts,
    regardless of what audio bytes were actually buffered. Lets tests drive
    StreamingTranscriber's finalization logic without a real model.
    """

    def __init__(self, segment_text_batches: list[list[str]]):
        self._batches = iter(segment_text_batches)

    def transcribe(self, audio, vad_filter=True):
        try:
            texts = next(self._batches)
        except StopIteration:
            texts = []
        return [FakeSegment(text) for text in texts], object()


class FakeTranscriber:
    def __init__(self, model: ScriptedWhisperModel):
        self._model = model


class FakeRateLimiter:
    def __init__(self, allowed: bool):
        self.allowed = allowed

    def is_allowed(self, key: str) -> bool:
        return self.allowed


def build_client(model: ScriptedWhisperModel, *, rate_limiter=None) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_transcriber] = lambda: FakeTranscriber(model)
    # Disable the re-transcription batching window -- these tests send
    # scripted chunks back-to-back with no real time elapsing, so every
    # push_chunk() call should transcribe immediately and deterministically.
    app.dependency_overrides[get_streaming_interval_seconds] = lambda: 0
    if rate_limiter is not None:
        app.dependency_overrides[get_rate_limiter] = lambda: rate_limiter
    return TestClient(app)


def test_health_returns_200_ok():
    client = build_client(ScriptedWhisperModel([]))

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_ws_sends_finalized_segment_once_a_later_segment_confirms_it():
    # First push: only one (still-growing) segment -- nothing finalized yet.
    # Second push: a second segment appears, so "hello" is now confirmed done.
    model = ScriptedWhisperModel([["hello"], ["hello", "world"]])
    client = build_client(model)

    with client.websocket_connect("/transcribe/ws") as ws:
        ws.send_bytes(b"chunk-1")
        ws.send_bytes(b"chunk-2")

        message = ws.receive_json()

    assert message == {"text": "hello", "final": True}


def test_ws_flushes_trailing_segment_on_stop():
    model = ScriptedWhisperModel([["hello", "world"], ["hello", "world"]])
    client = build_client(model)

    with client.websocket_connect("/transcribe/ws") as ws:
        ws.send_bytes(b"chunk-1")
        first = ws.receive_json()

        ws.send_text("stop")
        second = ws.receive_json()

    assert first == {"text": "hello", "final": True}
    assert second == {"text": "world", "final": True}


def test_ws_closes_after_stop_is_handled():
    model = ScriptedWhisperModel([["hello"], ["hello"]])
    client = build_client(model)

    with client.websocket_connect("/transcribe/ws") as ws:
        ws.send_bytes(b"chunk-1")
        ws.send_text("stop")
        message = ws.receive_json()

        with pytest.raises(WebSocketDisconnect):
            ws.receive_json()

    assert message == {"text": "hello", "final": True}


def test_ws_rejects_connection_when_rate_limiter_denies():
    client = build_client(ScriptedWhisperModel([]), rate_limiter=FakeRateLimiter(allowed=False))

    with pytest.raises(WebSocketDisconnect):
        with client.websocket_connect("/transcribe/ws"):
            pass
