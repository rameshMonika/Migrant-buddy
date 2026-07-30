import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from migrantbuddy.tts.routes import get_rate_limiter, router


class FakeRateLimiter:
    def __init__(self, allowed: bool):
        self.allowed = allowed

    def is_allowed(self, key: str) -> bool:
        return self.allowed


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_health_returns_200_ok(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_session_start_returns_token_and_id(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.tts.routes.create_session_token",
        lambda: {"session_token": "the-token", "session_id": "the-session"},
    )

    response = client.post("/session/start")

    assert response.status_code == 200
    assert response.json() == {"session_token": "the-token", "session_id": "the-session"}


def test_speak_returns_audio_base64(client, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.tts.routes.generate_speech", lambda text, previous_text=None: "the-audio"
    )

    response = client.post("/speak", json={"text": "hi"})

    assert response.status_code == 200
    assert response.json() == {"audio_base64": "the-audio"}


def test_speak_passes_text_and_previous_text_through(client, monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_generate_speech(text, previous_text=None):
        captured["text"] = text
        captured["previous_text"] = previous_text
        return "audio"

    monkeypatch.setattr("migrantbuddy.tts.routes.generate_speech", fake_generate_speech)

    client.post("/speak", json={"text": "second.", "previous_text": "first."})

    assert captured["text"] == "second."
    assert captured["previous_text"] == "first."


def test_session_start_returns_429_when_rate_limiter_denies(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_rate_limiter] = lambda: FakeRateLimiter(allowed=False)
    monkeypatch.setattr(
        "migrantbuddy.tts.routes.create_session_token",
        lambda: {"session_token": "t", "session_id": "s"},
    )
    test_client = TestClient(app)

    response = test_client.post("/session/start")

    assert response.status_code == 429


def test_speak_returns_429_when_rate_limiter_denies(monkeypatch: pytest.MonkeyPatch):
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_rate_limiter] = lambda: FakeRateLimiter(allowed=False)
    monkeypatch.setattr(
        "migrantbuddy.tts.routes.generate_speech", lambda text, previous_text=None: "audio"
    )
    test_client = TestClient(app)

    response = test_client.post("/speak", json={"text": "hi"})

    assert response.status_code == 429


def test_speak_succeeds_when_no_rate_limiter_configured(client, monkeypatch: pytest.MonkeyPatch):
    # The default `client` fixture never sets get_rate_limiter, so it
    # should resolve to None (disabled) and behave exactly as before.
    monkeypatch.setattr(
        "migrantbuddy.tts.routes.generate_speech", lambda text, previous_text=None: "audio"
    )

    response = client.post("/speak", json={"text": "hi"})

    assert response.status_code == 200
