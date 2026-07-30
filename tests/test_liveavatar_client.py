import pytest

from migrantbuddy.tts.liveavatar_client import create_session_token


class FakeResponse:
    def __init__(self, session_token: str = "token-1", session_id: str = "session-1"):
        self._data = {"session_token": session_token, "session_id": session_id}

    def raise_for_status(self):
        pass

    def json(self):
        return {"data": self._data}


def test_create_session_token_returns_token_and_id(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, headers, json, timeout):
        return FakeResponse(session_token="the-token", session_id="the-session")

    monkeypatch.setattr("migrantbuddy.tts.liveavatar_client.requests.post", fake_post)

    result = create_session_token(avatar_id="avatar-1", is_sandbox=True, api_key="key-1")

    assert result == {"session_token": "the-token", "session_id": "the-session"}


def test_create_session_token_sends_correct_url_headers_and_body(
    monkeypatch: pytest.MonkeyPatch,
):
    captured = {}

    def fake_post(url, headers, json, timeout):
        captured["url"] = url
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("migrantbuddy.tts.liveavatar_client.requests.post", fake_post)

    create_session_token(
        avatar_id="avatar-1", is_sandbox=False, api_key="key-1", api_url="https://api.example.com"
    )

    assert captured["url"] == "https://api.example.com/v1/sessions/token"
    assert captured["headers"]["X-API-KEY"] == "key-1"
    assert captured["json"] == {"mode": "LITE", "avatar_id": "avatar-1", "is_sandbox": False}


def test_create_session_token_propagates_http_errors(monkeypatch: pytest.MonkeyPatch):
    class FailingResponse:
        def raise_for_status(self):
            raise RuntimeError("HTTP error")

    def fake_post(url, headers, json, timeout):
        return FailingResponse()

    monkeypatch.setattr("migrantbuddy.tts.liveavatar_client.requests.post", fake_post)

    with pytest.raises(RuntimeError, match="HTTP error"):
        create_session_token()
