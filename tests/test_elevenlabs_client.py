import pytest

from migrantbuddy.tts.elevenlabs_client import generate_speech


class FakeResponse:
    def __init__(self, audio_base64: str = "aaaa", status_ok: bool = True):
        self._audio_base64 = audio_base64
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return {"audio_base64": self._audio_base64}


def test_generate_speech_returns_audio_base64(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, params, headers, json, timeout):
        return FakeResponse(audio_base64="the-audio")

    monkeypatch.setattr("migrantbuddy.tts.elevenlabs_client.requests.post", fake_post)

    result = generate_speech("hello", voice_id="voice-1", api_key="key-1")

    assert result == "the-audio"


def test_generate_speech_sends_correct_url_and_auth_header(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, params, headers, json, timeout):
        captured["url"] = url
        captured["params"] = params
        captured["headers"] = headers
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("migrantbuddy.tts.elevenlabs_client.requests.post", fake_post)

    generate_speech("hello", voice_id="voice-1", model_id="model-1", api_key="key-1")

    assert (
        captured["url"] == "https://api.elevenlabs.io/v1/text-to-speech/voice-1/with-timestamps"
    )
    assert captured["headers"]["xi-api-key"] == "key-1"
    assert captured["json"]["text"] == "hello"
    assert captured["json"]["model_id"] == "model-1"
    assert "previous_text" not in captured["json"]


def test_generate_speech_requests_pcm_24000_output_format(monkeypatch: pytest.MonkeyPatch):
    # Regression check: LiveAvatar LITE mode's repeatAudio() ingest is
    # hardcoded to 24kHz PCM -- see elevenlabs_client.py's module docstring.
    captured = {}

    def fake_post(url, params, headers, json, timeout):
        captured["params"] = params
        return FakeResponse()

    monkeypatch.setattr("migrantbuddy.tts.elevenlabs_client.requests.post", fake_post)

    generate_speech("hello")

    assert captured["params"] == {"output_format": "pcm_24000"}


def test_generate_speech_includes_previous_text_when_given(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, params, headers, json, timeout):
        captured["json"] = json
        return FakeResponse()

    monkeypatch.setattr("migrantbuddy.tts.elevenlabs_client.requests.post", fake_post)

    generate_speech("hello", previous_text="earlier sentence.")

    assert captured["json"]["previous_text"] == "earlier sentence."


def test_generate_speech_propagates_http_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, params, headers, json, timeout):
        return FakeResponse(status_ok=False)

    monkeypatch.setattr("migrantbuddy.tts.elevenlabs_client.requests.post", fake_post)

    with pytest.raises(RuntimeError, match="HTTP error"):
        generate_speech("hello")
