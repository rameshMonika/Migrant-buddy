import io

import pytest

from migrantbuddy.speech.transcription import Transcriber


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class FakeWhisperModel:
    def __init__(self, model_size: str, device: str, compute_type: str):
        self.model_size = model_size
        self.device = device
        self.compute_type = compute_type
        self.last_audio = None

    def transcribe(self, audio):
        self.last_audio = audio
        return [FakeSegment(" hello "), FakeSegment(" world ")], object()


def test_transcribe_joins_and_strips_segment_texts(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("migrantbuddy.speech.transcription.WhisperModel", FakeWhisperModel)
    transcriber = Transcriber()

    result = transcriber.transcribe(io.BytesIO(b"fake audio bytes"))

    assert result == "hello world"


def test_transcribe_handles_no_segments(monkeypatch: pytest.MonkeyPatch):
    class EmptyModel(FakeWhisperModel):
        def transcribe(self, audio):
            return [], object()

    monkeypatch.setattr("migrantbuddy.speech.transcription.WhisperModel", EmptyModel)
    transcriber = Transcriber()

    assert transcriber.transcribe(io.BytesIO(b"fake audio bytes")) == ""


def test_transcriber_passes_configured_model_params(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    class CapturingModel(FakeWhisperModel):
        def __init__(self, model_size, device, compute_type):
            captured["model_size"] = model_size
            captured["device"] = device
            captured["compute_type"] = compute_type
            super().__init__(model_size, device, compute_type)

    monkeypatch.setattr("migrantbuddy.speech.transcription.WhisperModel", CapturingModel)

    Transcriber(model_size="medium", device="cuda", compute_type="float16")

    assert captured == {"model_size": "medium", "device": "cuda", "compute_type": "float16"}


def test_transcribe_passes_audio_through_to_model(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr("migrantbuddy.speech.transcription.WhisperModel", FakeWhisperModel)
    transcriber = Transcriber()
    audio = io.BytesIO(b"fake audio bytes")

    transcriber.transcribe(audio)

    assert transcriber._model.last_audio is audio
