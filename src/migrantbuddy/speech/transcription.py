"""Speech-to-text via faster-whisper. Multilingual by default -- Whisper
auto-detects the spoken language per clip (no language pinned), matching
the target population (Burmese, Tamil, Thai, Vietnamese, etc.) the same way
BGE-M3/SEA-LION handle multilingual text without a translation step.

A class, not a bare function -- loading the model is expensive (weights
into memory), so it's built once at app startup and reused across
requests, same pattern as Retriever.

Requires ffmpeg on PATH (used internally for audio decoding) -- a system
dependency, not something `pip install` covers.
"""

from typing import BinaryIO

from faster_whisper import WhisperModel

from migrantbuddy.config import WHISPER_COMPUTE_TYPE, WHISPER_DEVICE, WHISPER_MODEL_SIZE


class Transcriber:
    def __init__(
        self,
        model_size: str = WHISPER_MODEL_SIZE,
        *,
        device: str = WHISPER_DEVICE,
        compute_type: str = WHISPER_COMPUTE_TYPE,
    ):
        self._model = WhisperModel(model_size, device=device, compute_type=compute_type)

    def transcribe(self, audio: BinaryIO) -> str:
        segments, _info = self._model.transcribe(audio)
        return " ".join(segment.text.strip() for segment in segments).strip()
