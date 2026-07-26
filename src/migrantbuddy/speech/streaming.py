"""Near-real-time transcription for the WebSocket endpoint. faster-whisper
doesn't support incremental decoding -- there's no way to feed it "one more
second of audio" and get an update. Instead, StreamingTranscriber re-runs a
full batch transcription (vad_filter=True, which segments speech from
silence) over everything received so far, and only emits segments that are
"finalized" -- followed by a silence gap, so a later segment (or the
recording ending) has confirmed they're done. The most recent segment is
never emitted as final while more audio might still be coming, since
VAD/Whisper can revise it as trailing context arrives.

Re-transcription is throttled to roughly once per MIN_RETRANSCRIBE_INTERVAL_SECONDS
of wall-clock time, not on every push_chunk() call. The frontend streams
audio in small chunks (e.g. every 250ms) so the UI can react quickly, but
re-running Whisper on the whole accumulated buffer that often means most of
the compute is spent re-decoding audio that was already decoded moments ago
for a result that hasn't meaningfully changed -- on CPU this can fall behind
realtime within a few seconds of recording. Batching several chunks between
transcriptions cuts that redundant work by roughly the same factor as the
interval, without losing any accuracy (flush() always transcribes
everything buffered, regardless of the throttle, so nothing is ever
dropped -- only the intermediate updates arrive less often).

Deliberately does not trim the buffer once a segment is finalized: the
buffer holds the raw compressed container bytes (webm/opus) exactly as the
browser's MediaRecorder emits them, and that format isn't byte-sliceable at
an arbitrary time offset -- only the first chunk carries the header needed
to decode anything after it. So re-transcription cost still grows with
total elapsed recording time, which is an acceptable tradeoff for a single
spoken question (typically well under a minute), not a live-captioning
session of arbitrary length.
"""

import io
import time
from dataclasses import dataclass
from typing import Callable

from migrantbuddy.speech.transcription import Transcriber

MIN_RETRANSCRIBE_INTERVAL_SECONDS = 1.5


@dataclass
class TranscriptSegment:
    text: str
    is_final: bool


class StreamingTranscriber:
    def __init__(
        self,
        transcriber: Transcriber | None = None,
        *,
        min_retranscribe_interval_seconds: float = MIN_RETRANSCRIBE_INTERVAL_SECONDS,
        clock: Callable[[], float] = time.monotonic,
    ):
        self._transcriber = transcriber if transcriber is not None else Transcriber()
        self._buffer = bytearray()
        self._finalized_count = 0
        self._min_interval = min_retranscribe_interval_seconds
        self._clock = clock
        self._last_transcribe_time: float | None = None

    def push_chunk(self, chunk: bytes) -> list[TranscriptSegment]:
        """Feed one more chunk of audio. Returns newly-finalized segments
        (each emitted exactly once) -- may be empty either because nothing
        new has been confirmed yet, or because this call landed inside the
        current batching window and skipped re-transcribing altogether.
        """
        self._buffer.extend(chunk)

        now = self._clock()
        if self._last_transcribe_time is not None and now - self._last_transcribe_time < self._min_interval:
            return []
        self._last_transcribe_time = now

        segments = self._transcribe_buffer()
        if not segments:
            return []

        # The last segment might still be growing -- only segments before it
        # are confirmed finished.
        newly_finalized = segments[self._finalized_count : -1]
        self._finalized_count = len(segments) - 1
        return self._to_results(newly_finalized)

    def flush(self) -> list[TranscriptSegment]:
        """Called once the client signals no more audio is coming. Always
        transcribes regardless of the batching window -- this is the last
        chance to return whatever's left, including the previously
        in-progress trailing segment.
        """
        segments = self._transcribe_buffer()
        remaining = segments[self._finalized_count :]
        self._finalized_count = len(segments)
        return self._to_results(remaining)

    def _transcribe_buffer(self) -> list:
        if not self._buffer:
            return []
        segments, _info = self._transcriber._model.transcribe(io.BytesIO(bytes(self._buffer)), vad_filter=True)
        return list(segments)

    @staticmethod
    def _to_results(segments: list) -> list[TranscriptSegment]:
        return [
            TranscriptSegment(text=text, is_final=True)
            for segment in segments
            if (text := segment.text.strip())
        ]
