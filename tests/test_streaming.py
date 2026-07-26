from migrantbuddy.speech.streaming import StreamingTranscriber


class FakeSegment:
    def __init__(self, text: str):
        self.text = text


class ScriptedWhisperModel:
    """See tests/test_speech_routes.py for the rationale -- each transcribe()
    call returns the next scripted batch of segment texts, independent of
    the actual buffered bytes.
    """

    def __init__(self, segment_text_batches: list[list[str]]):
        self._batches = iter(segment_text_batches)
        self.call_count = 0

    def transcribe(self, audio, vad_filter=True):
        self.call_count += 1
        try:
            texts = next(self._batches)
        except StopIteration:
            texts = []
        return [FakeSegment(text) for text in texts], object()


class FakeTranscriber:
    def __init__(self, model: ScriptedWhisperModel):
        self._model = model


class FakeClock:
    """A controllable clock so tests can simulate time passing without
    real sleeps."""

    def __init__(self, start: float = 0.0):
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


# Most tests disable the batching window entirely (interval=0) so every
# push_chunk() call transcribes immediately and deterministically -- the
# throttling behavior itself is covered separately below.
def make_streaming(model: ScriptedWhisperModel, **kwargs) -> StreamingTranscriber:
    kwargs.setdefault("min_retranscribe_interval_seconds", 0)
    return StreamingTranscriber(FakeTranscriber(model), **kwargs)


def test_push_chunk_does_not_finalize_the_only_segment():
    model = ScriptedWhisperModel([["hello"]])
    streaming = make_streaming(model)

    result = streaming.push_chunk(b"chunk-1")

    assert result == []


def test_push_chunk_finalizes_once_a_later_segment_appears():
    model = ScriptedWhisperModel([["hello"], ["hello", "world"]])
    streaming = make_streaming(model)

    streaming.push_chunk(b"chunk-1")
    result = streaming.push_chunk(b"chunk-2")

    assert [segment.text for segment in result] == ["hello"]
    assert all(segment.is_final for segment in result)


def test_push_chunk_only_emits_each_finalized_segment_once():
    model = ScriptedWhisperModel([["hello", "world"], ["hello", "world", "again"]])
    streaming = make_streaming(model)

    first = streaming.push_chunk(b"chunk-1")
    second = streaming.push_chunk(b"chunk-2")

    assert [segment.text for segment in first] == ["hello"]
    assert [segment.text for segment in second] == ["world"]


def test_flush_emits_the_trailing_segment():
    model = ScriptedWhisperModel([["hello", "world"], ["hello", "world"]])
    streaming = make_streaming(model)

    streaming.push_chunk(b"chunk-1")
    result = streaming.flush()

    assert [segment.text for segment in result] == ["world"]


def test_flush_on_empty_buffer_emits_nothing():
    model = ScriptedWhisperModel([])
    streaming = make_streaming(model)

    assert streaming.flush() == []
    assert model.call_count == 0


def test_blank_segments_are_dropped():
    model = ScriptedWhisperModel([["  "], ["  ", "hello"]])
    streaming = make_streaming(model)

    streaming.push_chunk(b"chunk-1")
    result = streaming.push_chunk(b"chunk-2")

    assert result == []


# --- re-transcription batching window ---


def test_push_chunk_skips_retranscription_within_the_batching_window():
    model = ScriptedWhisperModel([["hello"], ["hello", "world"]])
    clock = FakeClock()
    streaming = StreamingTranscriber(
        FakeTranscriber(model), min_retranscribe_interval_seconds=1.5, clock=clock
    )

    streaming.push_chunk(b"chunk-1")  # first call always transcribes
    clock.advance(0.5)
    result = streaming.push_chunk(b"chunk-2")  # still within the 1.5s window

    assert result == []
    assert model.call_count == 1


def test_push_chunk_retranscribes_once_the_window_elapses():
    model = ScriptedWhisperModel([["hello"], ["hello", "world"]])
    clock = FakeClock()
    streaming = StreamingTranscriber(
        FakeTranscriber(model), min_retranscribe_interval_seconds=1.5, clock=clock
    )

    streaming.push_chunk(b"chunk-1")
    clock.advance(1.5)
    result = streaming.push_chunk(b"chunk-2")

    assert [segment.text for segment in result] == ["hello"]
    assert model.call_count == 2


def test_flush_transcribes_even_within_the_batching_window():
    model = ScriptedWhisperModel([["hello", "world"], ["hello", "world", "again"]])
    clock = FakeClock()
    streaming = StreamingTranscriber(
        FakeTranscriber(model), min_retranscribe_interval_seconds=1.5, clock=clock
    )

    streaming.push_chunk(b"chunk-1")  # finalizes "hello", call_count == 1
    clock.advance(0.1)  # nowhere near the window elapsing
    result = streaming.flush()

    assert [segment.text for segment in result] == ["world", "again"]
    assert model.call_count == 2
