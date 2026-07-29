import json

import pytest

from migrantbuddy.generation.ollama_client import generate, generate_stream


class FakeResponse:
    def __init__(self, payload: dict, status_ok: bool = True):
        self._payload = payload
        self._status_ok = status_ok
        self.raised = False

    def raise_for_status(self):
        self.raised = True
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


class FakeStreamResponse:
    """Stands in for a requests.Response from a streaming call -- lines is
    a list of dicts, each rendered as one NDJSON line (Ollama's actual
    streaming format: one raw JSON object per line, no SSE "data: " prefix).
    """

    def __init__(self, lines: list[dict], status_ok: bool = True):
        self._lines = lines
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def iter_lines(self):
        for line in self._lines:
            yield json.dumps(line).encode("utf-8")


def test_generate_sends_system_and_user_messages(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse({"message": {"content": "the answer"}})

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    result = generate(
        "system prompt", "user prompt", model_name="test-model", base_url="http://fake:1234"
    )

    assert result == "the answer"
    assert captured["url"] == "http://fake:1234/api/chat"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]
    assert captured["json"]["stream"] is False


def test_generate_propagates_http_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, json, timeout):
        return FakeResponse({}, status_ok=False)

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    with pytest.raises(RuntimeError, match="HTTP error"):
        generate("system prompt", "user prompt")


def test_generate_uses_configured_timeout(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["timeout"] = timeout
        return FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    generate("system", "user", timeout=42.0)

    assert captured["timeout"] == 42.0


def test_generate_caps_output_with_num_predict(monkeypatch: pytest.MonkeyPatch):
    # Ollama's cap on generated tokens is named "num_predict" and lives
    # nested under "options" -- not a top-level field like vLLM's
    # "max_tokens". Getting this wrong means the cap silently does nothing.
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    generate("system", "user", max_tokens=256)

    assert captured["json"]["options"]["num_predict"] == 256


def test_generate_defaults_max_tokens_from_config(monkeypatch: pytest.MonkeyPatch):
    from migrantbuddy.config import GENERATION_MAX_TOKENS

    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    generate("system", "user")

    assert captured["json"]["options"]["num_predict"] == GENERATION_MAX_TOKENS


def test_generate_sends_keep_alive_to_avoid_model_reload(monkeypatch: pytest.MonkeyPatch):
    # Ollama's default keep_alive (5 minutes) unloads the model between
    # requests spaced further apart than that, forcing a multi-GB reload
    # from disk on the next request -- a large, avoidable chunk of latency.
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    generate("system", "user", keep_alive="10m")

    assert captured["json"]["keep_alive"] == "10m"


def test_generate_defaults_keep_alive_from_config(monkeypatch: pytest.MonkeyPatch):
    from migrantbuddy.config import OLLAMA_KEEP_ALIVE

    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return FakeResponse({"message": {"content": "ok"}})

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    generate("system", "user")

    assert captured["json"]["keep_alive"] == OLLAMA_KEEP_ALIVE


def test_generate_trims_dangling_sentence_when_done_reason_is_length(
    monkeypatch: pytest.MonkeyPatch,
):
    # done_reason "length" means num_predict cut the model off mid-thought --
    # trim back to the last complete sentence instead of returning a
    # dangling fragment.
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "message": {"content": "First complete sentence. Second sentence cut off mid-wo"},
                "done_reason": "length",
            }
        )

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    result = generate("system", "user")

    assert result == "First complete sentence."


def test_generate_does_not_trim_when_done_reason_is_stop(monkeypatch: pytest.MonkeyPatch):
    # A natural stop should never be touched, even if the text happens not
    # to end in recognized punctuation.
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "message": {"content": "A complete answer with no trailing period"},
                "done_reason": "stop",
            }
        )

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    result = generate("system", "user")

    assert result == "A complete answer with no trailing period"


# --- generate_stream ---


def test_generate_stream_yields_deltas_in_order(monkeypatch: pytest.MonkeyPatch):
    lines = [
        {"message": {"content": "Hello"}, "done": False},
        {"message": {"content": " world"}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "stop"},
    ]

    def fake_post(url, json, timeout, stream):
        return FakeStreamResponse(lines)

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    result = list(generate_stream("system", "user"))

    assert result == ["Hello", " world"]


def test_generate_stream_requests_streaming_mode(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, json, timeout, stream):
        captured["json"] = json
        captured["stream"] = stream
        return FakeStreamResponse(
            [{"message": {"content": ""}, "done": True, "done_reason": "stop"}]
        )

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    list(generate_stream("system", "user"))

    assert captured["json"]["stream"] is True
    assert captured["stream"] is True


def test_generate_stream_records_done_reason_in_result_info(monkeypatch: pytest.MonkeyPatch):
    lines = [
        {"message": {"content": "partial"}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "length"},
    ]

    def fake_post(url, json, timeout, stream):
        return FakeStreamResponse(lines)

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    info = {}
    list(generate_stream("system", "user", result_info=info))

    assert info["done_reason"] == "length"


def test_generate_stream_does_not_trim_yielded_tokens_even_when_truncated(
    monkeypatch: pytest.MonkeyPatch,
):
    # Unlike generate(), generate_stream() must never trim what it yields --
    # tokens already streamed live to a client can't be un-shown. Trimming
    # (if wanted) is the caller's job, applied to the accumulated text only.
    lines = [
        {"message": {"content": "First complete sentence. Second cut off mid-wo"}, "done": False},
        {"message": {"content": ""}, "done": True, "done_reason": "length"},
    ]

    def fake_post(url, json, timeout, stream):
        return FakeStreamResponse(lines)

    monkeypatch.setattr("migrantbuddy.generation.ollama_client.requests.post", fake_post)

    result = list(generate_stream("system", "user"))

    assert result == ["First complete sentence. Second cut off mid-wo"]
