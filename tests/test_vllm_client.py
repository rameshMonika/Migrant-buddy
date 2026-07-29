import pytest

from migrantbuddy.generation.vllm_client import generate


class FakeResponse:
    def __init__(self, payload: dict, status_ok: bool = True):
        self._payload = payload
        self._status_ok = status_ok

    def raise_for_status(self):
        if not self._status_ok:
            raise RuntimeError("HTTP error")

    def json(self):
        return self._payload


def test_generate_sends_openai_style_messages(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_post(url, json, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["timeout"] = timeout
        return FakeResponse({"choices": [{"message": {"content": "the answer"}}]})

    monkeypatch.setattr("migrantbuddy.generation.vllm_client.requests.post", fake_post)

    result = generate(
        "system prompt", "user prompt", model_name="test-model", base_url="http://fake:8001"
    )

    assert result == "the answer"
    assert captured["url"] == "http://fake:8001/v1/chat/completions"
    assert captured["json"]["model"] == "test-model"
    assert captured["json"]["messages"] == [
        {"role": "system", "content": "system prompt"},
        {"role": "user", "content": "user prompt"},
    ]


def test_generate_parses_choices_content_not_ollama_shape(monkeypatch: pytest.MonkeyPatch):
    # Regression check: vLLM's OpenAI-compatible response is
    # choices[0].message.content, not Ollama's message.content -- mixing
    # these up would raise a KeyError, not silently return the wrong thing.
    def fake_post(url, json, timeout):
        return FakeResponse({"choices": [{"message": {"content": "vllm answer"}}]})

    monkeypatch.setattr("migrantbuddy.generation.vllm_client.requests.post", fake_post)

    result = generate("system", "user")

    assert result == "vllm answer"


def test_generate_propagates_http_errors(monkeypatch: pytest.MonkeyPatch):
    def fake_post(url, json, timeout):
        return FakeResponse({}, status_ok=False)

    monkeypatch.setattr("migrantbuddy.generation.vllm_client.requests.post", fake_post)

    with pytest.raises(RuntimeError, match="HTTP error"):
        generate("system prompt", "user prompt")


def test_generate_caps_output_with_top_level_max_tokens(monkeypatch: pytest.MonkeyPatch):
    # vLLM's OpenAI-compatible cap is a top-level "max_tokens" field --
    # unlike Ollama's nested options.num_predict. Getting this wrong means
    # the cap silently does nothing.
    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("migrantbuddy.generation.vllm_client.requests.post", fake_post)

    generate("system", "user", max_tokens=256)

    assert captured["json"]["max_tokens"] == 256


def test_generate_defaults_max_tokens_from_config(monkeypatch: pytest.MonkeyPatch):
    from migrantbuddy.config import GENERATION_MAX_TOKENS

    captured = {}

    def fake_post(url, json, timeout):
        captured["json"] = json
        return FakeResponse({"choices": [{"message": {"content": "ok"}}]})

    monkeypatch.setattr("migrantbuddy.generation.vllm_client.requests.post", fake_post)

    generate("system", "user")

    assert captured["json"]["max_tokens"] == GENERATION_MAX_TOKENS


def test_generate_trims_dangling_sentence_when_finish_reason_is_length(
    monkeypatch: pytest.MonkeyPatch,
):
    # finish_reason "length" means max_tokens cut the model off mid-thought --
    # trim back to the last complete sentence instead of returning a
    # dangling fragment.
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {
                            "content": "First complete sentence. Second sentence cut off mid-wo"
                        },
                        "finish_reason": "length",
                    }
                ]
            }
        )

    monkeypatch.setattr("migrantbuddy.generation.vllm_client.requests.post", fake_post)

    result = generate("system", "user")

    assert result == "First complete sentence."


def test_generate_does_not_trim_when_finish_reason_is_stop(monkeypatch: pytest.MonkeyPatch):
    # A natural stop should never be touched, even if the text happens not
    # to end in recognized punctuation.
    def fake_post(url, json, timeout):
        return FakeResponse(
            {
                "choices": [
                    {
                        "message": {"content": "A complete answer with no trailing period"},
                        "finish_reason": "stop",
                    }
                ]
            }
        )

    monkeypatch.setattr("migrantbuddy.generation.vllm_client.requests.post", fake_post)

    result = generate("system", "user")

    assert result == "A complete answer with no trailing period"
