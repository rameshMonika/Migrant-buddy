import pytest

from migrantbuddy.generation.service import generate_answer, stream_answer
from migrantbuddy.indexing import Chunk


def make_chunk(url: str, text: str) -> Chunk:
    return Chunk(
        chunk_id=f"{url}::chunk-0",
        document_id=url,
        url=url,
        heading_path="Section",
        text=text,
        token_count=len(text) // 4,
    )


def test_generate_answer_returns_query_answer_and_sources(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_ollama_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        captured["system_prompt"] = system_prompt
        captured["user_prompt"] = user_prompt
        captured["model_name"] = model_name
        captured["max_tokens"] = max_tokens
        return "the generated answer"

    monkeypatch.setattr("migrantbuddy.generation.service.ollama_generate", fake_ollama_generate)
    chunks = [make_chunk("https://example.com/a", "context about salary")]

    result = generate_answer(
        "When must my employer pay my salary?", chunks, model_name="test-model"
    )

    assert result.query == "When must my employer pay my salary?"
    assert result.answer == "the generated answer"
    assert result.sources == ["https://example.com/a"]
    assert captured["model_name"] == "test-model"
    assert "context about salary" in captured["user_prompt"]
    assert "When must my employer pay my salary?" in captured["user_prompt"]


def test_generate_answer_collects_sources_from_all_chunks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.generation.service.ollama_generate",
        lambda system_prompt, user_prompt, *, model_name, max_tokens: "answer",
    )
    chunks = [
        make_chunk("https://example.com/a", "text a"),
        make_chunk("https://example.com/b", "text b"),
    ]

    result = generate_answer("query", chunks)

    assert result.sources == ["https://example.com/a", "https://example.com/b"]


def test_generate_answer_handles_no_context_chunks(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.generation.service.ollama_generate",
        lambda system_prompt, user_prompt, *, model_name, max_tokens: "I don't know",
    )

    result = generate_answer("query", [])

    assert result.sources == []
    assert result.answer == "I don't know"


def test_generate_answer_defaults_to_ollama_backend(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.generation.service.ollama_generate",
        lambda system_prompt, user_prompt, *, model_name, max_tokens: "from ollama",
    )
    monkeypatch.setattr(
        "migrantbuddy.generation.service.vllm_generate",
        lambda system_prompt, user_prompt, *, model_name, max_tokens: "from vllm",
    )

    result = generate_answer("query", [])

    assert result.answer == "from ollama"


def test_generate_answer_uses_vllm_backend_when_selected(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(
        "migrantbuddy.generation.service.ollama_generate",
        lambda system_prompt, user_prompt, *, model_name, max_tokens: "from ollama",
    )
    captured = {}

    def fake_vllm_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        captured["model_name"] = model_name
        captured["max_tokens"] = max_tokens
        return "from vllm"

    monkeypatch.setattr("migrantbuddy.generation.service.vllm_generate", fake_vllm_generate)

    result = generate_answer("query", [], model_name="vllm-model", backend="vllm")

    assert result.answer == "from vllm"
    assert captured["model_name"] == "vllm-model"


def test_generate_answer_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown generation backend"):
        generate_answer("query", [], backend="something-else")


def test_generate_answer_passes_configured_max_tokens(monkeypatch: pytest.MonkeyPatch):
    captured = {}

    def fake_ollama_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        captured["max_tokens"] = max_tokens
        return "answer"

    monkeypatch.setattr("migrantbuddy.generation.service.ollama_generate", fake_ollama_generate)

    generate_answer("query", [], max_tokens=256)

    assert captured["max_tokens"] == 256


def test_generate_answer_defaults_max_tokens_from_config(monkeypatch: pytest.MonkeyPatch):
    from migrantbuddy.config import GENERATION_MAX_TOKENS

    captured = {}

    def fake_ollama_generate(system_prompt, user_prompt, *, model_name, max_tokens):
        captured["max_tokens"] = max_tokens
        return "answer"

    monkeypatch.setattr("migrantbuddy.generation.service.ollama_generate", fake_ollama_generate)

    generate_answer("query", [])

    assert captured["max_tokens"] == GENERATION_MAX_TOKENS


# --- stream_answer ---


def test_stream_answer_yields_deltas_from_ollama_by_default(monkeypatch: pytest.MonkeyPatch):
    def fake_stream(system_prompt, user_prompt, *, model_name, max_tokens, result_info):
        result_info["done_reason"] = "stop"
        yield "Hello"
        yield " world"

    monkeypatch.setattr("migrantbuddy.generation.service.ollama_generate_stream", fake_stream)

    result = list(stream_answer("system", "user"))

    assert result == ["Hello", " world"]


def test_stream_answer_uses_vllm_backend_when_selected(monkeypatch: pytest.MonkeyPatch):
    def fake_stream(system_prompt, user_prompt, *, model_name, max_tokens, result_info):
        result_info["finish_reason"] = "stop"
        yield "from vllm"

    monkeypatch.setattr("migrantbuddy.generation.service.vllm_generate_stream", fake_stream)

    result = list(stream_answer("system", "user", backend="vllm"))

    assert result == ["from vllm"]


def test_stream_answer_normalizes_ollama_truncation_into_result_info(
    monkeypatch: pytest.MonkeyPatch,
):
    def fake_stream(system_prompt, user_prompt, *, model_name, max_tokens, result_info):
        result_info["done_reason"] = "length"
        yield "partial"

    monkeypatch.setattr("migrantbuddy.generation.service.ollama_generate_stream", fake_stream)

    info = {}
    list(stream_answer("system", "user", result_info=info))

    assert info["truncated"] is True


def test_stream_answer_normalizes_vllm_truncation_into_result_info(monkeypatch: pytest.MonkeyPatch):
    def fake_stream(system_prompt, user_prompt, *, model_name, max_tokens, result_info):
        result_info["finish_reason"] = "length"
        yield "partial"

    monkeypatch.setattr("migrantbuddy.generation.service.vllm_generate_stream", fake_stream)

    info = {}
    list(stream_answer("system", "user", backend="vllm", result_info=info))

    assert info["truncated"] is True


def test_stream_answer_reports_not_truncated_on_natural_stop(monkeypatch: pytest.MonkeyPatch):
    def fake_stream(system_prompt, user_prompt, *, model_name, max_tokens, result_info):
        result_info["done_reason"] = "stop"
        yield "complete"

    monkeypatch.setattr("migrantbuddy.generation.service.ollama_generate_stream", fake_stream)

    info = {}
    list(stream_answer("system", "user", result_info=info))

    assert info["truncated"] is False


def test_stream_answer_rejects_unknown_backend():
    with pytest.raises(ValueError, match="Unknown generation backend"):
        list(stream_answer("system", "user", backend="something-else"))
