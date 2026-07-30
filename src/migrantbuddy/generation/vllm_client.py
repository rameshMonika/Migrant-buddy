"""Thin HTTP client for vLLM's OpenAI-compatible chat completions API.

Same generate(...) -> str signature as ollama_client.py -- this is the
production swap CLAUDE.md's Generation serving decision calls for
(continuous batching / PagedAttention for throughput). Which one actually
runs is picked by migrantbuddy.config.GENERATION_BACKEND, dispatched in
generation/service.py.

Request/response shape differs from Ollama's /api/chat: vLLM serves the
standard OpenAI /v1/chat/completions schema (choices[0].message.content),
not Ollama's message.content.

generate_stream() is the streaming counterpart used by /chat -- see
ollama_client.py's generate_stream() docstring for the rationale (plain
sync generator, trim applied only to the saved/history copy, not to
already-displayed live tokens).
"""

import json

import requests

from migrantbuddy.config import GENERATION_MAX_TOKENS, GENERATION_MODEL_NAME, VLLM_BASE_URL
from migrantbuddy.generation.truncation import trim_to_last_complete_sentence
from migrantbuddy.observability import observe


@observe(as_type="generation")
def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    model_name: str = GENERATION_MODEL_NAME,
    base_url: str = VLLM_BASE_URL,
    max_tokens: int = GENERATION_MAX_TOKENS,
    timeout: float = 120.0,
) -> str:
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            # vLLM's OpenAI-compatible cap on generated tokens -- top-level
            # field, unlike Ollama's nested options.num_predict.
            "max_tokens": max_tokens,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    choice = response.json()["choices"][0]
    content = choice["message"]["content"]
    # finish_reason "length" means max_tokens cut generation off mid-thought
    # (vs. "stop", a natural end) -- only trim in that case, never touch a
    # clean stop even if it happens not to end in recognized punctuation.
    if choice.get("finish_reason") == "length":
        content = trim_to_last_complete_sentence(content)
    return content


def generate_stream(
    system_prompt: str,
    user_prompt: str,
    *,
    model_name: str = GENERATION_MODEL_NAME,
    base_url: str = VLLM_BASE_URL,
    max_tokens: int = GENERATION_MAX_TOKENS,
    timeout: float = 120.0,
    result_info: dict | None = None,
):
    """Yields text deltas as vLLM generates them (OpenAI-compatible SSE --
    `data: {...}` lines, final `data: [DONE]` sentinel). Unlike generate(),
    does NOT apply the truncation trim to what's yielded -- tokens already
    streamed to a live client can't be un-shown. Instead, once the stream
    ends, records finish_reason into `result_info` (if given) so the caller
    can trim the *saved* (not displayed) version of the accumulated text --
    see rag/service.py's finalize_turn().
    """
    response = requests.post(
        f"{base_url}/v1/chat/completions",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "max_tokens": max_tokens,
            "stream": True,
        },
        timeout=timeout,
        stream=True,
    )
    response.raise_for_status()

    for line in response.iter_lines(decode_unicode=True):
        if not line or not line.startswith("data: "):
            continue
        payload = line[len("data: ") :]
        if payload == "[DONE]":
            break
        choice = json.loads(payload)["choices"][0]
        content = choice.get("delta", {}).get("content") or ""
        if content:
            yield content
        finish_reason = choice.get("finish_reason")
        if finish_reason is not None and result_info is not None:
            result_info["finish_reason"] = finish_reason
