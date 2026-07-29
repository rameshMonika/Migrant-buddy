"""Thin HTTP client for vLLM's OpenAI-compatible chat completions API.

Same generate(...) -> str signature as ollama_client.py -- this is the
production swap CLAUDE.md's Generation serving decision calls for
(continuous batching / PagedAttention for throughput). Which one actually
runs is picked by migrantbuddy.config.GENERATION_BACKEND, dispatched in
generation/service.py.

Request/response shape differs from Ollama's /api/chat: vLLM serves the
standard OpenAI /v1/chat/completions schema (choices[0].message.content),
not Ollama's message.content.
"""

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
