"""Thin HTTP client for Ollama's chat API. Extracted from
notebooks/05_generation.ipynb.

Kept as a plain function, not a class -- each call is a single stateless
HTTP request, nothing expensive to load once and reuse (unlike Retriever's
models/index). This is also the seam CLAUDE.md's Ollama -> vLLM production
swap goes through: a vllm_client.py with the same `generate(...) -> str`
signature can stand in for this module without generation/service.py
changing at all.
"""

import requests

from migrantbuddy.config import (
    GENERATION_MAX_TOKENS,
    GENERATION_MODEL_NAME,
    OLLAMA_BASE_URL,
    OLLAMA_KEEP_ALIVE,
)
from migrantbuddy.generation.truncation import trim_to_last_complete_sentence
from migrantbuddy.observability import observe


@observe(as_type="generation")
def generate(
    system_prompt: str,
    user_prompt: str,
    *,
    model_name: str = GENERATION_MODEL_NAME,
    base_url: str = OLLAMA_BASE_URL,
    max_tokens: int = GENERATION_MAX_TOKENS,
    keep_alive: str = OLLAMA_KEEP_ALIVE,
    timeout: float = 120.0,
) -> str:
    response = requests.post(
        f"{base_url}/api/chat",
        json={
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "stream": False,
            # Ollama's cap on generated tokens -- named num_predict, nested
            # under options (not a top-level field like vLLM's max_tokens).
            "options": {"num_predict": max_tokens},
            # Keeps the model loaded in memory for this long after the
            # request, instead of Ollama's 5-minute default -- avoids
            # paying a multi-GB model reload on the next request if it
            # comes in after a short gap.
            "keep_alive": keep_alive,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    data = response.json()
    content = data["message"]["content"]
    # done_reason "length" means num_predict cut generation off mid-thought
    # (vs. "stop", a natural end) -- only trim in that case, never touch a
    # clean stop even if it happens not to end in recognized punctuation.
    if data.get("done_reason") == "length":
        content = trim_to_last_complete_sentence(content)
    return content
