"""Coordinates prompt construction + the generation backend call into a full
answer. Extracted from notebooks/05_generation.ipynb's generate() function.

Dispatches between ollama_client and vllm_client based on `backend`
(defaults to migrantbuddy.config.GENERATION_BACKEND). Deliberately an
if/elif on the imported client functions rather than a dict built at import
time -- a dict would capture a snapshot of the function references at
import time, which would break tests that monkeypatch
`migrantbuddy.generation.service.ollama_generate`/`vllm_generate` after the
module has already loaded.
"""

from dataclasses import dataclass
from typing import Sequence

from migrantbuddy.config import GENERATION_BACKEND, GENERATION_MAX_TOKENS, GENERATION_MODEL_NAME
from migrantbuddy.generation.ollama_client import generate as ollama_generate
from migrantbuddy.generation.ollama_client import generate_stream as ollama_generate_stream
from migrantbuddy.generation.prompts import SYSTEM_PROMPT, build_prompt
from migrantbuddy.generation.vllm_client import generate as vllm_generate
from migrantbuddy.generation.vllm_client import generate_stream as vllm_generate_stream
from migrantbuddy.indexing import Chunk
from migrantbuddy.observability import observe


@dataclass
class GenerationResult:
    query: str
    answer: str
    sources: list[str]


@observe()
def generate_answer(
    query: str,
    context_chunks: Sequence[Chunk],
    *,
    model_name: str = GENERATION_MODEL_NAME,
    backend: str = GENERATION_BACKEND,
    max_tokens: int = GENERATION_MAX_TOKENS,
) -> GenerationResult:
    prompt = build_prompt(query, context_chunks)

    if backend == "ollama":
        answer = ollama_generate(
            SYSTEM_PROMPT, prompt, model_name=model_name, max_tokens=max_tokens
        )
    elif backend == "vllm":
        answer = vllm_generate(SYSTEM_PROMPT, prompt, model_name=model_name, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown generation backend: {backend!r} (expected 'ollama' or 'vllm')")

    return GenerationResult(
        query=query,
        answer=answer,
        sources=[chunk.url for chunk in context_chunks],
    )


def stream_answer(
    system_prompt: str,
    user_prompt: str,
    *,
    model_name: str = GENERATION_MODEL_NAME,
    backend: str = GENERATION_BACKEND,
    max_tokens: int = GENERATION_MAX_TOKENS,
    result_info: dict | None = None,
):
    """Streaming counterpart to generate_answer() -- yields text deltas
    instead of returning a complete string, for /chat's token-by-token
    response. Takes an already-built prompt (not query/context_chunks),
    since the caller (rag/service.py's prepare_turn) already has it.

    Normalizes each backend's own truncation signal (Ollama's done_reason,
    vLLM's finish_reason) into a single result_info["truncated"] bool, so
    callers don't need to know which backend produced the answer.
    """
    raw_info: dict = {}

    if backend == "ollama":
        yield from ollama_generate_stream(
            system_prompt,
            user_prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            result_info=raw_info,
        )
        truncated = raw_info.get("done_reason") == "length"
    elif backend == "vllm":
        yield from vllm_generate_stream(
            system_prompt,
            user_prompt,
            model_name=model_name,
            max_tokens=max_tokens,
            result_info=raw_info,
        )
        truncated = raw_info.get("finish_reason") == "length"
    else:
        raise ValueError(f"Unknown generation backend: {backend!r} (expected 'ollama' or 'vllm')")

    if result_info is not None:
        result_info["truncated"] = truncated
