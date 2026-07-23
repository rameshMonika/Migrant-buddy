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
from migrantbuddy.generation.prompts import SYSTEM_PROMPT, build_prompt
from migrantbuddy.generation.vllm_client import generate as vllm_generate
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
        answer = ollama_generate(SYSTEM_PROMPT, prompt, model_name=model_name, max_tokens=max_tokens)
    elif backend == "vllm":
        answer = vllm_generate(SYSTEM_PROMPT, prompt, model_name=model_name, max_tokens=max_tokens)
    else:
        raise ValueError(f"Unknown generation backend: {backend!r} (expected 'ollama' or 'vllm')")

    return GenerationResult(
        query=query,
        answer=answer,
        sources=[chunk.url for chunk in context_chunks],
    )
