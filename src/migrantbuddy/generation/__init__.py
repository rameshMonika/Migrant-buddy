from migrantbuddy.generation.ollama_client import generate as ollama_generate
from migrantbuddy.generation.prompts import (
    QUERY_REWRITE_SYSTEM_PROMPT,
    SUMMARY_SYSTEM_PROMPT,
    SYSTEM_PROMPT,
    build_prompt,
    build_query_rewrite_prompt,
    build_summary_prompt,
)
from migrantbuddy.generation.service import GenerationResult, generate_answer
from migrantbuddy.generation.vllm_client import generate as vllm_generate

__all__ = [
    "GenerationResult",
    "QUERY_REWRITE_SYSTEM_PROMPT",
    "SUMMARY_SYSTEM_PROMPT",
    "SYSTEM_PROMPT",
    "build_prompt",
    "build_query_rewrite_prompt",
    "build_summary_prompt",
    "generate_answer",
    "ollama_generate",
    "vllm_generate",
]
