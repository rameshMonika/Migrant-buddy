"""Paths, model names, and settings shared across migrantbuddy modules.

Single source of truth for the settled architecture decisions in CLAUDE.md
(embedding, reranker, generation model) -- previously copy-pasted as
per-notebook constants across 05_generation.ipynb, 06_eval.ipynb,
05b_generation_sealion.ipynb, 06b_eval_sealion_generation.ipynb.
"""

import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Loads secrets/overrides (e.g. LANGFUSE_*) from a gitignored .env file at
# the project root for local dev. A no-op if that file doesn't exist --
# safe in a deployed environment, where env vars are set directly by the
# platform instead. Existing environment variables always take priority
# over .env (load_dotenv's default), so a real deployment's env vars are
# never accidentally shadowed by a stray .env file.
load_dotenv(PROJECT_ROOT / ".env")

DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = PROCESSED_DIR / "chroma"

# Settled per CLAUDE.md Architecture decisions.
EMBEDDING_MODEL_NAME = "BAAI/bge-m3"
RERANKER_MODEL_NAME = "aisingapore/SEA-LION-E5-Embedding-600M"
GENERATION_MODEL_NAME = "aisingapore/Llama-SEA-LION-v3-8B-IT"
CHROMA_COLLECTION_NAME = "bge_m3"

# Structure-aware chunking defaults -- 500 tokens, ~20% overlap (see
# notebooks/02_chunking.ipynb).
CHUNK_SIZE = 500
CHUNK_OVERLAP = 100

# Caps the generation backend's output length -- without this, a model that
# starts rambling or repeating (a known failure mode for smaller quantized
# models) has no ceiling and keeps generating until it hits a natural stop
# or the model's max context, which can take a long time. 512 is generous
# for a multi-paragraph answer with a few bullet points (the typical shape
# of answers seen in notebooks/05_generation.ipynb) while still bounding
# the worst case. Does not help "time to first token" (prompt processing) --
# only bounds how long the output itself can run.
GENERATION_MAX_TOKENS = 512

# Conversational memory (rag/) -- once a conversation exceeds this many
# messages, older ones get condensed into a running summary instead of
# being kept verbatim in every prompt, keeping prompt size (and therefore
# prefill latency) roughly bounded regardless of conversation length.
# MESSAGES_KEPT_VERBATIM stay untouched through summarization -- recent
# exchanges are the ones a follow-up question is most likely to depend on.
SUMMARY_TRIGGER_MESSAGE_COUNT = 12
MESSAGES_KEPT_VERBATIM = 6

# Separate, tighter output caps for the two short, structured generation
# calls in rag/nodes.py -- a running summary and a rewritten standalone
# query are each meant to be a few sentences or a single question, not a
# full answer. Without their own cap they'd default to
# GENERATION_MAX_TOKENS (512, sized for a complete multi-paragraph answer),
# giving the model far more room than either output needs and no real
# pressure to stay concise.
SUMMARY_MAX_TOKENS = 150
QUERY_REWRITE_MAX_TOKENS = 100

# Overridable via env var since this differs between local dev and a
# deployed environment; everything else above is a fixed architecture
# decision, not an environment-specific setting.
OLLAMA_BASE_URL = os.getenv("MIGRANTBUDDY_OLLAMA_BASE_URL", "http://localhost:11434")
VLLM_BASE_URL = os.getenv("MIGRANTBUDDY_VLLM_BASE_URL", "http://localhost:8001")

# How long Ollama keeps the model loaded in memory after a request before
# unloading it. Ollama's own default is 5 minutes -- too short for
# request-to-request gaps of a few minutes, which forces a full model
# reload from disk (multiple GB) on the next request, on top of actual
# generation time. "30m" trades idle memory usage for avoiding that reload
# cost. Only meaningful for Ollama -- vLLM keeps its model loaded for the
# lifetime of the server process, no equivalent setting needed there.
OLLAMA_KEEP_ALIVE = os.getenv("MIGRANTBUDDY_OLLAMA_KEEP_ALIVE", "30m")

# "ollama" for local dev (fast iteration, no GPU box needed) or "vllm" for
# production serving (continuous batching / PagedAttention for throughput)
# -- see CLAUDE.md's Generation serving decision. Defaults to "ollama" since
# that's still what local dev uses; set this to "vllm" in a deployed
# environment once vLLM is actually running there.
GENERATION_BACKEND = os.getenv("MIGRANTBUDDY_GENERATION_BACKEND", "ollama")

# The frontend's origin -- needed for CORS since the Next.js dev server
# (localhost:3000) and this API (localhost:8000) are different origins.
FRONTEND_ORIGIN = os.getenv("MIGRANTBUDDY_FRONTEND_ORIGIN", "http://localhost:3001")

# "memory" (LangGraph's built-in in-memory checkpointer -- no external
# service, conversations lost on server restart) or "redis" (persists
# across restarts, thread_id-keyed, with a TTL so old conversations expire
# instead of growing Redis unboundedly). Defaults to "memory" so nothing
# breaks for anyone without Redis running; see CLAUDE.md's Conversational
# memory decision for the two-stage rollout this implements.
CHECKPOINTER_BACKEND = os.getenv("MIGRANTBUDDY_CHECKPOINTER_BACKEND", "memory")
REDIS_URL = os.getenv("MIGRANTBUDDY_REDIS_URL", "redis://localhost:6379")
CONVERSATION_TTL_SECONDS = int(os.getenv("MIGRANTBUDDY_CONVERSATION_TTL_SECONDS", str(24 * 60 * 60)))

# Caches retrieval results (Redis) keyed by (query, top_k) -- unlike
# conversation state, retrieval results aren't time-sensitive: the corpus
# only changes on a deliberate re-ingestion, so a long TTL is safe. Off by
# default (separate toggle from CHECKPOINTER_BACKEND -- you may want one
# without the other). A cache miss or Redis being unavailable just means
# "compute it, like before" -- this is a latency optimization only, never
# allowed to break retrieval.
RETRIEVAL_CACHE_ENABLED = os.getenv("MIGRANTBUDDY_RETRIEVAL_CACHE_ENABLED", "false").lower() == "true"
RETRIEVAL_CACHE_TTL_SECONDS = int(os.getenv("MIGRANTBUDDY_RETRIEVAL_CACHE_TTL_SECONDS", str(7 * 24 * 60 * 60)))

# Rate limiting for /chat (Redis, fixed-window INCR+EXPIRE) -- protects the
# single Ollama/vLLM backend from being overwhelmed: it can't meaningfully
# serve concurrent requests (CPU-bound generation), so an unbounded burst
# would otherwise just queue up and time out ugly instead of being
# rejected cleanly (429). Off by default -- separate toggle from the
# checkpointer/cache, since you may want any subset of the three. Fails
# open (allows the request through) if Redis is unreachable -- like the
# retrieval cache, this is a protective measure, not a correctness
# requirement, and Redis being optional infra should never lock out real
# users. 10 requests / 60s per client IP is generous for normal chat use
# (each request already takes 10s+ end to end) while still stopping a
# runaway loop or abuse from hammering Ollama.
RATE_LIMIT_ENABLED = os.getenv("MIGRANTBUDDY_RATE_LIMIT_ENABLED", "false").lower() == "true"
RATE_LIMIT_MAX_REQUESTS = int(os.getenv("MIGRANTBUDDY_RATE_LIMIT_MAX_REQUESTS", "10"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("MIGRANTBUDDY_RATE_LIMIT_WINDOW_SECONDS", "60"))

# Rate limiting for the standalone Whisper service's /transcribe/ws -- same
# fixed-window Redis INCR+EXPIRE mechanism as RATE_LIMIT_* above, but its own
# toggle/budget since transcription (CPU-bound Whisper inference) and chat
# generation (Ollama/vLLM) are separate services with separate capacity, not
# one shared budget.
WHISPER_RATE_LIMIT_ENABLED = os.getenv("MIGRANTBUDDY_WHISPER_RATE_LIMIT_ENABLED", "false").lower() == "true"
WHISPER_RATE_LIMIT_MAX_REQUESTS = int(os.getenv("MIGRANTBUDDY_WHISPER_RATE_LIMIT_MAX_REQUESTS", "10"))
WHISPER_RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("MIGRANTBUDDY_WHISPER_RATE_LIMIT_WINDOW_SECONDS", "60"))

# Speech-to-text (faster-whisper) -- "small" balances multilingual accuracy
# (needs to handle Burmese/Tamil/Thai/Vietnamese etc., where a smaller
# model like "tiny"/"base" degrades noticeably more) against CPU-only
# inference speed, since GPU availability on the target machine isn't
# confirmed. Requires ffmpeg on PATH (used by faster-whisper's audio
# decoding) -- not a pip package, a separate system install.
WHISPER_MODEL_SIZE = os.getenv("MIGRANTBUDDY_WHISPER_MODEL_SIZE", "small")
WHISPER_DEVICE = os.getenv("MIGRANTBUDDY_WHISPER_DEVICE", "cpu")
# int8 quantization -- meaningfully faster on CPU than float32 with only a
# small accuracy cost; irrelevant if WHISPER_DEVICE=cuda later (GPU
# inference would typically use float16 instead).
WHISPER_COMPUTE_TYPE = os.getenv("MIGRANTBUDDY_WHISPER_COMPUTE_TYPE", "int8")

# Langfuse (observability/tracing) -- deliberately no default keys/host
# here. Tracing only activates when both keys are set (see
# migrantbuddy.observability); this app can handle sensitive queries (e.g.
# workplace complaints), so sending trace data to a service -- even a
# self-hosted one -- should be an explicit opt-in, not silent by default.
LANGFUSE_PUBLIC_KEY = os.getenv("LANGFUSE_PUBLIC_KEY", "")
LANGFUSE_SECRET_KEY = os.getenv("LANGFUSE_SECRET_KEY", "")
LANGFUSE_HOST = os.getenv("LANGFUSE_HOST", "")
LANGFUSE_ENABLED = bool(LANGFUSE_PUBLIC_KEY and LANGFUSE_SECRET_KEY)
