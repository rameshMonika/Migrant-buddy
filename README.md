# migrantBuddy

RAG system for Singapore migrant workers — answers employment questions (salary,
working hours, work permits, medical insurance, contact info) grounded in official
MOM documents, in whatever language the question was asked in.

See `CLAUDE.md` for the full architecture and the decisions behind it. This file is
just "how do I run it."

## Prerequisites

- Python 3.12+
- Node.js 18+ (for the frontend)
- [Ollama](https://ollama.com/), running locally, with the generation model pulled:
  ```
  ollama pull aisingapore/Llama-SEA-LION-v3-8B-IT
  ```
- **ffmpeg**, installed and on your `PATH` — required for speech-to-text
  (`faster-whisper` uses it to decode the audio the browser records). Not a pip
  package; install it separately (e.g. `winget install ffmpeg` on Windows, or grab a
  build from [ffmpeg.org](https://ffmpeg.org/download.html)) and confirm with
  `ffmpeg -version`.

## 1. Backend setup

From the project root:

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

Run the test suite (all mocked — no Ollama/Chroma/network needed):

```powershell
pytest -v
```

## 2. Build the data (ingestion → chunking → embedding → indexing)

The FastAPI backend reads `data/processed/chunks.json` and a Chroma collection at
`data/processed/chroma/` — these don't exist until the pipeline has been run once.
That pipeline currently lives in the notebooks (not yet wired into a single `src/`
script), so run these in order and let each one finish:

1. `notebooks/01_ingestion.ipynb` — fetches the MOM pages, writes `data/processed/ingested.json`
2. `notebooks/02_chunking.ipynb` — writes `data/processed/chunks.json`
3. `notebooks/03_embedding.ipynb` — writes `data/processed/embeddings_bge_m3.npy` + `chunk_ids_bge_m3.json`
4. `notebooks/04_retrieval.ipynb` — Step 2 builds the Chroma collection at `data/processed/chroma/`

Once these have run, `data/processed/` has everything the backend needs.

## 3. Run the backend services

Two independent services, run as two separate processes -- the RAG service (chat)
and the Whisper service (speech-to-text) don't depend on each other at runtime, so
either can be started, stopped, or restarted alone:

```powershell
uvicorn migrantbuddy.api.main:app --reload --port 8000       # RAG (chat)
uvicorn migrantbuddy.speech.main:app --reload --port 8002    # Whisper (speech-to-text)
```

(Port 8001 is reserved for vLLM, hence 8002 here.)

Check they're up:

```powershell
curl http://localhost:8000/health
curl http://localhost:8002/health
```

Only need voice input? Just skip the second command -- the chat UI works fine
without the Whisper service running, it just means the 🎤 button won't connect.

## 4. Run the frontend

```powershell
cd frontend
npm install
copy .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`. The chat UI calls the RAG service at
`NEXT_PUBLIC_API_BASE_URL` (set in `.env.local`, defaults to `http://localhost:8000`)
and the Whisper service directly at `NEXT_PUBLIC_WHISPER_WS_URL` (defaults to
`ws://localhost:8002/transcribe/ws`) -- there's no proxying between the two.

## Configuration

For local secrets (Langfuse keys) and any config overrides, copy `.env.example` to
`.env` and fill it in:

```powershell
copy .env.example .env
```

`.env` is loaded automatically (see `src/migrantbuddy/config.py`) and is gitignored —
never commit real keys. In a deployed environment, set real environment variables
directly instead of using `.env` (env vars always take priority over it anyway).

A few settings are overridable this way (everything else — model names, chunk size,
etc. — is a fixed architecture decision, not meant to vary by environment):

| Env var | Default | Purpose |
|---|---|---|
| `MIGRANTBUDDY_OLLAMA_BASE_URL` | `http://localhost:11434` | Where the backend calls Ollama |
| `MIGRANTBUDDY_OLLAMA_KEEP_ALIVE` | `30m` | How long Ollama keeps the model loaded after a request (avoids a multi-GB reload if the next request comes in after Ollama's 5-min default) |
| `MIGRANTBUDDY_VLLM_BASE_URL` | `http://localhost:8001` | Where the backend calls vLLM (if `GENERATION_BACKEND=vllm`) |
| `MIGRANTBUDDY_GENERATION_BACKEND` | `ollama` | `ollama` or `vllm` — which generation server to call |
| `MIGRANTBUDDY_FRONTEND_ORIGIN` | `http://localhost:3000` | Allowed CORS origin for the API |
| `MIGRANTBUDDY_CHECKPOINTER_BACKEND` | `memory` | `memory` or `redis` — where conversation state (message history, running summary) is persisted — see below |
| `MIGRANTBUDDY_REDIS_URL` | `redis://localhost:6379` | Redis connection string (only used when `CHECKPOINTER_BACKEND=redis`) |
| `MIGRANTBUDDY_CONVERSATION_TTL_SECONDS` | `86400` (24h) | How long an idle conversation survives in Redis before expiring (only used when `CHECKPOINTER_BACKEND=redis`) |
| `MIGRANTBUDDY_RETRIEVAL_CACHE_ENABLED` | `false` | Cache retrieval results in Redis, keyed by (query, top_k) — see below |
| `MIGRANTBUDDY_RETRIEVAL_CACHE_TTL_SECONDS` | `604800` (7 days) | How long a cached retrieval result lives (only used when `RETRIEVAL_CACHE_ENABLED=true`) |
| `MIGRANTBUDDY_RATE_LIMIT_ENABLED` | `false` | Rate-limit `/chat` (Redis) — see below |
| `MIGRANTBUDDY_RATE_LIMIT_MAX_REQUESTS` | `10` | Max requests per client IP per window (only used when `RATE_LIMIT_ENABLED=true`) |
| `MIGRANTBUDDY_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window, in seconds (only used when `RATE_LIMIT_ENABLED=true`) |
| `MIGRANTBUDDY_WHISPER_MODEL_SIZE` | `small` | Whisper model size for speech-to-text (`tiny`/`base`/`small`/`medium`/`large-v3`, etc.) — see below |
| `MIGRANTBUDDY_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` — set to `cuda` if you have an NVIDIA GPU |
| `MIGRANTBUDDY_WHISPER_COMPUTE_TYPE` | `int8` | faster-whisper quantization — `int8` is fastest on CPU; use `float16` with `cuda` |
| `MIGRANTBUDDY_WHISPER_RATE_LIMIT_ENABLED` | `false` | Rate-limit the Whisper service's `/transcribe/ws` (Redis) — own toggle/budget, separate from `/chat`'s |
| `MIGRANTBUDDY_WHISPER_RATE_LIMIT_MAX_REQUESTS` | `10` | Max connections per client IP per window (only used when `WHISPER_RATE_LIMIT_ENABLED=true`) |
| `MIGRANTBUDDY_WHISPER_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window, in seconds (only used when `WHISPER_RATE_LIMIT_ENABLED=true`) |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | unset (tracing off) | Enables Langfuse tracing when **all three** are set — see below |

### Conversational memory backend

Defaults to `memory` — LangGraph's built-in in-memory checkpointer, no external service,
but conversations are lost on every server restart. Setting `MIGRANTBUDDY_CHECKPOINTER_BACKEND=redis`
persists conversations across restarts, keyed by `thread_id`, with a TTL so idle
conversations expire instead of growing Redis forever.

Redis itself needs to be running somewhere first. Official Redis doesn't support Windows
well; on Windows, [Memurai](https://www.memurai.com/) (free Developer Edition) is a
Redis-API-compatible server that installs as a normal Windows service — no Docker, no
WSL2. It listens on `localhost:6379` by default, matching `MIGRANTBUDDY_REDIS_URL`'s
default.

### Retrieval caching

Off by default, separate toggle from the checkpointer (you may want one without the
other). Setting `MIGRANTBUDDY_RETRIEVAL_CACHE_ENABLED=true` caches retrieval results in
Redis keyed by (query, top_k) — safe with a long TTL since the corpus only changes on a
deliberate re-ingestion, unlike conversation state. A Redis error or cache miss always
falls back to computing retrieval fresh, and a stale cache entry (e.g. a chunk_id from
before a re-ingestion) is detected and discarded rather than crashing — this cache is
purely a latency optimization, never a correctness requirement.

### Rate limiting

Off by default. Setting `MIGRANTBUDDY_RATE_LIMIT_ENABLED=true` limits `/chat` (only —
`/health` is never limited) to `RATE_LIMIT_MAX_REQUESTS` requests per client IP per
`RATE_LIMIT_WINDOW_SECONDS` (Redis, fixed-window `INCR`+`EXPIRE`), returning `429` once
exceeded. This exists because Ollama/vLLM can't meaningfully serve concurrent requests
(generation is CPU/GPU-bound) — an unbounded burst would otherwise just queue up and
time out ugly instead of failing cleanly. Like the retrieval cache, this fails open: a
Redis error allows the request through rather than locking everyone out.

### Speech-to-text

Runs as its own service (`migrantbuddy.speech.main`, port 8002 by default), independent
of the RAG service — see "Run the backend services" above. The 🎤 button next to the
chat input opens a WebSocket directly to it (`/transcribe/ws`) and streams audio live as
you speak (`faster-whisper`, multilingual — no language is pinned, so it auto-detects
Burmese/Tamil/Thai/Vietnamese/etc.); transcribed text appears in the input box
incrementally as each spoken segment is confirmed, for you to review and edit rather
than auto-sent — transcription errors are common, and this app answers
employment/legal-rights questions, where getting the question right matters.
`MIGRANTBUDDY_WHISPER_MODEL_SIZE=small` by default, a balance of multilingual accuracy
against CPU-only inference speed; bump it up if you have a GPU
(`MIGRANTBUDDY_WHISPER_DEVICE=cuda`) or down if `small` is too slow.

### Observability (Langfuse)

Tracing is opt-in and off by default. Setting `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`,
and `LANGFUSE_HOST` (self-hosted or [Langfuse Cloud](https://cloud.langfuse.com)) turns on
per-stage tracing — each retrieval stage (`dense`, `bm25_search`, `hybrid`, `rerank`) and
each conversation node (`summarize`, `rewrite_query`, `retrieve`, `generate`) show up as
timed, nested spans under a top-level trace for `ConversationService.answer()`, viewable
in the Langfuse UI. Because this app can handle sensitive queries (e.g. workplace
complaints), think about whether self-hosting Langfuse makes more sense than sending
query text to a third-party cloud instance.

## Project status

Research/prototyping phase. Core pipeline choices (BGE-M3 embedding, hybrid
BM25+dense retrieval, SEA-LION-E5 reranker, SEA-LION 8B generation) are settled and
extracted into `src/migrantbuddy/`; Docker/CI/CD/deploy haven't been set up yet.
