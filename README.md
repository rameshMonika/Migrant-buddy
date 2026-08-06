# migrantBuddy

RAG system for Singapore migrant workers, answers employment questions (salary,
working hours, work permits, medical insurance, contact info) grounded in official
MOM documents, in whatever language the question was asked in.

## Contents

- [What it does](#what-it-does)
- [Tech stack](#tech-stack)
- [Architecture](#architecture)
- [Prerequisites](#prerequisites)
  - [Configuration](#configuration)
- [1. Backend setup](#1-backend-setup)
- [2. Build the data (ingestion → chunking → embedding → indexing)](#2-build-the-data)
- [3. Run the app](#3-run-the-app)
  - [Option A: Local](#option-a-local)
  - [Option B: Docker](#option-b-docker)
- [How the RAG pipeline works](#how-the-rag-pipeline-works)
  - [Ingestion, fetch, extract, validate](#ingestion--fetch-extract-validate)
  - [Chunking, structure-aware, five deterministic passes](#chunking--structure-aware-five-deterministic-passes)
  - [Embedding, BGE-M3](#embedding--bge-m3-settled)
  - [Retrieval, hybrid BM25 + dense, then rerank](#retrieval--hybrid-bm25--dense-then-rerank-settled)
  - [Generation, SEA-LION 8B](#generation--sea-lion-8b-settled)
  - [Conversation memory, LangGraph](#conversation-memory--langgraph)
- [Evaluation results](#evaluation-results)
  - [Reranker comparison](#reranker-comparison)
  - [Generation model comparison](#generation-model-comparison)
- [Architectural decisions](#architectural-decisions)
  - [Conversational memory backend](#conversational-memory-backend)
  - [Retrieval caching](#retrieval-caching)
  - [Rate limiting](#rate-limiting)
  - [Speech-to-text](#speech-to-text)
  - [Text-to-speech + avatar](#text-to-speech--avatar)
  - [Streaming answers](#streaming-answers)
  - [Observability (Langfuse)](#observability-langfuse)

<a id="what-it-does"></a>
## What it does

A migrant worker asks a question, by typing, or by speaking into the browser, in
English, Tamil, Burmese, Thai, Vietnamese, Malay, Filipino, or Indonesian. The system
retrieves the relevant passage from official MOM (Ministry of Manpower) guidance and
answers **in the language the question was asked in**, with an optional
talking-avatar readback for workers who find spoken answers easier to trust or
follow than text.

The corpus is intentionally narrow and real, not a broad scrape: five MOM source
pages across salary, working hours, work-permit conditions, medical insurance, and
help/contact info. Three other candidate categories (employment rights, work injury/
WICA, housing) were evaluated and dropped; their candidate URLs turned out to be
navigation-only landing pages with no substantive content, confirmed by fetching them
directly rather than assumed.

The core design bet: a multilingual embedding model (BGE-M3) retrieves directly
against non-English queries with no translation step, and a SEA-LION-tuned generation
model answers natively in that language off English-source context, so translation
never becomes a separate pipeline stage on either side of the request. See
[How the RAG pipeline works](#how-the-rag-pipeline-works) below for how that's proven
out, not just asserted.

<a id="tech-stack"></a>
## Tech stack

| Layer | Technology | Used for |
|---|---|---|
| Frontend | Next.js (TypeScript) | Chat UI, mic capture, streamed-answer rendering, hosts the LiveAvatar WebRTC session |
| API | FastAPI (3 services: `rag`, `speech`, `tts`) | REST + SSE + WebSocket endpoints per service |
| Conversation orchestration | LangGraph | Multi-turn state graph (summarize → rewrite query → retrieve → generate), checkpointed per `thread_id` |
| Embedding | BGE-M3 (`sentence-transformers`) | Multilingual dense embeddings — retrieves non-English queries directly against the English corpus |
| Lexical retrieval | `rank_bm25` | Sparse/lexical half of hybrid retrieval |
| Vector store | Chroma | Persistent index of BGE-M3 embeddings, embedded in-process (no separate service) |
| Reranker | SEA-LION-E5-Embedding-600M | Reorders hybrid retrieval's candidates — settled winner of a 6-way comparison, see [Evaluation results](#evaluation-results) |
| Generation (dev) | Ollama | Local, fast-iteration generation serving |
| Generation (prod) | vLLM | Continuous-batching/PagedAttention generation serving, 4-bit bitsandbytes-quantized for small-VRAM GPUs |
| Generation model | `aisingapore/Llama-SEA-LION-v3-8B-IT` | Answers natively in the query's language — settled winner over `qwen3:8b`, see [Evaluation results](#evaluation-results) |
| Ingestion | `trafilatura` | HTML → markdown extraction (tables kept inline, boilerplate stripped) |
| Speech-to-text | `faster-whisper` | Streaming, multilingual, auto-detects language — no manual language toggle |
| Text-to-speech | ElevenLabs | Synthesizes the spoken answer |
| Avatar | LiveAvatar (HeyGen ecosystem), WebRTC | Lip-synced talking-avatar readback; browser connects to HeyGen directly after a server-minted session token |
| Cache / state | Redis (Memurai on Windows) | LangGraph checkpointer, retrieval cache, rate limiting |
| Evaluation | Ragas | Retrieval metrics (MRR/precision/recall/nDCG) + generation metrics (faithfulness, answer relevancy), judged by `llama3.1:8b` |
| Observability | Langfuse | Per-stage tracing across retrieval and the conversation graph (opt-in) |
| Orchestration | Docker Compose | 8-container stack, alternative to running services as local processes — see [Option B: Docker](#option-b-docker) |

<a id="architecture"></a>
## Architecture

Four services the team owns, plus backing stores/model runtimes, plus two
third-party APIs; every internal hop is plain HTTP/WS on the Docker network's
service-name DNS, and every browser-facing hop goes through a published host port.
The same four-service topology runs two ways; what actually changes between them is
the generation backend and where conversation/cache state lives, shown below.

![migrantBuddy system architecture, local dev vs Docker Compose, with icons](architecture.svg)

HTTP (Hypertext Transfer Protocol) carries request/response calls between services;
WS (WebSocket) carries the persistent, bidirectional connections used for streaming
(e.g. speech-to-text audio, token-by-token chat responses). Redis backs the retrieval
cache and rate limiting in the Docker profile today; the LangGraph checkpointer itself
still defaults to in-memory even there (`MIGRANTBUDDY_CHECKPOINTER_BACKEND=redis` is
supported, just not yet the compose default), see
[Conversational memory backend](#conversational-memory-backend).

**Why four separate services instead of one monolith:**
- `rag` is latency-critical (streams tokens) and CPU/GPU-bound on embedding +
  generation, kept lean, no audio dependencies weighing down its image or
  cold-start.
- `speech` pulls in `faster-whisper` and a large model cache (its own named volume)
  independent of RAG's lifecycle, restarted/scaled without touching chat.
- `tts` is two thin outbound HTTP clients (ElevenLabs, LiveAvatar) with no local
  model, stays a lightweight, fast-building container.
- Each ships as its own Docker build `target` from one shared `Dockerfile`, with its
  own pushable image, for independent deploys.

<a id="prerequisites"></a>
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

<a id="configuration"></a>
### Configuration

For local secrets (Langfuse keys) and any config overrides, copy `.env.example` to
`.env` and fill it in:

```powershell
copy .env.example .env
```

`.env` is loaded automatically (see `src/migrantbuddy/config.py`) and is gitignored —
never commit real keys. In a deployed environment, set real environment variables
directly instead of using `.env` (env vars always take priority over it anyway).

Two of these are required just to run the app at all, since the TTS service fails on
first use without them; the rest are optional. Where to get each value:

- **`ELEVENLABS_API_KEY` / `ELEVENLABS_VOICE_ID`** (required): create an
  [ElevenLabs](https://elevenlabs.io) account, copy the API key from your profile, and
  pick a voice ID from the Voices tab.
- **`LIVEAVATAR_API_KEY` / `LIVEAVATAR_AVATAR_ID`** (required): create a
  [LiveAvatar](https://docs.liveavatar.com) account, copy the API key and the
  `avatar_id` of whichever avatar you've picked/created from your dashboard.
- **`LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST`** (optional,
  tracing is off unless all three are set): from a [Langfuse](https://langfuse.com)
  project, under Project Settings → API Keys.

Everything else in `.env.example` is commented out by default and only needs
uncommenting if you want to override a default (see the full table below).

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
| `MIGRANTBUDDY_RETRIEVAL_CACHE_ENABLED` | `true` | Cache retrieval results in Redis, keyed by (query, top_k) — see below |
| `MIGRANTBUDDY_RETRIEVAL_CACHE_TTL_SECONDS` | `604800` (7 days) | How long a cached retrieval result lives (only used when `RETRIEVAL_CACHE_ENABLED=true`) |
| `MIGRANTBUDDY_RATE_LIMIT_ENABLED` | `true` | Rate-limit `/chat` (Redis) — see below |
| `MIGRANTBUDDY_RATE_LIMIT_MAX_REQUESTS` | `10` | Max requests per client IP per window (only used when `RATE_LIMIT_ENABLED=true`) |
| `MIGRANTBUDDY_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window, in seconds (only used when `RATE_LIMIT_ENABLED=true`) |
| `MIGRANTBUDDY_WHISPER_MODEL_SIZE` | `small` | Whisper model size for speech-to-text (`tiny`/`base`/`small`/`medium`/`large-v3`, etc.) — see below |
| `MIGRANTBUDDY_WHISPER_DEVICE` | `cpu` | `cpu` or `cuda` — set to `cuda` if you have an NVIDIA GPU |
| `MIGRANTBUDDY_WHISPER_COMPUTE_TYPE` | `int8` | faster-whisper quantization — `int8` is fastest on CPU; use `float16` with `cuda` |
| `MIGRANTBUDDY_WHISPER_RATE_LIMIT_ENABLED` | `false` | Rate-limit the Whisper service's `/transcribe/ws` (Redis) — own toggle/budget, separate from `/chat`'s |
| `MIGRANTBUDDY_WHISPER_RATE_LIMIT_MAX_REQUESTS` | `10` | Max connections per client IP per window (only used when `WHISPER_RATE_LIMIT_ENABLED=true`) |
| `MIGRANTBUDDY_WHISPER_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window, in seconds (only used when `WHISPER_RATE_LIMIT_ENABLED=true`) |
| `ELEVENLABS_API_KEY` | unset (required) | Your ElevenLabs API key — the TTS service won't work without it |
| `ELEVENLABS_VOICE_ID` | unset (required) | Which ElevenLabs voice to speak with — see below |
| `MIGRANTBUDDY_ELEVENLABS_MODEL_ID` | `eleven_multilingual_v2` | ElevenLabs model — multilingual to match this project's target languages |
| `MIGRANTBUDDY_TTS_RATE_LIMIT_ENABLED` | `true` | Rate-limit the TTS service's `/session/start` and `/speak` (Redis) — own toggle/budget, separate from `/chat`'s and Whisper's (ElevenLabs and LiveAvatar are both metered) |
| `MIGRANTBUDDY_TTS_RATE_LIMIT_MAX_REQUESTS` | `10` | Max requests per client IP per window (only used when `TTS_RATE_LIMIT_ENABLED=true`) |
| `MIGRANTBUDDY_TTS_RATE_LIMIT_WINDOW_SECONDS` | `60` | Rate limit window, in seconds (only used when `TTS_RATE_LIMIT_ENABLED=true`) |
| `LIVEAVATAR_API_KEY` | unset (required) | Your LiveAvatar API key — `/session/start` won't work without it |
| `LIVEAVATAR_AVATAR_ID` | unset (required) | Which LiveAvatar avatar to render — from your LiveAvatar dashboard |
| `MIGRANTBUDDY_LIVEAVATAR_API_URL` | `https://api.liveavatar.com` | LiveAvatar API base URL |
| `MIGRANTBUDDY_LIVEAVATAR_IS_SANDBOX` | `true` | Sandbox mode — test without consuming LiveAvatar credits; turn off for a real end-to-end check |
| `LANGFUSE_PUBLIC_KEY` / `LANGFUSE_SECRET_KEY` / `LANGFUSE_HOST` | unset (tracing off) | Enables Langfuse tracing when **all three** are set — see [Architectural decisions](#architectural-decisions) below |

<a id="1-backend-setup"></a>
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

<a id="2-build-the-data"></a>
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

<a id="3-run-the-app"></a>
## 3. Run the app

**Step 2 ("Build the data") is a prerequisite either way** — neither option below runs
the ingestion/chunking/embedding notebook pipeline for you.

<a id="option-a-local"></a>
### Option A: Local

Three independent backend services, run as three separate processes, the RAG service
(chat), the Whisper service (speech-to-text), and the TTS service (voice output)
don't depend on each other at runtime, so any of them can be started, stopped, or
restarted alone:

```powershell
uvicorn migrantbuddy.api.main:app --reload --port 8000       # RAG (chat)
uvicorn migrantbuddy.speech.main:app --reload --port 8002    # Whisper (speech-to-text)
uvicorn migrantbuddy.tts.main:app --reload --port 8003       # TTS (voice output)
```

(Port 8001 is reserved for vLLM, hence 8002/8003 here.)

Check they're up:

```powershell
curl http://localhost:8000/health
curl http://localhost:8002/health
curl http://localhost:8003/health
```

Only need some of these? The chat UI works fine without the Whisper/TTS services
running, it just means the 🎤 button won't connect and answers won't be spoken.

Then run the frontend:

```powershell
cd frontend
npm install
copy .env.local.example .env.local
npm run dev
```

Open `http://localhost:3000`. The chat UI calls the RAG service at
`NEXT_PUBLIC_API_BASE_URL` (set in `.env.local`, defaults to `http://localhost:8000`),
the Whisper service directly at `NEXT_PUBLIC_WHISPER_WS_URL` (defaults to
`ws://localhost:8002/transcribe/ws`), and the TTS service directly at
`NEXT_PUBLIC_TTS_API_BASE_URL` (defaults to `http://localhost:8003`), there's no
proxying between any of them.

<a id="option-b-docker"></a>
### Option B: Docker

Runs the RAG service, Whisper service, TTS service, frontend, Redis, Ollama, and vLLM
as seven containers via one `docker-compose.yml`, an alternative to backend setup and
Option A above, no local Python/Node install needed. Docker still just bind-mounts
whatever `data/processed/` the notebooks already produced on your machine (read-write,
not read-only, Chroma writes its own SQLite WAL/lock files even when only querying).

Prerequisites:
- Docker Desktop
- For vLLM specifically: an NVIDIA GPU with Docker Desktop's WSL2 GPU passthrough
  configured (NVIDIA Container Toolkit). Without it, the `vllm` service will fail to
  start, either set that up first, or remove the `deploy:` block from `vllm` in
  `docker-compose.yml` and set `MIGRANTBUDDY_GENERATION_BACKEND=ollama` under `rag`'s
  `environment:` to fall back to Ollama (CPU-friendly, no GPU needed).
- `.env` at the project root (`copy .env.example .env`), same file the local setup
  uses, for Langfuse keys etc.

```powershell
docker compose build
docker compose up
```

One-time step, once `ollama` is up (pulls the model into the `ollama-data` volume,
persisted across restarts):

```powershell
docker compose exec ollama ollama pull aisingapore/Llama-SEA-LION-v3-8B-IT
```

Check everything's up:

```powershell
curl http://localhost:8010/health   # RAG (8000 is remapped to 8010)
curl http://localhost:8002/health   # Whisper
curl http://localhost:8003/health   # TTS
curl http://localhost:8001/v1/models  # vLLM (once the model's finished loading)
```

Open `http://localhost:3001` (also remapped). The `NEXT_PUBLIC_*` URLs are
baked in as Docker build args instead of read from `.env.local` (see
`frontend/Dockerfile`), so they don't need to match your local dev setup.

`rag` defaults to `MIGRANTBUDDY_GENERATION_BACKEND=vllm` in `docker-compose.yml` (the
whole reason vLLM's here, it was previously blocked by a Windows-only Long Path error
installing natively), switch back to `ollama` any time by editing that one env var in
`docker-compose.yml`, no rebuild needed, just `docker compose up -d rag`.

<a id="how-the-rag-pipeline-works"></a>
## How the RAG pipeline works

One request flows through offline ingestion (notebook-driven, run once per corpus
change) and five online stages per turn: summarize → rewrite query → retrieve →
rerank → generate.

![migrantBuddy RAG pipeline, offline ingestion and online per-turn stages, with icons](ragPipeline.svg)

<a id="ingestion--fetch-extract-validate"></a>
### Ingestion, fetch, extract, validate

Five MOM source URLs, each hand-tagged with a `category` (salary, working-hours,
work-permit, medical, help), deliberately not a corpus-wide crawl:

1. **Fetch & cache raw HTML**: plain `requests.get`, then the untouched HTML is
   written to disk *before* any parsing, so extraction can be re-run and debugged
   offline without re-hitting mom.gov.sg.
2. **Extract to markdown**: `trafilatura.extract(..., include_tables=True,
   favor_precision=False)`. `favor_precision=False` is deliberate: trafilatura's
   precision mode trims aggressively and risks cutting real guidance text along with
   boilerplate; recall matters more here.
3. **Assemble the record**: stamps a stable `document_id` (slugified URL),
   `authority="MOM"`, the per-source `category`, and `content_type=
   "official_guidance"`. The corpus itself is monolingual English; multilinguality
   lives entirely on the query side, not in the source documents.
4. **Validate before trusting it**: hard-fails on a missing title, empty text, or
   text under 500 characters (catches a page reduced to near-nothing after
   boilerplate stripping). This is exactly how the WICA and housing pages got caught
   and dropped.

<a id="chunking--structure-aware-five-deterministic-passes"></a>
### Chunking, structure-aware, five deterministic passes

Pure text transformation, no I/O or model loading. Token counting is a heuristic
(`len(text) // 4`), not a real tokenizer, accurate enough to budget chunk sizes
without a tokenizer dependency. Target: 300–500 tokens, 15–20% overlap, tuned
against a naive fixed-window baseline across 300/500/800-token and 10%/20%-overlap
variants.

1. **Split on headings**: regex-matches `##`/`###` lines and walks them with a
   level-aware stack, so each section keeps a full breadcrumb (e.g. *"Overtime pay >
   How overtime pay is calculated"*).
2. **Merge undersized sections**: heading-splitting alone leaves plenty of sections
   smaller than the 300-token floor; adjacent sections are merged forward greedily
   until the next one would exceed `chunk_size`.
3. **Protect tables, then split oversized sections**: any section still over budget
   is recursively split at paragraph boundaries with overlap. Every markdown table
   is swapped for a placeholder before splitting, so the splitter can never land
   inside one; tables stay whole even if that pushes a chunk over the nominal size.
4. **Prefix a breadcrumb**: every chunk is prefixed with `"{document title} >
   {heading path}"` before embedding, so a chunk retrieved on its own still carries
   enough context to be understood.
5. **Validate the invariant**: counts table blocks in the source vs. summed across
   that document's chunks; raises if they don't match, catching a split table rather
   than silently shipping a broken chunk.

<a id="embedding--bge-m3-settled"></a>
### Embedding, BGE-M3

`BAAI/bge-m3`, 1024-dim, multilingual, chosen specifically so queries in Tamil,
Burmese, Thai, etc. retrieve directly against the English-source corpus with no
pre-retrieval translation step.

<a id="retrieval--hybrid-bm25--dense-then-rerank-settled"></a>
### Retrieval, hybrid BM25 + dense, then rerank

Chroma indexes BGE-M3 dense vectors; `rank_bm25` runs lexical scoring in parallel;
results are combined via Reciprocal Rank Fusion (constant 60) over the top-10
candidates from each, then passed through a reranker. Six strategies were run
head-to-head, see [Evaluation results](#evaluation-results) for why SEA-LION-E5
reranking on top of hybrid retrieval won.

**Open question, not yet resolved:** BM25's lexical overlap mostly disappears for
non-English queries against an English corpus; worth splitting hybrid-vs-dense-only
by query language rather than assuming BM25 helps uniformly.

**Deferred, not built:** query rewriting + dual retrieval (retrieve on both the
original and an English-translated query, then merge) as a targeted fix for that
BM25 weakness, parked until real user query examples exist to confirm it's an
actual problem.

<a id="generation--sea-lion-8b-settled"></a>
### Generation, SEA-LION 8B

`aisingapore/Llama-SEA-LION-v3-8B-IT`, chosen over `qwen3:8b` at the same parameter
class specifically to isolate language specialization from model size. Generates
directly in the query's language, confirmed by a 10-query smoke test (7
non-English) with no "answer in the query's language" instruction in the system
prompt; it answered natively every time anyway.

<a id="conversation-memory--langgraph"></a>
### Conversation memory, LangGraph

Multi-turn state lives server-side, keyed by `thread_id`; the frontend only ever
sends `{message, thread_id}`, never full history. A compiled `StateGraph` runs four
nodes per turn:

- **summarize**: no-op until conversation length crosses a threshold, then
  condenses older turns instead of growing the prompt unboundedly.
- **rewrite_query**: folds history + the latest message into a standalone
  retrieval query. A bare follow-up like "what about for daily-rated workers?" has
  almost no signal for the retriever on its own.
- **retrieve**: the hybrid + SEA-LION-E5 rerank path above, optionally hitting the
  Redis retrieval cache first.
- **generate**: the graph is compiled with `interrupt_before=["generate"]`, it
  pauses right before this node so `/chat` can stream the answer token-by-token from
  outside the graph, then write the finished text back in and resume to `END`. A
  deliberate two-phase invoke, not a single blocking call, purely to make streaming
  and checkpointing compatible.

<a id="evaluation-results"></a>
## Evaluation results

Every retrieval or prompt change that graduates out of a notebook carries a
[Ragas](https://github.com/explodinggradients/ragas) run showing it didn't regress.
Retrieval is scored on MRR/precision/recall/nDCG against a labeled (query,
relevant-chunk) set; generation on faithfulness (groundedness to retrieved context)
and answer relevancy, both judged by `llama3.1:8b`.

<a id="reranker-comparison"></a>
### Reranker comparison

10 labeled queries (7 non-English), 16-chunk corpus, k=3. Each query has exactly one
labeled-relevant chunk, so `recall@3` is always identical to `hit@3`, and
`precision@3` is always `hit@3 / 3` (capped at 0.333 even under perfect retrieval). The
real signal here is MRR and nDCG@3, since both are sensitive to rank position rather
than just presence in the top 3:

| Strategy | MRR | Hit@3 | P@3 | Recall@3 | nDCG@3 | Latency |
|---|---|---|---|---|---|---|
| BM25 only | 0.403 | 0.600 | 0.200 | 0.600 | 0.439 | 0.1ms |
| hybrid_rerank + ms-marco-MiniLM | 0.425 | 0.500 | 0.167 | 0.500 | 0.426 | 267ms |
| hybrid (BM25+dense, no rerank) | 0.783 | 0.900 | 0.300 | 0.900 | 0.813 | 55ms |
| dense only (BGE-M3) | 0.875 | 0.900 | 0.300 | 0.900 | 0.863 | 137ms |
| hybrid_rerank + bge-reranker-v2-m3 | 0.817 | 1.000 | 0.333 | 1.000 | 0.863 | 4772ms |
| **hybrid_rerank + SEA-LION-E5** | **0.950** | 1.000 | 0.333 | 1.000 | **0.963** | 4872ms |

Why each strategy lands where it does:
- **BM25 only** misses the correct chunk in the top 3 on 4 of 10 queries, almost all
  non-English, where lexical overlap against the English corpus barely exists.
- **hybrid_rerank + ms-marco-MiniLM** is *worse than no rerank at all*: an
  English-only cross-encoder demotes correct chunks in a 70%-non-English pool.
- **hybrid (no rerank) scores below dense-only** despite matching hit@3: the noisy
  BM25 signal dilutes dense's ordering, often knocking the right chunk to 2nd or 3rd.
- **SEA-LION-E5 beats bge-reranker-v2-m3 on MRR and nDCG@3** despite tying on hit@3,
  landing the correct chunk at rank 1 more consistently, at the same latency.

(Two numbers are unexplained rather than papered over: hybrid-no-rerank benchmarking
faster than dense-only despite doing more work, and SEA-LION-E5 matching a
cross-encoder's latency despite being a bi-encoder. See `notebooks/FINDINGS.ipynb`
for the full run.)

<a id="generation-model-comparison"></a>
### Generation model comparison

Ragas, judge `llama3.1:8b`, embeddings BGE-M3, retrieval held constant (hybrid +
SEA-LION-E5 rerank) except where noted:

| Run | Retrieval | Generation | Faithfulness | Answer Relevancy |
|---|---|---|---|---|
| A | dense only | qwen3:8b | 0.846 | 0.792 |
| B | hybrid + SEA-LION-E5 | qwen3:8b | 0.771 | 0.774 |
| C | hybrid + SEA-LION-E5 | qwen3:8b | 0.794 | 0.828 |
| **D** | hybrid + SEA-LION-E5 | **SEA-LION-v3-8B-IT** | **0.900** | 0.813 |

Runs C and D drop the Burmese and Thai queries from the original 10-query set. Burmese
scored 1.0 in Run A but 0.25 in Run B on the *identical* query, which we traced to
qwen3 generation randomness rather than a retrieval regression. Thai scored 0.0 in
every run regardless of model or retrieval, a judge limitation, since `llama3.1:8b`
struggles to verify Thai-language answers against English-source context. Both were
excluded so the generation-model comparison wasn't swamped by judge noise unrelated
to what was being tested.

Net, SEA-LION generation gives a real faithfulness edge over qwen3:8b on the same 8
queries (+0.106, with every SEA-LION answer scoring 0.7 or higher versus two flagged
qwen3:8b answers), at a small enough answer-relevancy cost (−0.015) to read as noise
on 8 queries. It's a directional result, not a statistically robust one, but a real
edge on the metric that matters most for a compliance-adjacent domain: don't say
things that aren't in the source.

<a id="architectural-decisions"></a>
## Architectural decisions

<a id="conversational-memory-backend"></a>
### Conversational memory backend

Defaults to `memory` (LangGraph's built-in in-memory checkpointer), no external
service needed, but conversations are lost on every restart. Set
`MIGRANTBUDDY_CHECKPOINTER_BACKEND=redis` to persist them across restarts instead,
keyed by `thread_id` with a TTL so idle conversations expire. Redis itself has to be
running first, on Windows use [Memurai](https://www.memurai.com/) (free tier, installs
as a normal service, no Docker/WSL2 needed), listening on `localhost:6379` to match
`MIGRANTBUDDY_REDIS_URL`'s default.

<a id="retrieval-caching"></a>
### Retrieval caching

Off by default, and independent of the checkpointer toggle. Set
`MIGRANTBUDDY_RETRIEVAL_CACHE_ENABLED=true` to cache retrieval results in Redis by
`(query, top_k)`. It's a pure latency optimization: a cache miss or Redis error just
falls back to computing retrieval fresh, so it's safe to enable without affecting
correctness.

<a id="rate-limiting"></a>
### Rate limiting

Off by default. Set `MIGRANTBUDDY_RATE_LIMIT_ENABLED=true` to cap `/chat` (not
`/health`) at `RATE_LIMIT_MAX_REQUESTS` requests per client IP per
`RATE_LIMIT_WINDOW_SECONDS`, returning `429` once exceeded, since Ollama/vLLM can't
meaningfully serve concurrent generation requests. Like the retrieval cache, it fails
open: a Redis error lets requests through rather than blocking everyone.

<a id="speech-to-text"></a>
### Speech-to-text

Its own service (`migrantbuddy.speech.main`, port 8002), independent of `rag`, see
[Run the app](#3-run-the-app). The 🎤 button opens a WebSocket to it and streams audio
live via `faster-whisper`, which auto-detects the spoken language. Transcribed text
lands in the input box for you to review and edit rather than auto-sending, since
transcription errors matter more for employment/legal-rights questions.
`MIGRANTBUDDY_WHISPER_MODEL_SIZE=small` is the default balance of accuracy vs.
CPU-only speed, adjust it (and `MIGRANTBUDDY_WHISPER_DEVICE=cuda` if you have a GPU)
as needed.

<a id="text-to-speech--avatar"></a>
### Text-to-speech + avatar

Its own service (`migrantbuddy.tts.main`, port 8003), independent of `rag` and
`speech`. Answers are read aloud by a lip-synced [LiveAvatar](https://docs.liveavatar.com)
video (WebRTC, HeyGen ecosystem) using audio we generate via ElevenLabs. A mute toggle
next to the mic button skips this entirely, both providers are metered.

**Requires two accounts**: `ELEVENLABS_API_KEY`/`ELEVENLABS_VOICE_ID` for audio, and
`LIVEAVATAR_API_KEY`/`LIVEAVATAR_AVATAR_ID` for the avatar, all four are required.
`MIGRANTBUDDY_LIVEAVATAR_IS_SANDBOX` defaults to `true` so local testing doesn't burn
credits, turn it off for a real end-to-end check.

<a id="streaming-answers"></a>
### Streaming answers

`/chat` streams the answer back token-by-token over Server-Sent Events instead of
waiting for the full response, so text (and audio) starts appearing well before
generation finishes.

<a id="observability-langfuse"></a>
### Observability (Langfuse)

Off by default. Setting `LANGFUSE_PUBLIC_KEY`, `LANGFUSE_SECRET_KEY`, and
`LANGFUSE_HOST` together turns on per-stage tracing (each retrieval stage and
conversation node as a timed span, viewable in the Langfuse UI). Since this app can
handle sensitive queries like workplace complaints, consider self-hosting Langfuse
rather than sending query text to a third-party cloud instance.
