# migrantBuddy

RAG system for SG (Singapore) migrants — answers questions grounded in migrant-relevant
documents (work pass rules, MOM guidelines, support services, etc.).

## Current implementation

The project is currently in the **research and prototyping phase** — all work so far is
in `notebooks/`; `src/migrantbuddy/` does not exist on disk yet.

Development starts in Jupyter notebooks to experiment with retrieval strategies,
embedding models, chunking approaches and evaluation before productionizing the
pipeline. Core pipeline decisions are now **settled** (see Architecture decisions
below) — embedding (BGE-M3), retrieval (hybrid BM25 + dense), reranker (SEA-LION-E5),
and generation model (SEA-LION 8B) — so the next step is extracting this into
`src/migrantbuddy/`, not further pipeline experimentation, unless a specific
regression or open question below calls for it.

Current workflow:

1. Notebook exploration (`notebooks/`) — done for the core pipeline
   - Document ingestion
   - Chunking experiments
   - Embedding evaluation (BGE-M3 — settled default)
   - Hybrid retrieval (BM25 + dense, Chroma) + rerank experiments (SEA-LION-E5 —
     settled reranker)
   - Ollama RAG (local generation, fast iteration) — SEA-LION 8B is the settled
     generation model, evaluated against qwen3:8b
   - Ragas evaluation (retrieval + generation quality)

2. Productionization
   - Refactor validated notebook code into `src/migrantbuddy/`
   - Swap Ollama → vLLM for production generation serving (continuous batching /
     PagedAttention for throughput) — **deferred**: `vllm_client.py` exists and is
     tested, but no vLLM server has been run yet (Windows install hit a Long Path
     error; native Windows support is rough besides — WSL2 recommended if/when
     this is picked back up). Ollama remains the active generation backend.
   - Add FastAPI backend
   - Build React + TypeScript frontend
   - Dockerize — **deferred** alongside vLLM, same conversation
   - Add CI/CD
   - Deploy

Notebooks are for experiments and throwaway analysis, not a second home for production
code. Once a notebook approach is stable and evaluated (via Ragas), extract it into
`src/migrantbuddy/` with type hints and tests — don't leave working logic stranded in a
notebook. When a notebook cell graduates to `src/`, the notebook should import and call
it, not duplicate it.

## Project structure

Notebooks are numbered in pipeline order so execution/reading order is obvious at a
glance: `01_ingestion.ipynb`, `02_chunking.ipynb`, `03_embedding.ipynb`,
`04_retrieval.ipynb`, `05_generation.ipynb`, `06_eval.ipynb`. No separate translation
stage — see the Translation decision below.
Follow-up experiments on a step reuse its number with a suffix (e.g.
`02b_chunking_structure_aware.ipynb`) rather than being appended at the end.

`notebooks/` and `data/` below exist today. `src/migrantbuddy/` and `tests/` are the
**planned** layout for the productionization step — not created yet.

```
notebooks/              exploration & eval scratch work (exists)
src/migrantbuddy/        [planned, not yet created]
  ingestion/             document loading (HTML via trafilatura, PDF via pymupdf),
                         structure-aware chunking, table extraction (kept whole,
                         markdown-formatted)
  retrieval/             hybrid BM25 + dense (BGE-M3 embeddings, Chroma vector store)
                         + SEA-LION-E5 reranking
  generation/            prompt construction, Ollama (dev) / vLLM (prod) calls —
                         SEA-LION generates directly in the query's language, no
                         separate translation step
  eval/                  Ragas-based eval harness + metrics
tests/                   [planned, not yet created] mirrors src/migrantbuddy structure
data/                    raw docs, eval sets (gitignored) (exists)
  processed/eval_results/ saved per-notebook eval-run JSON snapshots (metrics +
                         per-query results), so retrieval/reranker experiments
                         across notebooks stay comparable without re-running them
```

## Architecture decisions

- **Embedding**: BGE-M3 (multilingual) — retrieves directly against non-English
  queries, no pre-retrieval translation needed.
  - Experiment: `03_embedding` builds and smoke-tests the full pipeline with
    `BAAI/bge-m3` (1024-dim; natively supports dense/sparse/multi-vector retrieval)
    first, since it's the committed default, not just a candidate — build the
    reference implementation around the model you actually expect to ship. Once the
    full pipeline (retrieval → generation) works end-to-end with it,
    swap in `paraphrase-multilingual-MiniLM-L12-v2` (384-dim multilingual variant —
    not `all-MiniLM-L6-v2`, which is English-only and wouldn't give a meaningful
    multilingual comparison) as a follow-up notebook (`03b_embedding_minilm`, own
    Chroma collection — dimensions differ, can't share one) and compare the two full
    pipeline runs via Ragas, ideally with test queries covering multiple target
    languages — especially Tamil and Burmese, where multilingual coverage is least
    certain for smaller models like MiniLM.
- **Vector store**: Chroma.
- **Retrieval**: hybrid BM25 + dense (BGE-M3, indexed in Chroma) + rerank using
  `aisingapore/SEA-LION-E5-Embedding-600M` as default — **settled**, see Stage 04
  in `notebooks/FINDINGS.ipynb` for the full metrics comparison. SEA-LION-E5 beat
  `ms-marco-MiniLM-L-6-v2` (English-only, breaks on cross-lingual queries) and
  `bge-reranker-v2-m3` on MRR against the labeled eval set. `04_retrieval.ipynb`
  keeps its original `ms-marco-MiniLM-L-6-v2` reranker as the historical baseline
  that motivated the comparison; `05_generation.ipynb`/`06_eval.ipynb` onward use
  `hybrid` + SEA-LION reranking as the default.
  - Open question: BM25 relies on lexical term overlap, which mostly disappears
    for non-English queries against the English-source corpus (aside from proper
    nouns/codes like "S Pass" or dollar figures). Worth evaluating hybrid vs.
    dense-only split by query language rather than assuming BM25 helps uniformly.
  - Deferred: query rewriting + dual retrieval (retrieve on both the original and
    a rewritten/English-translated query, merge results) — a candidate fix for the
    BM25 non-English weakness above. No training required, just another prompt to
    an existing LLM before retrieval. Deferred until real user query examples
    exist (see Domain notes) to confirm the problem it would fix actually shows up
    in practice, rather than adding pipeline complexity speculatively.
- **Chunking**: structure-aware (split on document sections/headings first, then
  sub-chunk oversized sections by tokens), 300–500 tokens, ~15–20% overlap. Tables
  extracted and kept whole, represented as markdown in chunk text — never split
  across chunks.
  - Experiment: chunk size (300/500/800), overlap (10%/20%), structure-aware vs
    naive fixed-window baseline.
- **Generation serving**: Ollama for notebook exploration (fast local iteration,
  no GPU box needed), vLLM for production backend (throughput via continuous
  batching / PagedAttention).
- **Generation model**: `aisingapore/Llama-SEA-LION-v3-8B-IT` — chosen over
  `qwen3:8b` (the original default in `05_generation.ipynb`/`06_eval.ipynb`).
  - Experiment: `05b_generation_sealion.ipynb` swapped the generation model to
    SEA-LION's 8B instruct model (same size class as qwen3:8b, so the
    comparison isolates language specialization rather than parameter count),
    keeping retrieval (BGE-M3 + SEA-LION reranker) and prompt unchanged.
    `06b_eval_sealion_generation.ipynb` scored both with the identical Ragas
    harness (Faithfulness, AnswerRelevancy; judge `llama3.1:8b`) on the same
    8-query labeled set: SEA-LION **0.900 faithfulness / 0.813
    answer_relevancy** vs qwen3:8b **0.794 / 0.828** — a real faithfulness
    edge (+0.106, every SEA-LION query scored ≥0.7 vs two flagged qwen3:8b
    queries) at an answer_relevancy difference (-0.015) small enough to be
    noise on 8 queries.
  - **Decision**: SEA-LION is the chosen generation model going forward
    (e.g. when extracting to `src/migrantbuddy/generation/`). `qwen3:8b`
    remains in `05_generation.ipynb`/`06_eval.ipynb` as the historical
    baseline that motivated the comparison, same as `04_retrieval.ipynb`
    keeping its original `ms-marco-MiniLM-L-6-v2` reranker after
    `04b_retrieval_sealion_rerank.ipynb` settled on SEA-LION there.
  - Caveat: directional signal from an 8-query eval set, not a statistically
    robust result — see `06_eval.ipynb`'s Burmese/Thai episode (Stage 06 in
    `notebooks/FINDINGS.ipynb`) for how much a single query flip can move
    these averages.
- **Translation**: not a separate pipeline stage — `aisingapore/Llama-SEA-LION-v3-8B-IT`
  generates directly in the query's language, so no post-generation translation step
  is needed. Confirmed by `05b_generation_sealion.ipynb`'s 10-query smoke test (7
  non-English: ms/ta/my/th/vi): with no "answer in the query's language" instruction
  in `SYSTEM_PROMPT`, SEA-LION answered every non-English query natively in-language,
  grounded in context retrieved from the English-source corpus. This also settles the
  earlier query-side translation question — no pre-translation of incoming queries to
  English either, since BGE-M3 retrieves directly against non-English queries (see
  Embedding decision above) and SEA-LION generates directly off the native-language
  query. Full pipeline is retrieval → generation, no translation stage on either side.
- **Conversational memory**: multi-turn conversation support via LangGraph
  (`MessagesState` + checkpointing) — **planned, not yet implemented**. Replaces the
  current single-turn `RagService.answer(query)` orchestration (retrieval +
  generation, no memory of prior turns) once built. Frontend sends `{message,
  thread_id}` per turn, not the full history — conversation state lives
  server-side in the checkpointer, keyed by `thread_id`.
  - **Rollout, two stages** (deliberately split so "conversation memory works"
    doesn't get blocked on new infra):
    - **Stage 1**: a LangGraph state graph using the built-in in-memory
      checkpointer (no Redis, no new infra). Nodes: summarize (once conversation
      size exceeds a threshold, condense older turns into a summary instead of
      growing the prompt unboundedly) → query rewrite (condense conversation
      history + the latest message into a standalone retrieval query, since a
      bare follow-up like "what about for daily-rated workers?" has almost no
      signal on its own for `Retriever.hybrid_rerank`) → retrieve (existing
      `Retriever.hybrid_rerank`, unchanged) → generate (existing
      `ollama_client`/`vllm_client`, unchanged).
    - **Stage 2**: swap the in-memory checkpointer for a Redis-backed one
      (`thread_id`-keyed, TTL e.g. 24h so conversations expire instead of
      growing Redis unboundedly). Deferred until Redis is actually running
      somewhere for local dev — native Windows Redis support is poor; WSL2,
      Memurai, or a hosted free tier are the realistic options. Docker itself
      is separately deferred (see Current implementation), so Redis needs its
      own answer independent of "just containerize everything."
  - **New dependency**: `langgraph` (Stage 1), plus `redis` /
    `langgraph-checkpoint-redis` (Stage 2) — the first genuine LangChain-
    ecosystem *runtime* dependency in `src/migrantbuddy/`, which was otherwise
    deliberately framework-free (plain `requests`/`chromadb`/
    `sentence-transformers` calls, per Coding standards below).
  - **Caveat**: query rewriting is a second LLM call every turn, on top of
    generation — a deliberate latency tradeoff (needed for follow-up retrieval
    to work at all), made consciously despite the effort already spent
    reducing single-turn latency (Ollama `keep_alive` tuning,
    `GENERATION_MAX_TOKENS` cap).

## Coding standards

- Type hints on all function signatures in `src/`.
- Docstrings only where behavior isn't obvious from the name/signature — no restating
  the obvious.
- Formatting/linting enforced via `ruff` + `black` (configured in `pyproject.toml`,
  once added) — run before committing, don't hand-format.
- No premature abstraction: build the concrete version first, generalize only when a
  second real use case shows up.
- Prefer explicit, small functions over deep class hierarchies for pipeline stages
  (ingestion/retrieval/generation are naturally functional).

## Evaluation

Evaluated with **Ragas**, tracking retrieval and generation quality separately:
- **Retrieval**: precision/recall/MRR against a labeled eval set of (query, relevant
  doc) pairs.
- **Generation**: faithfulness/groundedness to retrieved context, answer relevance.

Every retrieval or prompt change that graduates to `src/` should have a Ragas eval run
attached to it (in `notebooks/` or `src/migrantbuddy/eval/`) showing it didn't regress.

## Domain notes

- Target languages: aligned to SEA-LION coverage (Burmese, Filipino, Indonesian, Malay,
  Tamil, Thai, Vietnamese) — confirm against actual target migrant worker population in
  SG before finalizing.
- Corpus: English-source MOM documents across 5 categories — salary, working hours,
  work permit conditions, medical insurance, and help/contact info. 5 seed pages
  total. Three categories deferred pending real content pages: Employment Rights
  (termination, dismissal, repatriation, contracts) has no page sourced yet; Work
  Injury (WICA) and Housing both had candidate URLs that turned out to be
  landing/hub pages (~90% navigation, one intro sentence, real content on
  un-sourced sub-pages) — confirmed by fetching them directly, dropped from
  `01_ingestion` rather than ingesting near-empty stubs. Some pages tabular (salary
  thresholds, eligibility criteria) — table structure preserved through ingestion,
  not flattened to prose.
- Query examples: TBD — add 5–10 representative real user questions once gathered, use
  as seed for eval set.
- Ingestion metadata: each document now carries `category` (assigned per-URL, e.g.
  "salary", "work-permit" — not a blanket constant, since the corpus now spans
  multiple real categories) and `content_type` (e.g. "official_guidance" — constant
  for now since all sources are MOM guidance pages, kept as a separate field for when
  other source types are added later). `category` was previously deferred while the
  corpus was only 2 seed pages; added now that real category structure exists.
