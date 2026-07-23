"""Redis-backed cache for retrieval results, keyed by (query, top_k). The
corpus (data/processed/chunks.json + the Chroma collection) only changes on
a deliberate re-ingestion, so a repeated/similar query can safely reuse a
cached result for a long time -- unlike conversation state, retrieval
results aren't inherently time-sensitive.

Deliberately fails soft: a Redis error on get/set is swallowed and treated
as a cache miss -- this is purely a latency optimization, never a
correctness requirement, so Redis being unavailable must never break
retrieval. The caller (rag/nodes.py's retrieve_node) has its own separate
fallback for a *stale* cache hit (a chunk_id from before a re-ingestion) --
this module only handles Redis-level failures, not corpus-version staleness.
"""

import hashlib
import json

import redis

from migrantbuddy.config import REDIS_URL, RETRIEVAL_CACHE_TTL_SECONDS
from migrantbuddy.retrieval.service import RetrievalResult


def _cache_key(query: str, top_k: int) -> str:
    digest = hashlib.sha256(query.encode("utf-8")).hexdigest()
    return f"retrieval:{top_k}:{digest}"


class RetrievalCache:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        *,
        ttl_seconds: int = RETRIEVAL_CACHE_TTL_SECONDS,
        client: redis.Redis | None = None,
    ):
        self._client = client if client is not None else redis.Redis.from_url(redis_url, decode_responses=True)
        self._ttl_seconds = ttl_seconds

    def get(self, query: str, top_k: int) -> list[RetrievalResult] | None:
        try:
            raw = self._client.get(_cache_key(query, top_k))
        except redis.RedisError:
            return None
        if raw is None:
            return None
        return [RetrievalResult(chunk_id=item["chunk_id"], score=item["score"]) for item in json.loads(raw)]

    def set(self, query: str, top_k: int, results: list[RetrievalResult]) -> None:
        payload = json.dumps([{"chunk_id": r.chunk_id, "score": r.score} for r in results])
        try:
            self._client.set(_cache_key(query, top_k), payload, ex=self._ttl_seconds)
        except redis.RedisError:
            pass
