import redis

from migrantbuddy.retrieval import RetrievalResult
from migrantbuddy.retrieval.cache import RetrievalCache, _cache_key


class FakeRedisClient:
    """Stands in for redis.Redis -- an in-memory dict, so tests don't need
    a real Redis/Memurai instance running.
    """

    def __init__(self, *, raise_on_call: bool = False):
        self.store: dict[str, str] = {}
        self.raise_on_call = raise_on_call
        self.last_set_ttl = None

    def get(self, key: str):
        if self.raise_on_call:
            raise redis.RedisError("connection refused")
        return self.store.get(key)

    def set(self, key: str, value: str, ex: int | None = None):
        if self.raise_on_call:
            raise redis.RedisError("connection refused")
        self.store[key] = value
        self.last_set_ttl = ex


def test_cache_key_is_deterministic_for_same_query_and_top_k():
    assert _cache_key("overtime pay", 5) == _cache_key("overtime pay", 5)


def test_cache_key_differs_for_different_queries():
    assert _cache_key("overtime pay", 5) != _cache_key("salary timing", 5)


def test_cache_key_differs_for_different_top_k():
    assert _cache_key("overtime pay", 5) != _cache_key("overtime pay", 10)


def test_get_returns_none_on_cache_miss():
    cache = RetrievalCache(client=FakeRedisClient())

    assert cache.get("overtime pay", 5) is None


def test_set_then_get_round_trips_results():
    cache = RetrievalCache(client=FakeRedisClient())
    results = [RetrievalResult(chunk_id="c1", score=0.9), RetrievalResult(chunk_id="c2", score=0.5)]

    cache.set("overtime pay", 5, results)
    cached = cache.get("overtime pay", 5)

    assert cached == results


def test_set_uses_configured_ttl():
    client = FakeRedisClient()
    cache = RetrievalCache(client=client, ttl_seconds=12345)

    cache.set("overtime pay", 5, [RetrievalResult(chunk_id="c1", score=0.9)])

    assert client.last_set_ttl == 12345


def test_get_returns_none_when_redis_errors():
    cache = RetrievalCache(client=FakeRedisClient(raise_on_call=True))

    assert cache.get("overtime pay", 5) is None


def test_set_does_not_raise_when_redis_errors():
    cache = RetrievalCache(client=FakeRedisClient(raise_on_call=True))

    cache.set("overtime pay", 5, [RetrievalResult(chunk_id="c1", score=0.9)])  # should not raise
