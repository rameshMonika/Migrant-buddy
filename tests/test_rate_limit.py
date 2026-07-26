import redis

from migrantbuddy.rate_limit import RateLimiter


class FakeRedisClient:
    """Stands in for redis.Redis -- an in-memory counter dict, so tests
    don't need a real Redis/Memurai instance running.
    """

    def __init__(self, *, raise_on_call: bool = False):
        self.counts: dict[str, int] = {}
        self.expire_calls: list[tuple[str, int]] = []
        self.raise_on_call = raise_on_call

    def incr(self, key: str) -> int:
        if self.raise_on_call:
            raise redis.RedisError("connection refused")
        self.counts[key] = self.counts.get(key, 0) + 1
        return self.counts[key]

    def expire(self, key: str, seconds: int) -> None:
        if self.raise_on_call:
            raise redis.RedisError("connection refused")
        self.expire_calls.append((key, seconds))


def test_is_allowed_true_under_the_limit():
    limiter = RateLimiter(client=FakeRedisClient(), max_requests=3, window_seconds=60)

    assert limiter.is_allowed("client-1") is True
    assert limiter.is_allowed("client-1") is True
    assert limiter.is_allowed("client-1") is True


def test_is_allowed_false_once_over_the_limit():
    limiter = RateLimiter(client=FakeRedisClient(), max_requests=3, window_seconds=60)

    for _ in range(3):
        limiter.is_allowed("client-1")

    assert limiter.is_allowed("client-1") is False


def test_is_allowed_tracks_different_keys_independently():
    limiter = RateLimiter(client=FakeRedisClient(), max_requests=1, window_seconds=60)

    assert limiter.is_allowed("client-1") is True
    assert limiter.is_allowed("client-2") is True  # separate client, own budget
    assert limiter.is_allowed("client-1") is False  # client-1 already used its one request


def test_is_allowed_sets_expiry_only_on_the_first_increment():
    client = FakeRedisClient()
    limiter = RateLimiter(client=client, max_requests=5, window_seconds=60)

    limiter.is_allowed("client-1")
    limiter.is_allowed("client-1")
    limiter.is_allowed("client-1")

    assert client.expire_calls == [("ratelimit:client-1", 60)]


def test_is_allowed_true_when_redis_errors():
    # Fail open -- Redis being unavailable must never lock out real users.
    limiter = RateLimiter(client=FakeRedisClient(raise_on_call=True), max_requests=1)

    assert limiter.is_allowed("client-1") is True
    assert limiter.is_allowed("client-1") is True  # still True even after "many" calls
