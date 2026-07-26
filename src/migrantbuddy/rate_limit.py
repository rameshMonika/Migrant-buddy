"""Redis-backed rate limiting (fixed-window INCR+EXPIRE) -- shared by the RAG
service (/chat) and the Whisper service (/transcribe/ws), each configuring
and instantiating its own RateLimiter with its own budget (see
migrantbuddy.config for the tunables and the fail-open rationale). Lives
here rather than under api/ since it isn't specific to either service.

Fails open: a Redis error is treated as "allow the request" -- rate
limiting is a protective measure, not a correctness requirement, and
Redis being optional infrastructure should never lock out real users.
"""

import redis

from migrantbuddy.config import RATE_LIMIT_MAX_REQUESTS, RATE_LIMIT_WINDOW_SECONDS, REDIS_URL


class RateLimiter:
    def __init__(
        self,
        redis_url: str = REDIS_URL,
        *,
        max_requests: int = RATE_LIMIT_MAX_REQUESTS,
        window_seconds: int = RATE_LIMIT_WINDOW_SECONDS,
        client: redis.Redis | None = None,
    ):
        self._client = client if client is not None else redis.Redis.from_url(redis_url, decode_responses=True)
        self._max_requests = max_requests
        self._window_seconds = window_seconds

    def is_allowed(self, key: str) -> bool:
        cache_key = f"ratelimit:{key}"
        try:
            count = self._client.incr(cache_key)
            if count == 1:
                # Only the request that actually creates the counter sets
                # its expiry -- subsequent increments within the window
                # must not keep pushing the expiry back out, or a
                # continuously-active client would never reset.
                self._client.expire(cache_key, self._window_seconds)
        except redis.RedisError:
            return True
        return count <= self._max_requests
