import json
from collections.abc import Callable
from typing import Any, TypeVar

from redis import Redis

from app.core.config import get_settings

T = TypeVar("T")


def get_redis() -> Redis:
    return Redis.from_url(get_settings().redis_url, decode_responses=True)


def cache_json(key: str, ttl_seconds: int, producer: Callable[[], T]) -> T:
    redis = get_redis()
    cached = redis.get(key)
    if cached:
        return json.loads(cached)
    value = producer()
    redis.setex(key, ttl_seconds, json.dumps(value, default=str))
    return value
