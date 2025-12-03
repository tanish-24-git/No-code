# app/redis_client.py
import os
import anyio
import logging
from config.settings import settings
logger = logging.getLogger(__name__)
REDIS_URL = os.getenv("REDIS_URL", settings.redis_url)
# Preferred path: redis.asyncio (redis-py 4.x)
try:
    import redis.asyncio as redis_async # type: ignore
    redis_client = redis_async.from_url(REDIS_URL, decode_responses=True)
    logger.info("Using redis.asyncio client")
except Exception as exc:
    # Fallback: synchronous redis client wrapped into async methods using anyio.to_thread.run_sync
    logger.warning("redis.asyncio not available, falling back to sync redis wrapped with anyio: %s", exc)
    import redis as redis_sync # type: ignore
    sync_client = redis_sync.from_url(REDIS_URL, decode_responses=True)
    class AsyncRedisWrapper:
        """Small wrapper to expose async methods backed by a sync redis client via anyio.to_thread.run_sync."""
        def __init__(self, client):
            self._client = client
        async def hset(self, name, key=None, value=None, mapping=None, **kwargs):
            # mapping supported: call hset(name, mapping=mapping) if provided
            def _call():
                if mapping is not None:
                    return self._client.hset(name, mapping=mapping, **kwargs)
                if key is not None:
                    return self._client.hset(name, key, value, **kwargs)
                # fallback
                return self._client.hset(name, **kwargs)
            return await anyio.to_thread.run_sync(_call)
        async def hget(self, name, key):
            return await anyio.to_thread.run_sync(lambda: self._client.hget(name, key))
        async def hgetall(self, name):
            return await anyio.to_thread.run_sync(lambda: self._client.hgetall(name))
        async def set(self, key, value, *args, **kwargs):
            return await anyio.to_thread.run_sync(lambda: self._client.set(key, value, *args, **kwargs))
        async def get(self, key, *args, **kwargs):
            return await anyio.to_thread.run_sync(lambda: self._client.get(key, *args, **kwargs))
        async def delete(self, *keys):
            return await anyio.to_thread.run_sync(lambda: self._client.delete(*keys))
        async def exists(self, key):
            return await anyio.to_thread.run_sync(lambda: self._client.exists(key))
        async def ping(self):
            return await anyio.to_thread.run_sync(lambda: self._client.ping())
        # if you need more methods later, add here similarly
    redis_client = AsyncRedisWrapper(sync_client)