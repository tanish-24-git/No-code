"""
Redis client wrapper with connection pooling and retry logic.
"""
import redis.asyncio as redis
from typing import Optional, Dict, Any
from tenacity import retry, stop_after_attempt, wait_exponential
from app.utils.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class RedisClient:
    """Redis client with connection pooling and helper methods."""
    
    def __init__(self):
        self._pool: Optional[redis.ConnectionPool] = None
        self._client: Optional[redis.Redis] = None
    
    async def connect(self):
        """Initialize Redis connection pool."""
        if self._pool is None:
            self._pool = redis.ConnectionPool.from_url(
                settings.redis_url,
                max_connections=settings.redis_max_connections,
                decode_responses=True
            )
            self._client = redis.Redis(connection_pool=self._pool)
            logger.info("Redis connection pool initialized", url=settings.redis_url)
    
    async def disconnect(self):
        """Close Redis connection pool."""
        if self._client:
            await self._client.close()
        if self._pool:
            await self._pool.disconnect()
        logger.info("Redis connection pool closed")
    
    @property
    def client(self) -> redis.Redis:
        """Get Redis client instance."""
        if self._client is None:
            raise RuntimeError("Redis client not initialized. Call connect() first.")
        return self._client
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def ping(self) -> bool:
        """Ping Redis server."""
        return await self.client.ping()
    
    async def set_job_state(self, job_id: str, state: Dict[str, Any]):
        """Set job state in Redis."""
        key = f"job:{job_id}"
        await self.client.hset(key, mapping=state)
    
    async def get_job_state(self, job_id: str) -> Optional[Dict[str, Any]]:
        """Get job state from Redis."""
        key = f"job:{job_id}"
        data = await self.client.hgetall(key)
        return data if data else None
    
    async def update_job_state(self, job_id: str, updates: Dict[str, Any]):
        """Update specific fields in job state."""
        key = f"job:{job_id}"
        await self.client.hset(key, mapping=updates)
    
    async def delete_job_state(self, job_id: str):
        """Delete job state from Redis."""
        key = f"job:{job_id}"
        await self.client.delete(key)
    
    async def set_agent_state(self, run_id: str, agent_name: str, state: Dict[str, Any]):
        """Set agent execution state."""
        key = f"agent:{run_id}:{agent_name}"
        await self.client.hset(key, mapping=state)
    
    async def get_agent_state(self, run_id: str, agent_name: str) -> Optional[Dict[str, Any]]:
        """Get agent execution state."""
        key = f"agent:{run_id}:{agent_name}"
        data = await self.client.hgetall(key)
        return data if data else None


# Global Redis client instance
redis_client = RedisClient()
