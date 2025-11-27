# app/redis_client.py
import os
import redis.asyncio as redis
from config.settings import settings

REDIS_URL = os.getenv("REDIS_URL", settings.redis_url)
redis_client = redis.from_url(REDIS_URL, decode_responses=True)
