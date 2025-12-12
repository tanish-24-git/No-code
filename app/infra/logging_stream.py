"""
Real-time log streaming using Redis Streams.
Supports publishing structured logs and consuming via Server-Sent Events.
"""
import json
from typing import Dict, Any, Optional, AsyncGenerator
from datetime import datetime, timedelta
import redis.asyncio as redis
from app.utils.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class LogStream:
    """
    Real-time log streaming using Redis Streams.
    Each run has its own stream for isolated log consumption.
    """
    
    def __init__(self, redis_client: redis.Redis):
        self.redis = redis_client
        self.retention_days = settings.log_retention_days
    
    def _stream_key(self, run_id: str) -> str:
        """Get Redis stream key for a run."""
        return f"logs:{run_id}"
    
    async def publish_log(
        self,
        run_id: str,
        agent: str,
        level: str,
        message: str,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Publish a log event to the stream.
        
        Args:
            run_id: Unique run identifier
            agent: Agent name that generated the log
            level: Log level (INFO, WARN, ERROR, METRIC)
            message: Log message
            metadata: Additional metadata (step, epoch, loss, etc.)
        """
        stream_key = self._stream_key(run_id)
        
        log_event = {
            "timestamp": datetime.utcnow().isoformat(),
            "run_id": run_id,
            "agent": agent,
            "level": level,
            "message": message,
            **(metadata or {})
        }
        
        # Convert all values to strings for Redis
        log_data = {k: json.dumps(v) if isinstance(v, (dict, list)) else str(v) 
                   for k, v in log_event.items()}
        
        # Add to stream with automatic ID
        await self.redis.xadd(stream_key, log_data)
        
        # Set expiration on the stream
        expiration = timedelta(days=self.retention_days)
        await self.redis.expire(stream_key, int(expiration.total_seconds()))
    
    async def consume_logs(
        self,
        run_id: str,
        start_id: str = "0",
        block_ms: int = 5000
    ) -> AsyncGenerator[Dict[str, Any], None]:
        """
        Consume logs from a stream (async generator for SSE).
        
        Args:
            run_id: Run identifier
            start_id: Stream ID to start from ('0' for beginning, '$' for new only)
            block_ms: Milliseconds to block waiting for new messages
        
        Yields:
            Log event dictionaries
        """
        stream_key = self._stream_key(run_id)
        last_id = start_id
        
        while True:
            # Read from stream
            messages = await self.redis.xread(
                {stream_key: last_id},
                count=10,
                block=block_ms
            )
            
            if not messages:
                # No new messages, yield keepalive
                yield {"type": "keepalive"}
                continue
            
            # Process messages
            for stream, entries in messages:
                for entry_id, data in entries:
                    # Parse log event
                    log_event = {}
                    for key, value in data.items():
                        try:
                            # Try to parse JSON values
                            log_event[key] = json.loads(value)
                        except (json.JSONDecodeError, TypeError):
                            log_event[key] = value
                    
                    yield log_event
                    last_id = entry_id
    
    async def get_log_history(
        self,
        run_id: str,
        count: int = 100
    ) -> list[Dict[str, Any]]:
        """
        Get historical logs for a run.
        
        Args:
            run_id: Run identifier
            count: Maximum number of logs to retrieve
        
        Returns:
            List of log events
        """
        stream_key = self._stream_key(run_id)
        
        # Read from beginning
        messages = await self.redis.xrange(stream_key, count=count)
        
        logs = []
        for entry_id, data in messages:
            log_event = {"_id": entry_id}
            for key, value in data.items():
                try:
                    log_event[key] = json.loads(value)
                except (json.JSONDecodeError, TypeError):
                    log_event[key] = value
            logs.append(log_event)
        
        return logs
    
    async def delete_stream(self, run_id: str):
        """Delete a log stream."""
        stream_key = self._stream_key(run_id)
        await self.redis.delete(stream_key)
        logger.info("Log stream deleted", run_id=run_id)
