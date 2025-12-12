"""
Task queue abstraction using RQ (Redis Queue).
Supports job submission, status tracking, and retries.
"""
from typing import Dict, Any, Optional, Callable
from rq import Queue
from rq.job import Job
from redis import Redis
from app.utils.config import settings
from app.utils.logging import get_logger

logger = get_logger(__name__)


class TaskQueue:
    """
    Task queue abstraction for background job execution.
    Uses RQ (Redis Queue) for simplicity and reliability.
    """
    
    def __init__(self, redis_url: str = None):
        self.redis_url = redis_url or settings.redis_url
        # Create synchronous Redis connection for RQ
        self.redis_conn = Redis.from_url(self.redis_url, decode_responses=False)
        
        # Define queues with different priorities
        self.default_queue = Queue('default', connection=self.redis_conn)
        self.training_queue = Queue('training', connection=self.redis_conn)
        self.evaluation_queue = Queue('evaluation', connection=self.redis_conn)
        self.orchestration_queue = Queue('orchestration', connection=self.redis_conn)
        
        logger.info("Task queues initialized", redis_url=self.redis_url)
    
    def enqueue_task(
        self,
        func: Callable,
        queue_name: str = 'default',
        job_id: Optional[str] = None,
        timeout: int = 3600,
        retry_max: int = 3,
        **kwargs
    ) -> Job:
        """
        Enqueue a task to the specified queue.
        
        Args:
            func: Function to execute
            queue_name: Queue name ('default', 'training', 'evaluation', 'orchestration')
            job_id: Optional custom job ID
            timeout: Job timeout in seconds
            retry_max: Maximum number of retries
            **kwargs: Arguments to pass to the function
        
        Returns:
            RQ Job instance
        """
        queue = self._get_queue(queue_name)
        
        job = queue.enqueue(
            func,
            job_id=job_id,
            timeout=timeout,
            retry=retry_max,
            **kwargs
        )
        
        logger.info(
            "Task enqueued",
            job_id=job.id,
            queue=queue_name,
            func=func.__name__
        )
        
        return job
    
    def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """
        Get job status and result.
        
        Returns:
            Dictionary with status, result, and error information
        """
        job = Job.fetch(job_id, connection=self.redis_conn)
        
        return {
            "job_id": job.id,
            "status": job.get_status(),
            "created_at": job.created_at.isoformat() if job.created_at else None,
            "started_at": job.started_at.isoformat() if job.started_at else None,
            "ended_at": job.ended_at.isoformat() if job.ended_at else None,
            "result": job.result,
            "exc_info": job.exc_info,
            "meta": job.meta
        }
    
    def cancel_job(self, job_id: str):
        """Cancel a job."""
        job = Job.fetch(job_id, connection=self.redis_conn)
        job.cancel()
        logger.info("Job cancelled", job_id=job_id)
    
    def _get_queue(self, queue_name: str) -> Queue:
        """Get queue by name."""
        queues = {
            'default': self.default_queue,
            'training': self.training_queue,
            'evaluation': self.evaluation_queue,
            'orchestration': self.orchestration_queue
        }
        return queues.get(queue_name, self.default_queue)


# Global task queue instance
task_queue = TaskQueue()
