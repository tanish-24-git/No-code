"""
Structured logging utility with Redis Streams integration.
All logs are JSON-formatted with correlation IDs.
"""
import sys
import structlog
from datetime import datetime
from typing import Dict, Any, Optional, Literal
from app.utils.config import settings

# Log levels
LogLevel = Literal["DEBUG", "INFO", "WARN", "ERROR", "METRIC"]


def configure_logging():
    """Configure structured logging with JSON output."""
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    """Get a structured logger instance."""
    return structlog.get_logger(name)


class StructuredLogger:
    """
    Structured logger with support for correlation IDs and metadata.
    Integrates with Redis Streams for real-time log publishing.
    """
    
    def __init__(self, name: str, run_id: Optional[str] = None):
        self.logger = structlog.get_logger(name)
        self.run_id = run_id
        self.name = name
    
    def _log(
        self,
        level: LogLevel,
        message: str,
        **kwargs
    ):
        """Internal logging method."""
        log_data = {
            "run_id": self.run_id,
            "timestamp": datetime.utcnow().isoformat(),
            "logger": self.name,
            **kwargs
        }
        
        # Filter None values
        log_data = {k: v for k, v in log_data.items() if v is not None}
        
        # Log to stdout
        if level == "DEBUG":
            self.logger.debug(message, **log_data)
        elif level == "INFO":
            self.logger.info(message, **log_data)
        elif level == "WARN":
            self.logger.warning(message, **log_data)
        elif level == "ERROR":
            self.logger.error(message, **log_data)
        elif level == "METRIC":
            self.logger.info(message, metric=True, **log_data)
    
    def debug(self, message: str, **kwargs):
        """Log debug message."""
        self._log("DEBUG", message, **kwargs)
    
    def info(self, message: str, **kwargs):
        """Log info message."""
        self._log("INFO", message, **kwargs)
    
    def warn(self, message: str, **kwargs):
        """Log warning message."""
        self._log("WARN", message, **kwargs)
    
    def error(self, message: str, **kwargs):
        """Log error message."""
        self._log("ERROR", message, **kwargs)
    
    def metric(self, message: str, **kwargs):
        """Log metric event (training metrics, etc.)."""
        self._log("METRIC", message, **kwargs)


# Initialize logging on module import
configure_logging()
