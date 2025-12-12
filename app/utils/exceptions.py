"""
Custom exception hierarchy for the LLM fine-tuning platform.
All exceptions include error codes and structured metadata.
"""
from typing import Dict, Any, Optional


class AgentException(Exception):
    """Base exception for all agent-related errors."""
    
    def __init__(
        self,
        message: str,
        error_code: str = "AGENT_ERROR",
        metadata: Optional[Dict[str, Any]] = None
    ):
        self.message = message
        self.error_code = error_code
        self.metadata = metadata or {}
        super().__init__(self.message)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert exception to dictionary for API responses."""
        return {
            "error": self.error_code,
            "message": self.message,
            "metadata": self.metadata
        }


class ValidationException(AgentException):
    """Raised when data validation fails."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="VALIDATION_ERROR", metadata=metadata)


class TrainingException(AgentException):
    """Raised when training fails."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="TRAINING_ERROR", metadata=metadata)


class StorageException(AgentException):
    """Raised when object storage operations fail."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="STORAGE_ERROR", metadata=metadata)


class ConfigurationException(AgentException):
    """Raised when configuration is invalid."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="CONFIG_ERROR", metadata=metadata)


class DatasetException(AgentException):
    """Raised when dataset operations fail."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="DATASET_ERROR", metadata=metadata)


class PreprocessingException(AgentException):
    """Raised when preprocessing fails."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="PREPROCESSING_ERROR", metadata=metadata)


class EvaluationException(AgentException):
    """Raised when evaluation fails."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="EVALUATION_ERROR", metadata=metadata)


class ExportException(AgentException):
    """Raised when model export fails."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="EXPORT_ERROR", metadata=metadata)


class OrchestrationException(AgentException):
    """Raised when pipeline orchestration fails."""
    
    def __init__(self, message: str, metadata: Optional[Dict[str, Any]] = None):
        super().__init__(message, error_code="ORCHESTRATION_ERROR", metadata=metadata)
