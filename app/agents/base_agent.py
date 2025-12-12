"""
Base agent abstract class.
All agents must inherit from this class and implement the execute method.
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from pydantic import BaseModel, ValidationError as PydanticValidationError
from app.utils.logging import StructuredLogger
from app.utils.exceptions import AgentException, ValidationException
from app.infra.logging_stream import LogStream
from app.infra.redis import redis_client


class AgentInput(BaseModel):
    """Base input schema for all agents."""
    run_id: str
    

class AgentOutput(BaseModel):
    """Base output schema for all agents."""
    success: bool
    data: Dict[str, Any] = {}
    error: Optional[str] = None


class BaseAgent(ABC):
    """
    Abstract base class for all agents.
    
    Agents are stateless execution units that:
    - Receive structured JSON input
    - Emit structured JSON output
    - Publish logs and status events
    - Persist artifacts to object storage
    - Never communicate via free text
    """
    
    def __init__(self, agent_name: str):
        self.agent_name = agent_name
        self.logger: Optional[StructuredLogger] = None
        self.log_stream: Optional[LogStream] = None
    
    def _initialize_logging(self, run_id: str):
        """Initialize logging for this agent execution."""
        self.logger = StructuredLogger(self.agent_name, run_id=run_id)
        self.log_stream = LogStream(redis_client.client)
    
    async def _emit_log(
        self,
        run_id: str,
        level: str,
        message: str,
        **metadata
    ):
        """Emit a structured log event."""
        if self.logger:
            # Log to stdout
            if level == "INFO":
                self.logger.info(message, **metadata)
            elif level == "WARN":
                self.logger.warn(message, **metadata)
            elif level == "ERROR":
                self.logger.error(message, **metadata)
            elif level == "METRIC":
                self.logger.metric(message, **metadata)
        
        # Publish to Redis Stream for real-time consumption
        if self.log_stream:
            await self.log_stream.publish_log(
                run_id=run_id,
                agent=self.agent_name,
                level=level,
                message=message,
                metadata=metadata
            )
    
    async def _set_state(self, run_id: str, state: Dict[str, Any]):
        """Persist agent state to Redis."""
        await redis_client.set_agent_state(run_id, self.agent_name, state)
    
    async def _get_state(self, run_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve agent state from Redis."""
        return await redis_client.get_agent_state(run_id, self.agent_name)
    
    def validate_input(self, input_data: Dict[str, Any], schema: type[BaseModel]) -> BaseModel:
        """
        Validate input data against a Pydantic schema.
        
        Args:
            input_data: Raw input dictionary
            schema: Pydantic model class
        
        Returns:
            Validated Pydantic model instance
        
        Raises:
            ValidationException: If validation fails
        """
        try:
            return schema(**input_data)
        except PydanticValidationError as e:
            raise ValidationException(
                f"Input validation failed for {self.agent_name}",
                metadata={"errors": e.errors()}
            )
    
    @abstractmethod
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute agent logic. Must be implemented by subclasses.
        
        Args:
            input_data: Structured input dictionary
        
        Returns:
            Structured output dictionary
        
        Raises:
            AgentException: If execution fails
        """
        pass
    
    async def run(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main entry point for agent execution.
        Handles logging initialization, state management, and error handling.
        
        Args:
            input_data: Structured input dictionary
        
        Returns:
            Structured output dictionary
        """
        # Extract run_id
        run_id = input_data.get("run_id")
        if not run_id:
            raise ValidationException("run_id is required in input_data")
        
        # Initialize logging
        self._initialize_logging(run_id)
        
        try:
            # Set initial state
            await self._set_state(run_id, {"status": "running", "agent": self.agent_name})
            await self._emit_log(run_id, "INFO", f"{self.agent_name} started")
            
            # Execute agent logic
            output = await self.execute(input_data)
            
            # Set success state
            await self._set_state(run_id, {"status": "completed", "agent": self.agent_name})
            await self._emit_log(run_id, "INFO", f"{self.agent_name} completed successfully")
            
            return {
                "success": True,
                "data": output,
                "error": None
            }
        
        except AgentException as e:
            # Handle known agent exceptions
            await self._set_state(run_id, {
                "status": "failed",
                "agent": self.agent_name,
                "error": e.error_code
            })
            await self._emit_log(
                run_id,
                "ERROR",
                f"{self.agent_name} failed: {e.message}",
                error_code=e.error_code,
                metadata=e.metadata
            )
            
            return {
                "success": False,
                "data": {},
                "error": e.to_dict()
            }
        
        except Exception as e:
            # Handle unexpected exceptions
            await self._set_state(run_id, {
                "status": "failed",
                "agent": self.agent_name,
                "error": "UNKNOWN_ERROR"
            })
            await self._emit_log(
                run_id,
                "ERROR",
                f"{self.agent_name} failed with unexpected error: {str(e)}"
            )
            
            return {
                "success": False,
                "data": {},
                "error": {
                    "error": "UNKNOWN_ERROR",
                    "message": str(e),
                    "metadata": {}
                }
            }
