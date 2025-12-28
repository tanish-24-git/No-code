"""
Client for TinyLlama base model inference via vLLM server (Container #4).
Used by TrainingAgent for model loading and evaluation.
"""
import httpx
from typing import Dict, Any, Optional
from app.utils.logging import get_logger
from app.utils.config import settings

logger = get_logger(__name__)


class TinyLlamaBaseClient:
    """
    Client for TinyLlama base model inference.
    Connects to vLLM server running in Container #4.
    """
    
    def __init__(self, base_url: str = "http://tinyllama-base:8001"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(timeout=60.0)
    
    async def health_check(self) -> bool:
        """Check if TinyLlama base service is healthy."""
        try:
            response = await self.client.get(f"{self.base_url}/health")
            return response.status_code == 200
        except Exception as e:
            logger.error("TinyLlama base health check failed", error=str(e))
            return False
    
    async def generate(
        self,
        prompt: str,
        max_tokens: int = 256,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generate text using TinyLlama base model.
        
        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_p: Nucleus sampling parameter
        
        Returns:
            Generated text
        """
        try:
            payload = {
                "model": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                "prompt": prompt,
                "max_tokens": max_tokens,
                "temperature": temperature,
                "top_p": top_p,
            }
            
            response = await self.client.post(
                f"{self.base_url}/v1/completions",
                json=payload
            )
            response.raise_for_status()
            
            result = response.json()
            generated_text = result["choices"][0]["text"]
            
            logger.info("Generated text", prompt_length=len(prompt), output_length=len(generated_text))
            return generated_text
        
        except Exception as e:
            logger.error("Text generation failed", error=str(e))
            raise
    
    async def get_model_info(self) -> Dict[str, Any]:
        """Get model information from vLLM server."""
        try:
            response = await self.client.get(f"{self.base_url}/v1/models")
            response.raise_for_status()
            return response.json()
        except Exception as e:
            logger.error("Failed to get model info", error=str(e))
            raise
    
    async def close(self):
        """Close HTTP client."""
        await self.client.aclose()


# Global instance
tinyllama_base_client = TinyLlamaBaseClient()
