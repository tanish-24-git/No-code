# services/llm_service.py
import os
import json
from typing import Dict, Any, Optional
import httpx
import structlog
from config.settings import settings

logger = structlog.get_logger()

class LLMService:
    def __init__(self):
        self.provider = settings.llm_provider or "generic"
        self.api_url = settings.llm_api_url
        self.api_key = settings.llm_api_key
        # Vertex/Google config
        self.google_project = settings.google_project
        self.google_location = settings.google_location
        self.vertex_model_id = settings.vertex_model_id

    async def analyze_dataset(self, summary: Dict[str, Any], sample_rows: str) -> Dict[str, Any]:
        prompt = self._build_prompt(summary, sample_rows)
        logger.info("LLM analyze dataset", provider=self.provider)

        if self.provider == "vertex":
            return await self._call_vertex_predict(prompt)
        else:
            return await self._call_generic_http(prompt)

    def _build_prompt(self, summary: Dict[str, Any], sample_rows: str) -> str:
        return f"""
You are a data scientist assistant. Given the dataset summary and a small sample of rows, return strictly valid JSON with keys:
 - suggested_task: one of ["classification","regression","clustering"]
 - suggested_target: column name or null
 - missing_value_strategy: one of ["mean","median","mode","drop","custom"]
 - feature_engineering: list of strings
 - recommended_models: list of objects with keys {{"model","reason","hyperparams"}}
 - confidence: number between 0 and 1

Dataset summary: {json.dumps(summary)}
Sample rows (CSV): 
{sample_rows}

Return ONLY valid JSON and nothing else.
"""

    async def _call_generic_http(self, prompt: str) -> Dict[str, Any]:
        if not self.api_url:
            logger.warning("LLM API URL not configured")
            return {}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"prompt": prompt, "max_tokens": 512, "temperature": 0.0}
        async with httpx.AsyncClient(timeout=60) as client:
            r = await client.post(self.api_url, headers=headers, json=payload)
            r.raise_for_status()
            data = r.json()
        # Try to extract text
        text = None
        for k in ("text", "generated_text", "output", "content"):
            if k in data:
                text = data[k]
                break
        if not text:
            # try to find nested
            text = json.dumps(data)
        try:
            return json.loads(text)
        except Exception:
            logger.warning("LLM response not JSON, returning raw string in 'raw' key")
            return {"raw": text}

    async def _call_vertex_predict(self, prompt: str) -> Dict[str, Any]:
        # Vertex AI predict endpoint usage — requires GOOGLE_PROJECT & VERTEX_MODEL_ID
        if not self.google_project or not self.vertex_model_id:
            logger.error("Vertex config missing")
            return {}
        url = f"https://{self.google_location}-aiplatform.googleapis.com/v1/projects/{self.google_project}/locations/{self.google_location}/models/{self.vertex_model_id}:predict"
        body = {"instances": [{"content": prompt}], "parameters": {"temperature": 0.0, "maxOutputTokens": 800}}
        headers = {"Content-Type": "application/json"}
        params = {}
        if self.api_key:
            params["key"] = self.api_key
        async with httpx.AsyncClient(timeout=120) as client:
            r = await client.post(url, headers=headers, params=params, json=body)
            r.raise_for_status()
            resp = r.json()
        # Attempt to extract JSON from response
        candidates = []
        if "predictions" in resp and len(resp["predictions"]) > 0:
            pred0 = resp["predictions"][0]
            for key in ("content", "text", "output", "response"):
                if key in pred0:
                    candidates.append(pred0[key])
        # try parse
        for c in candidates:
            try:
                if isinstance(c, dict):
                    return c
                if isinstance(c, str):
                    return json.loads(c)
            except Exception:
                continue
        logger.warning("Vertex returned no JSON-parsable content")
        return {}
