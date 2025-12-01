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

        # sensible defaults
        self._http_timeout = 60  # seconds for generic provider
        self._vertex_timeout = 120

    async def analyze_dataset(self, summary: Dict[str, Any], sample_rows: str) -> Dict[str, Any]:
        """
        Returns LLM suggestions or {} on any failure.
        This ensures LLM is optional and doesn't break the upload flow.
        """
        prompt = self._build_prompt(summary, sample_rows)
        logger.info("LLM analyze dataset", provider=self.provider)

        try:
            # if there's no api_url configured, skip (unless using vertex)
            if not self.api_url and self.provider != "vertex":
                logger.info("No LLM API URL configured; skipping LLM call")
                return {}

            if self.provider == "vertex":
                return await self._call_vertex_predict(prompt)
            else:
                return await self._call_generic_http(prompt)
        except Exception as e:
            logger.warning("LLM analyze_dataset failed — continuing without LLM", error=str(e))
            return {}

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
            logger.info("LLM API URL not configured; skipping LLM call")
            return {}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        payload = {"prompt": prompt, "max_tokens": 512, "temperature": 0.0}

        try:
            async with httpx.AsyncClient(timeout=self._http_timeout) as client:
                r = await client.post(self.api_url, headers=headers, json=payload)
                r.raise_for_status()
                # prefer text and attempt to parse JSON strictly
                text = r.text
        except httpx.HTTPStatusError as e:
            logger.warning("LLM returned non-200 status", status=e.response.status_code, url=str(e.request.url))
            return {}
        except Exception as e:
            logger.warning("LLM generic HTTP call failed", error=str(e))
            return {}

        # Try to extract JSON text from common shapes
        try:
            # Many providers return {"text": "..."} or {"output": "..."} - try to parse top-level JSON first
            parsed_top = None
            try:
                parsed_top = json.loads(text)
            except Exception:
                parsed_top = None

            if isinstance(parsed_top, dict):
                # prefer fields that usually contain the model output
                for k in ("text", "generated_text", "output", "content", "result"):
                    if k in parsed_top:
                        candidate = parsed_top[k]
                        # if it's already structured JSON
                        if isinstance(candidate, dict):
                            return candidate
                        if isinstance(candidate, str):
                            try:
                                return json.loads(candidate)
                            except Exception:
                                # try to extract JSON substring
                                pass
                # if parsed_top looks like final result, return it
                # but ensure it contains expected keys; otherwise fall through to fallback below
                if set(parsed_top.keys()) & {"suggested_task", "suggested_target", "missing_value_strategy"}:
                    return parsed_top

            # If we didn't return above, try to find a JSON substring in the text body
            # find first '{' and last '}' and attempt parsing that slice (robust heuristic)
            start = text.find("{")
            end = text.rfind("}")
            if start != -1 and end != -1 and end > start:
                snippet = text[start:end+1]
                try:
                    return json.loads(snippet)
                except Exception:
                    pass

            # final fallback: return the raw text under 'raw'
            return {"raw": text}
        except Exception as e:
            logger.warning("Failed parsing LLM response", error=str(e))
            return {"raw": text}

    async def _call_vertex_predict(self, prompt: str) -> Dict[str, Any]:
        if not self.google_project or not self.vertex_model_id:
            logger.warning("Vertex config missing; skipping Vertex call")
            return {}

        url = f"https://{self.google_location}-aiplatform.googleapis.com/v1/projects/{self.google_project}/locations/{self.google_location}/models/{self.vertex_model_id}:predict"
        body = {"instances": [{"content": prompt}], "parameters": {"temperature": 0.0, "maxOutputTokens": 800}}
        headers = {"Content-Type": "application/json"}
        params = {}
        if self.api_key:
            params["key"] = self.api_key

        try:
            async with httpx.AsyncClient(timeout=self._vertex_timeout) as client:
                r = await client.post(url, headers=headers, params=params, json=body)
                r.raise_for_status()
                resp_text = r.text
        except httpx.HTTPStatusError as e:
            logger.warning("Vertex returned non-200", status=e.response.status_code)
            return {}
        except Exception as e:
            logger.warning("Vertex call failed", error=str(e))
            return {}

        # Try parsing Vertex's response for JSON content
        try:
            resp = json.loads(resp_text)
        except Exception:
            resp = {}

        candidates = []
        if "predictions" in resp and len(resp["predictions"]) > 0:
            pred0 = resp["predictions"][0]
            for key in ("content", "text", "output", "response"):
                if key in pred0:
                    candidates.append(pred0[key])

        for c in candidates:
            try:
                if isinstance(c, dict):
                    return c
                if isinstance(c, str):
                    return json.loads(c)
            except Exception:
                continue

        logger.warning("Vertex returned no JSON-parsable content; returning empty dict")
        return {}
