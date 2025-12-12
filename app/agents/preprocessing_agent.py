"""
PreprocessingAgent - Handles LLM-native data preprocessing.
"""
from typing import Dict, Any, Optional
import pandas as pd
import json
from pathlib import Path
from pydantic import BaseModel
from app.agents.base_agent import BaseAgent
from app.preprocessing import (
    clean_text,
    exact_dedup_dataframe,
    chunk_text,
    count_tokens,
    format_prompt
)
from app.storage.object_store import object_store
from app.utils.exceptions import PreprocessingException


class PreprocessingAgentInput(BaseModel):
    """Input schema for PreprocessingAgent."""
    run_id: str
    dataset_id: str
    local_path: str
    text_column: str
    format: str
    base_model: str  # For tokenizer
    config: Dict[str, Any] = {}  # Preprocessing configuration


class PreprocessingAgent(BaseAgent):
    """
    LLM-native preprocessing agent.
    
    Responsibilities:
    - Text normalization (unicode, whitespace)
    - Deduplication (if enabled)
    - Chunking (if texts exceed max length)
    - Tokenization with target model tokenizer
    - Apply instruction/chat templates
    - Save processed dataset to object storage
    """
    
    def __init__(self):
        super().__init__("PreprocessingAgent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute preprocessing pipeline.
        
        Input:
            {
                "run_id": "abc123",
                "dataset_id": "ds_xyz",
                "local_path": "/tmp/data.csv",
                "text_column": "text",
                "format": "csv",
                "base_model": "meta-llama/Llama-2-7b-hf",
                "config": {
                    "clean": true,
                    "dedup": true,
                    "chunk": false,
                    "max_length": 512,
                    "template": "alpaca",
                    "template_mapping": {...}
                }
            }
        
        Output:
            {
                "processed_dataset_path": "s3://datasets/processed_abc123.jsonl",
                "num_samples": 9500,
                "avg_tokens": 256,
                "processing_stats": {...}
            }
        """
        # Validate input
        validated = self.validate_input(input_data, PreprocessingAgentInput)
        run_id = validated.run_id
        config = validated.config
        
        await self._emit_log(run_id, "INFO", "Starting preprocessing",
                           dataset_id=validated.dataset_id)
        
        # Load dataset
        df = self._load_dataset(validated.local_path, validated.format)
        original_count = len(df)
        
        await self._emit_log(run_id, "INFO", f"Loaded {original_count} samples")
        
        # Step 1: Text cleaning
        if config.get("clean", True):
            await self._emit_log(run_id, "INFO", "Cleaning text")
            df[validated.text_column] = df[validated.text_column].apply(
                lambda x: clean_text(str(x), aggressive=config.get("aggressive_clean", False))
            )
        
        # Step 2: Deduplication
        if config.get("dedup", True):
            await self._emit_log(run_id, "INFO", "Removing duplicates")
            before_dedup = len(df)
            df = exact_dedup_dataframe(df, validated.text_column)
            after_dedup = len(df)
            removed = before_dedup - after_dedup
            await self._emit_log(run_id, "INFO", f"Removed {removed} duplicates")
        
        # Step 3: Apply prompt template (if specified)
        if config.get("template"):
            await self._emit_log(run_id, "INFO", f"Applying template: {config['template']}")
            df = self._apply_template(df, config)
        
        # Step 4: Chunking (if needed)
        if config.get("chunk", False):
            await self._emit_log(run_id, "INFO", "Chunking long texts")
            df = self._chunk_dataset(df, validated.text_column, config)
        
        # Step 5: Token counting
        await self._emit_log(run_id, "INFO", "Counting tokens")
        token_counts = []
        for text in df[validated.text_column]:
            try:
                count = count_tokens(str(text), validated.base_model)
                token_counts.append(count)
            except Exception:
                token_counts.append(0)
        
        df['token_count'] = token_counts
        avg_tokens = sum(token_counts) / len(token_counts) if token_counts else 0
        
        # Step 6: Save processed dataset
        await self._emit_log(run_id, "INFO", "Saving processed dataset")
        processed_path = await self._save_processed_dataset(df, run_id, validated.text_column)
        
        # Compute stats
        processing_stats = {
            "original_samples": original_count,
            "final_samples": len(df),
            "removed_samples": original_count - len(df),
            "avg_tokens": round(avg_tokens, 2),
            "min_tokens": min(token_counts) if token_counts else 0,
            "max_tokens": max(token_counts) if token_counts else 0
        }
        
        await self._emit_log(run_id, "INFO", "Preprocessing completed",
                           final_samples=len(df),
                           avg_tokens=round(avg_tokens, 2))
        
        return {
            "processed_dataset_path": processed_path,
            "num_samples": len(df),
            "avg_tokens": round(avg_tokens, 2),
            "processing_stats": processing_stats
        }
    
    def _load_dataset(self, file_path: str, file_format: str) -> pd.DataFrame:
        """Load dataset."""
        if file_format == "csv":
            return pd.read_csv(file_path)
        elif file_format == "json":
            return pd.read_json(file_path)
        elif file_format == "jsonl":
            return pd.read_json(file_path, lines=True)
        elif file_format == "parquet":
            return pd.read_parquet(file_path)
        else:
            raise PreprocessingException(f"Unsupported format: {file_format}")
    
    def _apply_template(self, df: pd.DataFrame, config: Dict[str, Any]) -> pd.DataFrame:
        """Apply prompt template to dataset."""
        template_name = config["template"]
        field_mapping = config.get("template_mapping", {})
        
        # Convert DataFrame rows to dicts
        data = df.to_dict('records')
        
        # Apply template
        from app.preprocessing.prompt_formatting import apply_template_to_dataset
        formatted_texts = apply_template_to_dataset(data, template_name, field_mapping)
        
        # Create new DataFrame with formatted text
        return pd.DataFrame({"text": formatted_texts})
    
    def _chunk_dataset(self, df: pd.DataFrame, text_column: str, config: Dict[str, Any]) -> pd.DataFrame:
        """Chunk long texts in dataset."""
        max_length = config.get("max_length", 512)
        overlap = config.get("chunk_overlap", 50)
        
        chunked_rows = []
        
        for _, row in df.iterrows():
            text = str(row[text_column])
            chunks = chunk_text(text, max_length=max_length, overlap=overlap)
            
            for chunk in chunks:
                new_row = row.copy()
                new_row[text_column] = chunk["text"]
                chunked_rows.append(new_row)
        
        return pd.DataFrame(chunked_rows)
    
    async def _save_processed_dataset(self, df: pd.DataFrame, run_id: str, text_column: str) -> str:
        """Save processed dataset to object storage."""
        # Save as JSONL for LLM training
        temp_path = Path(f"/tmp/processed_{run_id}.jsonl")
        
        # Convert to JSONL format
        df.to_json(temp_path, orient='records', lines=True)
        
        # Upload to object storage
        object_name = f"{run_id}/processed_dataset.jsonl"
        s3_path = object_store.upload_file(
            str(temp_path),
            object_name,
            bucket_type='datasets',
            content_type='application/jsonl'
        )
        
        # Clean up temp file
        temp_path.unlink()
        
        return s3_path
