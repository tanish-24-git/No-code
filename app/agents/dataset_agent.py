"""
DatasetAgent - Handles dataset ingestion and analysis.
"""
from typing import Dict, Any, Optional
from pathlib import Path
import pandas as pd
from pydantic import BaseModel
from app.agents.base_agent import BaseAgent
from app.storage.object_store import object_store
from app.utils.exceptions import DatasetException


class DatasetAgentInput(BaseModel):
    """Input schema for DatasetAgent."""
    run_id: str
    file_path: str  # S3 path or local path
    format: Optional[str] = None  # csv, json, jsonl, txt, parquet


class DatasetAgent(BaseAgent):
    """
    Dataset ingestion and analysis agent.
    
    Responsibilities:
    - Download file from object storage (if S3 path)
    - Infer schema (columns, types, row count)
    - Compute basic statistics
    - Detect text columns for LLM training
    - Store metadata
    """
    
    def __init__(self):
        super().__init__("DatasetAgent")
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute dataset ingestion and analysis.
        
        Input:
            {
                "run_id": "abc123",
                "file_path": "s3://datasets/upload.csv",
                "format": "csv"
            }
        
        Output:
            {
                "dataset_id": "ds_xyz",
                "rows": 10000,
                "columns": ["text", "label"],
                "text_column": "text",
                "format": "csv",
                "stats": {...}
            }
        """
        # Validate input
        validated = self.validate_input(input_data, DatasetAgentInput)
        run_id = validated.run_id
        
        await self._emit_log(run_id, "INFO", "Starting dataset ingestion",
                           file_path=validated.file_path)
        
        # Download file if S3 path
        local_path = await self._get_local_path(validated.file_path, run_id)
        
        # Infer format if not provided
        file_format = validated.format or self._infer_format(local_path)
        
        await self._emit_log(run_id, "INFO", f"Detected format: {file_format}")
        
        # Load and analyze dataset
        df = self._load_dataset(local_path, file_format)
        
        await self._emit_log(run_id, "INFO", f"Dataset loaded: {len(df)} rows, {len(df.columns)} columns")
        
        # Analyze schema and content
        analysis = self._analyze_dataset(df, run_id)
        
        # Generate dataset ID
        dataset_id = f"ds_{run_id[:8]}"
        
        result = {
            "dataset_id": dataset_id,
            "rows": len(df),
            "columns": list(df.columns),
            "format": file_format,
            "local_path": str(local_path),
            **analysis
        }
        
        await self._emit_log(run_id, "INFO", "Dataset analysis completed",
                           dataset_id=dataset_id,
                           rows=result["rows"])
        
        return result
    
    async def _get_local_path(self, file_path: str, run_id: str) -> Path:
        """Download file from S3 if needed, otherwise return local path."""
        if file_path.startswith("s3://"):
            # Parse S3 path
            s3_path = file_path.replace("s3://", "")
            parts = s3_path.split("/", 1)
            bucket_type = parts[0]
            object_name = parts[1] if len(parts) > 1 else ""
            
            # Download to temp location
            local_path = Path(f"/tmp/{run_id}_{Path(object_name).name}")
            
            await self._emit_log(run_id, "INFO", "Downloading from object storage",
                               s3_path=file_path)
            
            object_store.download_file(object_name, str(local_path), bucket_type=bucket_type)
            
            return local_path
        else:
            return Path(file_path)
    
    def _infer_format(self, file_path: Path) -> str:
        """Infer file format from extension."""
        suffix = file_path.suffix.lower()
        format_map = {
            ".csv": "csv",
            ".json": "json",
            ".jsonl": "jsonl",
            ".txt": "txt",
            ".parquet": "parquet"
        }
        return format_map.get(suffix, "txt")
    
    def _load_dataset(self, file_path: Path, file_format: str) -> pd.DataFrame:
        """Load dataset into pandas DataFrame."""
        try:
            if file_format == "csv":
                return pd.read_csv(file_path)
            elif file_format == "json":
                return pd.read_json(file_path)
            elif file_format == "jsonl":
                return pd.read_json(file_path, lines=True)
            elif file_format == "parquet":
                return pd.read_parquet(file_path)
            elif file_format == "txt":
                # Load as single column
                with open(file_path, 'r', encoding='utf-8') as f:
                    lines = f.readlines()
                return pd.DataFrame({"text": lines})
            else:
                raise DatasetException(f"Unsupported format: {file_format}")
        
        except Exception as e:
            raise DatasetException(f"Failed to load dataset: {str(e)}")
    
    def _analyze_dataset(self, df: pd.DataFrame, run_id: str) -> Dict[str, Any]:
        """Analyze dataset schema and content."""
        analysis = {}
        
        # Detect text columns (for LLM training)
        text_columns = []
        for col in df.columns:
            if df[col].dtype == 'object':  # String columns
                # Check if column contains substantial text
                avg_length = df[col].astype(str).str.len().mean()
                if avg_length > 10:  # Arbitrary threshold
                    text_columns.append(col)
        
        # Select primary text column (longest average length)
        if text_columns:
            text_col = max(text_columns, 
                          key=lambda c: df[c].astype(str).str.len().mean())
            analysis["text_column"] = text_col
            analysis["text_columns"] = text_columns
        else:
            analysis["text_column"] = None
            analysis["text_columns"] = []
        
        # Compute statistics
        stats = {
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()},
            "missing_values": df.isnull().sum().to_dict(),
            "unique_values": {col: int(df[col].nunique()) for col in df.columns}
        }
        
        # Text statistics
        if analysis["text_column"]:
            text_col = analysis["text_column"]
            stats["text_stats"] = {
                "avg_length": float(df[text_col].astype(str).str.len().mean()),
                "min_length": int(df[text_col].astype(str).str.len().min()),
                "max_length": int(df[text_col].astype(str).str.len().max())
            }
        
        analysis["stats"] = stats
        
        return analysis
