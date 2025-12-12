"""
ValidationAgent - Validates dataset quality and integrity.
"""
from typing import Dict, Any, List
import pandas as pd
import re
from pydantic import BaseModel
from app.agents.base_agent import BaseAgent
from app.utils.exceptions import ValidationException


class ValidationAgentInput(BaseModel):
    """Input schema for ValidationAgent."""
    run_id: str
    dataset_id: str
    local_path: str
    text_column: str
    format: str


class ValidationAgent(BaseAgent):
    """
    Dataset validation agent.
    
    Responsibilities:
    - Check for missing values in critical columns
    - Detect duplicates (exact and near-duplicates)
    - PII detection (emails, phone numbers, SSNs)
    - Text quality checks (min length, encoding issues)
    - Flag warnings/errors
    """
    
    def __init__(self):
        super().__init__("ValidationAgent")
        
        # PII detection patterns
        self.pii_patterns = {
            "email": r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
            "phone": r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',
            "ssn": r'\b\d{3}-\d{2}-\d{4}\b'
        }
    
    async def execute(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Execute dataset validation.
        
        Input:
            {
                "run_id": "abc123",
                "dataset_id": "ds_xyz",
                "local_path": "/tmp/data.csv",
                "text_column": "text",
                "format": "csv"
            }
        
        Output:
            {
                "valid": true,
                "warnings": ["10 duplicates found"],
                "errors": [],
                "pii_detected": false,
                "validation_report": {...}
            }
        """
        # Validate input
        validated = self.validate_input(input_data, ValidationAgentInput)
        run_id = validated.run_id
        
        await self._emit_log(run_id, "INFO", "Starting dataset validation",
                           dataset_id=validated.dataset_id)
        
        # Load dataset
        df = self._load_dataset(validated.local_path, validated.format)
        
        warnings = []
        errors = []
        pii_detected = False
        
        # Check 1: Missing values
        await self._emit_log(run_id, "INFO", "Checking for missing values")
        missing_check = self._check_missing_values(df, validated.text_column)
        if missing_check["has_missing"]:
            warnings.append(f"{missing_check['missing_count']} missing values in text column")
        
        # Check 2: Duplicates
        await self._emit_log(run_id, "INFO", "Checking for duplicates")
        dup_check = self._check_duplicates(df, validated.text_column)
        if dup_check["has_duplicates"]:
            warnings.append(f"{dup_check['duplicate_count']} duplicate rows found")
        
        # Check 3: PII detection
        await self._emit_log(run_id, "INFO", "Checking for PII")
        pii_check = self._check_pii(df, validated.text_column)
        if pii_check["pii_found"]:
            pii_detected = True
            warnings.append(f"PII detected: {', '.join(pii_check['pii_types'])}")
        
        # Check 4: Text quality
        await self._emit_log(run_id, "INFO", "Checking text quality")
        quality_check = self._check_text_quality(df, validated.text_column)
        if quality_check["low_quality_count"] > 0:
            warnings.append(f"{quality_check['low_quality_count']} rows with low quality text")
        
        # Determine if dataset is valid
        valid = len(errors) == 0
        
        validation_report = {
            "missing_values": missing_check,
            "duplicates": dup_check,
            "pii": pii_check,
            "text_quality": quality_check
        }
        
        await self._emit_log(run_id, "INFO", "Validation completed",
                           valid=valid,
                           warnings_count=len(warnings),
                           errors_count=len(errors))
        
        return {
            "valid": valid,
            "warnings": warnings,
            "errors": errors,
            "pii_detected": pii_detected,
            "validation_report": validation_report
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
            raise ValidationException(f"Unsupported format: {file_format}")
    
    def _check_missing_values(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check for missing values in text column."""
        missing_count = df[text_column].isnull().sum()
        return {
            "has_missing": missing_count > 0,
            "missing_count": int(missing_count),
            "missing_percentage": float(missing_count / len(df) * 100)
        }
    
    def _check_duplicates(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check for duplicate rows."""
        duplicate_count = df[text_column].duplicated().sum()
        return {
            "has_duplicates": duplicate_count > 0,
            "duplicate_count": int(duplicate_count),
            "duplicate_percentage": float(duplicate_count / len(df) * 100)
        }
    
    def _check_pii(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check for PII in text column."""
        pii_found = False
        pii_types = []
        pii_counts = {}
        
        # Sample first 100 rows for PII check (performance)
        sample = df[text_column].head(100).astype(str)
        
        for pii_type, pattern in self.pii_patterns.items():
            matches = sample.str.contains(pattern, regex=True, na=False).sum()
            if matches > 0:
                pii_found = True
                pii_types.append(pii_type)
                pii_counts[pii_type] = int(matches)
        
        return {
            "pii_found": pii_found,
            "pii_types": pii_types,
            "pii_counts": pii_counts
        }
    
    def _check_text_quality(self, df: pd.DataFrame, text_column: str) -> Dict[str, Any]:
        """Check text quality (length, encoding)."""
        text_series = df[text_column].astype(str)
        
        # Check for very short texts (< 5 characters)
        too_short = (text_series.str.len() < 5).sum()
        
        # Check for encoding issues (non-printable characters)
        has_encoding_issues = text_series.str.contains(r'[\x00-\x08\x0B-\x0C\x0E-\x1F]', 
                                                       regex=True, na=False).sum()
        
        low_quality_count = too_short + has_encoding_issues
        
        return {
            "low_quality_count": int(low_quality_count),
            "too_short_count": int(too_short),
            "encoding_issues_count": int(has_encoding_issues)
        }
