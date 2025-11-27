# utils/validators.py
import os
import hashlib
from pathlib import Path
from typing import List, Set
from fastapi import UploadFile, HTTPException
from config.settings import settings
import io
import pandas as pd

# Use settings-defined allowed extensions
ALLOWED_EXTENSIONS = set(settings.allowed_file_extensions)

MALICIOUS_SIGNATURES = {
    b'\x4d\x5a',  # MZ
    b'\x50\x4b\x03\x04',  # PK (zip)
    b'\x89\x50\x4e\x47',  # PNG
}

class EnhancedFileValidator:
    def __init__(self):
        self.max_file_size = settings.max_file_size_mb * 1024 * 1024
        self.uploaded_hashes: Set[str] = set()
    
    async def validate_file(self, file: UploadFile) -> bool:
        if not file.filename:
            raise HTTPException(status_code=400, detail="No filename provided")
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(status_code=400, detail=f"File type {ext} not allowed. Allowed: {ALLOWED_EXTENSIONS}")

        # Read a bounded prefix to check signatures and content type without loading whole file
        await file.seek(0)
        prefix = await file.read(1024 * 64)  # read first 64KB for signature & header checks
        # Reset pointer for downstream stream
        try:
            await file.seek(0)
        except Exception:
            try:
                file.file.seek(0)
            except Exception:
                pass

        if len(prefix) == 0:
            raise HTTPException(status_code=400, detail="Empty file not allowed")

        # Check malicious signatures
        for signature in MALICIOUS_SIGNATURES:
            if prefix.startswith(signature):
                raise HTTPException(status_code=400, detail="Potentially malicious file detected")

        # If file is small (<= max_file_size * 2), read and compute hash; else compute rolling hash by reading in chunks
        # We'll compute MD5 by streaming in background if the caller later fully streams to disk.
        # For now, compute hash from prefix only (fast duplicate detection heuristic)
        file_hash = hashlib.md5(prefix).hexdigest()
        if file_hash in self.uploaded_hashes:
            raise HTTPException(status_code=400, detail="Duplicate file detected")
        self.uploaded_hashes.add(file_hash)

        # If CSV: try reading the first few rows only to detect structure (nrows)
        if ext == '.csv':
            try:
                # read a small chunk - uses pandas but bounded
                await file.seek(0)
                chunk = await file.read(1024 * 512)  # read 512KB
                await file.seek(0)
                df = pd.read_csv(io.BytesIO(chunk), nrows=50)
                if df.shape[1] == 0:
                    raise HTTPException(status_code=400, detail="CSV file has no columns")
            except pd.errors.EmptyDataError:
                raise HTTPException(status_code=400, detail="Empty CSV file")
            except Exception as e:
                raise HTTPException(status_code=400, detail=f"CSV validation failed: {str(e)}")

        return True

file_validator = EnhancedFileValidator()

def validate_preprocessing_params(missing_strategy: str, encoding: str, target_column: str = None) -> bool:
    allowed_missing = ["mean", "median", "mode", "drop"]
    if missing_strategy not in allowed_missing:
        raise HTTPException(status_code=422, detail={
            "error": "Invalid missing strategy",
            "allowed_values": allowed_missing,
            "received": missing_strategy
        })
    allowed_encoding = ["onehot", "label", "target", "kfold"]
    if encoding not in allowed_encoding:
        raise HTTPException(status_code=422, detail={
            "error": "Invalid encoding method",
            "allowed_values": allowed_encoding,
            "received": encoding
        })
    if encoding in ["target", "kfold"] and not target_column:
        raise HTTPException(status_code=422, detail={
            "error": f"Target column required for {encoding} encoding",
            "required_field": "target_column"
        })
    return True
