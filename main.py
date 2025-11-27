# main.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any
import anyio
import inspect
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
# slowapi imports are optional now — handled below
import structlog
import uvicorn
from pydantic import BaseModel
from config.settings import settings
from models.requests import PreprocessRequest, TrainRequest
from models.responses import UploadResponse, PreprocessResponse, TrainResponse
from services.file_service import FileService
from services.preprocessing_service import PreprocessingService
from services.ml_service import MLService, AsyncMLService
from utils.validators import file_validator, validate_preprocessing_params
from utils.exceptions import (
    MLPlatformException, mlplatform_exception_handler,
    general_exception_handler, DatasetError, ModelTrainingError, ValidationError, PreprocessingError
)
from app.redis_client import redis_client

# Logging
structlog.configure(processors=[structlog.stdlib.filter_by_level, structlog.stdlib.add_log_level,
                                 structlog.processors.TimeStamper(fmt="iso"), structlog.processors.JSONRenderer() ],
                    wrapper_class=structlog.stdlib.BoundLogger, logger_factory=structlog.stdlib.LoggerFactory())
logger = structlog.get_logger()

app = FastAPI(title="No-Code ML Platform API", description="Backend API for training ML models without code", version="1.0.0")

# Optional: rate limiting with slowapi
# slowapi is not included in requirements.txt by default to avoid redis version conflicts.
# If slowapi is installed in your environment and compatible, this will enable the handler.
try:
    import importlib
    slowapi_module = importlib.util.find_spec("slowapi")
    if slowapi_module is not None:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("slowapi loaded: rate limiting enabled")
    else:
        logger.warning("slowapi not installed — rate limiting disabled (safe to proceed)")
except Exception:
    logger.warning("slowapi not installed — rate limiting disabled (safe to proceed)")

# Platform-level exception handlers
app.add_exception_handler(MLPlatformException, mlplatform_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.add_middleware(CORSMiddleware, allow_origins=settings.cors_origins, allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

# services
file_service = FileService()
preprocessing_service = PreprocessingService()
ml_service = MLService()
async_ml_service = AsyncMLService()

# Ensure upload dir
os.makedirs(settings.upload_directory, exist_ok=True)
UPLOAD_DIR = Path(settings.upload_directory).resolve()
MAX_BYTES = getattr(settings, "max_file_size_mb", 100) * 1024 * 1024
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", 10))

# Helper: job keys in redis
def job_key(job_id: str) -> str:
    return f"job:{job_id}"

# Train job background function
async def train_job_async(preprocessed_file: str, task_type: str, model_type: str, target_column: Any, job_id: str):
    logger.info("Job started", job_id=job_id, file=preprocessed_file)
    await redis_client.hset(job_key(job_id), mapping={"status": "running", "started_at": anyio.current_time().__str__()})
    try:
        train_fn = async_ml_service.train_model_async
        if inspect.iscoroutinefunction(train_fn):
            result = await train_fn(file_path=preprocessed_file, task_type=task_type, model_type=model_type, target_column=target_column)
        else:
            result = await anyio.to_thread.run_sync(train_fn, preprocessed_file, task_type, model_type, target_column)
        # persist result
        await redis_client.hset(job_key(job_id), mapping={"status":"completed", "result": json.dumps(result)})
        logger.info("Job completed", job_id=job_id)
    except Exception as e:
        logger.error("Job failed", job_id=job_id, error=str(e))
        await redis_client.hset(job_key(job_id), mapping={"status":"failed", "error": str(e)})

# Pydantic predict request
class PredictRequest(BaseModel):
    inputs: Dict[str, Any]

# Endpoints
@app.post("/upload", response_model=UploadResponse)
async def upload_files(request: Request, files: List[UploadFile] = File(...)):
    try:
        if len(files) > MAX_FILES_PER_UPLOAD:
            raise HTTPException(status_code=400, detail=f"Max {MAX_FILES_PER_UPLOAD} files allowed per upload")
        results = {}
        for upload in files:
            original_name = Path(upload.filename).name
            logger.info("Processing file", filename=original_name)
            # Save + validate (streaming)
            stored_path = await file_service.save_uploaded_file(upload)
            # Analyze dataset (async)
            analysis = await file_service.analyze_dataset(stored_path)
            results[original_name] = analysis
        logger.info("Successfully processed files", count=len(files))
        return UploadResponse(message="Files uploaded successfully", files=results)
    except Exception as e:
        logger.error("Upload error", error=str(e))
        if isinstance(e, HTTPException):
            raise e
        raise DatasetError(f"Failed to process uploaded files: {str(e)}")

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(request: Request, files: List[UploadFile] = File(...), missing_strategy: str = Form(...),
                          scaling: bool = Form(...), encoding: str = Form(...), target_column: str = Form(None),
                          selected_features_json: str = Form(None)):
    try:
        validate_preprocessing_params(missing_strategy, encoding, target_column)
        selected_features_dict = {}
        if selected_features_json:
            try:
                selected_features_dict = json.loads(selected_features_json)
            except json.JSONDecodeError:
                raise ValidationError("Invalid selected_features_json format")
        results = {}
        for upload in files:
            original_name = Path(upload.filename).name
            logger.info("Preprocessing file", filename=original_name)
            # save
            saved_path = await file_service.save_uploaded_file(upload)
            selected_features = selected_features_dict.get(original_name)
            preprocessed_path = await anyio.to_thread.run_sync(preprocessing_service.preprocess_dataset,
                                                                saved_path, missing_strategy, scaling, encoding, target_column, selected_features)
            # Optionally register mapping in redis
            await redis_client.hset("file_metadata", Path(preprocessed_path).name, original_name)
            results[original_name] = {"preprocessed_file": preprocessed_path}
        logger.info("Successfully preprocessed files", count=len(files))
        return PreprocessResponse(message="Preprocessing completed", files=results)
    except Exception as e:
        logger.error("Preprocessing error", error=str(e))
        if isinstance(e, (HTTPException, MLPlatformException)):
            raise e
        raise PreprocessingError(f"Preprocessing failed: {str(e)}")

@app.post("/train")
async def train_models(request: Request, background_tasks: BackgroundTasks,
                       preprocessed_filenames: List[str] = Form(...), target_column: str = Form(None),
                       task_type: str = Form(...), model_type: str = Form(None)):
    try:
        target_columns = {}
        if target_column:
            try:
                target_columns = json.loads(target_column)
            except json.JSONDecodeError:
                target_columns = {}
        returned_jobs = []
        for preprocessed_file in preprocessed_filenames:
            resolved_path = Path(preprocessed_file)
            if not resolved_path.exists():
                # try resolve via redis file_metadata map
                mapped = await redis_client.hget("file_metadata", Path(preprocessed_file).name)
                if mapped:
                    # find stored name
                    resolved_path = Path(settings.upload_directory) / preprocessed_file
                else:
                    raise ModelTrainingError(f"Preprocessed file not found: {preprocessed_file}")
            filename = resolved_path.name
            if filename.startswith("preprocessed_"):
                filename_no_prefix = filename[len("preprocessed_"):]
            else:
                filename_no_prefix = filename
            file_target = target_columns.get(filename_no_prefix, target_column if isinstance(target_column, str) else None)
            job_id = uuid4().hex
            await redis_client.hset(job_key(job_id), mapping={"status":"pending", "file": str(resolved_path)})
            background_tasks.add_task(train_job_async, str(resolved_path), task_type, model_type, file_target, job_id)
            returned_jobs.append({"file": str(resolved_path), "job_id": job_id})
        logger.info("Training jobs scheduled", count=len(returned_jobs))
        return JSONResponse(content={"message":"Training started", "jobs": returned_jobs})
    except Exception as e:
        logger.error("Training scheduling error", error=str(e))
        if isinstance(e, (HTTPException, MLPlatformException)):
            raise e
        raise ModelTrainingError(f"Model training scheduling failed: {str(e)}")

@app.get("/jobs/{job_id}")
async def get_job_status(job_id: str):
    data = await redis_client.hgetall(job_key(job_id))
    if not data:
        raise HTTPException(status_code=404, detail="Job not found")
    # parse result if present
    if "result" in data:
        try:
            data["result"] = json.loads(data["result"])
        except Exception:
            pass
    return data

@app.post("/models/{model_id}/predict")
async def predict(request: Request, model_id: str, payload: PredictRequest = Body(...)):
    try:
        safe_model_id = Path(model_id).name
        model_filename = f"trained_model_{safe_model_id}.pkl"
        model_path = UPLOAD_DIR / model_filename
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model not found")
        logger.info("Prediction requested", model_id=safe_model_id, input_keys=list(payload.inputs.keys()))
        result = await async_ml_service.predict_async(model_path, payload.inputs)
        logger.info("Prediction completed", model_id=safe_model_id)
        return {"result": result}
    except Exception as e:
        logger.error("Prediction error", error=str(e))
        if isinstance(e, HTTPException):
            raise e
        raise ModelTrainingError(f"Prediction failed: {str(e)}")

@app.get("/download-model/{filename}")
async def download_model(filename: str):
    try:
        raw = Path(filename).name
        name_no_csv = raw.replace(".csv", "")
        model_candidate_name = f"trained_model_{name_no_csv}.pkl"
        model_path = UPLOAD_DIR / model_candidate_name
        if not model_path.exists():
            raise HTTPException(status_code=404, detail="Model file not found")
        return FileResponse(path=str(model_path), filename=model_path.name, media_type='application/octet-stream')
    except HTTPException:
        raise
    except Exception as e:
        logger.error("Model download error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/health")
async def health_check():
    return {"status":"healthy", "message":"No-Code ML Platform API is running", "version":"1.0.0", "timestamp": anyio.current_time().__str__()}

if __name__ == "__main__":
    uvicorn.run(app, host=settings.api_host, port=settings.api_port, reload=settings.debug)
