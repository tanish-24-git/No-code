# main.py
import os
import json
from pathlib import Path
from typing import List, Dict, Any, Optional
import anyio
import inspect
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Request, BackgroundTasks, Body
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
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
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_log_level,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.JSONRenderer()
    ],
    wrapper_class=structlog.stdlib.BoundLogger,
    logger_factory=structlog.stdlib.LoggerFactory()
)
logger = structlog.get_logger()

app = FastAPI(
    title="No-Code ML Platform API",
    description="Backend API for training ML models without code",
    version="1.0.0"
)

# Optional: rate limiting with slowapi (safe if not installed)
try:
    import importlib
    slowapi_module = importlib.util.find_spec("slowapi")
    if slowapi_module is not None:
        from slowapi import _rate_limit_exceeded_handler
        from slowapi.errors import RateLimitExceeded
        app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
        logger.info("slowapi loaded: rate limiting enabled")
    else:
        logger.debug("slowapi not installed — rate limiting disabled")
except Exception:
    logger.debug("slowapi not installed — rate limiting disabled (safe to proceed)")

# Platform-level exception handlers
app.add_exception_handler(MLPlatformException, mlplatform_exception_handler)
app.add_exception_handler(Exception, general_exception_handler)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)

# services
file_service = FileService()
preprocessing_service = PreprocessingService()
ml_service = MLService()
async_ml_service = AsyncMLService()

# Ensure upload dir
Path(settings.upload_directory).mkdir(parents=True, exist_ok=True)
UPLOAD_DIR = Path(settings.upload_directory).resolve()
MAX_BYTES = getattr(settings, "max_file_size_mb", 100) * 1024 * 1024
MAX_FILES_PER_UPLOAD = int(os.getenv("MAX_FILES_PER_UPLOAD", 10))

# Helper: job keys in redis
def job_key(job_id: str) -> str:
    return f"job:{job_id}"

# Startup / shutdown events
@app.on_event("startup")
async def on_startup():
    logger.info("Application starting", host=settings.api_host, port=settings.api_port)
    try:
        Path(settings.upload_directory).mkdir(parents=True, exist_ok=True)
        logger.info("Upload directory ready", upload_dir=str(Path(settings.upload_directory).resolve()))
    except Exception as e:
        logger.error("Failed to ensure upload directory", error=str(e))
    # Redis health (non-fatal)
    try:
        pong = await redis_client.ping()
        logger.info("Redis ping success", pong=pong)
    except Exception as e:
        logger.warning("Redis ping failed at startup (will attempt reconnects during runtime)", error=str(e))

@app.on_event("shutdown")
async def on_shutdown():
    logger.info("Shutting down application")

# Train job background function (defensive + logs)
async def train_job_async(preprocessed_file: str, task_type: str, model_type: str, target_column: Any, job_id: str):
    logger.info("Job started", job_id=job_id, file=preprocessed_file)
    try:
        await redis_client.hset(job_key(job_id), mapping={"status": "running", "started_at": anyio.current_time().__str__()})
    except Exception as e:
        logger.warning("Could not set redis job metadata (non-fatal)", error=str(e))

    try:
        train_fn = async_ml_service.train_model_async
        if inspect.iscoroutinefunction(train_fn):
            result = await train_fn(file_path=preprocessed_file, task_type=task_type, model_type=model_type, target_column=target_column)
        else:
            result = await anyio.to_thread.run_sync(train_fn, preprocessed_file, task_type, model_type, target_column)

        # persist result
        try:
            await redis_client.hset(job_key(job_id), mapping={"status":"completed", "result": json.dumps(result)})
        except Exception as e:
            logger.warning("Failed to persist job result to redis (non-fatal)", error=str(e))
        logger.info("Job completed", job_id=job_id)
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        logger.error("Job failed", job_id=job_id, error=str(e), traceback=tb)
        try:
            await redis_client.hset(job_key(job_id), mapping={"status":"failed", "error": str(e)})
        except Exception:
            logger.warning("Failed to write job failure to redis")

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
            # Analyze dataset (async) -> returns heuristics + optionally llm suggestions
            analysis = await file_service.analyze_dataset(stored_path)
            # Return analysis which includes "llm_suggestions" and we also include suggested column actions if available
            results[original_name] = analysis
        logger.info("Successfully processed files", count=len(files))
        return UploadResponse(message="Files uploaded successfully", files=results)
    except Exception as e:
        logger.error("Upload error", error=str(e))
        if isinstance(e, HTTPException):
            raise e
        raise DatasetError(f"Failed to process uploaded files: {str(e)}")

@app.post("/preprocess", response_model=PreprocessResponse)
async def preprocess_data(
    request: Request,
    files: Optional[List[UploadFile]] = File(None),
    preprocessing_plan: Optional[str] = Form(None),  # JSON string when using multipart/form-data
    body_plan: Optional[Dict[str, Any]] = Body(None)  # JSON body alternative
):
    """
    Accepts either:
     - multipart form with files + preprocessing_plan (JSON string in form), OR
     - JSON body with {"files": ["uploaded_filename.csv"], "plan": {...}} where files are names already uploaded.
    """
    try:
        # harmonize plan
        plan = {}
        if preprocessing_plan:
            try:
                plan = json.loads(preprocessing_plan)
            except Exception:
                raise ValidationError("Invalid preprocessing_plan JSON")
        elif body_plan:
            # If JSON passed in body, it may contain the plan directly or a wrapper.
            # Accept either {"plan": {...}, "files": [...]} or {...} as the plan.
            if "plan" in body_plan and isinstance(body_plan["plan"], dict):
                plan = body_plan["plan"]
            else:
                plan = body_plan

        results = {}
        if files:
            # If files uploaded now, process them
            for upload in files:
                original_name = Path(upload.filename).name
                logger.info("Preprocessing uploaded file", filename=original_name)
                saved_path = await file_service.save_uploaded_file(upload)
                # Call preprocess with plan (blocking on thread to avoid CPU blocking the event loop)
                preprocessed_path = await anyio.to_thread.run_sync(preprocessing_service.preprocess_dataset,
                                                                    saved_path, plan)
                # Save plan sidecar path is handled by preprocess_dataset
                # Optionally register mapping in redis
                try:
                    await redis_client.hset("file_metadata", Path(preprocessed_path).name, original_name)
                except Exception as e:
                    logger.warning("Could not register file metadata in redis (non-fatal)", error=str(e))
                results[original_name] = {"preprocessed_file": preprocessed_path}
        else:
            # No files in multipart — maybe the user sent paths in the body_plan
            files_list = body_plan.get("files") if isinstance(body_plan, dict) else None
            if not files_list or not isinstance(files_list, list):
                raise ValidationError("No files provided for preprocessing")
            for server_filename in files_list:
                # Resolve to upload directory
                candidate = Path(settings.upload_directory) / Path(server_filename).name
                if not candidate.exists():
                    raise ValidationError(f"Referenced file not found on server: {server_filename}")
                preprocessed_path = await anyio.to_thread.run_sync(preprocessing_service.preprocess_dataset,
                                                                    str(candidate), plan)
                try:
                    await redis_client.hset("file_metadata", Path(preprocessed_path).name, candidate.name)
                except Exception:
                    pass
                results[candidate.name] = {"preprocessed_file": preprocessed_path}

        logger.info("Successfully preprocessed files", count=len(results))
        return PreprocessResponse(message="Preprocessing completed", files=results)
    except Exception as e:
        logger.error("Preprocessing error", error=str(e))
        if isinstance(e, (HTTPException, MLPlatformException)):
            raise e
        raise PreprocessingError(f"Preprocessing failed: {str(e)}")

@app.post("/train")
async def train_models(request: Request, background_tasks: BackgroundTasks,
                       preprocessed_filenames: List[str] = Form(None),  # form-list style
                       body_json: Optional[Dict[str, Any]] = Body(None),  # JSON alternative
                       target_column: str = Form(None),
                       task_type: str = Form(...), model_type: str = Form(None)):
    """
    Accepts either:
     - form fields: preprocessed_filenames repeated, OR
     - JSON body: { "preprocessed_filenames": ["a","b"], "target_column": "...", "task_type":"...", "model_type":"..." }
    """
    try:
        # harmonize JSON body if provided
        if (not preprocessed_filenames or len(preprocessed_filenames) == 0) and body_json:
            preprocessed_filenames = body_json.get("preprocessed_filenames", [])
            target_column = target_column or body_json.get("target_column")
            task_type = body_json.get("task_type", task_type)
            model_type = body_json.get("model_type", model_type)

        returned_jobs = []
        for preprocessed_file in preprocessed_filenames:
            resolved_path = Path(preprocessed_file)
            # if not absolute/existing, try inside upload directory
            if not resolved_path.exists():
                candidate = Path(settings.upload_directory) / preprocessed_file
                if candidate.exists():
                    resolved_path = candidate
                else:
                    candidate2 = Path(preprocessed_file).name
                    candidate3 = Path(settings.upload_directory) / candidate2
                    if candidate3.exists():
                        resolved_path = candidate3
                    else:
                        # try to look up mapping in redis
                        mapped = await redis_client.hget("file_metadata", Path(preprocessed_file).name)
                        if mapped:
                            resolved_path = Path(settings.upload_directory) / mapped
                        else:
                            raise ModelTrainingError(f"Preprocessed file not found: {preprocessed_file}")

            filename = resolved_path.name
            if filename.startswith("preprocessed_"):
                filename_no_prefix = filename[len("preprocessed_"):]
            else:
                filename_no_prefix = filename
            file_target = None
            if target_column:
                try:
                    # target_column might be JSON map in some clients
                    parsed = json.loads(target_column)
                    file_target = parsed.get(filename_no_prefix)
                except Exception:
                    file_target = target_column
            job_id = uuid4().hex
            try:
                await redis_client.hset(job_key(job_id), mapping={"status":"pending", "file": str(resolved_path)})
            except Exception as e:
                logger.warning("Could not write job pending status to redis (non-fatal)", error=str(e))
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
    """
    Accept model_id either with or without extension (e.g., preprocessed_small OR preprocessed_small.csv).
    Prediction aligns inputs to model metadata if available.
    """
    try:
        # sanitize model_id and strip common extensions
        safe_model_id = Path(model_id).stem
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
