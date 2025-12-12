"""
Main FastAPI application.
Entry point for the LLM fine-tuning platform API.
"""
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.utils.config import settings
from app.utils.logging import get_logger
from app.utils.exceptions import AgentException
from app.infra.redis import redis_client
from app.storage.object_store import object_store

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    logger.info("Starting LLM Fine-Tuning Platform API", version="1.0.0")
    
    # Initialize Redis connection
    try:
        await redis_client.connect()
        await redis_client.ping()
        logger.info("Redis connection established")
    except Exception as e:
        logger.error("Failed to connect to Redis", error=str(e))
    
    # Initialize object storage buckets
    try:
        object_store._ensure_buckets()
        logger.info("Object storage initialized")
    except Exception as e:
        logger.error("Failed to initialize object storage", error=str(e))
    
    logger.info("Application startup complete")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application")
    await redis_client.disconnect()
    logger.info("Application shutdown complete")


# Create FastAPI app
app = FastAPI(
    title="LLM Fine-Tuning Platform API",
    description="Agent-based backend for drag-and-drop LLM fine-tuning pipelines",
    version="1.0.0",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


# Exception handlers
@app.exception_handler(AgentException)
async def agent_exception_handler(request, exc: AgentException):
    """Handle agent exceptions."""
    logger.error("Agent exception", error=exc.to_dict())
    return HTTPException(
        status_code=400,
        detail=exc.to_dict()
    )


@app.exception_handler(Exception)
async def general_exception_handler(request, exc: Exception):
    """Handle general exceptions."""
    logger.error("Unhandled exception", error=str(exc))
    return HTTPException(
        status_code=500,
        detail={"error": "INTERNAL_ERROR", "message": str(exc)}
    )


# Health check endpoint
@app.get("/health")
async def health_check():
    """Health check endpoint."""
    try:
        # Check Redis
        redis_healthy = await redis_client.ping()
        
        return {
            "status": "healthy",
            "version": "1.0.0",
            "services": {
                "redis": "healthy" if redis_healthy else "unhealthy",
                "api": "healthy"
            }
        }
    except Exception as e:
        logger.error("Health check failed", error=str(e))
        return {
            "status": "unhealthy",
            "error": str(e)
        }


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "message": "LLM Fine-Tuning Platform API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


# Import and include routers (will be created next)
# from app.api.routes import jobs, datasets, models, logs
# app.include_router(jobs.router, prefix="/api/v1/jobs", tags=["jobs"])
# app.include_router(datasets.router, prefix="/api/v1/datasets", tags=["datasets"])
# app.include_router(models.router, prefix="/api/v1/models", tags=["models"])
# app.include_router(logs.router, prefix="/api/v1/logs", tags=["logs"])


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "app.main:app",
        host=settings.api_host,
        port=settings.api_port,
        reload=settings.debug
    )
