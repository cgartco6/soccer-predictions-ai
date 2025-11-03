from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import logging
from typing import Dict, Any

from .routes import predictions, models, health
from ..utils.config import config
from ..utils.logger import setup_logging

# Setup logging
setup_logging(
    level=config.get('logging.level', 'INFO'),
    json_format=config.get('logging.json_format', False)
)

logger = logging.getLogger(__name__)

# Create FastAPI application
app = FastAPI(
    title="AI Soccer Predictions API",
    description="Advanced soccer predictions using synthetic and strategic intelligence",
    version="1.0.0",
    docs_url="/docs" if config.get('api.enable_docs', True) else None,
    redoc_url="/redoc" if config.get('api.enable_docs', True) else None
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=config.get('api.cors_origins', ["*"]),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include routers
app.include_router(predictions.router, prefix="/api/v1", tags=["predictions"])
app.include_router(models.router, prefix="/api/v1", tags=["models"])
app.include_router(health.router, prefix="/api/v1", tags=["health"])

@app.on_event("startup")
async def startup_event():
    """Actions to perform on application startup"""
    logger.info("Starting AI Soccer Predictions API")
    
    # Initialize AI agents and models
    try:
        from ..agents import SyntheticIntelligenceAgent, StrategicIntelligenceAgent
        from ..agents import DataCollectorAgent, PredictionEnsembleAgent
        
        # Initialize agents (these would be managed properly in a real application)
        logger.info("Initializing AI agents...")
        
    except Exception as e:
        logger.error(f"Error during startup: {e}")

@app.on_event("shutdown")
async def shutdown_event():
    """Actions to perform on application shutdown"""
    logger.info("Shutting down AI Soccer Predictions API")

@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    """Global exception handler"""
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "message": str(exc),
            "request_id": getattr(request.state, 'request_id', 'unknown')
        }
    )

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "AI Soccer Predictions API",
        "version": "1.0.0",
        "status": "operational"
    }

# Additional middleware for request logging
@app.middleware("http")
async def log_requests(request, call_next):
    """Middleware for logging requests"""
    request_id = request.headers.get('X-Request-ID', 'unknown')
    
    logger.info(f"Request started: {request.method} {request.url.path} - ID: {request_id}")
    
    response = await call_next(request)
    
    logger.info(f"Request completed: {request.method} {request.url.path} - Status: {response.status_code}")
    
    return response
