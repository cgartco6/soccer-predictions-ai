"""
API module for soccer predictions system
"""

from .main import app
from .routes.predictions import router as predictions_router
from .routes.models import router as models_router
from .routes.health import router as health_router

__all__ = [
    'app',
    'predictions_router',
    'models_router',
    'health_router'
]
