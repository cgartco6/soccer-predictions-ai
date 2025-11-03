from fastapi import Request, HTTPException
from fastapi.security import APIKeyHeader
from typing import Optional
import logging
from ....utils.config import config

logger = logging.getLogger(__name__)

api_key_header = APIKeyHeader(name="X-API-Key")

class AuthMiddleware:
    """
    Authentication middleware for API endpoints
    """
    
    def __init__(self):
        self.valid_api_keys = self._load_api_keys()
    
    def _load_api_keys(self) -> set:
        """Load valid API keys from configuration"""
        api_keys = config.get('api.auth.api_keys', [])
        return set(api_keys)
    
    async def __call__(self, request: Request, call_next):
        """Middleware to authenticate requests"""
        
        # Skip authentication for health endpoints
        if request.url.path in ['/health', '/health/ready', '/health/live']:
            return await call_next(request)
        
        # Skip authentication for docs in development
        if config.is_development and request.url.path in ['/docs', '/redoc', '/openapi.json']:
            return await call_next(request)
        
        # Extract API key
        api_key = request.headers.get('X-API-Key')
        
        if not api_key:
            raise HTTPException(
                status_code=401, 
                detail="API key required"
            )
        
        if api_key not in self.valid_api_keys:
            logger.warning(f"Invalid API key attempt from {request.client.host}")
            raise HTTPException(
                status_code=401, 
                detail="Invalid API key"
            )
        
        # Add user context to request state
        request.state.user = self._get_user_from_api_key(api_key)
        
        response = await call_next(request)
        return response
    
    def _get_user_from_api_key(self, api_key: str) -> dict:
        """Get user information from API key"""
        # In production, this would query a user database
        # For now, return basic user info
        return {
            'user_id': 'api_user',
            'role': 'api_client',
            'permissions': ['read_predictions', 'make_predictions']
        }

# Authentication dependency for endpoints
async def get_current_user(request: Request):
    """Dependency to get current user from request"""
    if not hasattr(request.state, 'user'):
        raise HTTPException(status_code=401, detail="Not authenticated")
    return request.state.user

async def require_permission(permission: str):
    """Dependency to require specific permission"""
    async def permission_dependency(user: dict = Depends(get_current_user)):
        if permission not in user.get('permissions', []):
            raise HTTPException(
                status_code=403, 
                detail=f"Permission denied: {permission}"
            )
        return user
    return permission_dependency
