import time
from typing import Dict, Tuple
from fastapi import Request, HTTPException
import logging
from ....utils.config import config

logger = logging.getLogger(__name__)

class RateLimiter:
    """
    Rate limiting middleware to prevent API abuse
    """
    
    def __init__(self):
        self.requests: Dict[str, list] = {}
        self.rate_limits = config.get('api.rate_limits', {
            'default': (100, 3600),  # 100 requests per hour
            'predictions': (50, 3600),  # 50 predictions per hour
            'training': (5, 86400)  # 5 training jobs per day
        })
    
    async def __call__(self, request: Request, call_next):
        """Middleware to apply rate limiting"""
        
        # Skip rate limiting for health endpoints
        if request.url.path in ['/health', '/health/ready', '/health/live']:
            return await call_next(request)
        
        client_id = self._get_client_identifier(request)
        endpoint_type = self._get_endpoint_type(request.url.path)
        
        if not self.is_allowed(client_id, endpoint_type):
            logger.warning(f"Rate limit exceeded for {client_id} on {endpoint_type}")
            raise HTTPException(
                status_code=429,
                detail="Rate limit exceeded. Please try again later."
            )
        
        response = await call_next(request)
        
        # Add rate limit headers
        self.add_rate_limit_headers(response, client_id, endpoint_type)
        
        return response
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get unique identifier for client"""
        # Use API key if available, otherwise use IP address
        api_key = request.headers.get('X-API-Key')
        if api_key:
            return f"api_key:{api_key}"
        else:
            client_host = request.client.host
            return f"ip:{client_host}"
    
    def _get_endpoint_type(self, path: str) -> str:
        """Categorize endpoint for rate limiting"""
        if '/predict' in path:
            return 'predictions'
        elif '/models/train' in path:
            return 'training'
        else:
            return 'default'
    
    def is_allowed(self, client_id: str, endpoint_type: str) -> bool:
        """Check if request is within rate limits"""
        current_time = time.time()
        
        # Get rate limit for endpoint type
        max_requests, window_seconds = self.rate_limits.get(
            endpoint_type, self.rate_limits['default']
        )
        
        # Initialize or clean up client request history
        if client_id not in self.requests:
            self.requests[client_id] = []
        
        # Remove old requests outside the time window
        self.requests[client_id] = [
            req_time for req_time in self.requests[client_id]
            if current_time - req_time < window_seconds
        ]
        
        # Check if under limit
        if len(self.requests[client_id]) < max_requests:
            self.requests[client_id].append(current_time)
            return True
        else:
            return False
    
    def add_rate_limit_headers(self, response, client_id: str, endpoint_type: str):
        """Add rate limit headers to response"""
        current_time = time.time()
        max_requests, window_seconds = self.rate_limits.get(
            endpoint_type, self.rate_limits['default']
        )
        
        if client_id in self.requests:
            requests_in_window = [
                req_time for req_time in self.requests[client_id]
                if current_time - req_time < window_seconds
            ]
            remaining = max(0, max_requests - len(requests_in_window))
            
            # Find when the oldest request will expire
            if requests_in_window:
                reset_time = min(requests_in_window) + window_seconds
                reset_in = int(reset_time - current_time)
            else:
                reset_in = 0
        else:
            remaining = max_requests
            reset_in = window_seconds
        
        response.headers["X-RateLimit-Limit"] = str(max_requests)
        response.headers["X-RateLimit-Remaining"] = str(remaining)
        response.headers["X-RateLimit-Reset"] = str(reset_in)

# Global rate limiter instance
rate_limiter = RateLimiter()
