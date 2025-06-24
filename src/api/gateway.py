#!/usr/bin/env python3
"""
Production API Gateway

Enterprise-grade API gateway with authentication, authorization, rate limiting,
request/response transformation, and comprehensive security features.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response, HTTPException, Depends, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.responses import JSONResponse
import uvicorn

from ..cache.rate_limiter import rate_limiter
from ..cache.session_manager import session_manager
from ..cache.cache_layers import cache_manager
from ..database.production_manager import db_manager

logger = logging.getLogger(__name__)


class APIGatewayConfig:
    """API Gateway configuration"""
    
    def __init__(self):
        # Server configuration
        self.host = "0.0.0.0"
        self.port = 8000
        self.workers = 4
        self.reload = False
        
        # Security configuration
        self.allowed_hosts = ["*"]  # Configure for production
        self.cors_origins = ["*"]   # Configure for production
        self.cors_methods = ["GET", "POST", "PUT", "DELETE", "OPTIONS"]
        self.cors_headers = ["*"]
        
        # Rate limiting configuration
        self.enable_rate_limiting = True
        self.default_rate_limit = "1000/hour"
        
        # Authentication configuration
        self.jwt_secret_key = "your-super-secret-jwt-key-change-in-production"
        self.jwt_algorithm = "HS256"
        self.jwt_expiration_hours = 24
        
        # Request/Response configuration
        self.enable_compression = True
        self.max_request_size = 10 * 1024 * 1024  # 10MB
        self.request_timeout = 30  # seconds
        
        # Monitoring configuration
        self.enable_metrics = True
        self.enable_logging = True
        self.log_requests = True
        self.log_responses = False  # Set to True for debugging


class APIGateway:
    """Production API Gateway with comprehensive features"""
    
    def __init__(self, config: Optional[APIGatewayConfig] = None):
        self.config = config or APIGatewayConfig()
        self.app = FastAPI(
            title="PyGent Factory API Gateway",
            description="Enterprise-grade API gateway for PyGent Factory",
            version="1.0.0",
            docs_url="/docs",
            redoc_url="/redoc"
        )
        
        # Performance metrics
        self.metrics = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "rate_limited_requests": 0,
            "authentication_failures": 0,
            "average_response_time": 0.0,
            "start_time": datetime.utcnow()
        }
        
        # Security components
        self.security = HTTPBearer(auto_error=False)
        
        # Initialize middleware and routes
        self._setup_middleware()
        self._setup_routes()
        self._setup_error_handlers()
    
    def _setup_middleware(self):
        """Setup API gateway middleware"""
        
        # CORS middleware
        self.app.add_middleware(
            CORSMiddleware,
            allow_origins=self.config.cors_origins,
            allow_credentials=True,
            allow_methods=self.config.cors_methods,
            allow_headers=self.config.cors_headers,
        )
        
        # Trusted host middleware
        self.app.add_middleware(
            TrustedHostMiddleware,
            allowed_hosts=self.config.allowed_hosts
        )
        
        # Compression middleware
        if self.config.enable_compression:
            self.app.add_middleware(GZipMiddleware, minimum_size=1000)
        
        # Custom middleware
        @self.app.middleware("http")
        async def gateway_middleware(request: Request, call_next):
            return await self._process_request(request, call_next)
    
    async def _process_request(self, request: Request, call_next):
        """Process incoming requests through the gateway"""
        start_time = time.time()
        
        try:
            # Update metrics
            self.metrics["total_requests"] += 1
            
            # Log request if enabled
            if self.config.log_requests:
                logger.info(f"Request: {request.method} {request.url.path}")
            
            # Check rate limiting
            if self.config.enable_rate_limiting:
                rate_limit_result = await self._check_rate_limit(request)
                if not rate_limit_result["allowed"]:
                    self.metrics["rate_limited_requests"] += 1
                    return JSONResponse(
                        status_code=429,
                        content={
                            "error": "Rate limit exceeded",
                            "retry_after": rate_limit_result["retry_after"],
                            "limit": rate_limit_result["limit"]
                        },
                        headers={
                            "Retry-After": str(rate_limit_result["retry_after"]),
                            "X-RateLimit-Limit": str(rate_limit_result["limit"]),
                            "X-RateLimit-Remaining": str(rate_limit_result["remaining"])
                        }
                    )
            
            # Process request
            response = await call_next(request)
            
            # Update metrics
            processing_time = time.time() - start_time
            self._update_response_metrics(response.status_code, processing_time)
            
            # Add response headers
            response.headers["X-Response-Time"] = f"{processing_time:.3f}s"
            response.headers["X-Gateway-Version"] = "1.0.0"
            
            return response
            
        except Exception as e:
            # Handle errors
            self.metrics["failed_requests"] += 1
            logger.error(f"Gateway error: {e}")
            
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal gateway error",
                    "message": str(e) if self.config.reload else "Internal server error"
                }
            )
    
    async def _check_rate_limit(self, request: Request) -> Dict[str, Any]:
        """Check rate limiting for request"""
        try:
            # Get client identifier
            client_id = self._get_client_identifier(request)
            
            # Determine rate limit rule based on endpoint
            rule_name = self._get_rate_limit_rule(request)
            
            # Check rate limit
            result = await rate_limiter.check_rate_limit(rule_name, client_id)
            
            return {
                "allowed": result.allowed,
                "remaining": result.remaining_requests,
                "retry_after": result.retry_after_seconds,
                "limit": result.limit
            }
            
        except Exception as e:
            logger.error(f"Rate limit check failed: {e}")
            # Fail open - allow request if rate limiting fails
            return {"allowed": True, "remaining": 999, "retry_after": 0, "limit": 1000}
    
    def _get_client_identifier(self, request: Request) -> str:
        """Get client identifier for rate limiting"""
        # Try to get user ID from authentication
        auth_header = request.headers.get("Authorization")
        if auth_header:
            try:
                # Extract user ID from JWT token (implement JWT validation)
                user_id = self._extract_user_id_from_token(auth_header)
                if user_id:
                    return f"user:{user_id}"
            except Exception:
                pass
        
        # Fall back to IP address
        client_ip = request.client.host
        forwarded_for = request.headers.get("X-Forwarded-For")
        if forwarded_for:
            client_ip = forwarded_for.split(",")[0].strip()
        
        return f"ip:{client_ip}"
    
    def _get_rate_limit_rule(self, request: Request) -> str:
        """Determine rate limit rule based on request"""
        path = request.url.path
        
        # Authentication endpoints
        if "/auth/" in path:
            return "api_auth"
        
        # Model inference endpoints
        if "/models/" in path or "/inference/" in path:
            return "model_inference"
        
        # File upload endpoints
        if "/upload/" in path:
            return "file_upload"
        
        # WebSocket endpoints
        if "/ws/" in path:
            return "websocket"
        
        # Default API rate limit
        return "api_general"
    
    def _extract_user_id_from_token(self, auth_header: str) -> Optional[str]:
        """Extract user ID from JWT token"""
        try:
            # Remove "Bearer " prefix
            token = auth_header.replace("Bearer ", "")
            
            # Implement JWT token validation and user ID extraction
            # This is a placeholder - implement proper JWT validation
            return None
            
        except Exception as e:
            logger.error(f"Token extraction failed: {e}")
            return None
    
    def _update_response_metrics(self, status_code: int, processing_time: float):
        """Update response metrics"""
        if 200 <= status_code < 400:
            self.metrics["successful_requests"] += 1
        else:
            self.metrics["failed_requests"] += 1
        
        # Update average response time
        total_requests = self.metrics["total_requests"]
        current_avg = self.metrics["average_response_time"]
        self.metrics["average_response_time"] = (
            (current_avg * (total_requests - 1) + processing_time) / total_requests
        )
    
    def _setup_routes(self):
        """Setup API gateway routes"""

        # Include authentication endpoints
        from .auth_endpoints import router as auth_router
        self.app.include_router(auth_router)

        # Include Redis monitoring endpoints
        from .redis_monitoring import router as redis_router
        self.app.include_router(redis_router)

        # Include GPU monitoring endpoints
        from .gpu_monitoring import router as gpu_router
        self.app.include_router(gpu_router)

        @self.app.get("/")
        async def root():
            """API Gateway root endpoint"""
            return {
                "service": "PyGent Factory API Gateway",
                "version": "1.0.0",
                "status": "operational",
                "timestamp": datetime.utcnow().isoformat(),
                "features": {
                    "authentication": "JWT with RBAC",
                    "caching": "Redis multi-layer",
                    "rate_limiting": "Advanced algorithms",
                    "gpu_optimization": "RTX 3080 optimized",
                    "database": "PostgreSQL production-ready"
                }
            }
        
        @self.app.get("/health")
        async def health_check():
            """Comprehensive health check"""
            try:
                health_status = {
                    "status": "healthy",
                    "timestamp": datetime.utcnow().isoformat(),
                    "uptime_seconds": (datetime.utcnow() - self.metrics["start_time"]).total_seconds(),
                    "components": {}
                }
                
                # Check database health
                try:
                    db_health = await db_manager.health_check()
                    health_status["components"]["database"] = {
                        "status": "healthy" if db_health.get("status") == "healthy" else "unhealthy",
                        "response_time_ms": db_health.get("response_time_ms", 0)
                    }
                except Exception as e:
                    health_status["components"]["database"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                
                # Check cache health
                try:
                    cache_stats = await cache_manager.get_cache_statistics()
                    health_status["components"]["cache"] = {
                        "status": "healthy" if cache_stats else "unhealthy",
                        "hit_rate": cache_stats.get("cache_layers", {}).get("db", {}).get("hit_rate_percent", 0)
                    }
                except Exception as e:
                    health_status["components"]["cache"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
                
                # Check rate limiter health
                health_status["components"]["rate_limiter"] = {
                    "status": "healthy" if rate_limiter.is_initialized else "unhealthy"
                }
                
                # Determine overall status
                component_statuses = [comp.get("status") for comp in health_status["components"].values()]
                if "unhealthy" in component_statuses:
                    health_status["status"] = "degraded"
                
                return health_status
                
            except Exception as e:
                logger.error(f"Health check failed: {e}")
                return {
                    "status": "unhealthy",
                    "error": str(e),
                    "timestamp": datetime.utcnow().isoformat()
                }
        
        @self.app.get("/metrics")
        async def get_metrics():
            """Get API gateway metrics"""
            if not self.config.enable_metrics:
                raise HTTPException(status_code=404, detail="Metrics disabled")
            
            uptime = (datetime.utcnow() - self.metrics["start_time"]).total_seconds()
            
            return {
                **self.metrics,
                "uptime_seconds": uptime,
                "requests_per_second": self.metrics["total_requests"] / max(uptime, 1),
                "success_rate_percent": (
                    self.metrics["successful_requests"] / max(self.metrics["total_requests"], 1) * 100
                ),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    def _setup_error_handlers(self):
        """Setup custom error handlers"""
        
        @self.app.exception_handler(HTTPException)
        async def http_exception_handler(request: Request, exc: HTTPException):
            """Handle HTTP exceptions"""
            return JSONResponse(
                status_code=exc.status_code,
                content={
                    "error": exc.detail,
                    "status_code": exc.status_code,
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
        
        @self.app.exception_handler(Exception)
        async def general_exception_handler(request: Request, exc: Exception):
            """Handle general exceptions"""
            logger.error(f"Unhandled exception: {exc}")
            return JSONResponse(
                status_code=500,
                content={
                    "error": "Internal server error",
                    "message": str(exc) if self.config.reload else "An unexpected error occurred",
                    "timestamp": datetime.utcnow().isoformat()
                }
            )
    
    async def initialize(self) -> bool:
        """Initialize API gateway"""
        try:
            logger.info("Initializing API gateway...")
            
            # Initialize dependencies
            if not rate_limiter.is_initialized:
                await rate_limiter.initialize()
            
            if not session_manager.is_initialized:
                await session_manager.initialize()
            
            if not cache_manager.is_initialized:
                await cache_manager.initialize()
            
            logger.info("API gateway initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize API gateway: {e}")
            return False
    
    def run(self):
        """Run the API gateway"""
        uvicorn.run(
            self.app,
            host=self.config.host,
            port=self.config.port,
            workers=1 if self.config.reload else self.config.workers,
            reload=self.config.reload,
            access_log=self.config.log_requests
        )


# Global API gateway instance
api_gateway = APIGateway()
