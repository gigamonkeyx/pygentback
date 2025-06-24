"""
Health Checker

Provides health check functionality for system components.
"""

import logging
import asyncio
from typing import Dict, Any, List
from datetime import datetime
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class HealthCheckResult:
    """Result of a health check"""
    component: str
    status: str  # "healthy", "degraded", "failed"
    message: str
    timestamp: datetime
    response_time_ms: float = 0.0
    details: Dict[str, Any] = None


class HealthChecker:
    """Performs health checks on system components."""
    
    def __init__(self):
        self.check_registry: Dict[str, callable] = {}
        logger.info("Health Checker initialized")
    
    def register_check(self, name: str, check_function: callable):
        """Register a health check function."""
        self.check_registry[name] = check_function
        logger.info(f"Registered health check: {name}")
    
    async def run_check(self, name: str) -> HealthCheckResult:
        """Run a specific health check."""
        if name not in self.check_registry:
            return HealthCheckResult(
                component=name,
                status="failed",
                message=f"Health check '{name}' not registered",
                timestamp=datetime.now()
            )
        
        start_time = datetime.now()
        
        try:
            check_function = self.check_registry[name]
            
            if asyncio.iscoroutinefunction(check_function):
                result = await check_function()
            else:
                result = check_function()
            
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            if isinstance(result, dict):
                return HealthCheckResult(
                    component=name,
                    status=result.get('status', 'unknown'),
                    message=result.get('message', 'Health check completed'),
                    timestamp=start_time,
                    response_time_ms=response_time,
                    details=result.get('details', {})
                )
            else:
                return HealthCheckResult(
                    component=name,
                    status="healthy" if result else "failed",
                    message="Health check completed",
                    timestamp=start_time,
                    response_time_ms=response_time
                )
                
        except Exception as e:
            end_time = datetime.now()
            response_time = (end_time - start_time).total_seconds() * 1000
            
            return HealthCheckResult(
                component=name,
                status="failed",
                message=f"Health check failed: {str(e)}",
                timestamp=start_time,
                response_time_ms=response_time
            )
    
    async def run_all_checks(self) -> List[HealthCheckResult]:
        """Run all registered health checks."""
        results = []
        
        for name in self.check_registry.keys():
            result = await self.run_check(name)
            results.append(result)
        
        return results
    
    def get_registered_checks(self) -> List[str]:
        """Get list of registered health check names."""
        return list(self.check_registry.keys())
