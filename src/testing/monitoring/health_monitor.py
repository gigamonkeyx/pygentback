"""
Recipe Health Monitor

Monitors the health and performance of recipe execution systems,
providing real-time health checks and alerting capabilities.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum

logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status levels"""
    HEALTHY = "healthy"
    WARNING = "warning"
    CRITICAL = "critical"
    UNKNOWN = "unknown"


@dataclass
class HealthMetrics:
    """Health metrics for monitoring"""
    recipe_id: str
    status: HealthStatus
    success_rate: float
    average_execution_time: float
    error_count: int
    last_execution: Optional[datetime]
    uptime_percentage: float
    resource_usage: Dict[str, float] = field(default_factory=dict)


class RecipeHealthMonitor:
    """
    Recipe Health Monitoring System.
    
    Provides continuous monitoring of recipe health and performance
    with alerting and reporting capabilities.
    """
    
    def __init__(self, check_interval: float = 60.0):
        self.check_interval = check_interval
        self.monitored_recipes: Dict[str, HealthMetrics] = {}
        self.monitoring_active = False
        self.monitoring_task: Optional[asyncio.Task] = None
    
    async def start_monitoring(self):
        """Start health monitoring"""
        if self.monitoring_active:
            return
        
        self.monitoring_active = True
        self.monitoring_task = asyncio.create_task(self._monitoring_loop())
        logger.info("Started recipe health monitoring")
    
    async def stop_monitoring(self):
        """Stop health monitoring"""
        self.monitoring_active = False
        if self.monitoring_task:
            self.monitoring_task.cancel()
        logger.info("Stopped recipe health monitoring")
    
    async def _monitoring_loop(self):
        """Main monitoring loop"""
        while self.monitoring_active:
            try:
                await self._check_all_recipes()
                await asyncio.sleep(self.check_interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Monitoring loop error: {e}")
    
    async def _check_all_recipes(self):
        """Check health of all monitored recipes"""
        for recipe_id in list(self.monitored_recipes.keys()):
            await self._check_recipe_health(recipe_id)
    
    async def _check_recipe_health(self, recipe_id: str):
        """Check health of a specific recipe"""
        # Mock health check implementation
        metrics = self.monitored_recipes.get(recipe_id)
        if metrics:
            # Update health status based on metrics
            if metrics.success_rate > 0.9 and metrics.average_execution_time < 60:
                metrics.status = HealthStatus.HEALTHY
            elif metrics.success_rate > 0.7:
                metrics.status = HealthStatus.WARNING
            else:
                metrics.status = HealthStatus.CRITICAL
    
    def add_recipe(self, recipe_id: str):
        """Add a recipe to monitoring"""
        self.monitored_recipes[recipe_id] = HealthMetrics(
            recipe_id=recipe_id,
            status=HealthStatus.UNKNOWN,
            success_rate=0.0,
            average_execution_time=0.0,
            error_count=0,
            last_execution=None,
            uptime_percentage=0.0
        )
        logger.info(f"Added recipe {recipe_id} to health monitoring")
    
    def get_health_status(self, recipe_id: str) -> Optional[HealthMetrics]:
        """Get health status for a recipe"""
        return self.monitored_recipes.get(recipe_id)
    
    def get_all_health_status(self) -> Dict[str, HealthMetrics]:
        """Get health status for all monitored recipes"""
        return self.monitored_recipes.copy()
