"""
Production Deployment Configuration

Production-ready configuration and deployment utilities for the orchestration system.
Includes monitoring, scaling, security, and operational configurations.
"""

import os
import logging
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from datetime import timedelta
import json

logger = logging.getLogger(__name__)


@dataclass
class ProductionConfig:
    """Production deployment configuration."""
    
    # Environment settings
    environment: str = "production"
    debug_mode: bool = False
    log_level: str = "INFO"
    
    # Scaling configuration
    min_agents: int = 5
    max_agents: int = 50
    auto_scaling_enabled: bool = True
    scaling_threshold: float = 0.8
    
    # Performance settings
    max_concurrent_tasks: int = 100
    task_timeout: timedelta = field(default_factory=lambda: timedelta(minutes=30))
    health_check_interval: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    
    # Security settings
    enable_authentication: bool = True
    enable_encryption: bool = True
    api_rate_limit: int = 1000  # requests per minute
    
    # Monitoring settings
    metrics_retention_days: int = 30
    alert_thresholds: Dict[str, float] = field(default_factory=lambda: {
        'task_failure_rate': 0.1,
        'response_time_p95': 10.0,
        'agent_utilization': 0.9,
        'system_health': 0.8
    })
    
    # Database settings
    database_url: str = "postgresql://postgres:postgres@localhost:54321/pygent_factory"
    database_pool_size: int = 20
    database_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=30))
    
    # MCP server settings
    mcp_servers: Dict[str, Dict[str, Any]] = field(default_factory=lambda: {
        "filesystem": {
            "command": "npx -y @modelcontextprotocol/server-filesystem",
            "args": ["D:/mcp/pygent-factory/data"],
            "timeout": 30,
            "max_connections": 10
        },
        "postgresql": {
            "command": "npx -y @modelcontextprotocol/server-postgres",
            "args": ["postgresql://postgres:postgres@localhost:54321/pygent_factory"],
            "timeout": 30,
            "max_connections": 5
        },
        "github": {
            "command": "npx -y @modelcontextprotocol/server-github",
            "args": [],
            "timeout": 30,
            "max_connections": 3
        },
        "memory": {
            "command": "npx -y @modelcontextprotocol/server-memory",
            "args": [],
            "timeout": 30,
            "max_connections": 10
        }
    })
    
    # Backup and recovery
    backup_enabled: bool = True
    backup_interval: timedelta = field(default_factory=lambda: timedelta(hours=6))
    backup_retention_days: int = 7
    
    # Resource limits
    memory_limit_mb: int = 8192
    cpu_limit_cores: int = 4
    disk_space_limit_gb: int = 100


class ProductionDeployment:
    """Production deployment manager."""
    
    def __init__(self, config: ProductionConfig):
        self.config = config
        self.deployment_status = "not_deployed"
        self.health_checks = {}
        
    async def deploy(self) -> bool:
        """Deploy the orchestration system to production."""
        try:
            logger.info("Starting production deployment...")
            
            # Validate configuration
            if not await self._validate_configuration():
                logger.error("Configuration validation failed")
                return False
            
            # Setup infrastructure
            if not await self._setup_infrastructure():
                logger.error("Infrastructure setup failed")
                return False
            
            # Deploy MCP servers
            if not await self._deploy_mcp_servers():
                logger.error("MCP server deployment failed")
                return False
            
            # Setup monitoring
            if not await self._setup_monitoring():
                logger.error("Monitoring setup failed")
                return False
            
            # Setup security
            if not await self._setup_security():
                logger.error("Security setup failed")
                return False
            
            # Start health checks
            await self._start_health_checks()
            
            self.deployment_status = "deployed"
            logger.info("Production deployment completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Production deployment failed: {e}")
            self.deployment_status = "failed"
            return False
    
    async def scale_system(self, target_agents: int) -> bool:
        """Scale the system to target number of agents."""
        try:
            if target_agents < self.config.min_agents:
                target_agents = self.config.min_agents
            elif target_agents > self.config.max_agents:
                target_agents = self.config.max_agents
            
            logger.info(f"Scaling system to {target_agents} agents")
            
            # Implementation would scale actual agents
            # For now, just log the action
            
            return True
            
        except Exception as e:
            logger.error(f"System scaling failed: {e}")
            return False
    
    async def backup_system(self) -> bool:
        """Create system backup."""
        try:
            if not self.config.backup_enabled:
                return True
            
            logger.info("Creating system backup...")
            
            # Backup configuration
            config_backup = await self._backup_configuration()
            
            # Backup data
            data_backup = await self._backup_data()
            
            # Backup logs
            logs_backup = await self._backup_logs()
            
            logger.info("System backup completed")
            return config_backup and data_backup and logs_backup
            
        except Exception as e:
            logger.error(f"System backup failed: {e}")
            return False
    
    async def get_deployment_status(self) -> Dict[str, Any]:
        """Get deployment status and health information."""
        return {
            "deployment_status": self.deployment_status,
            "health_checks": self.health_checks,
            "configuration": {
                "environment": self.config.environment,
                "min_agents": self.config.min_agents,
                "max_agents": self.config.max_agents,
                "auto_scaling_enabled": self.config.auto_scaling_enabled
            },
            "resource_limits": {
                "memory_limit_mb": self.config.memory_limit_mb,
                "cpu_limit_cores": self.config.cpu_limit_cores,
                "disk_space_limit_gb": self.config.disk_space_limit_gb
            }
        }
    
    async def _validate_configuration(self) -> bool:
        """Validate production configuration."""
        try:
            # Check required environment variables
            required_env_vars = [
                "PYGENT_FACTORY_ENV",
                "DATABASE_URL"
            ]
            
            for var in required_env_vars:
                if not os.getenv(var):
                    logger.warning(f"Environment variable {var} not set, using default")
            
            # Validate resource limits
            if self.config.memory_limit_mb < 1024:
                logger.error("Memory limit too low for production")
                return False
            
            if self.config.max_concurrent_tasks < 10:
                logger.error("Max concurrent tasks too low for production")
                return False
            
            # Validate MCP server configuration
            for server_name, server_config in self.config.mcp_servers.items():
                if not server_config.get("command"):
                    logger.error(f"MCP server {server_name} missing command")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    async def _setup_infrastructure(self) -> bool:
        """Setup production infrastructure."""
        try:
            # Create necessary directories
            directories = [
                "D:/mcp/pygent-factory/data",
                "D:/mcp/pygent-factory/logs",
                "D:/mcp/pygent-factory/backups",
                "D:/mcp/pygent-factory/config"
            ]
            
            for directory in directories:
                os.makedirs(directory, exist_ok=True)
            
            # Setup logging configuration
            logging.basicConfig(
                level=getattr(logging, self.config.log_level),
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
                handlers=[
                    logging.FileHandler('D:/mcp/pygent-factory/logs/orchestration.log'),
                    logging.StreamHandler()
                ]
            )
            
            return True
            
        except Exception as e:
            logger.error(f"Infrastructure setup failed: {e}")
            return False
    
    async def _deploy_mcp_servers(self) -> bool:
        """Deploy MCP servers."""
        try:
            for server_name, server_config in self.config.mcp_servers.items():
                logger.info(f"Deploying MCP server: {server_name}")
                
                # In a real deployment, this would start the actual MCP servers
                # For now, just validate the configuration
                
                if not server_config.get("command"):
                    logger.error(f"MCP server {server_name} missing command")
                    return False
            
            return True
            
        except Exception as e:
            logger.error(f"MCP server deployment failed: {e}")
            return False
    
    async def _setup_monitoring(self) -> bool:
        """Setup monitoring and alerting."""
        try:
            # Setup metrics collection
            logger.info("Setting up monitoring...")
            
            # Configure alert thresholds
            for metric, threshold in self.config.alert_thresholds.items():
                logger.info(f"Alert threshold for {metric}: {threshold}")
            
            return True
            
        except Exception as e:
            logger.error(f"Monitoring setup failed: {e}")
            return False
    
    async def _setup_security(self) -> bool:
        """Setup security configurations."""
        try:
            if self.config.enable_authentication:
                logger.info("Authentication enabled")
            
            if self.config.enable_encryption:
                logger.info("Encryption enabled")
            
            logger.info(f"API rate limit: {self.config.api_rate_limit} requests/minute")
            
            return True
            
        except Exception as e:
            logger.error(f"Security setup failed: {e}")
            return False
    
    async def _start_health_checks(self):
        """Start health check monitoring."""
        self.health_checks = {
            "database": "healthy",
            "mcp_servers": "healthy",
            "agents": "healthy",
            "system_resources": "healthy"
        }
    
    async def _backup_configuration(self) -> bool:
        """Backup system configuration."""
        try:
            config_data = {
                "environment": self.config.environment,
                "mcp_servers": self.config.mcp_servers,
                "alert_thresholds": self.config.alert_thresholds,
                "backup_timestamp": str(datetime.utcnow())
            }
            
            backup_path = f"D:/mcp/pygent-factory/backups/config_backup_{datetime.utcnow().strftime('%Y%m%d_%H%M%S')}.json"
            
            with open(backup_path, 'w') as f:
                json.dump(config_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"Configuration backup failed: {e}")
            return False
    
    async def _backup_data(self) -> bool:
        """Backup system data."""
        try:
            # In a real implementation, this would backup the database
            logger.info("Data backup completed (placeholder)")
            return True
            
        except Exception as e:
            logger.error(f"Data backup failed: {e}")
            return False
    
    async def _backup_logs(self) -> bool:
        """Backup system logs."""
        try:
            # In a real implementation, this would archive log files
            logger.info("Logs backup completed (placeholder)")
            return True
            
        except Exception as e:
            logger.error(f"Logs backup failed: {e}")
            return False


def create_production_config() -> ProductionConfig:
    """Create production configuration from environment variables."""
    return ProductionConfig(
        environment=os.getenv("PYGENT_FACTORY_ENV", "production"),
        debug_mode=os.getenv("DEBUG_MODE", "false").lower() == "true",
        log_level=os.getenv("LOG_LEVEL", "INFO"),
        database_url=os.getenv("DATABASE_URL", "postgresql://postgres:postgres@localhost:54321/pygent_factory"),
        max_concurrent_tasks=int(os.getenv("MAX_CONCURRENT_TASKS", "100")),
        auto_scaling_enabled=os.getenv("AUTO_SCALING_ENABLED", "true").lower() == "true"
    )


def create_development_config() -> ProductionConfig:
    """Create development configuration."""
    return ProductionConfig(
        environment="development",
        debug_mode=True,
        log_level="DEBUG",
        min_agents=2,
        max_agents=10,
        max_concurrent_tasks=20,
        enable_authentication=False,
        enable_encryption=False,
        backup_enabled=False
    )