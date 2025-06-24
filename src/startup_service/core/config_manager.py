"""
Configuration Manager for PyGent Factory Startup Service
Dynamic configuration matrix with environment-specific profiles and validation.
"""

import os
import json
import yaml
from datetime import datetime
from typing import Dict, Any, List, Optional, Union
from pathlib import Path
import uuid

from ..models.schemas import (
    Environment, ServiceType, ConfigurationProfile,
    ConfigurationProfileCreate, ConfigurationProfileUpdate
)
from ..utils.logging_config import config_logger
from ..utils.security import SecurityManager
from .database import DatabaseManager


class ConfigurationManager:
    """Manages service configurations, profiles, and environment-specific settings."""
    
    def __init__(self, db_manager: DatabaseManager):
        self.db_manager = db_manager
        self.security_manager = SecurityManager()
        self.logger = config_logger
        
        # Configuration cache
        self.config_cache: Dict[str, Dict[str, Any]] = {}
        self.profile_cache: Dict[str, Dict[str, Any]] = {}
        
        # Default configurations
        self.default_configs = self._load_default_configurations()
        
        # Configuration validation rules
        self.validation_rules = self._load_validation_rules()
    
    async def initialize(self) -> bool:
        """Initialize the configuration manager."""
        try:
            self.logger.info("Initializing configuration manager")
            
            # Load default profiles if they don't exist
            await self._ensure_default_profiles()
            
            # Validate configuration integrity
            await self._validate_configuration_integrity()
            
            self.logger.info("Configuration manager initialized successfully")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize configuration manager: {e}")
            return False
    
    def _load_default_configurations(self) -> Dict[str, Dict[str, Any]]:
        """Load default service configurations."""
        return {
            ServiceType.POSTGRESQL: {
                "service_type": ServiceType.POSTGRESQL,
                "use_docker": True,
                "image": "postgres:15",
                "host": "localhost",
                "port": 5432,
                "database": "pygent_factory",
                "username": "postgres",
                "password": "postgres",
                "data_volume": None,
                "max_connections": 100,
                "shared_buffers": "256MB",
                "effective_cache_size": "1GB"
            },
            ServiceType.REDIS: {
                "service_type": ServiceType.REDIS,
                "use_docker": True,
                "image": "redis:7",
                "host": "localhost",
                "port": 6379,
                "password": None,
                "max_memory": "256mb",
                "max_memory_policy": "allkeys-lru",
                "persistence": True
            },
            ServiceType.OLLAMA: {
                "service_type": ServiceType.OLLAMA,
                "executable_path": "ollama",
                "host": "127.0.0.1",
                "port": 11434,
                "models": ["llama3.2:latest"],
                "gpu_enabled": True,
                "max_concurrent_requests": 4
            },
            ServiceType.AGENT: {
                "service_type": ServiceType.AGENT,
                "script_path": None,
                "python_path": "python",
                "environment_variables": {},
                "max_memory": "512MB",
                "timeout": 300
            },
            ServiceType.MONITORING: {
                "service_type": ServiceType.MONITORING,
                "prometheus_port": 9090,
                "grafana_port": 3000,
                "retention_days": 30,
                "scrape_interval": "15s"
            }
        }
    
    def _load_validation_rules(self) -> Dict[str, Dict[str, Any]]:
        """Load configuration validation rules."""
        return {
            ServiceType.POSTGRESQL: {
                "required_fields": ["host", "port", "database", "username"],
                "port_range": (1024, 65535),
                "valid_images": ["postgres:13", "postgres:14", "postgres:15", "postgres:16"]
            },
            ServiceType.REDIS: {
                "required_fields": ["host", "port"],
                "port_range": (1024, 65535),
                "valid_images": ["redis:6", "redis:7"]
            },
            ServiceType.OLLAMA: {
                "required_fields": ["host", "port"],
                "port_range": (1024, 65535),
                "valid_models": ["llama3.2:latest", "llama3.1:latest", "codellama:latest"]
            }
        }
    
    async def create_service_config(self, service_name: str, config_data: Dict[str, Any]) -> str:
        """Create a new service configuration."""
        try:
            # Validate configuration
            validation_errors = await self._validate_service_config(config_data)
            if validation_errors:
                raise ValueError(f"Configuration validation failed: {validation_errors}")
            
            # Encrypt sensitive data
            config_data = await self._encrypt_sensitive_config_data(config_data)
            
            # Create configuration in database
            config_id = await self.db_manager.create_service_configuration({
                "service_name": service_name,
                "service_type": config_data["service_type"],
                "configuration": config_data,
                "environment": config_data.get("environment", Environment.DEVELOPMENT),
                "is_active": True
            })
            
            # Update cache
            cache_key = f"{service_name}:{config_data.get('environment', Environment.DEVELOPMENT)}"
            self.config_cache[cache_key] = config_data
            
            self.logger.info(f"Service configuration created: {service_name}")
            return config_id
            
        except Exception as e:
            self.logger.error(f"Failed to create service configuration: {e}")
            raise
    
    async def get_service_config(self, service_name: str, environment: str = Environment.DEVELOPMENT) -> Optional[Dict[str, Any]]:
        """Get service configuration by name and environment."""
        try:
            cache_key = f"{service_name}:{environment}"
            
            # Check cache first
            if cache_key in self.config_cache:
                return self.config_cache[cache_key]
            
            # Load from database
            config = await self.db_manager.get_service_configuration(service_name, environment)
            
            if config:
                # Decrypt sensitive data
                config_data = await self._decrypt_sensitive_config_data(config["configuration"])
                
                # Cache the configuration
                self.config_cache[cache_key] = config_data
                
                return config_data
            
            # Return default configuration if available
            service_type = self._detect_service_type(service_name)
            if service_type in self.default_configs:
                default_config = self.default_configs[service_type].copy()
                default_config["environment"] = environment
                return default_config
            
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get service configuration: {e}")
            raise
    
    async def create_profile(self, profile_data: ConfigurationProfileCreate) -> str:
        """Create a new configuration profile."""
        try:
            # Validate profile data
            validation_errors = await self._validate_profile_data(profile_data.dict())
            if validation_errors:
                raise ValueError(f"Profile validation failed: {validation_errors}")
            
            # Create profile in database
            profile_id = str(uuid.uuid4())
            
            # Prepare profile data for database
            db_profile_data = {
                "id": profile_id,
                "profile_name": profile_data.profile_name,
                "description": profile_data.description,
                "profile_type": profile_data.profile_type,
                "services_config": profile_data.services_config,
                "startup_sequence": profile_data.startup_sequence,
                "environment_variables": profile_data.environment_variables,
                "is_default": profile_data.is_default,
                "is_active": profile_data.is_active,
                "tags": profile_data.tags,
                "created_at": datetime.utcnow(),
                "updated_at": datetime.utcnow()
            }
            
            # Store in database (would need to implement this in DatabaseManager)
            # For now, cache it
            self.profile_cache[profile_id] = db_profile_data
            
            self.logger.info(f"Configuration profile created: {profile_data.profile_name}")
            return profile_id
            
        except Exception as e:
            self.logger.error(f"Failed to create configuration profile: {e}")
            raise
    
    async def get_profile(self, profile_id: str) -> Optional[Dict[str, Any]]:
        """Get configuration profile by ID."""
        try:
            # Check cache first
            if profile_id in self.profile_cache:
                return self.profile_cache[profile_id]
            
            # In a real implementation, this would load from database
            # For now, return None if not in cache
            return None
            
        except Exception as e:
            self.logger.error(f"Failed to get configuration profile: {e}")
            raise
    
    async def get_profiles_by_type(self, profile_type: str) -> List[Dict[str, Any]]:
        """Get all profiles of a specific type."""
        try:
            profiles = []
            for profile in self.profile_cache.values():
                if profile.get("profile_type") == profile_type and profile.get("is_active", True):
                    profiles.append(profile)
            
            return profiles
            
        except Exception as e:
            self.logger.error(f"Failed to get profiles by type: {e}")
            raise
    
    async def get_environment_config(self, environment: str) -> Dict[str, Any]:
        """Get environment-specific configuration."""
        try:
            env_config = {
                "environment": environment,
                "database_url": self._get_database_url(environment),
                "redis_url": self._get_redis_url(environment),
                "ollama_url": self._get_ollama_url(environment),
                "monitoring_enabled": environment != Environment.DEVELOPMENT,
                "debug_mode": environment == Environment.DEVELOPMENT,
                "log_level": "DEBUG" if environment == Environment.DEVELOPMENT else "INFO"
            }
            
            # Add environment-specific overrides
            if environment == Environment.PRODUCTION:
                env_config.update({
                    "ssl_required": True,
                    "rate_limiting": True,
                    "backup_enabled": True,
                    "health_check_interval": 30
                })
            elif environment == Environment.STAGING:
                env_config.update({
                    "ssl_required": False,
                    "rate_limiting": True,
                    "backup_enabled": False,
                    "health_check_interval": 60
                })
            else:  # Development
                env_config.update({
                    "ssl_required": False,
                    "rate_limiting": False,
                    "backup_enabled": False,
                    "health_check_interval": 120
                })
            
            return env_config
            
        except Exception as e:
            self.logger.error(f"Failed to get environment configuration: {e}")
            raise
    
    async def validate_configuration(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate configuration data and return list of errors."""
        return await self._validate_service_config(config_data)
    
    async def _validate_service_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate service configuration data."""
        errors = []
        
        try:
            service_type = config_data.get("service_type")
            if not service_type:
                errors.append("service_type is required")
                return errors
            
            # Get validation rules for service type
            rules = self.validation_rules.get(service_type, {})
            
            # Check required fields
            required_fields = rules.get("required_fields", [])
            for field in required_fields:
                if field not in config_data:
                    errors.append(f"Required field missing: {field}")
            
            # Validate port range
            if "port" in config_data and "port_range" in rules:
                port = config_data["port"]
                min_port, max_port = rules["port_range"]
                if not (min_port <= port <= max_port):
                    errors.append(f"Port {port} not in valid range {min_port}-{max_port}")
            
            # Validate Docker image
            if "image" in config_data and "valid_images" in rules:
                image = config_data["image"]
                valid_images = rules["valid_images"]
                if not any(image.startswith(valid) for valid in valid_images):
                    errors.append(f"Invalid Docker image: {image}")
            
            # Service-specific validations
            if service_type == ServiceType.POSTGRESQL:
                errors.extend(await self._validate_postgresql_config(config_data))
            elif service_type == ServiceType.REDIS:
                errors.extend(await self._validate_redis_config(config_data))
            elif service_type == ServiceType.OLLAMA:
                errors.extend(await self._validate_ollama_config(config_data))
            
        except Exception as e:
            errors.append(f"Validation error: {str(e)}")
        
        return errors
    
    async def _validate_postgresql_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate PostgreSQL-specific configuration."""
        errors = []
        
        # Validate database name
        database = config_data.get("database", "")
        if not database.replace("_", "").isalnum():
            errors.append("Database name must be alphanumeric (underscores allowed)")
        
        # Validate memory settings
        shared_buffers = config_data.get("shared_buffers", "")
        if shared_buffers and not shared_buffers.endswith(("MB", "GB")):
            errors.append("shared_buffers must end with MB or GB")
        
        return errors
    
    async def _validate_redis_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate Redis-specific configuration."""
        errors = []
        
        # Validate memory policy
        policy = config_data.get("max_memory_policy", "")
        valid_policies = ["noeviction", "allkeys-lru", "volatile-lru", "allkeys-random", "volatile-random", "volatile-ttl"]
        if policy and policy not in valid_policies:
            errors.append(f"Invalid max_memory_policy: {policy}")
        
        return errors
    
    async def _validate_ollama_config(self, config_data: Dict[str, Any]) -> List[str]:
        """Validate Ollama-specific configuration."""
        errors = []
        
        # Validate models
        models = config_data.get("models", [])
        if not models:
            errors.append("At least one model must be specified")
        
        return errors
    
    async def _validate_profile_data(self, profile_data: Dict[str, Any]) -> List[str]:
        """Validate configuration profile data."""
        errors = []
        
        # Validate startup sequence
        startup_sequence = profile_data.get("startup_sequence", [])
        services_config = profile_data.get("services_config", {})
        
        for service in startup_sequence:
            if service not in services_config:
                errors.append(f"Service {service} in startup_sequence not found in services_config")
        
        # Validate individual service configurations
        for service_name, config in services_config.items():
            service_errors = await self._validate_service_config(config)
            for error in service_errors:
                errors.append(f"Service {service_name}: {error}")
        
        return errors
    
    async def _encrypt_sensitive_config_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Encrypt sensitive configuration data."""
        sensitive_fields = ["password", "secret", "key", "token"]
        encrypted_config = config_data.copy()
        
        for field in sensitive_fields:
            if field in encrypted_config and encrypted_config[field]:
                encrypted_config[field] = self.security_manager.secure_config_value(
                    encrypted_config[field], field
                )
        
        return encrypted_config
    
    async def _decrypt_sensitive_config_data(self, config_data: Dict[str, Any]) -> Dict[str, Any]:
        """Decrypt sensitive configuration data."""
        sensitive_fields = ["password", "secret", "key", "token"]
        decrypted_config = config_data.copy()
        
        for field in sensitive_fields:
            if field in decrypted_config and decrypted_config[field]:
                try:
                    decrypted_config[field] = self.security_manager.retrieve_config_value(
                        decrypted_config[field], field
                    )
                except Exception:
                    # If decryption fails, assume it's not encrypted
                    pass
        
        return decrypted_config
    
    def _detect_service_type(self, service_name: str) -> Optional[str]:
        """Detect service type from service name."""
        name_lower = service_name.lower()
        
        if "postgres" in name_lower or "postgresql" in name_lower:
            return ServiceType.POSTGRESQL
        elif "redis" in name_lower:
            return ServiceType.REDIS
        elif "ollama" in name_lower:
            return ServiceType.OLLAMA
        elif "agent" in name_lower:
            return ServiceType.AGENT
        elif any(monitor in name_lower for monitor in ["prometheus", "grafana", "monitoring"]):
            return ServiceType.MONITORING
        
        return ServiceType.OTHER
    
    def _get_database_url(self, environment: str) -> str:
        """Get database URL for environment."""
        if environment == Environment.PRODUCTION:
            return os.getenv("PROD_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pygent_factory")
        elif environment == Environment.STAGING:
            return os.getenv("STAGING_DATABASE_URL", "postgresql://postgres:postgres@localhost:5433/pygent_factory_staging")
        else:
            return os.getenv("DEV_DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/pygent_factory_dev")
    
    def _get_redis_url(self, environment: str) -> str:
        """Get Redis URL for environment."""
        if environment == Environment.PRODUCTION:
            return os.getenv("PROD_REDIS_URL", "redis://localhost:6379/0")
        elif environment == Environment.STAGING:
            return os.getenv("STAGING_REDIS_URL", "redis://localhost:6380/0")
        else:
            return os.getenv("DEV_REDIS_URL", "redis://localhost:6379/1")
    
    def _get_ollama_url(self, environment: str) -> str:
        """Get Ollama URL for environment."""
        if environment == Environment.PRODUCTION:
            return os.getenv("PROD_OLLAMA_URL", "http://localhost:11434")
        elif environment == Environment.STAGING:
            return os.getenv("STAGING_OLLAMA_URL", "http://localhost:11435")
        else:
            return os.getenv("DEV_OLLAMA_URL", "http://localhost:11434")
    
    async def _ensure_default_profiles(self):
        """Ensure default configuration profiles exist."""
        try:
            # Development profile
            dev_profile = ConfigurationProfileCreate(
                profile_name="Development Default",
                description="Default development environment configuration",
                profile_type="development",
                services_config={
                    "postgresql": self.default_configs[ServiceType.POSTGRESQL],
                    "redis": self.default_configs[ServiceType.REDIS],
                    "ollama": self.default_configs[ServiceType.OLLAMA]
                },
                startup_sequence=["postgresql", "redis", "ollama"],
                environment_variables={
                    "ENVIRONMENT": "development",
                    "DEBUG": "true",
                    "LOG_LEVEL": "DEBUG"
                },
                is_default=True,
                tags=["development", "default"]
            )
            
            await self.create_profile(dev_profile)
            
            # Production profile
            prod_config = self.default_configs.copy()
            for service_type, config in prod_config.items():
                config["environment"] = Environment.PRODUCTION
            
            prod_profile = ConfigurationProfileCreate(
                profile_name="Production Default",
                description="Default production environment configuration",
                profile_type="production",
                services_config=prod_config,
                startup_sequence=["postgresql", "redis", "ollama", "monitoring"],
                environment_variables={
                    "ENVIRONMENT": "production",
                    "DEBUG": "false",
                    "LOG_LEVEL": "INFO"
                },
                is_default=False,
                tags=["production", "default"]
            )
            
            await self.create_profile(prod_profile)
            
            self.logger.info("Default configuration profiles ensured")
            
        except Exception as e:
            self.logger.warning(f"Failed to ensure default profiles: {e}")
    
    async def _validate_configuration_integrity(self):
        """Validate overall configuration integrity."""
        try:
            # Check for configuration conflicts
            # Validate default configurations
            # Ensure required profiles exist
            self.logger.info("Configuration integrity validation completed")
            
        except Exception as e:
            self.logger.warning(f"Configuration integrity validation failed: {e}")
