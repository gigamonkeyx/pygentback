"""
Configuration Manager

Centralized configuration management for PyGent Factory with support
for YAML files, environment variables, and runtime configuration updates.
"""

import os
import yaml
import logging
from typing import Dict, Any, Optional, Union
from pathlib import Path
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class ConfigError(Exception):
    """Configuration-related errors"""
    pass


@dataclass
class ConfigValidationResult:
    """Result of configuration validation"""
    is_valid: bool
    errors: list = field(default_factory=list)
    warnings: list = field(default_factory=list)


class ConfigManager:
    """
    Centralized configuration manager for PyGent Factory
    
    Supports:
    - YAML configuration files
    - Environment variable overrides
    - Runtime configuration updates
    - Configuration validation
    - Environment-specific settings
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        self.config_dir = Path(config_dir or "config")
        self.config_data: Dict[str, Any] = {}
        self.environment = os.getenv("PYGENT_ENV", "development")
        
        # Configuration file paths
        self.base_config_file = self.config_dir / "production.yaml"
        self.env_config_file = self.config_dir / f"{self.environment}.yaml"
        
        # Load configuration
        self._load_configuration()
        
        logger.info(f"Configuration manager initialized for environment: {self.environment}")
    
    def _load_configuration(self):
        """Load configuration from files and environment variables"""
        
        # Load base configuration
        if self.base_config_file.exists():
            with open(self.base_config_file, 'r') as f:
                self.config_data = yaml.safe_load(f) or {}
            logger.info(f"Loaded base configuration from {self.base_config_file}")
        else:
            logger.warning(f"Base configuration file not found: {self.base_config_file}")
            self.config_data = {}
        
        # Load environment-specific overrides
        if self.env_config_file.exists() and self.env_config_file != self.base_config_file:
            with open(self.env_config_file, 'r') as f:
                env_config = yaml.safe_load(f) or {}
            
            # Merge environment-specific configuration
            self._deep_merge(self.config_data, env_config)
            logger.info(f"Applied environment overrides from {self.env_config_file}")
        
        # Apply environment variable overrides
        self._apply_env_overrides()
        
        # Validate configuration
        validation_result = self._validate_configuration()
        if not validation_result.is_valid:
            raise ConfigError(f"Configuration validation failed: {validation_result.errors}")
        
        if validation_result.warnings:
            for warning in validation_result.warnings:
                logger.warning(f"Configuration warning: {warning}")
    
    def _deep_merge(self, base: Dict[str, Any], override: Dict[str, Any]):
        """Deep merge two dictionaries"""
        for key, value in override.items():
            if key in base and isinstance(base[key], dict) and isinstance(value, dict):
                self._deep_merge(base[key], value)
            else:
                base[key] = value
    
    def _apply_env_overrides(self):
        """Apply environment variable overrides"""
        
        # Define environment variable mappings
        env_mappings = {
            'PYGENT_LOG_LEVEL': 'system.log_level',
            'PYGENT_GPU_ENABLED': 'hardware.gpu.enabled',
            'PYGENT_GPU_MEMORY_FRACTION': 'hardware.gpu.memory_fraction',
            'PYGENT_OLLAMA_URL': 'ollama.base_url',
            'PYGENT_API_PORT': 'api.port',
            'PYGENT_MAX_WORKERS': 'hardware.cpu.max_workers',
            'PYGENT_CACHE_SIZE_MB': 'hardware.memory.max_cache_size_mb',
            'PYGENT_TOT_MAX_DEPTH': 'tot.default_config.max_depth',
            'PYGENT_EVOLUTION_POPULATION': 'evolution.population_size',
            'PYGENT_VECTOR_DIMENSION': 'vector_search.dimension'
        }
        
        for env_var, config_path in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                converted_value = self._convert_env_value(env_value)
                self._set_nested_value(self.config_data, config_path, converted_value)
                logger.debug(f"Applied environment override: {env_var} -> {config_path} = {converted_value}")
    
    def _convert_env_value(self, value: str) -> Union[str, int, float, bool]:
        """Convert environment variable string to appropriate type"""
        
        # Boolean conversion
        if value.lower() in ('true', 'false'):
            return value.lower() == 'true'
        
        # Integer conversion
        try:
            if '.' not in value:
                return int(value)
        except ValueError:
            pass
        
        # Float conversion
        try:
            return float(value)
        except ValueError:
            pass
        
        # Return as string
        return value
    
    def _set_nested_value(self, data: Dict[str, Any], path: str, value: Any):
        """Set a nested dictionary value using dot notation"""
        keys = path.split('.')
        current = data
        
        for key in keys[:-1]:
            if key not in current:
                current[key] = {}
            current = current[key]
        
        current[keys[-1]] = value
    
    def _validate_configuration(self) -> ConfigValidationResult:
        """Validate the loaded configuration"""
        
        errors = []
        warnings = []
        
        # Required sections
        required_sections = ['system', 'hardware', 'tot', 'unified_pipeline']
        for section in required_sections:
            if section not in self.config_data:
                errors.append(f"Missing required configuration section: {section}")
        
        # Hardware validation
        if 'hardware' in self.config_data:
            hardware_config = self.config_data['hardware']
            
            # GPU configuration
            if 'gpu' in hardware_config:
                gpu_config = hardware_config['gpu']
                if gpu_config.get('enabled', False):
                    memory_fraction = gpu_config.get('memory_fraction', 0.8)
                    if not 0.1 <= memory_fraction <= 1.0:
                        errors.append("GPU memory fraction must be between 0.1 and 1.0")
            
            # CPU configuration
            if 'cpu' in hardware_config:
                cpu_config = hardware_config['cpu']
                max_workers = cpu_config.get('max_workers', 8)
                if max_workers < 1 or max_workers > 64:
                    warnings.append("CPU max_workers should be between 1 and 64")
        
        # ToT configuration validation
        if 'tot' in self.config_data:
            tot_config = self.config_data['tot'].get('default_config', {})
            
            max_depth = tot_config.get('max_depth', 8)
            if max_depth < 1 or max_depth > 20:
                warnings.append("ToT max_depth should be between 1 and 20")
            
            temperature = tot_config.get('temperature', 0.7)
            if not 0.0 <= temperature <= 2.0:
                errors.append("ToT temperature must be between 0.0 and 2.0")
        
        # Vector search validation
        if 'vector_search' in self.config_data:
            vector_config = self.config_data['vector_search']
            
            dimension = vector_config.get('dimension', 768)
            if dimension < 1 or dimension > 4096:
                warnings.append("Vector dimension should be between 1 and 4096")
            
            if vector_config.get('use_gpu', False) and not self.config_data.get('hardware', {}).get('gpu', {}).get('enabled', False):
                warnings.append("Vector search GPU enabled but hardware GPU disabled")
        
        return ConfigValidationResult(
            is_valid=len(errors) == 0,
            errors=errors,
            warnings=warnings
        )
    
    def get(self, path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation"""
        
        keys = path.split('.')
        current = self.config_data
        
        try:
            for key in keys:
                current = current[key]
            return current
        except (KeyError, TypeError):
            return default
    
    def set(self, path: str, value: Any):
        """Set configuration value using dot notation"""
        self._set_nested_value(self.config_data, path, value)
        logger.debug(f"Configuration updated: {path} = {value}")
    
    def get_section(self, section: str) -> Dict[str, Any]:
        """Get entire configuration section"""
        return self.config_data.get(section, {})
    
    def update_section(self, section: str, updates: Dict[str, Any]):
        """Update configuration section"""
        if section not in self.config_data:
            self.config_data[section] = {}
        
        self._deep_merge(self.config_data[section], updates)
        logger.info(f"Configuration section updated: {section}")
    
    def reload(self):
        """Reload configuration from files"""
        logger.info("Reloading configuration...")
        self._load_configuration()
    
    def save(self, file_path: Optional[str] = None):
        """Save current configuration to file"""
        
        if file_path is None:
            file_path = self.config_dir / f"{self.environment}_current.yaml"
        
        file_path = Path(file_path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(file_path, 'w') as f:
            yaml.dump(self.config_data, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {file_path}")
    
    def get_environment(self) -> str:
        """Get current environment"""
        return self.environment
    
    def is_production(self) -> bool:
        """Check if running in production environment"""
        return self.environment == "production"
    
    def is_development(self) -> bool:
        """Check if running in development environment"""
        return self.environment == "development"
    
    def get_all(self) -> Dict[str, Any]:
        """Get all configuration data"""
        return self.config_data.copy()
    
    def get_hardware_config(self) -> Dict[str, Any]:
        """Get hardware configuration"""
        return self.get_section('hardware')
    
    def get_tot_config(self) -> Dict[str, Any]:
        """Get Tree of Thought configuration"""
        return self.get_section('tot')
    
    def get_s3_rag_config(self) -> Dict[str, Any]:
        """Get s3 RAG configuration"""
        return self.get_section('s3_rag')
    
    def get_vector_search_config(self) -> Dict[str, Any]:
        """Get vector search configuration"""
        return self.get_section('vector_search')
    
    def get_evolution_config(self) -> Dict[str, Any]:
        """Get evolution configuration"""
        return self.get_section('evolution')
    
    def get_ollama_config(self) -> Dict[str, Any]:
        """Get Ollama configuration"""
        return self.get_section('ollama')
    
    def get_api_config(self) -> Dict[str, Any]:
        """Get API configuration"""
        return self.get_section('api')
    
    def get_monitoring_config(self) -> Dict[str, Any]:
        """Get monitoring configuration"""
        return self.get_section('monitoring')


# Global configuration manager instance
_config_manager: Optional[ConfigManager] = None


def get_config_manager() -> ConfigManager:
    """Get global configuration manager instance"""
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager()
    
    return _config_manager


def initialize_config(config_dir: Optional[str] = None) -> ConfigManager:
    """Initialize global configuration manager"""
    global _config_manager
    
    _config_manager = ConfigManager(config_dir)
    return _config_manager
