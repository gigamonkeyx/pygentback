"""
Integration Configuration Management

Comprehensive configuration management for integration components,
environment-specific settings, validation, and dynamic updates.
"""

import logging
import os
import json
import yaml
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from enum import Enum

logger = logging.getLogger(__name__)


class ConfigEnvironment(Enum):
    """Configuration environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


class ConfigFormat(Enum):
    """Configuration file formats"""
    JSON = "json"
    YAML = "yaml"
    ENV = "env"


@dataclass
class ComponentConfig:
    """Configuration for a single component"""
    component_id: str
    component_type: str
    enabled: bool = True
    settings: Dict[str, Any] = field(default_factory=dict)
    environment_overrides: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    validation_rules: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntegrationSettings:
    """Core integration settings"""
    max_concurrent_workflows: int = 5
    default_timeout_seconds: int = 300
    retry_attempts: int = 3
    enable_monitoring: bool = True
    enable_events: bool = True
    log_level: str = "INFO"
    performance_tracking: bool = True


@dataclass
class SecuritySettings:
    """Security configuration"""
    enable_authentication: bool = True
    enable_authorization: bool = True
    secret_key: Optional[str] = None
    token_expiry_hours: int = 24
    allowed_origins: List[str] = field(default_factory=list)
    rate_limiting: bool = True
    max_requests_per_minute: int = 100


@dataclass
class DatabaseSettings:
    """Database configuration"""
    connection_string: Optional[str] = None
    pool_size: int = 10
    max_overflow: int = 20
    pool_timeout: int = 30
    enable_migrations: bool = True
    backup_enabled: bool = False


class IntegrationConfigManager:
    """
    Integration Configuration Manager.
    
    Manages configuration for all integration components with support for
    environment-specific settings, validation, and dynamic updates.
    """
    
    def __init__(self, config_dir: str = "config", environment: str = None):
        self.config_dir = Path(config_dir)
        self.environment = ConfigEnvironment(environment or os.getenv("ENVIRONMENT", "development"))
        
        # Configuration storage
        self.component_configs: Dict[str, ComponentConfig] = {}
        self.integration_settings = IntegrationSettings()
        self.security_settings = SecuritySettings()
        self.database_settings = DatabaseSettings()
        
        # Configuration metadata
        self.config_version = "1.0.0"
        self.last_loaded = None
        self.config_sources = []
        
        # Validation rules
        self.validation_rules = self._initialize_validation_rules()
        
        # Change tracking
        self.change_handlers: List[callable] = []
        self.config_history: List[Dict[str, Any]] = []
        
        # Load configuration
        self._load_configuration()
    
    def _initialize_validation_rules(self) -> Dict[str, Any]:
        """Initialize configuration validation rules"""
        return {
            'integration_settings': {
                'max_concurrent_workflows': {'type': int, 'min': 1, 'max': 50},
                'default_timeout_seconds': {'type': int, 'min': 10, 'max': 3600},
                'retry_attempts': {'type': int, 'min': 0, 'max': 10},
                'log_level': {'type': str, 'choices': ['DEBUG', 'INFO', 'WARNING', 'ERROR']}
            },
            'security_settings': {
                'token_expiry_hours': {'type': int, 'min': 1, 'max': 168},
                'max_requests_per_minute': {'type': int, 'min': 1, 'max': 10000}
            },
            'database_settings': {
                'pool_size': {'type': int, 'min': 1, 'max': 100},
                'max_overflow': {'type': int, 'min': 0, 'max': 100},
                'pool_timeout': {'type': int, 'min': 1, 'max': 300}
            }
        }
    
    def _load_configuration(self):
        """Load configuration from files and environment"""
        try:
            # Load base configuration
            self._load_base_config()
            
            # Load environment-specific configuration
            self._load_environment_config()
            
            # Load component configurations
            self._load_component_configs()
            
            # Apply environment variable overrides
            self._apply_environment_overrides()
            
            # Validate configuration
            self._validate_configuration()
            
            self.last_loaded = datetime.utcnow()
            logger.info(f"Configuration loaded for environment: {self.environment.value}")
            
        except Exception as e:
            logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_base_config(self):
        """Load base configuration file"""
        base_config_file = self.config_dir / "base.yaml"
        if base_config_file.exists():
            with open(base_config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self._apply_config_data(config_data)
            self.config_sources.append(str(base_config_file))
    
    def _load_environment_config(self):
        """Load environment-specific configuration"""
        env_config_file = self.config_dir / f"{self.environment.value}.yaml"
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                config_data = yaml.safe_load(f)
            
            self._apply_config_data(config_data)
            self.config_sources.append(str(env_config_file))
    
    def _load_component_configs(self):
        """Load component-specific configurations"""
        components_dir = self.config_dir / "components"
        if components_dir.exists():
            for config_file in components_dir.glob("*.yaml"):
                try:
                    with open(config_file, 'r') as f:
                        component_data = yaml.safe_load(f)
                    
                    component_id = config_file.stem
                    self._load_component_config(component_id, component_data)
                    self.config_sources.append(str(config_file))
                    
                except Exception as e:
                    logger.error(f"Failed to load component config {config_file}: {e}")
    
    def _load_component_config(self, component_id: str, config_data: Dict[str, Any]):
        """Load configuration for a specific component"""
        component_config = ComponentConfig(
            component_id=component_id,
            component_type=config_data.get('type', 'unknown'),
            enabled=config_data.get('enabled', True),
            settings=config_data.get('settings', {}),
            environment_overrides=config_data.get('environment_overrides', {}),
            validation_rules=config_data.get('validation_rules', {})
        )
        
        # Apply environment-specific overrides
        if self.environment.value in component_config.environment_overrides:
            overrides = component_config.environment_overrides[self.environment.value]
            component_config.settings.update(overrides)
        
        self.component_configs[component_id] = component_config
    
    def _apply_config_data(self, config_data: Dict[str, Any]):
        """Apply configuration data to settings objects"""
        if 'integration' in config_data:
            self._update_integration_settings(config_data['integration'])
        
        if 'security' in config_data:
            self._update_security_settings(config_data['security'])
        
        if 'database' in config_data:
            self._update_database_settings(config_data['database'])
    
    def _update_integration_settings(self, settings_data: Dict[str, Any]):
        """Update integration settings"""
        for key, value in settings_data.items():
            if hasattr(self.integration_settings, key):
                setattr(self.integration_settings, key, value)
    
    def _update_security_settings(self, settings_data: Dict[str, Any]):
        """Update security settings"""
        for key, value in settings_data.items():
            if hasattr(self.security_settings, key):
                setattr(self.security_settings, key, value)
    
    def _update_database_settings(self, settings_data: Dict[str, Any]):
        """Update database settings"""
        for key, value in settings_data.items():
            if hasattr(self.database_settings, key):
                setattr(self.database_settings, key, value)
    
    def _apply_environment_overrides(self):
        """Apply environment variable overrides"""
        # Integration settings
        if os.getenv("MAX_CONCURRENT_WORKFLOWS"):
            self.integration_settings.max_concurrent_workflows = int(os.getenv("MAX_CONCURRENT_WORKFLOWS"))
        
        if os.getenv("DEFAULT_TIMEOUT"):
            self.integration_settings.default_timeout_seconds = int(os.getenv("DEFAULT_TIMEOUT"))
        
        if os.getenv("LOG_LEVEL"):
            self.integration_settings.log_level = os.getenv("LOG_LEVEL")
        
        # Security settings
        if os.getenv("SECRET_KEY"):
            self.security_settings.secret_key = os.getenv("SECRET_KEY")
        
        if os.getenv("ENABLE_AUTH"):
            self.security_settings.enable_authentication = os.getenv("ENABLE_AUTH").lower() == "true"
        
        # Database settings
        if os.getenv("DATABASE_URL"):
            self.database_settings.connection_string = os.getenv("DATABASE_URL")
        
        if os.getenv("DB_POOL_SIZE"):
            self.database_settings.pool_size = int(os.getenv("DB_POOL_SIZE"))
    
    def _validate_configuration(self):
        """Validate all configuration settings"""
        errors = []
        
        # Validate integration settings
        errors.extend(self._validate_settings(
            'integration_settings', 
            asdict(self.integration_settings),
            self.validation_rules.get('integration_settings', {})
        ))
        
        # Validate security settings
        errors.extend(self._validate_settings(
            'security_settings',
            asdict(self.security_settings),
            self.validation_rules.get('security_settings', {})
        ))
        
        # Validate database settings
        errors.extend(self._validate_settings(
            'database_settings',
            asdict(self.database_settings),
            self.validation_rules.get('database_settings', {})
        ))
        
        # Validate component configurations
        for component_id, component_config in self.component_configs.items():
            component_errors = self._validate_component_config(component_config)
            errors.extend([f"{component_id}: {error}" for error in component_errors])
        
        if errors:
            error_message = "Configuration validation failed:\n" + "\n".join(errors)
            logger.error(error_message)
            if self.environment == ConfigEnvironment.PRODUCTION:
                raise ValueError(error_message)
            else:
                logger.warning("Configuration validation failed, but continuing in non-production environment")
    
    def _validate_settings(self, settings_name: str, settings_dict: Dict[str, Any], 
                          rules: Dict[str, Any]) -> List[str]:
        """Validate settings against rules"""
        errors = []
        
        for field_name, field_value in settings_dict.items():
            if field_name in rules:
                rule = rules[field_name]
                field_errors = self._validate_field(field_name, field_value, rule)
                errors.extend([f"{settings_name}.{error}" for error in field_errors])
        
        return errors
    
    def _validate_field(self, field_name: str, value: Any, rule: Dict[str, Any]) -> List[str]:
        """Validate a single field against its rule"""
        errors = []
        
        # Type validation
        if 'type' in rule and not isinstance(value, rule['type']):
            errors.append(f"{field_name}: expected {rule['type'].__name__}, got {type(value).__name__}")
        
        # Range validation for numbers
        if isinstance(value, (int, float)):
            if 'min' in rule and value < rule['min']:
                errors.append(f"{field_name}: value {value} is below minimum {rule['min']}")
            if 'max' in rule and value > rule['max']:
                errors.append(f"{field_name}: value {value} is above maximum {rule['max']}")
        
        # Choice validation for strings
        if 'choices' in rule and value not in rule['choices']:
            errors.append(f"{field_name}: value '{value}' not in allowed choices {rule['choices']}")
        
        return errors
    
    def _validate_component_config(self, component_config: ComponentConfig) -> List[str]:
        """Validate component configuration"""
        errors = []
        
        # Validate against component-specific rules
        for field_name, field_value in component_config.settings.items():
            if field_name in component_config.validation_rules:
                rule = component_config.validation_rules[field_name]
                field_errors = self._validate_field(field_name, field_value, rule)
                errors.extend(field_errors)
        
        return errors
    
    def get_component_config(self, component_id: str) -> Optional[ComponentConfig]:
        """Get configuration for a component"""
        return self.component_configs.get(component_id)
    
    def get_component_setting(self, component_id: str, setting_name: str, default: Any = None) -> Any:
        """Get a specific setting for a component"""
        component_config = self.get_component_config(component_id)
        if component_config:
            return component_config.settings.get(setting_name, default)
        return default
    
    def update_component_setting(self, component_id: str, setting_name: str, value: Any):
        """Update a component setting"""
        if component_id not in self.component_configs:
            raise ValueError(f"Component not found: {component_id}")
        
        old_value = self.component_configs[component_id].settings.get(setting_name)
        self.component_configs[component_id].settings[setting_name] = value
        
        # Track change
        self._track_change(f"component.{component_id}.{setting_name}", old_value, value)
        
        # Trigger change handlers
        self._trigger_change_handlers(component_id, setting_name, old_value, value)
    
    def register_component(self, component_id: str, component_type: str, 
                          default_settings: Dict[str, Any] = None):
        """Register a new component with default configuration"""
        if component_id in self.component_configs:
            logger.warning(f"Component {component_id} already registered")
            return
        
        component_config = ComponentConfig(
            component_id=component_id,
            component_type=component_type,
            settings=default_settings or {}
        )
        
        self.component_configs[component_id] = component_config
        logger.info(f"Registered component: {component_id}")
    
    def unregister_component(self, component_id: str):
        """Unregister a component"""
        if component_id in self.component_configs:
            del self.component_configs[component_id]
            logger.info(f"Unregistered component: {component_id}")
    
    def reload_configuration(self):
        """Reload configuration from files"""
        logger.info("Reloading configuration...")
        self.config_sources.clear()
        self._load_configuration()
    
    def export_configuration(self, format: ConfigFormat = ConfigFormat.YAML) -> str:
        """Export current configuration"""
        config_data = {
            'version': self.config_version,
            'environment': self.environment.value,
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None,
            'integration': asdict(self.integration_settings),
            'security': asdict(self.security_settings),
            'database': asdict(self.database_settings),
            'components': {
                comp_id: asdict(comp_config) 
                for comp_id, comp_config in self.component_configs.items()
            }
        }
        
        if format == ConfigFormat.JSON:
            return json.dumps(config_data, indent=2)
        elif format == ConfigFormat.YAML:
            return yaml.dump(config_data, default_flow_style=False)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _track_change(self, setting_path: str, old_value: Any, new_value: Any):
        """Track configuration changes"""
        change_record = {
            'timestamp': datetime.utcnow().isoformat(),
            'setting_path': setting_path,
            'old_value': old_value,
            'new_value': new_value,
            'environment': self.environment.value
        }
        
        self.config_history.append(change_record)
        
        # Keep only recent changes
        if len(self.config_history) > 1000:
            self.config_history = self.config_history[-500:]
    
    def _trigger_change_handlers(self, component_id: str, setting_name: str, 
                               old_value: Any, new_value: Any):
        """Trigger configuration change handlers"""
        for handler in self.change_handlers:
            try:
                handler(component_id, setting_name, old_value, new_value)
            except Exception as e:
                logger.error(f"Configuration change handler error: {e}")
    
    def add_change_handler(self, handler: callable):
        """Add a configuration change handler"""
        self.change_handlers.append(handler)
    
    def get_configuration_summary(self) -> Dict[str, Any]:
        """Get configuration summary"""
        return {
            'version': self.config_version,
            'environment': self.environment.value,
            'last_loaded': self.last_loaded.isoformat() if self.last_loaded else None,
            'config_sources': self.config_sources,
            'components_count': len(self.component_configs),
            'enabled_components': len([c for c in self.component_configs.values() if c.enabled]),
            'changes_tracked': len(self.config_history)
        }
