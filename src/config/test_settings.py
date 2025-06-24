"""
Test-specific settings that don't trigger production warnings
"""

import os
import logging
from typing import List, Dict, Any, Optional
from pathlib import Path
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings


class TestSettings(BaseSettings):
    """Test-specific settings that avoid production warnings"""
    
    # Basic settings for testing
    DEBUG: bool = True
    SECRET_KEY: str = "test_secret_key_for_testing_only_32_chars_long"
    
    # Database settings for testing
    DATABASE_URL: str = "sqlite:///test.db"
    ASYNC_DATABASE_URL: str = "sqlite+aiosqlite:///test.db"
    
    # AI settings for testing
    OPENAI_API_KEY: str = "test_key"
    ANTHROPIC_API_KEY: str = "test_key"
    OLLAMA_BASE_URL: str = "http://localhost:11434"
    OLLAMA_MODEL: str = "deepseek1:latest"
    
    # Vector store settings
    EMBEDDING_MODEL: str = "all-MiniLM-L6-v2"
    SIMILARITY_THRESHOLD: float = 0.7
    
    # Agent settings
    DEFAULT_AGENT_TIMEOUT: int = 30
    AGENT_MEMORY_LIMIT: int = 100
    
    # Logging
    LOG_LEVEL: str = "WARNING"  # Reduce log noise in tests
    
    def validate_configuration(self) -> List[str]:
        """Test configuration validation - no warnings"""
        return []  # No warnings for test configuration
    
    def setup_logging(self) -> None:
        """Minimal logging setup for tests"""
        logging.basicConfig(
            level=logging.WARNING,  # Only show warnings and errors
            format="%(levelname)s - %(message)s",
            handlers=[logging.StreamHandler()]
        )
    
    def create_directories(self) -> None:
        """Create minimal test directories"""
        Path("test_data").mkdir(exist_ok=True)
    
    class Config:
        env_prefix = "TEST_"
        extra = "allow"


@lru_cache()
def get_test_settings() -> TestSettings:
    """Get test settings without production warnings"""
    settings = TestSettings()
    settings.setup_logging()
    settings.create_directories()
    return settings


# Test settings instance
test_settings = get_test_settings()
