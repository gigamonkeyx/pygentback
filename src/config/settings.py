"""
Application Settings and Configuration

This module provides centralized configuration management for PyGent Factory.
It uses Pydantic for settings validation and supports environment variables,
configuration files, and runtime configuration updates.
"""

import os
import logging
from typing import List, Optional
from pathlib import Path
from functools import lru_cache

from pydantic import Field, validator
from pydantic_settings import BaseSettings


logger = logging.getLogger(__name__)


class DatabaseSettings(BaseSettings):
    """Database configuration settings"""
    
    # Database Configuration (PostgreSQL with pgvector)
    DATABASE_URL: str = Field(
        default="postgresql://postgres:postgres@localhost:54321/pygent_factory",
        description="Synchronous database URL"
    )
    ASYNC_DATABASE_URL: str = Field(
        default="postgresql+asyncpg://postgres:postgres@localhost:54321/pygent_factory",
        description="Asynchronous database URL"
    )
    
    # Connection Pool Settings
    DB_POOL_SIZE: int = Field(default=10, description="Database connection pool size")
    DB_MAX_OVERFLOW: int = Field(default=20, description="Maximum connection overflow")
    DB_POOL_RECYCLE: int = Field(default=3600, description="Connection recycle time in seconds")
    
    # Supabase Configuration
    SUPABASE_URL: str = Field(default="http://localhost:54321", description="Supabase URL")
    SUPABASE_KEY: str = Field(default="development_key", description="Supabase API key")
    
    class Config:
        env_prefix = ""
        extra = "allow"


class AISettings(BaseSettings):
    """AI and ML configuration settings"""
    
    # API Keys
    OPENAI_API_KEY: str = Field(default="", description="OpenAI API key")
    OPENAI_API_BASE: Optional[str] = Field(default=None, description="OpenAI API base URL for proxies")
    ANTHROPIC_API_KEY: str = Field(default="", description="Anthropic API key")
    OPENROUTER_API_KEY: str = Field(default="", description="OpenRouter API key")
    OPENROUTER_API_BASE: Optional[str] = Field(default="https://openrouter.ai/api/v1", description="OpenRouter API base URL")

    # Model Configuration
    DEFAULT_LLM_MODEL: str = Field(default="gpt-4", description="Default LLM model")
    DEFAULT_EMBEDDING_PROVIDER: Optional[str] = Field(
        default=None, 
        description="Default embedding provider ('ollama', 'openai', 'openrouter', 'sentence_transformer')"
    )
    EMBEDDING_MODEL: str = Field(default="all-MiniLM-L6-v2", description="Embedding model for sentence-transformer")
    EMBEDDING_DIMENSION: int = Field(default=1536, description="Embedding vector dimension")
    
    # Ollama Configuration
    OLLAMA_BASE_URL: str = Field(default="http://localhost:11434", description="Ollama base URL")
    OLLAMA_MODEL: str = Field(default="deepseek-r1:8b", description="Default Ollama model")
    OLLAMA_EMBED_MODEL: str = Field(default="deepseek-r1:8b", description="Ollama model for embeddings")

    # Generation Settings
    MAX_TOKENS: int = Field(default=2048, description="Maximum tokens for generation")
    TEMPERATURE: float = Field(default=0.7, description="Generation temperature")
    
    @validator("TEMPERATURE")
    def validate_temperature(cls, v):
        if not 0.0 <= v <= 2.0:
            raise ValueError("Temperature must be between 0.0 and 2.0")
        return v
    
    class Config:
        env_prefix = ""
        extra = "allow"


class MCPSettings(BaseSettings):
    """MCP (Model Context Protocol) configuration settings"""
    
    # MCP Server Configuration
    MCP_SERVER_PATH: str = Field(default="./mcp_servers", description="MCP servers directory")
    MCP_LOG_LEVEL: str = Field(default="INFO", description="MCP logging level")
    MCP_TIMEOUT: int = Field(default=30, description="MCP operation timeout in seconds")
    
    # Server-specific Configuration
    MCP_FILESYSTEM_ALLOWED_PATHS: str = Field(
        default="D:/mcp/pygent-factory,D:/mcp,./workspace,./data,./src,./docs,./tests,./tools,./examples",
        description="Allowed filesystem paths for MCP filesystem server"
    )
    MCP_POSTGRES_CONNECTION_STRING: str = Field(
        default="postgresql://postgres:postgres@localhost:54321/pygent_factory",
        description="PostgreSQL connection string for MCP postgres server"
    )
    MCP_GITHUB_TOKEN: str = Field(default="", description="GitHub token for MCP github server")
    MCP_BRAVE_SEARCH_API_KEY: str = Field(default="", description="Brave Search API key")
    
    class Config:
        env_prefix = ""
        extra = "allow"


class VectorStoreSettings(BaseSettings):
    """Vector store configuration settings"""
    
    # ChromaDB Configuration
    CHROMADB_URL: str = Field(default="http://localhost:8000", description="ChromaDB URL")
    CHROMADB_COLLECTION_NAME: str = Field(default="pygent_knowledge", description="Default collection name")
    
    # Vector Search Configuration
    SIMILARITY_THRESHOLD: float = Field(default=0.7, description="Similarity search threshold")
    MAX_SEARCH_RESULTS: int = Field(default=10, description="Maximum search results")
    
    # RAG Configuration
    CHUNK_SIZE: int = Field(default=1000, description="Document chunk size")
    CHUNK_OVERLAP: int = Field(default=200, description="Document chunk overlap")
    
    @validator("SIMILARITY_THRESHOLD")
    def validate_similarity_threshold(cls, v):
        if not 0.0 <= v <= 1.0:
            raise ValueError("Similarity threshold must be between 0.0 and 1.0")
        return v
    
    class Config:
        env_prefix = ""
        extra = "allow"


class SecuritySettings(BaseSettings):
    """Security configuration settings"""
    
    # JWT Configuration
    SECRET_KEY: str = Field(
        default="development_secret_key_change_in_production_12345",
        description="Secret key for JWT tokens"
    )
    ALGORITHM: str = Field(default="HS256", description="JWT algorithm")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = Field(default=30, description="Access token expiration")
    
    # API Security
    API_KEY_HEADER: str = Field(default="X-API-Key", description="API key header name")
    CORS_ORIGINS: str = Field(
        default="http://localhost:5173,https://timpayne.net,https://www.timpayne.net,https://pygent-factory.pages.dev,https://*.pygent-factory.pages.dev,https://*.trycloudflare.com",
        description="CORS allowed origins (comma-separated, supports wildcards)"
    )
    
    @validator("SECRET_KEY")
    def validate_secret_key(cls, v):
        if len(v) < 32:
            raise ValueError("Secret key must be at least 32 characters long")
        return v
    
    class Config:
        env_prefix = ""
        extra = "allow"


class AgentSettings(BaseSettings):
    """Agent configuration settings"""
    
    # Agent Limits
    DEFAULT_AGENT_TIMEOUT: int = Field(default=300, description="Default agent timeout in seconds")
    MAX_CONCURRENT_AGENTS: int = Field(default=10, description="Maximum concurrent agents")
    AGENT_MEMORY_LIMIT: int = Field(default=1000, description="Agent memory entry limit")
    
    # Agent Behavior
    AGENT_RETRY_COUNT: int = Field(default=3, description="Default retry count for agent operations")
    AGENT_HEARTBEAT_INTERVAL: int = Field(default=60, description="Agent heartbeat interval in seconds")
    
    class Config:
        env_prefix = ""
        extra = "allow"


class ApplicationSettings(BaseSettings):
    """Main application configuration settings"""
    
    # Application Info
    APP_NAME: str = Field(default="PyGent Factory", description="Application name")
    APP_VERSION: str = Field(default="1.0.0", description="Application version")
    DEBUG: bool = Field(default=True, description="Debug mode")
    
    # Server Configuration
    HOST: str = Field(default="0.0.0.0", description="Server host")
    PORT: int = Field(default=8000, description="Server port")
    RELOAD: bool = Field(default=True, description="Auto-reload on changes")
    
    # Logging Configuration
    LOG_LEVEL: str = Field(default="INFO", description="Logging level")
    LOG_FORMAT: str = Field(
        default="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        description="Log format"
    )
    
    # File Storage
    UPLOAD_DIR: str = Field(default="./data/uploads", description="Upload directory")
    MAX_FILE_SIZE: int = Field(default=10485760, description="Maximum file size in bytes")  # 10MB
    ALLOWED_EXTENSIONS: str = Field(
        default=".txt,.md,.pdf,.docx,.json,.yaml,.yml",
        description="Allowed file extensions (comma-separated)"
    )
    
    # Redis Configuration (Optional)
    REDIS_URL: str = Field(default="redis://localhost:6379", description="Redis URL")
    REDIS_PASSWORD: str = Field(default="", description="Redis password")
    REDIS_DB: int = Field(default=0, description="Redis database number")
    
    # Evaluation Configuration
    EVALUATION_TIMEOUT: int = Field(default=600, description="Evaluation timeout in seconds")
    BENCHMARK_SUITE_PATH: str = Field(default="./data/benchmarks", description="Benchmark suite path")
    
    @validator("PORT")
    def validate_port(cls, v):
        if not 1 <= v <= 65535:
            raise ValueError("Port must be between 1 and 65535")
        return v
    
    @validator("LOG_LEVEL")
    def validate_log_level(cls, v):
        valid_levels = ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        if v.upper() not in valid_levels:
            raise ValueError(f"Log level must be one of: {valid_levels}")
        return v.upper()
    
    class Config:
        env_prefix = ""
        extra = "allow"


class Settings(BaseSettings):
    """
    Main settings class that combines all configuration sections.
    
    This class provides a unified interface to all application settings
    and handles environment variable loading, validation, and defaults.
    """
    
    # Include all setting sections
    database: DatabaseSettings = DatabaseSettings()
    ai: AISettings = AISettings()
    mcp: MCPSettings = MCPSettings()
    vector: VectorStoreSettings = VectorStoreSettings()
    security: SecuritySettings = SecuritySettings()
    agent: AgentSettings = AgentSettings()
    app: ApplicationSettings = ApplicationSettings()
    
    # Convenience properties for commonly used settings
    @property
    def DATABASE_URL(self) -> str:
        return self.database.DATABASE_URL
    
    @property
    def ASYNC_DATABASE_URL(self) -> str:
        return self.database.ASYNC_DATABASE_URL
    
    @property
    def OPENAI_API_KEY(self) -> str:
        return self.ai.OPENAI_API_KEY
    
    @property
    def ANTHROPIC_API_KEY(self) -> str:
        return self.ai.ANTHROPIC_API_KEY
    
    @property
    def EMBEDDING_MODEL(self) -> str:
        return self.ai.EMBEDDING_MODEL
    
    @property
    def SIMILARITY_THRESHOLD(self) -> float:
        return self.vector.SIMILARITY_THRESHOLD
    
    @property
    def DEFAULT_AGENT_TIMEOUT(self) -> int:
        return self.agent.DEFAULT_AGENT_TIMEOUT
    
    @property
    def AGENT_MEMORY_LIMIT(self) -> int:
        return self.agent.AGENT_MEMORY_LIMIT
    
    @property
    def SECRET_KEY(self) -> str:
        return self.security.SECRET_KEY
    
    @property
    def DEBUG(self) -> bool:
        return self.app.DEBUG
    
    @property
    def MCP_FILESYSTEM_ALLOWED_PATHS(self) -> str:
        return self.mcp.MCP_FILESYSTEM_ALLOWED_PATHS
    
    @property
    def MCP_POSTGRES_CONNECTION_STRING(self) -> str:
        return self.mcp.MCP_POSTGRES_CONNECTION_STRING
    
    @property
    def MCP_GITHUB_TOKEN(self) -> str:
        return self.mcp.MCP_GITHUB_TOKEN
    
    @property
    def MCP_BRAVE_SEARCH_API_KEY(self) -> str:
        return self.mcp.MCP_BRAVE_SEARCH_API_KEY

    @property
    def ALLOWED_EXTENSIONS_LIST(self) -> List[str]:
        """Get allowed extensions as a list"""
        return [ext.strip() for ext in self.app.ALLOWED_EXTENSIONS.split(",") if ext.strip()]

    @property
    def CORS_ORIGINS_LIST(self) -> List[str]:
        """Get CORS origins as a list"""
        return [origin.strip() for origin in self.security.CORS_ORIGINS.split(",") if origin.strip()]
    
    def setup_logging(self) -> None:
        """Set up application logging"""
        logging.basicConfig(
            level=getattr(logging, self.app.LOG_LEVEL),
            format=self.app.LOG_FORMAT,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler("logs/pygent_factory.log") if os.path.exists("logs") else logging.NullHandler()
            ]
        )
        
        # Set specific logger levels
        logging.getLogger("sqlalchemy.engine").setLevel(logging.WARNING)
        logging.getLogger("uvicorn").setLevel(logging.INFO)
    
    def create_directories(self) -> None:
        """Create necessary directories"""
        directories = [
            self.app.UPLOAD_DIR,
            self.app.BENCHMARK_SUITE_PATH,
            "logs",
            "data/backups"
        ]
        
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)
    
    def validate_configuration(self) -> List[str]:
        """
        Validate configuration and return list of warnings/errors.
        
        Returns:
            List[str]: List of validation messages
        """
        messages = []
        
        # Check required API keys in production
        if not self.DEBUG:
            if not self.ai.OPENAI_API_KEY:
                messages.append("WARNING: OpenAI API key not set")
            if not self.ai.ANTHROPIC_API_KEY:
                messages.append("WARNING: Anthropic API key not set")
        
        # Check database connectivity
        if "localhost" in self.DATABASE_URL and not self.DEBUG:
            messages.append("WARNING: Using localhost database in production")
        
        # Check security settings
        if (self.security.SECRET_KEY == "development_secret_key_change_in_production_12345"
            and not self.DEBUG):
            messages.append("ERROR: Using default secret key in production")
        
        return messages
    
    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        case_sensitive = True
        extra = "allow"


@lru_cache()
def get_settings() -> Settings:
    """
    Get cached settings instance.
    
    Returns:
        Settings: Application settings
    """
    settings = Settings()
    
    # Set up logging
    settings.setup_logging()
    
    # Create directories
    settings.create_directories()
    
    # Validate configuration
    validation_messages = settings.validate_configuration()
    for message in validation_messages:
        if message.startswith("ERROR"):
            logger.error(message)
        else:
            logger.warning(message)
    
    return settings


# Global settings instance
settings = get_settings()
