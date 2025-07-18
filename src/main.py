#!/usr/bin/env python3
"""
PyGent Factory Main Application

Production-ready main application with API gateway, authentication,
caching, and comprehensive system initialization.
"""

import asyncio
import os
import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parent))

# Import UTF-8 logging infrastructure
from utils.utf8_logger import get_pygent_logger, configure_utf8_logging

from api.gateway import api_gateway, APIGatewayConfig
from database.production_manager import db_manager
from cache.integration_layer import integrated_cache
from core.gpu_optimization import gpu_optimizer
from core.ollama_gpu_integration import ollama_gpu_manager

# Configure UTF-8 logging
configure_utf8_logging()
logger = get_pygent_logger("pygent_factory_main")


class PyGentFactoryApp:
    """Main PyGent Factory application"""

    def __init__(self, autonomous=False):
        self.is_initialized = False
        self.autonomous = autonomous  # Grok4 Heavy JSON autonomy flag
        self.components = {
            "database": False,
            "cache": False,
            "gpu": False,
            "ollama": False,
            "api_gateway": False,
            "autonomy": False
        }
    
    async def initialize(self) -> bool:
        """Initialize all application components"""
        try:
            logger.info("Starting PyGent Factory initialization...")

            # Create logs directory if it doesn't exist
            os.makedirs("logs", exist_ok=True)

            # Initialize database
            logger.info("Initializing production database...")
            if await db_manager.initialize():
                self.components["database"] = True
                logger.info("Database initialized successfully")
            else:
                logger.error("Database initialization failed")
                return False

            # Initialize integrated cache system
            logger.info("Initializing Redis caching system...")
            if await integrated_cache.initialize():
                self.components["cache"] = True
                logger.info("Cache system initialized successfully")
            else:
                logger.error("Cache system initialization failed")
                return False

            # Initialize GPU optimization
            logger.info("Initializing GPU optimization...")
            if await gpu_optimizer.initialize():
                self.components["gpu"] = True
                logger.info("GPU optimization initialized successfully")
            else:
                logger.warning("GPU optimization initialization failed (continuing without GPU)")
                self.components["gpu"] = False

            # Initialize Ollama GPU integration
            logger.info("Initializing Ollama GPU integration...")
            if await ollama_gpu_manager.initialize():
                self.components["ollama"] = True
                logger.info("Ollama GPU integration initialized successfully")
            else:
                logger.warning("Ollama GPU integration failed (continuing without Ollama)")
                self.components["ollama"] = False

            # Initialize API gateway
            logger.info("Initializing API gateway...")
            if await api_gateway.initialize():
                self.components["api_gateway"] = True
                logger.info("API gateway initialized successfully")
            else:
                logger.error("API gateway initialization failed")
                return False

            # Initialize Grok4 Heavy JSON autonomy system if enabled
            if self.autonomous:
                logger.info("Initializing Grok4 Heavy JSON autonomy system...")
                try:
                    from autonomy.mode import enable_hands_off_mode
                    autonomy_enabled = enable_hands_off_mode()
                    if autonomy_enabled:
                        self.components["autonomy"] = True
                        logger.info("✅ Grok4 Heavy JSON autonomy system enabled - hands-off mode active")
                    else:
                        logger.warning("⚠️ Autonomy system initialization failed")
                        self.components["autonomy"] = False
                except Exception as e:
                    logger.error(f"❌ Autonomy system initialization failed: {e}")
                    self.components["autonomy"] = False
            else:
                logger.info("Autonomy system disabled (autonomous=False)")
                self.components["autonomy"] = False

            self.is_initialized = True
            logger.info("PyGent Factory initialized successfully!")

            # Log component status
            self._log_component_status()

            return True
            
        except Exception as e:
            logger.error(f"❌ PyGent Factory initialization failed: {e}")
            return False
    
    def _log_component_status(self):
        """Log the status of all components"""
        logger.info("📋 Component Status:")
        for component, status in self.components.items():
            status_icon = "✅" if status else "❌"
            logger.info(f"   {status_icon} {component.title()}: {'Initialized' if status else 'Failed'}")
    
    async def health_check(self) -> dict:
        """Comprehensive application health check"""
        try:
            health_status = {
                "status": "healthy",
                "timestamp": asyncio.get_event_loop().time(),
                "components": {}
            }
            
            # Check database health
            if self.components["database"]:
                try:
                    db_health = await db_manager.health_check()
                    health_status["components"]["database"] = db_health
                except Exception as e:
                    health_status["components"]["database"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            else:
                health_status["components"]["database"] = {
                    "status": "not_initialized"
                }
            
            # Check cache health
            if self.components["cache"]:
                try:
                    cache_health = await integrated_cache.get_integrated_health_status()
                    health_status["components"]["cache"] = cache_health
                except Exception as e:
                    health_status["components"]["cache"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            else:
                health_status["components"]["cache"] = {
                    "status": "not_initialized"
                }
            
            # Check GPU health
            if self.components["gpu"]:
                try:
                    gpu_health = gpu_optimizer.get_optimization_status()
                    health_status["components"]["gpu"] = {
                        "status": "healthy" if gpu_health["initialized"] else "unhealthy",
                        "details": gpu_health
                    }
                except Exception as e:
                    health_status["components"]["gpu"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            else:
                health_status["components"]["gpu"] = {
                    "status": "not_available"
                }
            
            # Check Ollama health
            if self.components["ollama"]:
                try:
                    ollama_health = ollama_gpu_manager.get_performance_summary()
                    health_status["components"]["ollama"] = {
                        "status": "healthy" if ollama_gpu_manager.is_initialized else "unhealthy",
                        "details": ollama_health
                    }
                except Exception as e:
                    health_status["components"]["ollama"] = {
                        "status": "unhealthy",
                        "error": str(e)
                    }
            else:
                health_status["components"]["ollama"] = {
                    "status": "not_available"
                }
            
            # Determine overall status
            component_statuses = []
            for comp_health in health_status["components"].values():
                if isinstance(comp_health, dict):
                    component_statuses.append(comp_health.get("status", "unknown"))
            
            if "unhealthy" in component_statuses:
                health_status["status"] = "degraded"
            elif "not_initialized" in component_statuses:
                health_status["status"] = "degraded"
            
            return health_status
            
        except Exception as e:
            logger.error(f"Health check failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "timestamp": asyncio.get_event_loop().time()
            }
    
    async def shutdown(self):
        """Gracefully shutdown all components"""
        try:
            logger.info("🛑 Shutting down PyGent Factory...")
            
            # Shutdown components in reverse order
            if self.components["api_gateway"]:
                logger.info("Shutting down API gateway...")
                # API gateway shutdown handled by uvicorn
            
            if self.components["ollama"]:
                logger.info("Shutting down Ollama integration...")
                await ollama_gpu_manager.cleanup()
            
            if self.components["gpu"]:
                logger.info("Shutting down GPU optimization...")
                # GPU optimizer cleanup if needed
            
            if self.components["cache"]:
                logger.info("Shutting down cache system...")
                await integrated_cache.cleanup()
            
            if self.components["database"]:
                logger.info("Shutting down database...")
                await db_manager.cleanup()
            
            logger.info("✅ PyGent Factory shutdown completed")
            
        except Exception as e:
            logger.error(f"Error during shutdown: {e}")
    
    def run(self, host: str = "0.0.0.0", port: int = 8000, reload: bool = False):
        """Run the PyGent Factory application"""
        try:
            # Configure API gateway
            config = APIGatewayConfig()
            config.host = host
            config.port = port
            config.reload = reload
            
            api_gateway.config = config
            
            # Add health check endpoint to the gateway
            @api_gateway.app.get("/health")
            async def health_endpoint():
                return await self.health_check()
            
            # Add shutdown handler
            @api_gateway.app.on_event("shutdown")
            async def shutdown_event():
                await self.shutdown()
            
            # Run the API gateway
            logger.info(f"🌐 Starting PyGent Factory on {host}:{port}")
            api_gateway.run()
            
        except Exception as e:
            logger.error(f"Failed to run PyGent Factory: {e}")
            sys.exit(1)


# Global application instance (will be initialized with autonomy flag in main)
app = None


async def main():
    """Main application entry point"""
    try:
        # Initialize application
        success = await app.initialize()
        
        if not success:
            logger.error("❌ Application initialization failed")
            sys.exit(1)
        
        # Run application (this will be called by uvicorn in production)
        logger.info("✅ Application ready to serve requests")
        
    except KeyboardInterrupt:
        logger.info("🛑 Received shutdown signal")
        await app.shutdown()
    except Exception as e:
        logger.error(f"❌ Application error: {e}")
        await app.shutdown()
        sys.exit(1)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="PyGent Factory API Gateway")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--init-only", action="store_true", help="Initialize and exit")
    parser.add_argument("--autonomous", action="store_true", help="Enable Grok4 Heavy JSON autonomy mode")

    args = parser.parse_args()

    # Initialize app instance with autonomy flag
    app = PyGentFactoryApp(autonomous=args.autonomous)
    
    if args.init_only:
        # Just initialize and exit
        asyncio.run(main())
    else:
        # Initialize and run
        async def init_and_run():
            await app.initialize()
            app.run(host=args.host, port=args.port, reload=args.reload)
        
        asyncio.run(init_and_run())
