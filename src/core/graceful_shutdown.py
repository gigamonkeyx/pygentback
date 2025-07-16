#!/usr/bin/env python3
"""
Graceful Shutdown Manager for PyGent Factory

Implements graceful shutdown procedures and data persistence mechanisms
for Docker 4.43 integration with observer supervision.

Ensures clean shutdown of all components while preserving critical data.
"""

import asyncio
import logging
import signal
import sys
import time
from datetime import datetime
from typing import Dict, List, Any, Optional, Callable
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class GracefulShutdownManager:
    """Manages graceful shutdown of PyGent Factory components"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        self.shutdown_initiated = False
        self.shutdown_timeout = 30.0  # seconds
        self.shutdown_hooks = []
        self.cleanup_tasks = []
        self.data_persistence_tasks = []
        
        # Component shutdown order (higher priority first)
        self.shutdown_order = [
            "observer_supervision",
            "riperω_protocol",
            "agent_interactions",
            "emergent_behavior_detection",
            "evolution_system",
            "agent_factory",
            "docker_containers",
            "database_connections",
            "redis_connections",
            "file_systems"
        ]
        
        # Registered components
        self.components = {}
        
        # Shutdown statistics
        self.shutdown_stats = {
            "start_time": None,
            "end_time": None,
            "duration": 0.0,
            "components_shutdown": 0,
            "data_persisted": 0,
            "errors": []
        }
        
        # Setup signal handlers
        self._setup_signal_handlers()
    
    def _setup_signal_handlers(self):
        """Setup signal handlers for graceful shutdown"""
        try:
            # Handle SIGTERM (Docker stop)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
            # Handle SIGINT (Ctrl+C)
            signal.signal(signal.SIGINT, self._signal_handler)
            
            # Handle SIGHUP (reload)
            if hasattr(signal, 'SIGHUP'):
                signal.signal(signal.SIGHUP, self._signal_handler)
            
            self.logger.info("Signal handlers configured for graceful shutdown")
            
        except Exception as e:
            self.logger.error(f"Failed to setup signal handlers: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals"""
        signal_name = signal.Signals(signum).name
        self.logger.info(f"Received {signal_name} signal, initiating graceful shutdown...")
        
        # Run shutdown in event loop
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self.shutdown())
            else:
                asyncio.run(self.shutdown())
        except Exception as e:
            self.logger.error(f"Failed to initiate graceful shutdown: {e}")
            sys.exit(1)
    
    def register_component(self, name: str, component: Any, shutdown_method: str = "shutdown"):
        """Register component for graceful shutdown"""
        try:
            self.components[name] = {
                "component": component,
                "shutdown_method": shutdown_method,
                "priority": self.shutdown_order.index(name) if name in self.shutdown_order else 999,
                "registered_at": datetime.now()
            }
            
            self.logger.info(f"Component '{name}' registered for graceful shutdown")
            
        except Exception as e:
            self.logger.error(f"Failed to register component '{name}': {e}")
    
    def add_shutdown_hook(self, hook: Callable):
        """Add custom shutdown hook"""
        self.shutdown_hooks.append(hook)
        self.logger.info(f"Shutdown hook added: {hook.__name__}")
    
    def add_cleanup_task(self, task: Callable):
        """Add cleanup task"""
        self.cleanup_tasks.append(task)
        self.logger.info(f"Cleanup task added: {task.__name__}")
    
    def add_data_persistence_task(self, task: Callable):
        """Add data persistence task"""
        self.data_persistence_tasks.append(task)
        self.logger.info(f"Data persistence task added: {task.__name__}")
    
    async def shutdown(self):
        """Perform graceful shutdown"""
        if self.shutdown_initiated:
            self.logger.warning("Shutdown already initiated")
            return
        
        self.shutdown_initiated = True
        self.shutdown_stats["start_time"] = datetime.now()
        
        self.logger.info("Starting graceful shutdown sequence...")
        
        try:
            # Phase 1: Execute shutdown hooks
            await self._execute_shutdown_hooks()
            
            # Phase 2: Persist critical data
            await self._persist_critical_data()
            
            # Phase 3: Shutdown components in order
            await self._shutdown_components()
            
            # Phase 4: Execute cleanup tasks
            await self._execute_cleanup_tasks()
            
            # Phase 5: Final cleanup
            await self._final_cleanup()
            
            self.shutdown_stats["end_time"] = datetime.now()
            self.shutdown_stats["duration"] = (
                self.shutdown_stats["end_time"] - self.shutdown_stats["start_time"]
            ).total_seconds()
            
            self.logger.info(f"Graceful shutdown completed in {self.shutdown_stats['duration']:.2f}s")
            
        except Exception as e:
            self.logger.error(f"Error during graceful shutdown: {e}")
            self.shutdown_stats["errors"].append(str(e))
        
        finally:
            # Log shutdown statistics
            self._log_shutdown_statistics()
    
    async def _execute_shutdown_hooks(self):
        """Execute custom shutdown hooks"""
        self.logger.info("Executing shutdown hooks...")
        
        for hook in self.shutdown_hooks:
            try:
                if asyncio.iscoroutinefunction(hook):
                    await hook()
                else:
                    hook()
                
                self.logger.info(f"Shutdown hook executed: {hook.__name__}")
                
            except Exception as e:
                self.logger.error(f"Shutdown hook failed: {hook.__name__}: {e}")
                self.shutdown_stats["errors"].append(f"Hook {hook.__name__}: {e}")
    
    async def _persist_critical_data(self):
        """Persist critical data before shutdown"""
        self.logger.info("Persisting critical data...")
        
        persistence_tasks = []
        
        for task in self.data_persistence_tasks:
            try:
                if asyncio.iscoroutinefunction(task):
                    persistence_tasks.append(task())
                else:
                    # Run sync task in executor
                    persistence_tasks.append(asyncio.get_event_loop().run_in_executor(None, task))
                
            except Exception as e:
                self.logger.error(f"Data persistence task failed: {task.__name__}: {e}")
                self.shutdown_stats["errors"].append(f"Persistence {task.__name__}: {e}")
        
        # Execute all persistence tasks concurrently
        if persistence_tasks:
            try:
                await asyncio.wait_for(
                    asyncio.gather(*persistence_tasks, return_exceptions=True),
                    timeout=10.0  # 10 second timeout for data persistence
                )
                
                self.shutdown_stats["data_persisted"] = len(persistence_tasks)
                self.logger.info(f"Data persistence completed: {len(persistence_tasks)} tasks")
                
            except asyncio.TimeoutError:
                self.logger.warning("Data persistence timed out")
                self.shutdown_stats["errors"].append("Data persistence timeout")
    
    async def _shutdown_components(self):
        """Shutdown components in priority order"""
        self.logger.info("Shutting down components...")
        
        # Sort components by priority
        sorted_components = sorted(
            self.components.items(),
            key=lambda x: x[1]["priority"]
        )
        
        for name, component_info in sorted_components:
            try:
                component = component_info["component"]
                shutdown_method = component_info["shutdown_method"]
                
                self.logger.info(f"Shutting down component: {name}")
                
                # Get shutdown method
                if hasattr(component, shutdown_method):
                    method = getattr(component, shutdown_method)
                    
                    # Execute shutdown method
                    if asyncio.iscoroutinefunction(method):
                        await asyncio.wait_for(method(), timeout=5.0)
                    else:
                        await asyncio.get_event_loop().run_in_executor(None, method)
                    
                    self.shutdown_stats["components_shutdown"] += 1
                    self.logger.info(f"Component '{name}' shutdown completed")
                    
                else:
                    self.logger.warning(f"Component '{name}' has no shutdown method '{shutdown_method}'")
                
            except asyncio.TimeoutError:
                self.logger.warning(f"Component '{name}' shutdown timed out")
                self.shutdown_stats["errors"].append(f"Component {name}: timeout")
                
            except Exception as e:
                self.logger.error(f"Component '{name}' shutdown failed: {e}")
                self.shutdown_stats["errors"].append(f"Component {name}: {e}")
    
    async def _execute_cleanup_tasks(self):
        """Execute cleanup tasks"""
        self.logger.info("Executing cleanup tasks...")
        
        for task in self.cleanup_tasks:
            try:
                if asyncio.iscoroutinefunction(task):
                    await task()
                else:
                    await asyncio.get_event_loop().run_in_executor(None, task)
                
                self.logger.info(f"Cleanup task executed: {task.__name__}")
                
            except Exception as e:
                self.logger.error(f"Cleanup task failed: {task.__name__}: {e}")
                self.shutdown_stats["errors"].append(f"Cleanup {task.__name__}: {e}")
    
    async def _final_cleanup(self):
        """Perform final cleanup"""
        self.logger.info("Performing final cleanup...")
        
        try:
            # Close any remaining asyncio tasks
            tasks = [task for task in asyncio.all_tasks() if not task.done()]
            if tasks:
                self.logger.info(f"Cancelling {len(tasks)} remaining tasks")
                for task in tasks:
                    task.cancel()
                
                # Wait for tasks to complete cancellation
                await asyncio.gather(*tasks, return_exceptions=True)
            
            # Final log flush
            for handler in logging.getLogger().handlers:
                if hasattr(handler, 'flush'):
                    handler.flush()
            
        except Exception as e:
            self.logger.error(f"Final cleanup failed: {e}")
            self.shutdown_stats["errors"].append(f"Final cleanup: {e}")
    
    def _log_shutdown_statistics(self):
        """Log shutdown statistics"""
        stats = self.shutdown_stats
        
        self.logger.info("Shutdown Statistics:")
        self.logger.info(f"  Duration: {stats['duration']:.2f}s")
        self.logger.info(f"  Components shutdown: {stats['components_shutdown']}")
        self.logger.info(f"  Data persistence tasks: {stats['data_persisted']}")
        self.logger.info(f"  Errors: {len(stats['errors'])}")
        
        if stats["errors"]:
            self.logger.warning("Shutdown errors:")
            for error in stats["errors"]:
                self.logger.warning(f"  - {error}")
    
    @asynccontextmanager
    async def managed_shutdown(self):
        """Context manager for managed shutdown"""
        try:
            yield self
        finally:
            if not self.shutdown_initiated:
                await self.shutdown()


# Global shutdown manager instance
shutdown_manager = GracefulShutdownManager()


# Convenience functions
def register_for_shutdown(name: str, component: Any, shutdown_method: str = "shutdown"):
    """Register component for graceful shutdown"""
    shutdown_manager.register_component(name, component, shutdown_method)


def add_shutdown_hook(hook: Callable):
    """Add shutdown hook"""
    shutdown_manager.add_shutdown_hook(hook)


def add_cleanup_task(task: Callable):
    """Add cleanup task"""
    shutdown_manager.add_cleanup_task(task)


def add_data_persistence_task(task: Callable):
    """Add data persistence task"""
    shutdown_manager.add_data_persistence_task(task)


async def initiate_shutdown():
    """Initiate graceful shutdown"""
    await shutdown_manager.shutdown()


# Example usage for PyGent Factory components
async def example_component_registration():
    """Example of how to register components for graceful shutdown"""
    
    # Example component with async shutdown
    class ExampleComponent:
        async def shutdown(self):
            logger.info("Example component shutting down...")
            await asyncio.sleep(0.1)  # Simulate cleanup
    
    # Register component
    component = ExampleComponent()
    register_for_shutdown("example_component", component)
    
    # Add shutdown hook
    async def save_state():
        logger.info("Saving application state...")
        # Save critical state here
    
    add_shutdown_hook(save_state)
    
    # Add cleanup task
    def cleanup_temp_files():
        logger.info("Cleaning up temporary files...")
        # Cleanup logic here
    
    add_cleanup_task(cleanup_temp_files)
    
    # Add data persistence task
    async def persist_agent_data():
        logger.info("Persisting agent data...")
        # Data persistence logic here
    
    add_data_persistence_task(persist_agent_data)
