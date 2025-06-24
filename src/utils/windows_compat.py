"""
Windows Compatibility Utilities

Utilities to handle Windows-specific issues that cause import hanging
and other compatibility problems.
"""

import os
import sys
import time
import logging
from typing import Any, Optional, Callable
import threading
import signal

logger = logging.getLogger(__name__)


class WindowsCompatibilityManager:
    """
    Windows Compatibility Manager.
    
    Handles Windows-specific issues including import hanging,
    console problems, and process management.
    """
    
    def __init__(self):
        self.is_windows = sys.platform == 'win32'
        self.setup_complete = False
        
    def setup_windows_environment(self):
        """Setup Windows environment for optimal Python execution"""
        if not self.is_windows:
            return
        
        try:
            # Set environment variables
            os.environ['PYTHONDONTWRITEBYTECODE'] = '1'
            os.environ['PYTHONUNBUFFERED'] = '1'
            os.environ['PYTHONIOENCODING'] = 'utf-8'
            os.environ['PYTHONUTF8'] = '1'
            
            # Configure console
            self._configure_windows_console()
            
            # Setup signal handling
            self._setup_signal_handling()
            
            self.setup_complete = True
            logger.info("Windows environment setup completed")
            
        except Exception as e:
            logger.error(f"Windows environment setup failed: {e}")
    
    def _configure_windows_console(self):
        """Configure Windows console for better compatibility"""
        try:
            # Enable ANSI escape sequences
            import ctypes
            kernel32 = ctypes.windll.kernel32
            
            # Get console handle
            console_handle = kernel32.GetStdHandle(-11)  # STD_OUTPUT_HANDLE
            
            # Enable virtual terminal processing
            mode = ctypes.c_ulong()
            kernel32.GetConsoleMode(console_handle, ctypes.byref(mode))
            mode.value |= 0x0004  # ENABLE_VIRTUAL_TERMINAL_PROCESSING
            kernel32.SetConsoleMode(console_handle, mode)
            
            logger.debug("Windows console configured for ANSI support")
            
        except Exception as e:
            logger.debug(f"Could not configure Windows console: {e}")
    
    def _setup_signal_handling(self):
        """Setup signal handling for Windows"""
        try:
            # Windows doesn't support SIGALRM, use alternative
            if hasattr(signal, 'SIGBREAK'):
                signal.signal(signal.SIGBREAK, self._signal_handler)
            signal.signal(signal.SIGINT, self._signal_handler)
            signal.signal(signal.SIGTERM, self._signal_handler)
            
        except Exception as e:
            logger.debug(f"Could not setup signal handling: {e}")
    
    def _signal_handler(self, signum, frame):
        """Handle signals gracefully"""
        logger.info(f"Received signal {signum}, shutting down gracefully...")
        # Could add cleanup logic here
    
    def safe_import(self, module_name: str, timeout: float = 30.0) -> Optional[Any]:
        """
        Safely import a module with timeout protection.
        
        Args:
            module_name: Name of module to import
            timeout: Timeout in seconds
            
        Returns:
            Imported module or None if failed
        """
        if not self.is_windows:
            # On non-Windows, use normal import
            try:
                return __import__(module_name, fromlist=[''])
            except Exception as e:
                logger.error(f"Import failed for {module_name}: {e}")
                return None
        
        # Windows-specific safe import with timeout
        return self._import_with_timeout(module_name, timeout)
    
    def _import_with_timeout(self, module_name: str, timeout: float) -> Optional[Any]:
        """Import module with timeout using threading"""
        result = {'module': None, 'error': None, 'completed': False}
        
        def import_worker():
            try:
                result['module'] = __import__(module_name, fromlist=[''])
                result['completed'] = True
            except Exception as e:
                result['error'] = e
                result['completed'] = True
        
        # Start import in separate thread
        import_thread = threading.Thread(target=import_worker, daemon=True)
        import_thread.start()
        
        # Wait for completion or timeout
        import_thread.join(timeout)
        
        if not result['completed']:
            logger.error(f"Import of {module_name} timed out after {timeout}s")
            return None
        
        if result['error']:
            logger.error(f"Import of {module_name} failed: {result['error']}")
            return None
        
        logger.debug(f"Successfully imported {module_name}")
        return result['module']
    
    def safe_psutil_operations(self) -> dict:
        """
        Safely perform psutil operations that commonly hang on Windows.
        
        Returns:
            Dictionary with system metrics or safe defaults
        """
        try:
            import psutil
            
            # Use shorter intervals and timeouts
            metrics = {}
            
            # CPU usage with short interval
            try:
                metrics['cpu_percent'] = psutil.cpu_percent(interval=0.1)
            except:
                metrics['cpu_percent'] = 0.0
            
            # Memory usage
            try:
                memory = psutil.virtual_memory()
                metrics['memory_percent'] = memory.percent
                metrics['memory_available'] = memory.available
            except:
                metrics['memory_percent'] = 0.0
                metrics['memory_available'] = 0
            
            # Disk usage with Windows-compatible path
            try:
                disk_path = 'C:\\' if self.is_windows else '/'
                disk = psutil.disk_usage(disk_path)
                metrics['disk_percent'] = (disk.used / disk.total) * 100
            except:
                metrics['disk_percent'] = 0.0
            
            # Process count
            try:
                metrics['process_count'] = len(psutil.pids())
            except:
                metrics['process_count'] = 0
            
            # Network I/O
            try:
                network = psutil.net_io_counters()
                metrics['network_bytes_sent'] = network.bytes_sent
                metrics['network_bytes_recv'] = network.bytes_recv
            except:
                metrics['network_bytes_sent'] = 0
                metrics['network_bytes_recv'] = 0
            
            # Load average (Windows doesn't have this)
            if hasattr(psutil, 'getloadavg'):
                try:
                    metrics['load_average'] = psutil.getloadavg()[0]
                except:
                    metrics['load_average'] = metrics['cpu_percent'] / 100.0
            else:
                metrics['load_average'] = metrics['cpu_percent'] / 100.0
            
            # Boot time
            try:
                metrics['uptime_seconds'] = time.time() - psutil.boot_time()
            except:
                metrics['uptime_seconds'] = 0.0
            
            return metrics
            
        except Exception as e:
            logger.error(f"psutil operations failed: {e}")
            # Return safe defaults
            return {
                'cpu_percent': 0.0,
                'memory_percent': 0.0,
                'memory_available': 0,
                'disk_percent': 0.0,
                'process_count': 0,
                'network_bytes_sent': 0,
                'network_bytes_recv': 0,
                'load_average': 0.0,
                'uptime_seconds': 0.0
            }
    
    def create_safe_subprocess(self, *args, **kwargs):
        """Create subprocess with Windows-compatible settings"""
        if not self.is_windows:
            import subprocess
            return subprocess.Popen(*args, **kwargs)
        
        # Windows-specific subprocess creation
        import subprocess
        
        # Add Windows-specific flags
        creation_flags = kwargs.get('creationflags', 0)
        creation_flags |= 0x80000000  # CREATE_NO_WINDOW
        creation_flags |= 0x00000008  # DETACHED_PROCESS
        kwargs['creationflags'] = creation_flags
        
        # Set shell to False for better compatibility
        kwargs['shell'] = False
        
        return subprocess.Popen(*args, **kwargs)
    
    def get_safe_temp_dir(self) -> str:
        """Get safe temporary directory for Windows"""
        if self.is_windows:
            # Use Windows-specific temp directory
            temp_dir = os.environ.get('TEMP', os.environ.get('TMP', 'C:\\temp'))
        else:
            temp_dir = '/tmp'
        
        # Ensure directory exists
        os.makedirs(temp_dir, exist_ok=True)
        return temp_dir
    
    def fix_path_separators(self, path: str) -> str:
        """Fix path separators for current platform"""
        if self.is_windows:
            return path.replace('/', '\\')
        else:
            return path.replace('\\', '/')
    
    def is_admin(self) -> bool:
        """Check if running with administrator privileges"""
        if not self.is_windows:
            return os.geteuid() == 0
        
        try:
            import ctypes
            return ctypes.windll.shell32.IsUserAnAdmin()
        except:
            return False
    
    def get_system_info(self) -> dict:
        """Get safe system information"""
        info = {
            'platform': sys.platform,
            'python_version': sys.version,
            'is_windows': self.is_windows,
            'setup_complete': self.setup_complete,
            'is_admin': self.is_admin()
        }
        
        if self.is_windows:
            try:
                import platform
                info['windows_version'] = platform.win32_ver()
            except:
                info['windows_version'] = 'unknown'
        
        return info


# Global instance
windows_compat = WindowsCompatibilityManager()


def setup_windows_compatibility():
    """Setup Windows compatibility (call this early in application startup)"""
    windows_compat.setup_windows_environment()


def safe_import(module_name: str, timeout: float = 30.0) -> Optional[Any]:
    """Safely import a module with Windows compatibility"""
    return windows_compat.safe_import(module_name, timeout)


def get_safe_system_metrics() -> dict:
    """Get system metrics safely on Windows"""
    return windows_compat.safe_psutil_operations()


# Auto-setup on import
if windows_compat.is_windows:
    setup_windows_compatibility()
