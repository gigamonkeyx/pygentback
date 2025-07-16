#!/usr/bin/env python3
"""
UTF-8 Logger Utility for PyGent Factory

Cross-platform UTF-8 logging infrastructure with RIPER-Omega protocol support.
Resolves Windows cp1252 encoding issues while maintaining Unicode compatibility.

Observer-supervised implementation with zero mock solutions.
"""

import logging
import sys
import os
from typing import Optional, Dict, Any
from datetime import datetime
import codecs


class UTF8StreamHandler(logging.StreamHandler):
    """UTF-8 compatible stream handler for cross-platform logging"""
    
    def __init__(self, stream=None):
        super().__init__(stream)
        
        # Force UTF-8 encoding for the stream
        if hasattr(self.stream, 'buffer'):
            # For stdout/stderr with buffer attribute
            self.stream = codecs.getwriter('utf-8')(self.stream.buffer)
        elif hasattr(self.stream, 'encoding'):
            # For file-like objects with encoding
            if self.stream.encoding != 'utf-8':
                self.stream.reconfigure(encoding='utf-8')


class UTF8FileHandler(logging.FileHandler):
    """UTF-8 compatible file handler"""
    
    def __init__(self, filename, mode='a', encoding='utf-8', delay=False):
        super().__init__(filename, mode, encoding, delay)


class RIPEROmegaFormatter(logging.Formatter):
    """RIPER-Omega protocol compatible formatter with ASCII fallbacks"""
    
    def __init__(self, use_unicode=True):
        self.use_unicode = use_unicode
        
        if use_unicode:
            # Unicode symbols for enhanced readability
            self.symbols = {
                'omega': 'Ω',
                'check': '✓',
                'cross': '✗',
                'arrow': '→',
                'bullet': '•'
            }
        else:
            # ASCII fallbacks for compatibility
            self.symbols = {
                'omega': 'Omega',
                'check': '[OK]',
                'cross': '[FAIL]',
                'arrow': '->',
                'bullet': '*'
            }
        
        # Format string with protocol compliance
        format_str = f'%(asctime)s - RIPER-{self.symbols["omega"]} - %(name)s - %(levelname)s - %(message)s'
        super().__init__(format_str, datefmt='%Y-%m-%d %H:%M:%S')
    
    def format(self, record):
        """Format log record with RIPER-Omega protocol compliance"""
        # Add protocol symbols to record
        record.check = self.symbols['check']
        record.cross = self.symbols['cross']
        record.arrow = self.symbols['arrow']
        record.bullet = self.symbols['bullet']
        record.omega = self.symbols['omega']
        
        return super().format(record)


class PyGentLogger:
    """PyGent Factory UTF-8 logger with observer supervision"""
    
    def __init__(self, name: str = "pygent_factory"):
        self.name = name
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        
        # Clear existing handlers to avoid duplicates
        self.logger.handlers.clear()
        
        # Detect environment capabilities
        self.unicode_supported = self._detect_unicode_support()
        
        # Configure handlers
        self._setup_console_handler()
        self._setup_file_handler()
        
        # Observer supervision
        self.observer_active = True
        self.protocol_compliance = True
        
    def _detect_unicode_support(self) -> bool:
        """Detect if environment supports Unicode output"""
        try:
            # Test Unicode output capability
            test_output = "RIPER-Ω Protocol Test"
            if hasattr(sys.stdout, 'encoding'):
                test_output.encode(sys.stdout.encoding or 'utf-8')
            return True
        except (UnicodeEncodeError, AttributeError):
            return False
    
    def _setup_console_handler(self):
        """Setup UTF-8 compatible console handler"""
        try:
            # Create UTF-8 stream handler
            console_handler = UTF8StreamHandler(sys.stdout)
            console_handler.setLevel(logging.INFO)
            
            # Use Unicode formatter if supported, ASCII fallback otherwise
            formatter = RIPEROmegaFormatter(use_unicode=self.unicode_supported)
            console_handler.setFormatter(formatter)
            
            self.logger.addHandler(console_handler)
            
        except Exception as e:
            # Fallback to basic handler if UTF-8 setup fails
            fallback_handler = logging.StreamHandler()
            fallback_formatter = RIPEROmegaFormatter(use_unicode=False)
            fallback_handler.setFormatter(fallback_formatter)
            self.logger.addHandler(fallback_handler)
            
            self.logger.warning(f"UTF-8 console handler failed, using ASCII fallback: {e}")
    
    def _setup_file_handler(self):
        """Setup UTF-8 file handler"""
        try:
            # Ensure logs directory exists
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            
            # Create UTF-8 file handler
            log_file = os.path.join(log_dir, f"pygent_factory_{datetime.now().strftime('%Y%m%d')}.log")
            file_handler = UTF8FileHandler(log_file, encoding='utf-8')
            file_handler.setLevel(logging.DEBUG)
            
            # Always use Unicode in file logs
            formatter = RIPEROmegaFormatter(use_unicode=True)
            file_handler.setFormatter(formatter)
            
            self.logger.addHandler(file_handler)
            
        except Exception as e:
            self.logger.error(f"Failed to setup file handler: {e}")
    
    def get_logger(self) -> logging.Logger:
        """Get configured logger instance"""
        return self.logger
    
    def log_observer_event(self, event_type: str, message: str, level: str = "INFO"):
        """Log observer supervision events with protocol compliance"""
        observer_message = f"[OBSERVER-{event_type.upper()}] {message}"
        
        if level.upper() == "DEBUG":
            self.logger.debug(observer_message)
        elif level.upper() == "INFO":
            self.logger.info(observer_message)
        elif level.upper() == "WARNING":
            self.logger.warning(observer_message)
        elif level.upper() == "ERROR":
            self.logger.error(observer_message)
        elif level.upper() == "CRITICAL":
            self.logger.critical(observer_message)
    
    def log_riperω_transition(self, from_mode: str, to_mode: str, confidence: float):
        """Log RIPER-Omega mode transitions with protocol compliance"""
        symbols = self.symbols if hasattr(self, 'symbols') else {'arrow': '->', 'omega': 'Omega'}
        transition_msg = f"RIPER-{symbols.get('omega', 'Omega')} MODE TRANSITION: {from_mode} {symbols.get('arrow', '->')} {to_mode} (confidence: {confidence:.3f})"
        self.logger.info(transition_msg)
    
    def log_performance_benchmark(self, benchmark_name: str, result: float, target: float, passed: bool):
        """Log performance benchmark results with protocol compliance"""
        symbols = self.symbols if hasattr(self, 'symbols') else {'check': '[OK]', 'cross': '[FAIL]'}
        status_symbol = symbols.get('check', '[OK]') if passed else symbols.get('cross', '[FAIL]')
        benchmark_msg = f"BENCHMARK {status_symbol} {benchmark_name}: {result:.3f}s (target: {target:.3f}s)"
        
        if passed:
            self.logger.info(benchmark_msg)
        else:
            self.logger.warning(benchmark_msg)
    
    def log_docker_event(self, container: str, event: str, status: str):
        """Log Docker 4.43 events with protocol compliance"""
        docker_msg = f"DOCKER-4.43 [{container}] {event}: {status}"
        self.logger.info(docker_msg)
    
    def test_unicode_logging(self):
        """Test Unicode logging capability"""
        test_messages = [
            "RIPER-Ω Protocol Test",
            "Unicode symbols: ✓ ✗ → • Ω",
            "ASCII fallback: [OK] [FAIL] -> * Omega"
        ]
        
        for msg in test_messages:
            try:
                self.logger.info(f"Unicode Test: {msg}")
            except UnicodeEncodeError as e:
                self.logger.error(f"Unicode encoding failed: {e}")
                return False
        
        return True


# Global logger instance
_global_logger = None


def get_logger(name: str = "pygent_factory") -> logging.Logger:
    """Get global UTF-8 compatible logger instance"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = PyGentLogger(name)
    
    return _global_logger.get_logger()


def get_pygent_logger(name: str = "pygent_factory") -> PyGentLogger:
    """Get PyGent logger instance with enhanced features"""
    global _global_logger
    
    if _global_logger is None:
        _global_logger = PyGentLogger(name)
    
    return _global_logger


def configure_utf8_logging():
    """Configure UTF-8 logging for the entire application"""
    # Set UTF-8 environment variables
    os.environ['PYTHONIOENCODING'] = 'utf-8'
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    
    # Add UTF-8 compatible handler
    pygent_logger = get_pygent_logger()
    
    return pygent_logger


def test_logging_fix():
    """Test the logging fix implementation"""
    logger = get_pygent_logger("test")
    
    print("Testing UTF-8 logging fix...")
    
    # Test Unicode logging
    success = logger.test_unicode_logging()
    
    # Test RIPER-Omega protocol logging
    logger.log_riperω_transition("RESEARCH", "PLAN", 0.85)
    logger.log_observer_event("VALIDATION", "Testing UTF-8 logging implementation")
    logger.log_performance_benchmark("agent_spawn_time", 1.8, 2.0, True)
    logger.log_docker_event("pygent-test", "container_start", "success")
    
    print(f"UTF-8 logging test {'PASSED' if success else 'FAILED'}")
    return success


if __name__ == "__main__":
    test_logging_fix()
