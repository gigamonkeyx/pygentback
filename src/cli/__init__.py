"""
PyGent Factory CLI Module

Provides command-line interface for the PyGent Factory Testing Framework
with comprehensive recipe testing, optimization, and management capabilities.
"""

from .main import cli
from .commands import *

__all__ = ['cli']
