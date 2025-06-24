"""
API Module

REST API server for PyGent Factory AI system providing endpoints
for reasoning, evolution, search, and system management.
"""

from .server import create_app, APIServer
from .models import (
    ReasoningRequest,
    ReasoningResponse,
    EvolutionRequest,
    EvolutionResponse,
    SearchRequest,
    SearchResponse,
    HealthResponse
)

__all__ = [
    'create_app',
    'APIServer',
    'ReasoningRequest',
    'ReasoningResponse',
    'EvolutionRequest',
    'EvolutionResponse',
    'SearchRequest',
    'SearchResponse',
    'HealthResponse'
]
