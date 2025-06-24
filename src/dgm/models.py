from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class ImprovementType(str, Enum):
    PARAMETER_TUNING = "parameter_tuning"
    ALGORITHM_MODIFICATION = "algorithm_modification"
    ARCHITECTURE_CHANGE = "architecture_change"
    CONFIGURATION_UPDATE = "configuration_update"

class ImprovementStatus(str, Enum):
    PROPOSED = "proposed"
    TESTING = "testing"
    VALIDATED = "validated"
    APPLIED = "applied"
    REJECTED = "rejected"
    FAILED = "failed"

class PerformanceMetric(BaseModel):
    name: str
    value: float
    unit: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    context: Optional[Dict[str, Any]] = None

class ImprovementCandidate(BaseModel):
    id: str
    agent_id: str
    improvement_type: ImprovementType
    description: str
    code_changes: Dict[str, str]  # filename -> new_code
    expected_improvement: float
    risk_level: float = Field(ge=0.0, le=1.0)
    status: ImprovementStatus = ImprovementStatus.PROPOSED
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)
    
class ValidationResult(BaseModel):
    candidate_id: str
    success: bool
    performance_before: List[PerformanceMetric]
    performance_after: List[PerformanceMetric]
    improvement_score: float
    safety_score: float = Field(ge=0.0, le=1.0)
    test_results: Dict[str, Any]
    validation_time: float  # seconds
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DGMState(BaseModel):
    """Current state of the Darwin Gödel Machine"""
    agent_id: str
    current_performance: Dict[str, PerformanceMetric]
    improvement_history: List[ImprovementCandidate]
    active_experiments: List[str]  # candidate IDs
    best_configuration: Dict[str, Any]
    generation: int = 0
    last_improvement: Optional[datetime] = None
    
class EvolutionParameters(BaseModel):
    """Parameters controlling the evolution process"""
    mutation_rate: float = Field(default=0.1, ge=0.0, le=1.0)
    selection_pressure: float = Field(default=0.5, ge=0.0, le=1.0)
    population_size: int = Field(default=10, ge=1)
    max_generations: int = Field(default=100, ge=1)
    crossover_rate: float = Field(default=0.7, ge=0.0, le=1.0)
    elitism_ratio: float = Field(default=0.1, ge=0.0, le=1.0)
    
class GodelConstraint(BaseModel):
    """Formal constraint in the Gödel system"""
    id: str
    name: str
    description: str
    logical_expression: str
    priority: int = Field(default=1, ge=1)
    is_hard_constraint: bool = True
    
class SelfReflectionResult(BaseModel):
    """Result of agent self-reflection"""
    agent_id: str
    reflection_prompt: str
    insights: List[str]
    improvement_suggestions: List[str]
    confidence_score: float = Field(ge=0.0, le=1.0)
    timestamp: datetime = Field(default_factory=datetime.utcnow)

class DGMArchiveEntry(BaseModel):
    id: str
    agent_id: str
    improvement_candidate: ImprovementCandidate
    validation_result: Optional[ValidationResult] = None
    applied: bool = False
    application_timestamp: Optional[datetime] = None
    rollback_info: Optional[Dict[str, Any]] = None

class DGMProgram(BaseModel):
    """Represents a program in the DGM system"""
    id: str
    code: str
    description: str
    author: str = "system"
    version: str = "1.0"
    timestamp: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    tags: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None

class DGMEvolution(BaseModel):
    """Represents an evolution step in the DGM system"""
    id: str
    parent_id: Optional[str] = None
    program_id: str
    fitness: float
    generation: int
    timestamp: float = Field(default_factory=lambda: datetime.utcnow().timestamp())
    mutations: List[str] = Field(default_factory=list)
    metadata: Optional[Dict[str, Any]] = None
