"""
Predictive Data Models

Data structures and models for the predictive optimization system.
"""

from typing import List, Dict, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import uuid


class PredictionType(Enum):
    """Types of predictions"""
    PERFORMANCE = "performance"
    SUCCESS_RATE = "success_rate"
    RESOURCE_USAGE = "resource_usage"
    LATENCY = "latency"
    QUALITY = "quality"
    COST = "cost"
    RISK = "risk"
    TREND = "trend"


class OptimizationType(Enum):
    """Types of optimizations"""
    SINGLE_OBJECTIVE = "single_objective"
    MULTI_OBJECTIVE = "multi_objective"
    CONSTRAINED = "constrained"
    UNCONSTRAINED = "unconstrained"
    DISCRETE = "discrete"
    CONTINUOUS = "continuous"


class RecommendationType(Enum):
    """Types of recommendations"""
    PARAMETER_TUNING = "parameter_tuning"
    ARCHITECTURE_CHANGE = "architecture_change"
    RESOURCE_ALLOCATION = "resource_allocation"
    WORKFLOW_OPTIMIZATION = "workflow_optimization"
    PERFORMANCE_IMPROVEMENT = "performance_improvement"
    COST_REDUCTION = "cost_reduction"
    RISK_MITIGATION = "risk_mitigation"


class Priority(Enum):
    """Priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


@dataclass
class PredictionMetrics:
    """Metrics for prediction model performance"""
    accuracy: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    f1_score: float = 0.0
    mean_absolute_error: float = 0.0
    mean_squared_error: float = 0.0
    
    # Prediction statistics
    total_predictions: int = 0
    correct_predictions: int = 0
    avg_confidence: float = 0.0
    
    # Timing metrics
    avg_prediction_time_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.utcnow)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'mean_absolute_error': self.mean_absolute_error,
            'mean_squared_error': self.mean_squared_error,
            'total_predictions': self.total_predictions,
            'correct_predictions': self.correct_predictions,
            'avg_confidence': self.avg_confidence,
            'avg_prediction_time_ms': self.avg_prediction_time_ms,
            'last_updated': self.last_updated.isoformat()
        }


@dataclass
class Prediction:
    """Prediction result from a predictive model"""
    prediction_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    model_name: str = ""
    prediction_type: Union[PredictionType, str] = PredictionType.PERFORMANCE
    
    # Prediction data
    predicted_value: Any = None
    confidence: float = 0.0
    uncertainty: Optional[float] = None
    
    # Input and context
    input_features: Dict[str, Any] = field(default_factory=dict)
    prediction_context: Dict[str, Any] = field(default_factory=dict)
    
    # Validation and feedback
    actual_value: Optional[Any] = None
    prediction_error: Optional[float] = None
    feedback_received: bool = False
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    model_version: str = "1.0"
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.prediction_type, str):
            try:
                self.prediction_type = PredictionType(self.prediction_type)
            except ValueError:
                self.prediction_type = PredictionType.PERFORMANCE
    
    def is_expired(self) -> bool:
        """Check if prediction has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def calculate_error(self) -> Optional[float]:
        """Calculate prediction error if actual value is available"""
        if self.actual_value is None or self.predicted_value is None:
            return None
        
        try:
            error = abs(float(self.actual_value) - float(self.predicted_value))
            self.prediction_error = error
            return error
        except (ValueError, TypeError):
            return None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'prediction_id': self.prediction_id,
            'model_name': self.model_name,
            'prediction_type': self.prediction_type.value if isinstance(self.prediction_type, PredictionType) else self.prediction_type,
            'predicted_value': self.predicted_value,
            'confidence': self.confidence,
            'uncertainty': self.uncertainty,
            'input_features': self.input_features,
            'prediction_context': self.prediction_context,
            'actual_value': self.actual_value,
            'prediction_error': self.prediction_error,
            'feedback_received': self.feedback_received,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'model_version': self.model_version,
            'metadata': self.metadata
        }


@dataclass
class OptimizationConstraint:
    """Constraint for optimization problems"""
    constraint_name: str
    constraint_type: str  # 'equality', 'inequality', 'bound'
    constraint_function: Optional[str] = None  # Function definition as string
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    tolerance: float = 1e-6
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'constraint_name': self.constraint_name,
            'constraint_type': self.constraint_type,
            'constraint_function': self.constraint_function,
            'lower_bound': self.lower_bound,
            'upper_bound': self.upper_bound,
            'tolerance': self.tolerance
        }


@dataclass
class OptimizationObjective:
    """Objective function for optimization"""
    objective_name: str
    objective_type: str  # 'minimize', 'maximize'
    weight: float = 1.0
    priority: Priority = Priority.MEDIUM
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'objective_name': self.objective_name,
            'objective_type': self.objective_type,
            'weight': self.weight,
            'priority': self.priority.value
        }


@dataclass
class Optimization:
    """Optimization result"""
    optimization_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    optimizer_name: str = ""
    optimization_type: Union[OptimizationType, str] = OptimizationType.SINGLE_OBJECTIVE
    
    # Problem definition
    parameter_space: Dict[str, Any] = field(default_factory=dict)
    objectives: List[OptimizationObjective] = field(default_factory=list)
    constraints: List[OptimizationConstraint] = field(default_factory=list)
    
    # Results
    optimal_parameters: Dict[str, Any] = field(default_factory=dict)
    optimal_value: float = 0.0
    optimal_values: List[float] = field(default_factory=list)  # For multi-objective
    
    # Optimization process
    iterations: int = 0
    function_evaluations: int = 0
    optimization_time_seconds: float = 0.0
    convergence_achieved: bool = False
    convergence_criteria: Dict[str, float] = field(default_factory=dict)
    
    # Quality metrics
    improvement_percentage: float = 0.0
    confidence_interval: Optional[Tuple[float, float]] = None
    robustness_score: float = 0.0
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    algorithm_config: Dict[str, Any] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.optimization_type, str):
            try:
                self.optimization_type = OptimizationType(self.optimization_type)
            except ValueError:
                self.optimization_type = OptimizationType.SINGLE_OBJECTIVE
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'optimization_id': self.optimization_id,
            'optimizer_name': self.optimizer_name,
            'optimization_type': self.optimization_type.value if isinstance(self.optimization_type, OptimizationType) else self.optimization_type,
            'parameter_space': self.parameter_space,
            'objectives': [obj.to_dict() for obj in self.objectives],
            'constraints': [const.to_dict() for const in self.constraints],
            'optimal_parameters': self.optimal_parameters,
            'optimal_value': self.optimal_value,
            'optimal_values': self.optimal_values,
            'iterations': self.iterations,
            'function_evaluations': self.function_evaluations,
            'optimization_time_seconds': self.optimization_time_seconds,
            'convergence_achieved': self.convergence_achieved,
            'convergence_criteria': self.convergence_criteria,
            'improvement_percentage': self.improvement_percentage,
            'confidence_interval': self.confidence_interval,
            'robustness_score': self.robustness_score,
            'created_at': self.created_at.isoformat(),
            'algorithm_config': self.algorithm_config,
            'metadata': self.metadata
        }


@dataclass
class Recommendation:
    """Recommendation for improvement or optimization"""
    recommendation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    recommendation_type: Union[RecommendationType, str] = RecommendationType.PARAMETER_TUNING
    
    # Recommendation content
    title: str = ""
    description: str = ""
    recommended_action: str = ""
    recommended_parameters: Dict[str, Any] = field(default_factory=dict)
    
    # Impact and confidence
    confidence: float = 0.0
    predicted_impact: Optional[float] = None
    risk_level: Priority = Priority.LOW
    effort_level: Priority = Priority.MEDIUM
    
    # Context and reasoning
    reasoning: List[str] = field(default_factory=list)
    explanation: str = ""
    supporting_evidence: List[str] = field(default_factory=list)
    
    # Implementation
    implementation_steps: List[str] = field(default_factory=list)
    estimated_implementation_time: Optional[str] = None
    required_resources: List[str] = field(default_factory=list)
    
    # Validation and feedback
    feedback_received: bool = False
    feedback_positive: Optional[bool] = None
    feedback_notes: Optional[str] = None
    implementation_status: str = "pending"  # pending, implemented, rejected
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    expires_at: Optional[datetime] = None
    priority: Priority = Priority.MEDIUM
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if isinstance(self.recommendation_type, str):
            try:
                self.recommendation_type = RecommendationType(self.recommendation_type)
            except ValueError:
                self.recommendation_type = RecommendationType.PARAMETER_TUNING
    
    def is_expired(self) -> bool:
        """Check if recommendation has expired"""
        if self.expires_at is None:
            return False
        return datetime.utcnow() > self.expires_at
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'recommendation_id': self.recommendation_id,
            'recommendation_type': self.recommendation_type.value if isinstance(self.recommendation_type, RecommendationType) else self.recommendation_type,
            'title': self.title,
            'description': self.description,
            'recommended_action': self.recommended_action,
            'recommended_parameters': self.recommended_parameters,
            'confidence': self.confidence,
            'predicted_impact': self.predicted_impact,
            'risk_level': self.risk_level.value,
            'effort_level': self.effort_level.value,
            'reasoning': self.reasoning,
            'explanation': self.explanation,
            'supporting_evidence': self.supporting_evidence,
            'implementation_steps': self.implementation_steps,
            'estimated_implementation_time': self.estimated_implementation_time,
            'required_resources': self.required_resources,
            'feedback_received': self.feedback_received,
            'feedback_positive': self.feedback_positive,
            'feedback_notes': self.feedback_notes,
            'implementation_status': self.implementation_status,
            'created_at': self.created_at.isoformat(),
            'expires_at': self.expires_at.isoformat() if self.expires_at else None,
            'priority': self.priority.value,
            'tags': self.tags,
            'metadata': self.metadata
        }


@dataclass
class FeatureImportance:
    """Feature importance for model interpretability"""
    feature_name: str
    importance_score: float
    importance_type: str = "permutation"  # permutation, gain, split, etc.
    confidence_interval: Optional[Tuple[float, float]] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'feature_name': self.feature_name,
            'importance_score': self.importance_score,
            'importance_type': self.importance_type,
            'confidence_interval': self.confidence_interval
        }


@dataclass
class ModelExplanation:
    """Explanation for model predictions"""
    explanation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prediction_id: str = ""
    explanation_type: str = "feature_importance"
    
    # Explanation content
    feature_importances: List[FeatureImportance] = field(default_factory=list)
    explanation_text: str = ""
    confidence_factors: Dict[str, float] = field(default_factory=dict)
    
    # Visualization data
    visualization_data: Dict[str, Any] = field(default_factory=dict)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    explanation_method: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'explanation_id': self.explanation_id,
            'prediction_id': self.prediction_id,
            'explanation_type': self.explanation_type,
            'feature_importances': [fi.to_dict() for fi in self.feature_importances],
            'explanation_text': self.explanation_text,
            'confidence_factors': self.confidence_factors,
            'visualization_data': self.visualization_data,
            'created_at': self.created_at.isoformat(),
            'explanation_method': self.explanation_method,
            'metadata': self.metadata
        }


# Utility functions for model validation
def validate_prediction(prediction: Prediction) -> bool:
    """Validate prediction structure"""
    if not prediction.prediction_id or not isinstance(prediction.prediction_id, str):
        return False
    
    if not prediction.model_name or not isinstance(prediction.model_name, str):
        return False
    
    if not (0.0 <= prediction.confidence <= 1.0):
        return False
    
    if prediction.uncertainty is not None and prediction.uncertainty < 0:
        return False
    
    return True


def validate_optimization(optimization: Optimization) -> bool:
    """Validate optimization structure"""
    if not optimization.optimization_id or not isinstance(optimization.optimization_id, str):
        return False
    
    if not optimization.optimizer_name or not isinstance(optimization.optimizer_name, str):
        return False
    
    if optimization.iterations < 0 or optimization.function_evaluations < 0:
        return False
    
    if optimization.optimization_time_seconds < 0:
        return False
    
    return True


def validate_recommendation(recommendation: Recommendation) -> bool:
    """Validate recommendation structure"""
    if not recommendation.recommendation_id or not isinstance(recommendation.recommendation_id, str):
        return False
    
    if not recommendation.title or not isinstance(recommendation.title, str):
        return False
    
    if not (0.0 <= recommendation.confidence <= 1.0):
        return False
    
    if recommendation.predicted_impact is not None and not isinstance(recommendation.predicted_impact, (int, float)):
        return False
    
    return True
