"""
Predictive Core Components

Core engines for prediction, optimization, and recommendation generation.
"""

import logging
import asyncio
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from abc import ABC, abstractmethod

from .models import Prediction, Optimization, Recommendation, PredictionMetrics

logger = logging.getLogger(__name__)


class BasePredictiveModel(ABC):
    """
    Abstract base class for predictive models.
    """
    
    def __init__(self, model_name: str, model_type: str):
        self.model_name = model_name
        self.model_type = model_type
        self.is_trained = False
        self.training_data_size = 0
        self.last_training_time: Optional[datetime] = None
        
        # Model performance metrics
        self.metrics = PredictionMetrics()
        
        # Model configuration
        self.config = {
            'min_training_samples': 10,
            'retrain_threshold': 0.1,  # Retrain if accuracy drops below this
            'max_prediction_age_hours': 24,
            'confidence_threshold': 0.7
        }
    
    @abstractmethod
    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train the model with provided data"""
        pass
    
    @abstractmethod
    async def predict(self, input_data: Dict[str, Any]) -> Prediction:
        """Make prediction based on input data"""
        pass
    
    @abstractmethod
    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        pass
    
    def needs_retraining(self) -> bool:
        """Check if model needs retraining"""
        if not self.is_trained:
            return True
        
        # Check if accuracy has dropped
        if self.metrics.accuracy < self.config['retrain_threshold']:
            return True
        
        # Check if model is too old
        if self.last_training_time:
            age = datetime.utcnow() - self.last_training_time
            if age > timedelta(days=7):  # Retrain weekly
                return True
        
        return False
    
    def update_metrics(self, actual_value: Any, predicted_value: Any, confidence: float):
        """Update model performance metrics"""
        # Simple accuracy calculation (can be enhanced for specific model types)
        is_correct = abs(float(actual_value) - float(predicted_value)) < 0.1
        
        self.metrics.total_predictions += 1
        if is_correct:
            self.metrics.correct_predictions += 1
        
        self.metrics.accuracy = self.metrics.correct_predictions / self.metrics.total_predictions
        
        # Update confidence metrics
        total_confidence = (self.metrics.avg_confidence * (self.metrics.total_predictions - 1) + confidence)
        self.metrics.avg_confidence = total_confidence / self.metrics.total_predictions


class PredictiveEngine:
    """
    Main engine for coordinating predictive models and making predictions.
    """
    
    def __init__(self):
        self.models: Dict[str, BasePredictiveModel] = {}
        self.prediction_history: List[Prediction] = []
        self.feedback_queue: asyncio.Queue = asyncio.Queue()
        
        # Engine configuration
        self.config = {
            'max_history_size': 10000,
            'enable_ensemble_predictions': True,
            'auto_retrain_models': True,
            'prediction_timeout_seconds': 30.0
        }
        
        # Statistics
        self.stats = {
            'total_predictions': 0,
            'successful_predictions': 0,
            'failed_predictions': 0,
            'avg_prediction_time_ms': 0.0
        }
        
        # Background tasks
        self.is_running = False
        self.feedback_processor_task: Optional[asyncio.Task] = None
    
    async def start(self):
        """Start the predictive engine"""
        if self.is_running:
            return
        
        self.is_running = True
        
        # Start background tasks
        self.feedback_processor_task = asyncio.create_task(self._process_feedback_loop())
        
        logger.info("Predictive engine started")
    
    async def stop(self):
        """Stop the predictive engine"""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel background tasks
        if self.feedback_processor_task:
            self.feedback_processor_task.cancel()
            try:
                await self.feedback_processor_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Predictive engine stopped")
    
    def register_model(self, model: BasePredictiveModel):
        """Register a predictive model"""
        self.models[model.model_name] = model
        logger.info(f"Registered predictive model: {model.model_name}")
    
    def unregister_model(self, model_name: str):
        """Unregister a predictive model"""
        if model_name in self.models:
            del self.models[model_name]
            logger.info(f"Unregistered predictive model: {model_name}")
    
    async def predict(self, model_name: str, input_data: Dict[str, Any], 
                     use_ensemble: bool = False) -> Optional[Prediction]:
        """Make prediction using specified model or ensemble"""
        start_time = datetime.utcnow()
        
        try:
            if use_ensemble and self.config['enable_ensemble_predictions']:
                prediction = await self._ensemble_predict(input_data)
            else:
                if model_name not in self.models:
                    logger.error(f"Model {model_name} not found")
                    return None
                
                model = self.models[model_name]
                prediction = await asyncio.wait_for(
                    model.predict(input_data),
                    timeout=self.config['prediction_timeout_seconds']
                )
            
            # Record prediction
            prediction.created_at = start_time
            self.prediction_history.append(prediction)
            
            # Update statistics
            prediction_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_prediction_stats(True, prediction_time)
            
            # Limit history size
            if len(self.prediction_history) > self.config['max_history_size']:
                self.prediction_history = self.prediction_history[-self.config['max_history_size']//2:]
            
            logger.debug(f"Prediction made by {model_name}: confidence={prediction.confidence:.3f}")
            return prediction
            
        except asyncio.TimeoutError:
            logger.error(f"Prediction timeout for model {model_name}")
            self._update_prediction_stats(False, 0)
            return None
        except Exception as e:
            logger.error(f"Prediction failed for model {model_name}: {e}")
            self._update_prediction_stats(False, 0)
            return None
    
    async def _ensemble_predict(self, input_data: Dict[str, Any]) -> Prediction:
        """Make ensemble prediction using multiple models"""
        predictions = []
        
        # Get predictions from all available models
        for model_name, model in self.models.items():
            try:
                pred = await model.predict(input_data)
                if pred and pred.confidence > 0.5:  # Only use confident predictions
                    predictions.append(pred)
            except Exception as e:
                logger.warning(f"Ensemble prediction failed for {model_name}: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from ensemble models")
        
        # Combine predictions (weighted average by confidence)
        total_weight = sum(pred.confidence for pred in predictions)
        
        if isinstance(predictions[0].predicted_value, (int, float)):
            # Numeric prediction
            weighted_value = sum(pred.predicted_value * pred.confidence for pred in predictions) / total_weight
        else:
            # Non-numeric prediction - use most confident
            weighted_value = max(predictions, key=lambda p: p.confidence).predicted_value
        
        # Calculate ensemble confidence
        ensemble_confidence = min(1.0, total_weight / len(predictions))
        
        return Prediction(
            model_name="ensemble",
            predicted_value=weighted_value,
            confidence=ensemble_confidence,
            prediction_type="ensemble",
            input_features=input_data,
            metadata={
                'ensemble_size': len(predictions),
                'individual_predictions': [
                    {'model': p.model_name, 'value': p.predicted_value, 'confidence': p.confidence}
                    for p in predictions
                ]
            }
        )
    
    async def add_feedback(self, prediction_id: str, actual_value: Any, 
                          feedback_metadata: Optional[Dict[str, Any]] = None):
        """Add feedback for a prediction"""
        feedback = {
            'prediction_id': prediction_id,
            'actual_value': actual_value,
            'feedback_metadata': feedback_metadata or {},
            'timestamp': datetime.utcnow()
        }
        
        await self.feedback_queue.put(feedback)
        logger.debug(f"Added feedback for prediction {prediction_id}")
    
    async def _process_feedback_loop(self):
        """Process feedback and update models"""
        while self.is_running:
            try:
                # Get feedback with timeout
                feedback = await asyncio.wait_for(
                    self.feedback_queue.get(), timeout=1.0
                )
                
                await self._process_feedback(feedback)
                
            except asyncio.TimeoutError:
                # Check if models need retraining
                if self.config['auto_retrain_models']:
                    await self._check_and_retrain_models()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Feedback processing error: {e}")
    
    async def _process_feedback(self, feedback: Dict[str, Any]):
        """Process individual feedback"""
        prediction_id = feedback['prediction_id']
        actual_value = feedback['actual_value']
        
        # Find the prediction
        prediction = None
        for pred in self.prediction_history:
            if pred.prediction_id == prediction_id:
                prediction = pred
                break
        
        if not prediction:
            logger.warning(f"Prediction {prediction_id} not found for feedback")
            return
        
        # Update model metrics
        model = self.models.get(prediction.model_name)
        if model:
            model.update_metrics(actual_value, prediction.predicted_value, prediction.confidence)
            logger.debug(f"Updated metrics for model {prediction.model_name}")
    
    async def _check_and_retrain_models(self):
        """Check if models need retraining and retrain if necessary"""
        for model_name, model in self.models.items():
            if model.needs_retraining():
                logger.info(f"Model {model_name} needs retraining")
                # In a real implementation, would trigger retraining with new data
                # For now, just log the need
    
    def _update_prediction_stats(self, success: bool, prediction_time_ms: float):
        """Update prediction statistics"""
        self.stats['total_predictions'] += 1
        
        if success:
            self.stats['successful_predictions'] += 1
        else:
            self.stats['failed_predictions'] += 1
        
        # Update average prediction time
        if prediction_time_ms > 0:
            total_time = (self.stats['avg_prediction_time_ms'] * 
                         (self.stats['total_predictions'] - 1) + prediction_time_ms)
            self.stats['avg_prediction_time_ms'] = total_time / self.stats['total_predictions']
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get engine status and statistics"""
        model_status = {}
        for name, model in self.models.items():
            model_status[name] = {
                'model_type': model.model_type,
                'is_trained': model.is_trained,
                'training_data_size': model.training_data_size,
                'accuracy': model.metrics.accuracy,
                'total_predictions': model.metrics.total_predictions,
                'avg_confidence': model.metrics.avg_confidence,
                'needs_retraining': model.needs_retraining()
            }
        
        return {
            'is_running': self.is_running,
            'total_models': len(self.models),
            'prediction_history_size': len(self.prediction_history),
            'feedback_queue_size': self.feedback_queue.qsize(),
            'stats': self.stats.copy(),
            'models': model_status
        }


class OptimizationEngine:
    """
    Engine for optimizing recipe parameters and configurations.
    """
    
    def __init__(self, predictive_engine: PredictiveEngine):
        self.predictive_engine = predictive_engine
        self.optimizers: Dict[str, 'BaseOptimizer'] = {}
        self.optimization_history: List[Optimization] = []
        
        # Configuration
        self.config = {
            'max_optimization_time_seconds': 300,  # 5 minutes
            'max_iterations': 1000,
            'convergence_threshold': 0.001,
            'parallel_evaluations': True,
            'max_parallel_workers': 4
        }
        
        # Statistics
        self.stats = {
            'total_optimizations': 0,
            'successful_optimizations': 0,
            'avg_optimization_time_seconds': 0.0,
            'avg_improvement_percentage': 0.0
        }
    
    def register_optimizer(self, optimizer: 'BaseOptimizer'):
        """Register an optimizer"""
        self.optimizers[optimizer.optimizer_name] = optimizer
        logger.info(f"Registered optimizer: {optimizer.optimizer_name}")
    
    async def optimize(self, objective_function: Callable, parameter_space: Dict[str, Any],
                      optimizer_name: str = "genetic", constraints: Optional[Dict[str, Any]] = None) -> Optimization:
        """Optimize parameters using specified optimizer"""
        start_time = datetime.utcnow()
        
        try:
            if optimizer_name not in self.optimizers:
                raise ValueError(f"Optimizer {optimizer_name} not found")
            
            optimizer = self.optimizers[optimizer_name]
            
            # Run optimization
            result = await asyncio.wait_for(
                optimizer.optimize(objective_function, parameter_space, constraints),
                timeout=self.config['max_optimization_time_seconds']
            )
            
            # Create optimization record
            optimization_time = (datetime.utcnow() - start_time).total_seconds()
            
            optimization = Optimization(
                optimizer_name=optimizer_name,
                parameter_space=parameter_space,
                optimal_parameters=result.get('optimal_parameters', {}),
                optimal_value=result.get('optimal_value', 0.0),
                optimization_time_seconds=optimization_time,
                iterations=result.get('iterations', 0),
                convergence_achieved=result.get('converged', False),
                metadata=result.get('metadata', {})
            )
            
            # Record optimization
            self.optimization_history.append(optimization)
            
            # Update statistics
            self._update_optimization_stats(True, optimization_time, result.get('improvement', 0.0))
            
            logger.info(f"Optimization completed: {optimizer_name}, value={optimization.optimal_value:.4f}")
            return optimization
            
        except asyncio.TimeoutError:
            logger.error(f"Optimization timeout for {optimizer_name}")
            self._update_optimization_stats(False, 0, 0)
            raise
        except Exception as e:
            logger.error(f"Optimization failed for {optimizer_name}: {e}")
            self._update_optimization_stats(False, 0, 0)
            raise
    
    def _update_optimization_stats(self, success: bool, optimization_time: float, improvement: float):
        """Update optimization statistics"""
        self.stats['total_optimizations'] += 1
        
        if success:
            self.stats['successful_optimizations'] += 1
            
            # Update average optimization time
            total_time = (self.stats['avg_optimization_time_seconds'] * 
                         (self.stats['successful_optimizations'] - 1) + optimization_time)
            self.stats['avg_optimization_time_seconds'] = total_time / self.stats['successful_optimizations']
            
            # Update average improvement
            total_improvement = (self.stats['avg_improvement_percentage'] * 
                               (self.stats['successful_optimizations'] - 1) + improvement)
            self.stats['avg_improvement_percentage'] = total_improvement / self.stats['successful_optimizations']
    
    def get_optimization_history(self, limit: int = 100) -> List[Optimization]:
        """Get recent optimization history"""
        return self.optimization_history[-limit:]
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get optimization engine status"""
        return {
            'total_optimizers': len(self.optimizers),
            'optimizers': list(self.optimizers.keys()),
            'optimization_history_size': len(self.optimization_history),
            'stats': self.stats.copy()
        }


class RecommendationEngine:
    """
    Engine for generating intelligent recommendations based on predictions and optimizations.
    """
    
    def __init__(self, predictive_engine: PredictiveEngine, optimization_engine: OptimizationEngine):
        self.predictive_engine = predictive_engine
        self.optimization_engine = optimization_engine
        self.recommendation_rules: List[Callable] = []
        self.recommendation_history: List[Recommendation] = []
        
        # Configuration
        self.config = {
            'min_confidence_threshold': 0.7,
            'max_recommendations_per_request': 10,
            'enable_explanation_generation': True,
            'recommendation_expiry_hours': 24
        }
        
        # Statistics
        self.stats = {
            'total_recommendations': 0,
            'accepted_recommendations': 0,
            'rejected_recommendations': 0,
            'avg_confidence': 0.0
        }
    
    def add_recommendation_rule(self, rule: Callable):
        """Add a recommendation rule"""
        self.recommendation_rules.append(rule)
        logger.info("Added recommendation rule")
    
    async def generate_recommendations(self, context: Dict[str, Any], 
                                     recommendation_types: Optional[List[str]] = None) -> List[Recommendation]:
        """Generate recommendations based on context"""
        recommendations = []
        
        try:
            # Apply recommendation rules
            for rule in self.recommendation_rules:
                try:
                    rule_recommendations = await self._apply_rule(rule, context)
                    recommendations.extend(rule_recommendations)
                except Exception as e:
                    logger.warning(f"Recommendation rule failed: {e}")
            
            # Filter by type if specified
            if recommendation_types:
                recommendations = [r for r in recommendations if r.recommendation_type in recommendation_types]
            
            # Filter by confidence threshold
            recommendations = [r for r in recommendations if r.confidence >= self.config['min_confidence_threshold']]
            
            # Sort by confidence and limit
            recommendations.sort(key=lambda r: r.confidence, reverse=True)
            recommendations = recommendations[:self.config['max_recommendations_per_request']]
            
            # Generate explanations if enabled
            if self.config['enable_explanation_generation']:
                for rec in recommendations:
                    rec.explanation = await self._generate_explanation(rec, context)
            
            # Record recommendations
            self.recommendation_history.extend(recommendations)
            
            # Update statistics
            self._update_recommendation_stats(recommendations)
            
            logger.info(f"Generated {len(recommendations)} recommendations")
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation failed: {e}")
            return []
    
    async def _apply_rule(self, rule: Callable, context: Dict[str, Any]) -> List[Recommendation]:
        """Apply a recommendation rule"""
        # Rules should return list of recommendations
        if asyncio.iscoroutinefunction(rule):
            return await rule(context)
        else:
            return rule(context)
    
    async def _generate_explanation(self, recommendation: Recommendation, context: Dict[str, Any]) -> str:
        """Generate explanation for recommendation"""
        # Simple explanation generation - can be enhanced with NLP
        explanation_parts = []
        
        if recommendation.confidence > 0.9:
            explanation_parts.append("High confidence recommendation based on")
        elif recommendation.confidence > 0.7:
            explanation_parts.append("Moderate confidence recommendation based on")
        else:
            explanation_parts.append("Low confidence recommendation based on")
        
        if recommendation.predicted_impact:
            impact = recommendation.predicted_impact
            if impact > 0.2:
                explanation_parts.append(f"significant predicted improvement ({impact:.1%})")
            elif impact > 0.1:
                explanation_parts.append(f"moderate predicted improvement ({impact:.1%})")
            else:
                explanation_parts.append(f"minor predicted improvement ({impact:.1%})")
        
        explanation_parts.append("from historical data and predictive models.")
        
        return " ".join(explanation_parts)
    
    def _update_recommendation_stats(self, recommendations: List[Recommendation]):
        """Update recommendation statistics"""
        if not recommendations:
            return
        
        self.stats['total_recommendations'] += len(recommendations)
        
        # Update average confidence
        total_confidence = sum(rec.confidence for rec in recommendations)
        avg_new_confidence = total_confidence / len(recommendations)
        
        total_recs = self.stats['total_recommendations']
        prev_total_confidence = self.stats['avg_confidence'] * (total_recs - len(recommendations))
        self.stats['avg_confidence'] = (prev_total_confidence + total_confidence) / total_recs
    
    async def record_recommendation_feedback(self, recommendation_id: str, accepted: bool, 
                                           feedback_notes: Optional[str] = None):
        """Record feedback on recommendation"""
        # Find recommendation
        recommendation = None
        for rec in self.recommendation_history:
            if rec.recommendation_id == recommendation_id:
                recommendation = rec
                break
        
        if recommendation:
            recommendation.feedback_received = True
            recommendation.feedback_positive = accepted
            recommendation.feedback_notes = feedback_notes
            
            # Update statistics
            if accepted:
                self.stats['accepted_recommendations'] += 1
            else:
                self.stats['rejected_recommendations'] += 1
            
            logger.debug(f"Recorded feedback for recommendation {recommendation_id}: {'accepted' if accepted else 'rejected'}")
    
    def get_engine_status(self) -> Dict[str, Any]:
        """Get recommendation engine status"""
        return {
            'total_rules': len(self.recommendation_rules),
            'recommendation_history_size': len(self.recommendation_history),
            'stats': self.stats.copy(),
            'acceptance_rate': (self.stats['accepted_recommendations'] / 
                              max(self.stats['accepted_recommendations'] + self.stats['rejected_recommendations'], 1))
        }
