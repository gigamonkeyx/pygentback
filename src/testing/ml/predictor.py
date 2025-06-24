"""
Recipe Success Predictor

This module provides ML-powered prediction of recipe success using historical
test data, recipe features, and real-world execution patterns.
"""

import asyncio
import logging
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
import joblib
import json

from sklearn.ensemble import RandomForestClassifier, GradientBoostingRegressor
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from ..recipes.schema import RecipeDefinition, RecipeCategory, RecipeDifficulty
from ..core.framework import RecipeTestResult


logger = logging.getLogger(__name__)


@dataclass
class PredictionResult:
    """Result of recipe success prediction"""
    success_probability: float
    confidence: float
    execution_time_prediction: float
    memory_usage_prediction: float
    risk_factors: List[str]
    recommendations: List[str]
    feature_importance: Dict[str, float]


@dataclass
class ModelMetrics:
    """Model performance metrics"""
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    cross_val_score: float
    training_samples: int
    last_updated: datetime


class RecipeFeatureExtractor:
    """
    Extracts features from recipes for ML prediction.
    
    Converts recipe definitions into numerical features that can be
    used by machine learning models.
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_names = []
        self._initialize_feature_extractors()
    
    def _initialize_feature_extractors(self):
        """Initialize feature extraction components"""
        # Category and difficulty encoders
        self.label_encoders['category'] = LabelEncoder()
        self.label_encoders['difficulty'] = LabelEncoder()
        self.label_encoders['agent_types'] = LabelEncoder()
        self.label_encoders['server_types'] = LabelEncoder()
        
        # Fit with known values
        categories = [cat.value for cat in RecipeCategory]
        difficulties = [diff.value for diff in RecipeDifficulty]
        
        self.label_encoders['category'].fit(categories)
        self.label_encoders['difficulty'].fit(difficulties)
    
    def extract_features(self, recipe: RecipeDefinition) -> Dict[str, float]:
        """
        Extract numerical features from a recipe.
        
        Args:
            recipe: Recipe to extract features from
            
        Returns:
            Dict mapping feature names to values
        """
        features = {}
        
        # Basic recipe properties
        features['category_encoded'] = self._safe_encode('category', recipe.category.value)
        features['difficulty_encoded'] = self._safe_encode('difficulty', recipe.difficulty.value)
        features['num_steps'] = len(recipe.steps)
        features['num_agent_requirements'] = len(recipe.agent_requirements)
        features['num_mcp_requirements'] = len(recipe.mcp_requirements)
        features['num_tags'] = len(recipe.tags)
        
        # Complexity metrics
        features['total_timeout'] = sum(step.timeout_seconds for step in recipe.steps)
        features['critical_steps'] = sum(1 for step in recipe.steps if step.critical)
        features['retry_enabled_steps'] = sum(1 for step in recipe.steps if step.retry_on_failure)
        features['parallel_step_groups'] = len(recipe.parallel_steps)
        
        # Dependency complexity
        total_dependencies = sum(len(step.dependencies) for step in recipe.steps)
        features['avg_dependencies_per_step'] = total_dependencies / len(recipe.steps) if recipe.steps else 0
        features['max_dependencies'] = max((len(step.dependencies) for step in recipe.steps), default=0)
        
        # Resource requirements
        total_memory = sum(req.memory_limit_mb for req in recipe.agent_requirements)
        total_execution_time = sum(req.max_execution_time for req in recipe.agent_requirements)
        features['total_memory_limit'] = total_memory
        features['total_execution_time_limit'] = total_execution_time
        features['avg_memory_per_agent'] = total_memory / len(recipe.agent_requirements) if recipe.agent_requirements else 0
        
        # MCP tool complexity
        total_tool_timeout = sum(req.timeout_seconds for req in recipe.mcp_requirements)
        total_retry_count = sum(req.retry_count for req in recipe.mcp_requirements)
        features['total_mcp_timeout'] = total_tool_timeout
        features['total_mcp_retries'] = total_retry_count
        features['unique_servers'] = len(set(req.server_name for req in recipe.mcp_requirements))
        features['unique_tools'] = len(set(req.tool_name for req in recipe.mcp_requirements))
        
        # Validation criteria
        features['success_threshold'] = recipe.validation_criteria.success_threshold
        features['performance_budget'] = recipe.validation_criteria.performance_budget_ms
        features['memory_budget'] = recipe.validation_criteria.memory_budget_mb
        features['num_required_outputs'] = len(recipe.validation_criteria.required_outputs)
        features['num_quality_metrics'] = len(recipe.validation_criteria.quality_metrics)
        
        # Historical performance (if available)
        features['historical_success_rate'] = recipe.success_rate
        features['historical_avg_time'] = recipe.average_execution_time_ms
        features['usage_count'] = recipe.usage_count
        
        # Time-based features
        if recipe.last_tested:
            days_since_test = (datetime.utcnow() - recipe.last_tested).days
            features['days_since_last_test'] = days_since_test
        else:
            features['days_since_last_test'] = 999  # Never tested
        
        recipe_age_days = (datetime.utcnow() - recipe.created_at).days
        features['recipe_age_days'] = recipe_age_days
        
        # Version complexity (simple heuristic)
        version_parts = recipe.version.split('.')
        features['version_major'] = int(version_parts[0]) if version_parts else 1
        features['version_minor'] = int(version_parts[1]) if len(version_parts) > 1 else 0
        
        return features
    
    def _safe_encode(self, encoder_name: str, value: str) -> float:
        """Safely encode a categorical value"""
        try:
            encoder = self.label_encoders[encoder_name]
            if hasattr(encoder, 'classes_') and value in encoder.classes_:
                return float(encoder.transform([value])[0])
            else:
                # Handle unknown categories
                return -1.0
        except Exception:
            return -1.0
    
    def get_feature_names(self) -> List[str]:
        """Get list of feature names"""
        if not self.feature_names:
            # Extract from a dummy recipe to get feature names
            dummy_recipe = RecipeDefinition(name="dummy")
            features = self.extract_features(dummy_recipe)
            self.feature_names = list(features.keys())
        
        return self.feature_names


class RecipeSuccessPredictor:
    """
    ML-powered predictor for recipe success.
    
    Uses historical test data to predict the likelihood of recipe success,
    execution time, memory usage, and potential risk factors.
    """
    
    def __init__(self, model_dir: str = "./data/ml_models"):
        self.model_dir = Path(model_dir)
        self.model_dir.mkdir(parents=True, exist_ok=True)
        
        # Feature extraction
        self.feature_extractor = RecipeFeatureExtractor()
        self.scaler = StandardScaler()
        
        # Models
        self.success_classifier = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            class_weight='balanced'
        )
        
        self.execution_time_regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        self.memory_usage_regressor = GradientBoostingRegressor(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        # Model state
        self.models_trained = False
        self.last_training_time: Optional[datetime] = None
        self.model_metrics: Optional[ModelMetrics] = None
        self.feature_importance: Dict[str, float] = {}
        
        # Training data cache
        self.training_data: List[Tuple[RecipeDefinition, RecipeTestResult]] = []
        self.min_training_samples = 50
    
    async def initialize(self) -> None:
        """Initialize the predictor"""
        # Try to load existing models
        await self._load_models()
        
        logger.info(f"Recipe Success Predictor initialized (trained: {self.models_trained})")
    
    async def predict_success(self, recipe: RecipeDefinition) -> PredictionResult:
        """
        Predict success probability and metrics for a recipe.
        
        Args:
            recipe: Recipe to predict success for
            
        Returns:
            Prediction result with probability and recommendations
        """
        if not self.models_trained:
            # Return default prediction if models not trained
            return PredictionResult(
                success_probability=0.5,
                confidence=0.0,
                execution_time_prediction=5000.0,
                memory_usage_prediction=512.0,
                risk_factors=["Models not trained - insufficient historical data"],
                recommendations=["Collect more test data to improve predictions"],
                feature_importance={}
            )
        
        # Extract features
        features = self.feature_extractor.extract_features(recipe)
        feature_vector = np.array([list(features.values())]).reshape(1, -1)
        
        # Scale features
        feature_vector_scaled = self.scaler.transform(feature_vector)
        
        # Make predictions
        success_prob = self.success_classifier.predict_proba(feature_vector_scaled)[0][1]
        execution_time_pred = self.execution_time_regressor.predict(feature_vector_scaled)[0]
        memory_usage_pred = self.memory_usage_regressor.predict(feature_vector_scaled)[0]
        
        # Calculate confidence based on model certainty
        confidence = self._calculate_confidence(success_prob, feature_vector_scaled)
        
        # Identify risk factors
        risk_factors = self._identify_risk_factors(recipe, features)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(recipe, features, success_prob)
        
        return PredictionResult(
            success_probability=success_prob,
            confidence=confidence,
            execution_time_prediction=max(execution_time_pred, 100.0),  # Minimum 100ms
            memory_usage_prediction=max(memory_usage_pred, 64.0),       # Minimum 64MB
            risk_factors=risk_factors,
            recommendations=recommendations,
            feature_importance=self.feature_importance
        )
    
    async def add_training_data(self, recipe: RecipeDefinition, result: RecipeTestResult) -> None:
        """
        Add new training data from test results.
        
        Args:
            recipe: Recipe that was tested
            result: Test result
        """
        self.training_data.append((recipe, result))
        
        # Retrain if we have enough new data
        if len(self.training_data) >= self.min_training_samples:
            await self.retrain_models()
    
    async def retrain_models(self, force: bool = False) -> bool:
        """
        Retrain models with latest data.
        
        Args:
            force: Force retraining even with insufficient data
            
        Returns:
            bool: True if retraining was successful
        """
        if len(self.training_data) < self.min_training_samples and not force:
            logger.warning(f"Insufficient training data: {len(self.training_data)} < {self.min_training_samples}")
            return False
        
        logger.info(f"Retraining models with {len(self.training_data)} samples")
        
        try:
            # Prepare training data
            X, y_success, y_time, y_memory = self._prepare_training_data()
            
            if len(X) == 0:
                logger.warning("No valid training data available")
                return False
            
            # Split data
            X_train, X_test, y_success_train, y_success_test = train_test_split(
                X, y_success, test_size=0.2, random_state=42, stratify=y_success
            )
            
            _, _, y_time_train, y_time_test = train_test_split(
                X, y_time, test_size=0.2, random_state=42
            )
            
            _, _, y_memory_train, y_memory_test = train_test_split(
                X, y_memory, test_size=0.2, random_state=42
            )
            
            # Scale features
            X_train_scaled = self.scaler.fit_transform(X_train)
            X_test_scaled = self.scaler.transform(X_test)
            
            # Train models
            self.success_classifier.fit(X_train_scaled, y_success_train)
            self.execution_time_regressor.fit(X_train_scaled, y_time_train)
            self.memory_usage_regressor.fit(X_train_scaled, y_memory_train)
            
            # Evaluate models
            y_success_pred = self.success_classifier.predict(X_test_scaled)
            
            accuracy = accuracy_score(y_success_test, y_success_pred)
            precision = precision_score(y_success_test, y_success_pred, average='weighted')
            recall = recall_score(y_success_test, y_success_pred, average='weighted')
            f1 = f1_score(y_success_test, y_success_pred, average='weighted')
            
            # Cross-validation score
            cv_scores = cross_val_score(self.success_classifier, X_train_scaled, y_success_train, cv=5)
            cv_score = cv_scores.mean()
            
            # Update model metrics
            self.model_metrics = ModelMetrics(
                accuracy=accuracy,
                precision=precision,
                recall=recall,
                f1_score=f1,
                cross_val_score=cv_score,
                training_samples=len(X_train),
                last_updated=datetime.utcnow()
            )
            
            # Update feature importance
            feature_names = self.feature_extractor.get_feature_names()
            importance_scores = self.success_classifier.feature_importances_
            self.feature_importance = dict(zip(feature_names, importance_scores))
            
            self.models_trained = True
            self.last_training_time = datetime.utcnow()
            
            # Save models
            await self._save_models()
            
            logger.info(f"Model retraining completed - Accuracy: {accuracy:.3f}, F1: {f1:.3f}")
            return True
        
        except Exception as e:
            logger.error(f"Model retraining failed: {e}")
            return False
    
    def _prepare_training_data(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepare training data from collected samples"""
        X = []
        y_success = []
        y_time = []
        y_memory = []
        
        for recipe, result in self.training_data:
            try:
                # Extract features
                features = self.feature_extractor.extract_features(recipe)
                feature_vector = list(features.values())
                
                # Ensure all features are numeric
                if all(isinstance(f, (int, float)) and not np.isnan(f) for f in feature_vector):
                    X.append(feature_vector)
                    y_success.append(1 if result.success else 0)
                    y_time.append(result.execution_time_ms)
                    y_memory.append(result.memory_usage_mb)
            
            except Exception as e:
                logger.warning(f"Failed to process training sample: {e}")
                continue
        
        return (np.array(X), np.array(y_success), np.array(y_time), np.array(y_memory))
    
    def _calculate_confidence(self, success_prob: float, feature_vector: np.ndarray) -> float:
        """Calculate prediction confidence"""
        # Confidence based on how close to decision boundary
        distance_from_boundary = abs(success_prob - 0.5)
        confidence = min(distance_from_boundary * 2, 1.0)
        
        # Adjust based on training data size
        if self.model_metrics:
            data_confidence = min(self.model_metrics.training_samples / 1000, 1.0)
            confidence *= data_confidence
        
        return confidence
    
    def _identify_risk_factors(self, recipe: RecipeDefinition, features: Dict[str, float]) -> List[str]:
        """Identify potential risk factors for recipe failure"""
        risk_factors = []
        
        # High complexity risks
        if features.get('num_steps', 0) > 10:
            risk_factors.append("High number of execution steps")
        
        if features.get('total_timeout', 0) > 300:
            risk_factors.append("Long total timeout duration")
        
        if features.get('max_dependencies', 0) > 5:
            risk_factors.append("Complex step dependencies")
        
        # Resource risks
        if features.get('total_memory_limit', 0) > 2048:
            risk_factors.append("High memory requirements")
        
        if features.get('performance_budget', 0) < 1000:
            risk_factors.append("Tight performance budget")
        
        # Historical risks
        if features.get('historical_success_rate', 1.0) < 0.7:
            risk_factors.append("Poor historical success rate")
        
        if features.get('days_since_last_test', 0) > 30:
            risk_factors.append("Recipe not tested recently")
        
        # MCP complexity risks
        if features.get('unique_servers', 0) > 5:
            risk_factors.append("Many different MCP servers required")
        
        return risk_factors
    
    def _generate_recommendations(self, 
                                recipe: RecipeDefinition, 
                                features: Dict[str, float], 
                                success_prob: float) -> List[str]:
        """Generate recommendations to improve recipe success"""
        recommendations = []
        
        if success_prob < 0.7:
            recommendations.append("Consider simplifying recipe complexity")
            
            if features.get('num_steps', 0) > 8:
                recommendations.append("Reduce number of execution steps")
            
            if features.get('total_timeout', 0) > 200:
                recommendations.append("Optimize timeout values")
            
            if features.get('max_dependencies', 0) > 3:
                recommendations.append("Simplify step dependencies")
        
        if features.get('performance_budget', 0) < 2000:
            recommendations.append("Increase performance budget for more reliable execution")
        
        if features.get('historical_success_rate', 1.0) < 0.8:
            recommendations.append("Review and fix common failure patterns")
        
        if not recommendations:
            recommendations.append("Recipe looks good - no specific improvements needed")
        
        return recommendations
    
    async def _save_models(self) -> None:
        """Save trained models to disk"""
        try:
            # Save sklearn models
            joblib.dump(self.success_classifier, self.model_dir / "success_classifier.pkl")
            joblib.dump(self.execution_time_regressor, self.model_dir / "execution_time_regressor.pkl")
            joblib.dump(self.memory_usage_regressor, self.model_dir / "memory_usage_regressor.pkl")
            joblib.dump(self.scaler, self.model_dir / "feature_scaler.pkl")
            
            # Save metadata
            metadata = {
                "models_trained": self.models_trained,
                "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
                "feature_importance": self.feature_importance,
                "training_samples": len(self.training_data)
            }
            
            if self.model_metrics:
                metadata["model_metrics"] = {
                    "accuracy": self.model_metrics.accuracy,
                    "precision": self.model_metrics.precision,
                    "recall": self.model_metrics.recall,
                    "f1_score": self.model_metrics.f1_score,
                    "cross_val_score": self.model_metrics.cross_val_score,
                    "training_samples": self.model_metrics.training_samples,
                    "last_updated": self.model_metrics.last_updated.isoformat()
                }
            
            with open(self.model_dir / "model_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.info("Models saved successfully")
        
        except Exception as e:
            logger.error(f"Failed to save models: {e}")
    
    async def _load_models(self) -> None:
        """Load trained models from disk"""
        try:
            # Check if model files exist
            model_files = [
                "success_classifier.pkl",
                "execution_time_regressor.pkl", 
                "memory_usage_regressor.pkl",
                "feature_scaler.pkl"
            ]
            
            if all((self.model_dir / f).exists() for f in model_files):
                # Load sklearn models
                self.success_classifier = joblib.load(self.model_dir / "success_classifier.pkl")
                self.execution_time_regressor = joblib.load(self.model_dir / "execution_time_regressor.pkl")
                self.memory_usage_regressor = joblib.load(self.model_dir / "memory_usage_regressor.pkl")
                self.scaler = joblib.load(self.model_dir / "feature_scaler.pkl")
                
                # Load metadata
                metadata_file = self.model_dir / "model_metadata.json"
                if metadata_file.exists():
                    with open(metadata_file, 'r') as f:
                        metadata = json.load(f)
                    
                    self.models_trained = metadata.get("models_trained", False)
                    self.feature_importance = metadata.get("feature_importance", {})
                    
                    if metadata.get("last_training_time"):
                        self.last_training_time = datetime.fromisoformat(metadata["last_training_time"])
                    
                    if "model_metrics" in metadata:
                        metrics_data = metadata["model_metrics"]
                        self.model_metrics = ModelMetrics(
                            accuracy=metrics_data["accuracy"],
                            precision=metrics_data["precision"],
                            recall=metrics_data["recall"],
                            f1_score=metrics_data["f1_score"],
                            cross_val_score=metrics_data["cross_val_score"],
                            training_samples=metrics_data["training_samples"],
                            last_updated=datetime.fromisoformat(metrics_data["last_updated"])
                        )
                
                logger.info("Models loaded successfully")
            else:
                logger.info("No existing models found - will train when data is available")
        
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
            self.models_trained = False
    
    def get_model_status(self) -> Dict[str, Any]:
        """Get current model status and metrics"""
        status = {
            "models_trained": self.models_trained,
            "training_samples": len(self.training_data),
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "feature_importance": self.feature_importance
        }
        
        if self.model_metrics:
            status["metrics"] = {
                "accuracy": self.model_metrics.accuracy,
                "precision": self.model_metrics.precision,
                "recall": self.model_metrics.recall,
                "f1_score": self.model_metrics.f1_score,
                "cross_val_score": self.model_metrics.cross_val_score,
                "training_samples": self.model_metrics.training_samples
            }
        
        return status
