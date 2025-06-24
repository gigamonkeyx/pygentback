"""
Specialized Predictors

Concrete implementations of predictive models for different domains.
"""

import logging
import asyncio
import numpy as np
import random
from typing import List, Dict, Any, Optional
from datetime import datetime, timedelta

from .core import BasePredictiveModel
from .models import Prediction, PredictionType, PredictionMetrics

logger = logging.getLogger(__name__)


class PerformancePredictor(BasePredictiveModel):
    """
    Predictor for recipe and system performance metrics.
    """
    
    def __init__(self):
        super().__init__("performance_predictor", "regression")
        
        # Model parameters (simplified linear model for demonstration)
        self.weights = {}
        self.bias = 0.0
        self.feature_names = [
            'recipe_complexity', 'resource_allocation', 'parallel_tasks',
            'data_size', 'network_latency', 'cpu_cores', 'memory_gb'
        ]
        
        # Performance-specific configuration
        self.config.update({
            'learning_rate': 0.01,
            'regularization': 0.001,
            'feature_scaling': True,
            'outlier_threshold': 3.0
        })
    
    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train the performance prediction model"""
        try:
            if len(training_data) < self.config['min_training_samples']:
                logger.warning(f"Insufficient training data: {len(training_data)} samples")
                return False
            
            # Extract features and targets
            X, y = self._prepare_training_data(training_data)
            
            if len(X) == 0:
                logger.error("No valid training samples after preprocessing")
                return False
            
            # Simple linear regression training (simplified)
            await self._train_linear_model(X, y)
            
            # Update model state
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.utcnow()
            
            logger.info(f"Performance predictor trained on {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Performance predictor training failed: {e}")
            return False
    
    async def predict(self, input_data: Dict[str, Any]) -> Prediction:
        """Predict performance metrics"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            # Extract features
            features = self._extract_features(input_data)
            
            # Make prediction
            predicted_value = await self._predict_performance(features)
            
            # Calculate confidence based on feature similarity to training data
            confidence = self._calculate_confidence(features)
            
            # Calculate uncertainty estimate
            uncertainty = self._calculate_uncertainty(features)
            
            prediction = Prediction(
                model_name=self.model_name,
                prediction_type=PredictionType.PERFORMANCE,
                predicted_value=predicted_value,
                confidence=confidence,
                uncertainty=uncertainty,
                input_features=features,
                prediction_context={'model_type': self.model_type},
                expires_at=datetime.utcnow() + timedelta(hours=self.config['max_prediction_age_hours'])
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            raise
    
    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate model performance on test data"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        predictions = []
        actuals = []
        
        for sample in test_data:
            try:
                features = self._extract_features(sample)
                predicted = await self._predict_performance(features)
                actual = sample.get('performance_score', 0.0)
                
                predictions.append(predicted)
                actuals.append(actual)
            except Exception as e:
                logger.warning(f"Evaluation sample failed: {e}")
        
        if not predictions:
            return {'error': 'No valid predictions for evaluation'}
        
        # Calculate metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)
        
        # Calculate R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0
        
        return {
            'mean_absolute_error': float(mae),
            'mean_squared_error': float(mse),
            'root_mean_squared_error': float(rmse),
            'r_squared': float(r2),
            'sample_count': len(predictions)
        }
    
    def _prepare_training_data(self, training_data: List[Dict[str, Any]]) -> tuple:
        """Prepare training data for model"""
        X = []
        y = []
        
        for sample in training_data:
            try:
                features = self._extract_features(sample)
                target = sample.get('performance_score', 0.0)
                
                if features and isinstance(target, (int, float)):
                    X.append(list(features.values()))
                    y.append(target)
            except Exception as e:
                logger.warning(f"Skipping invalid training sample: {e}")
        
        return np.array(X), np.array(y)
    
    def _extract_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features from input data"""
        features = {}
        
        # Extract known features with defaults
        features['recipe_complexity'] = float(data.get('recipe_complexity', 5.0))
        features['resource_allocation'] = float(data.get('resource_allocation', 1.0))
        features['parallel_tasks'] = float(data.get('parallel_tasks', 1.0))
        features['data_size'] = float(data.get('data_size_mb', 100.0))
        features['network_latency'] = float(data.get('network_latency_ms', 50.0))
        features['cpu_cores'] = float(data.get('cpu_cores', 4.0))
        features['memory_gb'] = float(data.get('memory_gb', 8.0))
        
        # Feature scaling if enabled
        if self.config['feature_scaling']:
            features = self._scale_features(features)
        
        return features
    
    def _scale_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale features to normalized range"""
        # Simple min-max scaling with predefined ranges
        scaling_ranges = {
            'recipe_complexity': (1.0, 20.0),
            'resource_allocation': (0.1, 10.0),
            'parallel_tasks': (1.0, 50.0),
            'data_size': (1.0, 10000.0),
            'network_latency': (1.0, 1000.0),
            'cpu_cores': (1.0, 64.0),
            'memory_gb': (1.0, 256.0)
        }
        
        scaled_features = {}
        for name, value in features.items():
            if name in scaling_ranges:
                min_val, max_val = scaling_ranges[name]
                scaled_value = (value - min_val) / (max_val - min_val)
                scaled_features[name] = max(0.0, min(1.0, scaled_value))
            else:
                scaled_features[name] = value
        
        return scaled_features
    
    async def _train_linear_model(self, X: np.ndarray, y: np.ndarray):
        """Train simple linear model"""
        # Initialize weights randomly
        n_features = X.shape[1]
        self.weights = {self.feature_names[i]: random.uniform(-1, 1) for i in range(min(n_features, len(self.feature_names)))}
        self.bias = random.uniform(-1, 1)
        
        # Simple gradient descent (simplified)
        learning_rate = self.config['learning_rate']
        
        for epoch in range(100):  # Limited epochs for demo
            # Forward pass
            predictions = []
            for sample in X:
                pred = self.bias
                for i, feature_name in enumerate(self.feature_names[:len(sample)]):
                    if feature_name in self.weights:
                        pred += self.weights[feature_name] * sample[i]
                predictions.append(pred)
            
            predictions = np.array(predictions)
            
            # Calculate loss
            loss = np.mean((predictions - y) ** 2)
            
            # Backward pass (simplified)
            errors = predictions - y
            
            # Update weights
            for i, feature_name in enumerate(self.feature_names[:X.shape[1]]):
                if feature_name in self.weights:
                    gradient = np.mean(errors * X[:, i])
                    self.weights[feature_name] -= learning_rate * gradient
            
            # Update bias
            self.bias -= learning_rate * np.mean(errors)
            
            # Early stopping if loss is small
            if loss < 0.001:
                break
            
            # Yield control occasionally
            if epoch % 10 == 0:
                await asyncio.sleep(0.001)
    
    async def _predict_performance(self, features: Dict[str, float]) -> float:
        """Make performance prediction"""
        prediction = self.bias
        
        for feature_name, value in features.items():
            if feature_name in self.weights:
                prediction += self.weights[feature_name] * value
        
        # Ensure prediction is in reasonable range (0-100 performance score)
        return max(0.0, min(100.0, prediction * 50 + 50))  # Scale to 0-100
    
    def _calculate_confidence(self, features: Dict[str, float]) -> float:
        """Calculate prediction confidence"""
        # Simple confidence based on feature completeness and values
        base_confidence = 0.7
        
        # Boost confidence if all expected features are present
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        feature_completeness = len(present_features.intersection(expected_features)) / len(expected_features)
        
        confidence = base_confidence * feature_completeness
        
        # Adjust based on feature values (penalize extreme values)
        for value in features.values():
            if value < 0 or value > 1:  # Assuming scaled features
                confidence *= 0.9
        
        return min(1.0, max(0.1, confidence))
    
    def _calculate_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate prediction uncertainty"""
        # Simple uncertainty estimation
        base_uncertainty = 0.1
        
        # Increase uncertainty for incomplete features
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        missing_ratio = 1 - (len(present_features.intersection(expected_features)) / len(expected_features))
        
        uncertainty = base_uncertainty + missing_ratio * 0.3
        
        return min(1.0, uncertainty)


class SuccessPredictor(BasePredictiveModel):
    """
    Predictor for recipe success probability.
    """
    
    def __init__(self):
        super().__init__("success_predictor", "classification")
        
        # Model parameters for logistic regression
        self.weights = {}
        self.bias = 0.0
        self.feature_names = [
            'recipe_complexity', 'test_coverage', 'dependency_count',
            'error_history', 'resource_availability', 'team_experience'
        ]
    
    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train the success prediction model"""
        try:
            if len(training_data) < self.config['min_training_samples']:
                return False
            
            # Extract features and binary targets
            X, y = self._prepare_classification_data(training_data)
            
            if len(X) == 0:
                return False
            
            # Train logistic regression model
            await self._train_logistic_model(X, y)
            
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.utcnow()
            
            logger.info(f"Success predictor trained on {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Success predictor training failed: {e}")
            return False
    
    async def predict(self, input_data: Dict[str, Any]) -> Prediction:
        """Predict success probability"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            features = self._extract_success_features(input_data)
            success_probability = await self._predict_success_probability(features)
            
            # Confidence based on feature quality
            confidence = self._calculate_classification_confidence(features)
            
            prediction = Prediction(
                model_name=self.model_name,
                prediction_type=PredictionType.SUCCESS_RATE,
                predicted_value=success_probability,
                confidence=confidence,
                input_features=features,
                prediction_context={'model_type': self.model_type},
                expires_at=datetime.utcnow() + timedelta(hours=self.config['max_prediction_age_hours'])
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Success prediction failed: {e}")
            raise
    
    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate classification model performance"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        predictions = []
        actuals = []
        
        for sample in test_data:
            try:
                features = self._extract_success_features(sample)
                predicted_prob = await self._predict_success_probability(features)
                predicted_class = 1 if predicted_prob > 0.5 else 0
                actual = int(sample.get('success', 0))
                
                predictions.append(predicted_class)
                actuals.append(actual)
            except Exception as e:
                logger.warning(f"Evaluation sample failed: {e}")
        
        if not predictions:
            return {'error': 'No valid predictions for evaluation'}
        
        # Calculate classification metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)
        
        accuracy = np.mean(predictions == actuals)
        
        # Calculate precision, recall, F1
        tp = np.sum((predictions == 1) & (actuals == 1))
        fp = np.sum((predictions == 1) & (actuals == 0))
        fn = np.sum((predictions == 0) & (actuals == 1))
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        
        return {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1),
            'sample_count': len(predictions)
        }
    
    def _prepare_classification_data(self, training_data: List[Dict[str, Any]]) -> tuple:
        """Prepare classification training data"""
        X = []
        y = []
        
        for sample in training_data:
            try:
                features = self._extract_success_features(sample)
                target = int(sample.get('success', 0))  # Binary target
                
                if features:
                    X.append(list(features.values()))
                    y.append(target)
            except Exception as e:
                logger.warning(f"Skipping invalid training sample: {e}")
        
        return np.array(X), np.array(y)
    
    def _extract_success_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for success prediction"""
        features = {}
        
        features['recipe_complexity'] = float(data.get('recipe_complexity', 5.0)) / 20.0  # Normalize
        features['test_coverage'] = float(data.get('test_coverage', 0.8))
        features['dependency_count'] = min(float(data.get('dependency_count', 5.0)) / 50.0, 1.0)
        features['error_history'] = 1.0 - min(float(data.get('error_count', 0.0)) / 10.0, 1.0)
        features['resource_availability'] = float(data.get('resource_availability', 0.8))
        features['team_experience'] = float(data.get('team_experience', 0.7))
        
        return features
    
    async def _train_logistic_model(self, X: np.ndarray, y: np.ndarray):
        """Train logistic regression model"""
        n_features = X.shape[1]
        self.weights = {self.feature_names[i]: random.uniform(-1, 1) for i in range(min(n_features, len(self.feature_names)))}
        self.bias = random.uniform(-1, 1)
        
        learning_rate = 0.01
        
        for epoch in range(100):
            # Forward pass with sigmoid
            predictions = []
            for sample in X:
                logit = self.bias
                for i, feature_name in enumerate(self.feature_names[:len(sample)]):
                    if feature_name in self.weights:
                        logit += self.weights[feature_name] * sample[i]
                prob = 1 / (1 + np.exp(-logit))  # Sigmoid
                predictions.append(prob)
            
            predictions = np.array(predictions)
            
            # Calculate loss (cross-entropy)
            epsilon = 1e-15  # Prevent log(0)
            predictions = np.clip(predictions, epsilon, 1 - epsilon)
            loss = -np.mean(y * np.log(predictions) + (1 - y) * np.log(1 - predictions))
            
            # Backward pass
            errors = predictions - y
            
            # Update weights
            for i, feature_name in enumerate(self.feature_names[:X.shape[1]]):
                if feature_name in self.weights:
                    gradient = np.mean(errors * X[:, i])
                    self.weights[feature_name] -= learning_rate * gradient
            
            self.bias -= learning_rate * np.mean(errors)
            
            if loss < 0.01:
                break
            
            if epoch % 10 == 0:
                await asyncio.sleep(0.001)
    
    async def _predict_success_probability(self, features: Dict[str, float]) -> float:
        """Predict success probability using logistic function"""
        logit = self.bias
        
        for feature_name, value in features.items():
            if feature_name in self.weights:
                logit += self.weights[feature_name] * value
        
        # Apply sigmoid function
        probability = 1 / (1 + np.exp(-logit))
        return float(probability)
    
    def _calculate_classification_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence for classification prediction"""
        base_confidence = 0.8
        
        # Check feature completeness
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        completeness = len(present_features.intersection(expected_features)) / len(expected_features)
        
        confidence = base_confidence * completeness
        
        # Adjust for feature quality
        for value in features.values():
            if 0.0 <= value <= 1.0:  # Good normalized value
                confidence *= 1.05
            else:
                confidence *= 0.9
        
        return min(1.0, max(0.1, confidence))


class ResourcePredictor(BasePredictiveModel):
    """
    Predictor for resource usage (CPU, memory, storage, network).
    """
    
    def __init__(self):
        super().__init__("resource_predictor", "regression")
        
        # Separate models for different resource types
        self.resource_models = {
            'cpu_usage': {'weights': {}, 'bias': 0.0},
            'memory_usage': {'weights': {}, 'bias': 0.0},
            'storage_usage': {'weights': {}, 'bias': 0.0},
            'network_usage': {'weights': {}, 'bias': 0.0}
        }
        
        self.feature_names = [
            'data_size', 'processing_complexity', 'parallel_operations',
            'io_operations', 'network_requests', 'cache_size'
        ]
    
    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train resource usage prediction models"""
        try:
            if len(training_data) < self.config['min_training_samples']:
                return False
            
            # Train separate models for each resource type
            for resource_type in self.resource_models.keys():
                X, y = self._prepare_resource_data(training_data, resource_type)
                if len(X) > 0:
                    await self._train_resource_model(resource_type, X, y)
            
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.utcnow()
            
            logger.info(f"Resource predictor trained on {len(training_data)} samples")
            return True
            
        except Exception as e:
            logger.error(f"Resource predictor training failed: {e}")
            return False
    
    async def predict(self, input_data: Dict[str, Any]) -> Prediction:
        """Predict resource usage"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        try:
            features = self._extract_resource_features(input_data)
            
            # Predict all resource types
            resource_predictions = {}
            for resource_type in self.resource_models.keys():
                predicted_usage = await self._predict_resource_usage(resource_type, features)
                resource_predictions[resource_type] = predicted_usage
            
            confidence = self._calculate_resource_confidence(features)
            
            prediction = Prediction(
                model_name=self.model_name,
                prediction_type=PredictionType.RESOURCE_USAGE,
                predicted_value=resource_predictions,
                confidence=confidence,
                input_features=features,
                prediction_context={'model_type': self.model_type},
                expires_at=datetime.utcnow() + timedelta(hours=self.config['max_prediction_age_hours'])
            )
            
            return prediction
            
        except Exception as e:
            logger.error(f"Resource prediction failed: {e}")
            raise
    
    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate resource prediction model"""
        if not self.is_trained:
            raise ValueError("Model not trained")
        
        results = {}
        
        for resource_type in self.resource_models.keys():
            predictions = []
            actuals = []
            
            for sample in test_data:
                try:
                    features = self._extract_resource_features(sample)
                    predicted = await self._predict_resource_usage(resource_type, features)
                    actual = sample.get(f'{resource_type}_actual', 0.0)
                    
                    predictions.append(predicted)
                    actuals.append(actual)
                except Exception as e:
                    logger.warning(f"Evaluation sample failed for {resource_type}: {e}")
            
            if predictions:
                predictions = np.array(predictions)
                actuals = np.array(actuals)
                
                mae = np.mean(np.abs(predictions - actuals))
                mse = np.mean((predictions - actuals) ** 2)
                
                results[f'{resource_type}_mae'] = float(mae)
                results[f'{resource_type}_mse'] = float(mse)
        
        return results
    
    def _prepare_resource_data(self, training_data: List[Dict[str, Any]], resource_type: str) -> tuple:
        """Prepare training data for specific resource type"""
        X = []
        y = []
        
        for sample in training_data:
            try:
                features = self._extract_resource_features(sample)
                target = sample.get(f'{resource_type}_actual', 0.0)
                
                if features and isinstance(target, (int, float)):
                    X.append(list(features.values()))
                    y.append(target)
            except Exception as e:
                logger.warning(f"Skipping invalid training sample for {resource_type}: {e}")
        
        return np.array(X), np.array(y)
    
    def _extract_resource_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for resource prediction"""
        features = {}
        
        features['data_size'] = float(data.get('data_size_mb', 100.0)) / 10000.0  # Normalize
        features['processing_complexity'] = float(data.get('processing_complexity', 5.0)) / 10.0
        features['parallel_operations'] = float(data.get('parallel_operations', 1.0)) / 20.0
        features['io_operations'] = float(data.get('io_operations', 10.0)) / 1000.0
        features['network_requests'] = float(data.get('network_requests', 5.0)) / 100.0
        features['cache_size'] = float(data.get('cache_size_mb', 256.0)) / 1024.0
        
        return features
    
    async def _train_resource_model(self, resource_type: str, X: np.ndarray, y: np.ndarray):
        """Train model for specific resource type"""
        model = self.resource_models[resource_type]
        
        # Initialize weights
        n_features = X.shape[1]
        model['weights'] = {self.feature_names[i]: random.uniform(-1, 1) for i in range(min(n_features, len(self.feature_names)))}
        model['bias'] = random.uniform(-1, 1)
        
        # Simple training loop
        learning_rate = 0.01
        
        for epoch in range(50):
            predictions = []
            for sample in X:
                pred = model['bias']
                for i, feature_name in enumerate(self.feature_names[:len(sample)]):
                    if feature_name in model['weights']:
                        pred += model['weights'][feature_name] * sample[i]
                predictions.append(max(0, pred))  # Ensure non-negative resource usage
            
            predictions = np.array(predictions)
            errors = predictions - y
            
            # Update weights
            for i, feature_name in enumerate(self.feature_names[:X.shape[1]]):
                if feature_name in model['weights']:
                    gradient = np.mean(errors * X[:, i])
                    model['weights'][feature_name] -= learning_rate * gradient
            
            model['bias'] -= learning_rate * np.mean(errors)
            
            if epoch % 10 == 0:
                await asyncio.sleep(0.001)
    
    async def _predict_resource_usage(self, resource_type: str, features: Dict[str, float]) -> float:
        """Predict usage for specific resource type"""
        model = self.resource_models[resource_type]
        
        prediction = model['bias']
        for feature_name, value in features.items():
            if feature_name in model['weights']:
                prediction += model['weights'][feature_name] * value
        
        # Scale prediction to reasonable range and ensure non-negative
        if resource_type == 'cpu_usage':
            return max(0.0, min(100.0, prediction * 50 + 25))  # 0-100% CPU
        elif resource_type == 'memory_usage':
            return max(0.0, prediction * 1000 + 100)  # MB
        elif resource_type == 'storage_usage':
            return max(0.0, prediction * 5000 + 500)  # MB
        elif resource_type == 'network_usage':
            return max(0.0, prediction * 100 + 10)  # Mbps
        else:
            return max(0.0, prediction)
    
    def _calculate_resource_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence for resource prediction"""
        base_confidence = 0.75
        
        # Feature completeness
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        completeness = len(present_features.intersection(expected_features)) / len(expected_features)
        
        confidence = base_confidence * completeness
        
        # Adjust for reasonable feature values
        for value in features.values():
            if 0.0 <= value <= 1.0:
                confidence *= 1.02
            else:
                confidence *= 0.95
        
        return min(1.0, max(0.2, confidence))


class LatencyPredictor(BasePredictiveModel):
    """
    Predictor for system and operation latency.
    """

    def __init__(self):
        super().__init__("latency_predictor", "regression")

        # Model parameters for latency prediction
        self.weights = {}
        self.bias = 0.0
        self.feature_names = [
            'operation_complexity', 'data_size_mb', 'network_distance',
            'cpu_load', 'memory_usage', 'concurrent_operations', 'cache_hit_ratio'
        ]

        # Latency-specific configuration
        self.config.update({
            'learning_rate': 0.005,
            'regularization': 0.0001,
            'feature_scaling': True,
            'outlier_threshold': 2.5,
            'max_latency_ms': 10000.0  # Maximum expected latency
        })

    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train the latency prediction model"""
        try:
            if len(training_data) < self.config['min_training_samples']:
                logger.warning(f"Insufficient training data for latency predictor: {len(training_data)} samples")
                return False

            # Extract features and latency targets
            X, y = self._prepare_latency_training_data(training_data)

            if len(X) == 0:
                logger.error("No valid training samples for latency predictor")
                return False

            # Train latency prediction model
            await self._train_latency_model(X, y)

            # Update model state
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.utcnow()

            logger.info(f"Latency predictor trained on {len(training_data)} samples")
            return True

        except Exception as e:
            logger.error(f"Latency predictor training failed: {e}")
            return False

    async def predict(self, input_data: Dict[str, Any]) -> Prediction:
        """Predict operation latency"""
        if not self.is_trained:
            raise ValueError("Latency model not trained")

        try:
            # Extract latency-specific features
            features = self._extract_latency_features(input_data)

            # Make latency prediction
            predicted_latency = await self._predict_latency(features)

            # Calculate confidence based on feature quality and historical accuracy
            confidence = self._calculate_latency_confidence(features)

            # Calculate uncertainty estimate
            uncertainty = self._calculate_latency_uncertainty(features)

            prediction = Prediction(
                model_name=self.model_name,
                prediction_type=PredictionType.LATENCY,
                predicted_value=predicted_latency,
                confidence=confidence,
                uncertainty=uncertainty,
                input_features=features,
                prediction_context={
                    'model_type': self.model_type,
                    'latency_unit': 'milliseconds',
                    'prediction_range': f"0-{self.config['max_latency_ms']}ms"
                },
                expires_at=datetime.utcnow() + timedelta(minutes=30)  # Shorter expiry for latency
            )

            return prediction

        except Exception as e:
            logger.error(f"Latency prediction failed: {e}")
            raise

    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate latency model performance on test data"""
        if not self.is_trained:
            raise ValueError("Latency model not trained")

        predictions = []
        actuals = []

        for sample in test_data:
            try:
                features = self._extract_latency_features(sample)
                predicted = await self._predict_latency(features)
                actual = sample.get('actual_latency_ms', 0.0)

                predictions.append(predicted)
                actuals.append(actual)
            except Exception as e:
                logger.warning(f"Latency evaluation sample failed: {e}")

        if not predictions:
            return {'error': 'No valid predictions for latency evaluation'}

        # Calculate latency-specific metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 1.0))) * 100

        # Calculate R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Latency-specific metrics
        within_10_percent = np.mean(np.abs(predictions - actuals) / np.maximum(actuals, 1.0) <= 0.1) * 100
        within_50ms = np.mean(np.abs(predictions - actuals) <= 50.0) * 100

        return {
            'mean_absolute_error_ms': float(mae),
            'mean_squared_error': float(mse),
            'root_mean_squared_error_ms': float(rmse),
            'mean_absolute_percentage_error': float(mape),
            'r_squared': float(r2),
            'within_10_percent_accuracy': float(within_10_percent),
            'within_50ms_accuracy': float(within_50ms),
            'sample_count': len(predictions)
        }

    def _prepare_latency_training_data(self, training_data: List[Dict[str, Any]]) -> tuple:
        """Prepare training data for latency model"""
        X = []
        y = []

        for sample in training_data:
            try:
                features = self._extract_latency_features(sample)
                target = sample.get('actual_latency_ms', sample.get('latency_ms', 0.0))

                if features and isinstance(target, (int, float)) and target >= 0:
                    # Filter outliers
                    if target <= self.config['max_latency_ms']:
                        X.append(list(features.values()))
                        y.append(target)
            except Exception as e:
                logger.warning(f"Skipping invalid latency training sample: {e}")

        return np.array(X), np.array(y)

    def _extract_latency_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for latency prediction"""
        features = {}

        # Extract latency-relevant features with sensible defaults
        features['operation_complexity'] = float(data.get('operation_complexity', 5.0))
        features['data_size_mb'] = float(data.get('data_size_mb', data.get('data_size', 10.0)))
        features['network_distance'] = float(data.get('network_distance_ms', data.get('network_latency', 20.0)))
        features['cpu_load'] = float(data.get('cpu_load', data.get('cpu_usage', 0.5)))
        features['memory_usage'] = float(data.get('memory_usage', data.get('memory_percent', 0.6)))
        features['concurrent_operations'] = float(data.get('concurrent_operations', data.get('parallel_tasks', 1.0)))
        features['cache_hit_ratio'] = float(data.get('cache_hit_ratio', 0.8))

        # Feature scaling if enabled
        if self.config['feature_scaling']:
            features = self._scale_latency_features(features)

        return features

    def _scale_latency_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale latency features to normalized range"""
        # Latency-specific scaling ranges
        scaling_ranges = {
            'operation_complexity': (1.0, 50.0),
            'data_size_mb': (0.1, 1000.0),
            'network_distance': (1.0, 500.0),
            'cpu_load': (0.0, 1.0),
            'memory_usage': (0.0, 1.0),
            'concurrent_operations': (1.0, 100.0),
            'cache_hit_ratio': (0.0, 1.0)
        }

        scaled_features = {}
        for name, value in features.items():
            if name in scaling_ranges:
                min_val, max_val = scaling_ranges[name]
                if name in ['cpu_load', 'memory_usage', 'cache_hit_ratio']:
                    # These are already in 0-1 range
                    scaled_features[name] = max(0.0, min(1.0, value))
                else:
                    scaled_value = (value - min_val) / (max_val - min_val)
                    scaled_features[name] = max(0.0, min(1.0, scaled_value))
            else:
                scaled_features[name] = value

        return scaled_features

    async def _train_latency_model(self, X: np.ndarray, y: np.ndarray):
        """Train latency prediction model"""
        # Initialize weights for latency prediction
        n_features = X.shape[1]
        self.weights = {self.feature_names[i]: random.uniform(-0.5, 0.5) for i in range(min(n_features, len(self.feature_names)))}
        self.bias = random.uniform(0, 100)  # Positive bias for latency

        # Gradient descent with latency-specific learning rate
        learning_rate = self.config['learning_rate']
        regularization = self.config['regularization']

        for epoch in range(200):  # More epochs for latency prediction
            # Forward pass
            predictions = []
            for sample in X:
                pred = self.bias
                for i, feature_name in enumerate(self.feature_names[:len(sample)]):
                    if feature_name in self.weights:
                        pred += self.weights[feature_name] * sample[i]
                # Ensure non-negative latency
                pred = max(0.0, pred)
                predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate loss with regularization
            mse_loss = np.mean((predictions - y) ** 2)
            reg_loss = regularization * sum(w**2 for w in self.weights.values())
            total_loss = mse_loss + reg_loss

            # Backward pass
            errors = predictions - y

            # Update weights with regularization
            for i, feature_name in enumerate(self.feature_names[:X.shape[1]]):
                if feature_name in self.weights:
                    gradient = np.mean(errors * X[:, i]) + regularization * self.weights[feature_name]
                    self.weights[feature_name] -= learning_rate * gradient

            # Update bias
            self.bias -= learning_rate * np.mean(errors)
            self.bias = max(0.0, self.bias)  # Keep bias non-negative

            # Early stopping
            if total_loss < 1.0:
                break

            # Yield control
            if epoch % 20 == 0:
                await asyncio.sleep(0.001)

    async def _predict_latency(self, features: Dict[str, float]) -> float:
        """Make latency prediction"""
        prediction = self.bias

        for feature_name, value in features.items():
            if feature_name in self.weights:
                prediction += self.weights[feature_name] * value

        # Ensure non-negative latency and within reasonable bounds
        prediction = max(0.0, min(self.config['max_latency_ms'], prediction))

        # Scale prediction to realistic latency range (0-1000ms typical)
        if self.config['feature_scaling']:
            prediction = prediction * 1000  # Scale up from normalized range

        return prediction

    def _calculate_latency_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence for latency prediction"""
        base_confidence = 0.75

        # Boost confidence for complete feature sets
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        feature_completeness = len(present_features.intersection(expected_features)) / len(expected_features)

        confidence = base_confidence * feature_completeness

        # Adjust based on feature values (penalize extreme values)
        for name, value in features.items():
            if name in ['cpu_load', 'memory_usage'] and value > 0.9:
                confidence *= 0.9  # High load reduces confidence
            elif name == 'cache_hit_ratio' and value < 0.3:
                confidence *= 0.9  # Low cache hit reduces confidence

        return min(1.0, max(0.2, confidence))

    def _calculate_latency_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate uncertainty for latency prediction"""
        base_uncertainty = 0.15

        # Increase uncertainty for missing features
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        missing_ratio = 1 - (len(present_features.intersection(expected_features)) / len(expected_features))

        uncertainty = base_uncertainty + missing_ratio * 0.25

        # Increase uncertainty for high load conditions
        cpu_load = features.get('cpu_load', 0.5)
        memory_usage = features.get('memory_usage', 0.5)
        if cpu_load > 0.8 or memory_usage > 0.8:
            uncertainty += 0.1

        return min(0.8, uncertainty)


class QualityPredictor(BasePredictiveModel):
    """
    Predictor for code quality, recipe quality, and output quality metrics.
    """

    def __init__(self):
        super().__init__("quality_predictor", "regression")

        # Model parameters for quality prediction
        self.weights = {}
        self.bias = 0.0
        self.feature_names = [
            'code_complexity', 'test_coverage', 'documentation_score',
            'maintainability_index', 'error_density', 'review_score', 'experience_level'
        ]

        # Quality-specific configuration
        self.config.update({
            'learning_rate': 0.01,
            'regularization': 0.0005,
            'feature_scaling': True,
            'quality_scale': 100.0,  # 0-100 quality score
            'min_quality_threshold': 0.0,
            'max_quality_threshold': 100.0
        })

    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train the quality prediction model"""
        try:
            if len(training_data) < self.config['min_training_samples']:
                logger.warning(f"Insufficient training data for quality predictor: {len(training_data)} samples")
                return False

            # Extract features and quality targets
            X, y = self._prepare_quality_training_data(training_data)

            if len(X) == 0:
                logger.error("No valid training samples for quality predictor")
                return False

            # Train quality prediction model
            await self._train_quality_model(X, y)

            # Update model state
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.utcnow()

            logger.info(f"Quality predictor trained on {len(training_data)} samples")
            return True

        except Exception as e:
            logger.error(f"Quality predictor training failed: {e}")
            return False

    async def predict(self, input_data: Dict[str, Any]) -> Prediction:
        """Predict quality score"""
        if not self.is_trained:
            raise ValueError("Quality model not trained")

        try:
            # Extract quality-specific features
            features = self._extract_quality_features(input_data)

            # Make quality prediction
            predicted_quality = await self._predict_quality(features)

            # Calculate confidence based on feature completeness and historical accuracy
            confidence = self._calculate_quality_confidence(features)

            # Calculate uncertainty estimate
            uncertainty = self._calculate_quality_uncertainty(features)

            prediction = Prediction(
                model_name=self.model_name,
                prediction_type=PredictionType.QUALITY,
                predicted_value=predicted_quality,
                confidence=confidence,
                uncertainty=uncertainty,
                input_features=features,
                prediction_context={
                    'model_type': self.model_type,
                    'quality_scale': f"0-{self.config['quality_scale']}",
                    'quality_interpretation': self._interpret_quality_score(predicted_quality)
                },
                expires_at=datetime.utcnow() + timedelta(hours=6)  # Quality predictions valid for 6 hours
            )

            return prediction

        except Exception as e:
            logger.error(f"Quality prediction failed: {e}")
            raise

    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate quality model performance on test data"""
        if not self.is_trained:
            raise ValueError("Quality model not trained")

        predictions = []
        actuals = []

        for sample in test_data:
            try:
                features = self._extract_quality_features(sample)
                predicted = await self._predict_quality(features)
                actual = sample.get('actual_quality_score', sample.get('quality_score', 0.0))

                predictions.append(predicted)
                actuals.append(actual)
            except Exception as e:
                logger.warning(f"Quality evaluation sample failed: {e}")

        if not predictions:
            return {'error': 'No valid predictions for quality evaluation'}

        # Calculate quality-specific metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 1.0))) * 100

        # Calculate R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Quality-specific metrics
        within_5_points = np.mean(np.abs(predictions - actuals) <= 5.0) * 100
        within_10_points = np.mean(np.abs(predictions - actuals) <= 10.0) * 100

        # Quality category accuracy (High: 80+, Medium: 60-80, Low: <60)
        pred_categories = np.where(predictions >= 80, 'high', np.where(predictions >= 60, 'medium', 'low'))
        actual_categories = np.where(actuals >= 80, 'high', np.where(actuals >= 60, 'medium', 'low'))
        category_accuracy = np.mean(pred_categories == actual_categories) * 100

        return {
            'mean_absolute_error': float(mae),
            'mean_squared_error': float(mse),
            'root_mean_squared_error': float(rmse),
            'mean_absolute_percentage_error': float(mape),
            'r_squared': float(r2),
            'within_5_points_accuracy': float(within_5_points),
            'within_10_points_accuracy': float(within_10_points),
            'category_accuracy': float(category_accuracy),
            'sample_count': len(predictions)
        }

    def _prepare_quality_training_data(self, training_data: List[Dict[str, Any]]) -> tuple:
        """Prepare training data for quality model"""
        X = []
        y = []

        for sample in training_data:
            try:
                features = self._extract_quality_features(sample)
                target = sample.get('actual_quality_score', sample.get('quality_score', 0.0))

                if features and isinstance(target, (int, float)):
                    # Ensure quality score is in valid range
                    target = max(self.config['min_quality_threshold'],
                               min(self.config['max_quality_threshold'], target))
                    X.append(list(features.values()))
                    y.append(target)
            except Exception as e:
                logger.warning(f"Skipping invalid quality training sample: {e}")

        return np.array(X), np.array(y)

    def _extract_quality_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for quality prediction"""
        features = {}

        # Extract quality-relevant features with sensible defaults
        features['code_complexity'] = float(data.get('code_complexity', data.get('complexity', 5.0)))
        features['test_coverage'] = float(data.get('test_coverage', data.get('coverage', 0.8)))
        features['documentation_score'] = float(data.get('documentation_score', data.get('docs_score', 0.7)))
        features['maintainability_index'] = float(data.get('maintainability_index', data.get('maintainability', 0.75)))
        features['error_density'] = float(data.get('error_density', data.get('errors_per_kloc', 2.0)))
        features['review_score'] = float(data.get('review_score', data.get('peer_review', 0.8)))
        features['experience_level'] = float(data.get('experience_level', data.get('team_experience', 0.7)))

        # Feature scaling if enabled
        if self.config['feature_scaling']:
            features = self._scale_quality_features(features)

        return features

    def _scale_quality_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale quality features to normalized range"""
        # Quality-specific scaling ranges
        scaling_ranges = {
            'code_complexity': (1.0, 20.0),  # Cyclomatic complexity
            'test_coverage': (0.0, 1.0),     # Already normalized
            'documentation_score': (0.0, 1.0),  # Already normalized
            'maintainability_index': (0.0, 1.0),  # Already normalized
            'error_density': (0.0, 10.0),    # Errors per KLOC
            'review_score': (0.0, 1.0),      # Already normalized
            'experience_level': (0.0, 1.0)   # Already normalized
        }

        scaled_features = {}
        for name, value in features.items():
            if name in scaling_ranges:
                min_val, max_val = scaling_ranges[name]
                if name in ['test_coverage', 'documentation_score', 'maintainability_index', 'review_score', 'experience_level']:
                    # These are already in 0-1 range
                    scaled_features[name] = max(0.0, min(1.0, value))
                else:
                    scaled_value = (value - min_val) / (max_val - min_val)
                    scaled_features[name] = max(0.0, min(1.0, scaled_value))
            else:
                scaled_features[name] = value

        return scaled_features

    async def _train_quality_model(self, X: np.ndarray, y: np.ndarray):
        """Train quality prediction model"""
        # Initialize weights for quality prediction
        n_features = X.shape[1]
        self.weights = {self.feature_names[i]: random.uniform(-1.0, 1.0) for i in range(min(n_features, len(self.feature_names)))}
        self.bias = random.uniform(40, 60)  # Start with medium quality bias

        # Gradient descent with quality-specific learning rate
        learning_rate = self.config['learning_rate']
        regularization = self.config['regularization']

        for epoch in range(150):  # More epochs for quality prediction
            # Forward pass
            predictions = []
            for sample in X:
                pred = self.bias
                for i, feature_name in enumerate(self.feature_names[:len(sample)]):
                    if feature_name in self.weights:
                        pred += self.weights[feature_name] * sample[i]
                # Ensure quality is in valid range
                pred = max(0.0, min(100.0, pred))
                predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate loss with regularization
            mse_loss = np.mean((predictions - y) ** 2)
            reg_loss = regularization * sum(w**2 for w in self.weights.values())
            total_loss = mse_loss + reg_loss

            # Backward pass
            errors = predictions - y

            # Update weights with regularization
            for i, feature_name in enumerate(self.feature_names[:X.shape[1]]):
                if feature_name in self.weights:
                    gradient = np.mean(errors * X[:, i]) + regularization * self.weights[feature_name]
                    self.weights[feature_name] -= learning_rate * gradient

            # Update bias
            self.bias -= learning_rate * np.mean(errors)
            self.bias = max(0.0, min(100.0, self.bias))  # Keep bias in valid range

            # Early stopping
            if total_loss < 5.0:  # Quality-specific threshold
                break

            # Yield control
            if epoch % 15 == 0:
                await asyncio.sleep(0.001)

    async def _predict_quality(self, features: Dict[str, float]) -> float:
        """Make quality prediction"""
        prediction = self.bias

        for feature_name, value in features.items():
            if feature_name in self.weights:
                prediction += self.weights[feature_name] * value

        # Ensure quality is in valid range
        prediction = max(self.config['min_quality_threshold'],
                        min(self.config['max_quality_threshold'], prediction))

        # Scale prediction if features were scaled
        if self.config['feature_scaling']:
            # Apply quality-specific scaling
            prediction = prediction * self.config['quality_scale'] / 100.0

        return prediction

    def _calculate_quality_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence for quality prediction"""
        base_confidence = 0.8

        # Boost confidence for complete feature sets
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        feature_completeness = len(present_features.intersection(expected_features)) / len(expected_features)

        confidence = base_confidence * feature_completeness

        # Adjust based on feature values
        for name, value in features.items():
            if name == 'test_coverage' and value > 0.8:
                confidence *= 1.1  # High test coverage increases confidence
            elif name == 'documentation_score' and value > 0.8:
                confidence *= 1.05  # Good documentation increases confidence
            elif name == 'error_density' and value > 0.8:  # High error density (scaled)
                confidence *= 0.9  # High error density reduces confidence

        return min(1.0, max(0.3, confidence))

    def _calculate_quality_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate uncertainty for quality prediction"""
        base_uncertainty = 0.2

        # Increase uncertainty for missing features
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        missing_ratio = 1 - (len(present_features.intersection(expected_features)) / len(expected_features))

        uncertainty = base_uncertainty + missing_ratio * 0.3

        # Increase uncertainty for concerning quality indicators
        test_coverage = features.get('test_coverage', 0.8)
        error_density = features.get('error_density', 0.2)  # Scaled value

        if test_coverage < 0.5:  # Low test coverage
            uncertainty += 0.15
        if error_density > 0.7:  # High error density
            uncertainty += 0.1

        return min(0.7, uncertainty)

    def _interpret_quality_score(self, quality_score: float) -> str:
        """Interpret quality score into human-readable categories"""
        if quality_score >= 90:
            return "Excellent"
        elif quality_score >= 80:
            return "Good"
        elif quality_score >= 70:
            return "Acceptable"
        elif quality_score >= 60:
            return "Below Average"
        elif quality_score >= 50:
            return "Poor"
        else:
            return "Very Poor"


class CostPredictor(BasePredictiveModel):
    """
    Predictor for cost estimation and resource pricing.
    """

    def __init__(self):
        super().__init__("cost_predictor", "regression")

        # Model parameters for cost prediction
        self.weights = {}
        self.bias = 0.0
        self.feature_names = [
            'resource_usage', 'time_duration_hours', 'complexity_factor',
            'team_size', 'infrastructure_tier', 'region_multiplier', 'priority_level'
        ]

        # Cost-specific configuration
        self.config.update({
            'learning_rate': 0.008,
            'regularization': 0.0003,
            'feature_scaling': True,
            'currency': 'USD',
            'max_cost_estimate': 100000.0,  # Maximum cost estimate
            'cost_categories': ['compute', 'storage', 'network', 'labor', 'licensing']
        })

    async def train(self, training_data: List[Dict[str, Any]]) -> bool:
        """Train the cost prediction model"""
        try:
            if len(training_data) < self.config['min_training_samples']:
                logger.warning(f"Insufficient training data for cost predictor: {len(training_data)} samples")
                return False

            # Extract features and cost targets
            X, y = self._prepare_cost_training_data(training_data)

            if len(X) == 0:
                logger.error("No valid training samples for cost predictor")
                return False

            # Train cost prediction model
            await self._train_cost_model(X, y)

            # Update model state
            self.is_trained = True
            self.training_data_size = len(training_data)
            self.last_training_time = datetime.utcnow()

            logger.info(f"Cost predictor trained on {len(training_data)} samples")
            return True

        except Exception as e:
            logger.error(f"Cost predictor training failed: {e}")
            return False

    async def predict(self, input_data: Dict[str, Any]) -> Prediction:
        """Predict cost estimate"""
        if not self.is_trained:
            raise ValueError("Cost model not trained")

        try:
            # Extract cost-specific features
            features = self._extract_cost_features(input_data)

            # Make cost prediction
            predicted_cost = await self._predict_cost(features)

            # Calculate confidence based on feature quality and historical accuracy
            confidence = self._calculate_cost_confidence(features)

            # Calculate uncertainty estimate
            uncertainty = self._calculate_cost_uncertainty(features)

            # Generate cost breakdown
            cost_breakdown = self._generate_cost_breakdown(features, predicted_cost)

            prediction = Prediction(
                model_name=self.model_name,
                prediction_type=PredictionType.COST,
                predicted_value=predicted_cost,
                confidence=confidence,
                uncertainty=uncertainty,
                input_features=features,
                prediction_context={
                    'model_type': self.model_type,
                    'currency': self.config['currency'],
                    'cost_breakdown': cost_breakdown,
                    'cost_range': f"0-{self.config['max_cost_estimate']} {self.config['currency']}"
                },
                expires_at=datetime.utcnow() + timedelta(hours=12)  # Cost predictions valid for 12 hours
            )

            return prediction

        except Exception as e:
            logger.error(f"Cost prediction failed: {e}")
            raise

    async def evaluate(self, test_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate cost model performance on test data"""
        if not self.is_trained:
            raise ValueError("Cost model not trained")

        predictions = []
        actuals = []

        for sample in test_data:
            try:
                features = self._extract_cost_features(sample)
                predicted = await self._predict_cost(features)
                actual = sample.get('actual_cost', sample.get('cost', 0.0))

                predictions.append(predicted)
                actuals.append(actual)
            except Exception as e:
                logger.warning(f"Cost evaluation sample failed: {e}")

        if not predictions:
            return {'error': 'No valid predictions for cost evaluation'}

        # Calculate cost-specific metrics
        predictions = np.array(predictions)
        actuals = np.array(actuals)

        mae = np.mean(np.abs(predictions - actuals))
        mse = np.mean((predictions - actuals) ** 2)
        rmse = np.sqrt(mse)

        # Mean Absolute Percentage Error (MAPE)
        mape = np.mean(np.abs((actuals - predictions) / np.maximum(actuals, 1.0))) * 100

        # Calculate R-squared
        ss_res = np.sum((actuals - predictions) ** 2)
        ss_tot = np.sum((actuals - np.mean(actuals)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0

        # Cost-specific metrics
        within_10_percent = np.mean(np.abs(predictions - actuals) / np.maximum(actuals, 1.0) <= 0.1) * 100
        within_20_percent = np.mean(np.abs(predictions - actuals) / np.maximum(actuals, 1.0) <= 0.2) * 100

        # Over/under estimation analysis
        overestimations = np.mean(predictions > actuals) * 100
        underestimations = np.mean(predictions < actuals) * 100

        return {
            'mean_absolute_error': float(mae),
            'mean_squared_error': float(mse),
            'root_mean_squared_error': float(rmse),
            'mean_absolute_percentage_error': float(mape),
            'r_squared': float(r2),
            'within_10_percent_accuracy': float(within_10_percent),
            'within_20_percent_accuracy': float(within_20_percent),
            'overestimation_rate': float(overestimations),
            'underestimation_rate': float(underestimations),
            'sample_count': len(predictions)
        }

    def _prepare_cost_training_data(self, training_data: List[Dict[str, Any]]) -> tuple:
        """Prepare training data for cost model"""
        X = []
        y = []

        for sample in training_data:
            try:
                features = self._extract_cost_features(sample)
                target = sample.get('actual_cost', sample.get('cost', 0.0))

                if features and isinstance(target, (int, float)) and target >= 0:
                    # Filter unrealistic cost estimates
                    if target <= self.config['max_cost_estimate']:
                        X.append(list(features.values()))
                        y.append(target)
            except Exception as e:
                logger.warning(f"Skipping invalid cost training sample: {e}")

        return np.array(X), np.array(y)

    def _extract_cost_features(self, data: Dict[str, Any]) -> Dict[str, float]:
        """Extract features for cost prediction"""
        features = {}

        # Extract cost-relevant features with sensible defaults
        features['resource_usage'] = float(data.get('resource_usage', data.get('cpu_hours', 10.0)))
        features['time_duration_hours'] = float(data.get('time_duration_hours', data.get('duration', 8.0)))
        features['complexity_factor'] = float(data.get('complexity_factor', data.get('complexity', 1.0)))
        features['team_size'] = float(data.get('team_size', data.get('developers', 3.0)))
        features['infrastructure_tier'] = float(data.get('infrastructure_tier', data.get('tier', 2.0)))
        features['region_multiplier'] = float(data.get('region_multiplier', data.get('region_cost', 1.0)))
        features['priority_level'] = float(data.get('priority_level', data.get('priority', 1.0)))

        # Feature scaling if enabled
        if self.config['feature_scaling']:
            features = self._scale_cost_features(features)

        return features

    def _scale_cost_features(self, features: Dict[str, float]) -> Dict[str, float]:
        """Scale cost features to normalized range"""
        # Cost-specific scaling ranges
        scaling_ranges = {
            'resource_usage': (1.0, 1000.0),      # CPU hours or equivalent
            'time_duration_hours': (1.0, 2000.0), # Project duration
            'complexity_factor': (0.5, 5.0),      # Complexity multiplier
            'team_size': (1.0, 50.0),             # Number of team members
            'infrastructure_tier': (1.0, 5.0),    # Infrastructure tier (1-5)
            'region_multiplier': (0.5, 3.0),      # Regional cost multiplier
            'priority_level': (1.0, 5.0)          # Priority level (1-5)
        }

        scaled_features = {}
        for name, value in features.items():
            if name in scaling_ranges:
                min_val, max_val = scaling_ranges[name]
                scaled_value = (value - min_val) / (max_val - min_val)
                scaled_features[name] = max(0.0, min(1.0, scaled_value))
            else:
                scaled_features[name] = value

        return scaled_features

    async def _train_cost_model(self, X: np.ndarray, y: np.ndarray):
        """Train cost prediction model"""
        # Initialize weights for cost prediction
        n_features = X.shape[1]
        self.weights = {self.feature_names[i]: random.uniform(-2.0, 2.0) for i in range(min(n_features, len(self.feature_names)))}
        self.bias = random.uniform(100, 1000)  # Base cost bias

        # Gradient descent with cost-specific learning rate
        learning_rate = self.config['learning_rate']
        regularization = self.config['regularization']

        for epoch in range(250):  # More epochs for cost prediction
            # Forward pass
            predictions = []
            for sample in X:
                pred = self.bias
                for i, feature_name in enumerate(self.feature_names[:len(sample)]):
                    if feature_name in self.weights:
                        pred += self.weights[feature_name] * sample[i]
                # Ensure non-negative cost
                pred = max(0.0, pred)
                predictions.append(pred)

            predictions = np.array(predictions)

            # Calculate loss with regularization
            mse_loss = np.mean((predictions - y) ** 2)
            reg_loss = regularization * sum(w**2 for w in self.weights.values())
            total_loss = mse_loss + reg_loss

            # Backward pass
            errors = predictions - y

            # Update weights with regularization
            for i, feature_name in enumerate(self.feature_names[:X.shape[1]]):
                if feature_name in self.weights:
                    gradient = np.mean(errors * X[:, i]) + regularization * self.weights[feature_name]
                    self.weights[feature_name] -= learning_rate * gradient

            # Update bias
            self.bias -= learning_rate * np.mean(errors)
            self.bias = max(0.0, self.bias)  # Keep bias non-negative

            # Early stopping
            if total_loss < 10.0:  # Cost-specific threshold
                break

            # Yield control
            if epoch % 25 == 0:
                await asyncio.sleep(0.001)

    async def _predict_cost(self, features: Dict[str, float]) -> float:
        """Make cost prediction"""
        prediction = self.bias

        for feature_name, value in features.items():
            if feature_name in self.weights:
                prediction += self.weights[feature_name] * value

        # Ensure non-negative cost and within reasonable bounds
        prediction = max(0.0, min(self.config['max_cost_estimate'], prediction))

        # Scale prediction if features were scaled
        if self.config['feature_scaling']:
            # Apply cost-specific scaling
            prediction = prediction * 100  # Scale up from normalized range

        return prediction

    def _calculate_cost_confidence(self, features: Dict[str, float]) -> float:
        """Calculate confidence for cost prediction"""
        base_confidence = 0.7

        # Boost confidence for complete feature sets
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        feature_completeness = len(present_features.intersection(expected_features)) / len(expected_features)

        confidence = base_confidence * feature_completeness

        # Adjust based on feature values
        complexity = features.get('complexity_factor', 0.5)
        if complexity > 0.8:  # High complexity reduces confidence
            confidence *= 0.9

        team_size = features.get('team_size', 0.5)
        if team_size > 0.9:  # Very large teams reduce confidence
            confidence *= 0.95

        return min(1.0, max(0.3, confidence))

    def _calculate_cost_uncertainty(self, features: Dict[str, float]) -> float:
        """Calculate uncertainty for cost prediction"""
        base_uncertainty = 0.25

        # Increase uncertainty for missing features
        expected_features = set(self.feature_names)
        present_features = set(features.keys())
        missing_ratio = 1 - (len(present_features.intersection(expected_features)) / len(expected_features))

        uncertainty = base_uncertainty + missing_ratio * 0.3

        # Increase uncertainty for high complexity or large scale
        complexity = features.get('complexity_factor', 0.5)
        if complexity > 0.7:
            uncertainty += 0.1

        duration = features.get('time_duration_hours', 0.5)
        if duration > 0.8:  # Long duration projects
            uncertainty += 0.05

        return min(0.8, uncertainty)

    def _generate_cost_breakdown(self, features: Dict[str, float], total_cost: float) -> Dict[str, float]:
        """Generate cost breakdown by category"""
        # Simple cost breakdown based on features
        breakdown = {}

        # Labor costs (typically 60-80% of total)
        team_factor = features.get('team_size', 0.5)
        duration_factor = features.get('time_duration_hours', 0.5)
        labor_percentage = 0.6 + (team_factor * 0.2)
        breakdown['labor'] = total_cost * labor_percentage

        # Infrastructure costs (10-25% of total)
        infra_factor = features.get('infrastructure_tier', 0.5)
        resource_factor = features.get('resource_usage', 0.5)
        infra_percentage = 0.1 + (infra_factor * 0.15)
        breakdown['infrastructure'] = total_cost * infra_percentage

        # Licensing costs (5-15% of total)
        complexity_factor = features.get('complexity_factor', 0.5)
        license_percentage = 0.05 + (complexity_factor * 0.1)
        breakdown['licensing'] = total_cost * license_percentage

        # Remaining costs distributed among other categories
        remaining = total_cost - sum(breakdown.values())
        breakdown['miscellaneous'] = max(0.0, remaining)

        return breakdown
