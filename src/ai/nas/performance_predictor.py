"""
Performance Predictor for Recipe NAS

Predicts recipe performance without full execution using machine learning models.
Enables efficient architecture search by avoiding expensive evaluations.
"""

import logging
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import pickle
import json

from .architecture_encoder import RecipeArchitecture, NodeType, EdgeType
from .search_space import SearchSpace

# Set up logger first
logger = logging.getLogger(__name__)

try:
    from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error, r2_score
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    logger.warning("scikit-learn not available, using simple predictor")


@dataclass
class PerformancePrediction:
    """Prediction result for recipe performance"""
    success_probability: float
    execution_time_ms: float
    memory_usage_mb: float
    complexity_score: float
    confidence: float
    risk_factors: List[str] = field(default_factory=list)
    feature_importance: Dict[str, float] = field(default_factory=dict)
    prediction_timestamp: datetime = field(default_factory=datetime.utcnow)


@dataclass
class TrainingData:
    """Training data for performance prediction"""
    architecture_features: np.ndarray
    performance_targets: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class ArchitectureFeatureExtractor:
    """Extracts features from recipe architectures for ML models"""
    
    def __init__(self, search_space: SearchSpace):
        self.search_space = search_space
        self.feature_names = self._initialize_feature_names()
    
    def _initialize_feature_names(self) -> List[str]:
        """Initialize feature names for architecture representation"""
        features = []
        
        # Basic architecture features
        features.extend([
            "node_count", "edge_count", "depth", "width",
            "complexity_score", "connectivity_ratio"
        ])
        
        # Node type counts
        for node_type in NodeType:
            features.append(f"count_{node_type.value}")
        
        # Edge type counts
        for edge_type in EdgeType:
            features.append(f"edges_{edge_type.value}")
        
        # Operation category counts
        for category in self.search_space.operations_by_category:
            features.append(f"ops_{category.value}")
        
        # Graph topology features
        features.extend([
            "avg_node_degree", "max_node_degree", "clustering_coefficient",
            "path_length", "branching_factor", "parallel_paths"
        ])
        
        # Performance-related features
        features.extend([
            "estimated_compute_cost", "estimated_memory_cost",
            "dependency_complexity", "data_flow_efficiency"
        ])
        
        return features
    
    def extract_features(self, architecture: RecipeArchitecture) -> np.ndarray:
        """Extract feature vector from architecture"""
        features = []
        
        # Basic architecture features
        features.append(len(architecture.nodes))
        features.append(len(architecture.edges))
        features.append(self._calculate_depth(architecture))
        features.append(self._calculate_width(architecture))
        features.append(self._calculate_complexity_score(architecture))
        features.append(self._calculate_connectivity_ratio(architecture))
        
        # Node type counts
        node_type_counts = {nt: 0 for nt in NodeType}
        for node in architecture.nodes:
            node_type_counts[node.node_type] += 1
        
        for node_type in NodeType:
            features.append(node_type_counts[node_type])
        
        # Edge type counts
        edge_type_counts = {et: 0 for et in EdgeType}
        for edge in architecture.edges:
            edge_type_counts[edge.edge_type] += 1
        
        for edge_type in EdgeType:
            features.append(edge_type_counts[edge_type])
        
        # Operation category counts
        category_counts = {cat: 0 for cat in self.search_space.operations_by_category}
        for node in architecture.nodes:
            operation = self.search_space.get_operation_by_name(node.operation)
            if operation:
                category_counts[operation.category] += 1
        
        for category in self.search_space.operations_by_category:
            features.append(category_counts[category])
        
        # Graph topology features
        features.append(self._calculate_avg_node_degree(architecture))
        features.append(self._calculate_max_node_degree(architecture))
        features.append(self._calculate_clustering_coefficient(architecture))
        features.append(self._calculate_path_length(architecture))
        features.append(self._calculate_branching_factor(architecture))
        features.append(self._calculate_parallel_paths(architecture))
        
        # Performance-related features
        features.append(self._estimate_compute_cost(architecture))
        features.append(self._estimate_memory_cost(architecture))
        features.append(self._calculate_dependency_complexity(architecture))
        features.append(self._calculate_data_flow_efficiency(architecture))
        
        return np.array(features)
    
    def _calculate_depth(self, architecture: RecipeArchitecture) -> int:
        """Calculate maximum depth of the architecture"""
        # Simple approximation based on node positions
        if not architecture.nodes:
            return 0
        
        positions = [node.position[0] for node in architecture.nodes]
        return max(positions) - min(positions) + 1
    
    def _calculate_width(self, architecture: RecipeArchitecture) -> int:
        """Calculate maximum width of the architecture"""
        if not architecture.nodes:
            return 0
        
        # Group nodes by depth (x-coordinate)
        depth_groups = {}
        for node in architecture.nodes:
            depth = node.position[0]
            if depth not in depth_groups:
                depth_groups[depth] = 0
            depth_groups[depth] += 1
        
        return max(depth_groups.values()) if depth_groups else 0
    
    def _calculate_complexity_score(self, architecture: RecipeArchitecture) -> float:
        """Calculate complexity score based on operations"""
        total_complexity = 0.0
        
        for node in architecture.nodes:
            operation = self.search_space.get_operation_by_name(node.operation)
            if operation:
                total_complexity += operation.complexity_score
            else:
                total_complexity += 1.0  # Default complexity
        
        return total_complexity
    
    def _calculate_connectivity_ratio(self, architecture: RecipeArchitecture) -> float:
        """Calculate ratio of actual edges to maximum possible edges"""
        n_nodes = len(architecture.nodes)
        if n_nodes <= 1:
            return 0.0
        
        max_edges = n_nodes * (n_nodes - 1)  # Directed graph
        actual_edges = len(architecture.edges)
        
        return actual_edges / max_edges
    
    def _calculate_avg_node_degree(self, architecture: RecipeArchitecture) -> float:
        """Calculate average node degree"""
        if not architecture.nodes:
            return 0.0
        
        node_degrees = {}
        for node in architecture.nodes:
            node_degrees[node.id] = 0
        
        for edge in architecture.edges:
            if edge.source_node in node_degrees:
                node_degrees[edge.source_node] += 1
            if edge.target_node in node_degrees:
                node_degrees[edge.target_node] += 1
        
        return np.mean(list(node_degrees.values()))
    
    def _calculate_max_node_degree(self, architecture: RecipeArchitecture) -> int:
        """Calculate maximum node degree"""
        if not architecture.nodes:
            return 0
        
        node_degrees = {}
        for node in architecture.nodes:
            node_degrees[node.id] = 0
        
        for edge in architecture.edges:
            if edge.source_node in node_degrees:
                node_degrees[edge.source_node] += 1
            if edge.target_node in node_degrees:
                node_degrees[edge.target_node] += 1
        
        return max(node_degrees.values()) if node_degrees else 0
    
    def _calculate_clustering_coefficient(self, architecture: RecipeArchitecture) -> float:
        """Calculate clustering coefficient (simplified)"""
        # Simplified clustering coefficient calculation
        if len(architecture.nodes) < 3:
            return 0.0
        
        # Count triangles in the graph
        triangles = 0
        total_possible = 0
        
        for node in architecture.nodes:
            neighbors = set()
            for edge in architecture.edges:
                if edge.source_node == node.id:
                    neighbors.add(edge.target_node)
                elif edge.target_node == node.id:
                    neighbors.add(edge.source_node)
            
            if len(neighbors) >= 2:
                total_possible += len(neighbors) * (len(neighbors) - 1) // 2
                
                # Count actual connections between neighbors
                for n1 in neighbors:
                    for n2 in neighbors:
                        if n1 != n2:
                            for edge in architecture.edges:
                                if ((edge.source_node == n1 and edge.target_node == n2) or
                                    (edge.source_node == n2 and edge.target_node == n1)):
                                    triangles += 1
                                    break
        
        return triangles / max(total_possible, 1)
    
    def _calculate_path_length(self, architecture: RecipeArchitecture) -> float:
        """Calculate average path length (simplified)"""
        # Simplified path length calculation
        input_nodes = [n for n in architecture.nodes if n.node_type == NodeType.INPUT]
        output_nodes = [n for n in architecture.nodes if n.node_type == NodeType.OUTPUT]
        
        if not input_nodes or not output_nodes:
            return 0.0
        
        # Estimate path length based on depth
        return self._calculate_depth(architecture)
    
    def _calculate_branching_factor(self, architecture: RecipeArchitecture) -> float:
        """Calculate average branching factor"""
        if not architecture.nodes:
            return 0.0
        
        branching_factors = []
        for node in architecture.nodes:
            outgoing_edges = len([e for e in architecture.edges if e.source_node == node.id])
            branching_factors.append(outgoing_edges)
        
        return np.mean(branching_factors)
    
    def _calculate_parallel_paths(self, architecture: RecipeArchitecture) -> int:
        """Calculate number of parallel execution paths"""
        parallel_nodes = [n for n in architecture.nodes if n.node_type == NodeType.PARALLEL]
        return len(parallel_nodes)
    
    def _estimate_compute_cost(self, architecture: RecipeArchitecture) -> float:
        """Estimate computational cost"""
        cost = 0.0
        
        for node in architecture.nodes:
            operation = self.search_space.get_operation_by_name(node.operation)
            if operation:
                cost += operation.complexity_score * 100  # Scale to reasonable range
            else:
                cost += 100  # Default cost
        
        return cost
    
    def _estimate_memory_cost(self, architecture: RecipeArchitecture) -> float:
        """Estimate memory cost"""
        # Simple heuristic based on node count and complexity
        base_memory = len(architecture.nodes) * 50  # MB per node
        
        # Add complexity-based memory
        complexity_memory = self._calculate_complexity_score(architecture) * 20
        
        return base_memory + complexity_memory
    
    def _calculate_dependency_complexity(self, architecture: RecipeArchitecture) -> float:
        """Calculate dependency complexity"""
        dependency_edges = [e for e in architecture.edges if e.edge_type == EdgeType.DEPENDENCY]
        return len(dependency_edges) / max(len(architecture.edges), 1)
    
    def _calculate_data_flow_efficiency(self, architecture: RecipeArchitecture) -> float:
        """Calculate data flow efficiency"""
        data_flow_edges = [e for e in architecture.edges if e.edge_type == EdgeType.DATA_FLOW]
        total_edges = len(architecture.edges)
        
        if total_edges == 0:
            return 0.0
        
        return len(data_flow_edges) / total_edges


class PerformancePredictor:
    """
    Predicts recipe performance using machine learning models.
    
    Uses architecture features to predict success probability,
    execution time, memory usage, and other performance metrics.
    """
    
    def __init__(self, search_space: Optional[SearchSpace] = None):
        self.search_space = search_space or SearchSpace()
        self.feature_extractor = ArchitectureFeatureExtractor(self.search_space)
        
        # ML models
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        
        # Training history
        self.training_history = []
        self.feature_importance = {}
        
        # Initialize models if sklearn is available
        if SKLEARN_AVAILABLE:
            self._initialize_models()
    
    def _initialize_models(self):
        """Initialize ML models for different performance metrics"""
        self.models = {
            'success_probability': RandomForestRegressor(n_estimators=100, random_state=42),
            'execution_time': GradientBoostingRegressor(n_estimators=100, random_state=42),
            'memory_usage': RandomForestRegressor(n_estimators=100, random_state=42),
            'complexity_score': MLPRegressor(hidden_layer_sizes=(50, 25), random_state=42)
        }
        
        self.scalers = {
            metric: StandardScaler() for metric in self.models.keys()
        }
    
    async def predict_performance(self, architecture: RecipeArchitecture) -> PerformancePrediction:
        """
        Predict performance for a given architecture.
        
        Args:
            architecture: Architecture to predict performance for
            
        Returns:
            Performance prediction
        """
        try:
            # Extract features
            features = self.feature_extractor.extract_features(architecture)
            
            if SKLEARN_AVAILABLE and self.is_trained:
                # Use trained ML models
                predictions = self._predict_with_models(features)
            else:
                # Use heuristic predictions
                predictions = self._predict_with_heuristics(architecture, features)
            
            # Calculate confidence
            confidence = self._calculate_confidence(architecture, features)
            
            # Identify risk factors
            risk_factors = self._identify_risk_factors(architecture, features)
            
            prediction = PerformancePrediction(
                success_probability=predictions['success_probability'],
                execution_time_ms=predictions['execution_time'],
                memory_usage_mb=predictions['memory_usage'],
                complexity_score=predictions['complexity_score'],
                confidence=confidence,
                risk_factors=risk_factors,
                feature_importance=self.feature_importance.copy()
            )
            
            logger.debug(f"Predicted performance: success={prediction.success_probability:.3f}, "
                        f"time={prediction.execution_time_ms:.0f}ms, "
                        f"memory={prediction.memory_usage_mb:.0f}MB")
            
            return prediction
            
        except Exception as e:
            logger.error(f"Performance prediction failed: {e}")
            return self._create_default_prediction()
    
    def _predict_with_models(self, features: np.ndarray) -> Dict[str, float]:
        """Predict using trained ML models"""
        predictions = {}
        
        features_scaled = features.reshape(1, -1)
        
        for metric, model in self.models.items():
            try:
                # Scale features
                scaler = self.scalers[metric]
                features_normalized = scaler.transform(features_scaled)
                
                # Make prediction
                prediction = model.predict(features_normalized)[0]
                
                # Apply bounds
                if metric == 'success_probability':
                    prediction = np.clip(prediction, 0.0, 1.0)
                elif metric in ['execution_time', 'memory_usage']:
                    prediction = max(0.0, prediction)
                
                predictions[metric] = float(prediction)
                
            except Exception as e:
                logger.warning(f"Model prediction failed for {metric}: {e}")
                predictions[metric] = self._get_heuristic_prediction(metric, features)
        
        return predictions
    
    def _predict_with_heuristics(self, architecture: RecipeArchitecture, 
                                features: np.ndarray) -> Dict[str, float]:
        """Predict using heuristic methods"""
        predictions = {}
        
        # Extract key features
        node_count = features[0]
        edge_count = features[1]
        complexity_score = features[4]
        
        # Heuristic predictions
        predictions['success_probability'] = max(0.1, min(0.95, 
            0.8 - (complexity_score - 5.0) * 0.05))
        
        predictions['execution_time'] = (
            node_count * 200 +  # Base time per node
            edge_count * 50 +   # Time per edge
            complexity_score * 300  # Complexity penalty
        )
        
        predictions['memory_usage'] = (
            node_count * 64 +   # Base memory per node
            complexity_score * 32  # Complexity memory
        )
        
        predictions['complexity_score'] = complexity_score
        
        return predictions
    
    def _get_heuristic_prediction(self, metric: str, features: np.ndarray) -> float:
        """Get heuristic prediction for a specific metric"""
        node_count = features[0]
        complexity_score = features[4]
        
        if metric == 'success_probability':
            return max(0.1, min(0.95, 0.8 - complexity_score * 0.05))
        elif metric == 'execution_time':
            return node_count * 200 + complexity_score * 300
        elif metric == 'memory_usage':
            return node_count * 64 + complexity_score * 32
        elif metric == 'complexity_score':
            return complexity_score
        else:
            return 0.5
    
    def _calculate_confidence(self, architecture: RecipeArchitecture, 
                            features: np.ndarray) -> float:
        """Calculate confidence in the prediction"""
        confidence = 0.5  # Base confidence
        
        # Increase confidence if we have training data
        if self.is_trained:
            confidence += 0.3
        
        # Decrease confidence for very complex architectures
        complexity_score = features[4]
        if complexity_score > 10:
            confidence -= 0.2
        
        # Decrease confidence for unusual architectures
        node_count = features[0]
        if node_count > 30 or node_count < 3:
            confidence -= 0.1
        
        return max(0.1, min(0.95, confidence))
    
    def _identify_risk_factors(self, architecture: RecipeArchitecture, 
                             features: np.ndarray) -> List[str]:
        """Identify potential risk factors"""
        risk_factors = []
        
        node_count = features[0]
        edge_count = features[1]
        complexity_score = features[4]
        
        if complexity_score > 15:
            risk_factors.append("high_complexity")
        
        if node_count > 25:
            risk_factors.append("large_architecture")
        
        if edge_count > node_count * 2:
            risk_factors.append("high_connectivity")
        
        # Check for specific risky patterns
        parallel_nodes = [n for n in architecture.nodes if n.node_type == NodeType.PARALLEL]
        if len(parallel_nodes) > 3:
            risk_factors.append("excessive_parallelism")
        
        feedback_edges = [e for e in architecture.edges if e.edge_type == EdgeType.FEEDBACK]
        if len(feedback_edges) > 2:
            risk_factors.append("complex_feedback_loops")
        
        return risk_factors
    
    def _create_default_prediction(self) -> PerformancePrediction:
        """Create default prediction for error cases"""
        return PerformancePrediction(
            success_probability=0.5,
            execution_time_ms=5000,
            memory_usage_mb=512,
            complexity_score=5.0,
            confidence=0.1,
            risk_factors=["prediction_failed"]
        )
    
    def train_models(self, training_data: List[Tuple[RecipeArchitecture, Dict[str, float]]]):
        """Train the prediction models with historical data"""
        if not SKLEARN_AVAILABLE:
            logger.warning("Cannot train models: scikit-learn not available")
            return
        
        if len(training_data) < 10:
            logger.warning("Insufficient training data")
            return
        
        try:
            # Extract features and targets
            features_list = []
            targets = {metric: [] for metric in self.models.keys()}
            
            for architecture, performance_data in training_data:
                features = self.feature_extractor.extract_features(architecture)
                features_list.append(features)
                
                for metric in self.models.keys():
                    targets[metric].append(performance_data.get(metric, 0.0))
            
            X = np.array(features_list)
            
            # Train each model
            for metric, model in self.models.items():
                y = np.array(targets[metric])
                
                # Split data
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )
                
                # Scale features
                scaler = self.scalers[metric]
                X_train_scaled = scaler.fit_transform(X_train)
                X_test_scaled = scaler.transform(X_test)
                
                # Train model
                model.fit(X_train_scaled, y_train)
                
                # Evaluate
                y_pred = model.predict(X_test_scaled)
                mse = mean_squared_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                
                logger.info(f"Model {metric}: MSE={mse:.4f}, R2={r2:.4f}")
                
                # Store feature importance
                if hasattr(model, 'feature_importances_'):
                    importance_dict = dict(zip(
                        self.feature_extractor.feature_names,
                        model.feature_importances_
                    ))
                    self.feature_importance[metric] = importance_dict
            
            self.is_trained = True
            self.training_history.append({
                'timestamp': datetime.utcnow(),
                'training_size': len(training_data),
                'metrics_trained': list(self.models.keys())
            })
            
            logger.info(f"Successfully trained models with {len(training_data)} samples")
            
        except Exception as e:
            logger.error(f"Model training failed: {e}")
    
    def save_models(self, filepath: str):
        """Save trained models to file"""
        if not self.is_trained:
            logger.warning("No trained models to save")
            return
        
        model_data = {
            'models': self.models,
            'scalers': self.scalers,
            'feature_names': self.feature_extractor.feature_names,
            'training_history': self.training_history,
            'feature_importance': self.feature_importance
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(model_data, f)
        
        logger.info(f"Models saved to {filepath}")
    
    def load_models(self, filepath: str):
        """Load trained models from file"""
        try:
            with open(filepath, 'rb') as f:
                model_data = pickle.load(f)
            
            self.models = model_data['models']
            self.scalers = model_data['scalers']
            self.training_history = model_data.get('training_history', [])
            self.feature_importance = model_data.get('feature_importance', {})
            
            self.is_trained = True
            logger.info(f"Models loaded from {filepath}")
            
        except Exception as e:
            logger.error(f"Failed to load models: {e}")
    
    def get_predictor_status(self) -> Dict[str, Any]:
        """Get status of the performance predictor"""
        return {
            'is_trained': self.is_trained,
            'sklearn_available': SKLEARN_AVAILABLE,
            'models': list(self.models.keys()) if self.models else [],
            'feature_count': len(self.feature_extractor.feature_names),
            'training_history_count': len(self.training_history),
            'last_training': self.training_history[-1] if self.training_history else None
        }
