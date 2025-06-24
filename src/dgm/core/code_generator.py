import ast
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from ..models import ImprovementCandidate, ImprovementType, PerformanceMetric

logger = logging.getLogger(__name__)

class CodeGenerator:
    """Generate code improvements for DGM"""
    
    def __init__(self, config: Dict[str, Any]):
        self.config = config
        self.improvement_templates = self._load_improvement_templates()
    
    async def generate_improvement(
        self,
        agent_id: str,
        context: Dict[str, Any],
        baseline_performance: Optional[List[PerformanceMetric]] = None
    ) -> ImprovementCandidate:
        """Generate an improvement candidate"""
        
        # Analyze current performance bottlenecks
        bottlenecks = self._analyze_bottlenecks(baseline_performance)
        
        # Select improvement type based on context and bottlenecks
        improvement_type = self._select_improvement_type(context, bottlenecks)
        
        # Generate specific improvement
        if improvement_type == ImprovementType.PARAMETER_TUNING:
            return await self._generate_parameter_tuning(agent_id, context, bottlenecks)
        elif improvement_type == ImprovementType.ALGORITHM_MODIFICATION:
            return await self._generate_algorithm_modification(agent_id, context, bottlenecks)
        elif improvement_type == ImprovementType.CONFIGURATION_UPDATE:
            return await self._generate_configuration_update(agent_id, context, bottlenecks)
        else:
            # Default to parameter tuning
            return await self._generate_parameter_tuning(agent_id, context, bottlenecks)
    
    async def apply_changes(self, code_changes: Dict[str, str]):
        """Apply code changes to files"""
        for filename, new_code in code_changes.items():
            try:
                # Validate syntax before applying
                ast.parse(new_code)
                
                # Apply changes
                with open(filename, 'w') as f:
                    f.write(new_code)
                
                logger.info(f"Applied changes to {filename}")
                
            except SyntaxError as e:
                logger.error(f"Syntax error in generated code for {filename}: {e}")
                raise
            except Exception as e:
                logger.error(f"Failed to apply changes to {filename}: {e}")
                raise
    
    def _analyze_bottlenecks(self, performance_metrics: Optional[List[PerformanceMetric]]) -> List[str]:
        """Analyze performance bottlenecks"""
        if not performance_metrics:
            return []
        
        bottlenecks = []
        
        for metric in performance_metrics:
            if metric.name == "response_time" and metric.value > 1.0:
                bottlenecks.append("slow_response")
            elif metric.name == "memory_usage" and metric.value > 0.8:
                bottlenecks.append("high_memory")
            elif metric.name == "error_rate" and metric.value > 0.1:
                bottlenecks.append("high_errors")
            elif metric.name == "accuracy" and metric.value < 0.8:
                bottlenecks.append("low_accuracy")
        
        return bottlenecks
    
    def _select_improvement_type(self, context: Dict[str, Any], bottlenecks: List[str]) -> ImprovementType:
        """Select appropriate improvement type"""
        # Simple heuristic-based selection
        if "slow_response" in bottlenecks or "high_memory" in bottlenecks:
            return ImprovementType.PARAMETER_TUNING
        elif "low_accuracy" in bottlenecks:
            return ImprovementType.ALGORITHM_MODIFICATION
        elif "high_errors" in bottlenecks:
            return ImprovementType.CONFIGURATION_UPDATE
        else:
            return ImprovementType.PARAMETER_TUNING
    
    async def _generate_parameter_tuning(
        self, 
        agent_id: str, 
        context: Dict[str, Any], 
        bottlenecks: List[str]
    ) -> ImprovementCandidate:
        """Generate parameter tuning improvement"""
        
        # Example: Tune learning rate, batch size, etc.
        code_changes = {}
        
        # Find configuration file
        config_file = f"src/agents/{agent_id}/config.py"
        
        # Generate new configuration
        new_config = self._generate_tuned_parameters(bottlenecks)
        
        code_changes[config_file] = f"""# Auto-generated configuration improvement
# Generated at: {datetime.utcnow().isoformat()}

{new_config}
"""
        
        return ImprovementCandidate(
            id="",  # Will be set by caller
            agent_id=agent_id,
            improvement_type=ImprovementType.PARAMETER_TUNING,
            description=f"Parameter tuning to address: {', '.join(bottlenecks)}",
            code_changes=code_changes,
            expected_improvement=0.15,  # 15% expected improvement
            risk_level=0.2  # Low risk
        )
    
    async def _generate_algorithm_modification(
        self, 
        agent_id: str, 
        context: Dict[str, Any], 
        bottlenecks: List[str]
    ) -> ImprovementCandidate:
        """Generate algorithm modification improvement"""
        
        code_changes = {}
        
        # Example: Improve reasoning algorithm
        agent_file = f"src/agents/{agent_id}/reasoning.py"
        
        new_algorithm = self._generate_improved_algorithm(bottlenecks)
        
        code_changes[agent_file] = f"""# Auto-generated algorithm improvement
# Generated at: {datetime.utcnow().isoformat()}

{new_algorithm}
"""
        
        return ImprovementCandidate(
            id="",
            agent_id=agent_id,
            improvement_type=ImprovementType.ALGORITHM_MODIFICATION,
            description=f"Algorithm modification to address: {', '.join(bottlenecks)}",
            code_changes=code_changes,
            expected_improvement=0.25,  # 25% expected improvement
            risk_level=0.4  # Medium risk
        )
    
    async def _generate_configuration_update(
        self, 
        agent_id: str, 
        context: Dict[str, Any], 
        bottlenecks: List[str]
    ) -> ImprovementCandidate:
        """Generate configuration update improvement"""
        
        code_changes = {}
        
        # Example: Update error handling configuration
        config_file = f"src/agents/{agent_id}/config.json"
        
        new_config = self._generate_error_handling_config(bottlenecks)
        
        code_changes[config_file] = new_config
        
        return ImprovementCandidate(
            id="",
            agent_id=agent_id,
            improvement_type=ImprovementType.CONFIGURATION_UPDATE,
            description=f"Configuration update to address: {', '.join(bottlenecks)}",
            code_changes=code_changes,
            expected_improvement=0.1,  # 10% expected improvement
            risk_level=0.1  # Low risk
        )
    
    def _generate_tuned_parameters(self, bottlenecks: List[str]) -> str:
        """Generate tuned parameters based on bottlenecks"""
        config_lines = []
        
        if "slow_response" in bottlenecks:
            config_lines.append("RESPONSE_TIMEOUT = 0.5  # Reduced from 1.0")
            config_lines.append("BATCH_SIZE = 32  # Increased for efficiency")
        
        if "high_memory" in bottlenecks:
            config_lines.append("MAX_MEMORY_USAGE = 0.6  # Reduced from 0.8")
            config_lines.append("GARBAGE_COLLECTION_INTERVAL = 10  # More frequent GC")
        
        if "low_accuracy" in bottlenecks:
            config_lines.append("LEARNING_RATE = 0.001  # Reduced for stability")
            config_lines.append("EPOCHS = 50  # Increased training")
        
        return "\n".join(config_lines) if config_lines else "# No specific parameters to tune"
    
    def _generate_improved_algorithm(self, bottlenecks: List[str]) -> str:
        """Generate improved algorithm based on bottlenecks"""
        if "low_accuracy" in bottlenecks:
            return """
def improved_reasoning(self, input_data):
    '''Improved reasoning algorithm with better accuracy'''
    # Enhanced preprocessing
    preprocessed = self.enhanced_preprocess(input_data)
    
    # Multi-stage reasoning
    stage1_result = self.stage1_reasoning(preprocessed)
    stage2_result = self.stage2_reasoning(stage1_result)
    
    # Confidence scoring
    confidence = self.calculate_confidence(stage2_result)
    
    if confidence > 0.8:
        return stage2_result
    else:
        # Fallback to conservative approach
        return self.conservative_reasoning(preprocessed)
"""
        else:
            return """
def optimized_processing(self, data):
    '''Optimized processing for better performance'''
    # Batch processing for efficiency
    batch_size = 32
    results = []
    
    for i in range(0, len(data), batch_size):
        batch = data[i:i+batch_size]
        batch_result = self.process_batch(batch)
        results.extend(batch_result)
    
    return results
"""
    
    def _generate_error_handling_config(self, bottlenecks: List[str]) -> str:
        """Generate error handling configuration"""
        config = {
            "error_handling": {
                "max_retries": 3,
                "retry_delay": 1.0,
                "fallback_enabled": True
            },
            "logging": {
                "level": "INFO",
                "detailed_errors": True
            }
        }
        
        if "high_errors" in bottlenecks:
            config["error_handling"]["max_retries"] = 5
            config["error_handling"]["retry_delay"] = 0.5
            config["logging"]["level"] = "DEBUG"
        
        import json
        return json.dumps(config, indent=2)
    
    def _load_improvement_templates(self) -> Dict[str, str]:
        """Load improvement code templates"""
        # Placeholder - would load from files
        return {
            "parameter_tuning": "# Parameter tuning template",
            "algorithm_improvement": "# Algorithm improvement template",
            "configuration_update": "# Configuration update template"
        }
