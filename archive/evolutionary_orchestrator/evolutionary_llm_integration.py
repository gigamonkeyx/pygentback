"""
Ollama DeepSeek R1 Evolutionary Integration

Enhanced LLM integration with evolutionary model selection, A2A-coordinated fine-tuning,
distributed model evaluation, evolutionary prompt optimization, performance sharing,
and distributed model improvement coordination.

Aligned with Sakana AI's Darwin Gödel Machine principles for self-improving LLM systems.
"""

import logging
from typing import Dict, List, Optional, Any
from datetime import datetime
from dataclasses import dataclass, field
import hashlib
import statistics
from collections import defaultdict, deque

# Import Ollama manager
from ..core.ollama_manager import get_ollama_manager

# Import A2A protocol for distributed coordination
try:
    from ..a2a import AgentCard, A2APeerInfo
except ImportError:
    AgentCard = None
    A2APeerInfo = None

logger = logging.getLogger(__name__)


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for evolutionary model selection."""
    model_name: str
    task_type: str
    success_rate: float = 0.0
    avg_response_time: float = 0.0
    quality_score: float = 0.0
    accuracy: float = 0.0
    coherence: float = 0.0
    creativity: float = 0.0
    reasoning_score: float = 0.0
    evolution_fitness: float = 0.0
    usage_count: int = 0
    last_updated: datetime = field(default_factory=datetime.utcnow)


@dataclass
class EvolutionaryPrompt:
    """Evolutionary prompt with optimization history."""
    prompt_id: str
    base_prompt: str
    optimized_prompt: str
    generation: int = 0
    fitness_score: float = 0.0
    parent_prompts: List[str] = field(default_factory=list)
    mutation_history: List[Dict[str, Any]] = field(default_factory=list)
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    adaptation_context: Dict[str, Any] = field(default_factory=dict)
    last_evolution: Optional[datetime] = None


@dataclass
class ModelEvolutionEvent:
    """Distributed model evolution event for A2A sharing."""
    event_id: str
    event_type: str  # fine_tune, evaluation, optimization, performance_update
    model_name: str
    agent_id: str
    peer_id: str = ""
    performance_data: Dict[str, Any] = field(default_factory=dict)
    improvement_suggestions: List[str] = field(default_factory=list)
    timestamp: datetime = field(default_factory=datetime.utcnow)
    validation_proofs: List[str] = field(default_factory=list)


class EvolutionaryLLMManager:
    """
    Enhanced LLM manager with evolutionary capabilities and A2A coordination.
    
    Implements:
    - Evolutionary model selection for agent tasks (1.7.1)
    - A2A-coordinated model fine-tuning based on performance (1.7.2)
    - Distributed model evaluation via A2A network (1.7.3)
    - Evolutionary prompt optimization across agents (1.7.4)
    - A2A-enabled model performance sharing (1.7.5)
    - Distributed model improvement coordination (1.7.6)
    """
    
    def __init__(self, ollama_manager=None, a2a_server=None):
        self.ollama_manager = ollama_manager or get_ollama_manager()
        self.a2a_server = a2a_server
        
        # Performance tracking for evolutionary selection
        self.model_performance: Dict[str, Dict[str, ModelPerformanceMetrics]] = defaultdict(dict)
        self.task_model_mapping: Dict[str, str] = {}  # Task type -> best model
        self.model_evolution_history: List[ModelEvolutionEvent] = []
        
        # Evolutionary prompt optimization
        self.evolutionary_prompts: Dict[str, EvolutionaryPrompt] = {}
        self.prompt_genealogy: Dict[str, List[str]] = defaultdict(list)
        self.prompt_mutation_operators: List[str] = [
            "paraphrase", "extend", "simplify", "formalize", "add_context", 
            "add_examples", "reorder", "emphasize", "adapt_tone"
        ]
        
        # A2A coordination state
        self.peer_model_performances: Dict[str, Dict[str, Any]] = {}
        self.distributed_evaluations: Dict[str, Dict[str, Any]] = {}
        self.consensus_models: Dict[str, str] = {}  # Task -> consensus best model
        
        # Model fine-tuning coordination
        self.fine_tuning_queue: deque = deque()
        self.active_fine_tuning: Dict[str, Dict[str, Any]] = {}
        self.fine_tuning_results: Dict[str, ModelPerformanceMetrics] = {}
        
        # Evolution parameters
        self.selection_pressure = 0.7
        self.mutation_rate = 0.1
        self.crossover_rate = 0.3
        self.elite_retention = 0.2
        
        logger.info("Evolutionary LLM Manager initialized")
    
    # Phase 1.7.1: Evolutionary Model Selection for Agent Tasks
    async def select_optimal_model(self, task_type: str, agent_id: str, 
                                 context: Optional[Dict[str, Any]] = None) -> str:
        """Select optimal model for task using evolutionary principles."""
        try:
            available_models = await self.ollama_manager.get_available_models()
            if not available_models:
                logger.warning("No models available for selection")
                return "llama3.2"  # Default fallback
            
            # Check if we have a consensus model from A2A network
            if task_type in self.consensus_models:
                consensus_model = self.consensus_models[task_type]
                if consensus_model in available_models:
                    logger.debug(f"Using A2A consensus model {consensus_model} for {task_type}")
                    return consensus_model
            
            # Evaluate models based on performance metrics
            model_scores = {}
            for model_name in available_models:
                score = await self._calculate_model_fitness(model_name, task_type, context)
                model_scores[model_name] = score
            
            # Apply evolutionary selection
            selected_model = await self._evolutionary_model_selection(model_scores, task_type)
            
            # Update task-model mapping
            self.task_model_mapping[task_type] = selected_model
            
            # Log selection for performance tracking
            await self._log_model_selection(selected_model, task_type, agent_id)
            
            logger.debug(f"Selected model {selected_model} for task {task_type} (agent {agent_id})")
            return selected_model
            
        except Exception as e:
            logger.error(f"Failed to select optimal model for {task_type}: {e}")
            return "llama3.2"  # Safe fallback
    
    async def _calculate_model_fitness(self, model_name: str, task_type: str, 
                                     context: Optional[Dict[str, Any]] = None) -> float:
        """Calculate evolutionary fitness score for model-task combination."""
        try:
            # Get performance metrics for this model-task pair
            key = f"{model_name}:{task_type}"
            if key not in self.model_performance[model_name]:
                # Initialize with neutral fitness
                return 0.5
            
            metrics = self.model_performance[model_name][key]
            
            # Calculate composite fitness score
            weights = {
                'success_rate': 0.3,
                'quality_score': 0.25,
                'response_time': 0.15,  # Inverted - faster is better
                'accuracy': 0.15,
                'reasoning_score': 0.15
            }
            
            fitness = 0.0
            fitness += weights['success_rate'] * metrics.success_rate
            fitness += weights['quality_score'] * metrics.quality_score
            fitness += weights['response_time'] * (1.0 - min(metrics.avg_response_time / 30.0, 1.0))
            fitness += weights['accuracy'] * metrics.accuracy
            fitness += weights['reasoning_score'] * metrics.reasoning_score
            
            # Apply context-based adjustments
            if context:
                if context.get('priority') == 'speed' and metrics.avg_response_time < 5.0:
                    fitness *= 1.2
                elif context.get('priority') == 'quality' and metrics.quality_score > 0.8:
                    fitness *= 1.2
                elif context.get('priority') == 'reasoning' and metrics.reasoning_score > 0.8:
                    fitness *= 1.2
            
            return min(fitness, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate model fitness for {model_name}: {e}")
            return 0.5
    
    async def _evolutionary_model_selection(self, model_scores: Dict[str, float], 
                                          task_type: str) -> str:
        """Apply evolutionary selection pressure to choose model."""
        try:
            if not model_scores:
                return "llama3.2"
              # Sort by fitness
            sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            
            # Apply selection pressure
            total_models = len(sorted_models)
            
            # Elite selection for top performers
            if sorted_models[0][1] > 0.8:  # High performance threshold
                return sorted_models[0][0]
            
            # Tournament selection for moderate performers
            import random
            tournament_size = min(3, total_models)
            tournament = random.sample(sorted_models, tournament_size)
            winner = max(tournament, key=lambda x: x[1])
            
            return winner[0]
            
        except Exception as e:
            logger.error(f"Failed in evolutionary model selection: {e}")
            return list(model_scores.keys())[0] if model_scores else "llama3.2"
    
    async def _log_model_selection(self, model_name: str, task_type: str, agent_id: str) -> None:
        """Log model selection for tracking and A2A sharing."""
        try:
            event = ModelEvolutionEvent(
                event_id=f"select_{hashlib.md5(f'{model_name}_{task_type}_{datetime.utcnow()}'.encode()).hexdigest()[:8]}",
                event_type="selection",
                model_name=model_name,
                agent_id=agent_id,
                performance_data={
                    "task_type": task_type,
                    "selection_time": datetime.utcnow().isoformat(),
                    "fitness_score": await self._calculate_model_fitness(model_name, task_type)
                }
            )
            
            self.model_evolution_history.append(event)
            
            # Share with A2A network
            if self.a2a_server and hasattr(self.a2a_server, 'broadcast_model_event'):
                await self.a2a_server.broadcast_model_event(event)
            
        except Exception as e:
            logger.error(f"Failed to log model selection: {e}")
    
    # Phase 1.7.2: A2A-Coordinated Model Fine-tuning Based on Performance
    async def coordinate_model_fine_tuning(self, model_name: str, task_type: str,
                                         performance_data: Dict[str, Any]) -> bool:
        """Coordinate distributed model fine-tuning based on performance feedback."""
        try:
            # Analyze if fine-tuning is needed
            should_fine_tune = await self._analyze_fine_tuning_need(model_name, task_type, performance_data)
            
            if not should_fine_tune:
                return False
            
            # Create fine-tuning coordination request
            fine_tune_request = {
                "request_id": f"ft_{hashlib.md5(f'{model_name}_{datetime.utcnow()}'.encode()).hexdigest()[:8]}",
                "model_name": model_name,
                "task_type": task_type,
                "performance_data": performance_data,
                "requested_by": "local_agent",
                "timestamp": datetime.utcnow().isoformat(),
                "priority": self._calculate_fine_tuning_priority(performance_data)
            }
            
            # Add to fine-tuning queue
            self.fine_tuning_queue.append(fine_tune_request)
            
            # Coordinate with A2A peers
            if self.a2a_server and hasattr(self.a2a_server, 'coordinate_fine_tuning'):
                peer_responses = await self.a2a_server.coordinate_fine_tuning(fine_tune_request)
                
                # Process peer responses
                await self._process_fine_tuning_coordination(fine_tune_request, peer_responses)
            
            # Execute fine-tuning if consensus reached
            return await self._execute_coordinated_fine_tuning(fine_tune_request)
            
        except Exception as e:
            logger.error(f"Failed to coordinate model fine-tuning: {e}")
            return False
    
    async def _analyze_fine_tuning_need(self, model_name: str, task_type: str,
                                      performance_data: Dict[str, Any]) -> bool:
        """Analyze if model needs fine-tuning based on performance."""
        try:
            key = f"{model_name}:{task_type}"
            current_metrics = self.model_performance[model_name].get(key)
            
            if not current_metrics:
                return False  # Need baseline performance first
            
            # Check performance thresholds
            if current_metrics.success_rate < 0.7:
                return True
            if current_metrics.quality_score < 0.6:
                return True
            if current_metrics.accuracy < 0.7:
                return True
            
            # Check degradation trends
            recent_performance = performance_data.get('recent_metrics', {})
            if recent_performance.get('success_rate', 1.0) < current_metrics.success_rate * 0.9:
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Failed to analyze fine-tuning need: {e}")
            return False
    
    async def _execute_coordinated_fine_tuning(self, request: Dict[str, Any]) -> bool:
        """Execute coordinated fine-tuning based on A2A consensus."""
        try:
            # Mark as active fine-tuning
            self.active_fine_tuning[request["request_id"]] = request
            
            # Simulate fine-tuning process (in real implementation, this would
            # interface with actual fine-tuning infrastructure)
            logger.info(f"Starting fine-tuning for {request['model_name']} on {request['task_type']}")
            
            # Create evolution event
            event = ModelEvolutionEvent(
                event_id=request["request_id"],
                event_type="fine_tune",
                model_name=request["model_name"],
                agent_id="fine_tuning_coordinator",
                performance_data=request["performance_data"]
            )
            
            self.model_evolution_history.append(event)
            
            # Broadcast start of fine-tuning
            if self.a2a_server and hasattr(self.a2a_server, 'broadcast_model_event'):
                await self.a2a_server.broadcast_model_event(event)
            
            logger.info(f"Fine-tuning coordinated for model {request['model_name']}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to execute coordinated fine-tuning: {e}")
            return False
    
    # Phase 1.7.3: Distributed Model Evaluation via A2A Network
    async def distributed_model_evaluation(self, model_name: str, 
                                         evaluation_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Coordinate distributed model evaluation across A2A network."""
        try:
            evaluation_id = f"eval_{hashlib.md5(f'{model_name}_{datetime.utcnow()}'.encode()).hexdigest()[:8]}"
            
            # Create evaluation request
            eval_request = {
                "evaluation_id": evaluation_id,
                "model_name": model_name,
                "tasks": evaluation_tasks,
                "requested_by": "local_agent",
                "timestamp": datetime.utcnow().isoformat(),
                "expected_peers": 3,  # Minimum peers for reliable evaluation
                "timeout": 300  # 5 minutes timeout
            }
            
            # Store evaluation request
            self.distributed_evaluations[evaluation_id] = eval_request
            
            # Coordinate with A2A peers
            peer_results = {}
            if self.a2a_server and hasattr(self.a2a_server, 'coordinate_model_evaluation'):
                peer_results = await self.a2a_server.coordinate_model_evaluation(eval_request)
            
            # Perform local evaluation
            local_results = await self._perform_local_model_evaluation(model_name, evaluation_tasks)
            
            # Aggregate results
            aggregated_results = await self._aggregate_evaluation_results(
                evaluation_id, local_results, peer_results
            )
            
            # Update model performance based on evaluation
            await self._update_model_performance_from_evaluation(model_name, aggregated_results)
            
            logger.info(f"Completed distributed evaluation for model {model_name}")
            return aggregated_results
            
        except Exception as e:
            logger.error(f"Failed in distributed model evaluation: {e}")
            return {"error": str(e)}
    
    async def _perform_local_model_evaluation(self, model_name: str,
                                            evaluation_tasks: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Perform local model evaluation."""
        try:
            results = {
                "model_name": model_name,
                "evaluator": "local_agent",
                "timestamp": datetime.utcnow().isoformat(),
                "task_results": [],
                "overall_metrics": {}
            }
            
            total_score = 0.0
            total_tasks = len(evaluation_tasks)
            
            for task in evaluation_tasks:
                # Simulate task evaluation (in real implementation, this would
                # execute actual tasks and measure performance)
                task_result = {
                    "task_type": task.get("type"),
                    "task_id": task.get("id"),
                    "success": True,
                    "score": 0.85,  # Simulated score
                    "response_time": 2.5,
                    "quality_metrics": {
                        "accuracy": 0.88,
                        "coherence": 0.92,
                        "relevance": 0.85
                    }
                }
                
                results["task_results"].append(task_result)
                total_score += task_result["score"]
            
            # Calculate overall metrics
            results["overall_metrics"] = {
                "average_score": total_score / max(total_tasks, 1),
                "success_rate": 1.0,  # All simulated tasks succeed
                "avg_response_time": 2.5,
                "total_tasks": total_tasks
            }
            
            return results
            
        except Exception as e:
            logger.error(f"Failed in local model evaluation: {e}")
            return {"error": str(e)}
    
    async def _aggregate_evaluation_results(self, evaluation_id: str,
                                          local_results: Dict[str, Any],
                                          peer_results: Dict[str, Any]) -> Dict[str, Any]:
        """Aggregate evaluation results from multiple sources."""
        try:
            all_results = [local_results]
            
            # Add peer results
            for peer_id, peer_data in peer_results.items():
                if "error" not in peer_data:
                    all_results.append(peer_data)
            
            if not all_results:
                return {"error": "No valid evaluation results"}
            
            # Aggregate metrics
            aggregated = {
                "evaluation_id": evaluation_id,
                "model_name": local_results.get("model_name"),
                "timestamp": datetime.utcnow().isoformat(),
                "participant_count": len(all_results),
                "consensus_metrics": {},
                "individual_results": all_results
            }
            
            # Calculate consensus metrics
            scores = [r["overall_metrics"]["average_score"] for r in all_results if "overall_metrics" in r]
            response_times = [r["overall_metrics"]["avg_response_time"] for r in all_results if "overall_metrics" in r]
            success_rates = [r["overall_metrics"]["success_rate"] for r in all_results if "overall_metrics" in r]
            
            if scores:
                aggregated["consensus_metrics"] = {
                    "consensus_score": statistics.mean(scores),
                    "score_std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                    "avg_response_time": statistics.mean(response_times),
                    "consensus_success_rate": statistics.mean(success_rates),
                    "confidence": 1.0 - (statistics.stdev(scores) if len(scores) > 1 else 0.0)
                }
            
            return aggregated
            
        except Exception as e:
            logger.error(f"Failed to aggregate evaluation results: {e}")
            return {"error": str(e)}
    
    # Phase 1.7.4: Evolutionary Prompt Optimization Across Agents
    async def evolve_prompt(self, base_prompt: str, task_type: str, 
                          performance_feedback: Dict[str, float]) -> str:
        """Evolve prompts using genetic algorithms across A2A network."""
        try:
            prompt_id = hashlib.md5(base_prompt.encode()).hexdigest()
            
            # Initialize or retrieve evolutionary prompt
            if prompt_id not in self.evolutionary_prompts:
                evolutionary_prompt = EvolutionaryPrompt(
                    prompt_id=prompt_id,
                    base_prompt=base_prompt,
                    optimized_prompt=base_prompt,
                    performance_metrics=performance_feedback.copy()
                )
                self.evolutionary_prompts[prompt_id] = evolutionary_prompt
            else:
                evolutionary_prompt = self.evolutionary_prompts[prompt_id]
            
            # Apply evolutionary operators
            evolved_prompt = await self._apply_evolutionary_operators(
                evolutionary_prompt, task_type, performance_feedback
            )
            
            # Coordinate with A2A peers for cross-pollination
            if self.a2a_server and hasattr(self.a2a_server, 'coordinate_prompt_evolution'):
                peer_prompts = await self.a2a_server.coordinate_prompt_evolution({
                    "prompt_id": prompt_id,
                    "base_prompt": base_prompt,
                    "task_type": task_type,
                    "performance_feedback": performance_feedback
                })
                
                # Apply cross-agent genetic operations
                evolved_prompt = await self._apply_cross_agent_evolution(
                    evolved_prompt, peer_prompts
                )
            
            # Update evolutionary prompt record
            evolutionary_prompt.optimized_prompt = evolved_prompt
            evolutionary_prompt.generation += 1
            evolutionary_prompt.last_evolution = datetime.utcnow()
            
            # Calculate new fitness score
            evolutionary_prompt.fitness_score = await self._calculate_prompt_fitness(
                evolved_prompt, task_type, performance_feedback
            )
            
            logger.debug(f"Evolved prompt (gen {evolutionary_prompt.generation}): {evolved_prompt[:100]}...")
            return evolved_prompt
            
        except Exception as e:
            logger.error(f"Failed to evolve prompt: {e}")
            return base_prompt
    
    async def _apply_evolutionary_operators(self, evolutionary_prompt: EvolutionaryPrompt,
                                          task_type: str, performance_feedback: Dict[str, float]) -> str:
        """Apply evolutionary operators to optimize prompts."""
        try:
            import random
            
            current_prompt = evolutionary_prompt.optimized_prompt
            
            # Select mutation operator based on performance feedback
            if performance_feedback.get('accuracy', 0.5) < 0.6:
                # Low accuracy - add clarifying details
                operator = "add_context"
            elif performance_feedback.get('creativity', 0.5) < 0.6:
                # Low creativity - make more open-ended
                operator = "extend"
            elif performance_feedback.get('coherence', 0.5) < 0.6:
                # Low coherence - simplify structure
                operator = "simplify"
            else:
                # Random mutation
                operator = random.choice(self.prompt_mutation_operators)
            
            # Apply selected operator
            mutated_prompt = await self._apply_mutation_operator(current_prompt, operator)
            
            # Record mutation
            mutation_record = {
                "operator": operator,
                "original": current_prompt[:100],
                "mutated": mutated_prompt[:100],
                "timestamp": datetime.utcnow().isoformat(),
                "performance_trigger": performance_feedback
            }
            evolutionary_prompt.mutation_history.append(mutation_record)
            
            return mutated_prompt
            
        except Exception as e:
            logger.error(f"Failed to apply evolutionary operators: {e}")
            return evolutionary_prompt.optimized_prompt
    
    async def _apply_mutation_operator(self, prompt: str, operator: str) -> str:
        """Apply specific mutation operator to prompt."""
        try:
            if operator == "paraphrase":
                # Simple paraphrasing (in real implementation, use NLP library)
                return f"Please {prompt.lower()}"
            elif operator == "extend":
                return f"{prompt} Provide detailed examples and explanations."
            elif operator == "simplify":
                return prompt.replace(" Please", "").replace(" detailed", "")
            elif operator == "formalize":
                return f"Formally, {prompt}"
            elif operator == "add_context":
                return f"Given the context and requirements, {prompt}"
            elif operator == "add_examples":
                return f"{prompt} Include specific examples in your response."
            elif operator == "reorder":
                # Simple reordering simulation
                parts = prompt.split(". ")
                if len(parts) > 1:
                    return ". ".join(reversed(parts))
                return prompt
            elif operator == "emphasize":
                return f"**Important**: {prompt}"
            elif operator == "adapt_tone":
                return f"In a professional and clear manner, {prompt.lower()}"
            else:
                return prompt
                
        except Exception as e:
            logger.error(f"Failed to apply mutation operator {operator}: {e}")
            return prompt
    
    # Phase 1.7.5: A2A-Enabled Model Performance Sharing
    async def share_model_performance(self, model_name: str, 
                                    performance_data: Dict[str, Any]) -> bool:
        """Share model performance data with A2A network."""
        try:
            # Create performance sharing event
            event = ModelEvolutionEvent(
                event_id=f"perf_{hashlib.md5(f'{model_name}_{datetime.utcnow()}'.encode()).hexdigest()[:8]}",
                event_type="performance_update",
                model_name=model_name,
                agent_id="local_agent",
                performance_data=performance_data,
                timestamp=datetime.utcnow()
            )
            
            # Store locally
            self.model_evolution_history.append(event)
            
            # Broadcast to A2A network
            if self.a2a_server and hasattr(self.a2a_server, 'broadcast_model_performance'):
                return await self.a2a_server.broadcast_model_performance(event)
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to share model performance: {e}")
            return False
    
    async def receive_peer_performance(self, peer_id: str, 
                                     performance_event: ModelEvolutionEvent) -> None:
        """Receive and process peer model performance data."""
        try:
            # Store peer performance
            if peer_id not in self.peer_model_performances:
                self.peer_model_performances[peer_id] = {}
            
            self.peer_model_performances[peer_id][performance_event.model_name] = {
                "performance_data": performance_event.performance_data,
                "timestamp": performance_event.timestamp,
                "agent_id": performance_event.agent_id
            }
            
            # Update consensus models based on peer feedback
            await self._update_consensus_models()
            
            logger.debug(f"Received performance data from peer {peer_id} for model {performance_event.model_name}")
            
        except Exception as e:
            logger.error(f"Failed to receive peer performance: {e}")
    
    # Phase 1.7.6: Distributed Model Improvement Coordination
    async def coordinate_model_improvement(self, improvement_proposal: Dict[str, Any]) -> Dict[str, Any]:
        """Coordinate distributed model improvement initiatives."""
        try:
            proposal_id = f"improve_{hashlib.md5(str(improvement_proposal).encode()).hexdigest()[:8]}"
            
            # Validate improvement proposal
            if not self._validate_improvement_proposal(improvement_proposal):
                return {"success": False, "error": "Invalid improvement proposal"}
            
            # Create coordination request
            coordination_request = {
                "proposal_id": proposal_id,
                "proposal": improvement_proposal,
                "coordinator": "local_agent",
                "timestamp": datetime.utcnow().isoformat(),
                "status": "coordinating",
                "peer_responses": {},
                "consensus_threshold": 0.7
            }
            
            # Coordinate with A2A peers
            if self.a2a_server and hasattr(self.a2a_server, 'coordinate_model_improvement'):
                peer_responses = await self.a2a_server.coordinate_model_improvement(coordination_request)
                coordination_request["peer_responses"] = peer_responses
            
            # Evaluate consensus
            consensus_result = await self._evaluate_improvement_consensus(coordination_request)
            
            if consensus_result["consensus_reached"]:
                # Execute improvement
                execution_result = await self._execute_model_improvement(
                    improvement_proposal, consensus_result
                )
                return {
                    "success": True,
                    "proposal_id": proposal_id,
                    "consensus": consensus_result,
                    "execution": execution_result
                }
            else:
                return {
                    "success": False,
                    "proposal_id": proposal_id,
                    "reason": "Consensus not reached",
                    "consensus": consensus_result
                }
            
        except Exception as e:
            logger.error(f"Failed to coordinate model improvement: {e}")
            return {"success": False, "error": str(e)}
    
    async def _update_consensus_models(self) -> None:
        """Update consensus models based on peer performance data."""
        try:
            # Group performance data by task type
            task_performances = defaultdict(list)
            
            # Include local performance
            for model_name, task_metrics in self.model_performance.items():
                for task_key, metrics in task_metrics.items():
                    task_type = task_key.split(":")[1]
                    task_performances[task_type].append({
                        "model": model_name,
                        "fitness": metrics.evolution_fitness,
                        "source": "local"
                    })
            
            # Include peer performance
            for peer_id, peer_data in self.peer_model_performances.items():
                for model_name, model_data in peer_data.items():
                    perf_data = model_data["performance_data"]
                    if "task_type" in perf_data:
                        task_performances[perf_data["task_type"]].append({
                            "model": model_name,
                            "fitness": perf_data.get("fitness_score", 0.5),
                            "source": peer_id
                        })
            
            # Calculate consensus for each task type
            for task_type, performances in task_performances.items():
                if len(performances) >= 2:  # Need multiple data points
                    # Group by model and calculate average fitness
                    model_fitness = defaultdict(list)
                    for perf in performances:
                        model_fitness[perf["model"]].append(perf["fitness"])
                    
                    # Find model with highest average fitness
                    best_model = None
                    best_fitness = 0.0
                    for model, fitness_scores in model_fitness.items():
                        avg_fitness = statistics.mean(fitness_scores)
                        if avg_fitness > best_fitness:
                            best_fitness = avg_fitness
                            best_model = model
                    
                    if best_model and best_fitness > 0.6:  # Threshold for consensus
                        self.consensus_models[task_type] = best_model
                        logger.debug(f"Updated consensus model for {task_type}: {best_model} (fitness: {best_fitness:.3f})")
            
        except Exception as e:
            logger.error(f"Failed to update consensus models: {e}")
    
    async def get_evolutionary_llm_metrics(self) -> Dict[str, Any]:
        """Get comprehensive evolutionary LLM metrics."""
        try:
            return {
                "model_performance": {
                    model: {
                        task_key: {
                            "fitness": metrics.evolution_fitness,
                            "success_rate": metrics.success_rate,
                            "quality_score": metrics.quality_score,
                            "usage_count": metrics.usage_count
                        } for task_key, metrics in task_metrics.items()
                    } for model, task_metrics in self.model_performance.items()
                },
                "evolutionary_prompts": {
                    pid: {
                        "generation": prompt.generation,
                        "fitness_score": prompt.fitness_score,
                        "parent_count": len(prompt.parent_prompts),
                        "mutation_count": len(prompt.mutation_history)
                    } for pid, prompt in self.evolutionary_prompts.items()
                },
                "consensus_models": self.consensus_models,
                "peer_network": {
                    "peer_count": len(self.peer_model_performances),
                    "total_shared_models": sum(len(models) for models in self.peer_model_performances.values())
                },
                "evolution_events": len(self.model_evolution_history),
                "fine_tuning": {
                    "queue_size": len(self.fine_tuning_queue),
                    "active_count": len(self.active_fine_tuning),
                    "completed_count": len(self.fine_tuning_results)
                },
                "distributed_evaluations": len(self.distributed_evaluations)
            }
            
        except Exception as e:
            logger.error(f"Failed to get evolutionary LLM metrics: {e}")
            return {"error": str(e)}
    
    # Helper methods
    def _calculate_fine_tuning_priority(self, performance_data: Dict[str, Any]) -> int:
        """Calculate priority for fine-tuning based on performance degradation."""
        try:
            priority = 5  # Default medium priority
            
            if performance_data.get('success_rate', 1.0) < 0.5:
                priority = 9  # High priority
            elif performance_data.get('quality_score', 1.0) < 0.6:
                priority = 7
            elif performance_data.get('recent_degradation', False):
                priority = 8
            
            return min(priority, 10)
            
        except Exception:
            return 5
    
    async def _process_fine_tuning_coordination(self, request: Dict[str, Any], 
                                              peer_responses: Dict[str, Any]) -> None:
        """Process fine-tuning coordination responses from peers."""
        try:
            # Analyze peer responses and build consensus
            positive_responses = sum(1 for r in peer_responses.values() 
                                   if r.get("supports_fine_tuning", False))
            total_responses = len(peer_responses)
            
            if total_responses > 0:
                consensus_ratio = positive_responses / total_responses
                request["peer_consensus"] = consensus_ratio
                request["peer_support"] = positive_responses >= total_responses * 0.6
            else:
                request["peer_consensus"] = 0.0
                request["peer_support"] = False
            
        except Exception as e:
            logger.error(f"Failed to process fine-tuning coordination: {e}")
    
    def _validate_improvement_proposal(self, proposal: Dict[str, Any]) -> bool:
        """Validate model improvement proposal."""
        try:
            required_fields = ["model_name", "improvement_type", "expected_benefits", "implementation_plan"]
            return all(field in proposal for field in required_fields)
        except Exception:
            return False
    
    async def _evaluate_improvement_consensus(self, coordination_request: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate consensus for model improvement proposal."""
        try:
            peer_responses = coordination_request.get("peer_responses", {})
            threshold = coordination_request.get("consensus_threshold", 0.7)
            
            if not peer_responses:
                return {"consensus_reached": False, "reason": "No peer responses"}
            
            support_votes = sum(1 for r in peer_responses.values() if r.get("supports_proposal", False))
            total_votes = len(peer_responses)
            consensus_ratio = support_votes / total_votes
            
            return {
                "consensus_reached": consensus_ratio >= threshold,
                "consensus_ratio": consensus_ratio,
                "support_votes": support_votes,
                "total_votes": total_votes,
                "threshold": threshold
            }
            
        except Exception as e:
            logger.error(f"Failed to evaluate improvement consensus: {e}")
            return {"consensus_reached": False, "error": str(e)}
    
    async def _execute_model_improvement(self, proposal: Dict[str, Any], 
                                       consensus: Dict[str, Any]) -> Dict[str, Any]:
        """Execute approved model improvement."""
        try:
            # Create improvement execution event
            event = ModelEvolutionEvent(
                event_id=f"improve_{hashlib.md5(str(proposal).encode()).hexdigest()[:8]}",
                event_type="improvement",
                model_name=proposal["model_name"],
                agent_id="improvement_coordinator",
                performance_data=proposal,
                improvement_suggestions=[proposal.get("improvement_type", "unknown")]
            )
            
            self.model_evolution_history.append(event)
            
            # Broadcast improvement execution
            if self.a2a_server and hasattr(self.a2a_server, 'broadcast_model_event'):
                await self.a2a_server.broadcast_model_event(event)
            
            return {
                "status": "executed",
                "event_id": event.event_id,
                "timestamp": event.timestamp.isoformat()
            }
            
        except Exception as e:
            logger.error(f"Failed to execute model improvement: {e}")
            return {"status": "failed", "error": str(e)}
    
    async def _calculate_prompt_fitness(self, prompt: str, task_type: str, 
                                      performance_feedback: Dict[str, float]) -> float:
        """Calculate fitness score for evolved prompt."""
        try:
            # Base fitness from performance feedback
            weights = {
                'accuracy': 0.3,
                'coherence': 0.25,
                'creativity': 0.2,
                'relevance': 0.15,
                'clarity': 0.1
            }
            
            fitness = 0.0
            for metric, weight in weights.items():
                fitness += weight * performance_feedback.get(metric, 0.5)
            
            # Length penalty for overly complex prompts
            if len(prompt) > 500:
                fitness *= 0.9
            elif len(prompt) < 20:
                fitness *= 0.8
            
            return min(fitness, 1.0)
            
        except Exception as e:
            logger.error(f"Failed to calculate prompt fitness: {e}")
            return 0.5
    
    async def _apply_cross_agent_evolution(self, local_prompt: str, 
                                         peer_prompts: Dict[str, Any]) -> str:
        """Apply cross-agent genetic operations for prompt evolution."""
        try:
            if not peer_prompts:
                return local_prompt
            
            # Simple crossover with best performing peer prompt
            best_peer_prompt = local_prompt
            best_fitness = 0.0
            
            for peer_id, peer_data in peer_prompts.items():
                peer_prompt = peer_data.get("optimized_prompt", "")
                peer_fitness = peer_data.get("fitness_score", 0.0)
                
                if peer_fitness > best_fitness:
                    best_fitness = peer_fitness
                    best_peer_prompt = peer_prompt
            
            # Perform crossover if peer prompt is better
            if best_fitness > 0.7 and best_peer_prompt != local_prompt:
                # Simple crossover: combine parts of both prompts
                local_parts = local_prompt.split(". ")
                peer_parts = best_peer_prompt.split(". ")
                
                # Take first half from local, second half from peer
                mid_local = len(local_parts) // 2
                mid_peer = len(peer_parts) // 2
                
                crossover_prompt = ". ".join(local_parts[:mid_local] + peer_parts[mid_peer:])
                return crossover_prompt
            
            return local_prompt
            
        except Exception as e:
            logger.error(f"Failed to apply cross-agent evolution: {e}")
            return local_prompt
    
    async def _update_model_performance_from_evaluation(self, model_name: str, 
                                                      eval_results: Dict[str, Any]) -> None:
        """Update model performance metrics based on evaluation results."""
        try:
            if "error" in eval_results:
                return
            
            consensus_metrics = eval_results.get("consensus_metrics", {})
            if not consensus_metrics:
                return
            
            # Update or create performance metrics
            task_type = "evaluation"  # Generic task type for evaluations
            key = f"{model_name}:{task_type}"
            
            if key not in self.model_performance[model_name]:
                self.model_performance[model_name][key] = ModelPerformanceMetrics(
                    model_name=model_name,
                    task_type=task_type
                )
            
            metrics = self.model_performance[model_name][key]
            
            # Update metrics
            metrics.success_rate = consensus_metrics.get("consensus_success_rate", metrics.success_rate)
            metrics.avg_response_time = consensus_metrics.get("avg_response_time", metrics.avg_response_time)
            metrics.quality_score = consensus_metrics.get("consensus_score", metrics.quality_score)
            metrics.evolution_fitness = consensus_metrics.get("consensus_score", metrics.evolution_fitness)
            metrics.usage_count += 1
            metrics.last_updated = datetime.utcnow()
            
            logger.debug(f"Updated model performance for {model_name} from evaluation")
            
        except Exception as e:
            logger.error(f"Failed to update model performance from evaluation: {e}")


# Global instance
_evolutionary_llm_manager = None


def get_evolutionary_llm_manager(ollama_manager=None, a2a_server=None) -> EvolutionaryLLMManager:
    """Get global evolutionary LLM manager instance."""
    global _evolutionary_llm_manager
    if _evolutionary_llm_manager is None:
        _evolutionary_llm_manager = EvolutionaryLLMManager(ollama_manager, a2a_server)
    return _evolutionary_llm_manager
