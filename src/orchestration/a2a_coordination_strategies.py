"""
A2A Coordination Strategies

Implements various coordination patterns for multi-agent workflows using A2A protocol.
Provides sequential, parallel, hierarchical, and advanced coordination strategies.
"""

import asyncio
import logging
from datetime import datetime, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass

logger = logging.getLogger(__name__)


class CoordinationStrategy(Enum):
    """Available coordination strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    HIERARCHICAL = "hierarchical"
    PIPELINE = "pipeline"
    CONSENSUS = "consensus"
    AUCTION = "auction"
    SWARM = "swarm"


@dataclass
class CoordinationTask:
    """Task for coordination."""
    task_id: str
    description: str
    agent_id: Optional[str] = None
    dependencies: List[str] = None
    priority: int = 1
    timeout_seconds: int = 300
    metadata: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.dependencies is None:
            self.dependencies = []
        if self.metadata is None:
            self.metadata = {}


@dataclass
class CoordinationResult:
    """Result from coordination execution."""
    strategy: CoordinationStrategy
    total_tasks: int
    successful_tasks: int
    failed_tasks: int
    execution_time: float
    results: List[Dict[str, Any]]
    coordination_metadata: Dict[str, Any]
    
    @property
    def success_rate(self) -> float:
        """Calculate success rate."""
        return self.successful_tasks / self.total_tasks if self.total_tasks > 0 else 0.0


class A2ACoordinationEngine:
    """
    A2A Coordination Engine for multi-agent workflow orchestration.
    
    Implements various coordination strategies using A2A protocol for
    agent-to-agent communication and task coordination.
    """
    
    def __init__(self, a2a_manager, orchestration_manager):
        self.a2a_manager = a2a_manager
        self.orchestration_manager = orchestration_manager
        
        # Coordination state
        self.active_coordinations: Dict[str, Dict[str, Any]] = {}
        self.coordination_history: List[CoordinationResult] = []
        
        # Strategy performance tracking
        self.strategy_performance: Dict[CoordinationStrategy, Dict[str, float]] = {
            strategy: {"success_rate": 0.0, "avg_time": 0.0, "executions": 0}
            for strategy in CoordinationStrategy
        }
        
        logger.info("A2A Coordination Engine initialized")
    
    async def execute_coordination(self,
                                 coordination_id: str,
                                 strategy: CoordinationStrategy,
                                 tasks: List[CoordinationTask],
                                 coordination_metadata: Optional[Dict[str, Any]] = None) -> CoordinationResult:
        """Execute coordination using specified strategy."""
        start_time = datetime.utcnow()
        
        try:
            logger.info(f"Starting {strategy.value} coordination {coordination_id} with {len(tasks)} tasks")
            
            # Track active coordination
            self.active_coordinations[coordination_id] = {
                "strategy": strategy,
                "tasks": tasks,
                "start_time": start_time,
                "status": "running"
            }
            
            # Execute based on strategy
            if strategy == CoordinationStrategy.SEQUENTIAL:
                results = await self._execute_sequential(tasks)
            elif strategy == CoordinationStrategy.PARALLEL:
                results = await self._execute_parallel(tasks)
            elif strategy == CoordinationStrategy.HIERARCHICAL:
                results = await self._execute_hierarchical(tasks)
            elif strategy == CoordinationStrategy.PIPELINE:
                results = await self._execute_pipeline(tasks)
            elif strategy == CoordinationStrategy.CONSENSUS:
                results = await self._execute_consensus(tasks)
            elif strategy == CoordinationStrategy.AUCTION:
                results = await self._execute_auction(tasks)
            elif strategy == CoordinationStrategy.SWARM:
                results = await self._execute_swarm(tasks)
            else:
                raise ValueError(f"Unsupported coordination strategy: {strategy}")
            
            # Calculate execution time
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            
            # Create coordination result
            successful_tasks = sum(1 for r in results if r.get("status") == "success")
            failed_tasks = len(results) - successful_tasks
            
            coordination_result = CoordinationResult(
                strategy=strategy,
                total_tasks=len(tasks),
                successful_tasks=successful_tasks,
                failed_tasks=failed_tasks,
                execution_time=execution_time,
                results=results,
                coordination_metadata=coordination_metadata or {}
            )
            
            # Update performance tracking
            self._update_strategy_performance(strategy, coordination_result)
            
            # Store in history
            self.coordination_history.append(coordination_result)
            
            # Clean up active coordination
            if coordination_id in self.active_coordinations:
                del self.active_coordinations[coordination_id]
            
            logger.info(f"Completed {strategy.value} coordination {coordination_id}: "
                       f"{successful_tasks}/{len(tasks)} successful")
            
            return coordination_result
            
        except Exception as e:
            logger.error(f"Coordination {coordination_id} failed: {e}")
            
            # Clean up on failure
            if coordination_id in self.active_coordinations:
                del self.active_coordinations[coordination_id]
            
            # Return failure result
            execution_time = (datetime.utcnow() - start_time).total_seconds()
            return CoordinationResult(
                strategy=strategy,
                total_tasks=len(tasks),
                successful_tasks=0,
                failed_tasks=len(tasks),
                execution_time=execution_time,
                results=[{"status": "failed", "error": str(e)} for _ in tasks],
                coordination_metadata=coordination_metadata or {}
            )
    
    async def _execute_sequential(self, tasks: List[CoordinationTask]) -> List[Dict[str, Any]]:
        """Execute tasks sequentially, passing results between tasks."""
        results = []
        current_context = {}
        
        for i, task in enumerate(tasks):
            try:
                logger.debug(f"Executing sequential task {i+1}/{len(tasks)}: {task.task_id}")
                
                # Prepare task with context from previous tasks
                task_description = task.description
                if current_context and i > 0:
                    task_description += f"\n\nContext from previous tasks: {current_context}"
                
                # Execute task using A2A manager
                if task.agent_id:
                    # Execute on specific agent
                    a2a_results = await self.a2a_manager.coordinate_multi_agent_task(
                        task_description=task_description,
                        agent_ids=[task.agent_id],
                        coordination_strategy="sequential"
                    )
                else:
                    # Let A2A manager select appropriate agent
                    available_agents = await self._get_available_agents()
                    if available_agents:
                        a2a_results = await self.a2a_manager.coordinate_multi_agent_task(
                            task_description=task_description,
                            agent_ids=available_agents[:1],  # Use first available agent
                            coordination_strategy="sequential"
                        )
                    else:
                        raise ValueError("No available agents for task execution")
                
                if a2a_results and len(a2a_results) > 0:
                    result = {
                        "task_id": task.task_id,
                        "status": "success",
                        "result": a2a_results[0],
                        "execution_order": i + 1
                    }
                    
                    # Update context for next task
                    if hasattr(a2a_results[0], 'artifacts') and a2a_results[0].artifacts:
                        current_context[f"task_{i+1}"] = a2a_results[0].artifacts[-1].parts[0].text
                    
                else:
                    result = {
                        "task_id": task.task_id,
                        "status": "failed",
                        "error": "No results from A2A execution",
                        "execution_order": i + 1
                    }
                
                results.append(result)
                
                # If task failed and it's critical, stop execution
                if result["status"] == "failed" and task.priority > 3:
                    logger.warning(f"Critical task {task.task_id} failed, stopping sequential execution")
                    break
                
            except Exception as e:
                logger.error(f"Sequential task {task.task_id} failed: {e}")
                results.append({
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": str(e),
                    "execution_order": i + 1
                })
                
                # Stop on critical task failure
                if task.priority > 3:
                    break
        
        return results
    
    async def _execute_parallel(self, tasks: List[CoordinationTask]) -> List[Dict[str, Any]]:
        """Execute tasks in parallel."""
        logger.debug(f"Executing {len(tasks)} tasks in parallel")
        
        # Create coroutines for all tasks
        task_coroutines = []
        for task in tasks:
            coro = self._execute_single_task(task)
            task_coroutines.append(coro)
        
        # Execute all tasks concurrently
        try:
            results = await asyncio.gather(*task_coroutines, return_exceptions=True)
            
            # Process results and handle exceptions
            processed_results = []
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "task_id": tasks[i].task_id,
                        "status": "failed",
                        "error": str(result)
                    })
                else:
                    processed_results.append(result)
            
            return processed_results
            
        except Exception as e:
            logger.error(f"Parallel execution failed: {e}")
            return [
                {
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": str(e)
                }
                for task in tasks
            ]
    
    async def _execute_single_task(self, task: CoordinationTask) -> Dict[str, Any]:
        """Execute a single coordination task."""
        try:
            # Get available agents
            if task.agent_id:
                agent_ids = [task.agent_id]
            else:
                available_agents = await self._get_available_agents()
                agent_ids = available_agents[:1] if available_agents else []
            
            if not agent_ids:
                return {
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": "No available agents"
                }
            
            # Execute using A2A manager
            a2a_results = await self.a2a_manager.coordinate_multi_agent_task(
                task_description=task.description,
                agent_ids=agent_ids,
                coordination_strategy="sequential"
            )
            
            if a2a_results and len(a2a_results) > 0:
                return {
                    "task_id": task.task_id,
                    "status": "success",
                    "result": a2a_results[0],
                    "agent_id": agent_ids[0]
                }
            else:
                return {
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": "No results from A2A execution"
                }
                
        except Exception as e:
            logger.error(f"Single task execution failed for {task.task_id}: {e}")
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": str(e)
            }
    
    async def _get_available_agents(self) -> List[str]:
        """Get list of available A2A agents."""
        try:
            if self.a2a_manager:
                status = await self.a2a_manager.get_agent_status()
                agents = status.get("agents", [])
                return [agent.get("agent_id") for agent in agents if agent.get("agent_id")]
            return []
        except Exception as e:
            logger.error(f"Failed to get available agents: {e}")
            return []
    
    def _update_strategy_performance(self, strategy: CoordinationStrategy, result: CoordinationResult):
        """Update performance metrics for coordination strategy."""
        try:
            perf = self.strategy_performance[strategy]
            
            # Update running averages
            executions = perf["executions"]
            new_executions = executions + 1
            
            # Update success rate
            current_success_rate = perf["success_rate"]
            new_success_rate = ((current_success_rate * executions) + result.success_rate) / new_executions
            
            # Update average time
            current_avg_time = perf["avg_time"]
            new_avg_time = ((current_avg_time * executions) + result.execution_time) / new_executions
            
            # Store updated metrics
            perf["success_rate"] = new_success_rate
            perf["avg_time"] = new_avg_time
            perf["executions"] = new_executions
            
        except Exception as e:
            logger.error(f"Failed to update strategy performance: {e}")
    
    def get_strategy_performance(self) -> Dict[str, Dict[str, float]]:
        """Get performance metrics for all coordination strategies."""
        return {
            strategy.value: metrics 
            for strategy, metrics in self.strategy_performance.items()
        }
    
    def get_coordination_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recent coordination history."""
        recent_history = self.coordination_history[-limit:] if limit > 0 else self.coordination_history
        
        return [
            {
                "strategy": result.strategy.value,
                "total_tasks": result.total_tasks,
                "successful_tasks": result.successful_tasks,
                "success_rate": result.success_rate,
                "execution_time": result.execution_time,
                "metadata": result.coordination_metadata
            }
            for result in recent_history
        ]

    async def _execute_hierarchical(self, tasks: List[CoordinationTask]) -> List[Dict[str, Any]]:
        """Execute tasks in hierarchical coordination pattern."""
        logger.debug(f"Executing {len(tasks)} tasks in hierarchical pattern")

        try:
            # Sort tasks by priority (higher priority = coordinator)
            sorted_tasks = sorted(tasks, key=lambda t: t.priority, reverse=True)

            if not sorted_tasks:
                return []

            # First task becomes coordinator
            coordinator_task = sorted_tasks[0]
            subordinate_tasks = sorted_tasks[1:]

            results = []

            # Execute coordinator task first
            coordinator_result = await self._execute_single_task(coordinator_task)
            results.append(coordinator_result)

            # If coordinator succeeded, execute subordinate tasks
            if coordinator_result.get("status") == "success":
                # Execute subordinate tasks in parallel, using coordinator context
                subordinate_coroutines = []
                for task in subordinate_tasks:
                    # Enhance task description with coordinator context
                    enhanced_task = CoordinationTask(
                        task_id=task.task_id,
                        description=f"{task.description}\n\nCoordinator guidance: {coordinator_result.get('result', '')}",
                        agent_id=task.agent_id,
                        dependencies=task.dependencies,
                        priority=task.priority,
                        timeout_seconds=task.timeout_seconds,
                        metadata=task.metadata
                    )
                    subordinate_coroutines.append(self._execute_single_task(enhanced_task))

                if subordinate_coroutines:
                    subordinate_results = await asyncio.gather(*subordinate_coroutines, return_exceptions=True)

                    # Process subordinate results
                    for i, result in enumerate(subordinate_results):
                        if isinstance(result, Exception):
                            results.append({
                                "task_id": subordinate_tasks[i].task_id,
                                "status": "failed",
                                "error": str(result),
                                "role": "subordinate"
                            })
                        else:
                            result["role"] = "subordinate"
                            results.append(result)
            else:
                # Coordinator failed, mark all subordinate tasks as failed
                for task in subordinate_tasks:
                    results.append({
                        "task_id": task.task_id,
                        "status": "failed",
                        "error": "Coordinator task failed",
                        "role": "subordinate"
                    })

            return results

        except Exception as e:
            logger.error(f"Hierarchical execution failed: {e}")
            return [
                {
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": str(e)
                }
                for task in tasks
            ]

    async def _execute_pipeline(self, tasks: List[CoordinationTask]) -> List[Dict[str, Any]]:
        """Execute tasks in pipeline pattern with data flow."""
        logger.debug(f"Executing {len(tasks)} tasks in pipeline pattern")

        results = []
        pipeline_data = {}

        # Sort tasks by dependencies to create pipeline order
        ordered_tasks = self._resolve_task_dependencies(tasks)

        for i, task in enumerate(ordered_tasks):
            try:
                # Prepare task with pipeline data
                task_description = task.description
                if pipeline_data:
                    task_description += f"\n\nPipeline input data: {pipeline_data}"

                # Execute task
                result = await self._execute_single_task(CoordinationTask(
                    task_id=task.task_id,
                    description=task_description,
                    agent_id=task.agent_id,
                    dependencies=task.dependencies,
                    priority=task.priority,
                    timeout_seconds=task.timeout_seconds,
                    metadata=task.metadata
                ))

                # Extract output data for next stage
                if result.get("status") == "success":
                    pipeline_data[f"stage_{i+1}"] = result.get("result", {})

                result["pipeline_stage"] = i + 1
                results.append(result)

                # If task failed, break pipeline
                if result.get("status") == "failed":
                    logger.warning(f"Pipeline broken at stage {i+1}: {task.task_id}")
                    break

            except Exception as e:
                logger.error(f"Pipeline task {task.task_id} failed: {e}")
                results.append({
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": str(e),
                    "pipeline_stage": i + 1
                })
                break

        return results

    async def _execute_consensus(self, tasks: List[CoordinationTask]) -> List[Dict[str, Any]]:
        """Execute tasks using consensus coordination."""
        logger.debug(f"Executing {len(tasks)} tasks with consensus coordination")

        if len(tasks) < 2:
            # Need at least 2 tasks for consensus
            return await self._execute_parallel(tasks)

        try:
            # Execute all tasks in parallel
            parallel_results = await self._execute_parallel(tasks)

            # Analyze results for consensus
            successful_results = [r for r in parallel_results if r.get("status") == "success"]

            if len(successful_results) >= len(tasks) // 2 + 1:  # Majority consensus
                consensus_result = {
                    "consensus_achieved": True,
                    "consensus_type": "majority",
                    "participating_tasks": len(successful_results),
                    "total_tasks": len(tasks)
                }
            else:
                consensus_result = {
                    "consensus_achieved": False,
                    "consensus_type": "failed",
                    "participating_tasks": len(successful_results),
                    "total_tasks": len(tasks)
                }

            # Add consensus metadata to all results
            for result in parallel_results:
                result["consensus_metadata"] = consensus_result

            return parallel_results

        except Exception as e:
            logger.error(f"Consensus execution failed: {e}")
            return [
                {
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": str(e)
                }
                for task in tasks
            ]

    async def _execute_auction(self, tasks: List[CoordinationTask]) -> List[Dict[str, Any]]:
        """Execute tasks using auction-based coordination."""
        logger.debug(f"Executing {len(tasks)} tasks with auction coordination")

        try:
            available_agents = await self._get_available_agents()

            if not available_agents:
                return [
                    {
                        "task_id": task.task_id,
                        "status": "failed",
                        "error": "No available agents for auction"
                    }
                    for task in tasks
                ]

            results = []

            for task in tasks:
                # Conduct real A2A auction process
                best_agent = await self._conduct_task_auction(task, available_agents)

                if best_agent:
                    # Execute task with winning agent
                    enhanced_task = CoordinationTask(
                        task_id=task.task_id,
                        description=task.description,
                        agent_id=best_agent,
                        dependencies=task.dependencies,
                        priority=task.priority,
                        timeout_seconds=task.timeout_seconds,
                        metadata=task.metadata
                    )

                    result = await self._execute_single_task(enhanced_task)
                    result["auction_winner"] = best_agent
                    result["coordination_type"] = "auction"

                else:
                    result = {
                        "task_id": task.task_id,
                        "status": "failed",
                        "error": "No agent won the auction",
                        "coordination_type": "auction"
                    }

                results.append(result)

            return results

        except Exception as e:
            logger.error(f"Auction execution failed: {e}")
            return [
                {
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": str(e)
                }
                for task in tasks
            ]

    async def _execute_swarm(self, tasks: List[CoordinationTask]) -> List[Dict[str, Any]]:
        """Execute tasks using swarm coordination."""
        logger.debug(f"Executing {len(tasks)} tasks with swarm coordination")

        try:
            # Swarm coordination: distribute tasks dynamically based on agent availability
            available_agents = await self._get_available_agents()

            if not available_agents:
                return [
                    {
                        "task_id": task.task_id,
                        "status": "failed",
                        "error": "No available agents for swarm"
                    }
                    for task in tasks
                ]

            # Create task-agent assignments using swarm logic
            task_assignments = self._create_swarm_assignments(tasks, available_agents)

            # Execute tasks with swarm coordination
            swarm_coroutines = []
            for task, agent_id in task_assignments:
                enhanced_task = CoordinationTask(
                    task_id=task.task_id,
                    description=task.description,
                    agent_id=agent_id,
                    dependencies=task.dependencies,
                    priority=task.priority,
                    timeout_seconds=task.timeout_seconds,
                    metadata=task.metadata
                )
                swarm_coroutines.append(self._execute_single_task(enhanced_task))

            # Execute all swarm tasks
            swarm_results = await asyncio.gather(*swarm_coroutines, return_exceptions=True)

            # Process swarm results
            processed_results = []
            for i, result in enumerate(swarm_results):
                if isinstance(result, Exception):
                    processed_results.append({
                        "task_id": tasks[i].task_id,
                        "status": "failed",
                        "error": str(result),
                        "coordination_type": "swarm"
                    })
                else:
                    result["coordination_type"] = "swarm"
                    processed_results.append(result)

            return processed_results

        except Exception as e:
            logger.error(f"Swarm execution failed: {e}")
            return [
                {
                    "task_id": task.task_id,
                    "status": "failed",
                    "error": str(e)
                }
                for task in tasks
            ]

    def _resolve_task_dependencies(self, tasks: List[CoordinationTask]) -> List[CoordinationTask]:
        """Resolve task dependencies to create execution order."""
        # Simple topological sort for task dependencies
        ordered_tasks = []
        remaining_tasks = tasks.copy()

        while remaining_tasks:
            # Find tasks with no unresolved dependencies
            ready_tasks = []
            for task in remaining_tasks:
                if not task.dependencies or all(
                    dep in [t.task_id for t in ordered_tasks]
                    for dep in task.dependencies
                ):
                    ready_tasks.append(task)

            if not ready_tasks:
                # Circular dependency or unresolvable - add remaining tasks
                ordered_tasks.extend(remaining_tasks)
                break

            # Add ready tasks to ordered list
            ordered_tasks.extend(ready_tasks)

            # Remove ready tasks from remaining
            for task in ready_tasks:
                remaining_tasks.remove(task)

        return ordered_tasks

    async def _conduct_task_auction(self, task: CoordinationTask, available_agents: List[str]) -> Optional[str]:
        """Conduct real A2A auction for task assignment with bidding and negotiation."""
        try:
            if not available_agents or not self.a2a_manager:
                return None

            # Create auction announcement
            auction_announcement = {
                "type": "task_auction",
                "task_id": task.task_id,
                "task_description": task.description,
                "task_type": "coordination_task",
                "priority": task.priority,
                "timeout_seconds": task.timeout_seconds,
                "auction_deadline": (datetime.utcnow() + timedelta(seconds=30)).isoformat(),
                "bidding_criteria": {
                    "capability_match": 0.4,
                    "availability": 0.3,
                    "performance_history": 0.3
                }
            }

            # Collect bids from available agents
            bids = []
            for agent_id in available_agents:
                try:
                    # Send auction announcement to agent
                    bid_response = await self.a2a_manager.coordinate_multi_agent_task(
                        task_description=f"Auction bid request: {auction_announcement}",
                        agent_ids=[agent_id],
                        coordination_strategy="sequential"
                    )

                    if bid_response and len(bid_response) > 0:
                        # Parse bid from response
                        bid = self._parse_auction_bid(agent_id, bid_response[0], auction_announcement)
                        if bid:
                            bids.append(bid)

                except Exception as e:
                    logger.warning(f"Failed to get bid from agent {agent_id}: {e}")
                    continue

            # Evaluate bids and select winner
            if bids:
                winning_bid = self._evaluate_auction_bids(bids, auction_announcement)
                if winning_bid:
                    # Notify winner and other bidders
                    await self._notify_auction_results(winning_bid, bids, auction_announcement)
                    return winning_bid["agent_id"]

            # Fallback to first available agent if auction fails
            return available_agents[0] if available_agents else None

        except Exception as e:
            logger.error(f"Failed to conduct task auction: {e}")
            return available_agents[0] if available_agents else None

    def _parse_auction_bid(self, agent_id: str, bid_response: Any, auction_announcement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Parse auction bid from agent response."""
        try:
            # Extract bid information from response
            if hasattr(bid_response, 'artifacts') and bid_response.artifacts:
                response_text = bid_response.artifacts[-1].parts[0].text if bid_response.artifacts[-1].parts else ""

                # Parse bid components (simplified parsing)
                bid = {
                    "agent_id": agent_id,
                    "task_id": auction_announcement["task_id"],
                    "bid_amount": self._extract_bid_amount(response_text),
                    "capability_score": self._extract_capability_score(response_text),
                    "availability_score": self._extract_availability_score(response_text),
                    "estimated_completion_time": self._extract_completion_time(response_text),
                    "confidence_level": self._extract_confidence_level(response_text),
                    "bid_timestamp": datetime.utcnow().isoformat()
                }

                return bid

            return None

        except Exception as e:
            logger.error(f"Failed to parse auction bid from {agent_id}: {e}")
            return None

    def _extract_bid_amount(self, response_text: str) -> float:
        """Extract bid amount from response text."""
        try:
            import re
            # Look for numerical values that could be bid amounts
            numbers = re.findall(r'(\d+(?:\.\d+)?)', response_text)
            if numbers:
                return float(numbers[0])
            return 1.0  # Default bid
        except:
            return 1.0

    def _extract_capability_score(self, response_text: str) -> float:
        """Extract capability score from response text."""
        try:
            # Look for capability indicators
            if "expert" in response_text.lower() or "excellent" in response_text.lower():
                return 0.9
            elif "good" in response_text.lower() or "capable" in response_text.lower():
                return 0.7
            elif "basic" in response_text.lower() or "limited" in response_text.lower():
                return 0.4
            return 0.6  # Default
        except:
            return 0.6

    def _extract_availability_score(self, response_text: str) -> float:
        """Extract availability score from response text."""
        try:
            if "immediately" in response_text.lower() or "available" in response_text.lower():
                return 0.9
            elif "busy" in response_text.lower() or "occupied" in response_text.lower():
                return 0.3
            return 0.6  # Default
        except:
            return 0.6

    def _extract_completion_time(self, response_text: str) -> int:
        """Extract estimated completion time from response text."""
        try:
            import re
            # Look for time estimates
            time_matches = re.findall(r'(\d+)\s*(minute|hour|second)', response_text.lower())
            if time_matches:
                value, unit = time_matches[0]
                value = int(value)
                if unit.startswith('hour'):
                    return value * 3600
                elif unit.startswith('minute'):
                    return value * 60
                else:
                    return value
            return 300  # Default 5 minutes
        except:
            return 300

    def _extract_confidence_level(self, response_text: str) -> float:
        """Extract confidence level from response text."""
        try:
            if "confident" in response_text.lower() or "certain" in response_text.lower():
                return 0.8
            elif "uncertain" in response_text.lower() or "unsure" in response_text.lower():
                return 0.3
            return 0.6  # Default
        except:
            return 0.6

    def _evaluate_auction_bids(self, bids: List[Dict[str, Any]], auction_announcement: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Evaluate auction bids and select winner."""
        try:
            if not bids:
                return None

            criteria = auction_announcement.get("bidding_criteria", {
                "capability_match": 0.4,
                "availability": 0.3,
                "performance_history": 0.3
            })

            # Score each bid
            scored_bids = []
            for bid in bids:
                score = (
                    bid.get("capability_score", 0.5) * criteria.get("capability_match", 0.4) +
                    bid.get("availability_score", 0.5) * criteria.get("availability", 0.3) +
                    bid.get("confidence_level", 0.5) * criteria.get("performance_history", 0.3)
                )

                # Adjust score based on completion time (faster is better)
                time_factor = max(0.1, 1.0 - (bid.get("estimated_completion_time", 300) / 1800))  # Normalize to 30 min max
                score *= time_factor

                scored_bids.append((score, bid))

            # Select highest scoring bid
            scored_bids.sort(key=lambda x: x[0], reverse=True)
            return scored_bids[0][1] if scored_bids else None

        except Exception as e:
            logger.error(f"Failed to evaluate auction bids: {e}")
            return bids[0] if bids else None

    async def _notify_auction_results(self, winning_bid: Dict[str, Any], all_bids: List[Dict[str, Any]],
                                    auction_announcement: Dict[str, Any]):
        """Notify all bidders of auction results."""
        try:
            winner_id = winning_bid["agent_id"]

            # Notify winner
            winner_notification = {
                "type": "auction_winner",
                "task_id": auction_announcement["task_id"],
                "winning_bid": winning_bid,
                "auction_announcement": auction_announcement
            }

            await self.a2a_manager.coordinate_multi_agent_task(
                task_description=f"Auction winner notification: {winner_notification}",
                agent_ids=[winner_id],
                coordination_strategy="sequential"
            )

            # Notify other bidders
            for bid in all_bids:
                if bid["agent_id"] != winner_id:
                    loser_notification = {
                        "type": "auction_result",
                        "task_id": auction_announcement["task_id"],
                        "won": False,
                        "winning_agent": winner_id,
                        "your_bid": bid
                    }

                    await self.a2a_manager.coordinate_multi_agent_task(
                        task_description=f"Auction result notification: {loser_notification}",
                        agent_ids=[bid["agent_id"]],
                        coordination_strategy="sequential"
                    )

        except Exception as e:
            logger.error(f"Failed to notify auction results: {e}")

    def _create_swarm_assignments(self, tasks: List[CoordinationTask], available_agents: List[str]) -> List[Tuple[CoordinationTask, str]]:
        """Create task-agent assignments for swarm coordination."""
        assignments = []
        agent_index = 0

        for task in tasks:
            if task.agent_id and task.agent_id in available_agents:
                # Use specified agent
                assignments.append((task, task.agent_id))
            else:
                # Assign to next available agent (round-robin)
                agent_id = available_agents[agent_index % len(available_agents)]
                assignments.append((task, agent_id))
                agent_index += 1

        return assignments
