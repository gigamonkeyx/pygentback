"""
Darwin Gödel Machine (DGM) Engine
Self-improving agent engine implementing evolutionary programming principles.
"""

import asyncio
import logging
import uuid
from typing import Dict, Any, Optional
from datetime import datetime, timedelta

from ..models import (
    ImprovementCandidate, ImprovementStatus, ImprovementType,
    PerformanceMetric, DGMState, EvolutionParameters, DGMArchiveEntry
)
from .code_generator import CodeGenerator
from .validator import EmpiricalValidator
from .archive import DGMArchive
from .safety_monitor import SafetyMonitor

logger = logging.getLogger(__name__)


class DGMEngine:
    """Darwin Gödel Machine self-improvement engine"""
    
    def __init__(self, agent_id: str, config: Dict[str, Any]):
        self.agent_id = agent_id
        self.config = config
        
        # Core components
        self.code_generator = CodeGenerator(config.get("code_generation", {}))
        self.validator = EmpiricalValidator(config.get("validation", {}))
        self.archive = DGMArchive(config.get("archive_path", f"./data/dgm/{agent_id}"))
        self.safety_monitor = SafetyMonitor(config.get("safety", {}))
        
        # State
        self.state = DGMState(
            agent_id=agent_id,
            current_performance={},
            improvement_history=[],
            active_experiments=[],
            best_configuration=config.get("initial_configuration", {})
        )
        
        # Evolution parameters
        self.evolution_params = EvolutionParameters(**config.get("evolution", {}))
        
        # Configuration
        self.max_concurrent_improvements = config.get("max_concurrent_improvements", 3)
        self.improvement_interval = timedelta(
            minutes=config.get("improvement_interval_minutes", 30)
        )
        self.safety_threshold = config.get("safety_threshold", 0.8)
        
        # Background task
        self._improvement_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self):
        """Start the DGM improvement loop"""
        if self._running:
            return
        
        self._running = True
        logger.info(f"Starting DGM engine for agent {self.agent_id}")
        
        # Initialize baseline performance
        await self._establish_baseline()
        
        # Start improvement loop
        self._improvement_task = asyncio.create_task(self._improvement_loop())
    
    async def stop(self):
        """Stop the DGM improvement loop"""
        self._running = False
        
        if self._improvement_task:
            self._improvement_task.cancel()
            try:
                await self._improvement_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Stopped DGM engine for agent {self.agent_id}")
    
    async def attempt_improvement(self, context: Optional[Dict[str, Any]] = None) -> str:
        """Manually trigger improvement attempt"""
        if len(self.state.active_experiments) >= self.max_concurrent_improvements:
            raise ValueError("Maximum concurrent improvements reached")
        
        # Generate improvement candidate
        candidate = await self._generate_improvement_candidate(context or {})
        candidate_id = str(uuid.uuid4())
        candidate.id = candidate_id
        
        # Safety evaluation
        safety_evaluation = await self.safety_monitor.evaluate_candidate_safety(candidate)
        candidate.risk_level = 1.0 - safety_evaluation["safety_score"]
        
        if not safety_evaluation["safe"]:
            candidate.status = ImprovementStatus.REJECTED
            logger.warning(f"Candidate {candidate_id} rejected due to safety concerns")
            
            # Archive the rejected candidate
            archive_entry = DGMArchiveEntry(
                id=candidate_id,
                agent_id=self.agent_id,
                improvement_candidate=candidate
            )
            await self.archive.store_entry(archive_entry)
            return candidate_id
        
        # Store and start validation
        self.state.active_experiments.append(candidate_id)
        self.state.improvement_history.append(candidate)
        asyncio.create_task(self._validate_and_apply(candidate))
        
        logger.info(f"Started improvement attempt {candidate_id}")
        return candidate_id
    
    async def get_improvement_status(self, candidate_id: str) -> Optional[ImprovementCandidate]:
        """Get status of improvement candidate"""
        for candidate in self.state.improvement_history:
            if candidate.id == candidate_id:
                return candidate
        return None
    
    async def get_current_state(self) -> DGMState:
        """Get current DGM state"""
        return self.state
    
    async def _improvement_loop(self):
        """Main improvement loop"""
        while self._running:
            try:
                # Check if it's time for automatic improvement
                if await self._should_attempt_improvement():
                    await self.attempt_improvement()
                
                # Wait before next check
                await asyncio.sleep(60)  # Check every minute
                
            except Exception as e:
                logger.error(f"Error in improvement loop: {e}")
                await asyncio.sleep(60)
    
    async def _should_attempt_improvement(self) -> bool:
        """Check if automatic improvement should be attempted"""
        # Don't exceed concurrent limit
        if len(self.state.active_experiments) >= self.max_concurrent_improvements:
            return False
        
        # Check time-based criteria
        if self.state.last_improvement:
            time_since_last = datetime.utcnow() - self.state.last_improvement
            if time_since_last < self.improvement_interval:
                return False
        
        return True
    
    async def _generate_improvement_candidate(self, context: Dict[str, Any]) -> ImprovementCandidate:
        """Generate an improvement candidate"""
        # Use code generator to create improvements
        improvement_candidate = await self.code_generator.generate_improvement(
            agent_id=self.agent_id,
            context=context,
            baseline_performance=list(self.state.current_performance.values())
        )
        
        return improvement_candidate
    
    async def _validate_and_apply(self, candidate: ImprovementCandidate):
        """Validate and potentially apply an improvement candidate"""
        try:
            candidate.status = ImprovementStatus.TESTING
            
            # Run validation
            validation_result = await self.validator.validate_candidate(candidate)
            
            # Evaluate safety of validation results
            safety_evaluation = await self.safety_monitor.evaluate_candidate_safety(candidate)
            
            # Create archive entry
            archive_entry = DGMArchiveEntry(
                id=candidate.id,
                agent_id=self.agent_id,
                improvement_candidate=candidate,
                validation_result=validation_result
            )
            
            if validation_result.success and safety_evaluation["safe"]:
                candidate.status = ImprovementStatus.VALIDATED
                
                # Apply the improvement
                success = await self._apply_improvement(candidate)
                if success:
                    candidate.status = ImprovementStatus.APPLIED
                    archive_entry.applied = True
                    archive_entry.application_timestamp = datetime.utcnow()
                    
                    # Update state
                    self.state.last_improvement = datetime.utcnow()
                    self.state.generation += 1
                    
                    logger.info(f"Successfully applied improvement {candidate.id}")
                else:
                    candidate.status = ImprovementStatus.FAILED
                    logger.error(f"Failed to apply improvement {candidate.id}")
            else:
                candidate.status = ImprovementStatus.REJECTED
                logger.warning(f"Improvement {candidate.id} rejected after validation")
            
            # Store in archive
            await self.archive.store_entry(archive_entry)
            
        except Exception as e:
            logger.error(f"Error in validation and application for {candidate.id}: {e}")
            candidate.status = ImprovementStatus.FAILED
        finally:
            # Remove from active experiments
            if candidate.id in self.state.active_experiments:
                self.state.active_experiments.remove(candidate.id)
    
    async def _apply_improvement(self, candidate: ImprovementCandidate) -> bool:
        """Apply validated improvements to the system"""
        try:
            # In a real implementation, this would apply code changes
            # For now, simulate the application
            logger.info(f"Applying improvement {candidate.id}: {candidate.description}")
            
            # Simulate application time
            await asyncio.sleep(0.1)
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying improvement {candidate.id}: {e}")
            return False
    
    async def _establish_baseline(self):
        """Establish baseline performance metrics"""
        try:
            # Simulate baseline measurement
            baseline_metrics = {
                "response_time": PerformanceMetric(
                    name="response_time",
                    value=0.5,
                    unit="seconds"
                ),
                "accuracy": PerformanceMetric(
                    name="accuracy", 
                    value=0.85,
                    unit="percentage"
                )
            }
            
            self.state.current_performance = baseline_metrics
            logger.info(f"Established baseline performance for agent {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Error establishing baseline: {e}")
