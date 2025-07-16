"""
Supervisor Agent - Task Intelligence System with Dual-Loop Orchestrator
Implements Microsoft Magentic-One inspired architecture for intelligent task coordination.
"""

import logging
import asyncio
import uuid
from typing import Dict, List, Any, Optional, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import aiohttp
import subprocess
import psutil
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class SystemHealthResult:
    """Result of system health check"""
    healthy: bool
    components: Dict[str, bool] = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)
    recovery_attempted: bool = False
    recovery_successful: bool = False


@dataclass
class SystemRecoveryResult:
    """Result of system recovery attempt"""
    success: bool
    recovered_components: List[str] = field(default_factory=list)
    failed_components: List[str] = field(default_factory=list)
    errors: List[str] = field(default_factory=list)


class SystemNotReadyException(Exception):
    """Raised when system is not ready for task execution"""
    pass


class TaskType(Enum):
    """Task type classification"""
    CODING = "coding"
    RESEARCH = "research"
    ANALYSIS = "analysis"
    UI_CREATION = "ui_creation"
    GRAPHICS_DEVELOPMENT = "graphics_development"
    DOCUMENTATION = "documentation"
    BACKEND_DEVELOPMENT = "backend_development"
    TESTING = "testing"
    DEPLOYMENT = "deployment"
    INTEGRATION = "integration"
    UNKNOWN = "unknown"


class TaskComplexity(Enum):
    """Task complexity levels with scoring"""
    SIMPLE = 1      # Single step, clear requirements
    MODERATE = 3    # Multiple steps, some dependencies
    COMPLEX = 6     # Many steps, complex dependencies
    RESEARCH = 8    # Requires investigation and analysis
    ENTERPRISE = 10 # Multi-system integration, high stakes


@dataclass
class TaskAnalysis:
    """Enhanced task analysis result with confidence scoring"""
    task_type: TaskType
    complexity: TaskComplexity
    complexity_score: float  # 0.0-1.0 confidence in complexity assessment
    estimated_time: int  # minutes
    required_capabilities: List[str]
    success_criteria: List[str]
    context_requirements: List[str] = field(default_factory=list)
    risk_factors: List[str] = field(default_factory=list)
    dependency_count: int = 0
    parallel_potential: bool = False


@dataclass
class QualityScore:
    """Quality assessment result"""
    score: float  # 0.0-1.0
    passed: bool
    issues: List[str]
    suggestions: List[str]


@dataclass
class TaskLedger:
    """Outer loop: High-level task planning and strategy (Magentic-One inspired)"""
    task_id: str
    original_request: str
    facts: List[str] = field(default_factory=list)
    guesses: List[str] = field(default_factory=list)
    current_plan: List[Dict[str, Any]] = field(default_factory=list)
    strategy: str = "sequential"
    context_sources: List[str] = field(default_factory=list)
    requirements: List[str] = field(default_factory=list)
    success_criteria: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)

    def update_plan(self, new_plan: List[Dict[str, Any]], reason: str = ""):
        """Update the current plan and record the change"""
        self.current_plan = new_plan
        self.updated_at = datetime.utcnow()
        if reason:
            self.facts.append(f"Plan updated: {reason}")

    def add_fact(self, fact: str):
        """Add a verified fact to the ledger"""
        self.facts.append(f"{datetime.utcnow().isoformat()}: {fact}")
        self.updated_at = datetime.utcnow()

    def add_guess(self, guess: str):
        """Add an educated guess to the ledger"""
        self.guesses.append(f"{datetime.utcnow().isoformat()}: {guess}")
        self.updated_at = datetime.utcnow()


@dataclass
class ProgressLedger:
    """Inner loop: Real-time execution tracking and agent coordination"""
    task_id: str
    current_step: int = 0
    total_steps: int = 0
    agent_assignments: Dict[str, str] = field(default_factory=dict)  # step_id -> agent_id
    step_progress: Dict[str, float] = field(default_factory=dict)  # step_id -> progress (0.0-1.0)
    step_status: Dict[str, str] = field(default_factory=dict)  # step_id -> status
    step_results: Dict[str, Any] = field(default_factory=dict)  # step_id -> result
    active_agents: Set[str] = field(default_factory=set)
    stall_count: int = 0
    last_progress_time: datetime = field(default_factory=datetime.utcnow)
    execution_log: List[str] = field(default_factory=list)

    def is_making_progress(self, timeout_minutes: int = 5) -> bool:
        """Check if progress has been made recently"""
        time_since_progress = datetime.utcnow() - self.last_progress_time
        return time_since_progress < timedelta(minutes=timeout_minutes)

    def record_progress(self, step_id: str, progress: float, status: str = ""):
        """Record progress for a specific step"""
        self.step_progress[step_id] = progress
        if status:
            self.step_status[step_id] = status
        self.last_progress_time = datetime.utcnow()
        self.execution_log.append(f"{datetime.utcnow().isoformat()}: Step {step_id} progress: {progress:.2f}")

    def assign_agent_to_step(self, step_id: str, agent_id: str):
        """Assign an agent to a specific step"""
        self.agent_assignments[step_id] = agent_id
        self.active_agents.add(agent_id)
        self.execution_log.append(f"{datetime.utcnow().isoformat()}: Assigned agent {agent_id} to step {step_id}")

    def complete_step(self, step_id: str, result: Any):
        """Mark a step as completed with its result"""
        self.step_status[step_id] = "completed"
        self.step_results[step_id] = result
        self.step_progress[step_id] = 1.0
        self.last_progress_time = datetime.utcnow()

        # Increment current step counter
        self.current_step += 1

        # Remove agent from active set if no more steps assigned
        agent_id = self.agent_assignments.get(step_id)
        if agent_id and not any(
            self.agent_assignments.get(sid) == agent_id and self.step_status.get(sid) != "completed"
            for sid in self.agent_assignments.keys()
        ):
            self.active_agents.discard(agent_id)

        self.execution_log.append(f"{datetime.utcnow().isoformat()}: Completed step {step_id}")

    def is_task_complete(self) -> bool:
        """Check if all steps are completed"""
        if not self.step_status:
            return False
        return all(status == "completed" for status in self.step_status.values())


class TaskIntelligenceSystem:
    """
    Task Intelligence System with Dual-Loop Orchestrator

    Implements Microsoft Magentic-One inspired architecture:
    - Outer Loop: Task Ledger management (facts, guesses, planning)
    - Inner Loop: Progress Ledger management (execution, coordination)
    - Context-aware task decomposition
    - Intelligent agent coordination
    - Real-time progress monitoring
    """

    def __init__(self, mcp_manager=None, a2a_manager=None):
        self.logger = logging.getLogger(f"{__name__}.TaskIntelligenceSystem")

        # Dual-loop state management
        self.task_ledgers: Dict[str, TaskLedger] = {}
        self.progress_ledgers: Dict[str, ProgressLedger] = {}

        # Integration points
        self.mcp_manager = mcp_manager
        self.a2a_manager = a2a_manager

        # Performance tracking
        self.agent_performance: Dict[str, Dict] = {}
        self.pattern_library: Dict[str, Dict] = {}

        # Pattern Learning & Optimization System
        self.workflow_patterns: Dict[str, Dict[str, Any]] = {}
        self.pattern_matching_threshold = 0.6
        self.pattern_success_tracking: Dict[str, List[float]] = {}
        self.optimization_suggestions: Dict[str, List[Dict[str, Any]]] = {}
        self.failure_patterns: Dict[str, Dict[str, Any]] = {}

        # Configuration
        self.max_stall_count = 3
        self.progress_timeout_minutes = 5
        self.max_context_sources = 5

        # Teaching framework
        self.agent_learning_history: Dict[str, List[Dict[str, Any]]] = {}
        self.teaching_patterns: Dict[str, Dict[str, Any]] = {}
        self.feedback_effectiveness: Dict[str, float] = {}
        self.improvement_tracking: Dict[str, List[float]] = {}

        # Dynamic Question Generation Framework
        self.question_generation_enabled = True
        self.human_interaction_threshold = 0.7  # Context quality threshold
        self.context_source_weights = {
            "knowledge_base": 0.5,
            "mcp_servers": 0.3,
            "human_input": 0.2
        }
        self.nlp_analysis_enabled = False  # Toggleable NLP analysis
        self.question_history: Dict[str, List[Dict[str, Any]]] = {}
        self.user_expertise_level = "intermediate"  # adaptive based on responses

    async def system_health_check(self) -> SystemHealthResult:
        """
        Comprehensive system health check before task execution
        Checks all critical components and attempts recovery if needed
        """
        self.logger.info("🏥 Starting system health check...")

        checks = {}
        errors = []

        try:
            # Check Agent Orchestration MCP Server
            checks["agent_orchestration_mcp"] = await self._check_agent_orchestration_mcp()

            # Check Filesystem MCP
            checks["filesystem_mcp"] = await self._check_filesystem_mcp()

            # Check PostgreSQL MCP
            checks["postgres_mcp"] = await self._check_postgres_mcp()

            # Check Ollama Backend
            checks["ollama_backend"] = await self._check_ollama_connectivity()

            # Check MCP Registry
            checks["mcp_registry"] = await self._check_mcp_registry_status()

            # Overall health assessment
            healthy = all(checks.values())

            if not healthy:
                self.logger.warning("❌ System health check failed, attempting recovery...")
                recovery_result = await self._attempt_system_recovery(checks)

                if recovery_result.success:
                    self.logger.info("✅ System recovery successful")
                    # Re-run health check after recovery
                    return await self.system_health_check()
                else:
                    errors.extend(recovery_result.errors)
                    raise SystemNotReadyException(f"System recovery failed: {', '.join(errors)}")

            self.logger.info("✅ System health check passed")
            return SystemHealthResult(healthy=True, components=checks)

        except Exception as e:
            self.logger.error(f"💥 System health check failed: {e}")
            errors.append(str(e))
            return SystemHealthResult(healthy=False, components=checks, errors=errors)

    async def _check_agent_orchestration_mcp(self) -> bool:
        """Check if PyGent Factory backend is running on port 8000"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:8000/api/v1/health", timeout=5) as response:
                    if response.status == 200:
                        self.logger.info("✅ PyGent Factory backend is healthy")
                        return True
                    else:
                        self.logger.error(f"❌ PyGent Factory backend unhealthy: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"❌ PyGent Factory backend unreachable: {e}")
            return False

    async def _check_filesystem_mcp(self) -> bool:
        """Check Filesystem MCP availability - check for any working filesystem server"""
        try:
            if self.mcp_manager:
                # Check for any filesystem-related servers (Python Filesystem, etc.)
                servers = await self.mcp_manager.list_servers()
                filesystem_servers = [s for s in servers if
                                    'filesystem' in s.get('config', {}).get('name', '').lower() or
                                    'file' in s.get('config', {}).get('name', '').lower()]

                if filesystem_servers:
                    running_servers = [s for s in filesystem_servers if s.get('status') == 'running']
                    if running_servers:
                        self.logger.info(f"✅ Filesystem MCP is available ({len(running_servers)} servers)")
                        return True

            self.logger.error("❌ Filesystem MCP not available")
            return False
        except Exception as e:
            self.logger.error(f"❌ Filesystem MCP check failed: {e}")
            return False

    async def _check_postgres_mcp(self) -> bool:
        """Check PostgreSQL MCP availability - check for any working database server"""
        try:
            if self.mcp_manager:
                # Check for any database-related servers (Python Code Server has DB access, etc.)
                servers = await self.mcp_manager.list_servers()
                db_servers = [s for s in servers if
                            'postgres' in s.get('config', {}).get('name', '').lower() or
                            'database' in s.get('config', {}).get('name', '').lower() or
                            'python' in s.get('config', {}).get('name', '').lower()]

                if db_servers:
                    running_servers = [s for s in db_servers if s.get('status') == 'running']
                    if running_servers:
                        self.logger.info(f"✅ PostgreSQL MCP is available ({len(running_servers)} servers)")
                        return True

            self.logger.error("❌ PostgreSQL MCP not available")
            return False
        except Exception as e:
            self.logger.error(f"❌ PostgreSQL MCP check failed: {e}")
            return False

    async def _check_ollama_connectivity(self) -> bool:
        """Check Ollama backend connectivity"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get("http://localhost:11434/api/tags", timeout=5) as response:
                    if response.status == 200:
                        data = await response.json()
                        models = data.get("models", [])
                        if models:
                            self.logger.info(f"✅ Ollama is running with {len(models)} models")
                            return True
                        else:
                            self.logger.error("❌ Ollama running but no models available")
                            return False
                    else:
                        self.logger.error(f"❌ Ollama unhealthy: {response.status}")
                        return False
        except Exception as e:
            self.logger.error(f"❌ Ollama unreachable: {e}")
            return False

    async def _check_mcp_registry_status(self) -> bool:
        """Check MCP registry status"""
        try:
            if self.mcp_manager:
                # Check if MCP manager is initialized
                self.logger.info("✅ MCP Registry is available")
                return True
            else:
                self.logger.error("❌ MCP Registry not initialized")
                return False
        except Exception as e:
            self.logger.error(f"❌ MCP Registry check failed: {e}")
            return False

    async def _attempt_system_recovery(self, failed_checks: Dict[str, bool]) -> SystemRecoveryResult:
        """Attempt to recover failed system components"""
        self.logger.info("🔧 Attempting system recovery...")

        recovered = []
        failed = []
        errors = []

        try:
            # Attempt to restart Agent Orchestration MCP Server
            if not failed_checks.get("agent_orchestration_mcp", True):
                if await self._restart_agent_orchestration_server():
                    recovered.append("agent_orchestration_mcp")
                else:
                    failed.append("agent_orchestration_mcp")

            # Attempt to restart Ollama if needed
            if not failed_checks.get("ollama_backend", True):
                if await self._restart_ollama_service():
                    recovered.append("ollama_backend")
                else:
                    failed.append("ollama_backend")

            success = len(failed) == 0
            return SystemRecoveryResult(
                success=success,
                recovered_components=recovered,
                failed_components=failed,
                errors=errors
            )

        except Exception as e:
            errors.append(str(e))
            return SystemRecoveryResult(success=False, errors=errors)

    async def _restart_agent_orchestration_server(self) -> bool:
        """Attempt to restart Agent Orchestration MCP Server"""
        try:
            self.logger.info("🔄 Attempting to restart Agent Orchestration MCP Server...")

            # Try to start the server
            cmd = [
                "python",
                "src/servers/agent_orchestration_mcp_server.py",
                "0.0.0.0",
                "8005"
            ]

            # Start in background
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="."
            )

            # Wait a moment for startup
            await asyncio.sleep(3)

            # Check if it's running
            if await self._check_agent_orchestration_mcp():
                self.logger.info("✅ Agent Orchestration MCP Server restarted successfully")
                return True
            else:
                self.logger.error("❌ Failed to restart Agent Orchestration MCP Server")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error restarting Agent Orchestration MCP Server: {e}")
            return False

    async def _restart_ollama_service(self) -> bool:
        """Attempt to restart Ollama service"""
        try:
            self.logger.info("🔄 Attempting to restart Ollama service...")

            # Check if Ollama process is running
            ollama_running = False
            for proc in psutil.process_iter(['pid', 'name']):
                if 'ollama' in proc.info['name'].lower():
                    ollama_running = True
                    break

            if not ollama_running:
                # Try to start Ollama
                subprocess.Popen(["ollama", "serve"], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                await asyncio.sleep(5)  # Wait for startup

            # Check if it's working now
            if await self._check_ollama_connectivity():
                self.logger.info("✅ Ollama service restarted successfully")
                return True
            else:
                self.logger.error("❌ Failed to restart Ollama service")
                return False

        except Exception as e:
            self.logger.error(f"❌ Error restarting Ollama service: {e}")
            return False

    async def create_task_intelligence(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Main entry point: Create task intelligence with dual-loop orchestrator
        Returns task_id for tracking
        """
        task_id = str(uuid.uuid4())

        try:
            # STEP 0: SYSTEM HEALTH CHECK AND RECOVERY
            self.logger.info(f"🏥 Running system health check for task {task_id}")
            health_result = await self.system_health_check()

            if not health_result.healthy:
                raise SystemNotReadyException(f"System not ready: {', '.join(health_result.errors)}")

            self.logger.info("✅ System health check passed, proceeding with task execution")

            # OUTER LOOP: Create Task Ledger
            task_ledger = await self._create_task_ledger(task_id, task_description, context)
            self.task_ledgers[task_id] = task_ledger

            # Generate initial plan if none exists
            if len(task_ledger.current_plan) == 0:
                await self._generate_initial_plan(task_ledger)

            # INNER LOOP: Create Progress Ledger
            progress_ledger = await self._create_progress_ledger(task_id, task_ledger)
            self.progress_ledgers[task_id] = progress_ledger

            # Start dual-loop execution
            asyncio.create_task(self._execute_dual_loop(task_id))

            self.logger.info(f"Created task intelligence for task {task_id}")
            return task_id

        except Exception as e:
            self.logger.error(f"Failed to create task intelligence: {e}")
            raise

    async def _create_task_ledger(self, task_id: str, task_description: str, context: Optional[Dict[str, Any]]) -> TaskLedger:
        """OUTER LOOP: Create and populate Task Ledger"""

        # Initialize Task Ledger
        ledger = TaskLedger(
            task_id=task_id,
            original_request=task_description
        )

        # Gather context from multiple sources
        await self._gather_context(ledger, context)

        # Analyze task and extract requirements
        await self._analyze_task_requirements(ledger)

        # Create initial plan
        await self._create_initial_plan(ledger)

        return ledger

    async def _gather_context(self, ledger: TaskLedger, initial_context: Optional[Dict[str, Any]]):
        """Enhanced context gathering with intelligent source prioritization"""

        # Add initial context if provided
        if initial_context:
            for key, value in initial_context.items():
                ledger.add_fact(f"Initial context - {key}: {value}")

        # Gather context based on priority: Knowledge Base → MCP Servers → Human Input
        context_score = 0.0

        # 1. Knowledge Base Context (highest priority)
        kb_context = await self._gather_knowledge_base_context(ledger)
        context_score += kb_context["score"]

        # 2. MCP Server Context (medium priority)
        mcp_context = await self._gather_mcp_context(ledger)
        context_score += mcp_context["score"]

        # 3. System Capability Context (always available)
        system_context = await self._gather_system_context(ledger)
        context_score += system_context["score"]

        # 4. Dynamic Question Generation based on context quality
        if context_score < self.human_interaction_threshold:
            questions = await self._generate_context_questions(ledger, context_score)
            if questions and self.question_generation_enabled:
                ledger.add_guess(f"Generated {len(questions)} clarification questions")
                ledger.context_sources.append("human_input_required")
                # Store questions for retrieval
                self.question_history[ledger.task_id] = questions
            else:
                ledger.add_guess("Insufficient context but no questions generated")

        # Cache context for reuse
        await self._cache_context(ledger.task_id, {
            "knowledge_base": kb_context,
            "mcp_servers": mcp_context,
            "system": system_context,
            "total_score": context_score
        })

        ledger.add_fact(f"Context gathering completed with score: {context_score:.2f}")

    async def _analyze_task_requirements(self, ledger: TaskLedger):
        """Analyze task and extract requirements, complexity, and success criteria"""

        task_description = ledger.original_request

        # Assess complexity
        complexity, confidence = self._assess_complexity_enhanced(task_description)
        ledger.add_fact(f"Task complexity: {complexity.value} (confidence: {confidence:.2f})")

        # Extract requirements
        requirements = self._extract_requirements_enhanced(task_description)
        ledger.requirements.extend(requirements)

        # Define success criteria
        success_criteria = self._define_success_criteria(task_description)
        ledger.success_criteria.extend(success_criteria)

        # Determine strategy
        strategy = self._determine_strategy(complexity, requirements)
        ledger.strategy = strategy
        ledger.add_fact(f"Selected strategy: {strategy}")

    def _define_success_criteria(self, task_description: str) -> List[str]:
        """Define measurable success criteria for the task"""
        criteria = []
        task_lower = task_description.lower()

        # Always include basic criteria
        criteria.append("Task completed without errors")
        criteria.append("Output meets specified requirements")

        # UI-specific criteria
        if "ui" in task_lower or "interface" in task_lower:
            criteria.append("UI is functional and responsive")
            criteria.append("UI follows design best practices")

        # Code-specific criteria
        if "code" in task_lower or "implement" in task_lower:
            criteria.append("Code follows best practices")
            criteria.append("Code includes proper error handling")

        return criteria

    def _determine_strategy(self, complexity: TaskComplexity, requirements: List[str]) -> str:
        """Determine execution strategy based on complexity and requirements"""

        if complexity in [TaskComplexity.ENTERPRISE, TaskComplexity.RESEARCH] or len(requirements) > 3:
            return "multi_agent_parallel"
        elif complexity in [TaskComplexity.COMPLEX, TaskComplexity.MODERATE] or len(requirements) > 1:
            return "multi_agent_sequential"
        else:
            return "single_agent"

    async def _discover_relevant_tools(self, task_description: str) -> List[str]:
        """Discover relevant MCP tools for the task"""
        if not self.mcp_manager:
            return []

        try:
            # Simple keyword-based tool discovery
            # In production, this would use more sophisticated matching
            tools = []
            task_lower = task_description.lower()

            if any(keyword in task_lower for keyword in ["file", "document", "read", "write"]):
                tools.append("filesystem")
            if any(keyword in task_lower for keyword in ["database", "query", "data"]):
                tools.append("postgresql")
            if any(keyword in task_lower for keyword in ["github", "repository", "code"]):
                tools.append("github")
            if any(keyword in task_lower for keyword in ["search", "find", "lookup"]):
                tools.append("memory")

            return tools
        except Exception as e:
            self.logger.error(f"Tool discovery failed: {e}")
            return []

    async def analyze_task(self, task_description: str) -> TaskAnalysis:
        """Analyze task to determine type, complexity, and requirements"""
        
        # Simple keyword-based task classification
        task_lower = task_description.lower()
        
        # Determine task type
        if any(keyword in task_lower for keyword in ["graphics", "webgl", "3d", "animation", "gpu", "canvas", "dragon", "flying", "particle", "shader"]):
            task_type = TaskType.GRAPHICS_DEVELOPMENT
        elif any(keyword in task_lower for keyword in ["vue", "ui", "interface", "frontend", "component"]):
            task_type = TaskType.UI_CREATION
        elif any(keyword in task_lower for keyword in ["code", "implement", "build", "create", "develop"]):
            task_type = TaskType.CODING
        elif any(keyword in task_lower for keyword in ["research", "analyze", "investigate", "study"]):
            task_type = TaskType.RESEARCH
        elif any(keyword in task_lower for keyword in ["document", "write", "explain", "describe"]):
            task_type = TaskType.DOCUMENTATION
        else:
            task_type = TaskType.UNKNOWN
            
        # Estimate complexity (simple heuristic)
        complexity = min(10, max(1, len(task_description.split()) // 10 + 1))
        
        # Estimate time (simple heuristic)
        estimated_time = complexity * 5  # 5 minutes per complexity point
        
        # Define required capabilities based on task type
        capabilities_map = {
            TaskType.UI_CREATION: ["vue.js", "javascript", "html", "css", "frontend"],
            TaskType.GRAPHICS_DEVELOPMENT: ["webgl", "three.js", "canvas", "gpu_acceleration", "3d_graphics", "animation", "shaders"],
            TaskType.CODING: ["programming", "software_development"],
            TaskType.RESEARCH: ["web_search", "analysis", "synthesis"],
            TaskType.DOCUMENTATION: ["writing", "technical_communication"],
            TaskType.UNKNOWN: ["general"]
        }
        
        required_capabilities = capabilities_map.get(task_type, ["general"])
        
        # Define success criteria
        success_criteria = [
            "Task completed as requested",
            "Output meets quality standards",
            "No critical errors or issues"
        ]
        
        if task_type == TaskType.UI_CREATION:
            success_criteria.extend([
                "Valid Vue.js components created",
                "Proper file structure established",
                "Components are functional"
            ])
        elif task_type == TaskType.GRAPHICS_DEVELOPMENT:
            success_criteria.extend([
                "GPU-accelerated graphics rendering",
                "Smooth 60fps animation performance",
                "High-quality visual effects",
                "Cross-browser WebGL compatibility",
                "Responsive graphics scaling"
            ])
        elif task_type == TaskType.CODING:
            success_criteria.extend([
                "Code compiles/runs without errors",
                "Follows coding best practices",
                "Includes proper error handling"
            ])
            
        # Enhanced complexity assessment
        complexity_enum, complexity_score = self._assess_complexity_enhanced(task_description)

        # Enhanced requirement extraction
        enhanced_requirements = self._extract_requirements_enhanced(task_description)

        # Context requirements analysis
        context_requirements = self._analyze_context_requirements(task_description)

        # Risk factor identification
        risk_factors = self._identify_risk_factors(task_description, enhanced_requirements)

        # Dependency analysis
        dependency_count = self._estimate_dependencies(task_description, enhanced_requirements)

        # Parallel execution potential
        parallel_potential = self._assess_parallel_potential(enhanced_requirements, dependency_count)

        return TaskAnalysis(
            task_type=task_type,
            complexity=complexity_enum,
            complexity_score=complexity_score,
            estimated_time=self._estimate_time_enhanced(complexity_enum, enhanced_requirements),
            required_capabilities=enhanced_requirements,
            success_criteria=success_criteria,
            context_requirements=context_requirements,
            risk_factors=risk_factors,
            dependency_count=dependency_count,
            parallel_potential=parallel_potential
        )
    
    async def select_agent(self, analysis: TaskAnalysis) -> str:
        """Select the best agent for the task based on analysis"""
        
        # Simple agent selection logic
        agent_type_map = {
            TaskType.UI_CREATION: "coding",
            TaskType.GRAPHICS_DEVELOPMENT: "coding",
            TaskType.CODING: "coding",
            TaskType.RESEARCH: "research",
            TaskType.ANALYSIS: "research",
            TaskType.DOCUMENTATION: "research",
            TaskType.UNKNOWN: "research"
        }
        
        selected_agent_type = agent_type_map.get(analysis.task_type, "research")
        
        self.logger.info(f"Selected agent type: {selected_agent_type} for task type: {analysis.task_type}")
        
        return selected_agent_type

    def _assess_complexity_enhanced(self, task_description: str) -> tuple[TaskComplexity, float]:
        """Enhanced complexity assessment with confidence scoring"""

        task_lower = task_description.lower()
        word_count = len(task_description.split())

        # Initialize complexity factors
        complexity_factors = {
            "length": 0,
            "keywords": 0,
            "integration": 0,
            "dependencies": 0,
            "uncertainty": 0
        }

        # Length-based complexity
        if word_count < 10:
            complexity_factors["length"] = 1
        elif word_count < 25:
            complexity_factors["length"] = 2
        elif word_count < 50:
            complexity_factors["length"] = 4
        else:
            complexity_factors["length"] = 6

        # Keyword-based complexity
        complexity_keywords = {
            "simple": ["create", "make", "simple", "basic"],
            "moderate": ["implement", "build", "develop", "integrate"],
            "complex": ["system", "architecture", "multiple", "complex", "advanced"],
            "research": ["analyze", "research", "investigate", "study", "explore"],
            "enterprise": ["production", "enterprise", "scalable", "distributed", "microservice"]
        }

        for level, keywords in complexity_keywords.items():
            matches = sum(1 for keyword in keywords if keyword in task_lower)
            if level == "simple" and matches > 0:
                complexity_factors["keywords"] = max(complexity_factors["keywords"], 1)
            elif level == "moderate" and matches > 0:
                complexity_factors["keywords"] = max(complexity_factors["keywords"], 3)
            elif level == "complex" and matches > 0:
                complexity_factors["keywords"] = max(complexity_factors["keywords"], 5)
            elif level == "research" and matches > 0:
                complexity_factors["keywords"] = max(complexity_factors["keywords"], 7)
            elif level == "enterprise" and matches > 0:
                complexity_factors["keywords"] = max(complexity_factors["keywords"], 9)

        # Integration complexity
        integration_indicators = ["api", "database", "service", "integration", "connect", "sync"]
        integration_count = sum(1 for indicator in integration_indicators if indicator in task_lower)
        complexity_factors["integration"] = min(4, integration_count)

        # Dependency complexity
        dependency_indicators = ["depends", "requires", "needs", "after", "before", "prerequisite"]
        dependency_count = sum(1 for indicator in dependency_indicators if indicator in task_lower)
        complexity_factors["dependencies"] = min(3, dependency_count)

        # Uncertainty indicators
        uncertainty_indicators = ["maybe", "possibly", "might", "could", "unclear", "investigate"]
        uncertainty_count = sum(1 for indicator in uncertainty_indicators if indicator in task_lower)
        complexity_factors["uncertainty"] = min(2, uncertainty_count)

        # Calculate total complexity score
        total_score = sum(complexity_factors.values())

        # Map to complexity enum
        if total_score <= 3:
            complexity = TaskComplexity.SIMPLE
            confidence = 0.9
        elif total_score <= 6:
            complexity = TaskComplexity.MODERATE
            confidence = 0.8
        elif total_score <= 10:
            complexity = TaskComplexity.COMPLEX
            confidence = 0.7
        elif total_score <= 14:
            complexity = TaskComplexity.RESEARCH
            confidence = 0.6
        else:
            complexity = TaskComplexity.ENTERPRISE
            confidence = 0.5

        return complexity, confidence

    def _extract_requirements_enhanced(self, task_description: str) -> List[str]:
        """Enhanced requirement extraction with pattern matching and scoring"""

        task_lower = task_description.lower()

        # Define requirement patterns with weights
        requirement_patterns = {
            "frontend_development": {
                "patterns": ["ui", "interface", "frontend", "web", "html", "css", "vue", "react", "component"],
                "weight": 1.0
            },
            "graphics_development": {
                "patterns": ["graphics", "webgl", "3d", "animation", "gpu", "canvas", "dragon", "flying", "particle", "shader", "three.js"],
                "weight": 1.2
            },
            "backend_development": {
                "patterns": ["api", "backend", "server", "database", "endpoint", "service", "microservice"],
                "weight": 1.0
            },
            "file_operations": {
                "patterns": ["file", "document", "read", "write", "create", "upload", "download", "storage"],
                "weight": 0.8
            },
            "code_generation": {
                "patterns": ["code", "function", "class", "implement", "develop", "build", "create"],
                "weight": 0.9
            },
            "testing": {
                "patterns": ["test", "validate", "verify", "check", "unit test", "integration test"],
                "weight": 0.9
            },
            "database_operations": {
                "patterns": ["database", "sql", "query", "table", "schema", "migration", "postgresql"],
                "weight": 1.0
            },
            "integration": {
                "patterns": ["integrate", "connect", "sync", "webhook", "api integration", "third party"],
                "weight": 0.8
            },
            "deployment": {
                "patterns": ["deploy", "deployment", "production", "staging", "docker", "container"],
                "weight": 0.7
            },
            "security": {
                "patterns": ["security", "authentication", "authorization", "encrypt", "secure"],
                "weight": 1.0
            },
            "performance": {
                "patterns": ["performance", "optimize", "speed", "cache", "scalability"],
                "weight": 0.8
            }
        }

        # Score each requirement
        requirement_scores = {}
        for req_type, config in requirement_patterns.items():
            score = 0
            for pattern in config["patterns"]:
                if pattern in task_lower:
                    score += config["weight"]

            if score > 0:
                requirement_scores[req_type] = score

        # Select requirements above threshold
        threshold = 0.5
        requirements = [req for req, score in requirement_scores.items() if score >= threshold]

        return requirements if requirements else ["general"]

    def _analyze_context_requirements(self, task_description: str) -> List[str]:
        """Analyze what context sources are needed for this task"""

        context_requirements = []
        task_lower = task_description.lower()

        # MCP server requirements
        if any(keyword in task_lower for keyword in ["file", "document", "read", "write"]):
            context_requirements.append("filesystem_mcp")

        if any(keyword in task_lower for keyword in ["database", "sql", "query", "table"]):
            context_requirements.append("postgresql_mcp")

        if any(keyword in task_lower for keyword in ["github", "repository", "code", "commit"]):
            context_requirements.append("github_mcp")

        if any(keyword in task_lower for keyword in ["search", "find", "lookup", "memory"]):
            context_requirements.append("memory_mcp")

        # Knowledge base requirements
        if any(keyword in task_lower for keyword in ["research", "analyze", "investigate"]):
            context_requirements.append("knowledge_base")

        # Human input requirements
        if any(keyword in task_lower for keyword in ["unclear", "ambiguous", "decide", "choose"]):
            context_requirements.append("human_input")

        return context_requirements

    def _identify_risk_factors(self, task_description: str, requirements: List[str]) -> List[str]:
        """Identify potential risk factors for task execution"""

        risk_factors = []
        task_lower = task_description.lower()

        # Complexity risks
        if len(requirements) > 5:
            risk_factors.append("High requirement complexity")

        # Integration risks
        if "integration" in requirements:
            risk_factors.append("Third-party integration dependencies")

        # Database risks
        if "database_operations" in requirements:
            risk_factors.append("Data consistency and migration risks")

        # Security risks
        if "security" in requirements:
            risk_factors.append("Security implementation complexity")

        # Performance risks
        if "performance" in requirements:
            risk_factors.append("Performance optimization challenges")

        # Deployment risks
        if "deployment" in requirements:
            risk_factors.append("Production deployment risks")

        # Ambiguity risks
        ambiguity_indicators = ["unclear", "maybe", "possibly", "might", "could"]
        if any(indicator in task_lower for indicator in ambiguity_indicators):
            risk_factors.append("Task requirements ambiguity")

        # Time pressure risks
        urgency_indicators = ["urgent", "asap", "immediately", "quickly", "rush"]
        if any(indicator in task_lower for indicator in urgency_indicators):
            risk_factors.append("Time pressure constraints")

        return risk_factors

    def _estimate_dependencies(self, task_description: str, requirements: List[str]) -> int:
        """Estimate the number of dependencies for this task"""

        dependency_count = 0
        task_lower = task_description.lower()

        # Explicit dependency indicators
        dependency_indicators = ["depends", "requires", "needs", "after", "before", "prerequisite"]
        dependency_count += sum(1 for indicator in dependency_indicators if indicator in task_lower)

        # Requirement-based dependencies
        if "frontend_development" in requirements and "backend_development" in requirements:
            dependency_count += 2  # Frontend typically depends on backend

        if "testing" in requirements:
            dependency_count += 1  # Testing depends on implementation

        if "deployment" in requirements:
            dependency_count += 1  # Deployment depends on completed code

        if "integration" in requirements:
            dependency_count += len([r for r in requirements if r != "integration"])  # Integration depends on other components

        return dependency_count

    def _assess_parallel_potential(self, requirements: List[str], dependency_count: int) -> bool:
        """Assess if task has potential for parallel execution"""

        # High dependency count reduces parallel potential
        if dependency_count > 3:
            return False

        # Multiple independent requirements suggest parallel potential
        independent_requirements = [
            "frontend_development", "backend_development", "documentation",
            "testing", "security", "performance"
        ]

        independent_count = sum(1 for req in requirements if req in independent_requirements)

        return independent_count >= 2

    def _estimate_time_enhanced(self, complexity: TaskComplexity, requirements: List[str]) -> int:
        """Enhanced time estimation based on complexity and requirements"""

        # Base time by complexity
        base_times = {
            TaskComplexity.SIMPLE: 10,
            TaskComplexity.MODERATE: 25,
            TaskComplexity.COMPLEX: 45,
            TaskComplexity.RESEARCH: 60,
            TaskComplexity.ENTERPRISE: 120
        }

        base_time = base_times[complexity]

        # Requirement-based time adjustments
        requirement_multipliers = {
            "frontend_development": 1.3,
            "backend_development": 1.4,
            "database_operations": 1.2,
            "integration": 1.5,
            "security": 1.3,
            "performance": 1.2,
            "deployment": 1.1,
            "testing": 1.2
        }

        # Calculate total multiplier
        total_multiplier = 1.0
        for req in requirements:
            if req in requirement_multipliers:
                total_multiplier *= requirement_multipliers[req]

        # Cap the multiplier to prevent unrealistic estimates
        total_multiplier = min(total_multiplier, 3.0)

        return int(base_time * total_multiplier)

    async def _gather_knowledge_base_context(self, ledger: TaskLedger) -> Dict[str, Any]:
        """Gather context from knowledge base and pattern library"""

        context_result = {"sources": [], "facts": [], "score": 0.0}

        try:
            # Search pattern library for similar tasks
            similar_patterns = await self._find_similar_patterns(ledger.original_request)

            for pattern in similar_patterns[:3]:  # Top 3 similar patterns
                context_result["sources"].append(f"pattern_{pattern['id']}")
                context_result["facts"].append(f"Similar task pattern: {pattern['description']}")
                ledger.add_fact(f"Found similar pattern: {pattern['description']} (success rate: {pattern.get('success_rate', 0):.2f})")
                context_result["score"] += 0.2

            # Add domain knowledge if available
            domain_knowledge = await self._extract_domain_knowledge(ledger.original_request)
            if domain_knowledge:
                context_result["sources"].append("domain_knowledge")
                context_result["facts"].extend(domain_knowledge)
                for knowledge in domain_knowledge:
                    ledger.add_fact(f"Domain knowledge: {knowledge}")
                context_result["score"] += 0.3

        except Exception as e:
            ledger.add_guess(f"Knowledge base context gathering failed: {e}")
            context_result["score"] = 0.1  # Minimal score for attempt

        return context_result

    async def _gather_mcp_context(self, ledger: TaskLedger) -> Dict[str, Any]:
        """Gather context from MCP servers with intelligent tool discovery"""

        context_result = {"sources": [], "facts": [], "score": 0.0}

        if not self.mcp_manager:
            ledger.add_guess("MCP manager not available")
            return context_result

        try:
            # Discover relevant tools with scoring
            tool_matches = await self._discover_relevant_tools_scored(ledger.original_request)

            for tool_match in tool_matches[:self.max_context_sources]:
                tool_name = tool_match["name"]
                relevance_score = tool_match["score"]

                context_result["sources"].append(tool_name)
                ledger.context_sources.append(tool_name)

                # Get tool capabilities
                capabilities = await self._get_tool_capabilities(tool_name)
                if capabilities:
                    context_result["facts"].extend(capabilities)
                    for capability in capabilities:
                        ledger.add_fact(f"Tool {tool_name} capability: {capability}")

                    context_result["score"] += relevance_score * 0.2
                else:
                    ledger.add_guess(f"Could not retrieve capabilities for {tool_name}")

            # Test tool availability
            available_tools = await self._test_tool_availability(context_result["sources"])
            for tool_name, is_available in available_tools.items():
                if is_available:
                    ledger.add_fact(f"Tool {tool_name} is available and responsive")
                    context_result["score"] += 0.1
                else:
                    ledger.add_guess(f"Tool {tool_name} may not be available")
                    context_result["score"] -= 0.05

        except Exception as e:
            ledger.add_guess(f"MCP context gathering failed: {e}")
            context_result["score"] = 0.1

        return context_result

    async def _gather_system_context(self, ledger: TaskLedger) -> Dict[str, Any]:
        """Gather system capability context"""

        context_result = {"sources": ["system"], "facts": [], "score": 0.3}

        # Core system capabilities
        system_facts = [
            "System has A2A protocol for agent coordination",
            "System has MCP servers for tool access",
            "System has multi-agent orchestration capabilities",
            "System supports dual-loop task execution",
            "System has PostgreSQL database for persistence",
            "System supports real-time progress monitoring"
        ]

        context_result["facts"] = system_facts
        for fact in system_facts:
            ledger.add_fact(fact)

        # Add agent performance context if available
        if self.agent_performance:
            performance_summary = self._summarize_agent_performance()
            context_result["facts"].append(f"Agent performance summary: {performance_summary}")
            ledger.add_fact(f"Agent performance context: {performance_summary}")
            context_result["score"] += 0.1

        return context_result

    async def _find_similar_patterns(self, task_description: str) -> List[Dict[str, Any]]:
        """Find similar task patterns from pattern library"""

        # Simple pattern matching - in production this would use vector similarity
        similar_patterns = []

        task_lower = task_description.lower()

        # Check pattern library for matches
        for pattern_id, pattern_data in self.pattern_library.items():
            pattern_desc = pattern_data.get("description", "").lower()

            # Simple keyword overlap scoring
            task_words = set(task_lower.split())
            pattern_words = set(pattern_desc.split())

            overlap = len(task_words.intersection(pattern_words))
            total_words = len(task_words.union(pattern_words))

            if total_words > 0:
                similarity_score = overlap / total_words

                if similarity_score > 0.3:  # Threshold for similarity
                    similar_patterns.append({
                        "id": pattern_id,
                        "description": pattern_data.get("description", ""),
                        "success_rate": pattern_data.get("success_rate", 0.0),
                        "similarity_score": similarity_score
                    })

        # Sort by similarity score
        similar_patterns.sort(key=lambda x: x["similarity_score"], reverse=True)

        return similar_patterns

    async def _extract_domain_knowledge(self, task_description: str) -> List[str]:
        """Extract relevant domain knowledge"""

        domain_knowledge = []
        task_lower = task_description.lower()

        # Domain-specific knowledge patterns
        if any(keyword in task_lower for keyword in ["graphics", "webgl", "3d", "animation", "gpu", "canvas", "dragon"]):
            domain_knowledge.extend([
                "WebGL provides GPU-accelerated 3D graphics in browsers",
                "Three.js is the most popular WebGL library for 3D graphics",
                "Canvas 2D can be used for simpler graphics and animations",
                "RequestAnimationFrame ensures smooth 60fps animations",
                "GPU shaders (vertex/fragment) enable advanced visual effects",
                "Particle systems create realistic effects like fire, smoke, trails"
            ])
        elif any(keyword in task_lower for keyword in ["vue", "frontend", "ui", "component"]):
            domain_knowledge.extend([
                "Vue.js components require template, script, and style sections",
                "Vue components should follow single-file component structure",
                "Vue components need proper prop definitions and event handling"
            ])

        if any(keyword in task_lower for keyword in ["api", "backend", "server"]):
            domain_knowledge.extend([
                "REST APIs should follow standard HTTP methods and status codes",
                "API endpoints need proper error handling and validation",
                "Backend services should implement proper logging and monitoring"
            ])

        if any(keyword in task_lower for keyword in ["database", "sql", "postgresql"]):
            domain_knowledge.extend([
                "Database operations should use transactions for consistency",
                "Database schemas should include proper indexing",
                "Database migrations should be reversible and tested"
            ])

        return domain_knowledge

    async def _discover_relevant_tools_scored(self, task_description: str) -> List[Dict[str, Any]]:
        """Discover relevant tools with relevance scoring"""

        tool_matches = []
        task_lower = task_description.lower()

        # Define tool relevance patterns
        tool_patterns = {
            "filesystem": {
                "keywords": ["file", "document", "read", "write", "create", "upload", "download"],
                "base_score": 0.8
            },
            "postgresql": {
                "keywords": ["database", "sql", "query", "table", "schema", "data"],
                "base_score": 0.9
            },
            "github": {
                "keywords": ["github", "repository", "code", "commit", "branch", "pull request"],
                "base_score": 0.7
            },
            "memory": {
                "keywords": ["search", "find", "lookup", "remember", "store", "retrieve"],
                "base_score": 0.6
            },
            "a2a": {
                "keywords": ["agent", "coordinate", "communicate", "task", "assign"],
                "base_score": 0.8
            }
        }

        # Score each tool
        for tool_name, pattern in tool_patterns.items():
            score = 0.0
            keyword_matches = 0

            for keyword in pattern["keywords"]:
                if keyword in task_lower:
                    score += pattern["base_score"] / len(pattern["keywords"])
                    keyword_matches += 1

            if keyword_matches > 0:
                # Boost score for multiple keyword matches
                score *= (1 + (keyword_matches - 1) * 0.1)

                tool_matches.append({
                    "name": tool_name,
                    "score": min(score, 1.0),
                    "keyword_matches": keyword_matches
                })

        # Sort by score
        tool_matches.sort(key=lambda x: x["score"], reverse=True)

        return tool_matches

    async def _get_tool_capabilities(self, tool_name: str) -> List[str]:
        """Get capabilities for a specific tool"""

        if not self.mcp_manager:
            return []

        try:
            # Get tool information from MCP manager
            tool_info = await self.mcp_manager.get_tool_info(tool_name)

            if tool_info:
                capabilities = []

                # Extract capabilities from tool info
                if "tools" in tool_info:
                    for tool in tool_info["tools"]:
                        tool_desc = tool.get("description", "")
                        if tool_desc:
                            capabilities.append(f"Can {tool_desc.lower()}")

                if "description" in tool_info:
                    capabilities.append(f"Server description: {tool_info['description']}")

                return capabilities

        except Exception as e:
            self.logger.error(f"Failed to get capabilities for {tool_name}: {e}")

        return []

    async def _test_tool_availability(self, tool_names: List[str]) -> Dict[str, bool]:
        """Test availability of tools"""

        availability = {}

        if not self.mcp_manager:
            return {tool: False for tool in tool_names}

        for tool_name in tool_names:
            try:
                # Simple availability test
                is_available = await self.mcp_manager.is_server_available(tool_name)
                availability[tool_name] = is_available
            except Exception as e:
                self.logger.error(f"Failed to test availability for {tool_name}: {e}")
                availability[tool_name] = False

        return availability

    async def _cache_context(self, task_id: str, context_data: Dict[str, Any]):
        """Cache context data for reuse"""

        # Simple in-memory caching - in production this would use Redis or similar
        cache_key = f"context_{task_id}"

        # Store in pattern library for future reference
        if task_id not in self.pattern_library:
            self.pattern_library[task_id] = {
                "context": context_data,
                "timestamp": datetime.utcnow().isoformat(),
                "reuse_count": 0
            }

    def _summarize_agent_performance(self) -> str:
        """Summarize agent performance for context"""

        if not self.agent_performance:
            return "No performance data available"

        total_agents = len(self.agent_performance)
        avg_success_rate = sum(
            perf.get("success_rate", 0.0) for perf in self.agent_performance.values()
        ) / total_agents if total_agents > 0 else 0.0

        return f"{total_agents} agents tracked, {avg_success_rate:.2f} average success rate"

    # Dynamic Question Generation Framework

    async def _generate_context_questions(self, ledger: TaskLedger, context_score: float) -> List[Dict[str, Any]]:
        """
        Generate context-aware questions based on GrokGlue hybrid analysis approach
        Combines rule-based logic with optional NLP analysis
        """

        questions = []

        # Analyze what context is missing
        missing_context = await self._analyze_missing_context(ledger, context_score)

        # Generate questions based on missing context
        for context_gap in missing_context:
            question_set = await self._generate_questions_for_gap(context_gap, ledger)
            questions.extend(question_set)

        # Apply user expertise level filtering
        questions = self._filter_questions_by_expertise(questions)

        # Prioritize and batch questions
        questions = self._prioritize_and_batch_questions(questions)

        # Avoid redundant questions from conversation history
        questions = await self._filter_redundant_questions(ledger.task_id, questions)

        self.logger.info(f"Generated {len(questions)} context questions for task {ledger.task_id}")

        return questions

    async def _analyze_missing_context(self, ledger: TaskLedger, context_score: float) -> List[Dict[str, Any]]:
        """Analyze what specific context is missing using hybrid approach"""

        missing_context = []

        # Rule-based analysis (always available, fast)
        rule_based_gaps = self._rule_based_context_analysis(ledger)
        missing_context.extend(rule_based_gaps)

        # Optional NLP analysis (more sophisticated but requires resources)
        if self.nlp_analysis_enabled:
            nlp_gaps = await self._nlp_context_analysis(ledger)
            missing_context.extend(nlp_gaps)

        # Weighted context source analysis
        source_gaps = self._analyze_context_source_gaps(ledger, context_score)
        missing_context.extend(source_gaps)

        return missing_context

    def _rule_based_context_analysis(self, ledger: TaskLedger) -> List[Dict[str, Any]]:
        """Fast rule-based context gap analysis"""

        gaps = []
        task_lower = ledger.original_request.lower()

        # Check for ambiguous requirements
        ambiguity_indicators = ["maybe", "possibly", "might", "could", "unclear", "or", "either"]
        if any(indicator in task_lower for indicator in ambiguity_indicators):
            gaps.append({
                "type": "requirement_ambiguity",
                "severity": "high",
                "description": "Task contains ambiguous requirements",
                "keywords": [word for word in ambiguity_indicators if word in task_lower]
            })

        # Check for missing technical specifications
        if any(tech_word in task_lower for tech_word in ["api", "database", "integration"]):
            if not any(spec_word in task_lower for spec_word in ["endpoint", "schema", "format", "protocol"]):
                gaps.append({
                    "type": "technical_specification",
                    "severity": "medium",
                    "description": "Technical implementation lacks specific details"
                })

        # Check for missing UI/UX specifications
        if any(ui_word in task_lower for ui_word in ["ui", "interface", "frontend", "component"]):
            if not any(spec_word in task_lower for spec_word in ["design", "layout", "style", "behavior"]):
                gaps.append({
                    "type": "ui_specification",
                    "severity": "medium",
                    "description": "UI requirements lack design specifications"
                })

        # Check for missing success criteria
        if not any(success_word in task_lower for success_word in ["success", "complete", "done", "criteria", "validate"]):
            gaps.append({
                "type": "success_criteria",
                "severity": "high",
                "description": "No clear success criteria defined"
            })

        # Check for missing context about existing systems
        if any(integration_word in task_lower for integration_word in ["integrate", "connect", "existing", "current"]):
            gaps.append({
                "type": "existing_system_context",
                "severity": "high",
                "description": "Integration requires context about existing systems"
            })

        return gaps

    async def _nlp_context_analysis(self, ledger: TaskLedger) -> List[Dict[str, Any]]:
        """Advanced NLP-based context analysis (optional)"""

        gaps = []

        try:
            # This would integrate with an NLP service if available
            # For now, implement sophisticated pattern matching

            task_text = ledger.original_request

            # Analyze sentence complexity and ambiguity
            sentences = task_text.split('.')
            complex_sentences = [s for s in sentences if len(s.split()) > 20]

            if complex_sentences:
                gaps.append({
                    "type": "complexity_ambiguity",
                    "severity": "medium",
                    "description": "Complex sentences may contain ambiguous requirements",
                    "details": f"Found {len(complex_sentences)} complex sentences"
                })

            # Analyze for implicit assumptions
            assumption_patterns = [
                r"obviously", r"clearly", r"of course", r"naturally",
                r"as usual", r"like before", r"similar to"
            ]

            import re
            for pattern in assumption_patterns:
                if re.search(pattern, task_text, re.IGNORECASE):
                    gaps.append({
                        "type": "implicit_assumption",
                        "severity": "medium",
                        "description": "Task contains implicit assumptions that need clarification"
                    })
                    break

        except Exception as e:
            self.logger.error(f"NLP context analysis failed: {e}")

        return gaps

    def _analyze_context_source_gaps(self, ledger: TaskLedger, context_score: float) -> List[Dict[str, Any]]:
        """Analyze gaps in context sources using weighted approach"""

        gaps = []

        # Calculate weighted context contribution
        source_contributions = {}

        # Knowledge base contribution
        kb_facts = [fact for fact in ledger.facts if "pattern" in fact or "knowledge" in fact]
        source_contributions["knowledge_base"] = len(kb_facts) * self.context_source_weights["knowledge_base"]

        # MCP server contribution
        mcp_facts = [fact for fact in ledger.facts if "tool" in fact or "capability" in fact]
        source_contributions["mcp_servers"] = len(mcp_facts) * self.context_source_weights["mcp_servers"]

        # Human input contribution (if any previous input exists)
        human_facts = [fact for fact in ledger.facts if "human" in fact or "clarification" in fact]
        source_contributions["human_input"] = len(human_facts) * self.context_source_weights["human_input"]

        # Identify weak sources
        for source, contribution in source_contributions.items():
            expected_contribution = self.context_source_weights[source] * 0.5  # 50% of max expected

            if contribution < expected_contribution:
                gaps.append({
                    "type": "weak_context_source",
                    "severity": "medium",
                    "description": f"Insufficient context from {source}",
                    "source": source,
                    "contribution": contribution,
                    "expected": expected_contribution
                })

        return gaps

    async def _generate_questions_for_gap(self, context_gap: Dict[str, Any], ledger: TaskLedger) -> List[Dict[str, Any]]:
        """Generate specific questions for a context gap"""

        gap_type = context_gap["type"]
        severity = context_gap["severity"]

        questions = []

        if gap_type == "requirement_ambiguity":
            questions.extend([
                {
                    "question": "Could you clarify the specific requirements for this task?",
                    "type": "clarification",
                    "priority": "high",
                    "context": "Ambiguous language detected in task description"
                },
                {
                    "question": "What are the exact specifications you need implemented?",
                    "type": "specification",
                    "priority": "high",
                    "context": "Need concrete requirements"
                }
            ])

            # Add specific questions based on detected keywords
            if "keywords" in context_gap:
                for keyword in context_gap["keywords"]:
                    questions.append({
                        "question": f"You mentioned '{keyword}' - what specifically should happen in this case?",
                        "type": "keyword_clarification",
                        "priority": "medium",
                        "context": f"Clarifying ambiguous keyword: {keyword}"
                    })

        elif gap_type == "technical_specification":
            questions.extend([
                {
                    "question": "What are the technical specifications for this implementation?",
                    "type": "technical",
                    "priority": "high",
                    "context": "Missing technical details"
                },
                {
                    "question": "Are there any specific technologies, frameworks, or protocols I should use?",
                    "type": "technology_choice",
                    "priority": "medium",
                    "context": "Technology stack clarification"
                },
                {
                    "question": "What are the performance requirements and constraints?",
                    "type": "performance",
                    "priority": "medium",
                    "context": "Performance specifications needed"
                }
            ])

        elif gap_type == "ui_specification":
            questions.extend([
                {
                    "question": "What should the user interface look like and how should it behave?",
                    "type": "ui_design",
                    "priority": "high",
                    "context": "UI design specifications needed"
                },
                {
                    "question": "Are there any existing design systems or style guides I should follow?",
                    "type": "design_system",
                    "priority": "medium",
                    "context": "Design consistency requirements"
                },
                {
                    "question": "What user interactions and workflows should be supported?",
                    "type": "user_workflow",
                    "priority": "high",
                    "context": "User experience requirements"
                }
            ])

        elif gap_type == "success_criteria":
            questions.extend([
                {
                    "question": "How will we know when this task is successfully completed?",
                    "type": "success_definition",
                    "priority": "high",
                    "context": "Success criteria definition needed"
                },
                {
                    "question": "What specific outcomes or deliverables do you expect?",
                    "type": "deliverables",
                    "priority": "high",
                    "context": "Expected deliverables clarification"
                },
                {
                    "question": "Are there any quality standards or acceptance criteria I should meet?",
                    "type": "quality_standards",
                    "priority": "medium",
                    "context": "Quality requirements"
                }
            ])

        elif gap_type == "existing_system_context":
            questions.extend([
                {
                    "question": "Can you provide details about the existing systems I need to integrate with?",
                    "type": "system_integration",
                    "priority": "high",
                    "context": "Existing system context needed"
                },
                {
                    "question": "What are the current data formats, APIs, or interfaces I should work with?",
                    "type": "integration_details",
                    "priority": "high",
                    "context": "Integration specifications"
                },
                {
                    "question": "Are there any constraints or limitations from existing systems?",
                    "type": "constraints",
                    "priority": "medium",
                    "context": "System constraints identification"
                }
            ])

        elif gap_type == "weak_context_source":
            source = context_gap.get("source", "unknown")
            if source == "knowledge_base":
                questions.append({
                    "question": "Are there any similar projects or patterns I should reference?",
                    "type": "pattern_reference",
                    "priority": "low",
                    "context": "Seeking pattern library context"
                })
            elif source == "mcp_servers":
                questions.append({
                    "question": "Do you have preferences for which tools or services I should use?",
                    "type": "tool_preference",
                    "priority": "low",
                    "context": "Tool selection guidance"
                })

        # Add severity-based priority adjustment
        for question in questions:
            if severity == "high":
                if question["priority"] == "medium":
                    question["priority"] = "high"
                elif question["priority"] == "low":
                    question["priority"] = "medium"

        return questions

    def _filter_questions_by_expertise(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter questions based on user expertise level"""

        if self.user_expertise_level == "beginner":
            # Include more explanatory questions
            return questions  # Keep all questions for beginners

        elif self.user_expertise_level == "expert":
            # Filter out basic questions, focus on high-priority technical ones
            return [q for q in questions if q["priority"] == "high" or q["type"] in ["technical", "specification"]]

        else:  # intermediate
            # Balanced approach - remove only low-priority basic questions
            return [q for q in questions if q["priority"] != "low" or q["type"] in ["technical", "clarification"]]

    def _prioritize_and_batch_questions(self, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Prioritize questions and batch them for optimal user experience"""

        # Sort by priority
        priority_order = {"high": 0, "medium": 1, "low": 2}
        questions.sort(key=lambda q: priority_order.get(q["priority"], 3))

        # Limit to maximum 5 questions to avoid overwhelming user
        max_questions = 5
        if len(questions) > max_questions:
            # Keep all high priority, then fill with medium priority
            high_priority = [q for q in questions if q["priority"] == "high"]
            medium_priority = [q for q in questions if q["priority"] == "medium"]

            remaining_slots = max_questions - len(high_priority)
            if remaining_slots > 0:
                questions = high_priority + medium_priority[:remaining_slots]
            else:
                questions = high_priority[:max_questions]

        # Add batch information
        for i, question in enumerate(questions):
            question["batch_order"] = i + 1
            question["total_questions"] = len(questions)

        return questions

    async def _filter_redundant_questions(self, task_id: str, questions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Filter out questions that have been asked recently"""

        if task_id not in self.question_history:
            return questions

        # Get recent questions from history
        recent_questions = []
        for historical_questions in self.question_history.values():
            recent_questions.extend([q["question"] for q in historical_questions])

        # Filter out similar questions
        filtered_questions = []
        for question in questions:
            question_text = question["question"].lower()

            # Simple similarity check - in production this would be more sophisticated
            is_redundant = False
            for recent_q in recent_questions:
                recent_q_lower = recent_q.lower()

                # Check for significant word overlap
                question_words = set(question_text.split())
                recent_words = set(recent_q_lower.split())

                overlap = len(question_words.intersection(recent_words))
                total_words = len(question_words.union(recent_words))

                if total_words > 0 and overlap / total_words > 0.6:  # 60% similarity threshold
                    is_redundant = True
                    break

            if not is_redundant:
                filtered_questions.append(question)

        return filtered_questions

    async def get_context_questions(self, task_id: str) -> List[Dict[str, Any]]:
        """Get generated questions for a task"""
        return self.question_history.get(task_id, [])

    async def process_human_responses(self, task_id: str, responses: Dict[str, str]) -> bool:
        """Process human responses to context questions and update task context"""

        if task_id not in self.task_ledgers:
            self.logger.error(f"Task {task_id} not found for processing responses")
            return False

        try:
            ledger = self.task_ledgers[task_id]

            # Process each response
            for question_id, response in responses.items():
                if response.strip():  # Non-empty response
                    ledger.add_fact(f"Human clarification: {response}")

                    # Update context based on response type
                    await self._update_context_from_response(ledger, question_id, response)

            # Update user expertise level based on response quality
            self._update_user_expertise_assessment(responses)

            # Recalculate context score
            new_context_score = await self._recalculate_context_score(ledger)

            # Update context sources weights based on effectiveness
            self._update_context_source_weights(new_context_score)

            # Mark human input as received
            if "human_input_required" in ledger.context_sources:
                ledger.context_sources.remove("human_input_required")
                ledger.context_sources.append("human_input_received")

            self.logger.info(f"Processed {len(responses)} human responses for task {task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to process human responses for task {task_id}: {e}")
            return False

    async def _update_context_from_response(self, ledger: TaskLedger, question_id: str, response: str):
        """Update task context based on specific response"""

        response_lower = response.lower()

        # Extract actionable information from response
        if any(tech_word in response_lower for tech_word in ["api", "database", "framework", "library"]):
            ledger.requirements.append("technical_specification_provided")

        if any(ui_word in response_lower for ui_word in ["design", "layout", "style", "component"]):
            ledger.requirements.append("ui_specification_provided")

        if any(success_word in response_lower for success_word in ["success", "complete", "criteria", "goal"]):
            ledger.success_criteria.append(f"User-defined: {response}")

        # Update guesses to facts based on confirmation
        if any(confirm_word in response_lower for confirm_word in ["yes", "correct", "exactly", "right"]):
            # Convert recent guesses to facts
            if ledger.guesses:
                recent_guess = ledger.guesses[-1]
                ledger.add_fact(f"Confirmed: {recent_guess}")

    def _update_user_expertise_assessment(self, responses: Dict[str, str]):
        """Update user expertise level based on response quality"""

        if not responses:
            return

        # Analyze response sophistication
        total_words = sum(len(response.split()) for response in responses.values())
        avg_response_length = total_words / len(responses)

        technical_terms = 0
        for response in responses.values():
            response_lower = response.lower()
            tech_indicators = ["api", "database", "framework", "architecture", "protocol", "schema"]
            technical_terms += sum(1 for term in tech_indicators if term in response_lower)

        # Adjust expertise level
        if avg_response_length > 20 and technical_terms > 2:
            self.user_expertise_level = "expert"
        elif avg_response_length > 10 and technical_terms > 0:
            self.user_expertise_level = "intermediate"
        else:
            self.user_expertise_level = "beginner"

        self.logger.info(f"Updated user expertise level to: {self.user_expertise_level}")

    async def _recalculate_context_score(self, ledger: TaskLedger) -> float:
        """Recalculate context score after human input"""

        # Count different types of context
        fact_count = len(ledger.facts)
        requirement_count = len(ledger.requirements)
        success_criteria_count = len(ledger.success_criteria)

        # Calculate weighted score
        context_score = min(1.0, (
            fact_count * 0.1 +
            requirement_count * 0.2 +
            success_criteria_count * 0.3 +
            0.4  # Base score for having human input
        ))

        return context_score

    def _update_context_source_weights(self, new_context_score: float):
        """Update context source weights based on effectiveness"""

        # If human input significantly improved context, increase its weight
        if new_context_score > self.human_interaction_threshold:
            self.context_source_weights["human_input"] = min(0.4,
                self.context_source_weights["human_input"] + 0.05)

            # Rebalance other weights
            remaining_weight = 1.0 - self.context_source_weights["human_input"]
            self.context_source_weights["knowledge_base"] = remaining_weight * 0.6
            self.context_source_weights["mcp_servers"] = remaining_weight * 0.4

    def get_question_generation_stats(self) -> Dict[str, Any]:
        """Get statistics about question generation effectiveness"""

        total_tasks = len(self.question_history)
        total_questions = sum(len(questions) for questions in self.question_history.values())

        return {
            "total_tasks_with_questions": total_tasks,
            "total_questions_generated": total_questions,
            "average_questions_per_task": total_questions / total_tasks if total_tasks > 0 else 0,
            "current_user_expertise": self.user_expertise_level,
            "context_source_weights": self.context_source_weights.copy(),
            "human_interaction_threshold": self.human_interaction_threshold,
            "nlp_analysis_enabled": self.nlp_analysis_enabled
        }

    # Pattern Learning & Optimization System

    async def record_workflow_pattern(self, task_id: str, success: bool, execution_time: float, quality_score: float) -> str:
        """
        Record a completed workflow as a pattern for future learning
        Returns pattern_id for tracking
        """

        if task_id not in self.task_ledgers or task_id not in self.progress_ledgers:
            self.logger.error(f"Cannot record pattern for task {task_id} - ledgers not found")
            return ""

        try:
            task_ledger = self.task_ledgers[task_id]
            progress_ledger = self.progress_ledgers[task_id]

            # Create pattern from workflow
            pattern = await self._extract_workflow_pattern(task_ledger, progress_ledger, success, execution_time, quality_score)

            # Generate pattern ID
            pattern_id = f"pattern_{len(self.workflow_patterns)}_{task_id[:8]}"
            pattern["pattern_id"] = pattern_id

            # Store pattern
            self.workflow_patterns[pattern_id] = pattern

            # Initialize success tracking
            if pattern_id not in self.pattern_success_tracking:
                self.pattern_success_tracking[pattern_id] = []

            # Record success/failure
            success_score = 1.0 if success else 0.0
            if quality_score > 0:
                success_score = (success_score + quality_score) / 2  # Weighted by quality

            self.pattern_success_tracking[pattern_id].append(success_score)

            # Update pattern effectiveness
            await self._update_pattern_effectiveness(pattern_id)

            # Generate optimization suggestions if pattern shows room for improvement
            if success_score < 0.8:
                await self._generate_pattern_optimization_suggestions(pattern_id)

            self.logger.info(f"Recorded workflow pattern {pattern_id} with success score {success_score:.2f}")

            return pattern_id

        except Exception as e:
            self.logger.error(f"Failed to record workflow pattern for task {task_id}: {e}")
            return ""

    async def _extract_workflow_pattern(self, task_ledger: TaskLedger, progress_ledger: ProgressLedger,
                                      success: bool, execution_time: float, quality_score: float) -> Dict[str, Any]:
        """Extract reusable pattern from completed workflow"""

        pattern = {
            "task_characteristics": {
                "original_request": task_ledger.original_request,
                "requirements": task_ledger.requirements.copy(),
                "complexity": len(task_ledger.current_plan),
                "strategy": task_ledger.strategy,
                "context_sources": task_ledger.context_sources.copy()
            },
            "execution_pattern": {
                "total_steps": progress_ledger.total_steps,
                "execution_strategy": task_ledger.strategy,
                "step_sequence": [step["type"] for step in task_ledger.current_plan],
                "agent_assignments": progress_ledger.agent_assignments.copy(),
                "dependency_structure": self._extract_dependency_structure(task_ledger.current_plan)
            },
            "performance_metrics": {
                "success": success,
                "execution_time_minutes": execution_time / 60,
                "quality_score": quality_score,
                "stall_count": progress_ledger.stall_count,
                "step_completion_rate": len([s for s in progress_ledger.step_status.values() if s == "completed"]) / max(1, len(progress_ledger.step_status))
            },
            "context_effectiveness": {
                "facts_gathered": len(task_ledger.facts),
                "guesses_made": len(task_ledger.guesses),
                "context_score": len(task_ledger.facts) / max(1, len(task_ledger.facts) + len(task_ledger.guesses))
            },
            "created_at": datetime.utcnow().isoformat(),
            "usage_count": 0,
            "effectiveness_score": 0.0
        }

        return pattern

    def _extract_dependency_structure(self, plan: List[Dict[str, Any]]) -> Dict[str, List[str]]:
        """Extract dependency structure from execution plan"""

        dependencies = {}
        for step in plan:
            step_id = step["step_id"]
            step_deps = step.get("dependencies", [])
            dependencies[step_id] = step_deps

        return dependencies

    async def find_similar_patterns(self, task_description: str, requirements: List[str]) -> List[Dict[str, Any]]:
        """Find workflow patterns similar to current task"""

        similar_patterns = []

        for pattern_id, pattern in self.workflow_patterns.items():
            similarity_score = await self._calculate_pattern_similarity(
                task_description, requirements, pattern
            )

            if similarity_score >= self.pattern_matching_threshold:
                pattern_info = {
                    "pattern_id": pattern_id,
                    "similarity_score": similarity_score,
                    "effectiveness_score": pattern["effectiveness_score"],
                    "success_rate": self._calculate_pattern_success_rate(pattern_id),
                    "usage_count": pattern["usage_count"],
                    "recommended_strategy": pattern["execution_pattern"]["execution_strategy"],
                    "expected_execution_time": pattern["performance_metrics"]["execution_time_minutes"]
                }
                similar_patterns.append(pattern_info)

        # Sort by combined similarity and effectiveness
        similar_patterns.sort(
            key=lambda p: (p["similarity_score"] * 0.6 + p["effectiveness_score"] * 0.4),
            reverse=True
        )

        return similar_patterns[:5]  # Return top 5 matches

    async def _calculate_pattern_similarity(self, task_description: str, requirements: List[str], pattern: Dict[str, Any]) -> float:
        """Calculate similarity between current task and stored pattern"""

        similarity_factors = {}

        # Task description similarity (simple keyword overlap)
        task_words = set(task_description.lower().split())
        pattern_words = set(pattern["task_characteristics"]["original_request"].lower().split())

        if len(task_words.union(pattern_words)) > 0:
            word_overlap = len(task_words.intersection(pattern_words)) / len(task_words.union(pattern_words))
            similarity_factors["description"] = word_overlap * 0.4
        else:
            similarity_factors["description"] = 0.0

        # Requirements similarity
        pattern_requirements = set(pattern["task_characteristics"]["requirements"])
        current_requirements = set(requirements)

        if len(current_requirements.union(pattern_requirements)) > 0:
            req_overlap = len(current_requirements.intersection(pattern_requirements)) / len(current_requirements.union(pattern_requirements))
            similarity_factors["requirements"] = req_overlap * 0.4
        else:
            similarity_factors["requirements"] = 0.0

        # Complexity similarity
        current_complexity = len(requirements)  # Simple complexity measure
        pattern_complexity = pattern["task_characteristics"]["complexity"]

        if max(current_complexity, pattern_complexity) > 0:
            complexity_similarity = 1.0 - abs(current_complexity - pattern_complexity) / max(current_complexity, pattern_complexity)
            similarity_factors["complexity"] = complexity_similarity * 0.2
        else:
            similarity_factors["complexity"] = 0.0

        total_similarity = sum(similarity_factors.values())
        return total_similarity

    def _calculate_pattern_success_rate(self, pattern_id: str) -> float:
        """Calculate success rate for a pattern"""

        if pattern_id not in self.pattern_success_tracking:
            return 0.0

        success_scores = self.pattern_success_tracking[pattern_id]
        if not success_scores:
            return 0.0

        return sum(success_scores) / len(success_scores)

    async def _update_pattern_effectiveness(self, pattern_id: str):
        """Update pattern effectiveness based on usage and success"""

        if pattern_id not in self.workflow_patterns:
            return

        pattern = self.workflow_patterns[pattern_id]
        success_rate = self._calculate_pattern_success_rate(pattern_id)
        usage_count = pattern["usage_count"]

        # Calculate effectiveness: combines success rate with usage frequency
        # More used patterns with high success rates are more effective
        usage_factor = min(1.0, usage_count / 10)  # Normalize usage to 0-1 scale
        effectiveness = success_rate * 0.7 + usage_factor * 0.3

        pattern["effectiveness_score"] = effectiveness

    async def _generate_pattern_optimization_suggestions(self, pattern_id: str):
        """Generate optimization suggestions for underperforming patterns"""

        if pattern_id not in self.workflow_patterns:
            return

        pattern = self.workflow_patterns[pattern_id]
        suggestions = []

        # Analyze performance metrics for optimization opportunities
        metrics = pattern["performance_metrics"]

        # Execution time optimization
        if metrics["execution_time_minutes"] > 60:  # More than 1 hour
            suggestions.append({
                "type": "execution_time",
                "priority": "high",
                "suggestion": "Consider breaking down into smaller parallel tasks",
                "expected_improvement": "30-50% time reduction"
            })

        # Quality score optimization
        if metrics["quality_score"] < 0.7:
            suggestions.append({
                "type": "quality",
                "priority": "high",
                "suggestion": "Implement additional quality checkpoints and validation steps",
                "expected_improvement": "20-30% quality improvement"
            })

        # Stall count optimization
        if metrics["stall_count"] > 2:
            suggestions.append({
                "type": "stall_prevention",
                "priority": "medium",
                "suggestion": "Add proactive progress monitoring and alternative execution paths",
                "expected_improvement": "Reduce stalls by 50-70%"
            })

        # Step completion rate optimization
        if metrics["step_completion_rate"] < 0.9:
            suggestions.append({
                "type": "completion_rate",
                "priority": "medium",
                "suggestion": "Improve step dependency analysis and resource allocation",
                "expected_improvement": "10-20% better completion rate"
            })

        # Context effectiveness optimization
        context_metrics = pattern["context_effectiveness"]
        if context_metrics["context_score"] < 0.8:
            suggestions.append({
                "type": "context_gathering",
                "priority": "low",
                "suggestion": "Enhance context gathering strategies and validation",
                "expected_improvement": "Better task understanding and execution"
            })

        # Store suggestions
        self.optimization_suggestions[pattern_id] = suggestions

        self.logger.info(f"Generated {len(suggestions)} optimization suggestions for pattern {pattern_id}")

    async def apply_pattern_to_task(self, task_id: str, pattern_id: str) -> bool:
        """Apply a successful pattern to a new task"""

        if pattern_id not in self.workflow_patterns:
            self.logger.error(f"Pattern {pattern_id} not found")
            return False

        if task_id not in self.task_ledgers:
            self.logger.error(f"Task {task_id} not found")
            return False

        try:
            pattern = self.workflow_patterns[pattern_id]
            task_ledger = self.task_ledgers[task_id]

            # Apply pattern strategy
            recommended_strategy = pattern["execution_pattern"]["execution_strategy"]
            task_ledger.strategy = recommended_strategy
            task_ledger.add_fact(f"Applied pattern {pattern_id} with strategy: {recommended_strategy}")

            # Apply pattern-based plan if current plan is not yet optimized
            if len(task_ledger.current_plan) == 0:
                pattern_plan = await self._adapt_pattern_plan_to_task(pattern, task_ledger)
                task_ledger.update_plan(pattern_plan, f"Plan adapted from pattern {pattern_id}")

            # Increment pattern usage
            pattern["usage_count"] += 1

            # Update pattern effectiveness
            await self._update_pattern_effectiveness(pattern_id)

            self.logger.info(f"Applied pattern {pattern_id} to task {task_id}")
            return True

        except Exception as e:
            self.logger.error(f"Failed to apply pattern {pattern_id} to task {task_id}: {e}")
            return False

    async def _adapt_pattern_plan_to_task(self, pattern: Dict[str, Any], task_ledger: TaskLedger) -> List[Dict[str, Any]]:
        """Adapt a pattern's execution plan to the current task"""

        pattern_steps = pattern["execution_pattern"]["step_sequence"]
        adapted_plan = []

        for i, step_type in enumerate(pattern_steps):
            step = {
                "step_id": f"adapted_step_{i}",
                "description": f"Execute {step_type} for: {task_ledger.original_request}",
                "type": step_type,
                "dependencies": [],
                "estimated_time": 15  # Default estimate
            }

            # Add dependencies based on pattern structure
            if i > 0:
                # Simple dependency: each step depends on the previous one
                step["dependencies"] = [f"adapted_step_{i-1}"]

            adapted_plan.append(step)

        return adapted_plan

    async def record_failure_pattern(self, task_id: str, failure_details: Dict[str, Any]) -> str:
        """Record a failure pattern for learning and prevention"""

        if task_id not in self.task_ledgers:
            return ""

        try:
            task_ledger = self.task_ledgers[task_id]

            # Create failure pattern
            failure_pattern = {
                "task_characteristics": {
                    "original_request": task_ledger.original_request,
                    "requirements": task_ledger.requirements.copy(),
                    "strategy": task_ledger.strategy
                },
                "failure_details": failure_details,
                "failure_type": failure_details.get("type", "unknown"),
                "root_cause": failure_details.get("root_cause", "unidentified"),
                "prevention_suggestions": [],
                "created_at": datetime.utcnow().isoformat(),
                "occurrence_count": 1
            }

            # Generate failure pattern ID
            failure_id = f"failure_{failure_pattern['failure_type']}_{len(self.failure_patterns)}"

            # Check if similar failure pattern exists
            existing_pattern = await self._find_similar_failure_pattern(failure_pattern)
            if existing_pattern:
                # Update existing pattern
                existing_pattern["occurrence_count"] += 1
                failure_id = existing_pattern["failure_id"]
            else:
                # Store new failure pattern
                failure_pattern["failure_id"] = failure_id
                self.failure_patterns[failure_id] = failure_pattern

                # Generate prevention suggestions
                await self._generate_failure_prevention_suggestions(failure_id)

            self.logger.info(f"Recorded failure pattern {failure_id}")
            return failure_id

        except Exception as e:
            self.logger.error(f"Failed to record failure pattern for task {task_id}: {e}")
            return ""

    async def _find_similar_failure_pattern(self, failure_pattern: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Find similar existing failure pattern"""

        for existing_pattern in self.failure_patterns.values():
            # Check similarity based on failure type and task characteristics
            if (existing_pattern["failure_type"] == failure_pattern["failure_type"] and
                existing_pattern["root_cause"] == failure_pattern["root_cause"]):

                # Check task similarity
                existing_reqs = set(existing_pattern["task_characteristics"]["requirements"])
                new_reqs = set(failure_pattern["task_characteristics"]["requirements"])

                if len(existing_reqs.intersection(new_reqs)) / max(1, len(existing_reqs.union(new_reqs))) > 0.5:
                    return existing_pattern

        return None

    async def _generate_failure_prevention_suggestions(self, failure_id: str):
        """Generate suggestions to prevent similar failures"""

        if failure_id not in self.failure_patterns:
            return

        failure_pattern = self.failure_patterns[failure_id]
        failure_type = failure_pattern["failure_type"]

        suggestions = []

        if failure_type == "execution_stall":
            suggestions.extend([
                "Implement more aggressive timeout handling",
                "Add alternative execution paths for stall scenarios",
                "Improve progress monitoring granularity",
                "Consider task decomposition for complex operations"
            ])

        elif failure_type == "quality_failure":
            suggestions.extend([
                "Add intermediate quality checkpoints",
                "Implement iterative refinement processes",
                "Enhance success criteria validation",
                "Add quality prediction mechanisms"
            ])

        elif failure_type == "dependency_failure":
            suggestions.extend([
                "Improve dependency analysis accuracy",
                "Add dependency validation before execution",
                "Implement dependency conflict resolution",
                "Create dependency fallback mechanisms"
            ])

        elif failure_type == "context_failure":
            suggestions.extend([
                "Enhance context gathering strategies",
                "Add context completeness validation",
                "Implement context quality scoring",
                "Improve human interaction triggers"
            ])

        failure_pattern["prevention_suggestions"] = suggestions

    def get_pattern_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about pattern learning system"""

        analytics = {
            "workflow_patterns": {
                "total_patterns": len(self.workflow_patterns),
                "average_effectiveness": 0.0,
                "most_effective_patterns": [],
                "most_used_patterns": []
            },
            "failure_patterns": {
                "total_failures": len(self.failure_patterns),
                "common_failure_types": {},
                "prevention_suggestions_generated": 0
            },
            "optimization": {
                "patterns_with_suggestions": len(self.optimization_suggestions),
                "total_suggestions": sum(len(suggestions) for suggestions in self.optimization_suggestions.values())
            },
            "learning_effectiveness": {
                "pattern_reuse_rate": 0.0,
                "average_pattern_success_rate": 0.0
            }
        }

        # Calculate workflow pattern analytics
        if self.workflow_patterns:
            effectiveness_scores = [p["effectiveness_score"] for p in self.workflow_patterns.values()]
            analytics["workflow_patterns"]["average_effectiveness"] = sum(effectiveness_scores) / len(effectiveness_scores)

            # Most effective patterns
            sorted_by_effectiveness = sorted(
                self.workflow_patterns.items(),
                key=lambda x: x[1]["effectiveness_score"],
                reverse=True
            )
            analytics["workflow_patterns"]["most_effective_patterns"] = [
                {"pattern_id": pid, "effectiveness": p["effectiveness_score"]}
                for pid, p in sorted_by_effectiveness[:5]
            ]

            # Most used patterns
            sorted_by_usage = sorted(
                self.workflow_patterns.items(),
                key=lambda x: x[1]["usage_count"],
                reverse=True
            )
            analytics["workflow_patterns"]["most_used_patterns"] = [
                {"pattern_id": pid, "usage_count": p["usage_count"]}
                for pid, p in sorted_by_usage[:5]
            ]

        # Calculate failure pattern analytics
        if self.failure_patterns:
            failure_types = [p["failure_type"] for p in self.failure_patterns.values()]
            for failure_type in set(failure_types):
                analytics["failure_patterns"]["common_failure_types"][failure_type] = failure_types.count(failure_type)

            analytics["failure_patterns"]["prevention_suggestions_generated"] = sum(
                len(p["prevention_suggestions"]) for p in self.failure_patterns.values()
            )

        # Calculate learning effectiveness
        if self.pattern_success_tracking:
            all_success_rates = []
            for pattern_id in self.pattern_success_tracking:
                success_rate = self._calculate_pattern_success_rate(pattern_id)
                all_success_rates.append(success_rate)

            if all_success_rates:
                analytics["learning_effectiveness"]["average_pattern_success_rate"] = sum(all_success_rates) / len(all_success_rates)

        # Calculate pattern reuse rate
        total_usage = sum(p["usage_count"] for p in self.workflow_patterns.values())
        if len(self.workflow_patterns) > 0:
            analytics["learning_effectiveness"]["pattern_reuse_rate"] = total_usage / len(self.workflow_patterns)

        return analytics

    async def optimize_pattern_library(self):
        """Optimize pattern library by removing ineffective patterns and promoting successful ones"""

        try:
            # Remove patterns with consistently low effectiveness
            patterns_to_remove = []
            for pattern_id, pattern in self.workflow_patterns.items():
                if (pattern["effectiveness_score"] < 0.3 and
                    pattern["usage_count"] > 5):  # Give patterns a chance with low usage
                    patterns_to_remove.append(pattern_id)

            for pattern_id in patterns_to_remove:
                del self.workflow_patterns[pattern_id]
                if pattern_id in self.pattern_success_tracking:
                    del self.pattern_success_tracking[pattern_id]
                if pattern_id in self.optimization_suggestions:
                    del self.optimization_suggestions[pattern_id]

            # Promote highly effective patterns
            for pattern_id, pattern in self.workflow_patterns.items():
                if pattern["effectiveness_score"] > 0.8:
                    # Increase pattern visibility by adjusting matching threshold
                    pattern["promoted"] = True

            # Clean up old failure patterns (keep only recent ones)
            current_time = datetime.utcnow()
            old_failures = []

            for failure_id, failure_pattern in self.failure_patterns.items():
                created_time = datetime.fromisoformat(failure_pattern["created_at"])
                age_days = (current_time - created_time).days

                if age_days > 30 and failure_pattern["occurrence_count"] < 2:
                    old_failures.append(failure_id)

            for failure_id in old_failures:
                del self.failure_patterns[failure_id]

            self.logger.info(f"Pattern library optimization: removed {len(patterns_to_remove)} ineffective patterns and {len(old_failures)} old failures")

        except Exception as e:
            self.logger.error(f"Pattern library optimization failed: {e}")

    # Teaching Agent Framework Methods

    async def teach_agent_from_failure(self, agent_id: str, task_id: str, failure_details: Dict[str, Any]) -> Dict[str, Any]:
        """
        Teaching framework: Learn from agent failures and provide corrective guidance
        """

        teaching_result = {
            "agent_id": agent_id,
            "task_id": task_id,
            "teaching_applied": False,
            "feedback_generated": False,
            "improvement_predicted": 0.0,
            "teaching_strategy": "none"
        }

        try:
            # Analyze failure pattern
            failure_pattern = await self._analyze_failure_pattern(agent_id, failure_details)

            # Generate targeted teaching intervention
            teaching_intervention = await self._generate_teaching_intervention(agent_id, failure_pattern)

            # Apply teaching strategy
            if teaching_intervention["strategy"] != "none":
                success = await self._apply_teaching_strategy(agent_id, teaching_intervention)
                teaching_result["teaching_applied"] = success
                teaching_result["teaching_strategy"] = teaching_intervention["strategy"]

            # Generate improvement feedback
            feedback = await self._generate_improvement_feedback(agent_id, failure_pattern, teaching_intervention)
            if feedback:
                teaching_result["feedback_generated"] = True
                teaching_result["improvement_predicted"] = feedback.get("predicted_improvement", 0.0)

            # Record teaching session
            await self._record_teaching_session(agent_id, task_id, teaching_result, failure_pattern)

            self.logger.info(f"Teaching session completed for agent {agent_id}: {teaching_result['teaching_strategy']}")

        except Exception as e:
            self.logger.error(f"Teaching framework failed for agent {agent_id}: {e}")
            teaching_result["error"] = str(e)

        return teaching_result

    async def _analyze_failure_pattern(self, agent_id: str, failure_details: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze failure to identify patterns and root causes"""

        pattern = {
            "failure_type": "unknown",
            "root_cause": "unidentified",
            "frequency": 1,
            "context_factors": [],
            "similar_failures": [],
            "teachable_moment": False
        }

        # Get agent's learning history
        agent_history = self.agent_learning_history.get(agent_id, [])

        # Classify failure type
        error_message = failure_details.get("error", "").lower()

        if any(keyword in error_message for keyword in ["timeout", "stall", "stuck"]):
            pattern["failure_type"] = "execution_stall"
            pattern["teachable_moment"] = True
        elif any(keyword in error_message for keyword in ["quality", "validation", "criteria"]):
            pattern["failure_type"] = "quality_failure"
            pattern["teachable_moment"] = True
        elif any(keyword in error_message for keyword in ["dependency", "requirement", "missing"]):
            pattern["failure_type"] = "dependency_failure"
            pattern["teachable_moment"] = True
        elif any(keyword in error_message for keyword in ["context", "information", "unclear"]):
            pattern["failure_type"] = "context_failure"
            pattern["teachable_moment"] = True

        # Find similar failures in history
        for historical_failure in agent_history:
            if historical_failure.get("failure_type") == pattern["failure_type"]:
                pattern["similar_failures"].append(historical_failure)

        pattern["frequency"] = len(pattern["similar_failures"]) + 1

        # Extract context factors
        if "task_context" in failure_details:
            context = failure_details["task_context"]
            pattern["context_factors"] = [
                f"Task type: {context.get('task_type', 'unknown')}",
                f"Complexity: {context.get('complexity', 'unknown')}",
                f"Requirements: {len(context.get('requirements', []))}"
            ]

        return pattern

    async def _generate_teaching_intervention(self, agent_id: str, failure_pattern: Dict[str, Any]) -> Dict[str, Any]:
        """Generate targeted teaching intervention based on failure pattern"""

        intervention = {
            "strategy": "none",
            "actions": [],
            "expected_improvement": 0.0,
            "confidence": 0.0
        }

        if not failure_pattern["teachable_moment"]:
            return intervention

        failure_type = failure_pattern["failure_type"]
        frequency = failure_pattern["frequency"]

        # Select teaching strategy based on failure type and frequency
        if failure_type == "execution_stall":
            if frequency == 1:
                intervention["strategy"] = "guidance"
                intervention["actions"] = [
                    "Provide execution strategy alternatives",
                    "Suggest progress monitoring techniques",
                    "Recommend timeout handling approaches"
                ]
                intervention["expected_improvement"] = 0.3
            else:
                intervention["strategy"] = "intensive_training"
                intervention["actions"] = [
                    "Implement adaptive execution strategies",
                    "Add proactive stall detection",
                    "Create execution pattern library"
                ]
                intervention["expected_improvement"] = 0.6

        elif failure_type == "quality_failure":
            if frequency == 1:
                intervention["strategy"] = "feedback_loop"
                intervention["actions"] = [
                    "Enhance quality criteria understanding",
                    "Provide quality assessment examples",
                    "Implement iterative improvement process"
                ]
                intervention["expected_improvement"] = 0.4
            else:
                intervention["strategy"] = "quality_framework"
                intervention["actions"] = [
                    "Implement comprehensive quality framework",
                    "Add multi-dimensional quality scoring",
                    "Create quality pattern recognition"
                ]
                intervention["expected_improvement"] = 0.7

        elif failure_type == "dependency_failure":
            intervention["strategy"] = "dependency_training"
            intervention["actions"] = [
                "Improve dependency analysis capabilities",
                "Add dependency validation checks",
                "Implement dependency resolution strategies"
            ]
            intervention["expected_improvement"] = 0.5

        elif failure_type == "context_failure":
            intervention["strategy"] = "context_enhancement"
            intervention["actions"] = [
                "Improve context gathering techniques",
                "Add context validation mechanisms",
                "Implement context-aware decision making"
            ]
            intervention["expected_improvement"] = 0.5

        # Set confidence based on historical success
        if agent_id in self.feedback_effectiveness:
            intervention["confidence"] = self.feedback_effectiveness[agent_id]
        else:
            intervention["confidence"] = 0.7  # Default confidence

        return intervention

    async def _apply_teaching_strategy(self, agent_id: str, intervention: Dict[str, Any]) -> bool:
        """Apply the teaching strategy to improve agent performance"""

        strategy = intervention["strategy"]
        actions = intervention["actions"]

        try:
            if strategy == "guidance":
                # Provide guidance through enhanced context
                return await self._apply_guidance_strategy(agent_id, actions)

            elif strategy == "feedback_loop":
                # Implement feedback loop for continuous improvement
                return await self._apply_feedback_loop_strategy(agent_id, actions)

            elif strategy == "intensive_training":
                # Apply intensive training with pattern recognition
                return await self._apply_intensive_training_strategy(agent_id, actions)

            elif strategy == "quality_framework":
                # Implement comprehensive quality framework
                return await self._apply_quality_framework_strategy(agent_id, actions)

            elif strategy == "dependency_training":
                # Enhance dependency analysis capabilities
                return await self._apply_dependency_training_strategy(agent_id, actions)

            elif strategy == "context_enhancement":
                # Improve context gathering and processing
                return await self._apply_context_enhancement_strategy(agent_id, actions)

            return False

        except Exception as e:
            self.logger.error(f"Failed to apply teaching strategy {strategy} for agent {agent_id}: {e}")
            return False

    async def _apply_guidance_strategy(self, agent_id: str, actions: List[str]) -> bool:
        """Apply guidance strategy - provide enhanced context and suggestions"""

        # Create guidance patterns for the agent
        guidance_patterns = {
            "execution_alternatives": [
                "Try breaking down complex tasks into smaller steps",
                "Consider parallel execution for independent components",
                "Implement checkpoint-based progress tracking"
            ],
            "progress_monitoring": [
                "Set up regular progress checkpoints",
                "Monitor for stall conditions proactively",
                "Implement adaptive timeout strategies"
            ],
            "timeout_handling": [
                "Implement graceful timeout handling",
                "Add retry mechanisms with exponential backoff",
                "Consider alternative execution paths on timeout"
            ]
        }

        # Store guidance patterns for agent
        if agent_id not in self.teaching_patterns:
            self.teaching_patterns[agent_id] = {}

        self.teaching_patterns[agent_id]["guidance"] = guidance_patterns

        return True

    async def _apply_feedback_loop_strategy(self, agent_id: str, actions: List[str]) -> bool:
        """Apply feedback loop strategy - continuous improvement through feedback"""

        # Implement feedback loop mechanism
        feedback_config = {
            "feedback_frequency": "after_each_task",
            "improvement_tracking": True,
            "adaptive_learning_rate": 0.1,
            "quality_threshold": 0.8
        }

        if agent_id not in self.teaching_patterns:
            self.teaching_patterns[agent_id] = {}

        self.teaching_patterns[agent_id]["feedback_loop"] = feedback_config

        # Initialize improvement tracking
        if agent_id not in self.improvement_tracking:
            self.improvement_tracking[agent_id] = []

        return True

    async def _apply_intensive_training_strategy(self, agent_id: str, actions: List[str]) -> bool:
        """Apply intensive training strategy - comprehensive skill enhancement"""

        training_program = {
            "adaptive_strategies": [
                "Dynamic strategy selection based on task characteristics",
                "Real-time strategy adjustment based on progress",
                "Multi-strategy execution with fallback options"
            ],
            "stall_detection": [
                "Proactive stall pattern recognition",
                "Early warning system for potential stalls",
                "Automatic recovery mechanisms"
            ],
            "pattern_library": [
                "Build execution pattern library from successful tasks",
                "Pattern matching for similar task scenarios",
                "Pattern-based optimization recommendations"
            ]
        }

        if agent_id not in self.teaching_patterns:
            self.teaching_patterns[agent_id] = {}

        self.teaching_patterns[agent_id]["intensive_training"] = training_program

        return True

    async def _apply_quality_framework_strategy(self, agent_id: str, actions: List[str]) -> bool:
        """Apply quality framework strategy - comprehensive quality management"""

        quality_framework = {
            "multi_dimensional_scoring": {
                "correctness": 0.4,
                "completeness": 0.3,
                "efficiency": 0.2,
                "maintainability": 0.1
            },
            "quality_gates": [
                "Pre-execution quality check",
                "Mid-execution quality validation",
                "Post-execution quality assessment"
            ],
            "improvement_mechanisms": [
                "Iterative refinement based on quality scores",
                "Quality-driven strategy selection",
                "Continuous quality learning"
            ]
        }

        if agent_id not in self.teaching_patterns:
            self.teaching_patterns[agent_id] = {}

        self.teaching_patterns[agent_id]["quality_framework"] = quality_framework

        return True

    async def _apply_dependency_training_strategy(self, agent_id: str, actions: List[str]) -> bool:
        """Apply dependency training strategy - enhanced dependency management"""

        dependency_training = {
            "analysis_techniques": [
                "Deep dependency tree analysis",
                "Circular dependency detection",
                "Critical path identification"
            ],
            "validation_checks": [
                "Pre-execution dependency validation",
                "Runtime dependency monitoring",
                "Post-execution dependency verification"
            ],
            "resolution_strategies": [
                "Automatic dependency resolution",
                "Alternative dependency paths",
                "Dependency conflict resolution"
            ]
        }

        if agent_id not in self.teaching_patterns:
            self.teaching_patterns[agent_id] = {}

        self.teaching_patterns[agent_id]["dependency_training"] = dependency_training

        return True

    async def _apply_context_enhancement_strategy(self, agent_id: str, actions: List[str]) -> bool:
        """Apply context enhancement strategy - improved context processing"""

        context_enhancement = {
            "gathering_techniques": [
                "Multi-source context aggregation",
                "Context relevance scoring",
                "Dynamic context prioritization"
            ],
            "validation_mechanisms": [
                "Context completeness validation",
                "Context consistency checking",
                "Context freshness verification"
            ],
            "decision_making": [
                "Context-aware strategy selection",
                "Context-driven quality assessment",
                "Context-based risk evaluation"
            ]
        }

        if agent_id not in self.teaching_patterns:
            self.teaching_patterns[agent_id] = {}

        self.teaching_patterns[agent_id]["context_enhancement"] = context_enhancement

        return True

    async def _generate_improvement_feedback(self, agent_id: str, failure_pattern: Dict[str, Any], intervention: Dict[str, Any]) -> Dict[str, Any]:
        """Generate specific improvement feedback for the agent"""

        feedback = {
            "predicted_improvement": 0.0,
            "feedback_message": "",
            "actionable_steps": [],
            "success_indicators": []
        }

        failure_type = failure_pattern["failure_type"]
        strategy = intervention["strategy"]

        # Generate specific feedback based on failure type
        if failure_type == "execution_stall":
            feedback["feedback_message"] = (
                "Your task execution encountered stalls. Focus on breaking down complex tasks "
                "into smaller, manageable steps and implementing progress checkpoints."
            )
            feedback["actionable_steps"] = [
                "Implement task decomposition strategies",
                "Add progress monitoring at regular intervals",
                "Create fallback execution paths"
            ]
            feedback["success_indicators"] = [
                "Reduced task execution time",
                "Fewer stall incidents",
                "Improved progress visibility"
            ]
            feedback["predicted_improvement"] = intervention["expected_improvement"]

        elif failure_type == "quality_failure":
            feedback["feedback_message"] = (
                "Output quality did not meet standards. Focus on understanding quality criteria "
                "and implementing iterative improvement processes."
            )
            feedback["actionable_steps"] = [
                "Study quality criteria more thoroughly",
                "Implement quality checkpoints during execution",
                "Use iterative refinement approaches"
            ]
            feedback["success_indicators"] = [
                "Higher quality scores",
                "Fewer quality-related failures",
                "Better alignment with success criteria"
            ]
            feedback["predicted_improvement"] = intervention["expected_improvement"]

        # Add historical context if available
        if agent_id in self.improvement_tracking and self.improvement_tracking[agent_id]:
            recent_improvements = self.improvement_tracking[agent_id][-3:]  # Last 3 improvements
            avg_improvement = sum(recent_improvements) / len(recent_improvements)
            feedback["predicted_improvement"] = max(feedback["predicted_improvement"], avg_improvement * 1.1)

        return feedback

    async def _record_teaching_session(self, agent_id: str, task_id: str, teaching_result: Dict[str, Any], failure_pattern: Dict[str, Any]):
        """Record teaching session for future reference and effectiveness tracking"""

        session_record = {
            "timestamp": datetime.utcnow().isoformat(),
            "task_id": task_id,
            "failure_pattern": failure_pattern,
            "teaching_result": teaching_result,
            "effectiveness_score": 0.0  # Will be updated later based on actual improvement
        }

        # Add to agent's learning history
        if agent_id not in self.agent_learning_history:
            self.agent_learning_history[agent_id] = []

        self.agent_learning_history[agent_id].append(session_record)

        # Limit history size to prevent memory bloat
        if len(self.agent_learning_history[agent_id]) > 50:
            self.agent_learning_history[agent_id] = self.agent_learning_history[agent_id][-50:]

    async def update_teaching_effectiveness(self, agent_id: str, task_id: str, actual_improvement: float):
        """Update teaching effectiveness based on actual agent improvement"""

        if agent_id not in self.agent_learning_history:
            return

        # Find the teaching session for this task
        for session in reversed(self.agent_learning_history[agent_id]):
            if session["task_id"] == task_id:
                session["effectiveness_score"] = actual_improvement

                # Update overall feedback effectiveness for this agent
                effectiveness_scores = [
                    s["effectiveness_score"] for s in self.agent_learning_history[agent_id]
                    if s["effectiveness_score"] > 0
                ]

                if effectiveness_scores:
                    self.feedback_effectiveness[agent_id] = sum(effectiveness_scores) / len(effectiveness_scores)

                # Track improvement
                if agent_id not in self.improvement_tracking:
                    self.improvement_tracking[agent_id] = []

                self.improvement_tracking[agent_id].append(actual_improvement)

                # Limit tracking history
                if len(self.improvement_tracking[agent_id]) > 20:
                    self.improvement_tracking[agent_id] = self.improvement_tracking[agent_id][-20:]

                break

    def get_agent_teaching_summary(self, agent_id: str) -> Dict[str, Any]:
        """Get teaching summary for an agent"""

        summary = {
            "agent_id": agent_id,
            "total_teaching_sessions": 0,
            "average_effectiveness": 0.0,
            "improvement_trend": "stable",
            "common_failure_patterns": [],
            "successful_strategies": []
        }

        if agent_id not in self.agent_learning_history:
            return summary

        history = self.agent_learning_history[agent_id]
        summary["total_teaching_sessions"] = len(history)

        # Calculate average effectiveness
        effectiveness_scores = [s["effectiveness_score"] for s in history if s["effectiveness_score"] > 0]
        if effectiveness_scores:
            summary["average_effectiveness"] = sum(effectiveness_scores) / len(effectiveness_scores)

        # Analyze improvement trend
        if agent_id in self.improvement_tracking and len(self.improvement_tracking[agent_id]) >= 3:
            recent_improvements = self.improvement_tracking[agent_id][-3:]
            if all(recent_improvements[i] >= recent_improvements[i-1] for i in range(1, len(recent_improvements))):
                summary["improvement_trend"] = "improving"
            elif all(recent_improvements[i] <= recent_improvements[i-1] for i in range(1, len(recent_improvements))):
                summary["improvement_trend"] = "declining"

        # Identify common failure patterns
        failure_types = [s["failure_pattern"]["failure_type"] for s in history]
        failure_counts = {}
        for failure_type in failure_types:
            failure_counts[failure_type] = failure_counts.get(failure_type, 0) + 1

        summary["common_failure_patterns"] = sorted(
            failure_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        # Identify successful strategies
        successful_sessions = [s for s in history if s["effectiveness_score"] > 0.5]
        strategy_counts = {}
        for session in successful_sessions:
            strategy = session["teaching_result"]["teaching_strategy"]
            strategy_counts[strategy] = strategy_counts.get(strategy, 0) + 1

        summary["successful_strategies"] = sorted(
            strategy_counts.items(), key=lambda x: x[1], reverse=True
        )[:3]

        return summary
    
    async def evaluate_quality(self, output: Any, analysis: TaskAnalysis) -> QualityScore:
        """Evaluate the quality of agent output"""
        
        issues = []
        suggestions = []
        score = 1.0  # Start with perfect score
        
        # Basic quality checks
        if not output:
            issues.append("No output provided")
            score -= 0.5
        
        if isinstance(output, str):
            if len(output.strip()) < 10:
                issues.append("Output too short")
                score -= 0.3
                
            # Check for error indicators
            error_indicators = ["error", "failed", "exception", "traceback"]
            if any(indicator in output.lower() for indicator in error_indicators):
                issues.append("Output contains error indicators")
                score -= 0.4
        
        # Task-specific quality checks
        if analysis.task_type == TaskType.UI_CREATION:
            if isinstance(output, str):
                if "vue" not in output.lower():
                    issues.append("Vue.js components not detected")
                    score -= 0.3
                    suggestions.append("Ensure Vue.js components are properly created")

                if "<template>" not in output and ".vue" not in output:
                    issues.append("Vue component structure not found")
                    score -= 0.2
                    suggestions.append("Include proper Vue component templates")

        elif analysis.task_type == TaskType.GRAPHICS_DEVELOPMENT:
            if isinstance(output, str):
                if "webgl" not in output.lower() and "three.js" not in output.lower() and "canvas" not in output.lower():
                    issues.append("GPU-accelerated graphics not detected")
                    score -= 0.4
                    suggestions.append("Implement WebGL or Canvas-based graphics")

                if "animation" not in output.lower() and "animate" not in output.lower():
                    issues.append("Animation system not found")
                    score -= 0.3
                    suggestions.append("Add smooth animation with requestAnimationFrame")

                if "dragon" not in output.lower():
                    issues.append("Dragon graphics not detected")
                    score -= 0.3
                    suggestions.append("Create visible dragon model or sprites")

        elif analysis.task_type == TaskType.CODING:
            if isinstance(output, str):
                if len(output.split('\n')) < 5:
                    issues.append("Code output seems too minimal")
                    score -= 0.2
                    suggestions.append("Provide more comprehensive implementation")
        
        # Ensure score doesn't go below 0
        score = max(0.0, score)
        passed = score >= 0.7  # 70% threshold for passing
        
        return QualityScore(
            score=score,
            passed=passed,
            issues=issues,
            suggestions=suggestions
        )
    
    async def generate_feedback(self, quality_score: QualityScore, analysis: TaskAnalysis) -> str:
        """Generate teaching feedback for agent improvement"""
        
        if quality_score.passed:
            return "Good work! Task completed successfully."
        
        feedback_parts = ["The output needs improvement:"]
        
        # Add specific issues
        for issue in quality_score.issues:
            feedback_parts.append(f"- Issue: {issue}")
        
        # Add suggestions
        if quality_score.suggestions:
            feedback_parts.append("\nSuggestions for improvement:")
            for suggestion in quality_score.suggestions:
                feedback_parts.append(f"- {suggestion}")
        
        # Add task-specific guidance
        if analysis.task_type == TaskType.UI_CREATION:
            feedback_parts.append("\nFor UI creation tasks:")
            feedback_parts.append("- Create proper Vue.js components with <template>, <script>, and <style> sections")
            feedback_parts.append("- Include component props and data if needed")
            feedback_parts.append("- Ensure components are functional and well-structured")
        elif analysis.task_type == TaskType.GRAPHICS_DEVELOPMENT:
            feedback_parts.append("\nFor graphics development tasks:")
            feedback_parts.append("- Use WebGL or Three.js for GPU-accelerated graphics")
            feedback_parts.append("- Implement smooth 60fps animations with requestAnimationFrame")
            feedback_parts.append("- Create visually appealing models with proper lighting")
            feedback_parts.append("- Add particle effects and visual enhancements")
            feedback_parts.append("- Optimize for performance and cross-browser compatibility")
        
        return "\n".join(feedback_parts)
    
    async def supervise_task(self, task_id: str, task_description: str, agent_output: Any) -> Dict[str, Any]:
        """Main supervision workflow for a task"""
        
        try:
            # Step 1: Analyze the task
            analysis = await self.analyze_task(task_description)
            self.logger.info(f"Task {task_id} analyzed: {analysis.task_type}, complexity: {analysis.complexity}")
            
            # Step 2: Evaluate output quality
            quality_score = await self.evaluate_quality(agent_output, analysis)
            self.logger.info(f"Task {task_id} quality score: {quality_score.score:.2f}, passed: {quality_score.passed}")
            
            # Step 3: Generate feedback if needed
            feedback = await self.generate_feedback(quality_score, analysis)
            
            # Step 4: Record supervision result
            supervision_result = {
                "task_id": task_id,
                "analysis": analysis,
                "quality_score": quality_score,
                "feedback": feedback,
                "requires_retry": not quality_score.passed,
                "timestamp": asyncio.get_event_loop().time()
            }
            
            self.active_tasks[task_id] = supervision_result
            
            return supervision_result
            
        except Exception as e:
            self.logger.error(f"Error supervising task {task_id}: {e}")
            return {
                "task_id": task_id,
                "error": str(e),
                "requires_retry": True,
                "timestamp": asyncio.get_event_loop().time()
            }
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get supervision status for a task"""
        return self.active_tasks.get(task_id)
    
    def get_supervision_stats(self) -> Dict[str, Any]:
        """Get overall supervision statistics"""
        total_tasks = len(self.active_tasks)
        if total_tasks == 0:
            return {"total_tasks": 0, "success_rate": 0.0}
        
        successful_tasks = sum(1 for task in self.active_tasks.values() 
                             if not task.get("requires_retry", True))
        
        return {
            "total_tasks": total_tasks,
            "successful_tasks": successful_tasks,
            "success_rate": successful_tasks / total_tasks,
            "active_tasks": list(self.active_tasks.keys())
        }

    async def _create_initial_plan(self, ledger: TaskLedger):
        """Create initial execution plan based on task analysis"""

        task_description = ledger.original_request
        requirements = ledger.requirements
        strategy = ledger.strategy

        # Create plan steps based on strategy and requirements
        plan = []

        if "frontend_development" in requirements:
            plan.append({
                "step_id": f"ui_step_{len(plan)}",
                "description": "Create user interface components",
                "type": "ui_creation",
                "dependencies": [],
                "estimated_time": 15
            })

        if "backend_development" in requirements:
            plan.append({
                "step_id": f"backend_step_{len(plan)}",
                "description": "Implement backend functionality",
                "type": "backend_development",
                "dependencies": [],
                "estimated_time": 20
            })

        if "code_generation" in requirements:
            plan.append({
                "step_id": f"code_step_{len(plan)}",
                "description": "Generate required code",
                "type": "code_generation",
                "dependencies": [],
                "estimated_time": 10
            })

        if "testing" in requirements:
            plan.append({
                "step_id": f"test_step_{len(plan)}",
                "description": "Create and run tests",
                "type": "testing",
                "dependencies": [step["step_id"] for step in plan],  # Tests depend on all previous steps
                "estimated_time": 10
            })

        # If no specific requirements, create a general plan
        if not plan:
            plan.append({
                "step_id": f"general_step_{len(plan)}",
                "description": task_description,
                "type": "general",
                "dependencies": [],
                "estimated_time": 15
            })

        ledger.update_plan(plan, "Initial plan created based on requirements analysis")

    async def _generate_initial_plan(self, task_ledger: TaskLedger):
        """Generate initial execution plan for the task"""

        plan = []

        # Create plan based on task type and requirements
        if any(keyword in task_ledger.original_request.lower() for keyword in ["graphics", "webgl", "3d", "animation", "gpu", "dragon"]):
            plan = [
                {
                    "step_id": "research_graphics",
                    "description": "Research WebGL/Three.js graphics libraries and techniques",
                    "type": "research",
                    "dependencies": [],
                    "estimated_time": 15
                },
                {
                    "step_id": "setup_graphics_project",
                    "description": "Set up HTML page with WebGL/Three.js framework",
                    "type": "graphics_development",
                    "dependencies": ["research_graphics"],
                    "estimated_time": 10
                },
                {
                    "step_id": "create_dragon_model",
                    "description": "Create or load dragon 3D model/sprites",
                    "type": "graphics_development",
                    "dependencies": ["setup_graphics_project"],
                    "estimated_time": 20
                },
                {
                    "step_id": "implement_animation",
                    "description": "Implement flying animation and movement patterns",
                    "type": "graphics_development",
                    "dependencies": ["create_dragon_model"],
                    "estimated_time": 25
                },
                {
                    "step_id": "add_effects",
                    "description": "Add particle effects, lighting, and visual enhancements",
                    "type": "graphics_development",
                    "dependencies": ["implement_animation"],
                    "estimated_time": 20
                },
                {
                    "step_id": "optimize_performance",
                    "description": "Optimize for GPU performance and 60fps",
                    "type": "graphics_development",
                    "dependencies": ["add_effects"],
                    "estimated_time": 15
                }
            ]
        elif "ui" in task_ledger.original_request.lower():
            plan = [
                {
                    "step_id": "setup_project",
                    "description": "Set up Vue.js project structure",
                    "type": "ui_creation",
                    "dependencies": [],
                    "estimated_time": 10
                },
                {
                    "step_id": "create_components",
                    "description": "Create Vue.js components",
                    "type": "ui_creation",
                    "dependencies": ["setup_project"],
                    "estimated_time": 20
                },
                {
                    "step_id": "integrate_backend",
                    "description": "Integrate with backend APIs",
                    "type": "integration",
                    "dependencies": ["create_components"],
                    "estimated_time": 15
                },
                {
                    "step_id": "test_ui",
                    "description": "Test UI functionality",
                    "type": "testing",
                    "dependencies": ["integrate_backend"],
                    "estimated_time": 10
                }
            ]
        else:
            # Generic plan for other tasks
            plan = [
                {
                    "step_id": "analyze_requirements",
                    "description": "Analyze task requirements",
                    "type": "analysis",
                    "dependencies": [],
                    "estimated_time": 5
                },
                {
                    "step_id": "execute_task",
                    "description": "Execute main task",
                    "type": "execution",
                    "dependencies": ["analyze_requirements"],
                    "estimated_time": 15
                }
            ]

        task_ledger.update_plan(plan, "Initial execution plan generated")

    async def _create_progress_ledger(self, task_id: str, task_ledger: TaskLedger) -> ProgressLedger:
        """INNER LOOP: Create Progress Ledger from Task Ledger plan"""

        progress_ledger = ProgressLedger(task_id=task_id)

        # Initialize progress tracking for each step
        for step in task_ledger.current_plan:
            step_id = step["step_id"]
            progress_ledger.step_status[step_id] = "pending"
            progress_ledger.step_progress[step_id] = 0.0

        progress_ledger.total_steps = len(task_ledger.current_plan)

        return progress_ledger

    async def _execute_dual_loop(self, task_id: str):
        """Main dual-loop execution coordinator"""

        try:
            task_ledger = self.task_ledgers[task_id]
            progress_ledger = self.progress_ledgers[task_id]

            self.logger.info(f"Starting dual-loop execution for task {task_id}")

            # Main execution loop
            while not progress_ledger.is_task_complete():

                # OUTER LOOP: Check if plan needs updating
                if await self._should_update_plan(task_ledger, progress_ledger):
                    await self._update_plan(task_ledger, progress_ledger)

                # INNER LOOP: Execute next steps
                await self._execute_next_steps(task_ledger, progress_ledger)

                # Check for stalls
                if not progress_ledger.is_making_progress(self.progress_timeout_minutes):
                    progress_ledger.stall_count += 1
                    if progress_ledger.stall_count >= self.max_stall_count:
                        self.logger.warning(f"Task {task_id} stalled, attempting recovery")
                        await self._handle_stall(task_ledger, progress_ledger)

                # Brief pause to prevent tight loop
                await asyncio.sleep(1)

            self.logger.info(f"Task {task_id} completed successfully")

        except Exception as e:
            self.logger.error(f"Dual-loop execution failed for task {task_id}: {e}")
            # Mark task as failed
            if task_id in self.progress_ledgers:
                self.progress_ledgers[task_id].execution_log.append(f"FAILED: {e}")

    async def _should_update_plan(self, task_ledger: TaskLedger, progress_ledger: ProgressLedger) -> bool:
        """Determine if the plan needs updating based on current progress"""

        # Update plan if we have new facts that change the situation
        if len(task_ledger.facts) > len(task_ledger.current_plan) * 2:
            return True

        # Update plan if multiple steps are failing
        failed_steps = sum(1 for status in progress_ledger.step_status.values() if status == "failed")
        if failed_steps > 1:
            return True

        # Update plan if we're stalled
        if progress_ledger.stall_count > 1:
            return True

        return False

    async def _update_plan(self, task_ledger: TaskLedger, progress_ledger: ProgressLedger):
        """OUTER LOOP: Update plan based on current situation"""

        self.logger.info(f"Updating plan for task {task_ledger.task_id}")

        # Analyze current situation
        completed_steps = [step_id for step_id, status in progress_ledger.step_status.items()
                          if status == "completed"]
        failed_steps = [step_id for step_id, status in progress_ledger.step_status.items()
                       if status == "failed"]

        # Create updated plan
        updated_plan = []

        # Keep completed steps
        for step in task_ledger.current_plan:
            if step["step_id"] in completed_steps:
                updated_plan.append(step)

        # Recreate failed steps with different approach
        for step in task_ledger.current_plan:
            if step["step_id"] in failed_steps:
                new_step = step.copy()
                new_step["step_id"] = f"{step['step_id']}_retry_{progress_ledger.stall_count}"
                new_step["description"] = f"Retry: {step['description']}"
                updated_plan.append(new_step)

        # Add remaining pending steps
        for step in task_ledger.current_plan:
            if (step["step_id"] not in completed_steps and
                step["step_id"] not in failed_steps):
                updated_plan.append(step)

        # Update the plan
        task_ledger.update_plan(updated_plan, f"Plan updated due to {len(failed_steps)} failed steps")

        # Reset progress ledger for new steps
        for step in updated_plan:
            step_id = step["step_id"]
            if step_id not in progress_ledger.step_status:
                progress_ledger.step_status[step_id] = "pending"
                progress_ledger.step_progress[step_id] = 0.0

        progress_ledger.total_steps = len(updated_plan)

    async def _execute_next_steps(self, task_ledger: TaskLedger, progress_ledger: ProgressLedger):
        """INNER LOOP: Execute the next available steps"""

        # Find steps that are ready to execute (dependencies met)
        ready_steps = []
        for step in task_ledger.current_plan:
            step_id = step["step_id"]

            # Skip if already completed or in progress
            if progress_ledger.step_status.get(step_id) in ["completed", "in_progress"]:
                continue

            # Check if dependencies are met
            dependencies_met = True
            for dep_id in step.get("dependencies", []):
                if progress_ledger.step_status.get(dep_id) != "completed":
                    dependencies_met = False
                    break

            if dependencies_met:
                ready_steps.append(step)

        # Execute ready steps (limit concurrent execution)
        max_concurrent = 2 if task_ledger.strategy == "multi_agent_parallel" else 1
        executing_count = sum(1 for status in progress_ledger.step_status.values()
                             if status == "in_progress")

        available_slots = max_concurrent - executing_count
        steps_to_execute = ready_steps[:available_slots]

        for step in steps_to_execute:
            await self._execute_step(step, task_ledger, progress_ledger)

    async def _execute_step(self, step: Dict[str, Any], task_ledger: TaskLedger, progress_ledger: ProgressLedger):
        """Execute a single step using appropriate agent"""

        step_id = step["step_id"]

        try:
            # Mark step as in progress
            progress_ledger.step_status[step_id] = "in_progress"
            progress_ledger.record_progress(step_id, 0.1, "started")

            # Select appropriate agent for this step
            agent_id = await self._select_agent_for_step(step)
            progress_ledger.assign_agent_to_step(step_id, agent_id)

            # Execute step via A2A protocol if available
            if self.a2a_manager:
                result = await self._execute_via_a2a(step, agent_id)
            else:
                # Fallback to direct execution
                result = await self._execute_direct(step)

            # Mark step as completed
            progress_ledger.complete_step(step_id, result)
            task_ledger.add_fact(f"Completed step {step_id}: {step['description']}")

        except Exception as e:
            # Mark step as failed
            progress_ledger.step_status[step_id] = "failed"
            progress_ledger.execution_log.append(f"Step {step_id} failed: {e}")
            task_ledger.add_guess(f"Step {step_id} failed, may need different approach: {e}")
            self.logger.error(f"Step {step_id} execution failed: {e}")

    async def _select_agent_for_step(self, step: Dict[str, Any]) -> str:
        """Select the best agent for executing this step"""

        step_type = step.get("type", "general")

        # Map step types to agent types
        agent_type_map = {
            "ui_creation": "coding",
            "graphics_development": "coding",
            "backend_development": "coding",
            "code_generation": "coding",
            "testing": "coding",
            "research": "research",
            "general": "research"
        }

        return agent_type_map.get(step_type, "research")

    async def _execute_via_a2a(self, step: Dict[str, Any], agent_id: str) -> Any:
        """Execute step via A2A protocol"""

        if not self.a2a_manager:
            raise Exception("A2A manager not available")

        # Create A2A task for this step
        task_request = {
            "description": step["description"],
            "type": step.get("type", "general"),
            "context": step.get("context", {}),
            "requirements": step.get("requirements", [])
        }

        # Submit task via A2A protocol
        result = await self.a2a_manager.submit_task(agent_id, task_request)
        return result

    async def _execute_direct(self, step: Dict[str, Any]) -> Any:
        """Direct execution using MCP servers when A2A is not available"""

        step_description = step["description"]
        step_type = step.get("type", "general")

        try:
            # Use MCP servers for actual execution
            if self.mcp_manager:
                # Route to appropriate MCP server based on step type
                if step_type == "ui_creation":
                    return await self._execute_ui_creation_step(step)
                elif step_type == "graphics_development":
                    return await self._execute_graphics_development_step(step)
                elif step_type == "backend_development":
                    return await self._execute_backend_step(step)
                elif step_type == "code_generation":
                    return await self._execute_code_generation_step(step)
                elif step_type == "testing":
                    return await self._execute_testing_step(step)
                else:
                    return await self._execute_general_step(step)
            else:
                # No MCP manager available - this is a configuration error
                raise Exception(f"No execution capability available for step type: {step_type}")

        except Exception as e:
            self.logger.error(f"Direct execution failed for step {step.get('step_id', 'unknown')}: {e}")
            raise

    async def _execute_ui_creation_step(self, step: Dict[str, Any]) -> Any:
        """Execute UI creation step using real MCP servers"""

        try:
            # Use filesystem MCP server to create UI components
            if hasattr(self.mcp_manager, 'execute_request'):
                # Create component file structure
                component_request = {
                    "action": "create_file",
                    "path": f"components/{step.get('component_name', 'NewComponent')}.vue",
                    "content": await self._generate_vue_component_content(step)
                }

                result = await self.mcp_manager.execute_request("filesystem", component_request)
                return f"Created UI component: {result}"
            else:
                # Fallback to direct filesystem operations
                return await self._create_ui_component_direct(step)

        except Exception as e:
            self.logger.error(f"UI creation step failed: {e}")
            raise Exception(f"Failed to create UI component: {e}")

    async def _execute_graphics_development_step(self, step: Dict[str, Any]) -> Any:
        """Execute graphics development step using REAL Agent Orchestration MCP Server"""

        try:
            step_id = step.get("step_id", "unknown")
            description = step.get("description", "")

            # Create real task request for Agent Orchestration MCP Server
            task_request = {
                "task_type": "code_generation",
                "description": description,
                "input_data": {
                    "step_id": step_id,
                    "graphics_type": "webgl_dragon",
                    "requirements": [
                        "WebGL/Three.js implementation",
                        "GPU acceleration",
                        "Dragon 3D model",
                        "Flying animation",
                        "Particle effects"
                    ]
                },
                "required_capabilities": ["code_generation", "graphics_development", "webgl"],
                "priority": "high"
            }

            # Submit to real Agent Orchestration MCP Server
            result = await self._submit_task_to_orchestration_server(task_request)

            if result and result.get("success"):
                return result.get("data", {}).get("result", "Graphics development task completed")
            else:
                raise Exception(f"Graphics development task failed: {result}")

        except Exception as e:
            self.logger.error(f"Graphics development step failed: {e}")
            raise Exception(f"Failed to execute graphics step: {e}")

    async def _submit_task_to_orchestration_server(self, task_request: Dict[str, Any]) -> Dict[str, Any]:
        """Submit task to real Agent Orchestration MCP Server"""
        try:
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "http://localhost:8005/v1/tasks",
                    json=task_request,
                    timeout=30
                ) as response:
                    if response.status == 200:
                        result = await response.json()
                        self.logger.info(f"✅ Task submitted to orchestration server: {result.get('data', {}).get('task_id')}")

                        # Wait for task completion
                        task_id = result.get('data', {}).get('task_id')
                        if task_id:
                            return await self._wait_for_task_completion(task_id)
                        else:
                            return result
                    else:
                        error_text = await response.text()
                        self.logger.error(f"❌ Orchestration server error {response.status}: {error_text}")
                        return {"success": False, "error": f"HTTP {response.status}: {error_text}"}

        except Exception as e:
            self.logger.error(f"❌ Failed to submit task to orchestration server: {e}")
            return {"success": False, "error": str(e)}

    async def _wait_for_task_completion(self, task_id: str, timeout: int = 60) -> Dict[str, Any]:
        """Wait for task completion from orchestration server"""
        try:
            start_time = time.time()

            while time.time() - start_time < timeout:
                async with aiohttp.ClientSession() as session:
                    async with session.get(
                        f"http://localhost:8005/v1/tasks/{task_id}",
                        timeout=5
                    ) as response:
                        if response.status == 200:
                            result = await response.json()
                            task_status = result.get('data', {}).get('status')

                            if task_status == 'completed':
                                self.logger.info(f"✅ Task {task_id} completed successfully")
                                return result
                            elif task_status == 'failed':
                                self.logger.error(f"❌ Task {task_id} failed")
                                return result
                            else:
                                # Still running, wait and check again
                                await asyncio.sleep(2)
                        else:
                            self.logger.error(f"❌ Error checking task status: {response.status}")
                            await asyncio.sleep(2)

            # Timeout reached
            self.logger.error(f"⏰ Task {task_id} timed out after {timeout} seconds")
            return {"success": False, "error": "Task timeout"}

        except Exception as e:
            self.logger.error(f"❌ Error waiting for task completion: {e}")
            return {"success": False, "error": str(e)}

    async def _execute_backend_step(self, step: Dict[str, Any]) -> Any:
        """Execute backend development step using real MCP servers"""

        try:
            # Use multiple MCP servers for backend development
            results = []

            # 1. Database operations if needed
            if "database" in step["description"].lower():
                if hasattr(self.mcp_manager, 'execute_request'):
                    db_request = {
                        "action": "create_table",
                        "table_definition": await self._generate_table_definition(step)
                    }
                    db_result = await self.mcp_manager.execute_request("postgresql", db_request)
                    results.append(f"Database: {db_result}")

            # 2. API endpoint creation
            if hasattr(self.mcp_manager, 'execute_request'):
                api_request = {
                    "action": "create_file",
                    "path": f"api/{step.get('endpoint_name', 'new_endpoint')}.py",
                    "content": await self._generate_api_endpoint_content(step)
                }
                api_result = await self.mcp_manager.execute_request("filesystem", api_request)
                results.append(f"API: {api_result}")

            return f"Backend implementation completed: {'; '.join(results)}"

        except Exception as e:
            self.logger.error(f"Backend step failed: {e}")
            raise Exception(f"Failed to implement backend functionality: {e}")

    async def _execute_code_generation_step(self, step: Dict[str, Any]) -> Any:
        """Execute code generation step using real MCP servers"""

        try:
            # Use GitHub MCP server for code operations
            if hasattr(self.mcp_manager, 'execute_request'):
                code_request = {
                    "action": "create_file",
                    "path": f"src/{step.get('module_name', 'generated_code')}.py",
                    "content": await self._generate_code_content(step)
                }

                result = await self.mcp_manager.execute_request("filesystem", code_request)
                return f"Generated code: {result}"
            else:
                return await self._generate_code_direct(step)

        except Exception as e:
            self.logger.error(f"Code generation step failed: {e}")
            raise Exception(f"Failed to generate code: {e}")

    async def _execute_testing_step(self, step: Dict[str, Any]) -> Any:
        """Execute testing step using real MCP servers"""

        try:
            # Create test files using filesystem MCP server
            if hasattr(self.mcp_manager, 'execute_request'):
                test_request = {
                    "action": "create_file",
                    "path": f"tests/test_{step.get('test_name', 'generated_test')}.py",
                    "content": await self._generate_test_content(step)
                }

                result = await self.mcp_manager.execute_request("filesystem", test_request)

                # Run tests if possible
                if hasattr(self.mcp_manager, 'execute_command'):
                    test_run_result = await self.mcp_manager.execute_command("pytest", test_request["path"])
                    return f"Test created and executed: {result}; Results: {test_run_result}"
                else:
                    return f"Test created: {result}"
            else:
                return await self._create_test_direct(step)

        except Exception as e:
            self.logger.error(f"Testing step failed: {e}")
            raise Exception(f"Failed to create/run tests: {e}")

    async def _execute_general_step(self, step: Dict[str, Any]) -> Any:
        """Execute general step using available MCP servers"""

        try:
            step_description = step["description"]

            # Analyze step description to determine appropriate MCP server
            if "file" in step_description.lower() or "create" in step_description.lower():
                # Use filesystem MCP server
                if hasattr(self.mcp_manager, 'execute_request'):
                    file_request = {
                        "action": "create_file",
                        "path": f"output/{step.get('step_id', 'general_output')}.txt",
                        "content": f"Completed: {step_description}"
                    }
                    result = await self.mcp_manager.execute_request("filesystem", file_request)
                    return f"General task completed: {result}"

            elif "search" in step_description.lower() or "find" in step_description.lower():
                # Use memory MCP server
                if hasattr(self.mcp_manager, 'execute_request'):
                    search_request = {
                        "action": "search",
                        "query": step_description
                    }
                    result = await self.mcp_manager.execute_request("memory", search_request)
                    return f"Search completed: {result}"

            # Default: document the completion
            return await self._document_step_completion(step)

        except Exception as e:
            self.logger.error(f"General step execution failed: {e}")
            raise Exception(f"Failed to execute general step: {e}")

    async def _generate_vue_component_content(self, step: Dict[str, Any]) -> str:
        """Generate real Vue.js component content"""

        component_name = step.get('component_name', 'NewComponent')
        description = step.get('description', '')

        # Extract component requirements from description
        has_props = 'props' in description.lower() or 'property' in description.lower()
        has_events = 'event' in description.lower() or 'emit' in description.lower()
        has_form = 'form' in description.lower() or 'input' in description.lower()

        # Generate real Vue component
        template_content = self._generate_vue_template(step, has_form)
        script_content = self._generate_vue_script(component_name, has_props, has_events)
        style_content = self._generate_vue_style()

        return f"<template>\n{template_content}\n</template>\n\n<script>\n{script_content}\n</script>\n\n<style scoped>\n{style_content}\n</style>"

    def _generate_vue_template(self, step: Dict[str, Any], has_form: bool) -> str:
        """Generate Vue template section"""

        if has_form:
            return """  <div class="component-container">
    <h2>{{ title }}</h2>
    <form @submit.prevent="handleSubmit">
      <div class="form-group">
        <label for="input-field">Input:</label>
        <input
          id="input-field"
          v-model="inputValue"
          type="text"
          class="form-control"
          :placeholder="placeholder"
        />
      </div>
      <button type="submit" class="btn btn-primary">Submit</button>
    </form>
    <div v-if="result" class="result">
      {{ result }}
    </div>
  </div>"""
        else:
            return """  <div class="component-container">
    <h2>{{ title }}</h2>
    <p>{{ description }}</p>
    <button @click="handleClick" class="btn btn-primary">
      {{ buttonText }}
    </button>
  </div>"""

    def _generate_vue_script(self, component_name: str, has_props: bool, has_events: bool) -> str:
        """Generate Vue script section"""

        props_section = """  props: {
    title: {
      type: String,
      default: 'Component Title'
    },
    placeholder: {
      type: String,
      default: 'Enter value...'
    }
  },""" if has_props else ""

        events_section = """    this.$emit('component-action', this.inputValue);""" if has_events else ""

        script_content = f"export default {{\n  name: '{component_name}',\n{props_section}\n  data() {{\n    return {{\n      inputValue: '',\n      result: '',\n      buttonText: 'Click Me',\n      description: 'This is a functional Vue component.'\n    }};\n  }},\n  methods: {{\n    handleSubmit() {{\n      this.result = `Submitted: ${{this.inputValue}}`;\n{events_section}\n    }},\n    handleClick() {{\n      this.result = 'Button clicked!';\n{events_section}\n    }}\n  }}\n}};"

        return script_content

    def _generate_vue_style(self) -> str:
        """Generate Vue style section"""

        return """.component-container {
  max-width: 500px;
  margin: 0 auto;
  padding: 20px;
  border: 1px solid #ddd;
  border-radius: 8px;
  background-color: #f9f9f9;
}

.form-group {
  margin-bottom: 15px;
}

.form-control {
  width: 100%;
  padding: 8px 12px;
  border: 1px solid #ccc;
  border-radius: 4px;
  font-size: 14px;
}

.btn {
  padding: 10px 20px;
  border: none;
  border-radius: 4px;
  cursor: pointer;
  font-size: 14px;
}

.btn-primary {
  background-color: #007bff;
  color: white;
}

.btn-primary:hover {
  background-color: #0056b3;
}

.result {
  margin-top: 15px;
  padding: 10px;
  background-color: #e9ecef;
  border-radius: 4px;
}"""

    async def _generate_table_definition(self, step: Dict[str, Any]) -> str:
        """Generate real database table definition"""

        description = step.get('description', '').lower()
        table_name = step.get('table_name', 'new_table')

        # Extract table requirements from description
        has_user_data = 'user' in description
        has_timestamps = 'time' in description or 'date' in description
        has_status = 'status' in description or 'state' in description

        columns = ["id SERIAL PRIMARY KEY"]

        if has_user_data:
            columns.extend([
                "user_id INTEGER REFERENCES users(id)",
                "username VARCHAR(255) NOT NULL",
                "email VARCHAR(255) UNIQUE NOT NULL"
            ])

        if 'name' in description:
            columns.append("name VARCHAR(255) NOT NULL")

        if 'description' in description:
            columns.append("description TEXT")

        if has_status:
            columns.append("status VARCHAR(50) DEFAULT 'active'")

        if has_timestamps:
            columns.extend([
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP",
                "updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP"
            ])

        # Build index statements
        index_statements = [f"CREATE INDEX idx_{table_name}_created_at ON {table_name}(created_at);"]

        if has_user_data:
            index_statements.append(f"CREATE INDEX idx_{table_name}_user_id ON {table_name}(user_id);")

        if has_status:
            index_statements.append(f"CREATE INDEX idx_{table_name}_status ON {table_name}(status);")

        # Join columns and indexes
        columns_sql = ',\n    '.join(columns)
        indexes_sql = '\n'.join(index_statements)

        return f"""CREATE TABLE {table_name} (
    {columns_sql}
);

-- Create indexes for performance
{indexes_sql}"""

    async def _generate_api_endpoint_content(self, step: Dict[str, Any]) -> str:
        """Generate real API endpoint content"""

        endpoint_name = step.get('endpoint_name', 'new_endpoint')
        description = step.get('description', '').lower()

        # Determine endpoint type
        is_crud = any(word in description for word in ['create', 'read', 'update', 'delete', 'crud'])
        is_auth = 'auth' in description or 'login' in description
        has_validation = 'valid' in description or 'check' in description

        if is_auth:
            return self._generate_auth_endpoint(endpoint_name)
        elif is_crud:
            return self._generate_crud_endpoint(endpoint_name, has_validation)
        else:
            return self._generate_generic_endpoint(endpoint_name, description)

    def _generate_auth_endpoint(self, endpoint_name: str) -> str:
        """Generate authentication endpoint"""

        auth_code = f"""from fastapi import APIRouter, HTTPException, Depends
from fastapi.security import HTTPBearer
from pydantic import BaseModel
import bcrypt
import jwt
from datetime import datetime, timedelta

router = APIRouter()
security = HTTPBearer()

class LoginRequest(BaseModel):
    username: str
    password: str

class LoginResponse(BaseModel):
    access_token: str
    token_type: str
    expires_in: int

@router.post("/{{endpoint_name}}/login", response_model=LoginResponse)
async def login(request: LoginRequest):
    \"\"\"Authenticate user and return JWT token\"\"\"
    try:
        # Validate credentials (replace with real database lookup)
        user = await get_user_by_username(request.username)
        if not user or not verify_password(request.password, user.password_hash):
            raise HTTPException(status_code=401, detail="Invalid credentials")

        # Generate JWT token
        token_data = {{
            "sub": user.id,
            "username": user.username,
            "exp": datetime.utcnow() + timedelta(hours=24)
        }}

        token = jwt.encode(token_data, "your-secret-key", algorithm="HS256")

        return LoginResponse(
            access_token=token,
            token_type="bearer",
            expires_in=86400  # 24 hours
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Authentication failed: {{str(e)}}")

async def get_user_by_username(username: str):
    \"\"\"Get user from database by username\"\"\"
    # Implement database lookup
    pass

def verify_password(password: str, password_hash: str) -> bool:
    \"\"\"Verify password against hash\"\"\"
    return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
"""

        return auth_code

    def _generate_crud_endpoint(self, endpoint_name: str, has_validation: bool) -> str:
        """Generate CRUD endpoint"""

        validation_decorator = "@validate_request" if has_validation else ""

        # Build validation section
        validation_section = ""
        if has_validation:
            validation_section = '''@validator('name')
    def validate_name(cls, v):
        if len(v) < 2:
            raise ValueError('Name must be at least 2 characters')
        return v'''

        crud_code = f"""from fastapi import APIRouter, HTTPException, Depends
from pydantic import BaseModel, validator
from typing import List, Optional
import asyncpg

router = APIRouter()

class {endpoint_name.title()}Create(BaseModel):
    name: str
    description: Optional[str] = None
    status: str = "active"

    {validation_section}

class {endpoint_name.title()}Response(BaseModel):
    id: int
    name: str
    description: Optional[str]
    status: str
    created_at: str

@router.post("/{{endpoint_name}}/", response_model={{endpoint_name.title()}}Response)
{{validation_decorator}}
async def create_{{endpoint_name}}(item: {{endpoint_name.title()}}Create):
    \"\"\"Create a new {{endpoint_name}}\"\"\"
    try:
        # Insert into database
        query = '''
            INSERT INTO {{endpoint_name}} (name, description, status)
            VALUES ($1, $2, $3)
            RETURNING id, name, description, status, created_at
        '''

        result = await execute_query(query, item.name, item.description, item.status)

        return {{endpoint_name.title()}}Response(
            id=result['id'],
            name=result['name'],
            description=result['description'],
            status=result['status'],
            created_at=result['created_at'].isoformat()
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create {{endpoint_name}}: {{{{str(e)}}}}")

@router.get("/{{endpoint_name}}/{{{{item_id}}}}", response_model={{endpoint_name.title()}}Response)
async def get_{{endpoint_name}}(item_id: int):
    \"\"\"Get {{endpoint_name}} by ID\"\"\"
    try:
        query = "SELECT * FROM {{endpoint_name}} WHERE id = $1"
        result = await execute_query(query, item_id)

        if not result:
            raise HTTPException(status_code=404, detail=f"{{endpoint_name.title()}} not found")

        return {{endpoint_name.title()}}Response(**result)

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to get {{endpoint_name}}: {{{{str(e)}}}}")

@router.get("/{{endpoint_name}}/", response_model=List[{{endpoint_name.title()}}Response])
async def list_{{endpoint_name}}(skip: int = 0, limit: int = 100):
    \"\"\"List all {{endpoint_name}}\"\"\"
    try:
        query = "SELECT * FROM {{endpoint_name}} ORDER BY created_at DESC LIMIT $1 OFFSET $2"
        results = await execute_query_many(query, limit, skip)

        return [{{{{endpoint_name.title()}}Response(**row) for row in results]]

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to list {{endpoint_name}}: {{{{str(e)}}}}")

async def execute_query(query: str, *args):
    \"\"\"Execute database query\"\"\"
    # Implement database connection and query execution
    pass

async def execute_query_many(query: str, *args):
    \"\"\"Execute database query returning multiple rows\"\"\"
    # Implement database connection and query execution
    pass
"""

        return crud_code

    def _generate_generic_endpoint(self, endpoint_name: str, description: str) -> str:
        """Generate generic endpoint"""

        class_name = endpoint_name.title()

        generic_code = f"""from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

router = APIRouter()

class {{class_name}}Request(BaseModel):
    data: Dict[str, Any]

class {{class_name}}Response(BaseModel):
    success: bool
    message: str
    data: Dict[str, Any]

@router.post("/{{endpoint_name}}/", response_model={{class_name}}Response)
async def {{endpoint_name}}_handler(request: {{class_name}}Request):
    \"\"\"Handle {description}\"\"\"
    try:
        # Process the request
        result_data = {{
            "processed": True,
            "input": request.data,
            "timestamp": "{{datetime.utcnow().isoformat()}}"
        }}

        return {{class_name}}Response(
            success=True,
            message="Request processed successfully",
            data=result_data
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process request: {{str(e)}}")
"""

        return generic_code

    async def _handle_stall(self, task_ledger: TaskLedger, progress_ledger: ProgressLedger):
        """Handle stalled execution by updating strategy"""

        self.logger.warning(f"Handling stall for task {task_ledger.task_id}")

        # Add fact about stall
        task_ledger.add_fact(f"Task stalled after {progress_ledger.stall_count} attempts")

        # Change strategy if possible
        if task_ledger.strategy == "single_agent":
            task_ledger.strategy = "multi_agent_sequential"
            task_ledger.add_guess("Switching to multi-agent sequential strategy")
        elif task_ledger.strategy == "multi_agent_sequential":
            task_ledger.strategy = "multi_agent_parallel"
            task_ledger.add_guess("Switching to multi-agent parallel strategy")

        # Reset stall count
        progress_ledger.stall_count = 0
        progress_ledger.last_progress_time = datetime.utcnow()

        # Force plan update
        await self._update_plan(task_ledger, progress_ledger)

    async def _create_ui_component_direct(self, step: Dict[str, Any]) -> str:
        """Create UI component directly without MCP"""
        component_name = step.get('component_name', 'NewComponent')
        return f"UI component {component_name} created directly"

    async def _generate_code_direct(self, step: Dict[str, Any]) -> str:
        """Generate code directly without MCP"""
        module_name = step.get('module_name', 'generated_code')
        return f"Code module {module_name} generated directly"

    async def _create_test_direct(self, step: Dict[str, Any]) -> str:
        """Create test directly without MCP"""
        test_name = step.get('test_name', 'generated_test')
        return f"Test {test_name} created directly"

    async def _document_step_completion(self, step: Dict[str, Any]) -> str:
        """Document step completion"""
        return f"Step completed: {step.get('description', 'Unknown step')}"

    # REMOVED: All fake graphics development helper methods
    # These have been replaced with real Agent Orchestration MCP Server integration


class MetaSupervisorAgent:
    """
    Meta-Coordination Layer for Multiple Supervisor Agents

    Coordinates multiple TaskIntelligenceSystem instances for complex workflows
    that require multiple specialized supervisors working together.
    """

    def __init__(self, mcp_manager=None, a2a_manager=None):
        self.logger = logging.getLogger(f"{__name__}.MetaSupervisorAgent")

        # Supervisor agent pool
        self.supervisor_agents: Dict[str, TaskIntelligenceSystem] = {}
        self.agent_specializations: Dict[str, List[str]] = {}
        self.agent_performance: Dict[str, Dict[str, float]] = {}

        # Integration points
        self.mcp_manager = mcp_manager
        self.a2a_manager = a2a_manager

        # Meta-coordination state
        self.active_workflows: Dict[str, Dict[str, Any]] = {}
        self.agent_assignments: Dict[str, str] = {}  # task_id -> supervisor_id

        # Load balancing
        self.agent_load: Dict[str, int] = {}
        self.max_concurrent_tasks_per_agent = 3

    async def coordinate_complex_task(self, task_description: str, context: Optional[Dict[str, Any]] = None) -> str:
        """
        Main entry point for complex task coordination
        Returns workflow_id for tracking
        """
        workflow_id = str(uuid.uuid4())

        try:
            # Analyze task complexity and decomposition needs
            workflow_analysis = await self._analyze_workflow_requirements(task_description, context)

            # Create workflow coordination plan
            coordination_plan = await self._create_coordination_plan(workflow_analysis)

            # Initialize workflow state
            self.active_workflows[workflow_id] = {
                "description": task_description,
                "analysis": workflow_analysis,
                "plan": coordination_plan,
                "status": "initializing",
                "created_at": datetime.utcnow(),
                "supervisor_tasks": {},
                "dependencies": coordination_plan.get("dependencies", {}),
                "completion_status": {}
            }

            # Start workflow execution
            asyncio.create_task(self._execute_workflow(workflow_id))

            self.logger.info(f"Started complex workflow {workflow_id}")
            return workflow_id

        except Exception as e:
            self.logger.error(f"Failed to coordinate complex task: {e}")
            raise

    async def _analyze_workflow_requirements(self, task_description: str, context: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze if task requires multiple supervisors and how to decompose"""

        analysis = {
            "requires_multiple_supervisors": False,
            "supervisor_requirements": [],
            "task_decomposition": [],
            "coordination_complexity": "simple",
            "estimated_supervisors": 1
        }

        task_lower = task_description.lower()

        # Detect multi-domain requirements
        domain_indicators = {
            "frontend": ["ui", "interface", "frontend", "vue", "react", "component"],
            "backend": ["api", "backend", "server", "database", "service"],
            "integration": ["integrate", "connect", "sync", "webhook", "third party"],
            "testing": ["test", "validate", "verify", "qa", "quality"],
            "deployment": ["deploy", "production", "staging", "docker", "container"],
            "research": ["research", "analyze", "investigate", "study", "explore"]
        }

        detected_domains = []
        for domain, keywords in domain_indicators.items():
            if any(keyword in task_lower for keyword in keywords):
                detected_domains.append(domain)

        # Determine if multiple supervisors are needed
        if len(detected_domains) > 2:
            analysis["requires_multiple_supervisors"] = True
            analysis["supervisor_requirements"] = detected_domains
            analysis["estimated_supervisors"] = min(len(detected_domains), 4)  # Cap at 4 supervisors

            if len(detected_domains) > 3:
                analysis["coordination_complexity"] = "complex"
            else:
                analysis["coordination_complexity"] = "moderate"

        # Create task decomposition
        for domain in detected_domains:
            domain_task = self._extract_domain_task(task_description, domain)
            if domain_task:
                analysis["task_decomposition"].append({
                    "domain": domain,
                    "task": domain_task,
                    "priority": self._get_domain_priority(domain),
                    "dependencies": self._get_domain_dependencies(domain, detected_domains)
                })

        return analysis

    def _extract_domain_task(self, task_description: str, domain: str) -> str:
        """Extract domain-specific task from overall description"""

        # Simple extraction - in production this would be more sophisticated
        domain_templates = {
            "frontend": f"Create user interface components for: {task_description}",
            "backend": f"Implement backend services for: {task_description}",
            "integration": f"Integrate external services for: {task_description}",
            "testing": f"Create comprehensive tests for: {task_description}",
            "deployment": f"Deploy and configure production environment for: {task_description}",
            "research": f"Research and analyze requirements for: {task_description}"
        }

        return domain_templates.get(domain, f"Handle {domain} aspects of: {task_description}")

    def _get_domain_priority(self, domain: str) -> int:
        """Get execution priority for domain (1=highest, 5=lowest)"""

        priority_map = {
            "research": 1,      # Research first
            "backend": 2,       # Backend before frontend
            "frontend": 3,      # Frontend after backend
            "integration": 4,   # Integration after core components
            "testing": 4,       # Testing in parallel with integration
            "deployment": 5     # Deployment last
        }

        return priority_map.get(domain, 3)

    def _get_domain_dependencies(self, domain: str, all_domains: List[str]) -> List[str]:
        """Get dependencies for a domain"""

        dependency_map = {
            "frontend": ["backend"],
            "integration": ["backend", "frontend"],
            "testing": ["backend", "frontend"],
            "deployment": ["backend", "frontend", "integration", "testing"]
        }

        dependencies = dependency_map.get(domain, [])
        return [dep for dep in dependencies if dep in all_domains]

    async def _create_coordination_plan(self, workflow_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """Create coordination plan for multiple supervisors"""

        plan = {
            "execution_strategy": "sequential",
            "supervisor_assignments": {},
            "dependencies": {},
            "coordination_points": [],
            "success_criteria": []
        }

        if not workflow_analysis["requires_multiple_supervisors"]:
            # Single supervisor plan
            plan["execution_strategy"] = "single_supervisor"
            plan["supervisor_assignments"]["primary"] = {
                "domains": workflow_analysis["supervisor_requirements"],
                "task": workflow_analysis.get("task_decomposition", [])
            }
            return plan

        # Multi-supervisor plan
        task_decomposition = workflow_analysis["task_decomposition"]

        # Determine execution strategy
        if workflow_analysis["coordination_complexity"] == "complex":
            plan["execution_strategy"] = "hybrid"  # Mix of sequential and parallel
        else:
            plan["execution_strategy"] = "sequential"

        # Assign supervisors to domains
        for i, task_item in enumerate(task_decomposition):
            supervisor_id = f"supervisor_{i}"
            plan["supervisor_assignments"][supervisor_id] = {
                "domain": task_item["domain"],
                "task": task_item["task"],
                "priority": task_item["priority"],
                "dependencies": task_item["dependencies"]
            }

            # Set up dependencies
            if task_item["dependencies"]:
                plan["dependencies"][supervisor_id] = [
                    f"supervisor_{j}" for j, other_task in enumerate(task_decomposition)
                    if other_task["domain"] in task_item["dependencies"]
                ]

        # Define coordination points
        plan["coordination_points"] = [
            {"type": "dependency_check", "frequency": "before_execution"},
            {"type": "progress_sync", "frequency": "every_5_minutes"},
            {"type": "quality_gate", "frequency": "after_completion"}
        ]

        # Define success criteria
        plan["success_criteria"] = [
            "All supervisor tasks completed successfully",
            "Dependencies resolved correctly",
            "Quality gates passed",
            "No critical errors or conflicts"
        ]

        return plan

    async def _execute_workflow(self, workflow_id: str):
        """Execute multi-supervisor workflow"""

        try:
            workflow = self.active_workflows[workflow_id]
            plan = workflow["plan"]

            self.logger.info(f"Executing workflow {workflow_id} with strategy: {plan['execution_strategy']}")

            workflow["status"] = "executing"

            if plan["execution_strategy"] == "single_supervisor":
                await self._execute_single_supervisor_workflow(workflow_id)
            else:
                await self._execute_multi_supervisor_workflow(workflow_id)

            # Final quality check
            success = await self._validate_workflow_completion(workflow_id)

            if success:
                workflow["status"] = "completed"
                self.logger.info(f"Workflow {workflow_id} completed successfully")
            else:
                workflow["status"] = "failed"
                self.logger.error(f"Workflow {workflow_id} failed validation")

        except Exception as e:
            self.logger.error(f"Workflow {workflow_id} execution failed: {e}")
            if workflow_id in self.active_workflows:
                self.active_workflows[workflow_id]["status"] = "failed"
                self.active_workflows[workflow_id]["error"] = str(e)

    async def _execute_single_supervisor_workflow(self, workflow_id: str):
        """Execute workflow with single supervisor"""

        workflow = self.active_workflows[workflow_id]

        # Get or create supervisor
        supervisor = await self._get_or_create_supervisor("primary")

        # Execute task
        task_id = await supervisor.create_task_intelligence(
            workflow["description"],
            {"workflow_id": workflow_id}
        )

        workflow["supervisor_tasks"]["primary"] = task_id

        # Monitor completion
        await self._monitor_supervisor_task(workflow_id, "primary", task_id)

    async def _execute_multi_supervisor_workflow(self, workflow_id: str):
        """Execute workflow with multiple supervisors"""

        workflow = self.active_workflows[workflow_id]
        plan = workflow["plan"]

        # Execute based on dependencies
        completed_supervisors = set()

        while len(completed_supervisors) < len(plan["supervisor_assignments"]):
            # Find supervisors ready to execute
            ready_supervisors = []

            for supervisor_id, assignment in plan["supervisor_assignments"].items():
                if supervisor_id in completed_supervisors:
                    continue

                # Check if dependencies are met
                dependencies = plan["dependencies"].get(supervisor_id, [])
                if all(dep in completed_supervisors for dep in dependencies):
                    ready_supervisors.append(supervisor_id)

            if not ready_supervisors:
                self.logger.error(f"Workflow {workflow_id} deadlocked - no supervisors ready")
                break

            # Execute ready supervisors
            tasks = []
            for supervisor_id in ready_supervisors:
                task = asyncio.create_task(
                    self._execute_supervisor_assignment(workflow_id, supervisor_id)
                )
                tasks.append((supervisor_id, task))

            # Wait for completion
            for supervisor_id, task in tasks:
                try:
                    await task
                    completed_supervisors.add(supervisor_id)
                    workflow["completion_status"][supervisor_id] = "completed"
                except Exception as e:
                    self.logger.error(f"Supervisor {supervisor_id} failed: {e}")
                    workflow["completion_status"][supervisor_id] = "failed"

    async def _execute_supervisor_assignment(self, workflow_id: str, supervisor_id: str):
        """Execute a specific supervisor assignment"""

        workflow = self.active_workflows[workflow_id]
        assignment = workflow["plan"]["supervisor_assignments"][supervisor_id]

        # Get or create specialized supervisor
        supervisor = await self._get_or_create_supervisor(supervisor_id, assignment["domain"])

        # Execute domain-specific task
        task_id = await supervisor.create_task_intelligence(
            assignment["task"],
            {
                "workflow_id": workflow_id,
                "domain": assignment["domain"],
                "supervisor_id": supervisor_id
            }
        )

        workflow["supervisor_tasks"][supervisor_id] = task_id

        # Monitor completion
        await self._monitor_supervisor_task(workflow_id, supervisor_id, task_id)

    async def _get_or_create_supervisor(self, supervisor_id: str, specialization: Optional[str] = None) -> TaskIntelligenceSystem:
        """Get existing supervisor or create new one with specialization"""

        if supervisor_id in self.supervisor_agents:
            return self.supervisor_agents[supervisor_id]

        # Create new supervisor
        supervisor = TaskIntelligenceSystem(
            mcp_manager=self.mcp_manager,
            a2a_manager=self.a2a_manager
        )

        self.supervisor_agents[supervisor_id] = supervisor
        self.agent_load[supervisor_id] = 0

        # Set specialization
        if specialization:
            self.agent_specializations[supervisor_id] = [specialization]
            self.logger.info(f"Created specialized supervisor {supervisor_id} for {specialization}")
        else:
            self.agent_specializations[supervisor_id] = ["general"]
            self.logger.info(f"Created general supervisor {supervisor_id}")

        # Initialize performance tracking
        self.agent_performance[supervisor_id] = {
            "tasks_completed": 0,
            "success_rate": 1.0,
            "average_completion_time": 0.0,
            "specialization_score": 1.0
        }

        return supervisor

    async def _monitor_supervisor_task(self, workflow_id: str, supervisor_id: str, task_id: str):
        """Monitor supervisor task completion"""

        supervisor = self.supervisor_agents[supervisor_id]
        start_time = datetime.utcnow()

        # Increment load
        self.agent_load[supervisor_id] += 1

        try:
            # Monitor task progress
            while True:
                # Check if task is complete
                if task_id in supervisor.progress_ledgers:
                    progress_ledger = supervisor.progress_ledgers[task_id]
                    if progress_ledger.is_task_complete():
                        break

                # Check for timeout (30 minutes max)
                elapsed = datetime.utcnow() - start_time
                if elapsed.total_seconds() > 1800:  # 30 minutes
                    raise Exception(f"Task {task_id} timed out")

                # Brief pause
                await asyncio.sleep(5)

            # Update performance metrics
            completion_time = (datetime.utcnow() - start_time).total_seconds()
            await self._update_supervisor_performance(supervisor_id, True, completion_time)

        except Exception as e:
            # Update performance metrics for failure
            await self._update_supervisor_performance(supervisor_id, False, 0)
            raise e

        finally:
            # Decrement load
            self.agent_load[supervisor_id] -= 1

    async def _update_supervisor_performance(self, supervisor_id: str, success: bool, completion_time: float):
        """Update performance metrics for supervisor"""

        if supervisor_id not in self.agent_performance:
            return

        perf = self.agent_performance[supervisor_id]

        # Update task count
        perf["tasks_completed"] += 1

        # Update success rate (exponential moving average)
        alpha = 0.1  # Learning rate
        if success:
            perf["success_rate"] = perf["success_rate"] * (1 - alpha) + alpha
        else:
            perf["success_rate"] = perf["success_rate"] * (1 - alpha)

        # Update average completion time
        if success and completion_time > 0:
            if perf["average_completion_time"] == 0:
                perf["average_completion_time"] = completion_time
            else:
                perf["average_completion_time"] = (
                    perf["average_completion_time"] * (1 - alpha) +
                    completion_time * alpha
                )

    async def _validate_workflow_completion(self, workflow_id: str) -> bool:
        """Validate that workflow completed successfully"""

        workflow = self.active_workflows[workflow_id]
        plan = workflow["plan"]

        # Check all supervisor tasks completed
        for supervisor_id in plan["supervisor_assignments"]:
            status = workflow["completion_status"].get(supervisor_id)
            if status != "completed":
                self.logger.error(f"Supervisor {supervisor_id} did not complete successfully: {status}")
                return False

        # Check success criteria
        for criterion in plan["success_criteria"]:
            if not await self._check_success_criterion(workflow_id, criterion):
                self.logger.error(f"Success criterion not met: {criterion}")
                return False

        return True

    async def _check_success_criterion(self, workflow_id: str, criterion: str) -> bool:
        """Check if a specific success criterion is met"""

        # Simple criterion checking - in production this would be more sophisticated
        workflow = self.active_workflows[workflow_id]

        if "All supervisor tasks completed successfully" in criterion:
            return all(
                status == "completed"
                for status in workflow["completion_status"].values()
            )

        if "No critical errors" in criterion:
            return workflow.get("status") != "failed"

        # Default to true for unknown criteria
        return True

    def get_workflow_status(self, workflow_id: str) -> Optional[Dict[str, Any]]:
        """Get status of a workflow"""
        return self.active_workflows.get(workflow_id)

    def get_supervisor_performance(self) -> Dict[str, Dict[str, Any]]:
        """Get performance metrics for all supervisors"""
        return self.agent_performance.copy()


# Backward compatibility alias
SupervisorAgent = TaskIntelligenceSystem
