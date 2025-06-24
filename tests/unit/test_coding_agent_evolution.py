"""
Coding Agent Evolution Test Suite

This script:
1. Creates a baseline coding agent
2. Tests its basic functionality
3. Evolves it using genetic algorithms (50 iterations)
4. Tests the evolved agent
5. Adds MCP tools and tests again
"""

import asyncio
import logging
import json
import time
from datetime import datetime
from typing import Dict, Any, List

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import our systems
from src.core.agent_factory import AgentFactory
from src.core.agent.config import AgentConfig
from src.core.agent import AgentMessage, MessageType
from src.ai.evolution.genetic_algorithm import GeneticAlgorithm
from src.memory.memory_manager import MemoryManager
from src.mcp.server_registry import MCPServerManager
from src.config.settings import Settings

class CodingAgentEvolutionTest:
    """Test suite for coding agent evolution"""
    
    def __init__(self):
        self.factory = None
        self.memory_manager = None
        self.mcp_manager = None
        self.baseline_agent = None
        self.evolved_agent = None
        self.test_results = {
            "baseline_tests": {},
            "evolution_results": {},
            "evolved_tests": {},
            "mcp_enhanced_tests": {},
            "performance_comparison": {}
        }
        
        # Coding test cases for evaluation
        self.test_cases = [
            {
                "name": "simple_function",
                "request": "Write a Python function to calculate the factorial of a number",
                "type": "code_generation",
                "expected_keywords": ["def", "factorial", "return"]
            },
            {
                "name": "debug_issue",
                "request": "Fix this Python code: def add(a, b) return a + b",
                "type": "debugging",
                "expected_keywords": ["def", ":", "return"]
            },
            {
                "name": "code_analysis",
                "request": "Explain what this code does: def quicksort(arr): if len(arr) <= 1: return arr",
                "type": "code_analysis",
                "expected_keywords": ["quicksort", "algorithm", "sorting"]
            },
            {
                "name": "optimization",
                "request": "Optimize this Python code for better performance: for i in range(len(list)): print(list[i])",
                "type": "optimization",
                "expected_keywords": ["for", "in", "enumerate"]
            },
            {
                "name": "documentation",
                "request": "Generate documentation for: def binary_search(arr, target):",
                "type": "documentation",
                "expected_keywords": ["binary", "search", "parameter"]
            }
        ]
    
    async def setup(self):
        """Set up the test environment"""
        logger.info("Setting up test environment...")
        
        # Initialize managers
        self.memory_manager = MemoryManager()
        await self.memory_manager.initialize()
        
        self.mcp_manager = MCPServerManager()
        await self.mcp_manager.initialize()
        
        # Create agent factory
        self.factory = AgentFactory(
            mcp_manager=self.mcp_manager,
            memory_manager=self.memory_manager
        )
        await self.factory.initialize()
        
        logger.info("Test environment setup complete")
    
    async def test_baseline_agent(self):
        """Create and test baseline coding agent"""
        logger.info("Creating baseline coding agent...")
        
        # Create baseline agent
        self.baseline_agent = await self.factory.create_agent(
            agent_type="coding",
            name="baseline_coding_agent",
            capabilities=["code_generation", "debugging", "code_analysis"],
            custom_config={
                "model_name": "deepseek-coder-v2:latest",
                "temperature": 0.3,
                "max_tokens": 1000
            }
        )
        
        logger.info(f"Baseline agent created: {self.baseline_agent.agent_id}")
        
        # Test baseline agent
        baseline_results = await self._run_test_suite(self.baseline_agent, "baseline")
        self.test_results["baseline_tests"] = baseline_results
        
        logger.info(f"Baseline tests completed. Score: {baseline_results['overall_score']:.2f}")
        return baseline_results
    
    async def evolve_agent(self, iterations=50):
        """Evolve the coding agent using genetic algorithms"""
        logger.info(f"Starting agent evolution with {iterations} iterations...")
        
        # Create genetic algorithm for agent evolution
        ga = GeneticAlgorithm(
            population_size=10,
            mutation_rate=0.1,
            crossover_rate=0.8,
            elitism_rate=0.2
        )
        
        # Define evolution parameters for coding agents
        evolution_config = {
            "agent_type": "coding",
            "base_capabilities": ["code_generation", "debugging", "code_analysis"],
            "mutable_parameters": {
                "temperature": {"min": 0.1, "max": 0.7, "step": 0.1},
                "max_tokens": {"min": 500, "max": 2000, "step": 100},
                "reasoning_depth": {"min": 1, "max": 5, "step": 1},
                "code_style_preference": {"values": ["concise", "verbose", "balanced"]},
                "optimization_focus": {"values": ["speed", "readability", "memory"]}
            },
            "fitness_function": self._evaluate_coding_fitness
        }
        
        # Run evolution
        start_time = time.time()
        evolution_results = await ga.evolve_agent(
            base_agent=self.baseline_agent,
            config=evolution_config,
            iterations=iterations,
            factory=self.factory
        )
        evolution_time = time.time() - start_time
        
        # Get the best evolved agent
        self.evolved_agent = evolution_results["best_agent"]
        
        # Store evolution results
        self.test_results["evolution_results"] = {
            "iterations": iterations,
            "evolution_time": evolution_time,
            "initial_fitness": evolution_results["initial_fitness"],
            "final_fitness": evolution_results["final_fitness"],
            "improvement": evolution_results["final_fitness"] - evolution_results["initial_fitness"],
            "best_parameters": evolution_results["best_parameters"],
            "fitness_history": evolution_results["fitness_history"]
        }
        
        logger.info(f"Evolution completed in {evolution_time:.2f}s")
        logger.info(f"Fitness improved from {evolution_results['initial_fitness']:.3f} to {evolution_results['final_fitness']:.3f}")
        
        return evolution_results
    
    async def test_evolved_agent(self):
        """Test the evolved agent"""
        logger.info("Testing evolved agent...")
        
        if not self.evolved_agent:
            raise ValueError("No evolved agent available. Run evolve_agent() first.")
        
        # Test evolved agent
        evolved_results = await self._run_test_suite(self.evolved_agent, "evolved")
        self.test_results["evolved_tests"] = evolved_results
        
        logger.info(f"Evolved agent tests completed. Score: {evolved_results['overall_score']:.2f}")
        return evolved_results
    
    async def add_mcp_tools_and_test(self):
        """Add MCP tools to the evolved agent and test"""
        logger.info("Adding MCP tools to evolved agent...")
        
        if not self.evolved_agent:
            raise ValueError("No evolved agent available. Run evolve_agent() first.")
        
        # Add relevant MCP tools for coding
        mcp_tools = [
            "read_file",           # File system operations
            "write_file",
            "edit_file",
            "search_files",
            "resolve-library-id",  # Context7 documentation
            "get-library-docs",
            "directory_tree",      # Project structure
            "list_directory"
        ]
        
        # Register MCP tools with the agent
        for tool in mcp_tools:
            try:
                await self.evolved_agent.register_mcp_tool(tool, "filesystem" if "file" in tool else "context7")
                logger.info(f"Registered MCP tool: {tool}")
            except Exception as e:
                logger.warning(f"Failed to register MCP tool {tool}: {e}")
        
        # Test agent with MCP enhancements
        mcp_enhanced_results = await self._run_test_suite(self.evolved_agent, "mcp_enhanced")
        self.test_results["mcp_enhanced_tests"] = mcp_enhanced_results
        
        logger.info(f"MCP-enhanced tests completed. Score: {mcp_enhanced_results['overall_score']:.2f}")
        return mcp_enhanced_results
    
    async def _run_test_suite(self, agent, test_type: str) -> Dict[str, Any]:
        """Run the complete test suite on an agent"""
        results = {
            "test_type": test_type,
            "agent_id": agent.agent_id,
            "timestamp": datetime.now().isoformat(),
            "test_results": {},
            "performance_metrics": {},
            "overall_score": 0.0
        }
        
        total_score = 0.0
        successful_tests = 0
        
        for test_case in self.test_cases:
            logger.info(f"Running test: {test_case['name']}")
            
            try:
                # Create test message
                test_message = AgentMessage(
                    type=MessageType.REQUEST,
                    sender="test_runner",
                    recipient=agent.agent_id,
                    content=test_case["request"]
                )
                
                # Measure response time
                start_time = time.time()
                response = await agent.process_message(test_message)
                response_time = time.time() - start_time
                
                # Evaluate response
                score = self._evaluate_response(response, test_case)
                total_score += score
                successful_tests += 1
                
                results["test_results"][test_case["name"]] = {
                    "score": score,
                    "response_time": response_time,
                    "response_type": response.content.get("type", "unknown"),
                    "success": score > 0.5,
                    "response_length": len(str(response.content))
                }
                
                logger.info(f"Test {test_case['name']} completed. Score: {score:.2f}")
                
            except Exception as e:
                logger.error(f"Test {test_case['name']} failed: {e}")
                results["test_results"][test_case["name"]] = {
                    "score": 0.0,
                    "error": str(e),
                    "success": False
                }
        
        # Calculate overall metrics
        results["overall_score"] = total_score / len(self.test_cases) if self.test_cases else 0.0
        results["success_rate"] = successful_tests / len(self.test_cases) if self.test_cases else 0.0
        
        return results
    
    def _evaluate_response(self, response: AgentMessage, test_case: Dict[str, Any]) -> float:
        """Evaluate the quality of an agent response"""
        if not response or not response.content:
            return 0.0
        
        content = response.content
        response_text = str(content).lower()
        
        score = 0.0
        
        # Check for expected keywords
        keywords_found = 0
        for keyword in test_case.get("expected_keywords", []):
            if keyword.lower() in response_text:
                keywords_found += 1
        
        if test_case.get("expected_keywords"):
            score += (keywords_found / len(test_case["expected_keywords"])) * 0.4
        
        # Check response type match
        if content.get("type") == test_case.get("type"):
            score += 0.3
        
        # Check for code presence (for code generation tasks)
        if test_case["type"] == "code_generation":
            if any(indicator in response_text for indicator in ["def ", "function", "class", "{", "}"]):
                score += 0.2
        
        # Check response length (should be substantial)
        if len(response_text) > 50:
            score += 0.1
        
        # Avoid error responses
        if content.get("type") != "error" and not content.get("fallback"):
            score += 0.1
        else:
            score = max(0.1, score)  # Minimum score for non-error responses
        
        return min(1.0, score)
    
    async def _evaluate_coding_fitness(self, agent) -> float:
        """Fitness function for coding agent evolution"""
        # Run a subset of tests for fitness evaluation
        fitness_tests = self.test_cases[:3]  # Use first 3 tests for speed
        
        total_fitness = 0.0
        for test_case in fitness_tests:
            try:
                test_message = AgentMessage(
                    type=MessageType.REQUEST,
                    sender="fitness_evaluator",
                    recipient=agent.agent_id,
                    content=test_case["request"]
                )
                
                response = await agent.process_message(test_message)
                fitness = self._evaluate_response(response, test_case)
                total_fitness += fitness
                
            except Exception:
                total_fitness += 0.0  # Penalty for errors
        
        return total_fitness / len(fitness_tests)
    
    async def generate_report(self):
        """Generate comprehensive test report"""
        # Compare performance
        baseline_score = self.test_results.get("baseline_tests", {}).get("overall_score", 0.0)
        evolved_score = self.test_results.get("evolved_tests", {}).get("overall_score", 0.0)
        mcp_score = self.test_results.get("mcp_enhanced_tests", {}).get("overall_score", 0.0)
        
        self.test_results["performance_comparison"] = {
            "baseline_score": baseline_score,
            "evolved_score": evolved_score,
            "mcp_enhanced_score": mcp_score,
            "evolution_improvement": evolved_score - baseline_score,
            "mcp_improvement": mcp_score - evolved_score,
            "total_improvement": mcp_score - baseline_score
        }
        
        # Save detailed report
        report_file = f"coding_agent_evolution_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.test_results, f, indent=2)
        
        logger.info(f"Report saved to: {report_file}")
        
        # Print summary
        print("\n" + "="*80)
        print("CODING AGENT EVOLUTION TEST RESULTS")
        print("="*80)
        print(f"Baseline Score:      {baseline_score:.3f}")
        print(f"Evolved Score:       {evolved_score:.3f}")
        print(f"MCP Enhanced Score:  {mcp_score:.3f}")
        print("-"*40)
        print(f"Evolution Improvement: {evolved_score - baseline_score:+.3f}")
        print(f"MCP Improvement:       {mcp_score - evolved_score:+.3f}")
        print(f"Total Improvement:     {mcp_score - baseline_score:+.3f}")
        print("="*80)
        
        return self.test_results
    
    async def cleanup(self):
        """Clean up resources"""
        if self.baseline_agent:
            await self.factory.destroy_agent(self.baseline_agent.agent_id)
        if self.evolved_agent:
            await self.factory.destroy_agent(self.evolved_agent.agent_id)
        
        if self.factory:
            await self.factory.shutdown()
        if self.memory_manager:
            await self.memory_manager.shutdown()
        if self.mcp_manager:
            await self.mcp_manager.shutdown()

async def main():
    """Main test execution"""
    test_suite = CodingAgentEvolutionTest()
    
    try:
        # Setup
        await test_suite.setup()
        
        # Run baseline test
        print("Step 1: Testing baseline coding agent...")
        await test_suite.test_baseline_agent()
        
        # Evolve agent
        print("Step 2: Evolving agent with 50 iterations...")
        await test_suite.evolve_agent(iterations=50)
        
        # Test evolved agent
        print("Step 3: Testing evolved agent...")
        await test_suite.test_evolved_agent()
        
        # Add MCP tools and test
        print("Step 4: Adding MCP tools and testing...")
        await test_suite.add_mcp_tools_and_test()
        
        # Generate report
        print("Step 5: Generating final report...")
        await test_suite.generate_report()
        
    except Exception as e:
        logger.error(f"Test suite failed: {e}")
        raise
    finally:
        await test_suite.cleanup()

if __name__ == "__main__":
    asyncio.run(main())
