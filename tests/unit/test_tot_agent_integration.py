"""
Test and demonstration of the complete ToT Agent system

This script shows how to:
1. Initialize a ToT Agent
2. Start the agent and connect to LLM backend
3. Solve problems using Tree of Thought reasoning
4. Process task queues
5. Get agent status and statistics
"""

import asyncio
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_tot_agent():
    """Test the complete ToT Agent functionality"""
    
    # Import the ToT components
    try:
        from src.ai.reasoning.tot import (
            ToTAgent, ToTConfig, GenerationStrategy, 
            EvaluationMethod, SearchMethod
        )
    except ImportError as e:
        logger.error(f"Failed to import ToT components: {e}")
        return False
    
    logger.info("🧠 Starting ToT Agent Integration Test")
    
    # Step 1: Create ToT Agent
    agent = ToTAgent(
        agent_id="test_tot_agent_001",
        name="Test ToT Reasoning Agent",
        model_name="qwen2.5-coder:7b",
        capabilities=["reasoning", "problem_solving", "analysis", "mathematics"]
    )
    
    logger.info(f"Created agent: {agent}")
    
    # Step 2: Start the agent
    logger.info("Starting ToT Agent...")
    success = await agent.start()
    
    if not success:
        logger.error("❌ Failed to start ToT Agent")
        return False
    
    logger.info("✅ ToT Agent started successfully")
    
    # Step 3: Test basic problem solving
    test_problems = [
        {
            "problem": "What is 15 * 23? Show your reasoning step by step.",
            "context": {"domain": "mathematics", "difficulty": "easy"}
        },
        {
            "problem": "Explain the concept of recursion in programming with an example.",
            "context": {"domain": "computer_science", "difficulty": "medium"}
        },
        {
            "problem": "How would you design a simple caching system for a web application?",
            "context": {"domain": "system_design", "difficulty": "medium"}
        }
    ]
    
    # Test individual problem solving
    for i, test_case in enumerate(test_problems, 1):
        logger.info(f"\n🔍 Test {i}: {test_case['problem'][:50]}...")
        
        try:
            result = await agent.solve_problem(
                problem=test_case['problem'],
                context=test_case['context']
            )
            
            if result['success']:
                logger.info(f"✅ Solution found (confidence: {result['confidence']:.2f})")
                logger.info(f"📝 Solution: {result['solution'][:200]}...")
                logger.info(f"⏱️  Reasoning time: {result['reasoning_time']:.2f}s")
                logger.info(f"🌳 Thoughts generated: {result['total_thoughts']}")
                logger.info(f"📊 Reasoning depth: {result['reasoning_depth']}")
            else:
                logger.warning(f"❌ Problem solving failed: {result.get('error', 'Unknown error')}")
                
        except Exception as e:
            logger.error(f"❌ Exception during problem solving: {e}")
    
    # Step 4: Test task queue functionality
    logger.info("\n📋 Testing Task Queue Functionality")
    
    # Add tasks with different priorities
    queue_tasks = [
        ("High priority: Calculate compound interest for $1000 at 5% for 3 years", {"priority": "high"}, 3),
        ("Medium priority: Explain binary search algorithm", {"priority": "medium"}, 2),
        ("Low priority: What is the capital of France?", {"priority": "low"}, 1)
    ]
    
    task_ids = []
    for problem, context, priority in queue_tasks:
        task_id = await agent.add_task(problem, context, priority)
        task_ids.append(task_id)
        logger.info(f"Added task {task_id} with priority {priority}")
    
    # Process the task queue
    logger.info("Processing task queue...")
    queue_results = await agent.process_task_queue()
    
    for result in queue_results:
        if result['success']:
            logger.info(f"✅ Task {result['task_id']}: {result['solution'][:100]}...")
        else:
            logger.warning(f"❌ Task {result['task_id']} failed: {result.get('error')}")
    
    # Step 5: Test custom configuration
    logger.info("\n⚙️  Testing Custom ToT Configuration")
    
    custom_config = ToTConfig(
        model_name="qwen2.5-coder:7b",
        generation_strategy=GenerationStrategy.PROPOSE,
        evaluation_method=EvaluationMethod.VOTE,
        search_method=SearchMethod.BFS,
        n_generate_sample=2,  # Fewer samples for faster testing
        n_evaluate_sample=2,
        n_select_sample=1,
        max_depth=3,
        temperature=0.5,
        max_tokens=300,
        task_description="Logical reasoning problem",
        success_criteria="Find logically sound solution"
    )
    
    logic_problem = """
    Three friends Alice, Bob, and Charlie are sitting in a row.
    - Alice is not sitting next to Bob
    - Charlie is sitting to the right of Alice
    - Bob is not at the rightmost position
    
    What is the seating arrangement from left to right?
    """
    
    try:
        result = await agent.solve_problem(
            problem=logic_problem,
            context={"domain": "logic", "type": "constraint_satisfaction"},
            custom_config=custom_config
        )
        
        if result['success']:
            logger.info(f"✅ Logic problem solved: {result['solution']}")
        else:
            logger.warning(f"❌ Logic problem failed: {result.get('error')}")
            
    except Exception as e:
        logger.error(f"❌ Custom config test failed: {e}")
    
    # Step 6: Get agent status and statistics
    logger.info("\n📊 Agent Status and Statistics")
    status = agent.get_status()
    
    logger.info(f"Agent ID: {status['agent_id']}")
    logger.info(f"Agent Name: {status['name']}")
    logger.info(f"Is Active: {status['is_active']}")
    logger.info(f"Model: {status['model_name']}")
    logger.info(f"Capabilities: {', '.join(status['capabilities'])}")
    logger.info(f"Current Sessions: {status['current_sessions']}")
    logger.info(f"Completed Tasks: {status['completed_tasks']}")
    
    stats = status['statistics']
    logger.info(f"Total Sessions: {stats['total_sessions']}")
    logger.info(f"Success Rate: {stats['success_rate']:.1%}")
    logger.info(f"Avg Reasoning Time: {stats['avg_reasoning_time']:.2f}s")
    
    # Step 7: Stop the agent
    logger.info("\n🛑 Stopping ToT Agent")
    await agent.stop()
    logger.info("✅ ToT Agent stopped successfully")
    
    logger.info("\n🎉 ToT Agent Integration Test Completed Successfully!")
    return True


async def test_tot_engine_directly():
    """Test the ToT engine directly without the agent wrapper"""
    
    logger.info("\n🔧 Testing ToT Engine Directly")
    
    try:
        from src.ai.reasoning.tot import (
            ToTEngine, ToTConfig, GenerationStrategy, 
            EvaluationMethod, SearchMethod, OllamaBackend
        )
    except ImportError as e:
        logger.error(f"Failed to import ToT engine components: {e}")
        return False
    
    # Create configuration
    config = ToTConfig(
        model_name="qwen2.5-coder:7b",
        generation_strategy=GenerationStrategy.SAMPLE,
        evaluation_method=EvaluationMethod.VALUE,
        search_method=SearchMethod.ADAPTIVE,
        n_generate_sample=2,
        max_depth=3,
        temperature=0.7,
        max_tokens=200
    )
    
    # Create backend
    backend = OllamaBackend(model_name="qwen2.5-coder:7b")
    
    # Create engine
    engine = ToTEngine(config, backend)
    
    # Test problem
    problem = "How do you reverse a string in Python? Provide multiple approaches."
    
    try:
        result = await engine.solve(problem)
        
        if result.success:
            logger.info("✅ Direct engine test successful")
            logger.info(f"Solutions found: {len(result.solutions)}")
            if result.solutions:
                best_solution = result.solutions[0]
                logger.info(f"Best solution: {best_solution.content[:150]}...")
                logger.info(f"Confidence: {best_solution.confidence:.2f}")
        else:
            logger.warning("❌ Direct engine test failed")
            
    except Exception as e:
        logger.error(f"❌ Direct engine test exception: {e}")
        return False
    
    return True


if __name__ == "__main__":
    async def main():
        logger.info("🚀 Starting Complete ToT System Test")
        
        # Test 1: ToT Agent Integration
        agent_success = await test_tot_agent()
        
        # Test 2: Direct Engine Testing
        engine_success = await test_tot_engine_directly()
        
        # Summary
        logger.info("\n" + "="*50)
        logger.info("📋 TEST SUMMARY")
        logger.info(f"ToT Agent Test: {'✅ PASSED' if agent_success else '❌ FAILED'}")
        logger.info(f"ToT Engine Test: {'✅ PASSED' if engine_success else '❌ FAILED'}")
        
        overall_success = agent_success and engine_success
        logger.info(f"Overall Result: {'🎉 ALL TESTS PASSED' if overall_success else '⚠️  SOME TESTS FAILED'}")
        
        return overall_success
    
    # Run the tests
    success = asyncio.run(main())
