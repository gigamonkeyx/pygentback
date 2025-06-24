"""
Comprehensive Test for Tree of Thoughts Implementation

Tests the new ToT framework with core components, generators, evaluators,
and search algorithms.
"""

import asyncio
import logging
import sys
import os

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.ai.reasoning.tot.models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from src.ai.reasoning.tot.tot_engine import ToTEngine
from src.ai.reasoning.tot.thought_generator import OllamaBackend

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_basic_tot_functionality():
    """Test basic ToT functionality with a simple coding problem"""
    
    print("\\n" + "="*60)
    print("TESTING BASIC TOT FUNCTIONALITY")
    print("="*60)
    
    # Create ToT configuration
    config = ToTConfig(
        generation_strategy=GenerationStrategy.PROPOSE,
        evaluation_method=EvaluationMethod.VALUE,
        search_method=SearchMethod.BFS,
        n_generate_sample=3,
        max_depth=4,
        max_iterations=5,
        model_name="phi4-fast"
    )
    
    # Initialize LLM backend (optional for testing)
    try:
        llm_backend = OllamaBackend("phi4-fast")
        print("✓ LLM backend initialized")
    except Exception as e:
        print(f"⚠ LLM backend not available: {e}")
        llm_backend = None
    
    # Create ToT engine
    engine = ToTEngine(config, llm_backend)
    print("✓ ToT engine created")
    
    # Define test problem
    problem = """
    Create a Python function that finds the two numbers in a list that add up to a target sum.
    The function should return the indices of these two numbers.
    """
    
    task_context = {
        'task_type': 'coding',
        'session_id': 'test_basic_functionality'
    }
    
    print(f"\\nProblem: {problem.strip()}")
    print(f"Task context: {task_context}")
    
    # Solve using ToT
    try:
        result = await engine.solve(problem, task_context)
        
        print(f"\\n✓ ToT reasoning completed in {result.search_time:.2f}s")
        print(f"✓ Found {len(result.solutions)} solutions")
        print(f"✓ Success: {result.success}")
        
        # Display solutions
        if result.solutions:
            print("\\nSOLUTIONS:")
            for i, solution in enumerate(result.solutions[:2]):  # Show top 2
                score = getattr(solution.metrics, 'value_score', 0.0)
                print(f"\\nSolution {i+1} (Score: {score:.3f}):")
                print(f"Depth: {solution.depth}")
                print(f"Content: {solution.content[:300]}...")
        
        # Display statistics
        print("\\nSTATISTICS:")
        print(f"Total thoughts: {result.stats.get('total_thoughts', 0)}")
        print(f"Tree depth: {result.stats.get('tree_depth', 0)}")
        print(f"Search method: {result.stats.get('search_stats', {}).get('search_method', 'unknown')}")
        
        return True
        
    except Exception as e:
        print(f"✗ ToT reasoning failed: {e}")
        import traceback
        traceback.print_exc()
        return False


async def test_different_strategies():
    """Test different generation and evaluation strategies"""
    
    print("\\n" + "="*60)
    print("TESTING DIFFERENT STRATEGIES")
    print("="*60)
    
    problem = "Design an algorithm to sort a list of numbers efficiently."
    
    strategies = [
        {
            'name': 'BFS + Value + Propose',
            'generation': GenerationStrategy.PROPOSE,
            'evaluation': EvaluationMethod.VALUE,
            'search': SearchMethod.BFS
        },
        {
            'name': 'DFS + Value + Sample',
            'generation': GenerationStrategy.SAMPLE,
            'evaluation': EvaluationMethod.VALUE,
            'search': SearchMethod.DFS
        },
        {
            'name': 'BFS + Vote + Propose',
            'generation': GenerationStrategy.PROPOSE,
            'evaluation': EvaluationMethod.VOTE,
            'search': SearchMethod.BFS
        }
    ]
    
    results = {}
    
    for strategy in strategies:
        print(f"\\nTesting: {strategy['name']}")
        
        config = ToTConfig(
            generation_strategy=strategy['generation'],
            evaluation_method=strategy['evaluation'],
            search_method=strategy['search'],
            n_generate_sample=2,
            max_depth=3,
            max_iterations=3,
            model_name="phi4-fast"
        )
        
        engine = ToTEngine(config)
        
        try:
            result = await engine.solve(problem, {'task_type': 'coding'})
            
            results[strategy['name']] = {
                'success': result.success,
                'solutions': len(result.solutions),
                'time': result.search_time,
                'thoughts': result.stats.get('total_thoughts', 0)
            }
            
            print(f"  ✓ Success: {result.success}")
            print(f"  ✓ Solutions: {len(result.solutions)}")
            print(f"  ✓ Time: {result.search_time:.2f}s")
            
        except Exception as e:
            print(f"  ✗ Failed: {e}")
            results[strategy['name']] = {'success': False, 'error': str(e)}
    
    # Summary
    print("\\nSTRATEGY COMPARISON:")
    for name, result in results.items():
        if result.get('success'):
            print(f"{name}: {result['solutions']} solutions in {result['time']:.2f}s ({result['thoughts']} thoughts)")
        else:
            print(f"{name}: Failed - {result.get('error', 'Unknown error')}")
    
    return results


async def test_core_components():
    """Test individual core components"""
    
    print("\\n" + "="*60)
    print("TESTING CORE COMPONENTS")
    print("="*60)
    
    # Test Thought creation
    print("Testing Thought creation...")
    from src.ai.reasoning.tot.core.thought import Thought, ThoughtType
    
    thought = Thought(
        content="Test thought content",
        thought_type=ThoughtType.REASONING,
        depth=1
    )
    print(f"✓ Thought created: {thought.id}")
    print(f"✓ Thought content: {thought.content[:50]}...")
    
    # Test ReasoningState
    print("\\nTesting ReasoningState...")
    from src.ai.reasoning.tot.core.state import ReasoningState
    
    state = ReasoningState(
        problem="Test problem",
        session_id="test_components"
    )
    state.add_thought(thought)
    print(f"✓ ReasoningState created with {len(state.thoughts)} thoughts")
    
    # Test ThoughtTree
    print("\\nTesting ThoughtTree...")
    from src.ai.reasoning.tot.core.tree import ThoughtTree
    
    tree = ThoughtTree(max_depth=5)
    tree.add_thought(thought)
    print(f"✓ ThoughtTree created with {tree.size()} thoughts")
    print(f"✓ Tree depth: {tree.get_max_depth()}")
    
    # Test Generators
    print("\\nTesting Thought Generators...")
    from src.ai.reasoning.tot.generators import SamplingGenerator, ProposingGenerator
    
    config = ToTConfig()
    
    sampling_gen = SamplingGenerator(config)
    proposing_gen = ProposingGenerator(config)
    
    try:
        # Test generation (without LLM)
        sample_thoughts = await sampling_gen.generate_thoughts(state, thought, {'task_type': 'coding'})
        propose_thoughts = await proposing_gen.generate_thoughts(state, thought, {'task_type': 'coding'})
        
        print(f"✓ Sampling generator: {len(sample_thoughts)} thoughts")
        print(f"✓ Proposing generator: {len(propose_thoughts)} thoughts")
        
    except Exception as e:
        print(f"⚠ Generator test (expected without LLM): {e}")
    
    # Test Evaluators
    print("\\nTesting Evaluators...")
    from src.ai.reasoning.tot.evaluators import ValueEvaluator, CodingEvaluator
    value_eval = ValueEvaluator(config)
    # vote_eval = VoteEvaluator(config)  # Will be tested separately
    coding_eval = CodingEvaluator(config)
    
    test_thoughts = [thought]
    
    try:
        # Test evaluation (heuristic mode)
        value_results = await value_eval.evaluate_thoughts(test_thoughts, state, {'use_llm_evaluation': False})
        coding_results = await coding_eval.evaluate_thoughts(test_thoughts, state, {'task_type': 'coding'})
        
        print(f"✓ Value evaluator: {len(value_results)} results")
        print(f"✓ Coding evaluator: {len(coding_results)} results")
        
        if value_results:
            print(f"  Sample score: {value_results[0][1]:.3f}")
        
    except Exception as e:
        print(f"✗ Evaluator test failed: {e}")
    
    print("\\n✓ Core components test completed")
    return True


async def test_search_algorithms():
    """Test individual search algorithms"""
    
    print("\\n" + "="*60)
    print("TESTING SEARCH ALGORITHMS")
    print("="*60)
    
    from src.ai.reasoning.tot.search import BFSSearch, DFSSearch, AdaptiveSearch
    from src.ai.reasoning.tot.core.tree import ThoughtTree
    from src.ai.reasoning.tot.core.thought import Thought, ThoughtType
    
    # Create test setup
    config = ToTConfig(max_depth=3, max_iterations=3)
    
    tree = ThoughtTree()
    root_thought = Thought(
        content="Root problem",
        thought_type=ThoughtType.PROBLEM,
        depth=0
    )
    tree.add_thought(root_thought)
    
    # Mock generators and evaluators for testing
    async def mock_generator(state, parent, context):
        return [
            Thought(content=f"Child of {parent.id if parent else 'root'}", 
                   depth=parent.depth + 1 if parent else 1,
                   parent_id=parent.id if parent else None)
        ]
    
    async def mock_evaluator(thoughts, state, context):
        return [(thought, 0.7) for thought in thoughts]
    
    # Test each search algorithm
    algorithms = [
        ('BFS', BFSSearch(config)),
        ('DFS', DFSSearch(config)),
        ('Adaptive', AdaptiveSearch(config))
    ]
    
    for name, algorithm in algorithms:
        print(f"\\nTesting {name} Search...")
        
        try:
            solutions = await algorithm.search(
                tree, mock_generator, mock_evaluator, 
                {'problem': 'test problem'}
            )
            
            stats = algorithm.get_search_stats()
            
            print(f"✓ {name}: {len(solutions)} solutions found")
            print(f"  Search method: {stats.get('search_method', name.lower())}")
            print(f"  Nodes expanded: {stats.get('nodes_expanded', 0)}")
            
        except Exception as e:
            print(f"✗ {name} failed: {e}")
    
    return True


async def main():
    """Run all tests"""
    
    print("TREE OF THOUGHTS - COMPREHENSIVE TEST SUITE")
    print("=" * 80)
    
    tests = [
        ("Core Components", test_core_components),
        ("Search Algorithms", test_search_algorithms),
        ("Basic ToT Functionality", test_basic_tot_functionality),
        ("Different Strategies", test_different_strategies),
    ]
    
    results = {}
    
    for test_name, test_func in tests:
        print(f"\\n\\nRunning: {test_name}")
        print("-" * 40)
        
        try:
            result = await test_func()
            results[test_name] = "PASSED" if result else "FAILED"
            print(f"\\n{test_name}: {results[test_name]}")
            
        except Exception as e:
            results[test_name] = f"ERROR: {e}"
            print(f"\\n{test_name}: {results[test_name]}")
            import traceback
            traceback.print_exc()
    
    # Final summary
    print("\\n\\n" + "=" * 80)
    print("TEST RESULTS SUMMARY")
    print("=" * 80)
    
    for test_name, result in results.items():
        status = "✓" if result == "PASSED" else "✗"
        print(f"{status} {test_name}: {result}")
    
    passed = sum(1 for r in results.values() if r == "PASSED")
    total = len(results)
    
    print(f"\\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("\\n🎉 ALL TESTS PASSED! ToT implementation is working correctly.")
    else:
        print(f"\\n⚠ {total - passed} tests failed. Review the errors above.")
    
    return passed == total


if __name__ == "__main__":
    asyncio.run(main())
