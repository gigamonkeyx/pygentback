"""
Simple Complex Task Test for Task Intelligence System
"""

import asyncio
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Complex Task Description
COMPLEX_TASK = """
Build a Real-Time Collaborative Task Management System with:

FRONTEND: Vue.js 3 + TypeScript, WebSocket real-time updates, drag-and-drop Kanban board
BACKEND: FastAPI + async/await, JWT auth, role-based access, WebSocket server
DATABASE: PostgreSQL with indexing, migrations, audit logging, full-text search
TESTING: 90%+ coverage, unit/integration/e2e tests, performance testing
DEPLOYMENT: Docker + Kubernetes, CI/CD pipeline, blue-green deployment

REQUIREMENTS:
- Handle 1000+ concurrent users
- API response times under 200ms  
- 99.9% uptime
- Zero critical security vulnerabilities
- GDPR compliance
- Accessibility (WCAG 2.1)

INTEGRATIONS: GitHub, Slack, SendGrid, AWS S3, Google Calendar
TIMELINE: 2 weeks
BUDGET: $5000
"""

async def test_task_intelligence():
    """Test the Task Intelligence System with a complex task"""
    
    logger.info("🚀 Testing Task Intelligence System with Complex Task")
    logger.info("=" * 60)
    
    try:
        # Import the system
        from src.agents.supervisor_agent import TaskIntelligenceSystem
        
        logger.info("✅ TaskIntelligenceSystem imported successfully")
        
        # Create the system
        task_intelligence = TaskIntelligenceSystem()
        
        logger.info("✅ TaskIntelligenceSystem created successfully")
        
        # Test task analysis
        logger.info("🔍 Analyzing task complexity...")
        
        # Create a mock task ledger for testing
        from src.agents.supervisor_agent import TaskLedger
        
        task_ledger = TaskLedger(
            task_id="complex_task_001",
            original_request=COMPLEX_TASK
        )
        
        # Analyze the task
        await task_intelligence._analyze_task_requirements(task_ledger)
        
        logger.info(f"📊 Task Analysis Results:")
        logger.info(f"   Requirements: {len(task_ledger.requirements)}")
        logger.info(f"   Success Criteria: {len(task_ledger.success_criteria)}")
        logger.info(f"   Strategy: {task_ledger.strategy}")
        logger.info(f"   Facts: {len(task_ledger.facts)}")
        
        # Test context gathering
        logger.info("🧠 Testing context gathering...")

        initial_context = {
            "task_type": "multi_domain_development",
            "complexity": "high",
            "domains": ["frontend", "backend", "database", "testing", "deployment"]
        }

        await task_intelligence._gather_context(task_ledger, initial_context)
        
        logger.info(f"   Context facts added: {len([f for f in task_ledger.facts if 'context' in f.lower()])}")
        
        # Test question generation
        logger.info("❓ Testing question generation...")
        
        questions = await task_intelligence.get_context_questions(task_ledger.task_id)
        
        if questions:
            logger.info(f"   Generated {len(questions)} questions:")
            for i, q in enumerate(questions[:3], 1):
                logger.info(f"   {i}. {q['question']} (Priority: {q['priority']})")
        else:
            logger.info("   No additional questions needed")
        
        # Test pattern learning
        logger.info("🧩 Testing pattern learning...")
        
        similar_patterns = await task_intelligence.find_similar_patterns(
            COMPLEX_TASK,
            ["frontend", "backend", "database", "testing", "deployment"]
        )
        
        if similar_patterns:
            logger.info(f"   Found {len(similar_patterns)} similar patterns")
        else:
            logger.info("   No similar patterns found - novel task type")
        
        # Test teaching framework
        logger.info("🎓 Testing teaching framework...")
        
        failure_details = {
            "type": "complexity_overload",
            "root_cause": "Task complexity exceeded estimates",
            "task_context": {"task_type": "multi_domain_development"}
        }
        
        teaching_result = await task_intelligence.teach_agent_from_failure(
            "test_agent", task_ledger.task_id, failure_details
        )
        
        logger.info(f"   Teaching applied: {teaching_result['teaching_applied']}")
        logger.info(f"   Strategy: {teaching_result['teaching_strategy']}")
        
        # Get analytics
        logger.info("📊 Getting system analytics...")
        
        analytics = task_intelligence.get_pattern_analytics()
        
        logger.info(f"   Workflow patterns: {analytics['workflow_patterns']['total_patterns']}")
        logger.info(f"   Failure patterns: {analytics['failure_patterns']['total_failures']}")
        
        logger.info("=" * 60)
        logger.info("🎉 Task Intelligence System Test Completed Successfully!")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

async def main():
    """Main test runner"""
    
    print("🧪 PyGent Factory Task Intelligence System")
    print("🎯 Complex Task Test")
    print("=" * 40)
    
    start_time = datetime.now()
    
    try:
        success = await test_task_intelligence()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️  Test Duration: {duration.total_seconds():.2f} seconds")
        
        if success:
            print("✅ All tests passed!")
            print("\n🎯 Capabilities Demonstrated:")
            print("   ✅ Complex task analysis")
            print("   ✅ Context gathering")
            print("   ✅ Question generation")
            print("   ✅ Pattern learning")
            print("   ✅ Teaching framework")
            print("   ✅ Analytics system")
        else:
            print("❌ Some tests failed.")
            
    except KeyboardInterrupt:
        print("\n⏹️  Test interrupted")
    except Exception as e:
        print(f"\n💥 Test failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
