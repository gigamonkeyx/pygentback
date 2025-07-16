"""
REAL TASK INTELLIGENCE SYSTEM WITH REAL QA
Use the actual system as designed - let it fail if it's going to fail!
"""

import asyncio
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def use_real_task_intelligence_system():
    """Use the REAL Task Intelligence System with REAL QA - no shortcuts!"""
    
    logger.info("🎯 USING REAL TASK INTELLIGENCE SYSTEM WITH REAL QA")
    logger.info("🔥 NO SHORTCUTS, NO FIXES, LET IT FAIL IF IT FAILS!")
    logger.info("=" * 70)
    
    try:
        # Import the REAL Task Intelligence System
        logger.info("📋 Step 1: Import REAL Task Intelligence System")
        
        from src.agents.supervisor_agent import TaskIntelligenceSystem, TaskLedger
        
        # Create MCP manager first
        from src.mcp.server.manager import MCPServerManager
        from src.config.settings import Settings

        settings = Settings()
        mcp_manager = MCPServerManager(settings)
        await mcp_manager.initialize()

        # Create the REAL system with MCP manager
        task_intelligence = TaskIntelligenceSystem(mcp_manager=mcp_manager)
        
        logger.info("   ✅ Real Task Intelligence System imported and created")
        
        # Submit the REAL UI replacement task
        logger.info("📋 Step 2: Submit REAL UI replacement task")
        
        ui_replacement_task = """
        Create a complete Vue.js 3 UI replacement for PyGent Factory that:
        
        1. Provides full interface for all backend endpoints (/agents/*, /tasks/*, /mcp/*, /a2a/*, /research/*, /orchestration/*)
        2. Includes real-time agent status cards with live indicators
        3. Has a functional task queue with progress bars and drag-drop
        4. Shows system metrics dashboard with real charts
        5. Uses Vue.js 3 + Composition API + TypeScript
        6. Implements Pinia for state management
        7. Has responsive design with Tailwind CSS
        8. Includes real WebSocket connections for live updates
        9. Has proper error handling and loading states
        10. Is production-ready and deployable
        
        The UI must actually work and be testable by running npm run dev.
        """
        
        # Use the REAL create_task_intelligence method
        task_id = await task_intelligence.create_task_intelligence(
            task_description=ui_replacement_task,
            context={
                "task_type": "ui_replacement",
                "complexity": "enterprise", 
                "priority": "high",
                "framework": "vue3",
                "real_execution_required": True,
                "qa_required": True
            }
        )
        
        logger.info(f"   ✅ Task submitted to REAL system: {task_id}")
        
        # Let the REAL system work for a reasonable time
        logger.info("📋 Step 3: Let REAL system execute with REAL QA")
        logger.info("   ⏱️  Giving system 60 seconds to work autonomously...")
        
        # Monitor the REAL execution
        for i in range(12):  # 12 * 5 = 60 seconds
            await asyncio.sleep(5)
            
            # Check REAL progress
            if task_id in task_intelligence.task_ledgers:
                task_ledger = task_intelligence.task_ledgers[task_id]
                logger.info(f"   📊 Requirements: {len(task_ledger.requirements)}, Facts: {len(task_ledger.facts)}")
            
            if task_id in task_intelligence.progress_ledgers:
                progress = task_intelligence.progress_ledgers[task_id]
                logger.info(f"   📈 Progress: Step {progress.current_step}/{progress.total_steps}")
            
            logger.info(f"   ⏱️  Autonomous execution: {(i+1)*5} seconds")
        
        # Check what the REAL system actually produced
        logger.info("📋 Step 4: Check REAL system output and run REAL QA")
        
        if task_id in task_intelligence.task_ledgers:
            task_ledger = task_intelligence.task_ledgers[task_id]
            
            logger.info("🔍 REAL SYSTEM ANALYSIS RESULTS:")
            logger.info(f"   Requirements identified: {len(task_ledger.requirements)}")
            logger.info(f"   Success criteria: {len(task_ledger.success_criteria)}")
            logger.info(f"   Strategy selected: {task_ledger.strategy}")
            logger.info(f"   Facts gathered: {len(task_ledger.facts)}")
            
            # Show the requirements the REAL system found
            logger.info("📋 REAL REQUIREMENTS IDENTIFIED:")
            for i, req in enumerate(task_ledger.requirements, 1):
                logger.info(f"   {i}. {req}")
            
            # Show success criteria
            logger.info("🎯 REAL SUCCESS CRITERIA:")
            for i, criteria in enumerate(task_ledger.success_criteria, 1):
                logger.info(f"   {i}. {criteria}")
            
            # Now run the REAL QA system
            logger.info("🔍 RUNNING REAL QA EVALUATION...")
            
            # Import the real analysis types
            from src.agents.supervisor_agent import TaskAnalysis, TaskType, TaskComplexity
            
            # Create a real task analysis for QA
            analysis = TaskAnalysis(
                task_type=TaskType.UI_CREATION,
                complexity=TaskComplexity.ENTERPRISE,
                complexity_score=0.9,
                estimated_time=120,
                required_capabilities=["frontend_development", "vue_js", "typescript"],
                success_criteria=task_ledger.success_criteria
            )
            
            # Check if the system produced any actual output
            # Look for any execution results or generated content
            output_to_evaluate = None
            
            # Check if any files were created or if there's execution output
            import os
            if os.path.exists("pygent_ui_replacement"):
                output_to_evaluate = "Vue.js project structure created with components"
                logger.info("   📁 Found generated UI project")
            else:
                output_to_evaluate = "No tangible output detected"
                logger.info("   ❌ No generated files found")
            
            # Run the REAL quality evaluation
            quality_score = await task_intelligence.evaluate_quality(output_to_evaluate, analysis)
            
            logger.info("🎯 REAL QA RESULTS:")
            logger.info(f"   Quality Score: {quality_score.score:.2f}")
            logger.info(f"   Passed QA: {quality_score.passed}")
            logger.info(f"   Issues Found: {len(quality_score.issues)}")
            
            if quality_score.issues:
                logger.info("❌ QA ISSUES IDENTIFIED:")
                for i, issue in enumerate(quality_score.issues, 1):
                    logger.info(f"   {i}. {issue}")
            
            if quality_score.suggestions:
                logger.info("💡 QA SUGGESTIONS:")
                for i, suggestion in enumerate(quality_score.suggestions, 1):
                    logger.info(f"   {i}. {suggestion}")
            
            # Generate REAL feedback
            feedback = await task_intelligence.generate_feedback(quality_score, analysis)
            logger.info(f"📝 REAL SYSTEM FEEDBACK: {feedback}")
            
            # Check if the system wants to ask questions
            questions = await task_intelligence.get_context_questions(task_id)
            if questions:
                logger.info(f"❓ REAL SYSTEM QUESTIONS ({len(questions)}):")
                for i, q in enumerate(questions, 1):
                    logger.info(f"   {i}. {q['question']} (Priority: {q['priority']})")
            
            return {
                "task_id": task_id,
                "quality_score": quality_score,
                "requirements": len(task_ledger.requirements),
                "success_criteria": len(task_ledger.success_criteria),
                "strategy": task_ledger.strategy,
                "qa_passed": quality_score.passed,
                "feedback": feedback,
                "questions": len(questions) if questions else 0
            }
        else:
            logger.error("❌ Task not found in system - execution may have failed")
            return None
            
    except Exception as e:
        logger.error(f"💥 REAL SYSTEM EXECUTION FAILED: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Run the real system with real QA"""
    
    print("🎯 REAL TASK INTELLIGENCE SYSTEM WITH REAL QA")
    print("🔥 LET IT FAIL IF IT'S GOING TO FAIL!")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        results = await use_real_task_intelligence_system()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️  Total Execution Time: {duration.total_seconds():.2f} seconds")
        
        if results:
            print("\n🎯 REAL SYSTEM RESULTS:")
            print(f"   Task ID: {results['task_id']}")
            print(f"   Quality Score: {results['quality_score'].score:.2f}")
            print(f"   QA Passed: {'✅' if results['qa_passed'] else '❌'}")
            print(f"   Requirements: {results['requirements']}")
            print(f"   Success Criteria: {results['success_criteria']}")
            print(f"   Strategy: {results['strategy']}")
            print(f"   Questions Generated: {results['questions']}")
            print(f"   Feedback: {results['feedback']}")
            
            if results['qa_passed']:
                print("\n🎉 REAL QA PASSED - System produced quality output!")
            else:
                print("\n❌ REAL QA FAILED - System needs improvement!")
                print("   This is valuable learning data!")
        else:
            print("\n💥 REAL SYSTEM FAILED COMPLETELY")
            print("   This tells us what needs to be fixed!")
        
    except KeyboardInterrupt:
        print("\n⏹️  Execution interrupted")
    except Exception as e:
        print(f"\n💥 Failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
