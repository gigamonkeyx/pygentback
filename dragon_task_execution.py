"""
DRAGON FLYING HTML PAGE TASK
Submit to Task Intelligence System with full research capabilities
"""

import asyncio
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

async def execute_dragon_task():
    """Execute the dragon flying HTML page task with full system capabilities"""
    
    logger.info("🐉 DRAGON FLYING HTML PAGE TASK")
    logger.info("🎯 USING FULL TASK INTELLIGENCE SYSTEM WITH RESEARCH")
    logger.info("=" * 70)
    
    try:
        # Import and initialize the REAL Task Intelligence System
        logger.info("📋 Step 1: Initialize FULL Task Intelligence System")
        
        from src.agents.supervisor_agent import TaskIntelligenceSystem
        from src.mcp.server.manager import MCPServerManager
        from src.config.settings import Settings
        
        # Initialize MCP manager
        settings = Settings()
        mcp_manager = MCPServerManager(settings)
        await mcp_manager.initialize()
        
        # Create Task Intelligence System with full capabilities
        task_intelligence = TaskIntelligenceSystem(mcp_manager=mcp_manager)
        
        logger.info("   ✅ Full Task Intelligence System initialized")
        
        # Submit the dragon task
        logger.info("📋 Step 2: Submit Dragon Task with Research Requirements")
        
        dragon_task = """
        Create an HTML page with a beautiful dragon flying around the page using GPU acceleration.
        
        REQUIREMENTS:
        1. Beautiful, high-resolution dragon graphics
        2. Smooth flying animation around the page
        3. GPU acceleration for performance
        4. Cool visual effects (particles, trails, lighting)
        5. Responsive to different screen sizes
        6. Professional quality implementation
        
        RESEARCH NEEDED:
        - Best GPU-accelerated web graphics libraries (WebGL, Three.js, etc.)
        - Dragon 3D models or sprite techniques
        - Animation patterns for realistic flying
        - Particle systems for visual effects
        - Performance optimization techniques
        
        TECHNICAL REQUIREMENTS:
        - Use modern web technologies
        - Leverage GPU through WebGL/WebGPU
        - High frame rate (60fps target)
        - Cross-browser compatibility
        - Responsive design
        
        DELIVERABLES:
        - Complete HTML file with embedded CSS/JS
        - Dragon graphics (3D model or high-quality sprites)
        - Smooth animation system
        - GPU-accelerated rendering
        - Cool visual effects
        
        The result should look professional and impressive, suitable for showcasing advanced web graphics capabilities.
        """
        
        # Submit with research context
        task_id = await task_intelligence.create_task_intelligence(
            task_description=dragon_task,
            context={
                "task_type": "graphics_development",
                "complexity": "enterprise",
                "priority": "high",
                "requires_research": True,
                "gpu_acceleration": True,
                "visual_quality": "high",
                "target_platform": "web",
                "research_domains": [
                    "webgl", "three_js", "gpu_acceleration", 
                    "3d_graphics", "animation", "particle_systems"
                ]
            }
        )
        
        logger.info(f"   ✅ Dragon task submitted: {task_id}")
        
        # Monitor execution with research phase
        logger.info("📋 Step 3: Monitor Research and Execution")
        logger.info("   🔍 Allowing system to research and execute autonomously...")
        
        # Give the system time to research and execute
        for i in range(24):  # 24 * 5 = 120 seconds (2 minutes)
            await asyncio.sleep(5)
            
            # Check progress
            if task_id in task_intelligence.task_ledgers:
                task_ledger = task_intelligence.task_ledgers[task_id]
                logger.info(f"   📊 Requirements: {len(task_ledger.requirements)}, Facts: {len(task_ledger.facts)}")
            
            if task_id in task_intelligence.progress_ledgers:
                progress = task_intelligence.progress_ledgers[task_id]
                logger.info(f"   📈 Progress: Step {progress.current_step}/{progress.total_steps}")
            
            # Check for research questions
            questions = await task_intelligence.get_context_questions(task_id)
            if questions and i == 5:  # Show questions after 25 seconds
                logger.info(f"   ❓ Research questions generated: {len(questions)}")
                for j, q in enumerate(questions[:3], 1):
                    logger.info(f"      {j}. {q['question']}")
            
            logger.info(f"   ⏱️  Autonomous execution: {(i+1)*5} seconds")
        
        # Check final results
        logger.info("📋 Step 4: Check Results and Run QA")
        
        if task_id in task_intelligence.task_ledgers:
            task_ledger = task_intelligence.task_ledgers[task_id]
            
            logger.info("🔍 DRAGON TASK ANALYSIS RESULTS:")
            logger.info(f"   Requirements identified: {len(task_ledger.requirements)}")
            logger.info(f"   Success criteria: {len(task_ledger.success_criteria)}")
            logger.info(f"   Strategy selected: {task_ledger.strategy}")
            logger.info(f"   Facts gathered: {len(task_ledger.facts)}")
            
            # Show requirements
            logger.info("📋 REQUIREMENTS IDENTIFIED:")
            for i, req in enumerate(task_ledger.requirements, 1):
                logger.info(f"   {i}. {req}")
            
            # Check for generated files
            import os
            dragon_files = []
            
            # Look for HTML files
            for file in os.listdir("."):
                if file.endswith((".html", ".htm")) and "dragon" in file.lower():
                    dragon_files.append(file)
            
            if dragon_files:
                logger.info(f"   📁 Dragon files found: {dragon_files}")
                output_to_evaluate = f"Dragon HTML files created: {', '.join(dragon_files)}"
            else:
                logger.info("   ❌ No dragon HTML files found")
                output_to_evaluate = "No dragon files detected"
            
            # Run QA evaluation
            from src.agents.supervisor_agent import TaskAnalysis, TaskType, TaskComplexity
            
            analysis = TaskAnalysis(
                task_type=TaskType.UI_CREATION,
                complexity=TaskComplexity.ENTERPRISE,
                complexity_score=0.95,
                estimated_time=60,
                required_capabilities=["graphics_development", "webgl", "animation", "gpu_acceleration"],
                success_criteria=task_ledger.success_criteria
            )
            
            quality_score = await task_intelligence.evaluate_quality(output_to_evaluate, analysis)
            
            logger.info("🎯 DRAGON TASK QA RESULTS:")
            logger.info(f"   Quality Score: {quality_score.score:.2f}")
            logger.info(f"   Passed QA: {quality_score.passed}")
            logger.info(f"   Issues Found: {len(quality_score.issues)}")
            
            if quality_score.issues:
                logger.info("❌ QA ISSUES:")
                for i, issue in enumerate(quality_score.issues, 1):
                    logger.info(f"   {i}. {issue}")
            
            if quality_score.suggestions:
                logger.info("💡 QA SUGGESTIONS:")
                for i, suggestion in enumerate(quality_score.suggestions, 1):
                    logger.info(f"   {i}. {suggestion}")
            
            # Generate feedback
            feedback = await task_intelligence.generate_feedback(quality_score, analysis)
            logger.info(f"📝 SYSTEM FEEDBACK: {feedback}")
            
            return {
                "task_id": task_id,
                "quality_score": quality_score,
                "files_created": dragon_files,
                "requirements": len(task_ledger.requirements),
                "strategy": task_ledger.strategy,
                "qa_passed": quality_score.passed,
                "feedback": feedback
            }
        else:
            logger.error("❌ Dragon task not found in system")
            return None
            
    except Exception as e:
        logger.error(f"💥 Dragon task execution failed: {e}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main dragon task runner"""
    
    print("🐉 DRAGON FLYING HTML PAGE TASK")
    print("🎯 FULL TASK INTELLIGENCE SYSTEM WITH RESEARCH")
    print("=" * 60)
    
    start_time = datetime.now()
    
    try:
        results = await execute_dragon_task()
        
        end_time = datetime.now()
        duration = end_time - start_time
        
        print(f"\n⏱️  Total Execution Time: {duration.total_seconds():.2f} seconds")
        
        if results:
            print("\n🐉 DRAGON TASK RESULTS:")
            print(f"   Task ID: {results['task_id']}")
            print(f"   Quality Score: {results['quality_score'].score:.2f}")
            print(f"   QA Passed: {'✅' if results['qa_passed'] else '❌'}")
            print(f"   Files Created: {len(results['files_created'])}")
            print(f"   Requirements: {results['requirements']}")
            print(f"   Strategy: {results['strategy']}")
            print(f"   Feedback: {results['feedback']}")
            
            if results['files_created']:
                print(f"\n📁 Dragon Files Created:")
                for file in results['files_created']:
                    print(f"   • {file}")
                print("\n🚀 Open the HTML file in your browser to see the dragon!")
            
            if results['qa_passed']:
                print("\n🎉 DRAGON TASK COMPLETED SUCCESSFULLY!")
            else:
                print("\n⚠️ Dragon task needs improvement")
        else:
            print("\n💥 Dragon task failed")
        
    except KeyboardInterrupt:
        print("\n⏹️  Dragon task interrupted")
    except Exception as e:
        print(f"\n💥 Dragon task failed: {e}")

if __name__ == "__main__":
    asyncio.run(main())
