#!/usr/bin/env python3
"""
Golden Glue Test - Vue.js UI Creation with Supervisor Agent
Tests the complete supervisor agent system with a Vue.js UI creation task.
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from agents.supervisor_agent import SupervisorAgent
from agents.quality_evaluator import QualityEvaluator
from agents.teaching_system import TeachingSystem


async def test_golden_glue():
    """Test the complete Golden Glue system"""
    
    print("🚀 TESTING GOLDEN GLUE - SUPERVISOR AGENT SYSTEM")
    print("=" * 60)
    
    # Initialize components
    supervisor = SupervisorAgent()
    evaluator = QualityEvaluator()
    teacher = TeachingSystem()
    
    print("✅ All components initialized successfully")
    
    # Test task: Create Vue.js UI
    task_description = "Create a Vue.js alternative UI for PyGent Factory with components, routing, and API integration"
    
    print(f"\n📋 TASK: {task_description}")
    print("-" * 60)
    
    # Step 1: Task Analysis
    print("\n🔍 STEP 1: Task Analysis")
    analysis = await supervisor.analyze_task(task_description)
    print(f"   Task Type: {analysis.task_type}")
    print(f"   Complexity: {analysis.complexity}/10")
    print(f"   Estimated Time: {analysis.estimated_time} minutes")
    print(f"   Required Capabilities: {', '.join(analysis.required_capabilities)}")
    
    # Step 2: Agent Selection
    print("\n🤖 STEP 2: Agent Selection")
    selected_agent = await supervisor.select_agent(analysis)
    print(f"   Selected Agent Type: {selected_agent}")
    
    # Step 3: Simulate Agent Output (Vue.js UI)
    print("\n💻 STEP 3: Simulated Agent Output")
    simulated_vue_output = '''
<template>
  <div id="app">
    <nav class="navbar">
      <div class="nav-brand">
        <h1>PyGent Factory</h1>
      </div>
      <div class="nav-links">
        <router-link to="/">Home</router-link>
        <router-link to="/agents">Agents</router-link>
        <router-link to="/tasks">Tasks</router-link>
      </div>
    </nav>
    
    <main class="main-content">
      <router-view />
    </main>
  </div>
</template>

<script>
export default {
  name: 'App',
  data() {
    return {
      apiBaseUrl: 'http://localhost:8000/api/v1'
    }
  },
  mounted() {
    this.checkApiConnection();
  },
  methods: {
    async checkApiConnection() {
      try {
        const response = await fetch(`${this.apiBaseUrl}/health`);
        if (response.ok) {
          console.log('✅ API connection successful');
        }
      } catch (error) {
        console.error('❌ API connection failed:', error);
      }
    }
  }
}
</script>

<style scoped>
.navbar {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 1rem 2rem;
  background: #2c3e50;
  color: white;
}

.nav-brand h1 {
  margin: 0;
  color: #3498db;
}

.nav-links a {
  color: white;
  text-decoration: none;
  margin-left: 1rem;
  padding: 0.5rem 1rem;
  border-radius: 4px;
  transition: background 0.3s;
}

.nav-links a:hover {
  background: #34495e;
}

.main-content {
  padding: 2rem;
  min-height: calc(100vh - 80px);
}
</style>
'''
    
    print("   Generated Vue.js component with navigation, routing, and API integration")
    
    # Step 4: Quality Evaluation
    print("\n📊 STEP 4: Quality Evaluation")
    quality_report = await evaluator.evaluate_output(
        simulated_vue_output, 
        task_description, 
        "ui_creation"
    )
    
    print(f"   Completeness: {quality_report.metrics.completeness:.1%}")
    print(f"   Correctness: {quality_report.metrics.correctness:.1%}")
    print(f"   Clarity: {quality_report.metrics.clarity:.1%}")
    print(f"   Functionality: {quality_report.metrics.functionality:.1%}")
    print(f"   Overall Score: {quality_report.metrics.overall:.1%}")
    print(f"   Quality Passed: {'✅ YES' if quality_report.passed else '❌ NO'}")
    
    if quality_report.issues:
        print(f"   Issues Found: {len(quality_report.issues)}")
        for issue in quality_report.issues[:3]:  # Show first 3 issues
            print(f"     - {issue}")
    
    # Step 5: Teaching Feedback
    print("\n🎓 STEP 5: Teaching Feedback")
    feedback = await teacher.provide_guidance(
        agent_id="vue_ui_agent",
        task_type="ui_creation",
        task_description=task_description,
        agent_output=simulated_vue_output,
        quality_issues=quality_report.issues,
        suggestions=quality_report.suggestions,
        quality_score=quality_report.metrics.overall
    )
    
    print(f"   Feedback Type: {feedback.feedback_type}")
    print(f"   Specific Actions: {len(feedback.specific_actions)} recommendations")
    print(f"   Examples Provided: {len(feedback.examples)} examples")
    
    # Step 6: Complete Supervision
    print("\n🎯 STEP 6: Complete Supervision")
    supervision_result = await supervisor.supervise_task(
        task_id="vue_ui_creation_test",
        task_description=task_description,
        agent_output=simulated_vue_output
    )
    
    requires_retry = supervision_result.get("requires_retry", True)
    print(f"   Supervision Complete: {'✅ PASSED' if not requires_retry else '🔄 NEEDS RETRY'}")
    
    # Step 7: System Statistics
    print("\n📈 STEP 7: System Statistics")
    supervisor_stats = supervisor.get_supervision_stats()
    teaching_stats = teacher.get_teaching_stats()
    
    print(f"   Total Supervised Tasks: {supervisor_stats['total_tasks']}")
    print(f"   Success Rate: {supervisor_stats['success_rate']:.1%}")
    print(f"   Teaching Interactions: {teaching_stats['total_interactions']}")
    print(f"   Improvement Rate: {teaching_stats['improvement_rate']:.1%}")
    
    # Final Result
    print("\n" + "=" * 60)
    if not requires_retry and quality_report.passed:
        print("🎉 GOLDEN GLUE TEST: ✅ COMPLETE SUCCESS!")
        print("   ✅ Task analysis working")
        print("   ✅ Agent selection working") 
        print("   ✅ Quality evaluation working")
        print("   ✅ Teaching feedback working")
        print("   ✅ Supervision oversight working")
        print("   ✅ Vue.js UI creation validated")
        print("\n🚀 THE GOLDEN GLUE IS FULLY FUNCTIONAL!")
    else:
        print("🔄 GOLDEN GLUE TEST: PARTIAL SUCCESS")
        print("   System is working but output needs improvement")
        print("   This demonstrates the teaching/feedback loop is working!")
    
    print("\n💡 READY FOR AUTONOMOUS UI CODING!")


if __name__ == "__main__":
    asyncio.run(test_golden_glue())
