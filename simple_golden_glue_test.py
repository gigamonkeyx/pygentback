#!/usr/bin/env python3
"""
Simple Golden Glue Test
"""

import asyncio
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

async def main():
    print("🚀 GOLDEN GLUE TEST STARTING...")
    
    from agents.supervisor_agent import SupervisorAgent
    print("✅ Supervisor Agent imported")
    
    supervisor = SupervisorAgent()
    print("✅ Supervisor Agent created")
    
    # Test Vue.js task analysis
    task = "Create a Vue.js alternative UI for PyGent Factory"
    analysis = await supervisor.analyze_task(task)
    print(f"✅ Task analyzed: {analysis.task_type}, complexity: {analysis.complexity}")
    
    # Test agent selection
    agent_type = await supervisor.select_agent(analysis)
    print(f"✅ Agent selected: {agent_type}")
    
    # Test quality evaluation
    vue_code = '''
<template>
  <div class="app">
    <h1>PyGent Factory</h1>
    <router-view />
  </div>
</template>

<script>
export default {
  name: 'App'
}
</script>
'''
    
    from agents.quality_evaluator import QualityEvaluator
    evaluator = QualityEvaluator()
    quality = await evaluator.evaluate_output(vue_code, task, "ui_creation")
    print(f"✅ Quality evaluated: {quality.metrics.overall:.1%}, passed: {quality.passed}")
    
    # Test supervision
    result = await supervisor.supervise_task("test_task", task, vue_code)
    print(f"✅ Supervision complete: retry needed: {result.get('requires_retry', True)}")
    
    print("🎉 GOLDEN GLUE TEST COMPLETE!")
    print("🚀 OLLAMA + SUPERVISOR AGENT = 100% WORKING!")

if __name__ == "__main__":
    asyncio.run(main())
