"""
Teaching System - Agent Learning and Improvement
Provides corrective guidance and learning facilitation for agents.
"""

import logging
from typing import Dict, List, Any, Optional
from dataclasses import dataclass
from datetime import datetime

logger = logging.getLogger(__name__)


@dataclass
class LearningRecord:
    """Record of agent learning interaction"""
    agent_id: str
    task_type: str
    original_output: str
    feedback_provided: str
    improvement_achieved: bool
    timestamp: datetime


@dataclass
class TeachingFeedback:
    """Structured teaching feedback"""
    feedback_type: str  # "corrective", "guidance", "encouragement"
    message: str
    specific_actions: List[str]
    examples: List[str]
    success_criteria: List[str]


class TeachingSystem:
    """
    Teaching System for Agent Improvement
    
    Provides:
    - Corrective feedback generation
    - Learning pattern recognition
    - Improvement tracking
    - Success pattern storage
    """
    
    def __init__(self):
        self.logger = logging.getLogger(f"{__name__}.TeachingSystem")
        self.learning_records: List[LearningRecord] = []
        self.success_patterns: Dict[str, List[str]] = {}
        self.common_issues: Dict[str, List[str]] = {}
    
    async def generate_corrective_feedback(self, 
                                         task_type: str,
                                         task_description: str,
                                         agent_output: str,
                                         quality_issues: List[str],
                                         suggestions: List[str]) -> TeachingFeedback:
        """Generate corrective feedback for agent improvement"""
        
        feedback_parts = []
        specific_actions = []
        examples = []
        success_criteria = []
        
        # Start with encouraging tone
        feedback_parts.append("Let's improve this output together.")
        
        # Address specific issues
        if quality_issues:
            feedback_parts.append("\nAreas that need attention:")
            for issue in quality_issues:
                feedback_parts.append(f"• {issue}")
                
                # Generate specific actions based on issue type
                if "vue" in issue.lower() and "template" in issue.lower():
                    specific_actions.append("Add a <template> section with HTML structure")
                    examples.append("<template>\n  <div class=\"component-name\">\n    <!-- Your content here -->\n  </div>\n</template>")
                
                elif "script" in issue.lower():
                    specific_actions.append("Include a <script> section with component logic")
                    examples.append("<script>\nexport default {\n  name: 'ComponentName',\n  data() {\n    return {\n      // component data\n    }\n  }\n}\n</script>")
                
                elif "export" in issue.lower():
                    specific_actions.append("Add 'export default' to make the component usable")
                    examples.append("export default {\n  name: 'MyComponent'\n}")
                
                elif "indentation" in issue.lower():
                    specific_actions.append("Use consistent indentation (2 or 4 spaces)")
                    examples.append("Proper indentation:\nfunction example() {\n    return 'properly indented';\n}")
        
        # Add suggestions as actionable guidance
        if suggestions:
            feedback_parts.append("\nHere's how to improve:")
            for suggestion in suggestions:
                feedback_parts.append(f"• {suggestion}")
                specific_actions.append(suggestion)
        
        # Add task-specific guidance
        if "ui" in task_type.lower() or "vue" in task_description.lower():
            feedback_parts.append("\nFor Vue.js components, remember:")
            feedback_parts.append("• Every component needs <template>, <script>, and optionally <style>")
            feedback_parts.append("• Use PascalCase for component names")
            feedback_parts.append("• Export the component with 'export default'")
            
            success_criteria.extend([
                "Component has proper Vue.js structure",
                "Template section contains valid HTML",
                "Script section exports the component",
                "Component name follows PascalCase convention"
            ])
            
            examples.append("""Complete Vue component example:
<template>
  <div class="hello-world">
    <h1>{{ message }}</h1>
  </div>
</template>

<script>
export default {
  name: 'HelloWorld',
  data() {
    return {
      message: 'Hello from Vue!'
    }
  }
}
</script>

<style scoped>
.hello-world {
  text-align: center;
}
</style>""")
        
        elif "cod" in task_type.lower():
            feedback_parts.append("\nFor coding tasks, ensure:")
            feedback_parts.append("• Code is properly structured with functions/classes")
            feedback_parts.append("• Include error handling where appropriate")
            feedback_parts.append("• Add comments to explain complex logic")
            feedback_parts.append("• Use consistent formatting and indentation")
            
            success_criteria.extend([
                "Code follows proper structure",
                "Includes appropriate error handling",
                "Has clear comments and documentation",
                "Uses consistent formatting"
            ])
        
        # Combine feedback message
        message = "\n".join(feedback_parts)
        
        return TeachingFeedback(
            feedback_type="corrective",
            message=message,
            specific_actions=specific_actions,
            examples=examples,
            success_criteria=success_criteria
        )
    
    async def generate_encouragement(self, 
                                   task_type: str,
                                   quality_score: float) -> TeachingFeedback:
        """Generate encouraging feedback for good performance"""
        
        if quality_score >= 0.9:
            message = "Excellent work! Your output meets all quality standards."
        elif quality_score >= 0.8:
            message = "Great job! Your output is of high quality with minor room for improvement."
        elif quality_score >= 0.7:
            message = "Good work! You're meeting the basic requirements successfully."
        else:
            message = "You're making progress! Let's work together to improve the output."
        
        return TeachingFeedback(
            feedback_type="encouragement",
            message=message,
            specific_actions=[],
            examples=[],
            success_criteria=[]
        )
    
    async def provide_guidance(self,
                             agent_id: str,
                             task_type: str,
                             task_description: str,
                             agent_output: str,
                             quality_issues: List[str],
                             suggestions: List[str],
                             quality_score: float) -> TeachingFeedback:
        """Main method to provide appropriate guidance to agents"""
        
        try:
            if quality_score >= 0.7:
                # Provide encouragement for good work
                feedback = await self.generate_encouragement(task_type, quality_score)
            else:
                # Provide corrective feedback for improvement
                feedback = await self.generate_corrective_feedback(
                    task_type, task_description, agent_output, quality_issues, suggestions
                )
            
            # Record the learning interaction
            learning_record = LearningRecord(
                agent_id=agent_id,
                task_type=task_type,
                original_output=agent_output[:500],  # Truncate for storage
                feedback_provided=feedback.message[:500],  # Truncate for storage
                improvement_achieved=quality_score >= 0.7,
                timestamp=datetime.now()
            )
            
            self.learning_records.append(learning_record)
            
            # Update patterns
            await self._update_learning_patterns(task_type, quality_issues, quality_score >= 0.7)
            
            self.logger.info(f"Provided {feedback.feedback_type} feedback to agent {agent_id} for {task_type} task")
            
            return feedback
            
        except Exception as e:
            self.logger.error(f"Error providing guidance: {e}")
            
            # Return basic feedback on error
            return TeachingFeedback(
                feedback_type="guidance",
                message=f"I encountered an issue while analyzing your output: {str(e)}. Please try again.",
                specific_actions=["Retry the task with a different approach"],
                examples=[],
                success_criteria=["Task completed without errors"]
            )
    
    async def _update_learning_patterns(self, 
                                      task_type: str, 
                                      issues: List[str], 
                                      was_successful: bool) -> None:
        """Update learning patterns based on outcomes"""
        
        if was_successful:
            # Record success pattern
            if task_type not in self.success_patterns:
                self.success_patterns[task_type] = []
            
            # This is simplified - in a real system, we'd analyze what made it successful
            self.success_patterns[task_type].append("successful_completion")
        
        else:
            # Record common issues
            if task_type not in self.common_issues:
                self.common_issues[task_type] = []
            
            for issue in issues:
                if issue not in self.common_issues[task_type]:
                    self.common_issues[task_type].append(issue)
    
    def get_learning_history(self, agent_id: Optional[str] = None) -> List[LearningRecord]:
        """Get learning history for an agent or all agents"""
        
        if agent_id:
            return [record for record in self.learning_records if record.agent_id == agent_id]
        return self.learning_records
    
    def get_success_patterns(self, task_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Get success patterns for a task type or all task types"""
        
        if task_type:
            return {task_type: self.success_patterns.get(task_type, [])}
        return self.success_patterns
    
    def get_common_issues(self, task_type: Optional[str] = None) -> Dict[str, List[str]]:
        """Get common issues for a task type or all task types"""
        
        if task_type:
            return {task_type: self.common_issues.get(task_type, [])}
        return self.common_issues
    
    def get_teaching_stats(self) -> Dict[str, Any]:
        """Get teaching system statistics"""
        
        total_interactions = len(self.learning_records)
        if total_interactions == 0:
            return {
                "total_interactions": 0,
                "improvement_rate": 0.0,
                "most_common_issues": [],
                "success_patterns": {}
            }
        
        successful_interactions = sum(1 for record in self.learning_records if record.improvement_achieved)
        improvement_rate = successful_interactions / total_interactions
        
        # Get most common issues across all task types
        all_issues = []
        for issues_list in self.common_issues.values():
            all_issues.extend(issues_list)
        
        issue_counts = {}
        for issue in all_issues:
            issue_counts[issue] = issue_counts.get(issue, 0) + 1
        
        most_common_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            "total_interactions": total_interactions,
            "successful_interactions": successful_interactions,
            "improvement_rate": improvement_rate,
            "most_common_issues": [issue for issue, count in most_common_issues],
            "success_patterns": self.success_patterns,
            "task_type_distribution": self._get_task_type_distribution()
        }
    
    def _get_task_type_distribution(self) -> Dict[str, int]:
        """Get distribution of task types in learning records"""
        
        distribution = {}
        for record in self.learning_records:
            task_type = record.task_type
            distribution[task_type] = distribution.get(task_type, 0) + 1
        
        return distribution
