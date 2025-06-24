"""
Research Analysis Task for Tree of Thought

Specialized ToT implementation for academic research analysis
using multi-path reasoning to explore different analytical approaches.
"""

import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass

from ..models import ToTConfig, GenerationStrategy, EvaluationMethod, SearchMethod
from ..tot_engine import ToTEngine

logger = logging.getLogger(__name__)


@dataclass
class ResearchAnalysisConfig:
    """Configuration specific to research analysis"""
    analysis_types: List[str]      # e.g., ["thematic", "comparative", "historical"]
    research_domains: List[str]    # e.g., ["history", "science", "technology"]
    evidence_requirements: List[str] # e.g., ["primary sources", "peer review"]
    analysis_depth: str = "comprehensive"  # "surface", "moderate", "comprehensive"


class ResearchAnalysisTask:
    """
    Tree of Thought implementation for research analysis
    
    Uses multi-path reasoning to explore different analytical approaches,
    synthesize findings, and identify research gaps.
    """
    
    def __init__(self, config: Optional[ToTConfig] = None,
                 analysis_config: Optional[ResearchAnalysisConfig] = None):
        # Default ToT configuration for research analysis
        if config is None:
            config = ToTConfig(
                generation_strategy=GenerationStrategy.PROPOSE,
                evaluation_method=EvaluationMethod.VALUE,
                search_method=SearchMethod.BFS,
                n_generate_sample=4,
                n_evaluate_sample=3,
                n_select_sample=4,
                max_depth=8,
                temperature=0.6
            )
        
        self.config = config
        self.analysis_config = analysis_config or ResearchAnalysisConfig(
            analysis_types=["thematic", "comparative"],
            research_domains=["academic"],
            evidence_requirements=["peer review", "credible sources"]
        )
        
        self.tot_engine = ToTEngine(config)
    
    async def analyze_research_topic(self, topic: str,
                                   research_questions: List[str] = None,
                                   available_sources: List[Dict[str, Any]] = None,
                                   analysis_goals: List[str] = None) -> Dict[str, Any]:
        """
        Analyze a research topic using Tree of Thought reasoning
        
        Args:
            topic: Research topic to analyze
            research_questions: Specific questions to address
            available_sources: List of available research sources
            analysis_goals: Specific analysis objectives
            
        Returns:
            Dictionary containing analysis results and reasoning paths
        """
        # Prepare task context
        task_context = self._create_analysis_context(
            topic, research_questions, available_sources, analysis_goals
        )
        
        # Formulate the analysis problem
        problem = self._formulate_analysis_problem(
            topic, research_questions, analysis_goals
        )
        
        # Execute ToT reasoning
        result = await self.tot_engine.solve(problem, task_context)
        
        # Process and format results
        return self._process_analysis_results(result, topic)
    
    async def synthesize_literature(self, papers: List[Dict[str, Any]],
                                  synthesis_focus: str = "comprehensive") -> Dict[str, Any]:
        """
        Synthesize literature using multi-path reasoning
        
        Args:
            papers: List of research papers to synthesize
            synthesis_focus: Focus of synthesis ("themes", "gaps", "trends")
            
        Returns:
            Literature synthesis results
        """
        # Create synthesis-specific context
        task_context = self._create_synthesis_context(papers, synthesis_focus)
        
        # Formulate synthesis problem
        problem = f"Synthesize the following research literature with focus on {synthesis_focus}:\n"
        for i, paper in enumerate(papers[:10]):  # Limit to avoid token overflow
            problem += f"{i+1}. {paper.get('title', 'Unknown')}: {paper.get('abstract', 'No abstract')[:200]}...\n"
        
        # Execute synthesis reasoning
        result = await self.tot_engine.solve(problem, task_context)
        
        return self._process_synthesis_results(result, papers, synthesis_focus)
    
    async def identify_research_gaps(self, topic: str,
                                   existing_research: List[Dict[str, Any]],
                                   research_scope: str = "comprehensive") -> Dict[str, Any]:
        """
        Identify research gaps using systematic reasoning
        
        Args:
            topic: Research topic area
            existing_research: Current state of research
            research_scope: Scope of gap analysis
            
        Returns:
            Identified research gaps and opportunities
        """
        # Create gap analysis context
        task_context = self._create_gap_analysis_context(
            topic, existing_research, research_scope
        )
        
        # Formulate gap identification problem
        problem = self._formulate_gap_problem(topic, existing_research)
        
        # Execute gap analysis reasoning
        result = await self.tot_engine.solve(problem, task_context)
        
        return self._process_gap_analysis_results(result, topic)
    
    def _create_analysis_context(self, topic: str, research_questions: List[str],
                               available_sources: List[Dict[str, Any]],
                               analysis_goals: List[str]) -> Dict[str, Any]:
        """Create task-specific context for research analysis"""
        return {
            "topic": topic,
            "research_questions": research_questions or [],
            "available_sources": available_sources or [],
            "analysis_goals": analysis_goals or [],
            "analysis_types": self.analysis_config.analysis_types,
            "research_domains": self.analysis_config.research_domains,
            "evidence_requirements": self.analysis_config.evidence_requirements,
            "analysis_depth": self.analysis_config.analysis_depth,
            
            # Custom prompts for research analysis
            "propose_prompt": self._get_analysis_propose_prompt(),
            "value_prompt": self._get_analysis_value_prompt(),
            "solution_prompt": self._get_analysis_solution_prompt()
        }
    
    def _create_synthesis_context(self, papers: List[Dict[str, Any]],
                                synthesis_focus: str) -> Dict[str, Any]:
        """Create context for literature synthesis"""
        return {
            "papers": papers,
            "synthesis_focus": synthesis_focus,
            "paper_count": len(papers),
            
            "propose_prompt": self._get_synthesis_propose_prompt(),
            "value_prompt": self._get_synthesis_value_prompt(),
            "solution_prompt": self._get_synthesis_solution_prompt()
        }
    
    def _create_gap_analysis_context(self, topic: str,
                                   existing_research: List[Dict[str, Any]],
                                   research_scope: str) -> Dict[str, Any]:
        """Create context for research gap analysis"""
        return {
            "topic": topic,
            "existing_research": existing_research,
            "research_scope": research_scope,
            "research_count": len(existing_research),
            
            "propose_prompt": self._get_gap_propose_prompt(),
            "value_prompt": self._get_gap_value_prompt(),
            "solution_prompt": self._get_gap_solution_prompt()
        }
    
    def _formulate_analysis_problem(self, topic: str, research_questions: List[str],
                                  analysis_goals: List[str]) -> str:
        """Formulate the research analysis problem statement"""
        problem_parts = [f"Research Topic: {topic}"]
        
        if research_questions:
            questions_text = "\n".join([f"- {q}" for q in research_questions])
            problem_parts.append(f"Research Questions:\n{questions_text}")
        
        if analysis_goals:
            goals_text = "\n".join([f"- {g}" for g in analysis_goals])
            problem_parts.append(f"Analysis Goals:\n{goals_text}")
        
        analysis_types_text = ", ".join(self.analysis_config.analysis_types)
        problem_parts.append(f"Analysis Types: {analysis_types_text}")
        
        return "\n\n".join(problem_parts)
    
    def _formulate_gap_problem(self, topic: str,
                             existing_research: List[Dict[str, Any]]) -> str:
        """Formulate the research gap identification problem"""
        problem = f"Research Gap Analysis for: {topic}\n\n"
        problem += f"Existing Research ({len(existing_research)} studies):\n"
        
        for i, research in enumerate(existing_research[:5]):  # Limit for brevity
            title = research.get('title', 'Unknown Study')
            focus = research.get('focus', research.get('abstract', 'No description')[:100])
            problem += f"{i+1}. {title}: {focus}...\n"
        
        if len(existing_research) > 5:
            problem += f"... and {len(existing_research) - 5} more studies\n"
        
        return problem
    
    def _get_analysis_propose_prompt(self) -> str:
        """Get proposal prompt for research analysis"""
        return """Research Analysis Task: {task_description}

Current analysis step: {current_thought}
Depth: {current_depth}

Topic: {topic}
Research Questions: {research_questions}
Analysis Goals: {analysis_goals}
Analysis Types: {analysis_types}

Propose the next analytical step that builds on the current analysis. Consider:
- Available evidence and sources
- Analytical rigor and methodology
- Addressing research questions
- Meeting analysis goals

Next analytical step:"""
    
    def _get_analysis_value_prompt(self) -> str:
        """Get value evaluation prompt for research analysis"""
        return """Research Analysis Evaluation

Analysis step: {thought_content}

Evaluate this analytical step based on:
1. Methodological rigor: Is the approach sound?
2. Evidence quality: Does it use appropriate sources?
3. Analytical depth: How thorough is the analysis?
4. Relevance: Does it address the research questions?
5. Originality: Does it provide new insights?

Analysis requirements: {evidence_requirements}
Analysis depth: {analysis_depth}

Rate this analytical step on a scale of 0.0 to 1.0:

Score:"""
    
    def _get_analysis_solution_prompt(self) -> str:
        """Get solution check prompt for research analysis"""
        return """Research Analysis Solution Check

Analysis: {thought_content}

Does this represent a complete and rigorous analysis that:
- Addresses the main research questions
- Uses appropriate methodology
- Meets evidence requirements
- Provides meaningful insights

Answer yes or no:"""
    
    def _get_synthesis_propose_prompt(self) -> str:
        """Get proposal prompt for literature synthesis"""
        return """Literature Synthesis Task

Current synthesis: {current_thought}
Focus: {synthesis_focus}
Papers to synthesize: {paper_count}

Propose the next step in synthesizing the literature. Consider:
- Identifying common themes and patterns
- Noting contradictions and debates
- Highlighting methodological approaches
- Synthesizing key findings

Next synthesis step:"""
    
    def _get_synthesis_value_prompt(self) -> str:
        """Get value evaluation prompt for literature synthesis"""
        return """Literature Synthesis Evaluation

Synthesis step: {thought_content}
Focus: {synthesis_focus}

Evaluate based on:
1. Comprehensiveness: Does it cover key literature?
2. Accuracy: Are the syntheses accurate?
3. Insight: Does it provide meaningful connections?
4. Organization: Is it well-structured?

Score (0.0 to 1.0):"""
    
    def _get_synthesis_solution_prompt(self) -> str:
        """Get solution check prompt for literature synthesis"""
        return """Literature Synthesis Solution Check

Synthesis: {thought_content}

Is this a complete synthesis that effectively summarizes and connects the literature?

Answer yes or no:"""
    
    def _get_gap_propose_prompt(self) -> str:
        """Get proposal prompt for gap analysis"""
        return """Research Gap Analysis

Current gap analysis: {current_thought}
Topic: {topic}
Existing research: {research_count} studies

Propose the next step in identifying research gaps. Consider:
- Unexplored areas or questions
- Methodological limitations
- Temporal or geographical gaps
- Theoretical frameworks not applied

Next gap analysis step:"""
    
    def _get_gap_value_prompt(self) -> str:
        """Get value evaluation prompt for gap analysis"""
        return """Research Gap Analysis Evaluation

Gap analysis: {thought_content}

Evaluate based on:
1. Validity: Are these genuine gaps?
2. Significance: Are they important gaps?
3. Feasibility: Can they be researched?
4. Novelty: Are they previously unidentified?

Score (0.0 to 1.0):"""
    
    def _get_gap_solution_prompt(self) -> str:
        """Get solution check prompt for gap analysis"""
        return """Research Gap Analysis Solution Check

Gap analysis: {thought_content}

Does this identify clear, significant, and researchable gaps in the literature?

Answer yes or no:"""
    
    def _process_analysis_results(self, result, topic: str) -> Dict[str, Any]:
        """Process research analysis results"""
        return self._create_standard_result_format(result, topic, "research_analysis")
    
    def _process_synthesis_results(self, result, papers: List[Dict[str, Any]],
                                 synthesis_focus: str) -> Dict[str, Any]:
        """Process literature synthesis results"""
        processed = self._create_standard_result_format(result, synthesis_focus, "literature_synthesis")
        processed["papers_synthesized"] = len(papers)
        processed["synthesis_focus"] = synthesis_focus
        return processed
    
    def _process_gap_analysis_results(self, result, topic: str) -> Dict[str, Any]:
        """Process research gap analysis results"""
        return self._create_standard_result_format(result, topic, "gap_analysis")
    
    def _create_standard_result_format(self, result, topic: str, analysis_type: str) -> Dict[str, Any]:
        """Create standardized result format"""
        processed_result = {
            "analysis_type": analysis_type,
            "topic": topic,
            "success": result.success,
            "total_time": result.total_time,
            "reasoning_stats": {
                "nodes_explored": result.nodes_explored,
                "max_depth_reached": result.max_depth_reached,
                "total_llm_calls": result.total_llm_calls
            }
        }
        
        if result.best_solution:
            processed_result["primary_analysis"] = {
                "content": result.best_solution.content,
                "confidence_score": result.best_solution.value_score,
                "reasoning_path": self.tot_engine.format_solution_path(
                    result.best_solution, result.tree
                )
            }
            
            if len(result.all_solutions) > 1:
                processed_result["alternative_analyses"] = [
                    {
                        "content": sol.content,
                        "confidence_score": sol.value_score
                    }
                    for sol in sorted(result.all_solutions,
                                    key=lambda s: s.value_score, reverse=True)[1:]
                ]
        
        if not result.success:
            processed_result["error"] = result.error_message
        
        return processed_result
