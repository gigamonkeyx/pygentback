"""
Research-to-Analysis Orchestrator

Automated workflow that:
1. Conducts research using the Research Agent
2. Analyzes results using the Reasoning Agent with deepseek-r1
3. Provides comprehensive results with academic formatting
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List, Callable
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class WorkflowStatus(Enum):
    """Workflow execution status"""
    PENDING = "pending"
    RESEARCH_PHASE = "research_phase"
    ANALYSIS_PHASE = "analysis_phase"
    FORMATTING_PHASE = "formatting_phase"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class WorkflowProgress:
    """Progress tracking for the workflow"""
    status: WorkflowStatus
    current_step: str
    progress_percentage: float
    research_papers_found: int = 0
    analysis_confidence: float = 0.0
    estimated_time_remaining: Optional[float] = None
    error_message: Optional[str] = None


@dataclass
class ResearchAnalysisResult:
    """Complete result from the research-to-analysis workflow"""
    query: str
    research_data: Dict[str, Any]
    analysis_data: Dict[str, Any]
    formatted_output: str
    citations: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    execution_time: float
    success: bool
    error_message: Optional[str] = None


class ResearchAnalysisOrchestrator:
    """
    Orchestrates the complete Research-to-Analysis workflow
    """
    
    def __init__(self, agent_factory, progress_callback: Optional[Callable] = None):
        self.agent_factory = agent_factory
        self.progress_callback = progress_callback
        self.current_workflow_id: Optional[str] = None
        
    async def execute_workflow(
        self,
        query: str,
        analysis_model: str = "deepseek-r1:8b",
        max_papers: int = 15,
        analysis_depth: int = 3,
        workflow_id: Optional[str] = None
    ) -> ResearchAnalysisResult:
        """
        Execute the complete Research-to-Analysis workflow
        
        Args:
            query: Research query to investigate
            analysis_model: Model to use for analysis (default: deepseek-r1:8b)
            max_papers: Maximum papers to retrieve
            analysis_depth: Depth of Tree of Thought analysis
            workflow_id: Optional workflow ID for tracking
            
        Returns:
            Complete research and analysis results
        """
        start_time = datetime.now()
        self.current_workflow_id = workflow_id or f"workflow_{int(start_time.timestamp())}"
        
        try:
            logger.info(f"Starting Research-to-Analysis workflow for: {query}")
            
            # Phase 1: Research Phase
            await self._update_progress(
                WorkflowStatus.RESEARCH_PHASE,
                "Searching academic databases for relevant papers...",
                10.0
            )
            
            research_result = await self._execute_research_phase(query, max_papers)
            
            if not research_result["success"]:
                raise Exception(f"Research phase failed: {research_result.get('error', 'Unknown error')}")
            
            # Extract research metrics safely
            papers_analyzed = research_result.get('metadata', {}).get('papers_analyzed', 0)

            await self._update_progress(
                WorkflowStatus.RESEARCH_PHASE,
                f"Found {papers_analyzed} papers",
                40.0,
                research_papers_found=papers_analyzed
            )
            
            # Phase 2: Analysis Phase
            await self._update_progress(
                WorkflowStatus.ANALYSIS_PHASE,
                f"Analyzing research data with {analysis_model}...",
                50.0
            )
            
            analysis_result = await self._execute_analysis_phase(
                query, research_result, analysis_model, analysis_depth
            )
            
            if not analysis_result["success"]:
                raise Exception(f"Analysis phase failed: {analysis_result.get('error', 'Unknown error')}")
            
            await self._update_progress(
                WorkflowStatus.ANALYSIS_PHASE,
                "Analysis completed successfully",
                80.0,
                analysis_confidence=analysis_result.get('confidence', 0.8)
            )
            
            # Phase 3: Formatting Phase
            await self._update_progress(
                WorkflowStatus.FORMATTING_PHASE,
                "Formatting results with academic citations...",
                90.0
            )
            
            formatted_output, citations = await self._format_academic_output(
                query, research_result, analysis_result
            )
            
            # Complete workflow
            execution_time = (datetime.now() - start_time).total_seconds()
            
            await self._update_progress(
                WorkflowStatus.COMPLETED,
                "Workflow completed successfully",
                100.0
            )
            
            result = ResearchAnalysisResult(
                query=query,
                research_data=research_result,
                analysis_data=analysis_result,
                formatted_output=formatted_output,
                citations=citations,
                metadata={
                    "workflow_id": self.current_workflow_id,
                    "execution_time": execution_time,
                    "analysis_model": analysis_model,
                    "papers_analyzed": research_result.get('metadata', {}).get('papers_analyzed', 0),
                    "analysis_confidence": analysis_result.get('confidence', 0.8),
                    "timestamp": start_time.isoformat()
                },
                execution_time=execution_time,
                success=True
            )
            
            logger.info(f"Research-to-Analysis workflow completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Research-to-Analysis workflow failed: {e}")
            
            await self._update_progress(
                WorkflowStatus.FAILED,
                f"Workflow failed: {str(e)}",
                0.0,
                error_message=str(e)
            )
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return ResearchAnalysisResult(
                query=query,
                research_data={},
                analysis_data={},
                formatted_output="",
                citations=[],
                metadata={
                    "workflow_id": self.current_workflow_id,
                    "execution_time": execution_time,
                    "error": str(e)
                },
                execution_time=execution_time,
                success=False,
                error_message=str(e)
            )
    
    async def _execute_research_phase(self, query: str, max_papers: int) -> Dict[str, Any]:
        """Execute the research phase using the Research Agent"""
        try:
            # Create research agent
            research_agent = await self.agent_factory.create_agent(
                agent_type="research",
                name=f"research_agent_{self.current_workflow_id}",
                custom_config={
                    "enabled_capabilities": ["academic_research", "source_analysis"],
                    "max_papers": max_papers
                }
            )

            # Create proper AgentMessage for research request
            from core.agent import AgentMessage, MessageType

            research_message = AgentMessage(
                type=MessageType.REQUEST,
                sender="workflow_orchestrator",
                recipient=research_agent.agent_id,
                content={
                    "task": "research",
                    "prompt": query,
                    "research_type": "academic",
                    "max_papers": max_papers,
                    "context": {}
                }
            )

            # Execute research using proper agent interface
            result = await research_agent.process_message(research_message)

            # Extract response content
            if result and result.content:
                response_content = result.content.get("response", "")
                metadata = result.content.get("metadata", {})

                return {
                    "success": True,
                    "response": response_content,
                    "metadata": metadata,
                    "agent_id": research_agent.agent_id
                }
            else:
                return {
                    "success": False,
                    "error": "Research agent returned empty response",
                    "response": "",
                    "metadata": {}
                }

        except Exception as e:
            logger.error(f"Research phase failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "metadata": {}
            }
    
    async def _execute_analysis_phase(
        self, 
        query: str, 
        research_result: Dict[str, Any], 
        analysis_model: str,
        analysis_depth: int
    ) -> Dict[str, Any]:
        """Execute the analysis phase using the Reasoning Agent with deepseek-r1"""
        try:
            # Create reasoning agent with specified model
            reasoning_agent = await self.agent_factory.create_agent(
                agent_type="reasoning",
                name=f"analysis_agent_{self.current_workflow_id}",
                custom_config={
                    "model_name": analysis_model,
                    "enabled_capabilities": ["tree_of_thought", "document_analysis"],
                    "reasoning_depth": analysis_depth
                }
            )
            
            # Prepare analysis prompt
            analysis_prompt = self._create_analysis_prompt(query, research_result)
            
            # Execute analysis using ToT reasoning
            from core.agent import AgentMessage, MessageType

            analysis_message = AgentMessage(
                type=MessageType.REQUEST,
                sender="workflow_orchestrator",
                recipient=reasoning_agent.agent_id,
                content={
                    "task": "reasoning",
                    "prompt": analysis_prompt,
                    "reasoning_mode": "tree_of_thought",
                    "depth": analysis_depth,
                    "context": {}
                }
            )

            result = await reasoning_agent.process_message(analysis_message)

            # Extract response content properly
            if result and result.content:
                response_content = result.content.get("response", result.content)
                confidence = result.content.get("confidence", 0.8)
                reasoning_path = result.content.get("reasoning_path", [])

                return {
                    "success": True,
                    "response": response_content,
                    "confidence": confidence,
                    "reasoning_path": reasoning_path,
                    "agent_id": reasoning_agent.agent_id
                }
            else:
                return {
                    "success": False,
                    "error": "Reasoning agent returned empty response",
                    "response": "",
                    "confidence": 0.0
                }
            
        except Exception as e:
            logger.error(f"Analysis phase failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "response": "",
                "confidence": 0.0
            }
    
    def _create_analysis_prompt(self, query: str, research_result: Dict[str, Any]) -> str:
        """Create a comprehensive analysis prompt for the reasoning agent"""
        research_data = research_result.get("response", "")
        
        return f"""
Based on the following comprehensive research data about: "{query}"

RESEARCH DATA:
{research_data}

Please provide a thorough academic analysis that includes:

1. **Executive Summary**: Key findings and main conclusions from the research
2. **Arguments FOR**: Strongest evidence supporting the feasibility/viability
3. **Arguments AGAINST**: Key challenges, limitations, and counterarguments  
4. **Evidence Evaluation**: Assessment of the quality and strength of evidence
5. **Research Gaps**: Areas where more research is needed
6. **Future Outlook**: Predictions and conditions for success/failure
7. **Recommendations**: Actionable next steps based on the evidence

Use Tree of Thought reasoning to explore multiple perspectives and provide a balanced, evidence-based analysis. Consider the credibility of sources, recency of research, and methodological rigor in your evaluation.

Format your response with clear sections and academic rigor appropriate for scholarly publication.
"""
    
    async def _format_academic_output(
        self, 
        query: str, 
        research_result: Dict[str, Any], 
        analysis_result: Dict[str, Any]
    ) -> tuple[str, List[Dict[str, Any]]]:
        """Format the complete output with academic citations and professional formatting"""
        
        # Extract citations from research data
        citations = self._extract_citations(research_result)
        
        # Create formatted academic output
        formatted_output = f"""
# Research Analysis Report

**Research Query:** {query}

**Date:** {datetime.now().strftime("%B %d, %Y")}

**Methodology:** Automated literature review and AI-assisted analysis using Tree of Thought reasoning

---

## Research Summary

{research_result.get('response', 'Research data not available')}

---

## Analytical Assessment

{analysis_result.get('response', {}).get('solution', 'Analysis not available')}

---

## References

{self._format_citations(citations)}

---

## Methodology Notes

This report was generated using an automated Research-to-Analysis workflow that:
1. Conducted systematic searches of academic databases (arXiv, Semantic Scholar)
2. Applied relevance scoring and quality filtering to identify key papers
3. Used advanced AI reasoning (Tree of Thought) for comprehensive analysis
4. Formatted results according to academic standards

**Analysis Model:** {analysis_result.get('agent_id', 'Unknown')}
**Papers Analyzed:** {research_result.get('metadata', {}).get('papers_analyzed', 0)}
**Analysis Confidence:** {analysis_result.get('confidence', 0.0):.2f}
"""
        
        return formatted_output, citations
    
    def _extract_citations(self, research_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract citation information from research results"""
        citations = []

        # Try to extract real citations from research metadata
        metadata = research_result.get('metadata', {})

        # Check if there are papers in the metadata
        if 'papers' in metadata:
            papers = metadata['papers']
            for paper in papers[:10]:  # Limit to top 10 papers
                citation = {
                    "title": paper.get('title', 'Unknown Title'),
                    "authors": paper.get('authors', ['Unknown Author']),
                    "year": paper.get('publication_date', 'Unknown')[:4] if paper.get('publication_date') else 'Unknown',
                    "journal": paper.get('source', 'Unknown Source'),
                    "url": paper.get('url', ''),
                    "doi": paper.get('doi', '')
                }
                citations.append(citation)

        # If no real citations found, provide a minimal placeholder
        if not citations:
            citations = [
                {
                    "title": "Research data compiled from multiple academic sources",
                    "authors": ["PyGent Factory Research System"],
                    "year": "2024",
                    "journal": "Automated Research Compilation",
                    "url": "",
                    "doi": ""
                }
            ]

        return citations
    
    def _format_citations(self, citations: List[Dict[str, Any]]) -> str:
        """Format citations in academic style"""
        formatted_citations = []
        
        for i, citation in enumerate(citations, 1):
            authors = ", ".join(citation.get("authors", ["Unknown"]))
            title = citation.get("title", "Unknown Title")
            year = citation.get("year", "Unknown")
            journal = citation.get("journal", "Unknown Journal")
            url = citation.get("url", "")
            
            formatted_citation = f"{i}. {authors} ({year}). {title}. *{journal}*."
            if url:
                formatted_citation += f" Available at: {url}"
            
            formatted_citations.append(formatted_citation)
        
        return "\n".join(formatted_citations)
    
    async def _update_progress(
        self, 
        status: WorkflowStatus, 
        step: str, 
        percentage: float,
        research_papers_found: int = 0,
        analysis_confidence: float = 0.0,
        error_message: Optional[str] = None
    ):
        """Update workflow progress and notify callback if provided"""
        progress = WorkflowProgress(
            status=status,
            current_step=step,
            progress_percentage=percentage,
            research_papers_found=research_papers_found,
            analysis_confidence=analysis_confidence,
            error_message=error_message
        )
        
        logger.info(f"Workflow progress: {percentage:.1f}% - {step}")
        
        if self.progress_callback:
            await self.progress_callback(self.current_workflow_id, progress)
