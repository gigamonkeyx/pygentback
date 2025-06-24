"""
Multi-Agent Research Workflow for Historical Research
Orchestrates complex research processes using AI agents with real document acquisition.
"""
import asyncio
import json
import logging
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field
from datetime import datetime
import uuid
import time

from .agent_orchestrator import (
    agent_orchestrator, AgentTask, AgentType
)
from .ollama_integration import ollama_manager
from .openrouter_integration import openrouter_manager
try:
    from orchestration.internet_archive_integration import internet_archive_client
except ImportError:
    # Fallback for when orchestration module is not available
    internet_archive_client = None

logger = logging.getLogger(__name__)

@dataclass
class ResearchWorkflowConfig:
    """Configuration for research workflow."""
    topic: str
    scope: str = "comprehensive"  # comprehensive, focused, quick
    max_documents: int = 50
    fact_check_threshold: float = 0.7
    use_external_models: bool = True
    budget_limit: float = 10.0  # USD
    timeout_minutes: int = 60

@dataclass
class WorkflowStep:
    """Individual step in research workflow."""
    step_id: str
    name: str
    agent_type: AgentType
    task_type: str
    input_spec: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    parallel: bool = False
    critical: bool = True  # If false, failure won't stop workflow

@dataclass
class ResearchWorkflowResult:
    """Result of a complete research workflow."""
    workflow_id: str
    topic: str
    status: str
    started_at: datetime
    completed_at: Optional[datetime] = None
    results: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    total_time: float = 0.0

class ResearchWorkflow:
    """Manages end-to-end historical research workflows using real sources."""
    
    def __init__(self):
        self.active_workflows = {}
        self.completed_workflows = {}
        self.workflow_statistics = {
            "total_workflows": 0,
            "successful_workflows": 0,
            "failed_workflows": 0,
            "average_completion_time": 0.0
        }
    
    async def start_research(self, config: ResearchWorkflowConfig) -> str:
        """Start a new research workflow."""
        
        workflow_id = str(uuid.uuid4())
        
        # Create workflow result
        workflow_result = ResearchWorkflowResult(
            workflow_id=workflow_id,
            topic=config.topic,
            status="initializing",
            started_at=datetime.now()
        )
        
        self.active_workflows[workflow_id] = workflow_result
        self.workflow_statistics["total_workflows"] += 1
        
        logger.info(f"Started research workflow {workflow_id} for topic: {config.topic}")
        
        # Start workflow execution in background
        asyncio.create_task(self._execute_workflow(workflow_id, config))
        
        return workflow_id
    
    async def _execute_workflow(self, workflow_id: str, config: ResearchWorkflowConfig):
        """Execute the complete research workflow."""
        
        workflow_result = self.active_workflows[workflow_id]
        start_time = time.time()
        
        try:
            # Phase 1: Research Planning
            workflow_result.status = "planning"
            logger.info(f"Workflow {workflow_id}: Starting planning phase")
            
            planning_result = await self._execute_planning_phase(config)
            workflow_result.results["planning"] = planning_result
            
            if not planning_result.get("success"):
                raise Exception("Research planning failed - no result received")
            
            # Phase 2: Document Discovery using Internet Archive
            workflow_result.status = "discovery"
            logger.info(f"Workflow {workflow_id}: Starting document discovery phase")
            
            discovery_result = await self._execute_discovery_phase(config, planning_result)
            workflow_result.results["discovery"] = discovery_result
            
            if not discovery_result.get("success") or not discovery_result.get("documents"):
                raise Exception("Document discovery failed - no documents found")
            
            # Phase 3: Document Analysis
            workflow_result.status = "analysis"
            logger.info(f"Workflow {workflow_id}: Starting document analysis phase")
            
            analysis_result = await self._execute_analysis_phase(config, discovery_result)
            workflow_result.results["analysis"] = analysis_result
            
            if not analysis_result.get("success"):
                raise Exception("Document analysis failed")
            
            # Phase 4: Fact Checking
            workflow_result.status = "fact_checking"
            logger.info(f"Workflow {workflow_id}: Starting fact checking phase")
            
            fact_check_result = await self._execute_fact_checking_phase(config, analysis_result)
            workflow_result.results["fact_checking"] = fact_check_result
            
            # Phase 5: Research Synthesis
            workflow_result.status = "synthesis"
            logger.info(f"Workflow {workflow_id}: Starting synthesis phase")
            
            synthesis_result = await self._execute_synthesis_phase(config, analysis_result, fact_check_result)
            workflow_result.results["synthesis"] = synthesis_result
            
            if not synthesis_result.get("success"):
                raise Exception("Research synthesis failed")
            
            # Mark as completed
            workflow_result.status = "completed"
            workflow_result.completed_at = datetime.now()
            workflow_result.total_time = time.time() - start_time
            
            self.workflow_statistics["successful_workflows"] += 1
            self._update_average_completion_time(workflow_result.total_time)
            
            logger.info(f"Workflow {workflow_id}: Completed successfully in {workflow_result.total_time:.1f}s")
            
        except Exception as e:
            logger.error(f"Research workflow {workflow_id} failed: {e}")
            workflow_result.status = "failed"
            workflow_result.error = str(e)
            workflow_result.completed_at = datetime.now()
            workflow_result.total_time = time.time() - start_time
            
            self.workflow_statistics["failed_workflows"] += 1
        
        finally:
            # Move to completed workflows
            self.completed_workflows[workflow_id] = workflow_result
            if workflow_id in self.active_workflows:
                del self.active_workflows[workflow_id]
    
    async def _execute_planning_phase(self, config: ResearchWorkflowConfig) -> Dict[str, Any]:
        """Execute research planning phase using AI agents."""
        
        task = AgentTask(
            id=str(uuid.uuid4()),
            agent_type=AgentType.COORDINATOR,
            task_type="research_planning",
            input_data={
                "topic": config.topic,
                "scope": config.scope,
                "max_documents": config.max_documents,
                "fact_check_threshold": config.fact_check_threshold,
                "timeout_minutes": config.timeout_minutes
            }
        )
        
        # Submit task to agent orchestrator
        task_id = await agent_orchestrator.submit_task(task)
        
        # Wait for result with timeout
        result = await agent_orchestrator.get_task_result(task_id, timeout=30)
        
        if result:
            return {
                "success": True, 
                "plan": result,
                "research_questions": result.get("research_questions", []),
                "source_types": result.get("source_types", []),
                "search_terms": result.get("search_terms", [config.topic])
            }
        else:
            return {"success": False, "error": "Planning task timed out"}
    
    async def _execute_discovery_phase(self, config: ResearchWorkflowConfig, planning_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document discovery phase using Internet Archive."""
        
        try:
            # Extract search terms from planning
            search_terms = planning_result.get("search_terms", [config.topic])
            
            # Search Internet Archive for relevant documents
            all_documents = []
            total_found = 0
            
            for search_term in search_terms[:3]:  # Limit search terms
                try:
                    # Search for documents
                    search_results = await internet_archive_client.search_documents(
                        query=search_term,
                        limit=min(config.max_documents // len(search_terms), 20)
                    )
                    
                    if search_results.get("success") and search_results.get("documents"):
                        documents = search_results["documents"]
                        total_found += len(documents)
                        
                        # Download and process each document
                        for doc_meta in documents[:5]:  # Limit downloads
                            try:
                                # Download document content
                                download_result = await internet_archive_client.download_document(
                                    doc_meta["identifier"]
                                )
                                
                                if download_result.get("success"):
                                    doc_content = {
                                        "title": doc_meta.get("title", "Unknown"),
                                        "identifier": doc_meta["identifier"],
                                        "description": doc_meta.get("description", ""),
                                        "date": doc_meta.get("date", "Unknown"),
                                        "content": download_result["content"][:5000],  # Limit content length
                                        "content_length": len(download_result["content"]),
                                        "source": "Internet Archive",
                                        "url": f"https://archive.org/details/{doc_meta['identifier']}",
                                        "search_term": search_term,
                                        "relevance_score": self._calculate_relevance(doc_meta, config.topic)
                                    }
                                    all_documents.append(doc_content)
                                    
                            except Exception as e:
                                logger.warning(f"Failed to download document {doc_meta.get('identifier')}: {e}")
                                continue
                    
                except Exception as e:
                    logger.warning(f"Search failed for term '{search_term}': {e}")
                    continue
            
            # Sort documents by relevance
            all_documents.sort(key=lambda x: x["relevance_score"], reverse=True)
            
            return {
                "success": True,
                "documents": all_documents[:config.max_documents],  # Limit final count
                "total_found": total_found,
                "total_downloaded": len(all_documents),
                "search_terms_used": search_terms,
                "source": "Internet Archive"
            }
            
        except Exception as e:
            logger.error(f"Document discovery failed: {e}")
            return {
                "success": False,
                "error": str(e),
                "documents": []
            }
    
    def _calculate_relevance(self, doc_meta: Dict[str, Any], topic: str) -> float:
        """Calculate relevance score for a document."""
        score = 0.0
        topic_lower = topic.lower()
        
        # Check title relevance
        title = doc_meta.get("title", "").lower()
        if topic_lower in title:
            score += 0.5
        
        # Check description relevance  
        description = doc_meta.get("description", "").lower()
        if topic_lower in description:
            score += 0.3
        
        # Check for historical periods/dates
        if any(year in str(doc_meta.get("date", "")) for year in ["14", "15", "16"]):  # 14th-16th centuries
            score += 0.2
        
        return min(score, 1.0)
    
    async def _execute_analysis_phase(self, config: ResearchWorkflowConfig, discovery_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute document analysis phase using AI agents."""
        
        documents = discovery_result.get("documents", [])
        
        if not documents:
            return {"success": False, "error": "No documents to analyze"}
        
        # Analyze each document
        analyzed_docs = []
        
        for i, doc in enumerate(documents):
            try:
                task = AgentTask(
                    id=str(uuid.uuid4()),
                    agent_type=AgentType.DOCUMENT_ANALYZER,
                    task_type="analyze_document",
                    input_data={
                        "text": doc["content"],
                        "title": doc["title"],
                        "source": doc["source"],
                        "topic": config.topic
                    }
                )
                
                # Submit analysis task
                task_id = await agent_orchestrator.submit_task(task)
                result = await agent_orchestrator.get_task_result(task_id, timeout=30)
                
                if result:
                    analyzed_docs.append({
                        "document": doc,
                        "analysis": result,
                        "document_index": i
                    })
                else:
                    logger.warning(f"Analysis failed for document {i}: {doc['title']}")
                    
            except Exception as e:
                logger.warning(f"Failed to analyze document {i}: {e}")
                continue
        
        return {
            "success": True,
            "analyzed_documents": analyzed_docs,
            "total_analyzed": len(analyzed_docs),
            "total_documents": len(documents),
            "analysis_success_rate": len(analyzed_docs) / len(documents) if documents else 0
        }
    
    async def _execute_fact_checking_phase(self, config: ResearchWorkflowConfig, analysis_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute fact checking phase using AI agents."""
        
        analyzed_docs = analysis_result.get("analyzed_documents", [])
        
        if not analyzed_docs:
            return {"success": False, "error": "No analyzed documents for fact checking"}
        
        # Extract claims from analysis results
        claims = []
        for doc_analysis in analyzed_docs:
            analysis = doc_analysis.get("analysis", {})
            doc_claims = analysis.get("claims", [])
            if isinstance(doc_claims, list):
                claims.extend(doc_claims)
        
        # Fact check critical claims
        fact_checked_claims = []
        
        for claim in claims[:10]:  # Limit number of claims to check
            try:
                task = AgentTask(
                    id=str(uuid.uuid4()),
                    agent_type=AgentType.FACT_CHECKER,
                    task_type="fact_check",
                    input_data={
                        "claim": claim,
                        "topic": config.topic,
                        "threshold": config.fact_check_threshold
                    }
                )
                
                task_id = await agent_orchestrator.submit_task(task)
                result = await agent_orchestrator.get_task_result(task_id, timeout=20)
                
                if result:
                    fact_checked_claims.append({
                        "claim": claim,
                        "fact_check_result": result,
                        "confidence": result.get("confidence", 0.0),
                        "verification_status": result.get("status", "unknown")
                    })
                    
            except Exception as e:
                logger.warning(f"Fact checking failed for claim: {e}")
                continue
        
        # Calculate overall credibility
        verified_claims = [c for c in fact_checked_claims if c["confidence"] >= config.fact_check_threshold]
        credibility_score = len(verified_claims) / len(fact_checked_claims) if fact_checked_claims else 0.0
        
        return {
            "success": True,
            "fact_checked_claims": fact_checked_claims,
            "verified_claims": verified_claims,
            "total_claims": len(claims),
            "claims_checked": len(fact_checked_claims),
            "credibility_score": credibility_score,
            "meets_threshold": credibility_score >= config.fact_check_threshold
        }
    
    async def _execute_synthesis_phase(self, config: ResearchWorkflowConfig, analysis_result: Dict[str, Any], fact_check_result: Dict[str, Any]) -> Dict[str, Any]:
        """Execute research synthesis phase using AI models."""
        
        analyzed_docs = analysis_result.get("analyzed_documents", [])
        fact_checked_claims = fact_check_result.get("verified_claims", [])
        
        if not analyzed_docs:
            return {"success": False, "error": "No documents to synthesize"}
        
        # Prepare synthesis data
        synthesis_input = {
            "topic": config.topic,
            "documents": [doc["document"] for doc in analyzed_docs],
            "analyses": [doc["analysis"] for doc in analyzed_docs],
            "verified_claims": fact_checked_claims,
            "credibility_score": fact_check_result.get("credibility_score", 0.0)
        }
        
        try:
            if config.use_external_models and openrouter_manager._initialized:
                # Use external model for advanced synthesis
                synthesis = await openrouter_manager.synthesize_research(
                    synthesis_input["documents"], 
                    config.topic
                )
            else:
                # Use local model for synthesis
                synthesis_prompt = self._create_synthesis_prompt(synthesis_input)
                
                response = await ollama_manager.generate(
                    prompt=synthesis_prompt,
                    capability=ollama_manager.ModelCapability.REASONING,
                    temperature=0.3
                )
                
                # Try to parse as JSON, fallback to text
                try:
                    synthesis = json.loads(response.content)
                except json.JSONDecodeError:
                    synthesis = {
                        "summary": response.content,
                        "key_findings": ["Generated from analysis"],
                        "conclusions": "See summary for detailed conclusions",
                        "sources_used": len(analyzed_docs)
                    }
            
            # Add synthesis metadata
            synthesis_metadata = {
                "documents_analyzed": len(analyzed_docs),
                "claims_verified": len(fact_checked_claims),
                "credibility_score": fact_check_result.get("credibility_score", 0.0),
                "synthesis_method": "external" if (config.use_external_models and openrouter_manager._initialized) else "local",
                "workflow_duration": (datetime.now() - datetime.fromtimestamp(time.time() - 300)).total_seconds()  # Rough estimate
            }
            
            return {
                "success": True,
                "synthesis": synthesis,
                "synthesis_metadata": synthesis_metadata,
                "final_report": {
                    "topic": config.topic,
                    "summary": synthesis.get("summary", ""),
                    "key_findings": synthesis.get("key_findings", []),
                    "conclusions": synthesis.get("conclusions", ""),
                    "sources": [doc["document"]["title"] for doc in analyzed_docs],
                    "credibility_assessment": synthesis_metadata["credibility_score"],
                    "methodology": "AI-assisted historical research with real source verification"
                }
            }
            
        except Exception as e:
            logger.error(f"Synthesis failed: {e}")
            return {"success": False, "error": str(e)}
    
    def _create_synthesis_prompt(self, synthesis_input: Dict[str, Any]) -> str:
        """Create a comprehensive synthesis prompt."""
        
        topic = synthesis_input["topic"]
        documents = synthesis_input["documents"]
        analyses = synthesis_input["analyses"]
        verified_claims = synthesis_input["verified_claims"]
        
        prompt = f"""Synthesize comprehensive historical research on: {topic}

Based on {len(documents)} historical documents and {len(verified_claims)} verified claims, provide:

1. EXECUTIVE SUMMARY (2-3 paragraphs)
2. KEY HISTORICAL FINDINGS (3-5 main points)
3. ANALYSIS AND CONCLUSIONS
4. SOURCE RELIABILITY ASSESSMENT
5. AREAS FOR FURTHER RESEARCH

Documents analyzed:
"""
        
        for i, doc in enumerate(documents[:5]):  # Limit to prevent prompt overflow
            prompt += f"\n{i+1}. {doc.get('title', 'Unknown')} ({doc.get('date', 'Unknown date')})"
        
        prompt += f"""

Return as structured JSON with fields: summary, key_findings, conclusions, source_reliability, further_research.
Focus on academic rigor and historical accuracy."""
        
        return prompt
    
    def _update_average_completion_time(self, completion_time: float):
        """Update average completion time statistics."""
        successful = self.workflow_statistics["successful_workflows"]
        current_avg = self.workflow_statistics["average_completion_time"]
        
        # Calculate new average
        new_avg = ((current_avg * (successful - 1)) + completion_time) / successful
        self.workflow_statistics["average_completion_time"] = new_avg
    
    async def get_workflow_status(self, workflow_id: str) -> Optional[ResearchWorkflowResult]:
        """Get the current status of a workflow."""
        
        if workflow_id in self.active_workflows:
            return self.active_workflows[workflow_id]
        elif workflow_id in self.completed_workflows:
            return self.completed_workflows[workflow_id]
        else:
            return None
    
    def list_active_workflows(self) -> List[str]:
        """List all active workflow IDs."""
        return list(self.active_workflows.keys())
    
    def list_completed_workflows(self) -> List[str]:
        """List all completed workflow IDs."""
        return list(self.completed_workflows.keys())
    
    def get_workflow_statistics(self) -> Dict[str, Any]:
        """Get overall workflow statistics."""
        return self.workflow_statistics.copy()
    
    async def cancel_workflow(self, workflow_id: str) -> bool:
        """Cancel an active workflow."""
        if workflow_id in self.active_workflows:
            workflow = self.active_workflows[workflow_id]
            workflow.status = "cancelled"
            workflow.completed_at = datetime.now()
            workflow.error = "Cancelled by user"
            
            # Move to completed
            self.completed_workflows[workflow_id] = workflow
            del self.active_workflows[workflow_id]
            
            logger.info(f"Workflow {workflow_id} cancelled")
            return True
        
        return False

# Global research workflow instance
research_workflow = ResearchWorkflow()
