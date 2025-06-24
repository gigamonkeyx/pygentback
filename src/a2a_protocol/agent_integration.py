#!/usr/bin/env python3
"""
A2A Agent Integration

Integrates existing PyGent Factory agents with the A2A protocol.
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any
from datetime import datetime

from .protocol import (
    A2AProtocol, AgentCard, AgentSkill, AgentCapabilities, 
    AgentAuthentication, AgentProvider, Task, TaskState, 
    TaskStatus, Message, TextPart, Artifact, a2a_protocol
)

logger = logging.getLogger(__name__)


class A2AAgentWrapper:
    """Wraps existing PyGent Factory agents for A2A protocol compatibility"""
    
    def __init__(self, agent, base_url: str = "http://localhost:8000"):
        self.agent = agent
        self.base_url = base_url
        self.agent_url = f"{base_url}/agents/{agent.agent_id}"
        self.a2a_protocol = a2a_protocol
        
        # Register this agent with A2A protocol
        asyncio.create_task(self._register_with_a2a())
    
    async def _register_with_a2a(self):
        """Register this agent with the A2A protocol"""
        try:
            agent_card = await self._create_agent_card()
            success = await self.a2a_protocol.register_agent(agent_card)
            if success:
                logger.info(f"Successfully registered {self.agent.name} with A2A protocol")
            else:
                logger.error(f"Failed to register {self.agent.name} with A2A protocol")
        except Exception as e:
            logger.error(f"Error registering agent {self.agent.name}: {e}")
    
    async def _create_agent_card(self) -> AgentCard:
        """Create A2A AgentCard from PyGent Factory agent"""
        
        # Map agent capabilities to A2A skills
        skills = []
        for capability in self.agent.capabilities:
            skill = AgentSkill(
                id=capability.name.lower().replace(" ", "_"),
                name=capability.name,
                description=capability.description,
                tags=self._extract_tags_from_capability(capability),
                examples=self._generate_examples_for_capability(capability),
                inputModes=["text/plain"],
                outputModes=["text/plain", "application/json"]
            )
            skills.append(skill)
        
        # Determine agent type-specific skills
        agent_type = str(self.agent.agent_type).lower()
        if "research" in agent_type:
            skills.extend(self._get_research_skills())
        elif "analysis" in agent_type:
            skills.extend(self._get_analysis_skills())
        elif "generation" in agent_type:
            skills.extend(self._get_generation_skills())
        
        return AgentCard(
            name=self.agent.name,
            description=f"PyGent Factory {agent_type.title()} Agent - {self.agent.name}",
            url=self.agent_url,
            version="1.0.0",
            provider=AgentProvider(
                organization="PyGent Factory",
                url="https://github.com/gigamonkeyx/pygentback"
            ),
            authentication=AgentAuthentication(
                schemes=["Bearer"],
                credentials=None
            ),
            defaultInputModes=["text/plain", "application/json"],
            defaultOutputModes=["text/plain", "application/json"],
            capabilities=AgentCapabilities(
                streaming=True,
                pushNotifications=False,
                stateTransitionHistory=True
            ),
            skills=skills,
            documentationUrl=f"{self.base_url}/docs/agents/{self.agent.agent_id}"
        )
    
    def _extract_tags_from_capability(self, capability) -> List[str]:
        """Extract tags from agent capability"""
        tags = []
        
        # Add capability-based tags
        if "search" in capability.name.lower():
            tags.extend(["search", "retrieval", "documents"])
        if "analysis" in capability.name.lower():
            tags.extend(["analysis", "statistics", "data"])
        if "generation" in capability.name.lower():
            tags.extend(["generation", "creation", "content"])
        if "coordination" in capability.name.lower():
            tags.extend(["coordination", "management", "workflow"])
        
        # Add agent type tags
        agent_type = str(self.agent.agent_type).lower()
        tags.append(agent_type)
        
        return list(set(tags))  # Remove duplicates
    
    def _generate_examples_for_capability(self, capability) -> List[str]:
        """Generate example prompts for capability"""
        examples = []
        
        capability_name = capability.name.lower()
        
        if "search" in capability_name:
            examples.extend([
                "Search for documents about machine learning",
                "Find research papers on neural networks",
                "Retrieve documents containing 'artificial intelligence'"
            ])
        elif "analysis" in capability_name:
            examples.extend([
                "Analyze the statistical trends in this dataset",
                "Perform correlation analysis on these variables",
                "Generate summary statistics for the data"
            ])
        elif "generation" in capability_name:
            examples.extend([
                "Generate a summary of this research paper",
                "Create a report based on the analysis results",
                "Write documentation for this process"
            ])
        elif "coordination" in capability_name:
            examples.extend([
                "Coordinate a multi-agent research task",
                "Manage workflow between analysis and generation agents",
                "Orchestrate document retrieval and processing"
            ])
        
        return examples
    
    def _get_research_skills(self) -> List[AgentSkill]:
        """Get research-specific A2A skills"""
        return [
            AgentSkill(
                id="document_search",
                name="Document Search",
                description="Search and retrieve documents from various sources",
                tags=["search", "documents", "retrieval", "research"],
                examples=[
                    "Search for papers about quantum computing",
                    "Find documents related to climate change research",
                    "Retrieve academic articles on machine learning"
                ],
                inputModes=["text/plain"],
                outputModes=["text/plain", "application/json"]
            ),
            AgentSkill(
                id="literature_review",
                name="Literature Review",
                description="Conduct comprehensive literature reviews",
                tags=["literature", "review", "analysis", "research"],
                examples=[
                    "Conduct a literature review on renewable energy",
                    "Review recent papers in artificial intelligence",
                    "Analyze trends in biomedical research"
                ]
            )
        ]
    
    def _get_analysis_skills(self) -> List[AgentSkill]:
        """Get analysis-specific A2A skills"""
        return [
            AgentSkill(
                id="statistical_analysis",
                name="Statistical Analysis",
                description="Perform statistical analysis on datasets",
                tags=["statistics", "analysis", "data", "math"],
                examples=[
                    "Calculate descriptive statistics for this dataset",
                    "Perform regression analysis on the variables",
                    "Test for statistical significance"
                ]
            ),
            AgentSkill(
                id="data_visualization",
                name="Data Visualization",
                description="Create visualizations and charts from data",
                tags=["visualization", "charts", "graphs", "data"],
                examples=[
                    "Create a histogram of the data distribution",
                    "Generate a scatter plot showing correlation",
                    "Build a dashboard for the analysis results"
                ]
            )
        ]
    
    def _get_generation_skills(self) -> List[AgentSkill]:
        """Get generation-specific A2A skills"""
        return [
            AgentSkill(
                id="content_generation",
                name="Content Generation",
                description="Generate various types of content and documents",
                tags=["generation", "content", "writing", "creation"],
                examples=[
                    "Generate a research summary from these papers",
                    "Create documentation for this analysis",
                    "Write a report based on the findings"
                ]
            ),
            AgentSkill(
                id="code_generation",
                name="Code Generation",
                description="Generate code and scripts for various tasks",
                tags=["code", "programming", "scripts", "automation"],
                examples=[
                    "Generate Python code for data analysis",
                    "Create a script to automate this process",
                    "Write functions for the specified requirements"
                ]
            )
        ]
    
    async def handle_a2a_task(self, task: Task) -> Task:
        """Handle an A2A task using the wrapped agent"""
        try:
            # Update task status to working
            task.status.state = TaskState.WORKING
            task.status.timestamp = datetime.utcnow().isoformat()
            await self.a2a_protocol.update_task_status(task.id, task.status)
            
            # Extract the latest message
            if not task.history:
                raise ValueError("No messages in task history")
            
            latest_message = task.history[-1]
            
            # Convert A2A message to agent-compatible format
            agent_input = self._convert_a2a_message_to_agent_input(latest_message)
            
            # Execute task based on agent type and message content
            result = await self._execute_agent_task(agent_input)
            
            # Convert result to A2A artifact
            artifact = self._convert_agent_result_to_artifact(result)
            await self.a2a_protocol.add_artifact(task.id, artifact)
            
            # Update task status to completed
            task.status.state = TaskState.COMPLETED
            task.status.timestamp = datetime.utcnow().isoformat()
            await self.a2a_protocol.update_task_status(task.id, task.status)
            
            logger.info(f"Successfully completed A2A task {task.id} with agent {self.agent.name}")
            return task
            
        except Exception as e:
            logger.error(f"Error handling A2A task {task.id}: {e}")
            
            # Update task status to failed
            task.status.state = TaskState.FAILED
            task.status.message = Message(
                role="agent",
                parts=[TextPart(text=f"Task failed: {str(e)}")],
                metadata={"error": str(e)}
            )
            task.status.timestamp = datetime.utcnow().isoformat()
            await self.a2a_protocol.update_task_status(task.id, task.status)
            
            return task
    
    def _convert_a2a_message_to_agent_input(self, message: Message) -> Dict[str, Any]:
        """Convert A2A message to agent-compatible input"""
        # Extract text from message parts
        text_content = ""
        for part in message.parts:
            if hasattr(part, 'text'):
                text_content += part.text + " "
        
        return {
            "query": text_content.strip(),
            "role": message.role,
            "metadata": message.metadata
        }
    
    async def _execute_agent_task(self, agent_input: Dict[str, Any]) -> Dict[str, Any]:
        """Execute task using the wrapped agent"""
        query = agent_input.get("query", "")
        
        # Determine task type based on agent type and query content
        agent_type = str(self.agent.agent_type).lower()
        
        if "research" in agent_type:
            # Use document search capability
            return await self.agent._search_documents({
                "query": query,
                "limit": 5
            })
        elif "analysis" in agent_type:
            # Use analysis capability
            if "statistical" in query.lower() or "analyze" in query.lower():
                # Generate sample data for demonstration
                sample_data = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
                return await self.agent._perform_statistical_analysis({
                    "dataset": sample_data,
                    "analysis_type": "descriptive"
                })
        elif "generation" in agent_type:
            # Use generation capability
            return await self.agent._generate_content({
                "prompt": query,
                "content_type": "text"
            })
        
        # Default response
        return {
            "result": f"Processed query: {query}",
            "agent": self.agent.name,
            "agent_type": agent_type,
            "timestamp": datetime.utcnow().isoformat()
        }
    
    def _convert_agent_result_to_artifact(self, result: Dict[str, Any]) -> Artifact:
        """Convert agent result to A2A artifact"""
        # Create text representation of result
        if isinstance(result, dict):
            text_content = json.dumps(result, indent=2)
        else:
            text_content = str(result)
        
        return Artifact(
            name="agent_result",
            description=f"Result from {self.agent.name}",
            parts=[TextPart(text=text_content)],
            metadata={
                "agent_id": self.agent.agent_id,
                "agent_name": self.agent.name,
                "agent_type": str(self.agent.agent_type),
                "timestamp": datetime.utcnow().isoformat()
            },
            index=0,
            lastChunk=True
        )


class A2AAgentRegistry:
    """Registry for A2A-enabled agents"""
    
    def __init__(self):
        self.wrapped_agents: Dict[str, A2AAgentWrapper] = {}
    
    async def register_agent(self, agent, base_url: str = "http://localhost:8000") -> A2AAgentWrapper:
        """Register an agent with A2A protocol"""
        wrapper = A2AAgentWrapper(agent, base_url)
        self.wrapped_agents[agent.agent_id] = wrapper
        logger.info(f"Registered agent {agent.name} with A2A registry")
        return wrapper
    
    async def get_agent(self, agent_id: str) -> Optional[A2AAgentWrapper]:
        """Get A2A-wrapped agent by ID"""
        return self.wrapped_agents.get(agent_id)
    
    async def list_agents(self) -> List[A2AAgentWrapper]:
        """List all registered A2A agents"""
        return list(self.wrapped_agents.values())
    
    async def handle_task_for_agent(self, agent_id: str, task: Task) -> Task:
        """Handle A2A task for specific agent"""
        wrapper = await self.get_agent(agent_id)
        if not wrapper:
            raise ValueError(f"Agent {agent_id} not found in A2A registry")
        
        return await wrapper.handle_a2a_task(task)


# Global A2A agent registry
a2a_agent_registry = A2AAgentRegistry()
