#!/usr/bin/env python3
"""
A2A Agent Discovery Mechanism

Implements agent discovery system using Agent Cards at well-known URLs according to Google A2A specification.
Enables automatic agent finding and capability negotiation.
"""

import asyncio
import aiohttp
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from urllib.parse import urljoin, urlparse
import hashlib

logger = logging.getLogger(__name__)


@dataclass
class DiscoveredAgent:
    """Information about a discovered agent"""
    agent_id: str
    name: str
    description: str
    url: str
    well_known_url: str
    capabilities: Dict[str, Any]
    skills: List[Dict[str, Any]]
    provider: Dict[str, Any]
    security_schemes: Dict[str, Any] = field(default_factory=dict)
    discovered_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_verified: Optional[str] = None
    available: bool = True
    response_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class DiscoveryConfig:
    """Configuration for agent discovery"""
    discovery_timeout_seconds: int = 10
    verification_interval_minutes: int = 30
    max_concurrent_discoveries: int = 10
    cache_duration_minutes: int = 60
    retry_attempts: int = 3
    retry_delay_seconds: float = 1.0
    user_agent: str = "PyGent-Factory-A2A-Discovery/1.0"


@dataclass
class CapabilityMatch:
    """Result of capability matching"""
    agent_id: str
    agent_name: str
    match_score: float  # 0.0 to 1.0
    matching_skills: List[str]
    missing_capabilities: List[str]
    agent_url: str


class A2AAgentDiscovery:
    """A2A Agent Discovery System"""
    
    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self.discovered_agents: Dict[str, DiscoveredAgent] = {}
        self.discovery_cache: Dict[str, Dict[str, Any]] = {}
        self.verification_task: Optional[asyncio.Task] = None
        self.session: Optional[aiohttp.ClientSession] = None
        
        # Known agent registries and directories
        self.agent_registries: Set[str] = set()
        
        # Discovery patterns
        self.well_known_patterns = [
            "/.well-known/agent.json",
            "/.well-known/agent/{agent_id}.json",
            "/a2a/agents/{agent_id}/card"
        ]
        
        # Start verification task (only if event loop is running)
        self.verification_task = None
        try:
            # Check if there's a running event loop
            loop = asyncio.get_running_loop()
            if loop and not loop.is_closed():
                self.verification_task = asyncio.create_task(self._periodic_verification())
        except RuntimeError:
            # No event loop running, skip background tasks
            pass
    
    async def __aenter__(self):
        """Async context manager entry"""
        self.session = aiohttp.ClientSession(
            timeout=aiohttp.ClientTimeout(total=self.config.discovery_timeout_seconds),
            headers={"User-Agent": self.config.user_agent}
        )
        return self
    
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        await self.cleanup()

    async def cleanup(self):
        """Clean up resources and tasks"""
        # Close HTTP session
        if self.session:
            await self.session.close()
            self.session = None

        # Cancel verification task properly
        if self.verification_task and not self.verification_task.done():
            self.verification_task.cancel()
            try:
                await self.verification_task
            except asyncio.CancelledError:
                pass
            self.verification_task = None

    def __del__(self):
        """Destructor to ensure cleanup"""
        if self.verification_task and not self.verification_task.done():
            # If there's still a running task, try to cancel it
            try:
                self.verification_task.cancel()
            except Exception:
                pass

    async def register_agent(self, agent_id: str, agent_card: Dict[str, Any]) -> bool:
        """Register an agent with the discovery system"""
        try:
            # Add agent to registry
            self.agent_registry[agent_id] = {
                "agent_card": agent_card,
                "registered_at": datetime.utcnow().isoformat(),
                "last_seen": datetime.utcnow().isoformat(),
                "status": "active"
            }

            logger.info(f"Registered agent {agent_id} with A2A discovery")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id}: {e}")
            return False

    def register_agent_sync(self, agent_id: str, agent_card: Dict[str, Any]) -> bool:
        """Synchronous version of register_agent"""
        try:
            # Add agent to registry
            self.agent_registry[agent_id] = {
                "agent_card": agent_card,
                "registered_at": datetime.utcnow().isoformat(),
                "last_seen": datetime.utcnow().isoformat(),
                "status": "active"
            }

            logger.info(f"Registered agent {agent_id} with A2A discovery (sync)")
            return True

        except Exception as e:
            logger.error(f"Failed to register agent {agent_id} (sync): {e}")
            return False

    async def discover_agent_at_url(self, base_url: str, agent_id: Optional[str] = None) -> List[DiscoveredAgent]:
        """Discover agent(s) at a specific URL"""
        if not self.session:
            raise RuntimeError("Discovery session not initialized. Use async context manager.")
        
        discovered = []
        base_url = base_url.rstrip('/')
        
        try:
            # Try different well-known URL patterns
            for pattern in self.well_known_patterns:
                if agent_id and "{agent_id}" in pattern:
                    url = base_url + pattern.format(agent_id=agent_id)
                elif "{agent_id}" not in pattern:
                    url = base_url + pattern
                else:
                    continue
                
                agent = await self._fetch_agent_card(url, base_url)
                if agent:
                    discovered.append(agent)
                    
                    # If we found an agent list, try to discover individual agents
                    if not agent_id and "agents" in agent.metadata:
                        for agent_info in agent.metadata.get("agents", []):
                            individual_agent_id = agent_info.get("id")
                            if individual_agent_id:
                                individual_agents = await self.discover_agent_at_url(base_url, individual_agent_id)
                                discovered.extend(individual_agents)
            
            # Store discovered agents
            for agent in discovered:
                self.discovered_agents[agent.agent_id] = agent
                logger.info(f"Discovered agent {agent.name} ({agent.agent_id}) at {agent.url}")
            
            return discovered
            
        except Exception as e:
            logger.error(f"Error discovering agents at {base_url}: {e}")
            return []
    
    async def _fetch_agent_card(self, url: str, base_url: str) -> Optional[DiscoveredAgent]:
        """Fetch and parse agent card from URL"""
        try:
            start_time = asyncio.get_event_loop().time()
            
            async with self.session.get(url) as response:
                if response.status == 200:
                    content_type = response.headers.get('content-type', '')
                    if 'application/json' in content_type:
                        data = await response.json()
                        end_time = asyncio.get_event_loop().time()
                        response_time = (end_time - start_time) * 1000
                        
                        return self._parse_agent_card(data, url, base_url, response_time)
                    else:
                        logger.debug(f"Non-JSON response from {url}: {content_type}")
                else:
                    logger.debug(f"HTTP {response.status} from {url}")
                    
        except asyncio.TimeoutError:
            logger.debug(f"Timeout fetching agent card from {url}")
        except Exception as e:
            logger.debug(f"Error fetching agent card from {url}: {e}")
        
        return None
    
    def _parse_agent_card(self, data: Dict[str, Any], well_known_url: str, 
                         base_url: str, response_time: float) -> Optional[DiscoveredAgent]:
        """Parse agent card data into DiscoveredAgent"""
        try:
            # Handle agent list response
            if "agents" in data and isinstance(data["agents"], list):
                # This is a list of agents, create a virtual agent representing the service
                return DiscoveredAgent(
                    agent_id=f"service_{hashlib.md5(base_url.encode()).hexdigest()[:8]}",
                    name=f"Agent Service at {urlparse(base_url).netloc}",
                    description=f"Agent service hosting {len(data['agents'])} agents",
                    url=base_url,
                    well_known_url=well_known_url,
                    capabilities={"multi_agent": True, "agent_count": len(data["agents"])},
                    skills=[],
                    provider={"name": "Unknown", "organization": "Unknown"},
                    response_time_ms=response_time,
                    metadata=data
                )
            
            # Handle individual agent card
            required_fields = ["name", "description", "url", "capabilities", "skills"]
            if not all(field in data for field in required_fields):
                logger.debug(f"Agent card missing required fields: {well_known_url}")
                return None
            
            # Extract agent ID from URL or generate one
            agent_id = data.get("metadata", {}).get("agent_id")
            if not agent_id:
                # Try to extract from URL
                url_path = urlparse(data["url"]).path
                if "/agents/" in url_path:
                    agent_id = url_path.split("/agents/")[-1].split("/")[0]
                else:
                    agent_id = hashlib.md5(data["url"].encode()).hexdigest()[:16]
            
            return DiscoveredAgent(
                agent_id=agent_id,
                name=data["name"],
                description=data["description"],
                url=data["url"],
                well_known_url=well_known_url,
                capabilities=data["capabilities"],
                skills=data["skills"],
                provider=data.get("provider", {"name": "Unknown", "organization": "Unknown"}),
                security_schemes=data.get("securitySchemes", {}),
                response_time_ms=response_time,
                metadata=data.get("metadata", {})
            )
            
        except Exception as e:
            logger.error(f"Error parsing agent card from {well_known_url}: {e}")
            return None
    
    async def discover_agents_in_network(self, base_urls: List[str]) -> List[DiscoveredAgent]:
        """Discover agents across multiple network locations"""
        if not self.session:
            raise RuntimeError("Discovery session not initialized. Use async context manager.")
        
        # Limit concurrent discoveries
        semaphore = asyncio.Semaphore(self.config.max_concurrent_discoveries)
        
        async def discover_with_semaphore(url):
            async with semaphore:
                return await self.discover_agent_at_url(url)
        
        # Discover agents concurrently
        tasks = [discover_with_semaphore(url) for url in base_urls]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Flatten results
        all_discovered = []
        for result in results:
            if isinstance(result, list):
                all_discovered.extend(result)
            elif isinstance(result, Exception):
                logger.error(f"Discovery error: {result}")
        
        return all_discovered
    
    async def find_agents_by_capability(self, required_capabilities: List[str],
                                      required_skills: Optional[List[str]] = None) -> List[CapabilityMatch]:
        """Find agents that match required capabilities and skills"""
        matches = []
        
        for agent in self.discovered_agents.values():
            if not agent.available:
                continue
            
            match_score, matching_skills, missing_capabilities = self._calculate_capability_match(
                agent, required_capabilities, required_skills or []
            )
            
            if match_score > 0:
                matches.append(CapabilityMatch(
                    agent_id=agent.agent_id,
                    agent_name=agent.name,
                    match_score=match_score,
                    matching_skills=matching_skills,
                    missing_capabilities=missing_capabilities,
                    agent_url=agent.url
                ))
        
        # Sort by match score (highest first)
        matches.sort(key=lambda x: x.match_score, reverse=True)
        return matches
    
    def _calculate_capability_match(self, agent: DiscoveredAgent, 
                                  required_capabilities: List[str],
                                  required_skills: List[str]) -> Tuple[float, List[str], List[str]]:
        """Calculate how well an agent matches required capabilities"""
        # Check capabilities
        agent_capabilities = set()
        if isinstance(agent.capabilities, dict):
            # Extract capability names from capabilities object
            if "streaming" in agent.capabilities and agent.capabilities["streaming"]:
                agent_capabilities.add("streaming")
            if "pushNotifications" in agent.capabilities and agent.capabilities["pushNotifications"]:
                agent_capabilities.add("push_notifications")
            if "stateTransitionHistory" in agent.capabilities and agent.capabilities["stateTransitionHistory"]:
                agent_capabilities.add("state_history")
        
        # Check skills
        agent_skills = set()
        for skill in agent.skills:
            if isinstance(skill, dict):
                skill_id = skill.get("id", "")
                skill_tags = skill.get("tags", [])
                agent_skills.add(skill_id)
                agent_skills.update(skill_tags)
        
        # Calculate matches
        required_caps_set = set(required_capabilities)
        required_skills_set = set(required_skills)
        
        matching_capabilities = agent_capabilities.intersection(required_caps_set)
        matching_skills = agent_skills.intersection(required_skills_set)
        
        missing_capabilities = list(required_caps_set - agent_capabilities)
        
        # Calculate score
        total_required = len(required_capabilities) + len(required_skills)
        if total_required == 0:
            return 1.0, list(matching_skills), missing_capabilities
        
        total_matches = len(matching_capabilities) + len(matching_skills)
        match_score = total_matches / total_required
        
        return match_score, list(matching_skills), missing_capabilities
    
    async def verify_agent_availability(self, agent_id: str) -> bool:
        """Verify that a discovered agent is still available"""
        if agent_id not in self.discovered_agents:
            return False
        
        agent = self.discovered_agents[agent_id]
        
        try:
            # Try to fetch agent card again
            updated_agent = await self._fetch_agent_card(agent.well_known_url, agent.url)
            if updated_agent:
                # Update agent information
                agent.last_verified = datetime.utcnow().isoformat()
                agent.available = True
                agent.response_time_ms = updated_agent.response_time_ms
                return True
            else:
                agent.available = False
                return False
                
        except Exception as e:
            logger.debug(f"Error verifying agent {agent_id}: {e}")
            agent.available = False
            return False
    
    async def _periodic_verification(self):
        """Periodically verify discovered agents"""
        while True:
            try:
                await asyncio.sleep(self.config.verification_interval_minutes * 60)
                
                if not self.session:
                    continue
                
                # Verify all discovered agents
                verification_tasks = []
                for agent_id in list(self.discovered_agents.keys()):
                    task = asyncio.create_task(self.verify_agent_availability(agent_id))
                    verification_tasks.append(task)
                
                if verification_tasks:
                    await asyncio.gather(*verification_tasks, return_exceptions=True)
                    
                    # Log verification results
                    available_count = sum(1 for agent in self.discovered_agents.values() if agent.available)
                    total_count = len(self.discovered_agents)
                    logger.info(f"Agent verification complete: {available_count}/{total_count} agents available")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic verification: {e}")
    
    def get_discovered_agents(self, available_only: bool = True) -> List[DiscoveredAgent]:
        """Get list of discovered agents"""
        agents = list(self.discovered_agents.values())
        if available_only:
            agents = [agent for agent in agents if agent.available]
        return agents
    
    def get_agent_by_id(self, agent_id: str) -> Optional[DiscoveredAgent]:
        """Get discovered agent by ID"""
        return self.discovered_agents.get(agent_id)
    
    def add_agent_registry(self, registry_url: str):
        """Add an agent registry for discovery"""
        self.agent_registries.add(registry_url.rstrip('/'))
        logger.info(f"Added agent registry: {registry_url}")
    
    async def discover_from_registries(self) -> List[DiscoveredAgent]:
        """Discover agents from known registries"""
        if not self.agent_registries:
            return []
        
        return await self.discover_agents_in_network(list(self.agent_registries))
    
    def clear_cache(self):
        """Clear discovery cache"""
        self.discovery_cache.clear()
        logger.info("Discovery cache cleared")
    
    def get_discovery_stats(self) -> Dict[str, Any]:
        """Get discovery statistics"""
        total_agents = len(self.discovered_agents)
        available_agents = sum(1 for agent in self.discovered_agents.values() if agent.available)
        
        avg_response_time = 0.0
        if available_agents > 0:
            total_response_time = sum(
                agent.response_time_ms for agent in self.discovered_agents.values() 
                if agent.available
            )
            avg_response_time = total_response_time / available_agents
        
        return {
            "total_discovered": total_agents,
            "available_agents": available_agents,
            "unavailable_agents": total_agents - available_agents,
            "average_response_time_ms": avg_response_time,
            "registries_count": len(self.agent_registries),
            "cache_size": len(self.discovery_cache)
        }


class A2ADiscoveryClient:
    """High-level client for A2A agent discovery"""

    def __init__(self, config: Optional[DiscoveryConfig] = None):
        self.config = config or DiscoveryConfig()
        self._discovery: Optional[A2AAgentDiscovery] = None

    async def __aenter__(self):
        """Async context manager entry"""
        self._discovery = A2AAgentDiscovery(self.config)
        await self._discovery.__aenter__()
        return self

    async def __aexit__(self, exc_type, exc_val, exc_tb):
        """Async context manager exit"""
        if self._discovery:
            await self._discovery.__aexit__(exc_type, exc_val, exc_tb)

    async def discover_agents(self, urls: List[str]) -> List[DiscoveredAgent]:
        """Discover agents at specified URLs"""
        if not self._discovery:
            raise RuntimeError("Discovery client not initialized. Use async context manager.")

        return await self._discovery.discover_agents_in_network(urls)

    async def find_best_agent_for_task(self, task_description: str,
                                     required_capabilities: Optional[List[str]] = None,
                                     required_skills: Optional[List[str]] = None) -> Optional[CapabilityMatch]:
        """Find the best agent for a specific task"""
        if not self._discovery:
            raise RuntimeError("Discovery client not initialized. Use async context manager.")

        # Extract capabilities and skills from task description if not provided
        if not required_capabilities:
            required_capabilities = self._extract_capabilities_from_description(task_description)

        if not required_skills:
            required_skills = self._extract_skills_from_description(task_description)

        matches = await self._discovery.find_agents_by_capability(required_capabilities, required_skills)

        return matches[0] if matches else None

    def _extract_capabilities_from_description(self, description: str) -> List[str]:
        """Extract required capabilities from task description"""
        capabilities = []
        description_lower = description.lower()

        # Simple keyword matching
        if any(word in description_lower for word in ["stream", "streaming", "real-time"]):
            capabilities.append("streaming")

        if any(word in description_lower for word in ["notify", "notification", "alert"]):
            capabilities.append("push_notifications")

        if any(word in description_lower for word in ["history", "track", "log"]):
            capabilities.append("state_history")

        return capabilities

    def _extract_skills_from_description(self, description: str) -> List[str]:
        """Extract required skills from task description"""
        skills = []
        description_lower = description.lower()

        # Simple keyword matching for common skills
        skill_keywords = {
            "research": ["research", "search", "find", "investigate"],
            "analysis": ["analyze", "analysis", "examine", "study"],
            "generation": ["generate", "create", "write", "produce"],
            "data_processing": ["process", "transform", "convert"],
            "visualization": ["visualize", "chart", "graph", "plot"],
            "translation": ["translate", "translation", "language"],
            "summarization": ["summarize", "summary", "brief"]
        }

        for skill, keywords in skill_keywords.items():
            if any(keyword in description_lower for keyword in keywords):
                skills.append(skill)

        return skills

    async def get_agent_capabilities(self, agent_url: str) -> Optional[Dict[str, Any]]:
        """Get detailed capabilities of a specific agent"""
        if not self._discovery:
            raise RuntimeError("Discovery client not initialized. Use async context manager.")

        agents = await self._discovery.discover_agent_at_url(agent_url)
        if agents:
            agent = agents[0]
            return {
                "capabilities": agent.capabilities,
                "skills": agent.skills,
                "security_schemes": agent.security_schemes,
                "response_time_ms": agent.response_time_ms
            }

        return None

    async def test_agent_connectivity(self, agent_url: str) -> Dict[str, Any]:
        """Test connectivity to an agent"""
        if not self._discovery:
            raise RuntimeError("Discovery client not initialized. Use async context manager.")

        start_time = asyncio.get_event_loop().time()

        try:
            agents = await self._discovery.discover_agent_at_url(agent_url)
            end_time = asyncio.get_event_loop().time()

            if agents:
                return {
                    "success": True,
                    "response_time_ms": (end_time - start_time) * 1000,
                    "agent_count": len(agents),
                    "agents": [{"id": agent.agent_id, "name": agent.name} for agent in agents]
                }
            else:
                return {
                    "success": False,
                    "error": "No agents discovered",
                    "response_time_ms": (end_time - start_time) * 1000
                }

        except Exception as e:
            end_time = asyncio.get_event_loop().time()
            return {
                "success": False,
                "error": str(e),
                "response_time_ms": (end_time - start_time) * 1000
            }


# Global discovery instance
agent_discovery = A2AAgentDiscovery()
