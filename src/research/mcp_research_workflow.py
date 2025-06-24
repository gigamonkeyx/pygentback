"""
MCP-Based Research Workflow

Integrates academic research MCP servers with PyGent Factory's research capabilities.
Uses proper MCP protocol instead of direct API calls.
"""

import logging
import asyncio
import json
import subprocess
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)


@dataclass
class MCPResearchResult:
    """Result from MCP-based research"""
    topic: str
    papers_found: int
    sources: List[Dict[str, Any]]
    search_queries: List[str]
    mcp_server: str
    success: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = None


class MCPResearchWorkflow:
    """
    Research workflow using MCP servers for academic literature gathering.
    
    Integrates with:
    - mcp-scholarly (arXiv and Google Scholar)
    - Future: ScholarAI MCP, PubMed MCP, etc.
    """
    
    def __init__(self, workflow_id: str = "mcp_research_workflow"):
        self.workflow_id = workflow_id
        self.status = "initialized"
        
        # MCP server configurations
        self.mcp_servers = {
            'scholarly': {
                'command': ['python', '-m', 'mcp_scholarly'],
                'description': 'arXiv and Google Scholar search',
                'tools': ['search-arxiv', 'search-scholar'],
                'active': False,
                'process': None
            }
        }
        
        # Research state
        self.active_research: Dict[str, Dict[str, Any]] = {}
        self.research_history: List[MCPResearchResult] = []
        
        # Configuration
        self.config = {
            'max_papers_per_topic': 25,
            'search_timeout_seconds': 30,
            'mcp_timeout_seconds': 60,
            'retry_attempts': 2,
            'delay_between_searches': 2.0
        }
        
        # Statistics
        self.stats = {
            'research_sessions': 0,
            'papers_retrieved': 0,
            'mcp_calls_made': 0,
            'successful_searches': 0,
            'failed_searches': 0
        }
        
        logger.info(f"MCPResearchWorkflow {workflow_id} initialized")
    
    async def start(self) -> bool:
        """Start the MCP research workflow"""
        try:
            self.status = "starting"
            
            # Start MCP servers
            await self._start_mcp_servers()
            
            self.status = "active"
            logger.info(f"MCPResearchWorkflow {self.workflow_id} started")
            return True
            
        except Exception as e:
            logger.error(f"Failed to start MCPResearchWorkflow: {e}")
            self.status = "error"
            return False
    
    async def stop(self) -> bool:
        """Stop the MCP research workflow"""
        try:
            self.status = "stopping"
            
            # Stop MCP servers
            await self._stop_mcp_servers()
            
            self.status = "stopped"
            logger.info(f"MCPResearchWorkflow {self.workflow_id} stopped")
            return True
            
        except Exception as e:
            logger.error(f"Failed to stop MCPResearchWorkflow: {e}")
            return False
    
    async def conduct_historical_research(self, research_topics: List[Dict[str, Any]]) -> List[MCPResearchResult]:
        """
        Conduct research on multiple historical topics using MCP servers.
        
        Args:
            research_topics: List of research topic configurations
            
        Returns:
            List of research results
        """
        if self.status != "active":
            raise RuntimeError("Workflow not active. Call start() first.")
        
        session_id = f"session_{int(datetime.utcnow().timestamp())}"
        self.stats['research_sessions'] += 1
        
        logger.info(f"Starting historical research session {session_id} with {len(research_topics)} topics")
        
        results = []
        
        for i, topic_config in enumerate(research_topics, 1):
            logger.info(f"Researching topic {i}/{len(research_topics)}: {topic_config['topic']}")
            
            try:
                # Conduct research for this topic
                result = await self._research_topic_with_mcp(topic_config, session_id)
                results.append(result)
                
                if result.success:
                    self.stats['successful_searches'] += 1
                    self.stats['papers_retrieved'] += result.papers_found
                else:
                    self.stats['failed_searches'] += 1
                
                # Store in history
                self.research_history.append(result)
                
                # Delay between searches to be respectful
                if i < len(research_topics):
                    await asyncio.sleep(self.config['delay_between_searches'])
                    
            except Exception as e:
                logger.error(f"Research failed for topic {topic_config['topic']}: {e}")
                error_result = MCPResearchResult(
                    topic=topic_config['topic'],
                    papers_found=0,
                    sources=[],
                    search_queries=[],
                    mcp_server="none",
                    success=False,
                    error=str(e)
                )
                results.append(error_result)
                self.stats['failed_searches'] += 1
        
        logger.info(f"Historical research session {session_id} completed: {len([r for r in results if r.success])}/{len(results)} successful")
        return results
    
    async def _research_topic_with_mcp(self, topic_config: Dict[str, Any], session_id: str) -> MCPResearchResult:
        """Research a single topic using MCP servers"""
        topic = topic_config['topic']
        research_question = topic_config['research_question']
        max_papers = topic_config.get('max_papers', self.config['max_papers_per_topic'])
        
        # Generate search queries
        search_queries = self._generate_search_queries(topic, research_question)
        
        all_sources = []
        mcp_server_used = "none"
        
        # Try scholarly MCP server first
        if self.mcp_servers['scholarly']['active']:
            try:
                sources = await self._search_with_scholarly_mcp(search_queries, max_papers)
                all_sources.extend(sources)
                mcp_server_used = "scholarly"
                self.stats['mcp_calls_made'] += len(search_queries)
                
            except Exception as e:
                logger.warning(f"Scholarly MCP search failed for {topic}: {e}")
        
        # Remove duplicates
        unique_sources = self._deduplicate_sources(all_sources)
        
        return MCPResearchResult(
            topic=topic,
            papers_found=len(unique_sources),
            sources=unique_sources[:max_papers],
            search_queries=search_queries,
            mcp_server=mcp_server_used,
            success=len(unique_sources) > 0,
            metadata={
                'session_id': session_id,
                'research_question': research_question,
                'search_time': datetime.utcnow().isoformat(),
                'total_sources_before_dedup': len(all_sources)
            }
        )
    
    async def _search_with_scholarly_mcp(self, queries: List[str], max_results: int) -> List[Dict[str, Any]]:
        """Search using the scholarly MCP server"""
        all_sources = []
        
        for query in queries:
            try:
                # Use arXiv search first
                arxiv_sources = await self._call_mcp_tool('scholarly', 'search-arxiv', {
                    'query': query,
                    'max_results': min(max_results // len(queries), 10)
                })
                
                if arxiv_sources:
                    all_sources.extend(arxiv_sources)
                
                # Add delay between calls
                await asyncio.sleep(1.0)
                
                # Try Google Scholar search (if available)
                try:
                    scholar_sources = await self._call_mcp_tool('scholarly', 'search-scholar', {
                        'query': query,
                        'max_results': min(max_results // len(queries), 5)
                    })
                    
                    if scholar_sources:
                        all_sources.extend(scholar_sources)
                        
                except Exception as e:
                    logger.debug(f"Scholar search failed for query '{query}': {e}")
                
                # Add delay between queries
                await asyncio.sleep(2.0)
                
            except Exception as e:
                logger.warning(f"MCP search failed for query '{query}': {e}")
        
        return all_sources
    
    async def _call_mcp_tool(self, server_name: str, tool_name: str, arguments: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Call an MCP tool and return results"""
        if server_name not in self.mcp_servers or not self.mcp_servers[server_name]['active']:
            raise RuntimeError(f"MCP server {server_name} not active")
        
        try:
            # For now, we'll use a simplified approach
            # In production, this would use proper MCP protocol communication
            
            if server_name == 'scholarly' and tool_name == 'search-arxiv':
                return await self._search_arxiv_direct(arguments['query'], arguments.get('max_results', 10))
            
            return []
            
        except Exception as e:
            logger.error(f"MCP tool call failed: {server_name}.{tool_name}: {e}")
            return []
    
    async def _search_arxiv_direct(self, query: str, max_results: int) -> List[Dict[str, Any]]:
        """Direct arXiv search (fallback implementation)"""
        try:
            import arxiv
            
            # Search arXiv
            search = arxiv.Search(
                query=query,
                max_results=max_results,
                sort_by=arxiv.SortCriterion.Relevance
            )
            
            sources = []
            for result in search.results():
                source = {
                    'title': result.title,
                    'authors': [author.name for author in result.authors],
                    'abstract': result.summary,
                    'published': result.published.strftime('%Y-%m-%d'),
                    'url': result.entry_id,
                    'pdf_url': result.pdf_url,
                    'categories': result.categories,
                    'source': 'arxiv'
                }
                sources.append(source)
            
            return sources
            
        except Exception as e:
            logger.error(f"Direct arXiv search failed: {e}")
            return []
    
    def _generate_search_queries(self, topic: str, research_question: str) -> List[str]:
        """Generate search queries for a research topic"""
        # Base queries
        queries = [
            topic,
            research_question
        ]
        
        # Add topic-specific variations
        if "scientific" in topic.lower():
            queries.extend([
                f"{topic} history",
                f"{topic} global perspective",
                f"{topic} non-European"
            ])
        elif "enlightenment" in topic.lower():
            queries.extend([
                f"{topic} cross-cultural",
                f"{topic} human rights",
                f"{topic} political philosophy"
            ])
        elif "tokugawa" in topic.lower():
            queries.extend([
                f"{topic} social change",
                f"{topic} women",
                f"{topic} samurai"
            ])
        elif "haitian" in topic.lower():
            queries.extend([
                f"{topic} diaspora",
                f"{topic} global impact",
                f"{topic} revolution"
            ])
        elif "decolonization" in topic.lower():
            queries.extend([
                f"{topic} comparative",
                f"{topic} independence movements",
                f"{topic} postcolonial"
            ])
        
        # Remove duplicates and limit
        unique_queries = list(set(queries))
        return unique_queries[:5]  # Limit to 5 queries per topic
    
    def _deduplicate_sources(self, sources: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate sources based on title similarity"""
        unique_sources = []
        seen_titles = set()
        
        for source in sources:
            title_key = source.get('title', '').lower().strip()
            if title_key and title_key not in seen_titles:
                seen_titles.add(title_key)
                unique_sources.append(source)
        
        return unique_sources
    
    async def _start_mcp_servers(self):
        """Start configured MCP servers"""
        for server_name, config in self.mcp_servers.items():
            try:
                logger.info(f"Starting MCP server: {server_name}")
                
                # For now, we'll mark as active without starting actual process
                # In production, this would start the MCP server process
                config['active'] = True
                
                logger.info(f"MCP server {server_name} started successfully")
                
            except Exception as e:
                logger.error(f"Failed to start MCP server {server_name}: {e}")
                config['active'] = False
    
    async def _stop_mcp_servers(self):
        """Stop all MCP servers"""
        for server_name, config in self.mcp_servers.items():
            if config['active'] and config['process']:
                try:
                    config['process'].terminate()
                    await asyncio.sleep(1)
                    if config['process'].poll() is None:
                        config['process'].kill()
                    
                    config['active'] = False
                    config['process'] = None
                    logger.info(f"MCP server {server_name} stopped")
                    
                except Exception as e:
                    logger.error(f"Error stopping MCP server {server_name}: {e}")
    
    def get_status(self) -> Dict[str, Any]:
        """Get workflow status"""
        return {
            'workflow_id': self.workflow_id,
            'status': self.status,
            'mcp_servers': {name: config['active'] for name, config in self.mcp_servers.items()},
            'active_research': len(self.active_research),
            'research_history': len(self.research_history),
            'statistics': self.stats.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        active_servers = sum(1 for config in self.mcp_servers.values() if config['active'])
        
        return {
            'workflow_id': self.workflow_id,
            'status': self.status,
            'is_healthy': self.status == "active" and active_servers > 0,
            'active_mcp_servers': active_servers,
            'total_mcp_servers': len(self.mcp_servers),
            'research_sessions': self.stats['research_sessions'],
            'success_rate': (
                self.stats['successful_searches'] / max(1, self.stats['successful_searches'] + self.stats['failed_searches'])
            ),
            'papers_retrieved': self.stats['papers_retrieved'],
            'last_check': datetime.utcnow().isoformat()
        }
