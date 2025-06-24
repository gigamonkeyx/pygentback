"""
Agent Memory Management

This module provides memory management capabilities for agents in PyGent Factory.
It supports different types of memory (short-term, long-term, episodic) with
vector-based storage and retrieval for efficient context management.
"""

import asyncio
import logging
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
import uuid

from ..storage.vector_store import VectorStoreManager, VectorDocument, SimilarityResult
from ..config.settings import Settings


logger = logging.getLogger(__name__)


class MemoryType(Enum):
    """Types of agent memory"""
    SHORT_TERM = "short_term"      # Recent interactions and context
    LONG_TERM = "long_term"        # Persistent knowledge and experiences
    EPISODIC = "episodic"          # Specific events and experiences
    SEMANTIC = "semantic"          # Factual knowledge and concepts
    PROCEDURAL = "procedural"      # Skills and procedures


class MemoryImportance(Enum):
    """Memory importance levels for retention"""
    LOW = 1
    MEDIUM = 2
    HIGH = 3
    CRITICAL = 4


@dataclass
class MemoryEntry:
    """Represents a memory entry"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    agent_id: str = ""
    memory_type: MemoryType = MemoryType.SHORT_TERM
    content: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    importance: MemoryImportance = MemoryImportance.MEDIUM
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.utcnow)
    last_accessed: datetime = field(default_factory=datetime.utcnow)
    access_count: int = 0
    retention_score: float = 1.0
    
    def update_access(self) -> None:
        """Update access statistics"""
        self.last_accessed = datetime.utcnow()
        self.access_count += 1


@dataclass
class MemorySearchResult:
    """Memory search result with relevance scoring"""
    memory: MemoryEntry
    relevance_score: float
    similarity_score: float


class MemorySpace:
    """
    Memory space for an individual agent.
    
    Manages different types of memory with automatic retention,
    consolidation, and retrieval capabilities.
    """
    
    def __init__(self, 
                 agent_id: str,
                 vector_store_manager: VectorStoreManager,
                 config: Dict[str, Any]):
        self.agent_id = agent_id
        self.vector_store_manager = vector_store_manager
        self.config = config
        self.memory_stores: Dict[MemoryType, Any] = {}
        self.memory_index: Dict[str, MemoryEntry] = {}
        self.max_entries = config.get("max_entries", 1000)
        self.retention_threshold = config.get("retention_threshold", 0.1)
        self._lock = asyncio.Lock()
    
    async def initialize(self) -> None:
        """Initialize memory space and vector stores"""
        try:
            # Initialize vector stores for each memory type
            for memory_type in MemoryType:
                collection_name = f"agent_{self.agent_id}_{memory_type.value}"
                store = await self.vector_store_manager.get_store(collection_name)
                self.memory_stores[memory_type] = store
            
            logger.info(f"Memory space initialized for agent: {self.agent_id}")
            
        except Exception as e:
            logger.error(f"Failed to initialize memory space for {self.agent_id}: {str(e)}")
            raise
    
    async def store_memory(self, 
                          content: str,
                          memory_type: MemoryType = MemoryType.SHORT_TERM,
                          metadata: Optional[Dict[str, Any]] = None,
                          importance: MemoryImportance = MemoryImportance.MEDIUM,
                          embedding: Optional[List[float]] = None) -> str:
        """
        Store a new memory entry.
        
        Args:
            content: Memory content
            memory_type: Type of memory
            metadata: Additional metadata
            importance: Memory importance level
            embedding: Pre-computed embedding (optional)
            
        Returns:
            str: Memory entry ID
        """
        async with self._lock:
            try:
                # Create memory entry
                memory = MemoryEntry(
                    agent_id=self.agent_id,
                    memory_type=memory_type,
                    content=content,
                    metadata=metadata or {},
                    importance=importance,
                    embedding=embedding
                )
                
                # Store in vector store
                vector_doc = VectorDocument(
                    id=memory.id,
                    content=content,
                    embedding=embedding or [],  # Will be computed by embedding service
                    metadata={
                        "agent_id": self.agent_id,
                        "memory_type": memory_type.value,
                        "importance": importance.value,
                        "created_at": memory.created_at.isoformat(),
                        **memory.metadata
                    },
                    created_at=memory.created_at
                )
                
                store = self.memory_stores[memory_type]
                await store.add_documents([vector_doc])
                
                # Update local index
                self.memory_index[memory.id] = memory
                
                # Check if we need to consolidate memory
                await self._check_memory_limits()
                
                logger.debug(f"Stored memory {memory.id} for agent {self.agent_id}")
                return memory.id
                
            except Exception as e:
                logger.error(f"Failed to store memory for {self.agent_id}: {str(e)}")
                raise
    
    async def retrieve_memories(self,
                               query: str,
                               memory_types: Optional[List[MemoryType]] = None,
                               limit: int = 10,
                               similarity_threshold: float = 0.7,
                               query_embedding: Optional[List[float]] = None) -> List[MemorySearchResult]:
        """
        Retrieve relevant memories based on query.
        
        Args:
            query: Search query
            memory_types: Types of memory to search (default: all)
            limit: Maximum number of results
            similarity_threshold: Minimum similarity score
            query_embedding: Pre-computed query embedding
            
        Returns:
            List[MemorySearchResult]: Relevant memories
        """
        try:
            if memory_types is None:
                memory_types = list(MemoryType)
            
            all_results = []
            
            # Search each memory type
            for memory_type in memory_types:
                if memory_type not in self.memory_stores:
                    continue
                
                store = self.memory_stores[memory_type]
                
                # Perform similarity search
                if query_embedding:
                    results = await store.similarity_search(
                        query_embedding=query_embedding,
                        limit=limit,
                        similarity_threshold=similarity_threshold,
                        metadata_filter={"agent_id": self.agent_id}
                    )
                else:
                    # If no embedding provided, search by content (less efficient)
                    results = await store.similarity_search(
                        query_embedding=[0.0] * 1536,  # Placeholder
                        limit=limit,
                        similarity_threshold=0.0,
                        metadata_filter={"agent_id": self.agent_id}
                    )
                
                # Convert to memory search results
                for result in results:
                    memory_id = result.document.id
                    if memory_id in self.memory_index:
                        memory = self.memory_index[memory_id]
                        
                        # Update access statistics
                        memory.update_access()
                        
                        # Calculate relevance score (combines similarity and importance)
                        relevance_score = (
                            result.similarity_score * 0.7 + 
                            (memory.importance.value / 4.0) * 0.3
                        )
                        
                        all_results.append(MemorySearchResult(
                            memory=memory,
                            relevance_score=relevance_score,
                            similarity_score=result.similarity_score
                        ))
            
            # Sort by relevance and return top results
            all_results.sort(key=lambda x: x.relevance_score, reverse=True)
            return all_results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to retrieve memories for {self.agent_id}: {str(e)}")
            return []
    
    async def get_memory(self, memory_id: str) -> Optional[MemoryEntry]:
        """Get a specific memory by ID"""
        memory = self.memory_index.get(memory_id)
        if memory:
            memory.update_access()
        return memory
    
    async def update_memory(self, memory_id: str, 
                           content: Optional[str] = None,
                           metadata: Optional[Dict[str, Any]] = None) -> bool:
        """Update an existing memory entry"""
        async with self._lock:
            try:
                if memory_id not in self.memory_index:
                    return False
                
                memory = self.memory_index[memory_id]
                
                if content:
                    memory.content = content
                
                if metadata:
                    memory.metadata.update(metadata)
                
                # Update in vector store
                store = self.memory_stores[memory.memory_type]
                vector_doc = VectorDocument(
                    id=memory.id,
                    content=memory.content,
                    embedding=memory.embedding or [],
                    metadata={
                        "agent_id": self.agent_id,
                        "memory_type": memory.memory_type.value,
                        "importance": memory.importance.value,
                        "created_at": memory.created_at.isoformat(),
                        **memory.metadata
                    },
                    created_at=memory.created_at
                )
                
                await store.update_document(vector_doc)
                return True
                
            except Exception as e:
                logger.error(f"Failed to update memory {memory_id}: {str(e)}")
                return False
    
    async def delete_memory(self, memory_id: str) -> bool:
        """Delete a memory entry"""
        async with self._lock:
            try:
                if memory_id not in self.memory_index:
                    return False
                
                memory = self.memory_index[memory_id]
                
                # Delete from vector store
                store = self.memory_stores[memory.memory_type]
                await store.delete_document(memory_id)
                
                # Remove from local index
                del self.memory_index[memory_id]
                
                return True
                
            except Exception as e:
                logger.error(f"Failed to delete memory {memory_id}: {str(e)}")
                return False
    
    async def consolidate_memories(self) -> None:
        """Consolidate and optimize memory storage"""
        async with self._lock:
            try:
                # Implement memory consolidation logic
                # - Move important short-term memories to long-term
                # - Remove low-importance, rarely accessed memories
                # - Merge similar memories
                
                current_time = datetime.utcnow()
                memories_to_remove = []
                memories_to_promote = []
                
                for memory_id, memory in self.memory_index.items():
                    # Calculate retention score based on importance, recency, and access
                    age_days = (current_time - memory.created_at).days
                    recency_score = max(0, 1 - (age_days / 30))  # Decay over 30 days
                    access_score = min(1, memory.access_count / 10)  # Normalize access count
                    importance_score = memory.importance.value / 4.0
                    
                    retention_score = (
                        importance_score * 0.5 +
                        recency_score * 0.3 +
                        access_score * 0.2
                    )
                    
                    memory.retention_score = retention_score
                    
                    # Mark for removal if below threshold
                    if retention_score < self.retention_threshold:
                        memories_to_remove.append(memory_id)
                    
                    # Promote important short-term memories to long-term
                    elif (memory.memory_type == MemoryType.SHORT_TERM and
                          importance_score >= 0.75 and age_days >= 1):
                        memories_to_promote.append(memory)
                
                # Remove low-retention memories
                for memory_id in memories_to_remove:
                    await self.delete_memory(memory_id)
                
                # Promote memories to long-term
                for memory in memories_to_promote:
                    # Create long-term copy
                    await self.store_memory(
                        content=memory.content,
                        memory_type=MemoryType.LONG_TERM,
                        metadata=memory.metadata,
                        importance=memory.importance,
                        embedding=memory.embedding
                    )
                    
                    # Remove from short-term
                    await self.delete_memory(memory.id)
                
                logger.info(f"Memory consolidation complete for {self.agent_id}: "
                           f"removed {len(memories_to_remove)}, promoted {len(memories_to_promote)}")
                
            except Exception as e:
                logger.error(f"Failed to consolidate memories for {self.agent_id}: {str(e)}")
    
    async def _check_memory_limits(self) -> None:
        """Check and enforce memory limits"""
        if len(self.memory_index) > self.max_entries:
            await self.consolidate_memories()
    
    async def get_memory_stats(self) -> Dict[str, Any]:
        """Get memory space statistics"""
        stats = {
            "agent_id": self.agent_id,
            "total_memories": len(self.memory_index),
            "memory_types": {},
            "importance_distribution": {imp.name: 0 for imp in MemoryImportance},
            "average_retention_score": 0.0
        }
        
        total_retention = 0.0
        
        for memory in self.memory_index.values():
            # Count by type
            type_name = memory.memory_type.value
            if type_name not in stats["memory_types"]:
                stats["memory_types"][type_name] = 0
            stats["memory_types"][type_name] += 1
            
            # Count by importance
            stats["importance_distribution"][memory.importance.name] += 1
            
            # Sum retention scores
            total_retention += memory.retention_score
        
        if len(self.memory_index) > 0:
            stats["average_retention_score"] = total_retention / len(self.memory_index)
        
        return stats


class MemoryManager:
    """
    Global memory manager for all agents.
    
    Coordinates memory operations across agents and provides
    system-wide memory management capabilities.
    """
    
    def __init__(self, vector_store_manager: VectorStoreManager, settings: Settings):
        self.vector_store_manager = vector_store_manager
        self.settings = settings
        self.memory_spaces: Dict[str, MemorySpace] = {}
        self._consolidation_task: Optional[asyncio.Task] = None
        self._running = False
    
    async def start(self) -> None:
        """Start the memory manager"""
        self._running = True
        
        # Start periodic consolidation task
        self._consolidation_task = asyncio.create_task(self._periodic_consolidation())
        
        logger.info("Memory manager started")
    
    async def stop(self) -> None:
        """Stop the memory manager"""
        self._running = False
        
        if self._consolidation_task:
            self._consolidation_task.cancel()
            try:
                await self._consolidation_task
            except asyncio.CancelledError:
                pass
        
        logger.info("Memory manager stopped")
    
    async def create_memory_space(self, agent_id: str, config: Dict[str, Any]) -> MemorySpace:
        """Create a memory space for an agent"""
        if agent_id in self.memory_spaces:
            return self.memory_spaces[agent_id]
        
        memory_space = MemorySpace(agent_id, self.vector_store_manager, config)
        await memory_space.initialize()
        
        self.memory_spaces[agent_id] = memory_space
        logger.info(f"Created memory space for agent: {agent_id}")
        
        return memory_space
    
    async def get_memory_space(self, agent_id: str) -> Optional[MemorySpace]:
        """Get memory space for an agent"""
        return self.memory_spaces.get(agent_id)
    
    async def cleanup_memory_space(self, agent_id: str) -> None:
        """Clean up memory space for an agent"""
        if agent_id in self.memory_spaces:
            # Could implement cleanup logic here
            del self.memory_spaces[agent_id]
            logger.info(f"Cleaned up memory space for agent: {agent_id}")
    
    async def get_memory_spaces_count(self) -> int:
        """Get count of active memory spaces"""
        return len(self.memory_spaces)
    
    async def _periodic_consolidation(self) -> None:
        """Periodic memory consolidation task"""
        while self._running:
            try:
                # Consolidate memories for all agents
                for memory_space in self.memory_spaces.values():
                    await memory_space.consolidate_memories()
                
                # Wait for next consolidation cycle (1 hour)
                await asyncio.sleep(3600)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in periodic consolidation: {str(e)}")
                await asyncio.sleep(300)  # Wait 5 minutes before retry
    
    async def get_global_stats(self) -> Dict[str, Any]:
        """Get global memory statistics"""
        total_memories = 0
        total_agents = len(self.memory_spaces)
        
        type_distribution = {memory_type.value: 0 for memory_type in MemoryType}
        importance_distribution = {imp.name: 0 for imp in MemoryImportance}
        
        for memory_space in self.memory_spaces.values():
            stats = await memory_space.get_memory_stats()
            total_memories += stats["total_memories"]
            
            for memory_type, count in stats["memory_types"].items():
                type_distribution[memory_type] += count
            
            for importance, count in stats["importance_distribution"].items():
                importance_distribution[importance] += count
        
        return {
            "total_agents": total_agents,
            "total_memories": total_memories,
            "average_memories_per_agent": total_memories / max(1, total_agents),
            "memory_type_distribution": type_distribution,
            "importance_distribution": importance_distribution
        }
