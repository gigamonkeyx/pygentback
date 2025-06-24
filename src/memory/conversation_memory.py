"""
Conversation Memory Implementation

Provides conversation history storage and retrieval for PyGent Factory agents,
supporting context management and conversation threading.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


class MessageRole(Enum):
    """Message roles in conversation"""
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"
    TOOL = "tool"


@dataclass
class ConversationMessage:
    """Single message in conversation"""
    id: str
    role: MessageRole
    content: str
    timestamp: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tool_calls: Optional[List[Dict[str, Any]]] = None
    tool_results: Optional[List[Dict[str, Any]]] = None


@dataclass
class ConversationThread:
    """Conversation thread containing multiple messages"""
    thread_id: str
    agent_id: str
    title: Optional[str] = None
    messages: List[ConversationMessage] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True


class ConversationMemory:
    """
    Conversation memory manager for storing and retrieving conversation history.
    
    Provides thread-based conversation management with support for
    message storage, retrieval, and context management.
    """
    
    def __init__(self, max_threads: int = 1000, max_messages_per_thread: int = 1000):
        self.max_threads = max_threads
        self.max_messages_per_thread = max_messages_per_thread
        self.threads: Dict[str, ConversationThread] = {}
        self.agent_threads: Dict[str, List[str]] = {}  # agent_id -> thread_ids
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize conversation memory"""
        try:
            self.is_initialized = True
            logger.info("Conversation memory initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize conversation memory: {e}")
            raise
    
    async def create_thread(self, agent_id: str, title: Optional[str] = None) -> ConversationThread:
        """Create new conversation thread"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Generate thread ID
            thread_id = f"thread_{agent_id}_{int(datetime.utcnow().timestamp())}"
            
            # Create thread
            thread = ConversationThread(
                thread_id=thread_id,
                agent_id=agent_id,
                title=title or f"Conversation {len(self.threads) + 1}"
            )
            
            # Store thread
            self.threads[thread_id] = thread
            
            # Update agent threads mapping
            if agent_id not in self.agent_threads:
                self.agent_threads[agent_id] = []
            self.agent_threads[agent_id].append(thread_id)
            
            # Cleanup old threads if needed
            await self._cleanup_old_threads()
            
            logger.debug(f"Created conversation thread {thread_id} for agent {agent_id}")
            return thread
            
        except Exception as e:
            logger.error(f"Failed to create conversation thread: {e}")
            raise
    
    async def add_message(self, thread_id: str, message: ConversationMessage) -> bool:
        """Add message to conversation thread"""
        try:
            if thread_id not in self.threads:
                logger.error(f"Thread {thread_id} not found")
                return False
            
            thread = self.threads[thread_id]
            
            # Add message to thread
            thread.messages.append(message)
            thread.updated_at = datetime.utcnow()
            
            # Cleanup old messages if needed
            if len(thread.messages) > self.max_messages_per_thread:
                # Keep only the most recent messages
                thread.messages = thread.messages[-self.max_messages_per_thread:]
            
            logger.debug(f"Added message to thread {thread_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add message to thread {thread_id}: {e}")
            return False
    
    async def get_thread(self, thread_id: str) -> Optional[ConversationThread]:
        """Get conversation thread by ID"""
        return self.threads.get(thread_id)
    
    async def get_agent_threads(self, agent_id: str) -> List[ConversationThread]:
        """Get all threads for an agent"""
        thread_ids = self.agent_threads.get(agent_id, [])
        threads = []
        for thread_id in thread_ids:
            if thread_id in self.threads:
                threads.append(self.threads[thread_id])
        return threads
    
    async def get_recent_messages(self, thread_id: str, limit: int = 10) -> List[ConversationMessage]:
        """Get recent messages from thread"""
        thread = await self.get_thread(thread_id)
        if not thread:
            return []
        
        # Return most recent messages
        return thread.messages[-limit:] if thread.messages else []
    
    async def get_conversation_context(self, thread_id: str, max_tokens: int = 4000) -> str:
        """Get conversation context as formatted string"""
        thread = await self.get_thread(thread_id)
        if not thread:
            return ""
        
        context_parts = []
        total_length = 0
        
        # Build context from most recent messages
        for message in reversed(thread.messages):
            message_text = f"{message.role.value}: {message.content}\n"
            
            if total_length + len(message_text) > max_tokens:
                break
            
            context_parts.insert(0, message_text)
            total_length += len(message_text)
        
        return "\n".join(context_parts)
    
    async def search_messages(self, query: str, agent_id: Optional[str] = None) -> List[ConversationMessage]:
        """Search messages by content"""
        results = []
        
        threads_to_search = []
        if agent_id:
            threads_to_search = await self.get_agent_threads(agent_id)
        else:
            threads_to_search = list(self.threads.values())
        
        for thread in threads_to_search:
            for message in thread.messages:
                if query.lower() in message.content.lower():
                    results.append(message)
        
        return results
    
    async def delete_thread(self, thread_id: str) -> bool:
        """Delete conversation thread"""
        try:
            if thread_id not in self.threads:
                return False
            
            thread = self.threads[thread_id]
            agent_id = thread.agent_id
            
            # Remove from threads
            del self.threads[thread_id]
            
            # Remove from agent threads mapping
            if agent_id in self.agent_threads:
                self.agent_threads[agent_id] = [
                    tid for tid in self.agent_threads[agent_id] if tid != thread_id
                ]
            
            logger.debug(f"Deleted conversation thread {thread_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete thread {thread_id}: {e}")
            return False
    
    async def clear_agent_memory(self, agent_id: str):
        """Clear all conversation memory for an agent"""
        thread_ids = self.agent_threads.get(agent_id, []).copy()
        for thread_id in thread_ids:
            await self.delete_thread(thread_id)
        
        logger.info(f"Cleared conversation memory for agent {agent_id}")
    
    async def _cleanup_old_threads(self):
        """Cleanup old threads if limit exceeded"""
        if len(self.threads) <= self.max_threads:
            return
        
        # Sort threads by last update time
        sorted_threads = sorted(
            self.threads.items(),
            key=lambda x: x[1].updated_at
        )
        
        # Remove oldest threads
        threads_to_remove = len(self.threads) - self.max_threads
        for i in range(threads_to_remove):
            thread_id = sorted_threads[i][0]
            await self.delete_thread(thread_id)
    
    def get_stats(self) -> Dict[str, Any]:
        """Get conversation memory statistics"""
        total_messages = sum(len(thread.messages) for thread in self.threads.values())
        
        return {
            "total_threads": len(self.threads),
            "total_messages": total_messages,
            "total_agents": len(self.agent_threads),
            "is_initialized": self.is_initialized
        }
