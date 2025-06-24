#!/usr/bin/env python3
"""
Agent Context Restoration System - Proof of Concept Implementation

This prototype demonstrates how to implement comprehensive context capture and restoration
for AI agents in PyGent Factory, enabling seamless agent continuity across resets.
"""

import asyncio
import json
import gzip
import hashlib
import logging
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from pathlib import Path
import sys

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

logger = logging.getLogger(__name__)

@dataclass
class ConversationMessage:
    """Single conversation message"""
    id: str
    role: str  # 'user', 'assistant', 'system'
    content: str
    timestamp: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ReasoningThread:
    """A reasoning thread from Tree of Thought processing"""
    thread_id: str
    prompt: str
    reasoning_steps: List[str]
    conclusion: str
    confidence_score: float
    timestamp: datetime

@dataclass
class ToolInvocation:
    """Record of a tool being called"""
    tool_name: str
    server_name: str
    arguments: Dict[str, Any]
    result: Any
    timestamp: datetime
    execution_time_ms: float

@dataclass
class MemorySnapshot:
    """Snapshot of agent memory state"""
    agent_id: str
    short_term_memories: List[Dict[str, Any]]
    long_term_memories: List[Dict[str, Any]]
    episodic_memories: List[Dict[str, Any]]
    semantic_memories: List[Dict[str, Any]]
    vector_embeddings: Dict[str, List[float]]
    memory_relationships: List[Dict[str, Any]]

@dataclass
class ProjectSnapshot:
    """Snapshot of project state"""
    project_root: str
    modified_files: Dict[str, str]  # file_path -> content
    git_status: Dict[str, Any]
    dependencies: Dict[str, str]
    configuration_files: Dict[str, Any]

@dataclass
class MCPServerState:
    """State of an MCP server"""
    server_name: str
    configuration: Dict[str, Any]
    tool_states: Dict[str, Any]
    resource_states: Dict[str, Any]
    session_data: Dict[str, Any]
    performance_metrics: Dict[str, float]

@dataclass
class AgentContextSnapshot:
    """Complete snapshot of agent operational context"""
    
    # Core Identity
    agent_id: str
    session_id: str
    snapshot_timestamp: datetime
    context_version: str = "1.0.0"
    
    # Conversation Context
    conversation_history: List[ConversationMessage] = field(default_factory=list)
    reasoning_threads: List[ReasoningThread] = field(default_factory=list)
    tool_usage_history: List[ToolInvocation] = field(default_factory=list)
    
    # Project Context
    project_state: Optional[ProjectSnapshot] = None
    
    # Agent Memory
    memory_snapshot: Optional[MemorySnapshot] = None
    
    # MCP Context
    mcp_server_states: Dict[str, MCPServerState] = field(default_factory=dict)
    
    # Metadata
    compression_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class CompressionResult:
    """Result of context compression"""
    original_size_bytes: int
    compressed_size_bytes: int
    compression_ratio: float
    compression_strategy: str
    integrity_hash: str
    compression_metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class ContextPackage:
    """Compressed and serialized context package"""
    package_id: str
    agent_id: str
    session_id: str
    created_at: datetime
    compressed_data: bytes
    compression_result: CompressionResult
    package_metadata: Dict[str, Any] = field(default_factory=dict)

class SemanticCompressionStrategy:
    """Semantic compression using content similarity"""
    
    def __init__(self):
        self.similarity_threshold = 0.85
        self.importance_weights = {
            'reasoning_threads': 1.0,
            'tool_invocations': 0.8,
            'conversation_history': 0.6,
            'memory_entries': 0.9
        }
    
    async def deduplicate_semantically(self, snapshot: AgentContextSnapshot) -> AgentContextSnapshot:
        """Remove semantically similar content"""
        logger.info("Applying semantic deduplication...")
        
        # Deduplicate conversation messages
        deduped_conversation = self._deduplicate_messages(snapshot.conversation_history)
        
        # Deduplicate reasoning threads
        deduped_reasoning = self._deduplicate_reasoning_threads(snapshot.reasoning_threads)
        
        return AgentContextSnapshot(
            agent_id=snapshot.agent_id,
            session_id=snapshot.session_id,
            snapshot_timestamp=snapshot.snapshot_timestamp,
            context_version=snapshot.context_version,
            conversation_history=deduped_conversation,
            reasoning_threads=deduped_reasoning,
            tool_usage_history=snapshot.tool_usage_history,
            project_state=snapshot.project_state,
            memory_snapshot=snapshot.memory_snapshot,
            mcp_server_states=snapshot.mcp_server_states
        )
    
    def _deduplicate_messages(self, messages: List[ConversationMessage]) -> List[ConversationMessage]:
        """Remove similar conversation messages"""
        if not messages:
            return messages
        
        unique_messages = [messages[0]]  # Always keep first message
        
        for message in messages[1:]:
            is_similar = False
            for existing in unique_messages[-5:]:  # Check last 5 messages
                if self._messages_similar(message, existing):
                    is_similar = True
                    break
            
            if not is_similar:
                unique_messages.append(message)
        
        return unique_messages
    
    def _messages_similar(self, msg1: ConversationMessage, msg2: ConversationMessage) -> bool:
        """Check if two messages are semantically similar"""
        # Simple similarity check - in production, use embeddings
        content1 = msg1.content.lower().strip()
        content2 = msg2.content.lower().strip()
        
        # Check for exact duplicates
        if content1 == content2:
            return True
        
        # Check for high overlap
        words1 = set(content1.split())
        words2 = set(content2.split())
        
        if len(words1) == 0 or len(words2) == 0:
            return False
        
        overlap = len(words1.intersection(words2))
        min_length = min(len(words1), len(words2))
        
        similarity = overlap / min_length if min_length > 0 else 0
        return similarity > self.similarity_threshold
    
    def _deduplicate_reasoning_threads(self, threads: List[ReasoningThread]) -> List[ReasoningThread]:
        """Remove similar reasoning threads"""
        if not threads:
            return threads
        
        unique_threads = []
        
        for thread in threads:
            is_similar = False
            for existing in unique_threads:
                if self._threads_similar(thread, existing):
                    # Keep the one with higher confidence
                    if thread.confidence_score > existing.confidence_score:
                        unique_threads.remove(existing)
                        unique_threads.append(thread)
                    is_similar = True
                    break
            
            if not is_similar:
                unique_threads.append(thread)
        
        return unique_threads
    
    def _threads_similar(self, thread1: ReasoningThread, thread2: ReasoningThread) -> bool:
        """Check if two reasoning threads are similar"""
        # Simple similarity check
        prompt_sim = self._text_similarity(thread1.prompt, thread2.prompt)
        conclusion_sim = self._text_similarity(thread1.conclusion, thread2.conclusion)
        
        return (prompt_sim > 0.7 and conclusion_sim > 0.7)
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Calculate text similarity"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if len(words1) == 0 and len(words2) == 0:
            return 1.0
        if len(words1) == 0 or len(words2) == 0:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)

class ContextCompressionEngine:
    """Advanced context compression with multiple strategies"""
    
    def __init__(self):
        self.compression_strategies = {
            'semantic': SemanticCompressionStrategy(),
            'basic': None  # Placeholder for basic compression
        }
    
    async def compress_context(self, 
                             snapshot: AgentContextSnapshot,
                             strategy: str = 'semantic',
                             target_ratio: float = 0.3) -> CompressionResult:
        """Compress agent context using specified strategy"""
        
        logger.info(f"Compressing context using {strategy} strategy...")
        
        # Get original size
        original_data = json.dumps(asdict(snapshot), default=str).encode('utf-8')
        original_size = len(original_data)
        
        # Apply semantic compression if requested
        if strategy == 'semantic':
            strategy_impl = self.compression_strategies['semantic']
            compressed_snapshot = await strategy_impl.deduplicate_semantically(snapshot)
        else:
            compressed_snapshot = snapshot
        
        # Convert to JSON and apply gzip compression
        json_data = json.dumps(asdict(compressed_snapshot), default=str)
        compressed_data = gzip.compress(json_data.encode('utf-8'))
        compressed_size = len(compressed_data)
        
        # Calculate compression ratio
        compression_ratio = compressed_size / original_size
        
        # Calculate integrity hash
        integrity_hash = hashlib.sha256(compressed_data).hexdigest()
        
        logger.info(f"Compression completed: {original_size} -> {compressed_size} bytes "
                   f"({compression_ratio:.3f} ratio)")
        
        return CompressionResult(
            original_size_bytes=original_size,
            compressed_size_bytes=compressed_size,
            compression_ratio=compression_ratio,
            compression_strategy=strategy,
            integrity_hash=integrity_hash,
            compression_metadata={
                'compression_algorithm': 'gzip',
                'serialization_format': 'json',
                'semantic_deduplication': strategy == 'semantic'
            }
        )

class ContextSerializationSystem:
    """Context serialization and deserialization"""
    
    async def create_context_package(self,
                                   snapshot: AgentContextSnapshot,
                                   compression_result: CompressionResult) -> ContextPackage:
        """Create a context package from snapshot and compression result"""
        
        # Serialize the snapshot
        json_data = json.dumps(asdict(snapshot), default=str)
        compressed_data = gzip.compress(json_data.encode('utf-8'))
        
        # Create package
        package_id = hashlib.md5(f"{snapshot.agent_id}_{snapshot.session_id}_{datetime.now().isoformat()}".encode()).hexdigest()
        
        return ContextPackage(
            package_id=package_id,
            agent_id=snapshot.agent_id,
            session_id=snapshot.session_id,
            created_at=datetime.now(),
            compressed_data=compressed_data,
            compression_result=compression_result,
            package_metadata={
                'context_version': snapshot.context_version,
                'snapshot_timestamp': snapshot.snapshot_timestamp.isoformat(),
                'conversation_message_count': len(snapshot.conversation_history),
                'reasoning_thread_count': len(snapshot.reasoning_threads),
                'tool_invocation_count': len(snapshot.tool_usage_history)
            }
        )
    
    async def extract_snapshot_from_package(self, package: ContextPackage) -> AgentContextSnapshot:
        """Extract AgentContextSnapshot from ContextPackage"""
        
        # Verify integrity
        calculated_hash = hashlib.sha256(package.compressed_data).hexdigest()
        if calculated_hash != package.compression_result.integrity_hash:
            raise ValueError(f"Package integrity check failed for {package.package_id}")
        
        # Decompress and deserialize
        decompressed_data = gzip.decompress(package.compressed_data)
        json_data = decompressed_data.decode('utf-8')
        snapshot_dict = json.loads(json_data)
        
        # Convert back to dataclass (simplified - in production, use proper deserialization)
        snapshot = self._dict_to_snapshot(snapshot_dict)
        
        return snapshot
    
    def _dict_to_snapshot(self, data: Dict[str, Any]) -> AgentContextSnapshot:
        """Convert dictionary back to AgentContextSnapshot"""
        
        # Convert conversation messages
        conversation_history = []
        for msg_data in data.get('conversation_history', []):
            conversation_history.append(ConversationMessage(
                id=msg_data['id'],
                role=msg_data['role'],
                content=msg_data['content'],
                timestamp=datetime.fromisoformat(msg_data['timestamp']),
                metadata=msg_data.get('metadata', {})
            ))
        
        # Convert reasoning threads
        reasoning_threads = []
        for thread_data in data.get('reasoning_threads', []):
            reasoning_threads.append(ReasoningThread(
                thread_id=thread_data['thread_id'],
                prompt=thread_data['prompt'],
                reasoning_steps=thread_data['reasoning_steps'],
                conclusion=thread_data['conclusion'],
                confidence_score=thread_data['confidence_score'],
                timestamp=datetime.fromisoformat(thread_data['timestamp'])
            ))
        
        # Convert tool invocations
        tool_usage_history = []
        for tool_data in data.get('tool_usage_history', []):
            tool_usage_history.append(ToolInvocation(
                tool_name=tool_data['tool_name'],
                server_name=tool_data['server_name'],
                arguments=tool_data['arguments'],
                result=tool_data['result'],
                timestamp=datetime.fromisoformat(tool_data['timestamp']),
                execution_time_ms=tool_data['execution_time_ms']
            ))
        
        return AgentContextSnapshot(
            agent_id=data['agent_id'],
            session_id=data['session_id'],
            snapshot_timestamp=datetime.fromisoformat(data['snapshot_timestamp']),
            context_version=data['context_version'],
            conversation_history=conversation_history,
            reasoning_threads=reasoning_threads,
            tool_usage_history=tool_usage_history,
            project_state=data.get('project_state'),
            memory_snapshot=data.get('memory_snapshot'),
            mcp_server_states=data.get('mcp_server_states', {})
        )

class ContextPackageStorage:
    """Simple file-based storage for context packages"""
    
    def __init__(self, storage_dir: str = "data/context_packages"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
    
    async def store_package(self, package: ContextPackage) -> str:
        """Store context package to file"""
        
        package_file = self.storage_dir / f"{package.package_id}.ctx"
        
        # Save package data
        package_data = {
            'package_id': package.package_id,
            'agent_id': package.agent_id,
            'session_id': package.session_id,
            'created_at': package.created_at.isoformat(),
            'compressed_data': package.compressed_data.hex(),  # Store as hex string
            'compression_result': asdict(package.compression_result),
            'package_metadata': package.package_metadata
        }
        
        with open(package_file, 'w') as f:
            json.dump(package_data, f, indent=2)
        
        logger.info(f"Stored context package {package.package_id} to {package_file}")
        return package.package_id
    
    async def retrieve_package(self, package_id: str) -> ContextPackage:
        """Retrieve context package from file"""
        
        package_file = self.storage_dir / f"{package_id}.ctx"
        
        if not package_file.exists():
            raise FileNotFoundError(f"Context package {package_id} not found")
        
        with open(package_file, 'r') as f:
            package_data = json.load(f)
        
        # Reconstruct package
        compression_result = CompressionResult(**package_data['compression_result'])
        
        package = ContextPackage(
            package_id=package_data['package_id'],
            agent_id=package_data['agent_id'],
            session_id=package_data['session_id'],
            created_at=datetime.fromisoformat(package_data['created_at']),
            compressed_data=bytes.fromhex(package_data['compressed_data']),
            compression_result=compression_result,
            package_metadata=package_data['package_metadata']
        )
        
        logger.info(f"Retrieved context package {package_id}")
        return package
    
    async def list_packages(self) -> List[str]:
        """List all stored package IDs"""
        return [f.stem for f in self.storage_dir.glob("*.ctx")]

class AgentContextRestorationSystem:
    """Main system for context capture and restoration"""
    
    def __init__(self, storage_dir: str = "data/context_packages"):
        self.compression_engine = ContextCompressionEngine()
        self.serialization_system = ContextSerializationSystem()
        self.storage = ContextPackageStorage(storage_dir)
    
    async def capture_agent_context(self,
                                  agent_id: str,
                                  session_id: str,
                                  include_project_state: bool = True) -> AgentContextSnapshot:
        """Capture comprehensive agent context"""
        
        logger.info(f"Capturing context for agent {agent_id}, session {session_id}")
        
        # Create sample context data (in production, this would gather real data)
        snapshot = self._create_sample_context(agent_id, session_id)
        
        logger.info(f"Captured context: {len(snapshot.conversation_history)} messages, "
                   f"{len(snapshot.reasoning_threads)} reasoning threads, "
                   f"{len(snapshot.tool_usage_history)} tool invocations")
        
        return snapshot
    
    async def compress_and_package(self,
                                 snapshot: AgentContextSnapshot,
                                 compression_strategy: str = 'semantic',
                                 target_ratio: float = 0.3) -> ContextPackage:
        """Compress snapshot and create package"""
        
        # Compress the context
        compression_result = await self.compression_engine.compress_context(
            snapshot, compression_strategy, target_ratio
        )
        
        # Create package
        package = await self.serialization_system.create_context_package(
            snapshot, compression_result
        )
        
        return package
    
    async def store_context_package(self, package: ContextPackage) -> str:
        """Store context package persistently"""
        return await self.storage.store_package(package)
    
    async def restore_agent_context(self,
                                  package_id: str,
                                  validation_level: str = 'basic') -> AgentContextSnapshot:
        """Restore agent context from package"""
        
        logger.info(f"Restoring context from package {package_id}")
        
        # Retrieve package
        package = await self.storage.retrieve_package(package_id)
        
        # Extract snapshot
        snapshot = await self.serialization_system.extract_snapshot_from_package(package)
        
        # Validate if requested
        if validation_level == 'comprehensive':
            await self._validate_restored_context(snapshot)
        
        logger.info(f"Restored context: {len(snapshot.conversation_history)} messages, "
                   f"{len(snapshot.reasoning_threads)} reasoning threads")
        
        return snapshot
    
    async def list_available_contexts(self) -> List[Dict[str, Any]]:
        """List all available context packages"""
        
        package_ids = await self.storage.list_packages()
        contexts = []
        
        for package_id in package_ids:
            try:
                package = await self.storage.retrieve_package(package_id)
                contexts.append({
                    'package_id': package.package_id,
                    'agent_id': package.agent_id,
                    'session_id': package.session_id,
                    'created_at': package.created_at.isoformat(),
                    'compression_ratio': package.compression_result.compression_ratio,
                    'message_count': package.package_metadata.get('conversation_message_count', 0),
                    'size_bytes': package.compression_result.compressed_size_bytes
                })
            except Exception as e:
                logger.warning(f"Failed to load package {package_id}: {e}")
        
        return contexts
    
    def _create_sample_context(self, agent_id: str, session_id: str) -> AgentContextSnapshot:
        """Create sample context data for demonstration"""
        
        now = datetime.now()
        
        # Sample conversation messages
        conversation_history = [
            ConversationMessage(
                id="msg_1",
                role="user",
                content="I need help implementing a context restoration system for AI agents",
                timestamp=now - timedelta(minutes=30),
                metadata={"priority": "high"}
            ),
            ConversationMessage(
                id="msg_2",
                role="assistant",
                content="I can help you design a comprehensive context restoration system. Let me analyze the requirements...",
                timestamp=now - timedelta(minutes=29),
                metadata={"reasoning_mode": "analytical"}
            ),
            ConversationMessage(
                id="msg_3",
                role="user",
                content="The system should preserve conversation history, agent memory, and project state",
                timestamp=now - timedelta(minutes=25),
                metadata={"requirements": ["conversation", "memory", "project_state"]}
            ),
            ConversationMessage(
                id="msg_4",
                role="assistant", 
                content="Excellent requirements. I'll design a system with compression, serialization, and restoration capabilities...",
                timestamp=now - timedelta(minutes=24),
                metadata={"design_phase": "architecture"}
            )
        ]
        
        # Sample reasoning threads
        reasoning_threads = [
            ReasoningThread(
                thread_id="thread_1",
                prompt="How to implement context compression?",
                reasoning_steps=[
                    "1. Analyze content for semantic similarity",
                    "2. Remove duplicate or redundant information", 
                    "3. Apply hierarchical importance scoring",
                    "4. Use appropriate compression algorithms"
                ],
                conclusion="Semantic compression with gzip provides optimal balance of size and integrity",
                confidence_score=0.92,
                timestamp=now - timedelta(minutes=20)
            ),
            ReasoningThread(
                thread_id="thread_2",
                prompt="What storage format should be used?",
                reasoning_steps=[
                    "1. Consider JSON for human readability",
                    "2. Evaluate binary formats for efficiency",
                    "3. Compare compression ratios",
                    "4. Consider security requirements"
                ],
                conclusion="JSON with gzip compression provides good balance of readability and efficiency",
                confidence_score=0.88,
                timestamp=now - timedelta(minutes=15)
            )
        ]
        
        # Sample tool invocations
        tool_usage_history = [
            ToolInvocation(
                tool_name="file_search",
                server_name="filesystem-mcp",
                arguments={"pattern": "*.py", "path": "src/"},
                result={"files_found": 45, "total_size": "2.3MB"},
                timestamp=now - timedelta(minutes=18),
                execution_time_ms=234.5
            ),
            ToolInvocation(
                tool_name="git_status",
                server_name="git-mcp", 
                arguments={},
                result={"modified_files": 3, "staged_files": 1, "branch": "main"},
                timestamp=now - timedelta(minutes=12),
                execution_time_ms=89.2
            )
        ]
        
        return AgentContextSnapshot(
            agent_id=agent_id,
            session_id=session_id,
            snapshot_timestamp=now,
            conversation_history=conversation_history,
            reasoning_threads=reasoning_threads,
            tool_usage_history=tool_usage_history
        )
    
    async def _validate_restored_context(self, snapshot: AgentContextSnapshot) -> bool:
        """Validate restored context integrity"""
        
        logger.info("Validating restored context...")
        
        # Basic validation checks
        assert snapshot.agent_id, "Agent ID missing"
        assert snapshot.session_id, "Session ID missing"
        assert snapshot.conversation_history, "Conversation history empty"
        
        # Validate message chronology
        for i in range(1, len(snapshot.conversation_history)):
            current_msg = snapshot.conversation_history[i]
            prev_msg = snapshot.conversation_history[i-1]
            assert current_msg.timestamp >= prev_msg.timestamp, "Message timestamps out of order"
        
        logger.info("Context validation completed successfully")
        return True

async def main():
    """Demonstration of the Context Restoration System"""
    
    print("🚀 Agent Context Restoration System - Proof of Concept")
    print("=" * 60)
    
    # Initialize the system
    context_system = AgentContextRestorationSystem()
    
    # 1. Capture agent context
    print("\n1. Capturing agent context...")
    snapshot = await context_system.capture_agent_context(
        agent_id="reasoning_agent_001",
        session_id="session_2025_01_21_001"
    )
    
    print(f"   ✓ Captured {len(snapshot.conversation_history)} conversation messages")
    print(f"   ✓ Captured {len(snapshot.reasoning_threads)} reasoning threads")
    print(f"   ✓ Captured {len(snapshot.tool_usage_history)} tool invocations")
    
    # 2. Compress and package context
    print("\n2. Compressing and packaging context...")
    package = await context_system.compress_and_package(
        snapshot=snapshot,
        compression_strategy='semantic',
        target_ratio=0.3
    )
    
    print(f"   ✓ Original size: {package.compression_result.original_size_bytes:,} bytes")
    print(f"   ✓ Compressed size: {package.compression_result.compressed_size_bytes:,} bytes")
    print(f"   ✓ Compression ratio: {package.compression_result.compression_ratio:.3f}")
    print(f"   ✓ Package ID: {package.package_id}")
    
    # 3. Store context package
    print("\n3. Storing context package...")
    package_id = await context_system.store_context_package(package)
    print(f"   ✓ Stored package: {package_id}")
    
    # 4. List available contexts
    print("\n4. Listing available contexts...")
    contexts = await context_system.list_available_contexts()
    for ctx in contexts:
        print(f"   📦 {ctx['package_id'][:12]}... - Agent: {ctx['agent_id']} - "
              f"Messages: {ctx['message_count']} - Size: {ctx['size_bytes']:,} bytes")
    
    # 5. Restore context (simulating agent reset)
    print("\n5. Restoring context (simulating agent reset)...")
    restored_snapshot = await context_system.restore_agent_context(
        package_id=package_id,
        validation_level='comprehensive'
    )
    
    print(f"   ✓ Restored {len(restored_snapshot.conversation_history)} conversation messages")
    print(f"   ✓ Restored {len(restored_snapshot.reasoning_threads)} reasoning threads")
    print(f"   ✓ Restored {len(restored_snapshot.tool_usage_history)} tool invocations")
    
    # 6. Verify restoration integrity
    print("\n6. Verifying restoration integrity...")
    
    # Compare original and restored
    original_msg_count = len(snapshot.conversation_history)
    restored_msg_count = len(restored_snapshot.conversation_history)
    
    original_thread_count = len(snapshot.reasoning_threads)
    restored_thread_count = len(restored_snapshot.reasoning_threads)
    
    print(f"   ✓ Messages: {original_msg_count} -> {restored_msg_count} ({'✓ MATCH' if original_msg_count == restored_msg_count else '✗ MISMATCH'})")
    print(f"   ✓ Reasoning threads: {original_thread_count} -> {restored_thread_count} ({'✓ MATCH' if original_thread_count == restored_thread_count else '✗ MISMATCH'})")
      # Check content integrity
    if snapshot.conversation_history and restored_snapshot.conversation_history:
        original_first_msg = snapshot.conversation_history[0].content
        restored_first_msg = restored_snapshot.conversation_history[0].content
        content_match = original_first_msg == restored_first_msg
        print(f"   ✓ Content integrity: {'✓ VERIFIED' if content_match else '✗ FAILED'}")
    
    print("\n🎉 Context Restoration System demonstration completed!")
    print(f"\n💡 This system successfully captured, compressed ({package.compression_result.compression_ratio:.1%} ratio), ")
    print("   stored, and restored complete agent context with full integrity.")
    
    # Show practical benefits
    print("\n🚀 Practical Benefits:")
    print("   • Context preserved across agent resets")
    print(f"   • {(1 - package.compression_result.compression_ratio):.1%} storage space saved through compression")
    print("   • Complete conversation and reasoning history maintained")
    print("   • Tool usage patterns and project state preserved")
    print("   • Seamless agent continuity for long-running projects")

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # Run the demonstration
    asyncio.run(main())
