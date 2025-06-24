"""
Knowledge Graph Implementation

Provides knowledge graph storage and querying capabilities for PyGent Factory,
supporting entity relationships, graph traversal, and knowledge management.
"""

import logging
import asyncio
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import json

logger = logging.getLogger(__name__)


@dataclass
class Entity:
    """Knowledge graph entity"""
    id: str
    name: str
    entity_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    updated_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class Relationship:
    """Relationship between entities"""
    id: str
    source_id: str
    target_id: str
    relationship_type: str
    properties: Dict[str, Any] = field(default_factory=dict)
    weight: float = 1.0
    created_at: datetime = field(default_factory=datetime.utcnow)


@dataclass
class GraphQuery:
    """Graph query specification"""
    entity_types: Optional[List[str]] = None
    relationship_types: Optional[List[str]] = None
    properties: Optional[Dict[str, Any]] = None
    max_depth: int = 3
    limit: int = 100


@dataclass
class GraphPath:
    """Path through knowledge graph"""
    entities: List[Entity]
    relationships: List[Relationship]
    total_weight: float
    path_length: int


class KnowledgeGraph:
    """
    Knowledge graph for storing and querying entity relationships.
    
    Provides graph-based knowledge storage with support for
    entity management, relationship tracking, and graph traversal.
    """
    
    def __init__(self, max_entities: int = 10000, max_relationships: int = 50000):
        self.max_entities = max_entities
        self.max_relationships = max_relationships
        self.entities: Dict[str, Entity] = {}
        self.relationships: Dict[str, Relationship] = {}
        self.entity_relationships: Dict[str, Set[str]] = {}  # entity_id -> relationship_ids
        self.type_index: Dict[str, Set[str]] = {}  # entity_type -> entity_ids
        self.is_initialized = False
    
    async def initialize(self):
        """Initialize knowledge graph"""
        try:
            self.is_initialized = True
            logger.info("Knowledge graph initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize knowledge graph: {e}")
            raise
    
    async def add_entity(self, entity: Entity) -> bool:
        """Add entity to knowledge graph"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Check limits
            if len(self.entities) >= self.max_entities:
                logger.warning("Maximum entities limit reached")
                return False
            
            # Store entity
            self.entities[entity.id] = entity
            
            # Update type index
            if entity.entity_type not in self.type_index:
                self.type_index[entity.entity_type] = set()
            self.type_index[entity.entity_type].add(entity.id)
            
            # Initialize relationships set
            if entity.id not in self.entity_relationships:
                self.entity_relationships[entity.id] = set()
            
            logger.debug(f"Added entity {entity.id} of type {entity.entity_type}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add entity {entity.id}: {e}")
            return False
    
    async def add_relationship(self, relationship: Relationship) -> bool:
        """Add relationship to knowledge graph"""
        try:
            if not self.is_initialized:
                await self.initialize()
            
            # Check if entities exist
            if relationship.source_id not in self.entities:
                logger.error(f"Source entity {relationship.source_id} not found")
                return False
            
            if relationship.target_id not in self.entities:
                logger.error(f"Target entity {relationship.target_id} not found")
                return False
            
            # Check limits
            if len(self.relationships) >= self.max_relationships:
                logger.warning("Maximum relationships limit reached")
                return False
            
            # Store relationship
            self.relationships[relationship.id] = relationship
            
            # Update entity relationships
            self.entity_relationships[relationship.source_id].add(relationship.id)
            self.entity_relationships[relationship.target_id].add(relationship.id)
            
            logger.debug(f"Added relationship {relationship.id} between {relationship.source_id} and {relationship.target_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to add relationship {relationship.id}: {e}")
            return False
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID"""
        return self.entities.get(entity_id)
    
    async def get_entities_by_type(self, entity_type: str) -> List[Entity]:
        """Get all entities of a specific type"""
        entity_ids = self.type_index.get(entity_type, set())
        return [self.entities[entity_id] for entity_id in entity_ids if entity_id in self.entities]
    
    async def get_entity_relationships(self, entity_id: str) -> List[Relationship]:
        """Get all relationships for an entity"""
        if entity_id not in self.entity_relationships:
            return []
        
        relationship_ids = self.entity_relationships[entity_id]
        return [self.relationships[rel_id] for rel_id in relationship_ids if rel_id in self.relationships]
    
    async def get_connected_entities(self, entity_id: str, relationship_type: Optional[str] = None) -> List[Entity]:
        """Get entities connected to the given entity"""
        relationships = await self.get_entity_relationships(entity_id)
        
        connected_entities = []
        for rel in relationships:
            if relationship_type and rel.relationship_type != relationship_type:
                continue
            
            # Get the other entity in the relationship
            other_entity_id = rel.target_id if rel.source_id == entity_id else rel.source_id
            
            if other_entity_id in self.entities:
                connected_entities.append(self.entities[other_entity_id])
        
        return connected_entities
    
    async def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> Optional[GraphPath]:
        """Find shortest path between two entities"""
        if source_id not in self.entities or target_id not in self.entities:
            return None
        
        if source_id == target_id:
            return GraphPath(
                entities=[self.entities[source_id]],
                relationships=[],
                total_weight=0.0,
                path_length=0
            )
        
        # BFS to find shortest path
        queue = [(source_id, [source_id], [], 0.0)]
        visited = {source_id}
        
        while queue:
            current_id, path, rel_path, total_weight = queue.pop(0)
            
            if len(path) > max_depth + 1:
                continue
            
            # Get relationships for current entity
            relationships = await self.get_entity_relationships(current_id)
            
            for rel in relationships:
                next_id = rel.target_id if rel.source_id == current_id else rel.source_id
                
                if next_id == target_id:
                    # Found target
                    final_path = path + [next_id]
                    final_rel_path = rel_path + [rel]
                    final_weight = total_weight + rel.weight
                    
                    return GraphPath(
                        entities=[self.entities[eid] for eid in final_path],
                        relationships=final_rel_path,
                        total_weight=final_weight,
                        path_length=len(final_rel_path)
                    )
                
                if next_id not in visited and len(path) < max_depth + 1:
                    visited.add(next_id)
                    queue.append((
                        next_id,
                        path + [next_id],
                        rel_path + [rel],
                        total_weight + rel.weight
                    ))
        
        return None
    
    async def query_graph(self, query: GraphQuery) -> List[Entity]:
        """Query graph with specified criteria"""
        results = []
        
        # Start with all entities or filter by type
        candidate_entities = []
        if query.entity_types:
            for entity_type in query.entity_types:
                candidate_entities.extend(await self.get_entities_by_type(entity_type))
        else:
            candidate_entities = list(self.entities.values())
        
        # Filter by properties
        for entity in candidate_entities:
            if query.properties:
                match = True
                for key, value in query.properties.items():
                    if key not in entity.properties or entity.properties[key] != value:
                        match = False
                        break
                if not match:
                    continue
            
            results.append(entity)
            
            if len(results) >= query.limit:
                break
        
        return results
    
    async def delete_entity(self, entity_id: str) -> bool:
        """Delete entity and its relationships"""
        try:
            if entity_id not in self.entities:
                return False
            
            entity = self.entities[entity_id]
            
            # Delete all relationships involving this entity
            relationship_ids = self.entity_relationships.get(entity_id, set()).copy()
            for rel_id in relationship_ids:
                await self.delete_relationship(rel_id)
            
            # Remove from type index
            if entity.entity_type in self.type_index:
                self.type_index[entity.entity_type].discard(entity_id)
            
            # Remove entity
            del self.entities[entity_id]
            del self.entity_relationships[entity_id]
            
            logger.debug(f"Deleted entity {entity_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete entity {entity_id}: {e}")
            return False
    
    async def delete_relationship(self, relationship_id: str) -> bool:
        """Delete relationship"""
        try:
            if relationship_id not in self.relationships:
                return False
            
            relationship = self.relationships[relationship_id]
            
            # Remove from entity relationships
            self.entity_relationships[relationship.source_id].discard(relationship_id)
            self.entity_relationships[relationship.target_id].discard(relationship_id)
            
            # Remove relationship
            del self.relationships[relationship_id]
            
            logger.debug(f"Deleted relationship {relationship_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete relationship {relationship_id}: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """Get knowledge graph statistics"""
        return {
            "total_entities": len(self.entities),
            "total_relationships": len(self.relationships),
            "entity_types": len(self.type_index),
            "is_initialized": self.is_initialized
        }
