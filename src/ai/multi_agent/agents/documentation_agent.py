"""
Documentation Agent

Specialized agent for generating, maintaining, and improving documentation
in software development and research workflows.
"""

import logging
import asyncio
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum

logger = logging.getLogger(__name__)


class DocumentationType(Enum):
    """Types of documentation"""
    API = "api"
    USER_GUIDE = "user_guide"
    TECHNICAL = "technical"
    README = "readme"
    CHANGELOG = "changelog"
    TUTORIAL = "tutorial"
    REFERENCE = "reference"


@dataclass
class DocumentationItem:
    """Documentation item"""
    doc_id: str
    title: str
    content: str
    doc_type: DocumentationType
    language: str = "en"
    version: str = "1.0"
    created_at: datetime = None
    updated_at: datetime = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = datetime.utcnow()
        if self.updated_at is None:
            self.updated_at = self.created_at
        if self.tags is None:
            self.tags = []


class DocumentationAgent:
    """
    Agent specialized in documentation generation and maintenance.
    
    Capabilities:
    - Automatic documentation generation
    - Documentation quality assessment
    - Multi-language documentation support
    - Documentation maintenance and updates
    - Style and consistency checking
    """
    
    def __init__(self, agent_id: str = "documentation_agent"):
        self.agent_id = agent_id
        self.agent_type = "documentation"
        self.status = "initialized"
        self.capabilities = [
            "documentation_generation",
            "quality_assessment",
            "multi_language_support",
            "maintenance_updates",
            "style_checking"
        ]
        
        # Documentation state
        self.documentation_items: Dict[str, DocumentationItem] = {}
        self.templates: Dict[str, str] = {}
        
        # Configuration
        self.config = {
            'supported_languages': ['en', 'es', 'fr', 'de', 'zh', 'ja'],
            'default_language': 'en',
            'max_content_length': 50000,
            'quality_threshold': 0.7,
            'auto_update_enabled': True,
            'style_guide': 'standard'
        }
        
        # Statistics
        self.stats = {
            'documents_generated': 0,
            'documents_updated': 0,
            'quality_assessments': 0,
            'translations_performed': 0,
            'avg_generation_time_ms': 0.0,
            'successful_generations': 0
        }
        
        # Initialize templates
        self._initialize_templates()
        
        logger.info(f"DocumentationAgent {agent_id} initialized")
    
    async def start(self) -> bool:
        """Start the documentation agent"""
        try:
            self.status = "active"
            logger.info(f"DocumentationAgent {self.agent_id} started")
            return True
        except Exception as e:
            logger.error(f"Failed to start DocumentationAgent {self.agent_id}: {e}")
            self.status = "error"
            return False
    
    async def stop(self) -> bool:
        """Stop the documentation agent"""
        try:
            self.status = "stopped"
            logger.info(f"DocumentationAgent {self.agent_id} stopped")
            return True
        except Exception as e:
            logger.error(f"Failed to stop DocumentationAgent {self.agent_id}: {e}")
            return False
    
    async def generate_documentation(self, content_data: Dict[str, Any], 
                                   doc_type: str = "technical") -> Dict[str, Any]:
        """
        Generate documentation from content data.
        
        Args:
            content_data: Data to generate documentation from
            doc_type: Type of documentation to generate
            
        Returns:
            Generated documentation results
        """
        start_time = datetime.utcnow()
        
        try:
            # Validate document type
            try:
                documentation_type = DocumentationType(doc_type)
            except ValueError:
                documentation_type = DocumentationType.TECHNICAL
            
            # Generate documentation content
            generated_content = await self._generate_content(content_data, documentation_type)
            
            # Create documentation item
            doc_id = f"doc_{int(start_time.timestamp())}"
            doc_item = DocumentationItem(
                doc_id=doc_id,
                title=content_data.get('title', f"Generated {doc_type.title()} Documentation"),
                content=generated_content,
                doc_type=documentation_type,
                language=content_data.get('language', self.config['default_language']),
                tags=content_data.get('tags', [])
            )
            
            # Store documentation
            self.documentation_items[doc_id] = doc_item
            
            # Assess quality
            quality_score = await self._assess_quality(generated_content, documentation_type)
            
            # Update statistics
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_generation_stats(True, generation_time)
            
            result = {
                'doc_id': doc_id,
                'success': True,
                'title': doc_item.title,
                'content_length': len(generated_content),
                'doc_type': doc_type,
                'language': doc_item.language,
                'quality_score': quality_score,
                'generation_time_ms': generation_time,
                'timestamp': start_time.isoformat()
            }
            
            logger.debug(f"Documentation generated in {generation_time:.2f}ms")
            return result
            
        except Exception as e:
            generation_time = (datetime.utcnow() - start_time).total_seconds() * 1000
            self._update_generation_stats(False, generation_time)
            logger.error(f"Documentation generation failed: {e}")
            return {
                'success': False,
                'error': str(e),
                'generation_time_ms': generation_time,
                'timestamp': start_time.isoformat()
            }
    
    async def update_documentation(self, doc_id: str, updates: Dict[str, Any]) -> bool:
        """Update existing documentation"""
        try:
            if doc_id not in self.documentation_items:
                logger.warning(f"Documentation {doc_id} not found for update")
                return False
            
            doc_item = self.documentation_items[doc_id]
            
            # Apply updates
            if 'title' in updates:
                doc_item.title = updates['title']
            if 'content' in updates:
                doc_item.content = updates['content']
            if 'tags' in updates:
                doc_item.tags = updates['tags']
            
            doc_item.updated_at = datetime.utcnow()
            
            self.stats['documents_updated'] += 1
            logger.info(f"Documentation {doc_id} updated successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to update documentation {doc_id}: {e}")
            return False
    
    async def assess_documentation_quality(self, content: str, doc_type: str = "technical") -> Dict[str, Any]:
        """Assess the quality of documentation content"""
        try:
            documentation_type = DocumentationType(doc_type)
        except ValueError:
            documentation_type = DocumentationType.TECHNICAL
        
        quality_score = await self._assess_quality(content, documentation_type)
        
        # Detailed quality assessment
        assessment = {
            'overall_score': quality_score,
            'readability': self._assess_readability(content),
            'completeness': self._assess_completeness(content, documentation_type),
            'structure': self._assess_structure(content),
            'clarity': self._assess_clarity(content),
            'recommendations': self._generate_quality_recommendations(content, quality_score)
        }
        
        self.stats['quality_assessments'] += 1
        return assessment
    
    def get_documentation(self, doc_id: Optional[str] = None) -> Dict[str, Any]:
        """Get documentation items"""
        if doc_id:
            if doc_id in self.documentation_items:
                doc_item = self.documentation_items[doc_id]
                return {
                    'doc_id': doc_item.doc_id,
                    'title': doc_item.title,
                    'content': doc_item.content,
                    'doc_type': doc_item.doc_type.value,
                    'language': doc_item.language,
                    'version': doc_item.version,
                    'created_at': doc_item.created_at.isoformat(),
                    'updated_at': doc_item.updated_at.isoformat(),
                    'tags': doc_item.tags
                }
            else:
                return {'error': f'Documentation {doc_id} not found'}
        else:
            # Return all documentation
            return {
                doc_id: {
                    'title': doc.title,
                    'doc_type': doc.doc_type.value,
                    'language': doc.language,
                    'created_at': doc.created_at.isoformat(),
                    'updated_at': doc.updated_at.isoformat(),
                    'content_length': len(doc.content)
                }
                for doc_id, doc in self.documentation_items.items()
            }
    
    async def _generate_content(self, content_data: Dict[str, Any], 
                              doc_type: DocumentationType) -> str:
        """Generate documentation content based on type"""
        
        if doc_type == DocumentationType.API:
            return await self._generate_api_documentation(content_data)
        elif doc_type == DocumentationType.USER_GUIDE:
            return await self._generate_user_guide(content_data)
        elif doc_type == DocumentationType.README:
            return await self._generate_readme(content_data)
        elif doc_type == DocumentationType.TUTORIAL:
            return await self._generate_tutorial(content_data)
        else:
            return await self._generate_technical_documentation(content_data)
    
    async def _generate_api_documentation(self, content_data: Dict[str, Any]) -> str:
        """Generate API documentation"""
        template = self.templates.get('api', self.templates['default'])
        
        # Extract API information
        api_name = content_data.get('api_name', 'API')
        endpoints = content_data.get('endpoints', [])
        base_url = content_data.get('base_url', 'https://api.example.com')
        
        content = template.format(
            title=f"{api_name} API Documentation",
            description=content_data.get('description', f'Documentation for {api_name} API'),
            base_url=base_url,
            endpoints_section=self._format_endpoints(endpoints)
        )
        
        return content
    
    async def _generate_user_guide(self, content_data: Dict[str, Any]) -> str:
        """Generate user guide documentation"""
        template = self.templates.get('user_guide', self.templates['default'])
        
        content = template.format(
            title=content_data.get('title', 'User Guide'),
            description=content_data.get('description', 'User guide documentation'),
            features=self._format_features(content_data.get('features', [])),
            getting_started=content_data.get('getting_started', 'Getting started information')
        )
        
        return content
    
    async def _generate_readme(self, content_data: Dict[str, Any]) -> str:
        """Generate README documentation"""
        template = self.templates.get('readme', self.templates['default'])
        
        content = template.format(
            project_name=content_data.get('project_name', 'Project'),
            description=content_data.get('description', 'Project description'),
            installation=content_data.get('installation', 'Installation instructions'),
            usage=content_data.get('usage', 'Usage examples'),
            contributing=content_data.get('contributing', 'Contributing guidelines')
        )
        
        return content
    
    async def _generate_tutorial(self, content_data: Dict[str, Any]) -> str:
        """Generate tutorial documentation"""
        template = self.templates.get('tutorial', self.templates['default'])
        
        steps = content_data.get('steps', [])
        formatted_steps = '\n'.join([f"{i+1}. {step}" for i, step in enumerate(steps)])
        
        content = template.format(
            title=content_data.get('title', 'Tutorial'),
            description=content_data.get('description', 'Tutorial documentation'),
            prerequisites=content_data.get('prerequisites', 'No prerequisites'),
            steps=formatted_steps
        )
        
        return content
    
    async def _generate_technical_documentation(self, content_data: Dict[str, Any]) -> str:
        """Generate technical documentation"""
        template = self.templates.get('technical', self.templates['default'])
        
        content = template.format(
            title=content_data.get('title', 'Technical Documentation'),
            description=content_data.get('description', 'Technical documentation'),
            overview=content_data.get('overview', 'Technical overview'),
            details=content_data.get('details', 'Technical details')
        )
        
        return content
    
    async def _assess_quality(self, content: str, doc_type: DocumentationType) -> float:
        """Assess documentation quality"""
        score = 0.0
        
        # Length assessment
        if len(content) > 100:
            score += 0.2
        
        # Structure assessment
        if '# ' in content or '## ' in content:  # Has headers
            score += 0.2
        
        # Content completeness
        if len(content.split()) > 50:  # Has substantial content
            score += 0.3
        
        # Type-specific assessment
        if doc_type == DocumentationType.API:
            if 'endpoint' in content.lower() or 'api' in content.lower():
                score += 0.2
        elif doc_type == DocumentationType.README:
            if 'installation' in content.lower() and 'usage' in content.lower():
                score += 0.2
        else:
            score += 0.1
        
        # Readability
        sentences = content.split('.')
        if len(sentences) > 3:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_readability(self, content: str) -> float:
        """Assess content readability"""
        words = content.split()
        sentences = content.split('.')
        
        if not sentences:
            return 0.0
        
        avg_words_per_sentence = len(words) / len(sentences)
        
        # Simple readability score (lower is better)
        if avg_words_per_sentence <= 15:
            return 0.9
        elif avg_words_per_sentence <= 20:
            return 0.7
        elif avg_words_per_sentence <= 25:
            return 0.5
        else:
            return 0.3
    
    def _assess_completeness(self, content: str, doc_type: DocumentationType) -> float:
        """Assess content completeness"""
        required_sections = {
            DocumentationType.API: ['endpoint', 'parameter', 'response'],
            DocumentationType.README: ['installation', 'usage', 'description'],
            DocumentationType.USER_GUIDE: ['getting started', 'feature', 'example'],
            DocumentationType.TUTORIAL: ['step', 'example', 'prerequisite']
        }
        
        content_lower = content.lower()
        required = required_sections.get(doc_type, ['overview', 'detail'])
        
        found_sections = sum(1 for section in required if section in content_lower)
        return found_sections / len(required)
    
    def _assess_structure(self, content: str) -> float:
        """Assess content structure"""
        score = 0.0
        
        # Check for headers
        if '# ' in content:
            score += 0.4
        if '## ' in content:
            score += 0.3
        
        # Check for lists
        if '- ' in content or '* ' in content or '1. ' in content:
            score += 0.2
        
        # Check for code blocks
        if '```' in content or '`' in content:
            score += 0.1
        
        return min(1.0, score)
    
    def _assess_clarity(self, content: str) -> float:
        """Assess content clarity"""
        # Simple clarity assessment based on word complexity
        words = content.split()
        complex_words = sum(1 for word in words if len(word) > 10)
        
        if not words:
            return 0.0
        
        complexity_ratio = complex_words / len(words)
        
        # Lower complexity ratio = higher clarity
        return max(0.0, 1.0 - complexity_ratio * 2)
    
    def _generate_quality_recommendations(self, content: str, quality_score: float) -> List[str]:
        """Generate recommendations for improving documentation quality"""
        recommendations = []
        
        if quality_score < 0.7:
            recommendations.append("Consider adding more detailed content")
        
        if '# ' not in content:
            recommendations.append("Add section headers to improve structure")
        
        if len(content.split()) < 50:
            recommendations.append("Expand content with more detailed explanations")
        
        if '```' not in content and '`' not in content:
            recommendations.append("Add code examples to illustrate concepts")
        
        if '- ' not in content and '* ' not in content:
            recommendations.append("Use bullet points to organize information")
        
        return recommendations
    
    def _format_endpoints(self, endpoints: List[Dict]) -> str:
        """Format API endpoints for documentation"""
        if not endpoints:
            return "No endpoints documented"
        
        formatted = []
        for endpoint in endpoints:
            method = endpoint.get('method', 'GET')
            path = endpoint.get('path', '/')
            description = endpoint.get('description', 'No description')
            formatted.append(f"**{method} {path}**\n{description}\n")
        
        return '\n'.join(formatted)
    
    def _format_features(self, features: List[str]) -> str:
        """Format features list for documentation"""
        if not features:
            return "No features documented"
        
        return '\n'.join([f"- {feature}" for feature in features])
    
    def _initialize_templates(self):
        """Initialize documentation templates"""
        self.templates = {
            'default': """# {title}

{description}

## Overview

{overview}

## Details

{details}
""",
            'api': """# {title}

{description}

## Base URL
{base_url}

## Endpoints

{endpoints_section}
""",
            'readme': """# {project_name}

{description}

## Installation

{installation}

## Usage

{usage}

## Contributing

{contributing}
""",
            'user_guide': """# {title}

{description}

## Getting Started

{getting_started}

## Features

{features}
""",
            'tutorial': """# {title}

{description}

## Prerequisites

{prerequisites}

## Steps

{steps}
"""
        }
    
    def _update_generation_stats(self, success: bool, generation_time_ms: float):
        """Update generation statistics"""
        self.stats['documents_generated'] += 1
        
        if success:
            self.stats['successful_generations'] += 1
        
        # Update average generation time
        current_avg = self.stats['avg_generation_time_ms']
        count = self.stats['documents_generated']
        self.stats['avg_generation_time_ms'] = ((current_avg * (count - 1)) + generation_time_ms) / count
    
    def get_status(self) -> Dict[str, Any]:
        """Get documentation agent status"""
        return {
            'agent_id': self.agent_id,
            'agent_type': self.agent_type,
            'status': self.status,
            'capabilities': self.capabilities,
            'documents_stored': len(self.documentation_items),
            'supported_languages': self.config['supported_languages'],
            'templates_available': len(self.templates),
            'statistics': self.stats.copy(),
            'config': self.config.copy()
        }
    
    async def health_check(self) -> Dict[str, Any]:
        """Perform health check"""
        return {
            'agent_id': self.agent_id,
            'status': self.status,
            'is_healthy': self.status == "active",
            'documents_generated': self.stats['documents_generated'],
            'success_rate': (
                self.stats['successful_generations'] / max(1, self.stats['documents_generated'])
            ),
            'avg_generation_time_ms': self.stats['avg_generation_time_ms'],
            'templates_loaded': len(self.templates),
            'last_check': datetime.utcnow().isoformat()
        }
