"""
Documentation Generation Module

Modular components for generating intelligent documentation.
"""

import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime

from .core import NLPProcessor, ConfidenceCalculator, ProcessingResult
from .models import DocumentationType, GeneratedDoc, DocumentationSection

logger = logging.getLogger(__name__)


class TemplateEngine:
    """
    Template engine for documentation generation.
    """
    
    def __init__(self):
        self.templates = self._initialize_templates()
        self.section_templates = self._initialize_section_templates()
    
    def _initialize_templates(self) -> Dict[DocumentationType, Dict[str, Any]]:
        """Initialize documentation templates"""
        return {
            DocumentationType.API_DOCS: {
                'title_template': "{api_name} API Documentation",
                'sections': ['overview', 'authentication', 'endpoints', 'examples', 'errors'],
                'style': 'technical'
            },
            DocumentationType.USER_GUIDE: {
                'title_template': "{product_name} User Guide",
                'sections': ['introduction', 'getting_started', 'features', 'tutorials', 'faq'],
                'style': 'user_friendly'
            },
            DocumentationType.TUTORIAL: {
                'title_template': "{topic} Tutorial",
                'sections': ['prerequisites', 'step_by_step', 'examples', 'troubleshooting'],
                'style': 'instructional'
            },
            DocumentationType.REFERENCE: {
                'title_template': "{component_name} Reference",
                'sections': ['overview', 'configuration', 'methods', 'properties', 'examples'],
                'style': 'reference'
            },
            DocumentationType.TROUBLESHOOTING: {
                'title_template': "{product_name} Troubleshooting Guide",
                'sections': ['common_issues', 'diagnostics', 'solutions', 'contact_support'],
                'style': 'problem_solving'
            },
            DocumentationType.README: {
                'title_template': "{project_name}",
                'sections': ['description', 'installation', 'usage', 'contributing', 'license'],
                'style': 'project_overview'
            }
        }
    
    def _initialize_section_templates(self) -> Dict[str, str]:
        """Initialize section templates"""
        return {
            'overview': """## Overview

{description}

### Key Features
{features}

### Requirements
{requirements}
""",
            'getting_started': """## Getting Started

### Installation
{installation_steps}

### Quick Start
{quick_start_steps}

### First Example
{first_example}
""",
            'api_endpoint': """### {method} {endpoint}

{description}

**Parameters:**
{parameters}

**Response:**
{response_format}

**Example:**
```{language}
{example_code}
```
""",
            'tutorial_step': """### Step {step_number}: {step_title}

{step_description}

{step_code}

{step_notes}
""",
            'troubleshooting_item': """### {issue_title}

**Problem:** {problem_description}

**Cause:** {cause_description}

**Solution:**
{solution_steps}
""",
            'configuration_option': """#### {option_name}

- **Type:** {option_type}
- **Default:** {default_value}
- **Description:** {description}

{example_usage}
"""
        }
    
    def render_template(self, doc_type: DocumentationType, context: Dict[str, Any]) -> str:
        """Render documentation template"""
        template = self.templates.get(doc_type)
        if not template:
            return f"# {context.get('title', 'Documentation')}\n\nContent not available."
        
        title = template['title_template'].format(**context)
        return f"# {title}\n\n"
    
    def render_section(self, section_type: str, context: Dict[str, Any]) -> str:
        """Render section template"""
        template = self.section_templates.get(section_type, "## {title}\n\n{content}")
        
        try:
            return template.format(**context)
        except KeyError as e:
            logger.warning(f"Missing context key for section template: {e}")
            return f"## {context.get('title', section_type.title())}\n\n{context.get('content', 'Content not available.')}"


class DocumentationGenerator(NLPProcessor):
    """
    Intelligent documentation generator.
    """
    
    def __init__(self):
        super().__init__("DocumentationGenerator")
        self.template_engine = TemplateEngine()
        
        # Content generators for different types
        self.content_generators = {
            DocumentationType.API_DOCS: self._generate_api_docs,
            DocumentationType.USER_GUIDE: self._generate_user_guide,
            DocumentationType.TUTORIAL: self._generate_tutorial,
            DocumentationType.REFERENCE: self._generate_reference,
            DocumentationType.TROUBLESHOOTING: self._generate_troubleshooting,
            DocumentationType.README: self._generate_readme
        }
    
    async def process(self, text: str, context: Optional[Dict[str, Any]] = None) -> ProcessingResult:
        """Process documentation generation request"""
        start_time = time.time()
        
        try:
            # Extract documentation type and context
            doc_type, doc_context = self._extract_doc_context(text, context)
            
            # Generate documentation
            generated_doc = await self.generate_documentation(doc_type, doc_context)
            
            processing_time = (time.time() - start_time) * 1000
            
            # Update statistics
            self.update_stats(True, generated_doc.generation_confidence, processing_time)
            
            return ProcessingResult(
                success=True,
                confidence=generated_doc.generation_confidence,
                processing_time_ms=processing_time,
                metadata={'documentation': generated_doc.to_dict()}
            )
            
        except Exception as e:
            processing_time = (time.time() - start_time) * 1000
            self.update_stats(False, 0.0, processing_time)
            
            logger.error(f"Documentation generation failed: {e}")
            
            return ProcessingResult(
                success=False,
                confidence=0.0,
                processing_time_ms=processing_time,
                metadata={'error': str(e)}
            )
    
    def _extract_doc_context(self, text: str, context: Optional[Dict[str, Any]]) -> tuple:
        """Extract documentation type and context from input"""
        # Default values
        doc_type = DocumentationType.USER_GUIDE
        doc_context = context or {}
        
        # Try to detect documentation type from text
        text_lower = text.lower()
        
        if any(word in text_lower for word in ['api', 'endpoint', 'rest', 'graphql']):
            doc_type = DocumentationType.API_DOCS
        elif any(word in text_lower for word in ['tutorial', 'step by step', 'how to']):
            doc_type = DocumentationType.TUTORIAL
        elif any(word in text_lower for word in ['reference', 'documentation', 'manual']):
            doc_type = DocumentationType.REFERENCE
        elif any(word in text_lower for word in ['troubleshoot', 'problem', 'error', 'issue']):
            doc_type = DocumentationType.TROUBLESHOOTING
        elif any(word in text_lower for word in ['readme', 'project', 'repository']):
            doc_type = DocumentationType.README
        
        # Override with explicit context
        if context and 'doc_type' in context:
            try:
                doc_type = DocumentationType(context['doc_type'])
            except ValueError:
                pass
        
        # Add text to context
        doc_context['source_text'] = text
        
        return doc_type, doc_context
    
    async def generate_documentation(self, doc_type: DocumentationType, 
                                   context: Dict[str, Any]) -> GeneratedDoc:
        """Generate documentation of specified type"""
        # Get content generator
        generator = self.content_generators.get(doc_type, self._generate_generic_docs)
        
        # Generate content
        title, sections = await generator(context)
        
        # Calculate confidence
        confidence = self._calculate_generation_confidence(doc_type, sections, context)
        
        return GeneratedDoc(
            title=title,
            doc_type=doc_type,
            sections=sections,
            template_used=doc_type.value,
            generation_confidence=confidence,
            source_data=context,
            generation_context={'doc_type': doc_type.value}
        )
    
    async def _generate_api_docs(self, context: Dict[str, Any]) -> tuple:
        """Generate API documentation"""
        api_name = context.get('api_name', 'API')
        title = f"{api_name} Documentation"
        
        sections = []
        
        # Overview section
        overview_content = f"""
This documentation describes the {api_name} REST API.

The API provides endpoints for managing resources and performing operations.
All requests must be authenticated and responses are in JSON format.
"""
        sections.append(DocumentationSection(
            title="Overview",
            content=overview_content.strip(),
            section_type="content",
            level=2
        ))
        
        # Authentication section
        auth_content = """
## Authentication

All API requests require authentication using an API key.

Include your API key in the request header:
```
Authorization: Bearer YOUR_API_KEY
```
"""
        sections.append(DocumentationSection(
            title="Authentication",
            content=auth_content.strip(),
            section_type="content",
            level=2
        ))
        
        # Endpoints section
        endpoints = context.get('endpoints', [])
        if endpoints:
            for endpoint in endpoints:
                endpoint_content = self._generate_endpoint_docs(endpoint)
                sections.append(DocumentationSection(
                    title=f"{endpoint.get('method', 'GET')} {endpoint.get('path', '/')}",
                    content=endpoint_content,
                    section_type="code",
                    level=3
                ))
        else:
            sections.append(DocumentationSection(
                title="Endpoints",
                content="API endpoints will be documented here.",
                section_type="content",
                level=2
            ))
        
        return title, sections
    
    async def _generate_user_guide(self, context: Dict[str, Any]) -> tuple:
        """Generate user guide"""
        product_name = context.get('product_name', 'Product')
        title = f"{product_name} User Guide"
        
        sections = []
        
        # Introduction
        intro_content = f"""
Welcome to {product_name}! This guide will help you get started and make the most of the features available.

{product_name} is designed to be intuitive and powerful, providing you with the tools you need to accomplish your goals efficiently.
"""
        sections.append(DocumentationSection(
            title="Introduction",
            content=intro_content.strip(),
            section_type="content",
            level=2
        ))
        
        # Getting Started
        getting_started_content = """
## Getting Started

### Installation
1. Download the latest version
2. Follow the installation wizard
3. Launch the application

### First Steps
1. Create your account
2. Complete the setup wizard
3. Explore the main interface
"""
        sections.append(DocumentationSection(
            title="Getting Started",
            content=getting_started_content.strip(),
            section_type="content",
            level=2
        ))
        
        # Features
        features = context.get('features', [])
        if features:
            for feature in features:
                feature_content = f"""
### {feature.get('name', 'Feature')}

{feature.get('description', 'Feature description not available.')}

**How to use:**
{feature.get('usage', 'Usage instructions not available.')}
"""
                sections.append(DocumentationSection(
                    title=feature.get('name', 'Feature'),
                    content=feature_content.strip(),
                    section_type="content",
                    level=3
                ))
        
        return title, sections
    
    async def _generate_tutorial(self, context: Dict[str, Any]) -> tuple:
        """Generate tutorial"""
        topic = context.get('topic', 'Tutorial')
        title = f"{topic} Tutorial"
        
        sections = []
        
        # Prerequisites
        prereq_content = """
## Prerequisites

Before starting this tutorial, make sure you have:
- Basic understanding of the concepts
- Required software installed
- Access to the necessary resources
"""
        sections.append(DocumentationSection(
            title="Prerequisites",
            content=prereq_content.strip(),
            section_type="content",
            level=2
        ))
        
        # Steps
        steps = context.get('steps', [])
        if steps:
            for i, step in enumerate(steps, 1):
                step_content = f"""
### Step {i}: {step.get('title', f'Step {i}')}

{step.get('description', 'Step description not available.')}

{step.get('code', '')}

{step.get('notes', '')}
"""
                sections.append(DocumentationSection(
                    title=f"Step {i}: {step.get('title', f'Step {i}')}",
                    content=step_content.strip(),
                    section_type="content",
                    level=3
                ))
        else:
            sections.append(DocumentationSection(
                title="Tutorial Steps",
                content="Tutorial steps will be added here.",
                section_type="content",
                level=2
            ))
        
        return title, sections
    
    async def _generate_reference(self, context: Dict[str, Any]) -> tuple:
        """Generate reference documentation"""
        component_name = context.get('component_name', 'Component')
        title = f"{component_name} Reference"
        
        sections = []
        
        # Overview
        overview_content = f"""
## {component_name} Reference

This reference provides detailed information about {component_name}, including all available methods, properties, and configuration options.
"""
        sections.append(DocumentationSection(
            title="Overview",
            content=overview_content.strip(),
            section_type="content",
            level=2
        ))
        
        # Methods
        methods = context.get('methods', [])
        if methods:
            for method in methods:
                method_content = self._generate_method_docs(method)
                sections.append(DocumentationSection(
                    title=method.get('name', 'Method'),
                    content=method_content,
                    section_type="code",
                    level=3
                ))
        
        return title, sections
    
    async def _generate_troubleshooting(self, context: Dict[str, Any]) -> tuple:
        """Generate troubleshooting guide"""
        product_name = context.get('product_name', 'Product')
        title = f"{product_name} Troubleshooting Guide"
        
        sections = []
        
        # Common Issues
        issues = context.get('issues', [])
        if issues:
            for issue in issues:
                issue_content = f"""
### {issue.get('title', 'Issue')}

**Problem:** {issue.get('problem', 'Problem description not available.')}

**Cause:** {issue.get('cause', 'Cause not identified.')}

**Solution:**
{issue.get('solution', 'Solution not available.')}
"""
                sections.append(DocumentationSection(
                    title=issue.get('title', 'Issue'),
                    content=issue_content.strip(),
                    section_type="content",
                    level=3
                ))
        else:
            sections.append(DocumentationSection(
                title="Common Issues",
                content="Common issues and solutions will be documented here.",
                section_type="content",
                level=2
            ))
        
        return title, sections
    
    async def _generate_readme(self, context: Dict[str, Any]) -> tuple:
        """Generate README documentation"""
        project_name = context.get('project_name', 'Project')
        title = project_name
        
        sections = []
        
        # Description
        description = context.get('description', f'{project_name} is an innovative solution.')
        sections.append(DocumentationSection(
            title="Description",
            content=description,
            section_type="content",
            level=2
        ))
        
        # Installation
        installation_content = """
## Installation

```bash
# Clone the repository
git clone https://github.com/user/project.git

# Install dependencies
npm install

# Run the application
npm start
```
"""
        sections.append(DocumentationSection(
            title="Installation",
            content=installation_content.strip(),
            section_type="code",
            level=2
        ))
        
        # Usage
        usage_content = """
## Usage

Basic usage example:

```javascript
const project = require('project');

// Initialize
const instance = new project.Main();

// Use the API
instance.doSomething();
```
"""
        sections.append(DocumentationSection(
            title="Usage",
            content=usage_content.strip(),
            section_type="code",
            level=2
        ))
        
        return title, sections
    
    async def _generate_generic_docs(self, context: Dict[str, Any]) -> tuple:
        """Generate generic documentation"""
        title = context.get('title', 'Documentation')
        
        sections = [
            DocumentationSection(
                title="Overview",
                content="This documentation provides information about the system.",
                section_type="content",
                level=2
            )
        ]
        
        return title, sections
    
    def _generate_endpoint_docs(self, endpoint: Dict[str, Any]) -> str:
        """Generate documentation for an API endpoint"""
        method = endpoint.get('method', 'GET')
        path = endpoint.get('path', '/')
        description = endpoint.get('description', 'Endpoint description not available.')
        
        content = f"""
**{method} {path}**

{description}

**Parameters:**
"""
        
        parameters = endpoint.get('parameters', [])
        if parameters:
            for param in parameters:
                content += f"- `{param.get('name', 'param')}` ({param.get('type', 'string')}): {param.get('description', 'No description')}\n"
        else:
            content += "No parameters required.\n"
        
        content += "\n**Response:**\n"
        response = endpoint.get('response', {})
        if response:
            content += f"```json\n{response}\n```\n"
        else:
            content += "Response format not documented.\n"
        
        return content.strip()
    
    def _generate_method_docs(self, method: Dict[str, Any]) -> str:
        """Generate documentation for a method"""
        name = method.get('name', 'method')
        description = method.get('description', 'Method description not available.')
        
        content = f"""
#### {name}

{description}

**Syntax:**
```
{method.get('syntax', f'{name}()')}
```

**Parameters:**
"""
        
        parameters = method.get('parameters', [])
        if parameters:
            for param in parameters:
                content += f"- `{param.get('name', 'param')}` ({param.get('type', 'any')}): {param.get('description', 'No description')}\n"
        else:
            content += "No parameters.\n"
        
        content += f"\n**Returns:** {method.get('returns', 'void')}\n"
        
        example = method.get('example')
        if example:
            content += f"\n**Example:**\n```\n{example}\n```\n"
        
        return content.strip()
    
    def _calculate_generation_confidence(self, doc_type: DocumentationType, 
                                       sections: List[DocumentationSection],
                                       context: Dict[str, Any]) -> float:
        """Calculate confidence in generated documentation"""
        factors = []
        
        # Content completeness
        expected_sections = self.template_engine.templates.get(doc_type, {}).get('sections', [])
        if expected_sections:
            completeness = len(sections) / len(expected_sections)
            factors.append(min(1.0, completeness))
        else:
            factors.append(0.8)  # Default for unknown templates
        
        # Context richness
        context_richness = min(1.0, len(context) / 5.0)  # Normalize to 0-1
        factors.append(context_richness)
        
        # Content quality (based on section content length)
        avg_content_length = sum(len(section.content) for section in sections) / max(len(sections), 1)
        content_quality = min(1.0, avg_content_length / 200.0)  # Normalize
        factors.append(content_quality)
        
        return ConfidenceCalculator.combined_confidence(factors, [0.4, 0.3, 0.3])
