# Research and Documentation Agent Integration Guide

## Overview

PyGent Factory includes sophisticated research and documentation systems that seamlessly integrate with the frontend Documentation tab. This system enables automated research-to-documentation workflows, where research outputs can be automatically converted into structured documentation and published through the documentation interface.

## System Architecture

### Research Agent System
The research agent system consists of multiple specialized agents working together:

#### Academic Research Agent (`src/ai/multi_agent/agents/academic_research_agent.py`)
- **Primary Functions**: Conducts literature reviews, searches academic databases (arXiv, Semantic Scholar, Google Scholar)
- **Real Data Integration**: Pulls actual academic papers and performs quality scoring
- **Output Format**: Structured research reports with APA-style citations
- **Key Capabilities**:
  - Multi-database academic search
  - Literature review automation
  - Citation analysis and research gap identification
  - Primary source discovery

#### Historical Research Agent (`src/ai/multi_agent/agents/research_agent.py`)
- **Primary Functions**: Specialized historical research with global perspectives
- **Key Capabilities**:
  - Cross-cultural synthesis
  - Primary source analysis
  - Global perspective integration
  - Cultural insight generation

#### Research Agent Adapter (`src/agents/research_agent_adapter.py`)
- **Primary Functions**: Unified interface for different research types
- **Integration Point**: Handles routing between academic and historical research
- **Output Standardization**: Converts research outputs to consistent format

### Documentation Agent System

#### Documentation Agent (`src/ai/multi_agent/agents/documentation_agent.py`)
- **Primary Functions**: Automated documentation generation and quality assessment
- **Key Capabilities**:
  - Multi-format documentation generation (API docs, user guides, tutorials, README)
  - Documentation quality assessment and improvement recommendations
  - Template-based generation with customizable formats
  - Multi-language support (EN, ES, FR, DE, ZH, JA)

#### Documentation Types Supported
```python
class DocumentationType(Enum):
    TECHNICAL = "technical"
    API = "api"
    USER_GUIDE = "user_guide"
    README = "readme"
    TUTORIAL = "tutorial"
```

### Frontend Documentation Integration

#### Documentation API (`src/api/routes/documentation.py`)
The backend provides comprehensive API endpoints for documentation management:

- **`GET /api/documentation/`** - Documentation index with navigation structure
- **`GET /api/documentation/files`** - List all documentation files with filtering
- **`GET /api/documentation/file/{path}`** - Get specific documentation file content
- **`GET /api/documentation/search`** - Search documentation content
- **`GET /api/documentation/categories`** - Get documentation categories
- **`GET /api/documentation/stats`** - Get documentation statistics

#### Frontend Documentation Page (`src/ui/src/pages/DocumentationPage.tsx`)
- **Integration Method**: Iframe-based documentation display
- **Navigation**: Seamless integration with main UI navigation
- **Real-time Updates**: Supports live documentation updates

## Research-to-Documentation Workflow

### Automated Research-Analysis Pipeline

The system includes a complete automated pipeline (`RESEARCH_ANALYSIS_WORKFLOW_COMPLETE.md`) that:

1. **Research Phase**: 
   - Uses Research Agent to pull actual papers from arXiv and Semantic Scholar
   - Performs quality scoring and relevance filtering
   - No mock data - all research is real

2. **Analysis Phase**: 
   - Feeds research data to Reasoning Agent with deepseek-r1:8b
   - Applies Tree of Thought reasoning for multi-perspective evaluation
   - Generates evidence-based conclusions and recommendations

3. **Documentation Phase**: 
   - Creates academic report with proper APA citations
   - Formats output with professional typography
   - Provides multiple export options (Markdown, HTML, JSON)

### Workflow API Integration

#### Research-Analysis Workflow Endpoints (`src/api/routes/workflows.py`)
```http
POST /api/v1/workflows/research-analysis
GET /api/v1/workflows/research-analysis/{workflow_id}/status
GET /api/v1/workflows/research-analysis/{workflow_id}/result
GET /api/v1/workflows/research-analysis/{workflow_id}/export/{format}
```

#### Workflow Orchestrator (`src/workflows/research_analysis_orchestrator.py`)
- **Function**: Coordinates complete research-to-analysis pipeline
- **Integration**: Uses Agent Factory for research and reasoning agents
- **Progress Tracking**: Real-time status updates via WebSocket
- **Output Management**: Structured results ready for documentation publishing

## Integration Points and Workflows

### 1. Direct Research Output Publishing

Research agents can generate documentation-ready outputs:

```python
# Example: Academic Research Agent generates literature review
research_result = await academic_research_agent.conduct_literature_review(
    topic="quantum computing",
    research_question="What are recent advances in quantum error correction?"
)

# Result includes documentation-ready sections:
# - Executive summary
# - Literature summary with citations
# - Key findings and recommendations
# - Professional formatting
```

### 2. Documentation Agent Processing

Research outputs can be processed by the Documentation Agent:

```python
# Convert research output to documentation
doc_result = await documentation_agent.generate_documentation(
    content_data={
        'title': 'Quantum Error Correction Literature Review',
        'content': research_result['response'],
        'doc_type': 'technical',
        'tags': ['quantum', 'research', 'literature-review']
    },
    doc_type="technical"
)

# Generates structured documentation with:
# - Quality assessment
# - Professional formatting
# - Metadata and tags
```

### 3. Frontend Documentation Display

The frontend Documentation tab can display both:

#### Static Documentation Files
- Traditional markdown files in `/docs` directory
- Automatically indexed and categorized
- Full-text search capabilities

#### Dynamic Research Documentation
- Live research outputs from agent workflows
- Real-time updates as research completes
- Interactive navigation and export options

## Configuration and Customization

### Documentation Agent Configuration
```python
config = {
    'supported_languages': ['en', 'es', 'fr', 'de', 'zh', 'ja'],
    'default_language': 'en',
    'max_content_length': 50000,
    'quality_threshold': 0.7,
    'auto_update_enabled': True,
    'style_guide': 'standard'
}
```

### Research Agent Configuration
```python
config = {
    'max_papers_per_search': 50,
    'min_citation_threshold': 5,
    'preferred_publication_years': 10,
    'search_timeout_seconds': 30,
    'quality_filters': {
        'min_citations': 1,
        'peer_reviewed_only': True,
        'exclude_preprints': False
    }
}
```

## Advanced Features

### 1. Real-time Research Documentation

The system supports live documentation generation:
- Research starts → Progress tracking begins
- Intermediate results → Live preview updates
- Research completes → Final documentation published
- User access → Immediate availability in Documentation tab

### 2. Citation and Reference Management

Research agents automatically generate:
- APA-style citations for all sources
- Clickable links to original papers
- Reference lists with DOI links
- Citation metadata for academic use

### 3. Multi-format Export

Documentation can be exported in multiple formats:
- **Markdown**: For version control and editing
- **HTML**: For web publishing
- **JSON**: For programmatic access
- **PDF**: For academic submission (future enhancement)

## Implementation Status

### ✅ Completed Components
- Research Agent System (Academic + Historical)
- Documentation Agent System
- Research-Analysis Workflow Pipeline
- Backend Documentation API
- Frontend Documentation Page
- Navigation Integration

### 🔄 Integration Enhancements Available

#### Enhanced Documentation Generation
```python
# Future enhancement: Direct research-to-docs publishing
async def publish_research_as_documentation(research_id: str):
    """Automatically publish research results as documentation"""
    research_result = await get_research_result(research_id)
    
    # Generate documentation
    doc_result = await documentation_agent.generate_documentation(
        content_data=research_result,
        doc_type="technical"
    )
    
    # Save to docs directory
    save_path = f"docs/research/{research_id}.md"
    with open(save_path, 'w') as f:
        f.write(doc_result['content'])
    
    # Update documentation index
    await refresh_documentation_index()
    
    return doc_result
```

#### Live Research Documentation Dashboard
- Real-time research progress in Documentation tab
- Interactive research parameter adjustment
- Live preview of research outputs
- One-click publication to documentation

#### Research Documentation Templates
```python
research_templates = {
    'literature_review': """# {title}

## Executive Summary
{executive_summary}

## Methodology
{methodology}

## Key Findings
{key_findings}

## Citations
{citations}
""",
    'research_report': """# {title}

## Research Question
{research_question}

## Background
{background}

## Analysis
{analysis}

## Conclusions
{conclusions}

## References
{references}
"""
}
```

## Usage Examples

### Example 1: Academic Research to Documentation

```python
# 1. Start research workflow
workflow_id = await start_research_analysis_workflow({
    "query": "Recent advances in transformer architectures",
    "max_papers": 20,
    "analysis_depth": 3
})

# 2. Monitor progress
status = await get_workflow_status(workflow_id)

# 3. Get results
results = await get_workflow_results(workflow_id)

# 4. Auto-publish to documentation
doc_id = await publish_research_as_documentation(workflow_id)

# 5. Documentation now available at:
# /docs/research/transformer-architectures-{workflow_id}.md
```

### Example 2: Custom Documentation Generation

```python
# Generate custom documentation from research data
doc_result = await documentation_agent.generate_documentation(
    content_data={
        'title': 'AI Research Methodology Guide',
        'description': 'Guide for conducting AI research using PyGent Factory',
        'content': research_methodology_content,
        'tags': ['ai', 'research', 'methodology']
    },
    doc_type="user_guide"
)

# Result includes quality assessment and recommendations
print(f"Quality Score: {doc_result['quality_score']}")
print(f"Recommendations: {doc_result.get('recommendations', [])}")
```

## Conclusion

The PyGent Factory research and documentation system provides a comprehensive, integrated workflow for conducting research and automatically generating high-quality documentation. The system combines:

- **Real Data Integration**: No mock data - all research uses actual academic sources
- **AI-Powered Analysis**: Advanced reasoning with deepseek-r1 and Tree of Thought
- **Professional Output**: Academic-quality formatting with proper citations
- **Seamless Integration**: Direct connection between research and documentation systems
- **User-Friendly Interface**: Accessible through the main UI Documentation tab

This integration enables researchers and developers to conduct thorough research and automatically publish professional documentation, streamlining the knowledge creation and sharing process within the PyGent Factory ecosystem.
