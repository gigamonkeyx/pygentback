# Documentation Generation Capabilities

## PDF Workflow Generator

The PyGent Factory system includes advanced documentation generation capabilities for creating visual workflow diagrams.

### Features

- **Mermaid Diagram Rendering**: Converts Mermaid markdown diagrams to interactive visualizations
- **PDF Generation**: Creates print-ready PDFs from markdown documentation
- **Multi-Format Output**: HTML, PDF, and browser-printable formats
- **Professional Layout**: Optimized typography and spacing for technical documentation

### Usage

```python
# Generate workflow diagram PDF
python create_printable_workflow.py

# Generate any markdown file with Mermaid diagrams
python simple_pdf_generator.py
```

### API

```python
def create_print_ready_html():
    """
    Create an HTML version optimized for printing to PDF
    
    Features:
    - Extracts Mermaid diagrams from markdown
    - Converts markdown to semantic HTML
    - Applies print-optimized CSS
    - Includes interactive Mermaid rendering
    
    Returns:
        str: Path to generated HTML file
    """
```

### Generated Files

- `agent_workflow_printable.html` - Print-optimized version
- `agent_workflow.html` - Basic interactive version
- `agent_workflow_diagram.pdf` - Final PDF output

### Supported Diagram Types

- **Flowcharts**: Agent creation and lifecycle flows
- **State Diagrams**: Agent status transitions
- **Sequence Diagrams**: Message processing pipelines
- **Graph Diagrams**: System integration and feedback loops

This capability is essential for:
- System documentation
- Architecture visualization
- Process flow documentation
- Technical presentations
- Compliance documentation
