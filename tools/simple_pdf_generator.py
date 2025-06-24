"""
Simple PDF Generator for Agent Workflow Diagram

This script creates a clean PDF version of the workflow diagram.
"""

import re
from pathlib import Path

def create_simple_html():
    """Create a simple HTML version with embedded CSS and Mermaid"""
    
    # Read the markdown file
    with open('AGENT_WORKFLOW_DIAGRAM.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract Mermaid diagrams
    mermaid_pattern = r'```mermaid\n(.*?)\n```'
    mermaid_blocks = re.findall(mermaid_pattern, content, re.DOTALL)
    
    # Replace mermaid blocks with div placeholders
    content = re.sub(mermaid_pattern, '<div class="mermaid">MERMAID_PLACEHOLDER</div>', content, flags=re.DOTALL)
    
    # Basic markdown to HTML conversion
    content = re.sub(r'^# (.*)', r'<h1>\\1</h1>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*)', r'<h2>\\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.*)', r'<h3>\\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^\*\*(.*?)\*\*', r'<strong>\\1</strong>', content, flags=re.MULTILINE)
    content = re.sub(r'`([^`]+)`', r'<code>\\1</code>', content)
    content = content.replace('\\n\\n', '</p><p>')
    content = content.replace('\\n', '<br>')
    content = f'<p>{content}</p>'
    
    # Replace mermaid placeholders with actual diagrams
    for i, diagram in enumerate(mermaid_blocks):
        content = content.replace('<div class="mermaid">MERMAID_PLACEHOLDER</div>', 
                                f'<div class="mermaid">{diagram}</div>', 1)
    
    html_template = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Agent Workflow Diagram</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }}
        .mermaid {{
            text-align: center;
            margin: 20px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }}
        pre {{
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }}
        code {{
            background: #f1f1f1;
            padding: 2px 4px;
            border-radius: 3px;
        }}
        strong {{
            color: #2c3e50;
        }}
    </style>
</head>
<body>
    <script>
        mermaid.initialize({{ 
            startOnLoad: true,
            theme: 'default',
            themeVariables: {{
                primaryColor: '#3498db',
                primaryTextColor: '#2c3e50',
                primaryBorderColor: '#2980b9',
                lineColor: '#34495e',
                background: '#ffffff'
            }}
        }});
    </script>
    {content}
</body>
</html>'''
    
    with open('agent_workflow.html', 'w', encoding='utf-8') as f:
        f.write(html_template)
    
    print("✅ HTML file created: agent_workflow.html")
    return 'agent_workflow.html'

def convert_to_pdf():
    """Convert HTML to PDF using Playwright"""
    try:
        from playwright.sync_api import sync_playwright
        
        print("🔄 Converting HTML to PDF with Playwright...")
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Load the HTML file
            html_path = Path('agent_workflow.html').absolute()
            page.goto(f"file:///{html_path}")
            
            # Wait for Mermaid to render
            print("⏳ Waiting for Mermaid diagrams to render...")
            page.wait_for_timeout(5000)
            
            # Generate PDF
            print("📄 Generating PDF...")
            page.pdf(
                path="agent_workflow_diagram.pdf",
                format="A3",  # Larger format for better diagram visibility
                print_background=True,
                margin={{
                    "top": "20mm",
                    "bottom": "20mm", 
                    "left": "15mm",
                    "right": "15mm"
                }}
            )
            
            browser.close()
            print("✅ PDF created: agent_workflow_diagram.pdf")
            return True
            
    except ImportError:
        print("❌ Playwright not available")
        return False
    except Exception as e:
        print(f"❌ Error creating PDF: {e}")
        return False

def main():
    """Main execution"""
    print("🚀 Creating Agent Workflow PDF...")
    
    # Create HTML
    html_file = create_simple_html()
    
    # Convert to PDF
    if convert_to_pdf():
        print("✅ Success! Open agent_workflow_diagram.pdf to see the Mermaid diagrams")
        print("📁 Files created:")
        print("   - agent_workflow.html (can be opened in browser)")
        print("   - agent_workflow_diagram.pdf (final PDF with diagrams)")
    else:
        print("⚠️ PDF creation failed, but you can open agent_workflow.html in your browser")

if __name__ == "__main__":
    main()
