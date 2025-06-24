"""
PDF Generator for Agent Workflow Diagram

This script converts the AGENT_WORKFLOW_DIAGRAM.md file to PDF
with properly rendered Mermaid diagrams.
"""

import subprocess
import sys
from pathlib import Path

def install_requirements():
    """Install required packages"""
    packages = [
        "markdown",
        "weasyprint", 
        "pymdown-extensions",
        "markdown-mermaid",
        "playwright"
    ]
    
    for package in packages:
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", package])
            print(f"✅ Installed {package}")
        except subprocess.CalledProcessError:
            print(f"❌ Failed to install {package}")

def convert_to_html():
    """Convert Markdown to HTML with Mermaid support"""
    
    html_template = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Agent Workflow Diagram</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid/dist/mermaid.min.js"></script>
    <style>
        body {
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }
        h1, h2, h3 {
            color: #2c3e50;
            border-bottom: 2px solid #3498db;
            padding-bottom: 10px;
        }
        .mermaid {
            text-align: center;
            margin: 20px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
        }
        pre {
            background: #f4f4f4;
            padding: 15px;
            border-radius: 5px;
            overflow-x: auto;
        }
        code {
            background: #f1f1f1;
            padding: 2px 4px;
            border-radius: 3px;
        }
        blockquote {
            border-left: 4px solid #3498db;
            margin: 0;
            padding-left: 20px;
            font-style: italic;
        }
        .workflow-section {
            margin: 30px 0;
            padding: 20px;
            background: #ffffff;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        }
    </style>
</head>
<body>
    <script>
        mermaid.initialize({ 
            startOnLoad: true,
            theme: 'default',
            themeVariables: {
                primaryColor: '#3498db',
                primaryTextColor: '#2c3e50',
                primaryBorderColor: '#2980b9',
                lineColor: '#34495e',
                background: '#ffffff'
            }
        });
    </script>
    {content}
</body>
</html>
"""
    
    # Read the markdown file
    with open('AGENT_WORKFLOW_DIAGRAM.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Convert markdown code blocks to proper mermaid divs
    import re
    
    # Replace ```mermaid blocks with <div class="mermaid">
    content = re.sub(
        r'```mermaid\n(.*?)\n```', 
        r'<div class="mermaid">\1</div>', 
        content, 
        flags=re.DOTALL
    )
    
    # Convert basic markdown to HTML
    content = re.sub(r'^# (.*)', r'<h1>\1</h1>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*)', r'<h2>\1</h2>', content, flags=re.MULTILINE)
    content = re.sub(r'^### (.*)', r'<h3>\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^\*\*(.*?)\*\*', r'<strong>\1</strong>', content, flags=re.MULTILINE)
    content = re.sub(r'`([^`]+)`', r'<code>\1</code>', content)
    content = re.sub(r'\n\n', '<br><br>', content)
    content = re.sub(r'\n', '<br>', content)
    
    # Wrap sections
    content = re.sub(r'(<h2>.*?</h2>)', r'<div class="workflow-section">\1', content)
    content = re.sub(r'(<h3>.*?</h3>)', r'</div><div class="workflow-section">\1', content)
    content += '</div>'
    
    html_content = html_template.format(content=content)
    
    with open('agent_workflow.html', 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print("✅ HTML file created: agent_workflow.html")
    return 'agent_workflow.html'

def convert_html_to_pdf(html_file):
    """Convert HTML to PDF using Playwright"""
    try:
        from playwright.sync_api import sync_playwright
        
        with sync_playwright() as p:
            browser = p.chromium.launch()
            page = browser.new_page()
            
            # Load the HTML file
            html_path = Path(html_file).absolute()
            page.goto(f"file://{html_path}")
            
            # Wait for Mermaid to render
            page.wait_for_timeout(3000)
            
            # Generate PDF
            page.pdf(
                path="agent_workflow_diagram.pdf",
                format="A4",
                print_background=True,
                margin={
                    "top": "20mm",
                    "bottom": "20mm", 
                    "left": "15mm",
                    "right": "15mm"
                }
            )
            
            browser.close()
            print("✅ PDF created: agent_workflow_diagram.pdf")
            
    except ImportError:
        print("❌ Playwright not available, trying alternative method...")
        return convert_with_weasyprint(html_file)

def convert_with_weasyprint(html_file):
    """Fallback: Convert using WeasyPrint (no Mermaid support)"""
    try:
        import weasyprint
        
        weasyprint.HTML(filename=html_file).write_pdf('agent_workflow_simple.pdf')
        print("✅ Simple PDF created: agent_workflow_simple.pdf (no Mermaid diagrams)")
        
    except ImportError:
        print("❌ WeasyPrint not available. Please install: pip install weasyprint")

def main():
    """Main execution function"""
    print("🚀 Starting PDF generation...")
    
    # Install required packages
    print("📦 Installing requirements...")
    install_requirements()
      # Install Playwright browsers
    try:
        subprocess.check_call([sys.executable, "-m", "playwright", "install", "chromium"])
        print("✅ Playwright chromium installed")
    except Exception as e:
        print(f"⚠️ Playwright installation failed: {e}, will try fallback method")
    
    # Convert to HTML
    print("🔄 Converting Markdown to HTML...")
    html_file = convert_to_html()
    
    # Convert to PDF
    print("📄 Converting HTML to PDF...")
    convert_html_to_pdf(html_file)
    
    print("✅ PDF generation complete!")
    print("📁 Files created:")
    print("   - agent_workflow.html (intermediate)")
    print("   - agent_workflow_diagram.pdf (final PDF)")

if __name__ == "__main__":
    main()
