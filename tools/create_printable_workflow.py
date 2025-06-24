"""
Print-to-PDF Helper for Agent Workflow

This creates a properly formatted version that you can print to PDF from your browser.
"""

import re

def create_print_ready_html():
    """Create an HTML version optimized for printing to PDF"""
    
    # Read the markdown file
    with open('AGENT_WORKFLOW_DIAGRAM.md', 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Extract and convert Mermaid diagrams to text descriptions for PDF
    mermaid_blocks = []
    def extract_mermaid(match):
        diagram = match.group(1)
        mermaid_blocks.append(diagram)
        return f'<div class="mermaid">{diagram}</div>'
    
    content = re.sub(r'```mermaid\n(.*?)\n```', extract_mermaid, content, flags=re.DOTALL)
    
    # Convert markdown to HTML
    content = re.sub(r'^# (.*)', r'<h1>\\1</h1>', content, flags=re.MULTILINE)
    content = re.sub(r'^## (.*)', r'<h2>\\1</h2>', content, flags=re.MULTILINE) 
    content = re.sub(r'^### (.*)', r'<h3>\\1</h3>', content, flags=re.MULTILINE)
    content = re.sub(r'^\*\*(.*?)\*\*', r'<strong>\\1</strong>', content, flags=re.MULTILINE)
    content = re.sub(r'`([^`]+)`', r'<code>\\1</code>', content)
    content = content.replace('\\n\\n', '</p><p>')
    content = content.replace('\\n', '<br>')
    content = f'<div class="content">{content}</div>'

    html = f'''<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>PyGent Factory - Agent Workflow Diagram</title>
    <script src="https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.min.js"></script>
    <style>
        @page {{
            size: A3;
            margin: 15mm;
        }}
        
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.4;
            color: #2c3e50;
            margin: 0;
            padding: 20px;
        }}
        
        .content {{
            max-width: none;
        }}
        
        h1 {{
            color: #2c3e50;
            font-size: 28px;
            text-align: center;
            border-bottom: 3px solid #3498db;
            padding-bottom: 15px;
            margin-bottom: 30px;
            page-break-after: avoid;
        }}
        
        h2 {{
            color: #2980b9;
            font-size: 22px;
            border-bottom: 2px solid #3498db;
            padding-bottom: 8px;
            margin-top: 30px;
            margin-bottom: 20px;
            page-break-after: avoid;
        }}
        
        h3 {{
            color: #34495e;
            font-size: 18px;
            margin-top: 25px;
            margin-bottom: 15px;
            page-break-after: avoid;
        }}
        
        .mermaid {{
            text-align: center;
            margin: 30px 0;
            background: #f8f9fa;
            padding: 20px;
            border-radius: 8px;
            border: 1px solid #e9ecef;
            page-break-inside: avoid;
            transform: scale(0.8);
            transform-origin: center;
        }}
        
        code {{
            background: #f1f1f1;
            padding: 2px 6px;
            border-radius: 3px;
            font-family: 'Courier New', monospace;
            font-size: 14px;
        }}
        
        strong {{
            color: #2c3e50;
            font-weight: 600;
        }}
        
        p {{
            margin: 10px 0;
            text-align: justify;
        }}
        
        .print-instructions {{
            background: #e8f5e8;
            border: 1px solid #4CAF50;
            padding: 15px;
            border-radius: 5px;
            margin: 20px 0;
            font-weight: bold;
            text-align: center;
        }}
        
        @media print {{
            .print-instructions {{
                display: none;
            }}
            
            body {{
                padding: 0;
            }}
            
            .mermaid {{
                transform: scale(0.7);
                page-break-inside: avoid;
            }}
            
            h1, h2, h3 {{
                page-break-after: avoid;
            }}
        }}
    </style>
</head>
<body>
    <div class="print-instructions">
        📄 <strong>To save as PDF:</strong> Press Ctrl+P → Choose "Save as PDF" → Select A3 or A4 landscape → Save
    </div>
    
    <script>
        mermaid.initialize({{
            startOnLoad: true,
            theme: 'default',
            themeVariables: {{
                primaryColor: '#3498db',
                primaryTextColor: '#2c3e50',
                primaryBorderColor: '#2980b9',
                lineColor: '#34495e',
                background: '#ffffff',
                fontSize: '14px'
            }},
            flowchart: {{
                nodeSpacing: 50,
                rankSpacing: 50
            }}
        }});
        
        // Wait for diagrams to render then notify user
        setTimeout(() => {{
            console.log('Mermaid diagrams should be ready for printing');
        }}, 3000);
    </script>
    
    {content}
    
    <div style="text-align: center; margin-top: 40px; color: #7f8c8d; font-style: italic;">
        <p>Generated from PyGent Factory Agent Workflow Analysis - {len(mermaid_blocks)} interactive diagrams</p>
    </div>
</body>
</html>'''

    with open('agent_workflow_printable.html', 'w', encoding='utf-8') as f:
        f.write(html)
    
    print(f"✅ Print-ready HTML created: agent_workflow_printable.html")
    print(f"📊 Contains {len(mermaid_blocks)} Mermaid diagrams")
    print("🖨️  Open in browser and use Ctrl+P to save as PDF")

if __name__ == "__main__":
    create_print_ready_html()
