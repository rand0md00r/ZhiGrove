#!/usr/bin/env python3
"""
Markdown to PDF converter
"""

import sys
import os
import subprocess
from pathlib import Path

def install_requirements():
    """Install required packages"""
    try:
        import markdown
        import weasyprint
        print("Required packages already installed")
        return True
    except ImportError:
        print("Installing required packages...")
        try:
            subprocess.check_call([sys.executable, "-m", "pip", "install", "markdown", "weasyprint"])
            return True
        except subprocess.CalledProcessError:
            print("Failed to install packages. Please install manually:")
            print("pip install markdown weasyprint")
            return False

def md_to_pdf(md_file, pdf_file=None):
    """Convert Markdown file to PDF"""
    if not install_requirements():
        return False
    
    try:
        import markdown
        from weasyprint import HTML, CSS
        from weasyprint.text.fonts import FontConfiguration
    except ImportError:
        print("Failed to import required modules")
        return False
    
    # Read markdown file
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['tables', 'fenced_code', 'codehilite'])
    
    # Add CSS styling
    css_content = """
    body {
        font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
        line-height: 1.6;
        margin: 2cm;
        font-size: 12pt;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #333;
        margin-top: 1.5em;
        margin-bottom: 0.5em;
    }
    h1 { font-size: 24pt; }
    h2 { font-size: 20pt; }
    h3 { font-size: 16pt; }
    h4 { font-size: 14pt; }
    code {
        background-color: #f5f5f5;
        padding: 2px 4px;
        border-radius: 3px;
        font-family: 'Courier New', monospace;
    }
    pre {
        background-color: #f5f5f5;
        padding: 1em;
        border-radius: 5px;
        overflow-x: auto;
    }
    img {
        max-width: 100%;
        height: auto;
    }
    table {
        border-collapse: collapse;
        width: 100%;
        margin: 1em 0;
    }
    th, td {
        border: 1px solid #ddd;
        padding: 8px;
        text-align: left;
    }
    th {
        background-color: #f2f2f2;
    }
    blockquote {
        border-left: 4px solid #ddd;
        margin: 1em 0;
        padding-left: 1em;
        color: #666;
    }
    """
    
    # Create full HTML document
    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="utf-8">
        <title>{Path(md_file).stem}</title>
    </head>
    <body>
        {html_content}
    </body>
    </html>
    """
    
    # Generate PDF
    font_config = FontConfiguration()
    html_doc = HTML(string=full_html)
    css_doc = CSS(string=css_content, font_config=font_config)
    
    if pdf_file is None:
        pdf_file = Path(md_file).with_suffix('.pdf')
    
    html_doc.write_pdf(pdf_file, stylesheets=[css_doc], font_config=font_config)
    print(f"PDF generated successfully: {pdf_file}")
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python md_to_pdf.py <markdown_file> [output_pdf_file]")
        sys.exit(1)
    
    md_file = sys.argv[1]
    pdf_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    if not os.path.exists(md_file):
        print(f"Error: File {md_file} not found")
        sys.exit(1)
    
    success = md_to_pdf(md_file, pdf_file)
    sys.exit(0 if success else 1)
