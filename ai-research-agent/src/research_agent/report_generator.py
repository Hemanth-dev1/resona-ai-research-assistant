"""Report generation module - converts Markdown reports to PDF with professional styling."""

import os
from datetime import datetime
from pathlib import Path
from typing import Optional

import markdown


# Modern, clean CSS for professional-looking PDF reports
PDF_STYLES = """
@page {
    size: A4;
    margin: 2.5cm 2cm;
    @top-center {
        content: "AI Research Report";
        font-size: 9pt;
        color: #666;
    }
    @bottom-center {
        content: "Page " counter(page) " of " counter(pages);
        font-size: 9pt;
        color: #888;
    }
}

body {
    font-family: 'Helvetica Neue', Helvetica, Arial, sans-serif;
    font-size: 11pt;
    line-height: 1.6;
    color: #222;
    max-width: 100%;
}

h1 {
    font-size: 24pt;
    color: #1a1a2e;
    border-bottom: 3px solid #1a1a2e;
    padding-bottom: 8px;
    margin-top: 30px;
    page-break-before: always;
}

h1:first-of-type {
    page-break-before: avoid;
}

h2 {
    font-size: 18pt;
    color: #16213e;
    border-bottom: 1px solid #ccc;
    padding-bottom: 5px;
    margin-top: 25px;
}

h3 {
    font-size: 14pt;
    color: #0f3460;
    margin-top: 20px;
}

h4 {
    font-size: 12pt;
    color: #533483;
    margin-top: 15px;
}

p {
    margin: 10px 0;
    text-align: justify;
}

strong {
    color: #1a1a2e;
}

ul, ol {
    margin: 10px 0;
    padding-left: 25px;
}

li {
    margin: 5px 0;
}

blockquote {
    border-left: 4px solid #0f3460;
    margin: 15px 0;
    padding: 10px 15px;
    background: #f8f9fa;
    font-style: italic;
    color: #444;
}

code {
    font-family: 'Courier New', monospace;
    background: #f4f4f4;
    padding: 2px 5px;
    font-size: 10pt;
    border-radius: 3px;
}

pre {
    background: #f4f4f4;
    padding: 12px 15px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 10pt;
    border: 1px solid #ddd;
}

table {
    border-collapse: collapse;
    width: 100%;
    margin: 15px 0;
}

th, td {
    border: 1px solid #ddd;
    padding: 8px 12px;
    text-align: left;
}

th {
    background: #1a1a2e;
    color: white;
    font-weight: bold;
}

tr:nth-child(even) {
    background: #f8f9fa;
}

hr {
    border: none;
    border-top: 2px solid #eee;
    margin: 25px 0;
}

a {
    color: #0f3460;
    text-decoration: none;
}

a:hover {
    text-decoration: underline;
}

/* Title page styling */
.title-page {
    text-align: center;
    padding-top: 120px;
    page-break-after: always;
}

.title-page h1 {
    font-size: 28pt;
    color: #1a1a2e;
    border: none;
    margin-bottom: 10px;
}

.title-page .subtitle {
    font-size: 16pt;
    color: #555;
    margin-top: 10px;
}

.title-page .meta {
    margin-top: 50px;
    font-size: 11pt;
    color: #777;
}

.title-page .brand {
    margin-top: 80px;
    font-size: 10pt;
    color: #aaa;
    text-transform: uppercase;
    letter-spacing: 3px;
}

/* Quality review box */
.quality-review {
    background: #f0f7ff;
    border: 1px solid #0f3460;
    border-radius: 5px;
    padding: 15px 20px;
    margin: 20px 0;
}

.quality-review strong {
    color: #0f3460;
}

/* Source references */
.references {
    font-size: 10pt;
}

.references li {
    margin: 8px 0;
    word-break: break-all;
}
"""


def generate_markdown_report(
    topic: str,
    content: str,
    output_dir: str = "output",
    filename: Optional[str] = None,
) -> str:
    """Save the report content as a Markdown file.

    Args:
        topic: The research topic.
        content: The full report in Markdown format.
        output_dir: Directory to save the report.
        filename: Optional custom filename.

    Returns:
        Path to the saved Markdown file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        safe_topic = _sanitize_filename(topic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_topic}_{timestamp}.md"

    filepath = output_path / filename
    filepath.write_text(content, encoding="utf-8")
    print(f"  ✅ Markdown report saved: {filepath}")
    return str(filepath)


def generate_pdf_report(
    topic: str,
    content: str,
    output_dir: str = "output",
    filename: Optional[str] = None,
) -> str:
    """Convert the Markdown report to a styled PDF.

    Uses WeasyPrint to render HTML+CSS into a PDF with professional formatting.

    Args:
        topic: The research topic.
        content: The full report in Markdown format.
        output_dir: Directory to save the PDF.
        filename: Optional custom filename.

    Returns:
        Path to the saved PDF file.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    if filename is None:
        safe_topic = _sanitize_filename(topic)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{safe_topic}_{timestamp}.pdf"
    elif not filename.endswith(".pdf"):
        filename += ".pdf"

    filepath = output_path / filename

    # Convert Markdown to HTML
    html_body = markdown.markdown(
        content,
        extensions=["fenced_code", "tables", "sane_lists", "toc"],
    )

    # Create the full HTML document with title page
    today = datetime.now().strftime("%B %d, %Y")
    html_doc = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>{topic} - Research Report</title>
    <style>{PDF_STYLES}</style>
</head>
<body>

<div class="title-page">
    <h1>{topic}</h1>
    <div class="subtitle">Comprehensive Research Report</div>
    <div class="meta">
        <p>Generated: {today}</p>
        <p>Report Type: AI-Powered Research & Analysis</p>
    </div>
    <div class="brand">AI Research Agent</div>
</div>

{html_body}

</body>
</html>"""

    try:
        from weasyprint import HTML

        HTML(string=html_doc).write_pdf(str(filepath))
        print(f"  ✅ PDF report saved: {filepath}")
        return str(filepath)
    except ImportError:
        print("  ⚠️  WeasyPrint not installed. PDF generation skipped.")
        print("     Install with: pip install weasyprint")
        print(f"  📝 Markdown report available at: {output_path / filename.replace('.pdf', '.md')}")
        return ""
    except Exception as e:
        print(f"  ⚠️  PDF generation failed: {e}")
        print(f"  📝 Markdown report available at: {output_path / filename.replace('.pdf', '.md')}")
        return ""


def _sanitize_filename(topic: str) -> str:
    """Convert a topic string to a safe filename."""
    # Keep only alphanumeric and basic punctuation
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in topic)
    # Replace spaces with hyphens and collapse multiple hyphens
    safe = "-".join(safe.split())
    # Limit length
    return safe[:60].lower()
