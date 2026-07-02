"""AI Research Agent.

Usage:
    python main.py
"""

import os
import sys
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# ---------------------------------------------------------------------------
# Groq configuration (uses OpenAI-compatible API)
# ---------------------------------------------------------------------------
load_dotenv()

# Initialize LangSmith tracing at CLI startup
from tracing import setup_tracing
setup_tracing()

from llm_config import get_provider, get_model_name

# Validate API keys
provider = get_provider()
provider_key_map = {
    "groq": "GROQ_API_KEY",
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
}
api_key_env = provider_key_map.get(provider.value, "GROQ_API_KEY")
if not os.getenv(api_key_env):
    print(f"❌ {api_key_env} not found in .env file for provider '{provider.value}'.")
    sys.exit(1)

if not os.getenv("SERPER_API_KEY"):
    print("⚠️  SERPER_API_KEY not found. Web search will fall back to DuckDuckGo.\n")


# ---------------------------------------------------------------------------
# Report Generation
# ---------------------------------------------------------------------------

PDF_STYLES = """@page {
    size: A4;
    margin: 2.5cm 2cm;
    @top-center { content: "AI Research Report"; font-size: 9pt; color: #666; }
    @bottom-center { content: "Page " counter(page) " of " counter(pages); font-size: 9pt; color: #888; }
}
body { font-family: 'Helvetica Neue', Arial, sans-serif; font-size: 11pt; line-height: 1.6; color: #222; }
h1 { font-size: 24pt; color: #1a1a2e; border-bottom: 3px solid #1a1a2e; padding-bottom: 8px; margin-top: 30px; page-break-before: always; }
h1:first-of-type { page-break-before: avoid; }
h2 { font-size: 18pt; color: #16213e; border-bottom: 1px solid #ccc; padding-bottom: 5px; margin-top: 25px; }
h3 { font-size: 14pt; color: #0f3460; margin-top: 20px; }
p { margin: 10px 0; text-align: justify; }
strong { color: #1a1a2e; }
ul, ol { margin: 10px 0; padding-left: 25px; }
blockquote { border-left: 4px solid #0f3460; margin: 15px 0; padding: 10px 15px; background: #f8f9fa; font-style: italic; color: #444; }
.title-page { text-align: center; padding-top: 120px; page-break-after: always; }
.title-page h1 { font-size: 28pt; color: #1a1a2e; border: none; margin-bottom: 10px; }
.title-page .subtitle { font-size: 16pt; color: #555; margin-top: 10px; }
.title-page .meta { margin-top: 50px; font-size: 11pt; color: #777; }
.title-page .brand { margin-top: 80px; font-size: 10pt; color: #aaa; text-transform: uppercase; letter-spacing: 3px; }
.quality-review { background: #f0f7ff; border: 1px solid #0f3460; border-radius: 5px; padding: 15px 20px; margin: 20px 0; }
.references { font-size: 10pt; }
.references li { margin: 8px 0; word-break: break-all; }
"""


def _sanitize(topic: str) -> str:
    safe = "".join(c if c.isalnum() or c in " -_" else "_" for c in topic)
    return "-".join(safe.split())[:60].lower()


def save_report(topic: str, content: str, output_dir: str = "output") -> tuple[str, str]:
    """Save the report as both Markdown and PDF. Returns (md_path, pdf_path)."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"{_sanitize(topic)}_{stamp}"

    # Markdown
    md_path = output_path / f"{base}.md"
    md_path.write_text(content, encoding="utf-8")
    print(f"  ✅ Markdown: {md_path}")

    # PDF
    pdf_path = output_path / f"{base}.pdf"
    try:
        import markdown
        from weasyprint import HTML

        html_body = markdown.markdown(content, extensions=["fenced_code", "tables"])
        today = datetime.now().strftime("%B %d, %Y")
        html_doc = f"""<!DOCTYPE html><html><head><meta charset="UTF-8"><title>{topic}</title>
        <style>{PDF_STYLES}</style></head><body>
        <div class="title-page"><h1>{topic}</h1><div class="subtitle">Comprehensive Research Report</div>
        <div class="meta"><p>Generated: {today}</p></div><div class="brand">AI Research Agent</div></div>
        {html_body}</body></html>"""
        HTML(string=html_doc).write_pdf(str(pdf_path))
        print(f"  ✅ PDF:      {pdf_path}")
    except Exception as e:
        print(f"  ⚠️  PDF skipped ({e})")
        pdf_path = None

    return str(md_path), str(pdf_path) if pdf_path else None


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def run():
    """Run the research agent interactively via the unified orchestrator."""
    print("\n🤖 AI Research Agent" + "\n" + "=" * 40)
    topic = input("Enter a research topic: ").strip()

    if not topic:
        print("No topic entered. Exiting.")
        return

    from orchestrator import run_pipeline

    print(f"\n🚀 Starting research on: '{topic}'\n")

    result = run_pipeline(topic)

    if result.get("error"):
        print(f"❌ {result['error']}")
        return

    report = result["report"]
    critique_iterations = result["critique_iterations"]
    verification = result.get("verification_result", {})
    verify_passed = verification.get("passed", True) if verification else True
    verify_findings = verification.get("findings", []) if verification else []

    # Save outputs
    print(f"\n{'=' * 60}")
    print("  📄 Saving Report\n")
    md_path, pdf_path = save_report(topic, report)

    # Summary
    print(f"\n{'=' * 60}")
    print("  ✅ Research Complete!")
    print(f"{'=' * 60}")
    print(f"  Topic:        {topic}")
    print(f"  Engine:       LangGraph")
    print(f"  Markdown:     {md_path}")
    if pdf_path:
        print(f"  PDF:          {pdf_path}")
    print(f"  Critic iters: {critique_iterations}")
    print(f"  Verification: {'✅ Passed' if verify_passed else '⚠️ ' + str(len(verify_findings)) + ' issues found'}")
    if result.get("duration_seconds"):
        print(f"  Duration:     {result['duration_seconds']:.1f}s")
    print()

    # Show a preview
    print("\n--- REPORT PREVIEW ---\n")
    print(report[:500] + "\n..." if len(report) > 500 else report)


if __name__ == "__main__":
    run()
