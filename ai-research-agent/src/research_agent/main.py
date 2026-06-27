"""CLI entry point for the AI Research Agent.

Usage:
    research-agent "Your research topic here"
    research-agent --topic "Your topic" --output ./reports --no-pdf
"""

import argparse
import sys
from datetime import datetime

from research_agent.crew import ResearchCrew
from research_agent.report_generator import generate_markdown_report, generate_pdf_report


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="AI Research Agent - Researches any topic and generates structured reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s "Quantum Computing in 2026"
  %(prog)s -t "Rust vs Go for backend development" -o ./reports
  %(prog)s -t "Climate change technologies" --no-pdf --verbose
        """,
    )
    parser.add_argument(
        "topic",
        type=str,
        nargs="?",
        help="The research topic to investigate",
    )
    parser.add_argument(
        "-t",
        "--topic",
        type=str,
        dest="topic_flag",
        help="The research topic (alternative to positional argument)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default="output",
        help="Output directory for reports (default: ./output)",
    )
    parser.add_argument(
        "--no-pdf",
        action="store_true",
        help="Skip PDF generation (save only Markdown)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose agent logging",
    )
    parser.add_argument(
        "--version",
        action="store_true",
        help="Show version information",
    )
    return parser.parse_args()


def main() -> None:
    """Main entry point for the research agent CLI."""
    args = parse_args()

    if args.version:
        from research_agent import __version__

        print(f"AI Research Agent v{__version__}")
        return

    # Determine topic from positional arg or flag
    topic = args.topic or args.topic_flag
    if not topic:
        # Interactive mode if no topic provided
        topic = input("🔬 Enter a research topic: ").strip()
        if not topic:
            print("❌ No topic provided. Exiting.")
            sys.exit(1)

    # Validate OpenAI API key
    import os

    if not os.getenv("OPENAI_API_KEY"):
        print(
            "❌ OPENAI_API_KEY not found in environment.\n"
            "   Create a .env file with:\n"
            "   OPENAI_API_KEY=sk-your-key-here\n"
            "   Or export it: export OPENAI_API_KEY=sk-your-key-here"
        )
        sys.exit(1)

    # Run the research
    crew = ResearchCrew(verbose=args.verbose)
    report_content = crew.run(topic)

    if not report_content or not report_content.strip():
        print("\n❌ No report was generated. Check the logs above for errors.")
        sys.exit(1)

    # Generate output files
    print(f"\n{'='*60}")
    print(f"  📄 Generating Report Files")
    print(f"{'='*60}\n")

    output_dir = args.output

    # Save Markdown
    md_path = generate_markdown_report(topic, report_content, output_dir=output_dir)

    # Save PDF (skip if --no-pdf)
    pdf_path = ""
    if not args.no_pdf:
        pdf_path = generate_pdf_report(topic, report_content, output_dir=output_dir)

    # Summary
    print(f"\n{'='*60}")
    print(f"  ✅ Research Complete!")
    print(f"{'='*60}")
    print(f"  Topic: {topic}")
    print(f"  Markdown: {md_path}")
    if pdf_path:
        print(f"  PDF:      {pdf_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
