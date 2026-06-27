# AI Research Agent 🤖📄

An autonomous AI agent that researches any topic, synthesizes findings from multiple sources, and generates a professional structured report — saved as both Markdown and PDF.

Built with **CrewAI** and **OpenAI GPT-4o** to demonstrate multi-agent orchestration, tool use, and real output generation.

## Features

- 🔍 **Autonomous Web Research** — Searches the web for the latest information on any topic
- 🧠 **Multi-Agent Pipeline** — Three specialized AI agents collaborate (Researcher → Writer → Editor)
- 📊 **Structured Reports** — Generates professional reports with executive summary, analysis, insights, and sources
- 📄 **Dual Output** — Saves reports as both Markdown (`.md`) and PDF (`.pdf`) with clean, print-ready styling
- 🚀 **No API Key Required for Search** — Uses DuckDuckGo by default (free). Optionally use Serper for better results.

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                  AI Research Agent                   │
├─────────────────────────────────────────────────────┤
│                                                      │
│  ┌──────────────┐    ┌──────────────┐    ┌────────┐ │
│  │   Senior     │    │  Technical   │    │ Quality│ │
│  │   Research   │───▶│   Content    │───▶│ Editor │ │
│  │   Analyst    │    │   Writer     │    │        │ │
│  └──────────────┘    └──────────────┘    └────────┘ │
│         │                    │                │       │
│         ▼                    ▼                ▼       │
│  🌐 Web Search     ✍️ Markdown Report    ✅ Polish   │
│  📄 Scrape Sites    📑 8 Sections        🔍 Fact-Check │
│  📝 Research Brief  📝 1500-2000 words   📋 Review     │
│                                                      │
└─────────────────────────────────────────────────────┘
        │
        ▼
┌────────────────┐
│  📄 Output     │
├────────────────┤
│  report.md     │
│  report.pdf    │
└────────────────┘
```

### Agent Roles

| Agent | Role | Tools |
|-------|------|-------|
| **Senior Research Analyst** | Searches the web, cross-references sources, extracts key findings | Web Search, ScrapeWebsite |
| **Technical Content Writer** | Synthesizes research into a structured, engaging report | — (uses research output) |
| **Quality Editor** | Reviews for accuracy, clarity, formatting, and completeness | — (uses draft) |

### Report Sections

1. **Title Page** — Topic, date, branding
2. **Executive Summary** — Concise overview of findings
3. **Introduction** — Context and background
4. **Detailed Analysis** — 3-5 subsections covering main aspects
5. **Key Insights** — Bulleted takeaways
6. **Challenges & Considerations** — Controversies and limitations
7. **Future Outlook** — Trends and predictions
8. **Sources & References** — All cited sources

## Quick Start

### Prerequisites

- Python 3.10+
- OpenAI API key ([get one here](https://platform.openai.com/api-keys))

### Installation

```bash
# Clone the repository
git clone <your-repo-url>
cd ai-research-agent

# Create a virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -e .

# Set up your API key
cp .env.example .env
# Edit .env and add your OPENAI_API_KEY
```

### Usage

```bash
# Research a topic
research-agent "Quantum Computing in 2026"

# Or specify topic with flag and custom output directory
research-agent -t "Rust vs Go for backend development" -o ./reports

# Skip PDF generation (Markdown only)
research-agent -t "Climate change technologies" --no-pdf

# Enable verbose logging
research-agent "Artificial General Intelligence" --verbose

# See help
research-agent --help
```

### Interactive Mode

If you run the command without a topic, it will prompt you to enter one:

```bash
research-agent
> 🔬 Enter a research topic: <your topic here>
```

## Environment Variables

| Variable | Required | Description |
|----------|----------|-------------|
| `OPENAI_API_KEY` | **Yes** | Your OpenAI API key |
| `SERPER_API_KEY` | No | Serper API key (for better web search, [free tier available](https://serper.dev)) |
| `OPENAI_MODEL_NAME` | No | Override model (default: gpt-4o) |
| `VERBOSE` | No | Enable verbose logging (true/false) |

## Example Output

Check the `output/` directory for generated reports after running. Each report is timestamped:

```
output/
├── quantum-computing-in-2026_20260115_143022.md
├── quantum-computing-in-2026_20260115_143022.pdf
├── rust-vs-go-backend_20260115_150245.md
└── rust-vs-go-backend_20260115_150245.pdf
```

## Tech Stack

- **[CrewAI](https://crewai.com)** — Multi-agent orchestration framework
- **[OpenAI GPT-4o](https://platform.openai.com)** — LLM for agent reasoning
- **[DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)** — Free web search integration
- **[SerperDev](https://serper.dev)** — Optional professional search API
- **[WeasyPrint](https://weasyprint.org)** — HTML/CSS to PDF rendering
- **[Python-Markdown](https://python-markdown.github.io)** — Markdown to HTML conversion

## Why This Project?

This project demonstrates several skills that are valuable in AI engineering roles:

- **Multi-Agent Orchestration** — Designing and coordinating specialized AI agents
- **Tool-Use / Function Calling** — Integrating search and scraping tools
- **Prompt Engineering** — Crafting effective agent roles, goals, and task descriptions
- **Pipeline Architecture** — Building sequential data processing pipelines
- **Real Output Generation** — Producing professional, usable artifacts (Markdown + PDF)
- **Clean Python** — Modular, well-documented, production-style code

## License

MIT
