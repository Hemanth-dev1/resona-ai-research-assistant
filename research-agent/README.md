# 🔬 Resona — Autonomous Multi-Agent AI Research Assistant

**Self-correcting critic loop · RAG memory with ChromaDB · LangSmith observability · Pydantic-validated outputs · Unified LLM config (Groq / OpenAI / Anthropic)**

[![Live Demo →](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge)](https://resona-ai-research-assistant.onrender.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateMachine-blue)](https://langchain-ai.github.io/langgraph/)
[![LangChain](https://img.shields.io/badge/LangChain-LCEL-blue)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-Vector--Store-green)](https://chromadb.com)
[![Groq](https://img.shields.io/badge/Groq-LLaMA4-orange)](https://groq.com)
[![LangSmith](https://img.shields.io/badge/LangSmith-Tracing-purple)](https://smith.langchain.com)
[![RAGAS](https://img.shields.io/badge/RAGAS-Quality--Eval-red)](https://docs.ragas.io)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validated-920)](https://docs.pydantic.dev)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed)](https://docker.com)
[![Render](https://img.shields.io/badge/Render-Deploy-46e3b7)](https://render.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Resona is an autonomous AI research assistant that researches **any topic**, evaluates its own output through a **self-correcting critic loop**, and generates a professional structured report — saved as both **Markdown** and **PDF**. It runs on **LangGraph** (state machine) with **langchain_groq.ChatGroq** for direct LLM access, **ChromaDB** RAG memory, **RAGAS** quality evaluation, and full **LangSmith** observability.

**[Live demo →](https://resona-ai-research-assistant.onrender.com)** — enter a topic and watch the pipeline collaborate in real-time via SSE streaming.

---

## 🏛️ Architecture

```
┌──────────────────────────────────────────────────────────────────────┐
│                        User enters a topic                           │
└──────────────────────────┬───────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      LangGraph StateMachine (graph.py)                │
│                                                                      │
│  planner ──→ analysis_writer ──→ critic ──→ verifier ──→ END         │
│                    ↑                    │  ↑           │              │
│                    │                    ▼  │           │              │
│                    └───────── revise ◄────┘           │              │
│                                              (strict mode)            │
│                                          verifier ──→ revise ◄───────┘│
│                                                                      │
│  • planner: Decompose topic into sub-questions (fast model)          │
│  • analysis_writer: Analyze research + compose report (capable)      │
│  • critic: Score report 0-10 on 5 quality dimensions                 │
│  • revise: Regenerate report from critic feedback                    │
│  • verifier: Fact-check claims against research material              │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     RAGAS Quality Evaluation                         │
│                                                                      │
│  • Faithfulness — claims supported by context?                       │
│  • Answer Relevancy — report addresses the topic?                    │
│  • Context Recall — all relevant context retrieved?                  │
│                                                                      │
│  Scores stored to output/ragas_scores.json · Exposed via /api/ragas  │
└──────────────────────────────────────────────────────────────────────┘
                           │
                           ▼
               ┌──────────────────────┐
               │  📄 Markdown Report  │
               │  📄 Styled PDF       │
               │  🗄️  ChromaDB Memory │
               └──────────────────────┘
```

### Pipeline Flow

```
User → Planner (sub-questions) → Parallel Research (web search × N)
    → LangGraph StateMachine (analysis → critic ↔ revise → verifier)
    → RAGAS Evaluation → Report (MD + PDF) → ChromaDB Memory
```

---

## ✨ Features

### 1. 🧠 LangGraph State Machine

Always uses LangGraph — no mode selection needed:

| Component | Description |
|-----------|-------------|
| **LangGraph** | StateGraph with conditional critic/verifier edges — planner → analysis_writer → critic ↔ revise → verifier → END |
| **langchain_groq.ChatGroq** | Direct Groq API access — no LiteLLM/CrewAI translation layer |
| **LangChain LCEL** | Individual analysis/writing chains used as LangGraph node bodies |

> Previously supported CrewAI and standalone LangChain modes. Consolidated on LangGraph to eliminate a class of provider-compatibility bugs (LiteLLM cache_breakpoint, Pydantic validation errors). The archived CrewAI code (`archive/agents.py`, `archive/tasks.py`) remains in the repo as reference.

### 2. 🔄 Self-Correcting Critic Loop

Every report is automatically scored by an LLM judge across **5 quality dimensions**:

| Dimension | Score 0-10 | What It Measures |
|-----------|-----------|------------------|
| **Factual Accuracy** | ✅/❌ | Are claims supported? Any hallucinations? |
| **Structure** | ✅/❌ | Does it follow the 8-section order? |
| **Clarity** | ✅/❌ | Is the writing clear and professional? |
| **Completeness** | ✅/❌ | Are all sections adequately developed? |
| **Citation Quality** | ✅/❌ | Are sources properly cited? |

- If overall score < **7/10**, specific feedback is sent back to the Writer agent
- The report is regenerated with improvements (up to **3 iterations**)
- Configurable threshold via `QUALITY_THRESHOLD` or `RESONA_CRITIC_THRESHOLD`
- Each iteration logged: `📝 Critic iteration 2/3` → `📊 Score: 6/10` → `🔄 Revising report...`

### 3. 📊 RAGAS Quality Evaluation

Three LLM-as-judge metrics computed after every research run:

| Metric | Description | Target Range |
|--------|-------------|--------------|
| **Faithfulness** | Are claims in the answer supported by retrieved context? | 0.7 – 1.0 |
| **Answer Relevancy** | How relevant is the answer to the question? | 0.7 – 1.0 |
| **Context Recall** | Was all relevant context retrieved from ChromaDB? | 0.6 – 1.0 |

- Scores written to `output/ragas_scores.json` on each run
- Average scores exposed via `GET /api/ragas`
- ChromaDB is saved **before** RAGAS evaluation so context_recall is meaningful

### 4. 🗄️ RAG Memory with ChromaDB

- All reports are chunked (500-char with 50-char overlap) and stored in **ChromaDB** using `all-MiniLM-L6-v2` embeddings
- Subsequent research on related topics automatically retrieves prior findings as additional context
- Cross-session persistence — memory survives server restarts
- FAISS index for in-session document similarity search

### 5. 🔁 Retry Logic with Exponential Backoff

Every LLM call is wrapped with **tenacity** retry logic:

```
⚠️  Retry 1/3 after 1.0s: Temporary API failure
⚠️  Retry 2/3 after 2.0s: Temporary API failure
✅ Success on attempt 3
```

- Configurable via `RESONA_MAX_RETRIES` (default: 3), `RESONA_RETRY_MIN_WAIT` (1s), `RESONA_RETRY_MAX_WAIT` (10s)
- Graceful fallback — if all retries exhaust, returns a clean error dict instead of crashing

### 6. 🛡️ Pydantic-Validated Output Models

Every agent output is parsed into a typed Pydantic model:

```
ResearchReport(
    topic="Quantum computing 2026",
    executive_summary="...",
    sources=["https://...", "https://..."],  # min 2 validated
    key_insights=["...", "..."],
    word_count=1850,
)
```

- Catches hallucinated structure before content reaches the user
- Validation errors are caught gracefully — the pipeline continues with the raw text
- 15 typed models: `ResearchFinding`, `ResearchBrief`, `ThemeAnalysis`, `Analysis`, `ResearchReport`, `CritiqueResult`, `PipelineResult`, etc.

### 7. 📡 LangSmith Observability

Full tracing via OpenTelemetry for every research run:

- **LangGraph Mode:** Every LangGraph node, critic loop iteration, and LLM invocation
- **LLM Calls:** Token counts, latency, costs for every API call (via langchain-groq)
- **Pipeline Router:** Top-level `run_analysis()` call with inputs/outputs

#### LangSmith Screenshot

> **📸 Add your LangSmith trace screenshot here**
>
> After running a research topic with `LANGSMITH_API_KEY` configured:
> 1. Go to [smith.langchain.com](https://smith.langchain.com) → `resona-ai-research-assistant` project
> 2. Click on a completed trace run
> 3. Take a screenshot showing the agent steps, LLM calls, and token usage
> 4. Save it as `docs/langsmith_trace.png`
> 5. Replace the placeholder below:

```
![LangSmith Trace](docs/langsmith_trace.png)
```

*Example trace data: Agent step hierarchy, LLM token counts, latency breakdown, and cost estimates appear automatically for every research run.*

### 8. 🔌 Unified LLM Config (Groq / OpenAI / Anthropic)

Switch providers with a single env var:

```bash
# .env
LLM_PROVIDER=groq          # groq (default), openai, or anthropic
LLM_MODEL=llama-3.1-8b-instant
```

| Provider | Env Var | Default Model |
|----------|---------|---------------|
| **Groq** | `GROQ_API_KEY` | `llama-3.1-8b-instant` |
| **OpenAI** | `OPENAI_API_KEY` | `gpt-4o` |
| **Anthropic** | `ANTHROPIC_API_KEY` | `claude-3-5-sonnet-20240620` |

### 9. 🐳 Docker + Render Deployment

One-click deploy to Render:

```bash
# Build locally
docker build -t resona .

# Or deploy via render.yaml — auto-detected by Render
```

- **Dockerfile:** Python 3.12-slim with WeasyPrint system deps (Cairo, Pango, GDK-Pixbuf)
- **docker-compose.yml:** Named volumes for ChromaDB persistence, `.env` mount, resource limits
- **render.yaml:** Pre-configured for Render — sets env vars, health checks, 8080 port
- **Health endpoint:** `GET /health` → `{"status": "ok", "version": "1.1.0"}`

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- An API key ([Groq free](https://console.groq.com/keys), [OpenAI](https://platform.openai.com/api-keys), or [Anthropic](https://console.anthropic.com/))

### Local Setup

```bash
# Clone and enter the project
cd research-agent

# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
echo 'GROQ_API_KEY=gsk_your_key_here' >> .env
echo 'LLM_PROVIDER=groq' >> .env
```

### Usage — CLI

```bash
# Interactive mode
python main.py

# Or pipe a topic directly
echo "Quantum computing breakthroughs 2026" | python main.py
```

### Usage — Web UI

```bash
# Start the FastAPI server
python server.py

# Open in browser
open http://localhost:8080
```

Enter a topic in the chat UI and watch the agents stream progress via SSE — research phase, analysis phase, writing phase, critic loop iterations, and RAGAS scores all appear in real-time.

### Usage — API

```bash
# Run research via API (SSE streaming)
curl -X POST http://localhost:8080/api/run \
  -H "Content-Type: application/json" \
  -d '{"topic": "Quantum computing 2026"}'

# List reports
curl http://localhost:8080/api/reports

# Get RAGAS scores
curl http://localhost:8080/api/ragas

# Get stats
curl http://localhost:8080/api/stats

# Health check
curl http://localhost:8080/health
```

---

## 📁 Project Structure

```
research-agent/
├── .env                    # API keys & config
├── .gitignore
├── requirements.txt        # Python dependencies
├── README.md               # This file
├── render.yaml             # Render deployment config
├── Dockerfile              # Docker image
├── docker-compose.yml      # Local Docker compose
├── .dockerignore
│
├── main.py                 # CLI entry point + report generation
├── server.py               # FastAPI web server with SSE streaming
├── router.py               # Pipeline router (always LangGraph)
│
├── archive/                # Archived CrewAI code (reference only)
│   ├── agents.py
│   ├── tasks.py
│   └── tools.py
│
├── critic.py               # Self-correcting critic loop
├── ragas_eval.py           # RAGAS quality evaluation
├── retry_utils.py          # Tenacity retry logic
├── llm_config.py           # Unified LLM config (Groq/OpenAI/Anthropic)
├── tracing.py              # LangSmith OpenTelemetry setup
│
├── schemas/
│   ├── models.py           # 15 Pydantic-typed data models
│   └── parser.py           # Parse agent output into Pydantic models
│
├── chain/                  # LangChain LCEL pipeline
│   ├── __init__.py
│   ├── prompts.py
│   └── chain.py
│
├── memory/                 # RAG memory layer
│   ├── __init__.py
│   ├── chroma_store.py     # Persistent ChromaDB (cross-session)
│   └── faiss_index.py      # In-session FAISS index
│
├── static/
│   └── index.html          # Chat UI
│
├── output/                 # Generated reports (.md + .pdf)
├── chroma_db/              # ChromaDB persistent storage
├── tests/                  # Pytest test suite
└── docs/                   # Screenshots & documentation assets
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes* | — | Groq API key (if provider=groq) |
| `OPENAI_API_KEY` | Yes* | — | OpenAI API key (if provider=openai) |
| `ANTHROPIC_API_KEY` | Yes* | — | Anthropic API key (if provider=anthropic) |
| `LLM_PROVIDER` | No | `groq` | Provider: `groq`, `openai`, or `anthropic` |
| `LLM_MODEL` | No | per-provider default | Model name override |
| `SERPER_API_KEY` | No | — | Google search API ([free tier](https://serper.dev)). Falls back to DuckDuckGo. |
| `LANGSMITH_API_KEY` | No | — | LangSmith tracing ([smith.langchain.com](https://smith.langchain.com)) |
| `LANGSMITH_PROJECT` | No | `resona-ai-research-assistant` | LangSmith project name |
| `QUALITY_THRESHOLD` | No | `7` | Critic loop pass threshold (0-10) |
| `RESONA_CRITIC_THRESHOLD` | No | `7` | Alias for QUALITY_THRESHOLD |
| `RESONA_MAX_CRITIC_ITERATIONS` | No | `3` | Max critic loop iterations |
| `LLM_MODEL_FAST` | No | per-provider fast | Fast model for planner & research (e.g., `llama-3.1-8b-instant`) |
| `LLM_MODEL_CAPABLE` | No | per-provider capable | Capable model for analyst, writer, critic (e.g., `llama-3.3-70b-versatile`) |
| `RESEARCH_MAX_CONCURRENT` | No | `1` | Parallel research workers (1 avoids Groq rate limits, increase for paid tiers) |
| `MEMORY_CONTEXT` | No | — | Additional context from ChromaDB memory |
| `RESONA_MAX_RETRIES` | No | `3` | Max retry attempts |
| `RESONA_RETRY_MIN_WAIT` | No | `1.0` | Min retry backoff (seconds) |
| `RESONA_RETRY_MAX_WAIT` | No | `10.0` | Max retry backoff (seconds) |

*\* At least one API key must be set, corresponding to `LLM_PROVIDER`.*

### Report Sections

| # | Section | Description |
|---|---------|-------------|
| 1 | **Title Page** | Topic, date, AI Research Agent branding |
| 2 | **Executive Summary** | Concise overview of findings |
| 3 | **Introduction** | Context and background |
| 4 | **Detailed Analysis** | 3-5 subsections with research findings |
| 5 | **Key Insights** | Bulleted takeaways |
| 6 | **Challenges & Considerations** | Limitations, controversies, debates |
| 7 | **Future Outlook** | Trends and predictions |
| 8 | **Sources & References** | Numbered list with URLs |

---

## 📊 RAGAS Evaluation Scores

After running a real research topic, RAGAS scores appear in `output/ragas_scores.json`. See the `/api/ragas` endpoint for aggregate averages:

```json
// Example output/ragas_scores.json
[
  {
    "topic": "Quantum computing breakthroughs 2026",
    "faithfulness": 0.92,
    "answer_relevancy": 0.88,
    "context_recall": 0.75,
    "overall": 0.85,
    "timestamp": "2026-06-16T15:10:55"
  }
]
```

| Metric | Score | Interpretation |
|--------|-------|----------------|
| Faithfulness | 0.92 | ✅ Strong — claims well-supported by context |
| Answer Relevancy | 0.88 | ✅ Highly relevant to the query |
| Context Recall | 0.75 | ⚡ Good — most relevant context retrieved |
| **Overall** | **0.85** | **High quality research run** |

> *Scores are computed via LLM-as-judge after each run. Run a topic and check `output/ragas_scores.json` to see your real numbers.*

---

## 🌐 Deploy to Render

Resona is pre-configured for one-click deployment to [Render](https://render.com):

```yaml
# render.yaml (in repo root)
services:
  - type: web
    name: resona-ai-research-assistant
    runtime: docker
    plan: free
    healthCheckPath: /health
    envVars:
      - key: GROQ_API_KEY
        sync: false          # Set in Render dashboard
      - key: LANGSMITH_API_KEY
        sync: false
      - key: LANGSMITH_PROJECT
        value: resona-ai-research-assistant
      - key: LLM_PROVIDER
        value: groq
      - key: QUALITY_THRESHOLD
        value: "7"
      - key: SERPER_API_KEY
        sync: false
```

### Deployment Steps

1. **Push to GitHub:**
   ```bash
   git add .
   git commit -m "feat: complete Resona research assistant"
   git push origin main
   ```

2. **Connect to Render:**
   - Go to [render.com](https://render.com) → **New** → **Web Service**
   - Connect your GitHub repo
   - Render auto-detects `render.yaml` — the config is pre-loaded
   - In the **Environment** tab, add these secrets:
     - `GROQ_API_KEY` — your Groq key
     - `LANGSMITH_API_KEY` — your LangSmith key
     - `SERPER_API_KEY` — your Serper key
   - Click **Deploy** (~8 min first build)

3. **Your live URL:**
   ```
   https://resona-ai-research-assistant.onrender.com
   ```

---

## 🧪 Tests

```bash
# Run the test suite
PYTHONPATH=. pytest tests/ -v
```

Tests verify:
- Markdown file creation with correct content
- PDF file creation with valid PDF header
- Nested output directory creation
- Both file formats returned from `save_report()`
- LangGraph state machine routing (planner → analysis → critic → verifier)
- Fallback web research for CLI mode

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| **[LangGraph](https://langchain-ai.github.io/langgraph/)** | State machine orchestration |
| **[LangChain](https://langchain.com)** | LCEL chains used as LangGraph node bodies |
| **[Groq](https://groq.com)** | LLM inference (Llama 3.1, Mixtral, etc.) |
| **[OpenAI](https://openai.com)** | LLM inference (GPT-4o, GPT-4o-mini) |
| **[Anthropic](https://anthropic.com)** | LLM inference (Claude 3.5 Sonnet) |
| **[ChromaDB](https://chromadb.com)** | Persistent cross-session research memory |
| **[FAISS](https://faiss.ai)** | In-session document similarity search |
| **[LangSmith](https://smith.langchain.com)** | LLM observability & OpenTelemetry tracing |
| **[RAGAS](https://docs.ragas.io)** | LLM-as-judge quality evaluation |
| **[Pydantic](https://docs.pydantic.dev)** | Typed data validation for agent outputs |
| **[Tenacity](https://tenacity.readthedocs.io)** | Retry logic with exponential backoff |
| **[Sentence-Transformers](https://sbert.net)** | Embedding model (`all-MiniLM-L6-v2`) |
| **[DuckDuckGo Search](https://pypi.org/project/duckduckgo-search/)** | Free web search (no API key needed) |
| **[WeasyPrint](https://weasyprint.org)** | HTML/CSS to PDF rendering |
| **[FastAPI](https://fastapi.tiangolo.com)** | Web server with SSE streaming |
| **[Uvicorn](https://www.uvicorn.org)** | ASGI server |
| **[Render](https://render.com)** | Cloud deployment (Docker) |

---

## 📄 License

MIT

---

## 🙌 Why Resona?

Resona was built to demonstrate **production-grade AI engineering**:

- **Autonomous quality control** — the critic loop means the system checks its own work, not just generates text
- **Measurable quality** — RAGAS scores quantify how well the system performs, with real numbers you can track over time
- **Full observability** — every LLM call, token, and latency metric is traced to LangSmith
- **Typed data pipeline** — Pydantic models catch structural issues before they reach the user
- **Resilient by design** — retry logic with exponential backoff means transient API failures don't crash the pipeline
- **Deployable in minutes** — one-click Render deployment with Docker, health checks, and persistent storage

**[Live demo →](https://resona-ai-research-assistant.onrender.com)**
