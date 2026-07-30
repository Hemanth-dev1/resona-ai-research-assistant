# 🔬 Resona — AI Research Assistant (Chat + Deep Research)

**Multi-agent research pipeline · Chat with document upload · Tavily web search · Self-correcting critic loop · LangGraph orchestration**

[![Live Demo →](https://img.shields.io/badge/Live-Demo-blue?style=for-the-badge)](https://resona-ai-research-assistant.onrender.com)
[![Python](https://img.shields.io/badge/Python-3.12-blue)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-009688)](https://fastapi.tiangolo.com)
[![LangGraph](https://img.shields.io/badge/LangGraph-StateMachine-blue)](https://langchain-ai.github.io/langgraph/)
[![Groq](https://img.shields.io/badge/Groq-LLaMA-orange)](https://groq.com)
[![Google AI](https://img.shields.io/badge/Gemini-Flash-blue)](https://aistudio.google.com)
[![Tavily](https://img.shields.io/badge/Tavily-Search-purple)](https://tavily.com)
[![Docker](https://img.shields.io/badge/Docker-Ready-2496ed)](https://docker.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

Resona is an AI research assistant that works in **two modes**:

- **💬 Chat** — natural conversation with the ability to upload documents (`.pdf`, `.docx`, `.txt`, `.md`) and ask questions about them. Voice input/output supported.
- **🔎 Deep Research** — multi-step autonomous research: planner decomposes your topic into sub-questions → parallel web search → analyst synthesizes → writer generates a structured report → critic scores it → verifier fact-checks claims. Output saved as Markdown + PDF.

**[Live demo →](https://resona-ai-research-assistant.onrender.com)** — enter a topic or just start chatting.

---

## ✨ Features

### 💬 Chat Mode

- **Multi-session management** — each chat is isolated with its own history. Switch between sessions, delete them, or start fresh. Session list persisted in `localStorage` for privacy.
- **Document upload** — upload `.pdf`, `.docx`, `.txt`, or `.md` files and ask questions about their content. Files are chunked and indexed per session.
- **Voice input** — click the mic button to dictate using the Web Speech API.
- **Voice output** — toggle "Read replies aloud" in the + menu to hear responses via SpeechSynthesis.
- **SSE streaming** — responses stream token-by-token with visible "thinking" dots. Errors are shown inline, never silently swallowed.

### 🔎 Deep Research Mode

| Phase | Description |
|-------|-------------|
| **Planner** | Decomposes your topic into 3–5 focused sub-questions (fast model) |
| **Parallel Research** | Each sub-question searched independently via web (Tavily or ddgs) |
| **Analyst** | Synthesizes findings per sub-question with proper source IDs |
| **Writer** | Composes full report (Executive Summary → Detailed Analysis → Key Insights → Sources) |
| **Critic** | Scores report 0–10 on factual accuracy, structure, clarity, completeness, citation quality |
| **Verifier** | Fact-checks each claim against retrieved sources |
| **Report** | Output saved as Markdown + PDF in `output/` |

The critic may request up to 3 revision rounds if the score is below the quality threshold.

### 🔄 Source-ID Integrity

Sources from each sub-question are assigned **globally unique IDs** (`S1`, `S2`, etc.) so citations never collide across parallel research findings — no more `[S1]` pointing at multiple unrelated URLs.

### 🌐 Web Search

| Provider | API Key Required | Reliability |
|----------|-----------------|-------------|
| **Tavily** (primary) | `TAVILY_API_KEY` | ✅ Excellent — built for LLM agent search |
| **DuckDuckGo** (fallback) | None | ⚠️ Free but rate-limited, especially from cloud IPs |

Tavily is tried first when configured. If unset or unavailable, the system falls back to DuckDuckGo automatically.

### 🛡️ Anti-Hallucination

- **Writer prompt** includes explicit "Reinforced Citation Discipline": only cite source IDs that actually appear in the research data. Never invent a source or statistic.
- **Verifier** cross-checks every claim against the source material.
- **Temperature tuning**: writer runs at `0.15` (faithful synthesis, not improvisation), planner at `0.3` (diverse sub-questions).

### 🗄️ RAG Memory with ChromaDB

- Reports are chunked and stored in **ChromaDB** with `all-MiniLM-L6-v2` embeddings.
- Subsequent research on related topics automatically retrieves prior findings as additional context.
- Cross-session persistence survives server restarts.

### 📊 RAGAS Quality Evaluation

Resona evaluates every deep research report using three LLM-as-judge metrics:
- **Faithfulness** — are claims supported by retrieved context?
- **Answer Relevancy** — how relevant is the report to the topic?
- **Context Recall** — was all relevant context retrieved?

Scores are stored in `output/ragas_scores.json` and exposed via `GET /api/ragas`. Each run generates a quality scorecard alongside the report.

### 🔁 Retry Logic with Exponential Backoff

Every LLM call is wrapped with **tenacity** retry:

```
⚠️  Retry 1/3 after 1.0s: Temporary API failure
⚠️  Retry 2/3 after 2.0s: Temporary API failure
✅ Success on attempt 3
```

Configurable via `RESONA_MAX_RETRIES` (default: 3), `RESONA_RETRY_MIN_WAIT` (1s), `RESONA_RETRY_MAX_WAIT` (10s).

### 🔊 Voice Support

- **Input**: Web Speech API (Chrome/Edge) — click the microphone icon in the input bar
- **Output**: Browser SpeechSynthesis — toggle via the + menu ("Read replies aloud")

---

## 🏛️ Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                       User Interface (static/index.html)                 │
│                                                                         │
│   ┌─────────────────┐   ┌────────────────────┐   ┌──────────────────┐  │
│   │ Chat Mode        │   │ Deep Research Mode │   │ + Tools Menu     │  │
│   │ • Multi-session  │   │ • SSE streaming    │   │ • Upload file    │  │
│   │ • Voice in/out   │   │ • Phase progress   │   │ • Deep Research  │  │
│   │ • Doc upload     │   │ • Critic loop      │   │ • Read aloud     │  │
│   └────────┬─────────┘   └────────┬───────────┘   └──────────────────┘  │
│            │                       │                                    │
└────────────┼───────────────────────┼────────────────────────────────────┘
             │                       │
             ▼                       ▼
┌────────────────────────┐  ┌────────────────────────────────────────────┐
│  POST /api/chat/stream │  │       POST /api/run (SSE event stream)     │
│  • session_id          │  │                                            │
│  • message             │  │  planner → parallel research → analyst     │
│  • voice               │  │     → writer → critic ↔ revise → verifier  │
│  • (documents via      │  │                                            │
│     /api/chat/documents)│  └────────────────────────────────────────────┘
└───────────┬────────────┘                │
            │                             ▼
            ▼                     ┌─────────────────┐
┌──────────────────────┐         │ search_provider  │
│  chat/conversation   │         │ • Tavily (try)   │
│  _buffer.py          │         │ • ddgs (fallback)│
│  chat/chat_graph.py  │         └─────────────────┘
│  ingestion/          │
│  document_parser.py  │
└──────────────────────┘
```

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- At least one LLM API key ([Groq free](https://console.groq.com/keys), or [Gemini free](https://aistudio.google.com/apikey))
- Optional: [Tavily API key](https://tavily.com) for reliable web search

### Local Setup

```bash
# Create and activate virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Edit .env with your keys:
#   GROQ_API_KEY=gsk_...          (required for primary LLM)
#   TAVILY_API_KEY=tvly-...       (recommended for reliable web search)
#   GOOGLE_API_KEY=AIza...        (optional, automatic fallback)
```

### Usage — Web UI

```bash
python server.py
# Open http://localhost:8080
```

Enter a topic or just start chatting. The + menu next to the input bar offers:
- **Upload file** — attach `.pdf`, `.docx`, `.txt`, or `.md` documents to chat with
- **Deep Research** — toggle multi-step autonomous deep research mode
- **Read replies aloud** — toggle voice output for assistant replies

### Usage — CLI

```bash
# Interactive deep research mode
python main.py

# Pipe a topic directly
echo "Quantum computing breakthroughs 2026" | python main.py
```

### Usage — API

```bash
# Chat: create a session
curl -X POST http://localhost:8080/api/chat/session

# Chat: stream a message
curl -N -X POST http://localhost:8080/api/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"session_id":"<sid>","message":"Hello","voice":false}'

# Deep research (SSE streaming)
curl -X POST http://localhost:8080/api/run \
  -H "Content-Type: application/json" \
  -d '{"topic":"Quantum computing 2026"}'

# List reports
curl http://localhost:8080/api/reports

# Delete a report
curl -X DELETE http://localhost:8080/api/reports/<filename>

# Health check
curl http://localhost:8080/health
```

---

## 🌐 Deploy to Render

Resona is pre-configured for one-click deployment to [Render](https://render.com):

```bash
git push origin main
# Connect your GitHub repo at https://render.com → New → Web Service
# Render auto-detects render.yaml
```

Set these environment secrets in the Render dashboard:
- `GROQ_API_KEY` — your Groq API key
- `TAVILY_API_KEY` — your Tavily API key (recommended)
- `GOOGLE_API_KEY` — your Gemini API key (optional fallback)

---

## 📁 Project Structure

```
.
├── .env                      # API keys & config
├── .env.example              # Template for .env
├── .gitignore
├── requirements.txt
├── README.md
│
├── server.py                 # FastAPI web server with SSE streaming
├── main.py                   # CLI entry point
├── router.py                 # Pipeline router
├── orchestrator.py           # Research orchestration
│
├── chat/                     # Chat mode
│   ├── chat_graph.py         # LangGraph for chat conversation
│   ├── chat_prompts.py       # Chat system prompts
│   └── conversation_buffer.py# In-memory message history per session
│
├── chain/                    # Deep research pipeline
│   ├── chain.py              # LLM chains (planner, writer, etc.)
│   └── prompts.py            # System prompts for each agent
│
├── ingestion/                # Document upload
│   └── document_parser.py    # Extract text from .pdf/.docx/.txt/.md
│
├── memory/                   # RAG memory layer
│   ├── chroma_store.py       # Persistent ChromaDB (cross-session)
│   ├── faiss_index.py        # In-session FAISS index
│   └── session_store.py      # Per-session document storage
│
├── search_provider.py        # Unified web search (Tavily + ddgs fallback)
├── research_queue.py         # Parallel sub-question research
├── critic.py                 # Self-correcting critic loop
├── graph.py                   # LangGraph state machine (critic, verifier nodes)
│
├── schemas/
│   ├── models.py             # Pydantic data models
│   ├── chat_models.py        # Chat-specific models
│   └── parser.py             # Parse agent output
│
├── llm_config.py             # Unified LLM config (Groq/Gemini/OpenAI/Anthropic)
├── retry_utils.py            # Tenacity retry logic
├── token_tracker.py          # Token usage tracking
├── tracing.py                # LangSmith OpenTelemetry setup
│
├── static/
│   └── index.html            # Unified React UI (chat + deep research)
│
├── output/                   # Generated reports (.md + .pdf)
├── chroma_db/                # ChromaDB persistent storage
├── tests/                    # Pytest test suite
├── Dockerfile                # Docker image
├── docker-compose.yml        # Local Docker compose
└── render.yaml               # Render deployment config
```

---

## 🔧 Configuration

### Environment Variables

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `GROQ_API_KEY` | Yes* | — | Groq API key (primary LLM) |
| `GOOGLE_API_KEY` | No | — | Gemini Flash fallback when Groq budget exhausted |
| `OPENAI_API_KEY` | No | — | OpenAI key (if `LLM_PROVIDER=openai`) |
| `ANTHROPIC_API_KEY` | No | — | Anthropic key (if `LLM_PROVIDER=anthropic`) |
| `TAVILY_API_KEY` | No | — | Tavily web search (much more reliable than ddgs fallback) |
| `LLM_PROVIDER` | No | `groq` | `groq`, `openai`, or `anthropic` |
| `QUALITY_THRESHOLD` | No | `7` | Critic pass threshold (0–10) |
| `LLM_MODEL_FAST` | No | per-provider | Fast model (planner, research) |
| `LLM_MODEL_CAPABLE` | No | per-provider | Capable model (analyst, writer, critic) |
| `LANGSMITH_API_KEY` | No | — | LangSmith tracing |
| `RESEARCH_MAX_CONCURRENT` | No | `1` | Parallel research workers |

*At least one LLM API key required.*

---

## 🧪 Tests

```bash
PYTHONPATH=. pytest tests/ -v
```

Tests verify:
- Markdown and PDF report generation
- Output directory creation
- LangGraph mode routing
- Web search fallback (mocked tests, no API keys needed)
- Session lifecycle and chat buffer CRUD
- Hedge-phrase detection and critic scoring
- Citation integrity (no fabricated sources)

---

## 🛠 Tech Stack

| Technology | Purpose |
|------------|---------|
| **FastAPI** + **SSE-Starlette** | Web server with real-time streaming |
| **LangGraph** | State machine orchestration |
| **Groq** | Primary LLM inference (fast) |
| **Gemini Flash** | Automatic fallback LLM |
| **Tavily** | Primary web search (API, reliable) |
| **DuckDuckGo** | Free web search fallback |
| **ChromaDB** | Persistent research memory |
| **FAISS** | In-session document similarity |
| **Pydantic** | Typed data validation |
| **Tenacity** | Retry with exponential backoff |
| **WeasyPrint** | HTML/CSS to PDF |
| **Web Speech API** | Voice input (browser) |
| **SpeechSynthesis** | Voice output (browser) |
| **LangSmith** | OpenTelemetry tracing |
| **Docker** / **Render** | Deployment |

---

## 📄 License

MIT
