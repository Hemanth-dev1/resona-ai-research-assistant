"""
Resona — FastAPI Server

Serves a chat-style HTML UI and provides APIs to run the research agent,
stream progress via SSE, and browse/download reports.
"""

import asyncio
import glob
import json
import os
import re
import sys
from datetime import datetime
from pathlib import Path

import uvicorn
from dotenv import load_dotenv
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse, HTMLResponse
from sse_starlette.sse import EventSourceResponse

load_dotenv()

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

# Initialize LangSmith tracing at server startup
from tracing import setup_tracing
setup_tracing()

# Set up CrewAI environment from unified LLM config
from llm_config import setup_crewai_env, get_provider
setup_crewai_env()

# Initialize orchestrator config
from orchestrator import get_mode as get_orch_mode
get_orch_mode()  # Validate ORCHESTRATION env var at startup

app = FastAPI(title="Resona", version="1.1.0")

OUTPUT_DIR = Path(__file__).parent / "output"
STATIC_DIR = Path(__file__).parent / "static"


# ── Health check (for Render) ───────────────────────────────────────────────

@app.get("/health")
async def health():
    """Health check endpoint for Render deployment."""
    return {"status": "ok", "version": "1.1.0", "provider": get_provider().value}


# ── Serve frontend ──────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>Frontend not found — run the server from the project root.</h1>")


# ── SSE: run research agent with live progress ──────────────────────────────

def _chunk_report(text: str, min_chunk: int = 200) -> list[str]:
    """Split a report into natural chunks by section headings."""
    lines = text.split("\n")
    chunks: list[str] = []
    current: list[str] = []
    for line in lines:
        # Section boundary: h1 or h2 heading
        if line.startswith("# ") or line.startswith("## "):
            if current:
                section = "\n".join(current).strip()
                if len(section) >= min_chunk:
                    chunks.append(section)
                    current = []
                else:
                    # Merge small section into next one
                    current = [section]
                    if not chunks:
                        chunks = current
                        current = []
        current.append(line)
    remaining = "\n".join(current).strip()
    if remaining:
        if chunks:
            chunks.append(remaining)
        else:
            chunks = [remaining]
    return chunks if chunks else [text]


async def run_agent_events(topic: str, depth: str = "standard", fmt: str = "markdown+pdf", model: str = None, mode: str = None):
    """Yield SSE events as the agent pipeline progresses and content streams.

    Orchestrates: plan → parallel research (throttled) → analysis → writing → stream
    """

    def send(event: str, data: dict):
        return {"event": event, "data": json.dumps(data)}

    try:
        from llm_config import get_api_key, get_provider
        provider = get_provider()
        api_key = get_api_key()
        if not api_key:
            key_name = {"groq": "GROQ_API_KEY", "openai": "OPENAI_API_KEY", "anthropic": "ANTHROPIC_API_KEY"}.get(provider.value, "GROQ_API_KEY")
            yield send("error", {"message": f"{key_name} not found in .env file for provider '{provider.value}'."})
            return

        # Resolve mode from orchestrator if not specified
        if mode is None:
            from orchestrator import get_mode as get_orch_mode
            mode = get_orch_mode().value

        # ── Step 0: Plan ──────────────────────────────────────────────────
        mode_labels = {"crewai": "CrewAI", "langchain": "LangChain", "langgraph": "LangGraph"}
        mode_label = mode_labels.get(mode, mode.capitalize())
        yield send("phase", {
            "phase": "planning", "status": "running",
            "message": f"📋 {mode_label}: Planning research strategy...",
            "progress": 5,
        })

        from chain.chain import run_planner
        plan = await asyncio.to_thread(run_planner, topic)
        sub_questions = plan.get("sub_questions", []) if plan else []
        num_q = len(sub_questions)
        yield send("phase", {
            "phase": "planning", "status": "complete",
            "message": f"✅ Research plan ready ({num_q} sub-questions). Starting parallel research...",
            "progress": 10,
        })

        # ── Step 1: Parallel Research (throttled async queue) ──────────────
        yield send("phase", {
            "phase": "research", "status": "running",
            "message": f"🔍 {mode_label}: Researching {num_q} sub-questions in parallel...",
            "progress": 15,
        })

        from research_queue import run_parallel_research
        from memory.chroma_store import get_relevant_context
        memory_context = await asyncio.to_thread(get_relevant_context, topic) or ""

        # Run parallel research with real-time progress via async Queue
        merged_research = ""
        if num_q > 0:
            progress_queue: asyncio.Queue = asyncio.Queue()
            total_q = num_q

            # Launch research in a background task
            research_task = asyncio.create_task(
                run_parallel_research(
                    topic, sub_questions,
                    memory_context=memory_context,
                    max_concurrent=2,
                    progress_queue=progress_queue,
                )
            )

            # Read progress events from the queue in real-time
            done_count = 0
            while done_count < total_q:
                try:
                    idx, total, st = await asyncio.wait_for(
                        progress_queue.get(), timeout=30.0
                    )
                    labels = {"searching": "Searching", "synthesizing": "Analyzing", "complete": "Done"}
                    pct = 15 + int((idx + 1) / total * 20)
                    yield send("phase", {
                        "phase": "research", "status": "running",
                        "message": f"📡 {labels.get(st, st)} sub-question {idx+1}/{total}...",
                        "progress": pct,
                    })
                    if st == "complete":
                        done_count += 1
                except asyncio.TimeoutError:
                    # Check if research task failed
                    if research_task.done() and research_task.exception():
                        raise research_task.exception()
                    continue

            merged_research = await research_task
        else:
            merged_research = f"Research on: {topic}"

        yield send("phase", {
            "phase": "research", "status": "complete",
            "message": f"✅ Parallel research complete ({num_q} sub-questions). Beginning analysis...",
            "progress": 35,
        })
        await asyncio.sleep(0.1)

        # ── Step 2: Analysis + Writing ────────────────────────────────────
        yield send("phase", {
            "phase": "analysis", "status": "running",
            "message": f"🧠 {mode_label}: Analyzing findings...",
            "progress": 45,
        })
        await asyncio.sleep(0.1)

        yield send("phase", {
            "phase": "analysis", "status": "complete",
            "message": f"✅ Analysis complete. Writing report...",
            "progress": 55,
        })
        await asyncio.sleep(0.1)

        yield send("phase", {
            "phase": "writing", "status": "running",
            "message": f"✍️ {mode_label}: Writing report...",
            "progress": 65,
        })

        yield send("content_start", {"message": f"{mode_label} composing the report..."})

        # Run analysis + writing + verification via router (synchronous, in thread)
        from router import run_analysis
        report, critique_iterations, verification_result = await asyncio.to_thread(
            run_analysis, topic, merged_research, mode=mode
        )

        if report.startswith("❌"):
            yield send("error", {"message": report})
            return

        # ── Step 3: Verification ────────────────────────────────────────────
        verify_passed = verification_result.get("passed", True) if verification_result else True
        verify_findings = verification_result.get("findings", []) if verification_result else []
        verify_summary = verification_result.get("summary", "") if verification_result else ""

        yield send("phase", {
            "phase": "verifying", "status": "running",
            "message": f"✅ Verifying: fact-checking report against research...",
            "progress": 76,
        })
        await asyncio.sleep(0.1)

        yield send("phase", {
            "phase": "verifying", "status": "complete",
            "message": f"{'✅' if verify_passed else '⚠️'} Verification: checked {verification_result.get('total_claims_checked', 0) if verification_result else 0} claims, {len(verify_findings)} issues found",
            "progress": 80,
            "verification": {
                "passed": verify_passed,
                "findings": verify_findings,
                "summary": verify_summary,
                "total_claims_checked": verification_result.get("total_claims_checked", 0) if verification_result else 0,
            },
        })
        await asyncio.sleep(0.1)

        # ── Step 4: Save + Evaluate + Stream ────────────────────────────────
        yield send("phase", {
            "phase": "writing", "status": "complete",
            "message": "✅ Report written! Streaming content...",
            "progress": 80,
        })

        # Save to ChromaDB
        from memory.chroma_store import save_report as chroma_save
        try:
            await asyncio.to_thread(chroma_save, topic, report)
        except Exception as e:
            print(f"  ⚠️  ChromaDB save skipped: {e}")

        # Run RAGAS evaluation
        ragas_scores = None
        try:
            from ragas_eval import evaluate_rag
            rag_context = await asyncio.to_thread(get_relevant_context, topic, n_results=5)
            contexts = [c.strip() for c in rag_context.split("\n\n---\n\n") if c.strip()] if rag_context else []
            if contexts:
                ragas_scores = await asyncio.to_thread(evaluate_rag, topic, report, contexts, topic)
        except Exception:
            pass

        # Save files
        from main import save_report
        md_path, pdf_path = await asyncio.to_thread(save_report, topic, report)

        md_content = ""
        if md_path and os.path.exists(md_path):
            with open(md_path, encoding="utf-8") as f:
                md_content = f.read()

        # Stream report in chunks
        chunks = _chunk_report(md_content)
        for i, chunk in enumerate(chunks):
            yield send("chunk", {
                "text": chunk, "index": i, "total": len(chunks),
                "progress": 80 + int((i + 1) / len(chunks) * 15),
            })
            await asyncio.sleep(0.15)

        # Final completion
        yield send("complete", {
            "message": "✅ Research Complete!",
            "topic": topic,
            "markdown": md_path, "pdf": pdf_path,
            "markdown_basename": str(Path(md_path).name) if md_path else None,
            "pdf_basename": str(Path(pdf_path).name) if pdf_path else None,
            "has_pdf": pdf_path is not None and os.path.exists(pdf_path),
            "full_content": md_content,
            "progress": 100,
            "mode": mode,
            "iterations": critique_iterations,
            "ragas_scores": ragas_scores,
        })

    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Server error: {e}\n{tb}")
        yield send("error", {"message": str(e), "detail": tb})


@app.post("/api/run")
async def api_run(request: Request):
    """Run the research agent with SSE streaming progress."""
    body = await request.json()
    topic = body.get("topic", "").strip()
    if not topic:
        return {"error": "No topic provided"}

    from orchestrator import get_mode as get_orch_mode
    mode = body.get("mode") or get_orch_mode().value

    return EventSourceResponse(
        run_agent_events(
            topic=topic,
            depth=body.get("depth", "standard"),
            fmt=body.get("format", "markdown+pdf"),
            model=body.get("model"),
            mode=mode,
        )
    )


# ── Reports API ─────────────────────────────────────────────────────────────

@app.get("/api/reports")
async def list_reports():
    """List all generated reports, newest first."""
    files = sorted(
        glob.glob(str(OUTPUT_DIR / "*.md")),
        key=os.path.getmtime,
        reverse=True,
    )
    reports = []
    for f in files:
        mtime = datetime.fromtimestamp(os.path.getmtime(f))
        size_kb = os.path.getsize(f) / 1024
        name = os.path.basename(f).replace(".md", "")
        # Find corresponding PDF
        pdf_path = f.replace(".md", ".pdf")
        has_pdf = os.path.exists(pdf_path)
        # Clean name: strip the trailing timestamp pattern (_YYYYMMDD_HHMMSS)
        clean_name = re.sub(r"_\d{8}_\d{6}$", "", name)
        clean_name = clean_name.replace("-", " ").replace("_", " ").strip()
        reports.append({
            "name": clean_name,
            "filename": os.path.basename(f),
            "date": mtime.strftime("%b %d, %Y · %H:%M"),
            "size": f"{size_kb:.1f} KB",
            "has_pdf": has_pdf,
        })
    return {"reports": reports}


@app.get("/api/reports/{filename}")
async def get_report(filename: str):
    """Get the content of a specific report."""
    md_path = OUTPUT_DIR / filename
    if not md_path.exists():
        return {"error": "Report not found"}

    content = md_path.read_text(encoding="utf-8")
    pdf_path = md_path.with_suffix(".pdf")
    has_pdf = pdf_path.exists()

    topic = filename.replace("_", " ").rsplit(".", 1)[0]
    # Clean up the topic from the filename pattern
    topic = re.sub(r"_\d{8}_\d{6}$", "", topic)

    return {
        "topic": topic,
        "content": content,
        "filename": filename,
        "has_pdf": has_pdf,
        "pdf_filename": filename.replace(".md", ".pdf"),
    }


# ── PDF download ────────────────────────────────────────────────────────────

@app.get("/api/reports/{filename}/pdf")
async def get_report_pdf(filename: str):
    """Serve the PDF version of a report."""
    pdf_name = filename.replace(".md", ".pdf")
    pdf_path = OUTPUT_DIR / pdf_name
    if not pdf_path.exists():
        return {"error": "PDF not found"}
    return FileResponse(str(pdf_path), media_type="application/pdf", filename=pdf_name)


# ── Memory / Topics API ─────────────────────────────────────────────────────

@app.get("/memory/topics")
async def list_memory_topics():
    """Get all unique research topics stored in ChromaDB memory."""
    from memory.chroma_store import get_all_topics
    topics = get_all_topics()
    return {"topics": topics, "count": len(topics)}


# ── Stats / Analytics API ───────────────────────────────────────────────────

@app.get("/api/stats")
async def get_stats():
    """Get analytics data about generated reports."""
    files = sorted(
        glob.glob(str(OUTPUT_DIR / "*.md")),
        key=os.path.getmtime,
    )
    total = len(files)
    total_size = sum(os.path.getsize(f) for f in files) / 1024

    # Actual word count
    total_words = 0
    for f in files:
        try:
            with open(f, encoding="utf-8") as fh:
                total_words += len(fh.read().split())
        except Exception:
            pass

    return {
        "total_reports": total,
        "total_size_kb": f"{total_size:.0f}",
        "total_words_estimate": total_words,
        "model": os.getenv("OPENAI_MODEL_NAME", "llama-3.1-8b-instant"),
        "avg_words_per_report": int(total_words / total) if total else 0,
        "provider": get_provider().value if get_provider() else "groq",
        "critic_threshold": int(os.getenv("RESONA_CRITIC_THRESHOLD", "7")),
        "iterations": int(os.getenv("RESONA_MAX_CRITIC_ITERATIONS", "3")),
    }


# ── RAGAS Scores API ─────────────────────────────────────────────────────────

@app.get("/api/ragas")
async def get_ragas_scores():
    """Get average RAGAS evaluation scores across all research runs."""
    try:
        from ragas_eval import get_average_scores
        return get_average_scores()
    except Exception as e:
        return {"error": str(e), "total_evaluations": 0}


# ── Launch ──────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print("""
╔══════════════════════════════════════════════════╗
║         🤖  ResearchAgent  Server               ║
║                                                  ║
║   Open:  http://localhost:8080                   ║
║   Stop:  Ctrl+C                                  ║
╚══════════════════════════════════════════════════╝
    """)
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
