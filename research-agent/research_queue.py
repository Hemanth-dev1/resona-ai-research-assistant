"""Async parallel research queue with concurrency throttling.

Replaces the single sequential research call with per-sub-question parallel
web search + LLM synthesis, capped at MAX_CONCURRENT concurrent workers.

Usage:
    from research_queue import run_parallel_research
    merged = await run_parallel_research(topic, plan["sub_questions"], progress_queue=q)
"""

import asyncio
import os
import re
import time as time_module
from typing import Optional

# ── Configuration ──────────────────────────────────────────────────────────

MAX_CONCURRENT = int(os.getenv("RESEARCH_MAX_CONCURRENT", "1"))
"""Maximum number of concurrent research workers (default: 1, avoids Groq rate limits).

Groq free tier has 6000 TPM limit for llama-3.1-8b-instant. Set to 2+ if using
a paid tier or a provider with higher limits."""


# ── Web search helpers (sync, wrapped in asyncio.to_thread) ────────────────

def _search_web(query: str, max_results: int = 5) -> str:
    """Run a DuckDuckGo search and return formatted results."""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return f"No search results found for: {query}"
        formatted = []
        for i, r in enumerate(results, 1):
            title = r.get("title", "")
            body = r.get("body", "")
            href = r.get("href", "")
            formatted.append(f"**Result {i}:** {title}\n   {body}\n   URL: {href}")
        return "\n\n".join(formatted)
    except ImportError:
        return "Web search unavailable (duckduckgo_search not installed)."
    except Exception as e:
        return f"Search error: {e}"


def _synthesize_findings(
    sub_question: str,
    rationale: str,
    topic: str,
    search_results: str,
    memory_context: str,
) -> str:
    """Call the fast LLM to synthesize findings from search results.

    Includes automatic retry on 429 rate limit errors with the
    retry-after time extracted from Groq's error message.
    """
    from llm_config import get_fast_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_fast_llm(temperature=0.3, max_tokens=2048)
    messages = [
        SystemMessage(
            content=(
                "You are a focused research analyst. Answer ONE specific sub-question "
                "using the web search results provided. Be concise and factual.\n\n"
                "Structure: key finding, supporting evidence, source citations."
            )
        ),
        HumanMessage(
            content=(
                f"Research topic: {topic}\n"
                f"Sub-question: {sub_question}\n"
                f"Rationale: {rationale}\n\n"
                f"Web search results:\n{search_results}\n\n"
                f"Context: {memory_context}\n\n"
                "Synthesize your findings."
            )
        ),
    ]

    max_retries = 5
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            result = llm.invoke(messages)
            return result.content if hasattr(result, "content") else str(result)
        except Exception as e:
            last_error = e
            err_str = str(e).lower()

            # Check if this is a rate limit (429) error
            if "rate limit" in err_str or "rate_limit" in err_str or "429" in err_str:
                # Try to extract the suggested wait time from Groq's error message
                wait_match = re.search(r"try again in ([\d.]+)s", str(e))
                wait_time = float(wait_match.group(1)) + 2 if wait_match else min(5 * attempt, 60)
                print(f"  ⏳ Rate limit hit — waiting {wait_time:.1f}s before retry ({attempt}/{max_retries})...")
                time_module.sleep(wait_time)
            else:
                # Non-rate-limit error: short backoff then give up
                if attempt < max_retries:
                    time_module.sleep(min(2 ** attempt, 15))
                else:
                    print(f"  ❌ LLM synthesis failed: {e}")
                    return f"Synthesis error: {e}"

    print(f"  ❌ All {max_retries} retries exhausted: {last_error}")
    return f"Synthesis unavailable after {max_retries} retries: {last_error}"


# ── Parallel research worker ───────────────────────────────────────────────

async def _research_one(
    sub_q: dict,
    topic: str,
    memory_context: str,
    sem: asyncio.Semaphore,
    index: int,
    total: int,
    progress_queue: Optional[asyncio.Queue] = None,
) -> str:
    """Research a single sub-question: search web + LLM synthesis.

    Pushes progress events to progress_queue if provided.

    Args:
        sub_q: Sub-question dict.
        topic: The overall research topic.
        memory_context: Memory context.
        sem: Semaphore for concurrency control.
        index: 0-based index.
        total: Total number of sub-questions.
        progress_queue: Optional asyncio.Queue for real-time SSE progress.

    Returns:
        Synthesized findings string.
    """
    question = sub_q.get("question", str(sub_q)) if isinstance(sub_q, dict) else str(sub_q)
    rationale = sub_q.get("rationale", "") if isinstance(sub_q, dict) else ""

    _emit_progress(progress_queue, index, total, "searching")

    async with sem:
        search_results = await asyncio.to_thread(
            _search_web, f"{topic} {question}", max_results=5
        )

        _emit_progress(progress_queue, index, total, "synthesizing")

        findings = await asyncio.to_thread(
            _synthesize_findings,
            question, rationale, topic, search_results, memory_context,
        )

    _emit_progress(progress_queue, index, total, "complete")
    return findings


def _emit_progress(queue: Optional[asyncio.Queue], index: int, total: int, status: str):
    """Emit a progress event to the async queue (non-blocking)."""
    if queue is not None:
        queue.put_nowait((index, total, status))


# ── Main entry point ───────────────────────────────────────────────────────

async def run_parallel_research(
    topic: str,
    sub_questions: list,
    memory_context: str = "",
    max_concurrent: int = MAX_CONCURRENT,
    progress_queue: Optional[asyncio.Queue] = None,
) -> str:
    """Run parallel research on each sub-question with concurrency cap.

    Each sub-question gets its own DuckDuckGo web search + LLM synthesis.
    Results are merged into a single research brief.

    Args:
        topic: The overall research topic.
        sub_questions: List of sub-question dicts from the planner.
        memory_context: Optional context from ChromaDB memory.
        max_concurrent: Max concurrent research workers (default: 2).
        progress_queue: Optional asyncio.Queue for real-time SSE progress.

    Returns:
        Merged research string combining all sub-question findings.
    """
    if not sub_questions:
        return ""

    sem = asyncio.Semaphore(max_concurrent)
    tasks = [
        _research_one(q, topic, memory_context, sem, i, len(sub_questions), progress_queue)
        for i, q in enumerate(sub_questions)
    ]

    results = await asyncio.gather(*tasks)

    header = f"# Parallel Research: {topic}\n\n"
    merged = header + "\n\n---\n\n".join(
        f"## Research Finding {i+1}\n{r}"
        for i, r in enumerate(results) if r.strip()
    )
    return merged
