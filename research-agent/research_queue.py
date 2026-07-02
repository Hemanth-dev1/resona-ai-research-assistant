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

def _search_web(query: str, max_results: int = 5) -> list[dict]:
    """Run a DuckDuckGo search and return structured source dicts.

    Each result is a dict with keys: url, title, snippet (the body text).
    This replaces the old string formatting — sources are now structured
    so they can be assigned IDs (S1, S2, ...) and tracked through the
    pipeline to the final report.

    Returns:
        List of dicts, each with 'url', 'title', 'snippet' keys.
        Empty list on error or no results.
    """
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        if not results:
            return []
        return [
            {
                "url": r.get("href", ""),
                "title": r.get("title", ""),
                "snippet": r.get("body", ""),
            }
            for r in results
        ]
    except ImportError:
        print("  ⚠️  Web search unavailable (duckduckgo_search not installed).")
        return []
    except Exception as e:
        print(f"  ⚠️  Search error: {e}")
        return []


def _format_sources_with_ids(raw_sources: list[dict]) -> tuple[str, list[dict]]:
    """Assign IDs (S1, S2, ...) to raw search results and format for the prompt.

    Args:
        raw_sources: List of dicts with url, title, snippet keys from _search_web().

    Returns:
        Tuple of (formatted_sources_str, enriched_sources_list) where each
        enriched source dict also has an 'id' key ("S1", "S2", ...).
    """
    enriched: list[dict] = []
    lines: list[str] = []
    for i, src in enumerate(raw_sources):
        sid = f"S{i+1}"
        enriched.append({**src, "id": sid})
        lines.append(
            f"[{sid}] {src.get('title', 'Untitled')}\n"
            f"    URL: {src.get('url', '')}\n"
            f"    {src.get('snippet', '')[:500]}"
        )
    return "\n\n".join(lines) if lines else "No search results available for this query.", enriched


def _synthesize_findings(
    sub_question: str,
    rationale: str,
    topic: str,
    raw_sources: list[dict],
    memory_context: str,
) -> dict:
    """Call the fast LLM to synthesize findings from structured search sources.

    Sources are labeled with IDs (S1, S2, ...) before being passed to the LLM,
    and the LLM is instructed to cite them using [S1], [S2] tags inline.
    The result includes both the findings text and the enriched source list.

    Includes automatic retry on 429 rate limit errors with the
    retry-after time extracted from Groq's error message.

    Returns:
        Dict with keys:
            - findings (str): Synthesized text with inline [S#] citation tags
            - sources (list[dict]): Enriched source list with id, url, title, snippet
    """
    from llm_config import get_fast_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    # Format sources with IDs
    formatted_sources, enriched_sources = _format_sources_with_ids(raw_sources)

    llm = get_fast_llm(temperature=0.3, max_tokens=2048)
    messages = [
        SystemMessage(
            content=(
                "You are a focused research analyst. Answer ONE specific sub-question "
                "using the web search results provided below.\n\n"
                "Each source has an ID like [S1], [S2], etc. When you cite a fact, "
                "reference the source ID inline like [S1] or [S2].\n\n"
                "Structure your response as:\n"
                "- Key finding (with [S#] citations)\n"
                "- Supporting evidence (with [S#] citations)\n"
                "- Any relevant data points\n\n"
                "BE SPECIFIC AND FACTUAL. Do NOT write hedges or vague statements — "
                "every claim must trace back to a cited source."
            )
        ),
        HumanMessage(
            content=(
                f"Research topic: {topic}\n"
                f"Sub-question: {sub_question}\n"
                f"Rationale: {rationale}\n\n"
                f"Web search results (cite by [S#]):\n{formatted_sources}\n\n"
                f"Context: {memory_context}\n\n"
                "Synthesize your findings. Cite sources by [S#] for every factual claim."
            )
        ),
    ]

    max_retries = 5
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            result = llm.invoke(messages)
            findings_text = result.content if hasattr(result, "content") else str(result)
            return {
                "findings": findings_text,
                "sources": enriched_sources,
            }
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
                    return {"findings": f"Synthesis error: {e}", "sources": enriched_sources}

    print(f"  ❌ All {max_retries} retries exhausted: {last_error}")
    return {
        "findings": f"Synthesis unavailable after {max_retries} retries: {last_error}",
        "sources": enriched_sources,
    }


# ── Parallel research worker ───────────────────────────────────────────────

async def _research_one(
    sub_q: dict,
    topic: str,
    memory_context: str,
    sem: asyncio.Semaphore,
    index: int,
    total: int,
    progress_queue: Optional[asyncio.Queue] = None,
) -> dict:
    """Research a single sub-question: search web + LLM synthesis.

    Returns structured data with the findings text and tracked sources.
    Sources are assigned IDs (S1, S2, ...) and the LLM is instructed to
    cite them inline using [S#] tags.

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
        Dict with keys:
            - question (str): The sub-question text
            - findings (str): Synthesized findings with inline [S#] tags
            - sources (list[dict]): Enriched source list with id, url, title, snippet
    """
    question = sub_q.get("question", str(sub_q)) if isinstance(sub_q, dict) else str(sub_q)
    rationale = sub_q.get("rationale", "") if isinstance(sub_q, dict) else ""
    # Use the optimized search_query from the planner (Step 6) if available
    search_query = sub_q.get("search_query", "") if isinstance(sub_q, dict) else ""
    if not search_query:
        search_query = f"{topic} {question}"  # Fallback: use topic + question

    _emit_progress(progress_queue, index, total, "searching")

    async with sem:
        raw_sources = await asyncio.to_thread(
            _search_web, search_query, max_results=5
        )

        _emit_progress(progress_queue, index, total, "synthesizing")

        synthesis = await asyncio.to_thread(
            _synthesize_findings,
            question, rationale, topic, raw_sources, memory_context,
        )

    _emit_progress(progress_queue, index, total, "complete")
    return {
        "question": question,
        "findings": synthesis.get("findings", ""),
        "sources": synthesis.get("sources", []),
    }


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
    Sources are labeled with IDs (S1, S2, ...) and tracked through the
    pipeline. The merged output includes:
      1. Per sub-question findings with inline [S#] citation tags
      2. A consolidated Sources section listing every source with ID + URL

    Args:
        topic: The overall research topic.
        sub_questions: List of sub-question dicts from the planner.
        memory_context: Optional context from ChromaDB memory.
        max_concurrent: Max concurrent research workers (default: 2).
        progress_queue: Optional asyncio.Queue for real-time SSE progress.

    Returns:
        Merged research string combining all sub-question findings with
        structured source references. Sources are cited with [S#] tags
        that the analyst and writer can preserve into the final report.
    """
    if not sub_questions:
        return ""

    sem = asyncio.Semaphore(max_concurrent)
    tasks = [
        _research_one(q, topic, memory_context, sem, i, len(sub_questions), progress_queue)
        for i, q in enumerate(sub_questions)
    ]

    results: list[dict] = await asyncio.gather(*tasks)

    # Build merged findings with structured source tracking
    header = f"# Parallel Research: {topic}\n\n"

    # Per sub-question findings
    finding_sections: list[str] = []
    all_sources: dict[str, dict] = {}  # Deduplicate sources by URL

    for i, r in enumerate(results):
        question = r.get("question", f"Sub-question {i+1}")
        findings = r.get("findings", "")
        sources = r.get("sources", [])

        finding_sections.append(
            f"## Research Finding {i+1}: {question}\n{findings}"
        )

        # Collect all sources, deduplicating by URL
        for src in sources:
            url = src.get("url", "")
            if url and url not in all_sources:
                all_sources[url] = src

    # Consolidated Sources section at the end
    sources_section = "## Tracked Sources\n\n"
    if all_sources:
        for src in all_sources.values():
            sid = src.get("id", "?")
            title = src.get("title", "Untitled")
            url = src.get("url", "")
            snippet = src.get("snippet", "")[:200]
            sources_section += (
                f"- **[{sid}]** {title}\n"
                f"  URL: {url}\n"
                f"  {snippet}\n\n"
            )
    else:
        sources_section += "No sources were retrieved during research.\n"

    merged = header + "\n\n---\n\n".join(finding_sections) + "\n\n---\n\n" + sources_section
    return merged
