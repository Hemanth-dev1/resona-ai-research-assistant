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

MAX_CONCURRENT = int(os.getenv("RESEARCH_MAX_CONCURRENT", "2"))
"""Maximum number of concurrent research workers (default: 1, avoids Groq rate limits).

Groq free tier has 6000 TPM limit for llama-3.1-8b-instant. Set to 2+ if using
a paid tier or a provider with higher limits."""


# ── Web search helpers (sync, wrapped in asyncio.to_thread) ────────────────

def _search_web(query: str, max_results: int = 5) -> list[dict]:
    """Run a web search and return structured source dicts.

    Delegates to search_provider.web_search which tries Tavily first
    (if TAVILY_API_KEY is set), then falls back to ddgs (free, no key
    needed). Previously this function called ddgs directly and silently
    swallowed all errors — that was the root cause of "no sources
    retrieved" failures.

    Each result is a dict with keys: url, title, snippet (the body text).

    Returns:
        List of dicts, each with 'url', 'title', 'snippet' keys.
        Empty list on error or no results.
    """
    from search_provider import web_search
    return web_search(query, max_results=max_results)


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
                "## CRITICAL: NO FABRICATION\n"
                "You may ONLY cite from the sources listed below with their assigned "
                "[S#] IDs. Do NOT invent source IDs, statistics, study names, journal "
                "references, or data points that are not present in the provided sources. "
                "If the sources don't answer a question, say so explicitly rather than "
                "fabricating evidence.\n\n"
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
            from token_tracker import record_from_response
            record_from_response("llama-3.1-8b-instant", result)
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
    all_sources: dict[str, dict] = {}  # url -> source dict WITH a GLOBAL id

    for i, r in enumerate(results):
        question = r.get("question", f"Sub-question {i+1}")
        findings = r.get("findings", "")
        sources = r.get("sources", [])  # each has its own PER-SUB-QUESTION-LOCAL id, e.g. "S1"

        # _format_sources_with_ids() numbers sources independently per
        # sub-question (every sub-question starts at S1). Build a
        # local-id -> global-id remap here: reuse the existing global id
        # if this URL already appeared in an earlier sub-question,
        # otherwise mint a new one. Without this, unrelated sources from
        # different sub-questions collide under the same [S#] tag.
        local_to_global: dict[str, str] = {}
        for src in sources:
            url = src.get("url", "")
            local_id = src.get("id", "")
            if not url or not local_id:
                continue
            if url in all_sources:
                global_id = all_sources[url]["id"]
            else:
                global_id = f"S{len(all_sources) + 1}"
                all_sources[url] = {**src, "id": global_id}
            local_to_global[local_id] = global_id

        # Rewrite this sub-question's findings text so every [S#] tag now
        # points at the GLOBAL id. Sort local ids longest-first so "S10"
        # is replaced before "S1" — otherwise a naive replace could
        # corrupt "S10" into "<global-for-S1>0".
        rewritten = findings
        for local_id in sorted(local_to_global, key=len, reverse=True):
            rewritten = re.sub(
                rf"\[{re.escape(local_id)}\]",
                f"[{local_to_global[local_id]}]",
                rewritten,
            )

        finding_sections.append(
            f"## Research Finding {i+1}: {question}\n{rewritten}"
        )

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
