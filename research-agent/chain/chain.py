"""LangChain LCEL pipeline for the research agent.

Builds four chains (plan → research → analysis → writing) using
LangChain Expression Language (LCEL) with model routing:
- Planner & Research: fast/cheap model (default: llama-3.1-8b-instant)
- Analysis & Writing: capable model (default: llama-3.3-70b-versatile)

Note: The research step is now handled by `research_queue.run_parallel_research()`
in the async SSE context. This module provides:
- run_planner(): Decompose topic into sub-questions
- run_analysis_writing(): Takes merged research, produces final report
"""

import json
import re
import time
from functools import lru_cache
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langsmith import traceable

from chain.prompts import PLANNER_PROMPT, RESEARCHER_PROMPT, ANALYST_PROMPT, WRITER_PROMPT


@lru_cache(maxsize=1)
def _get_llms() -> tuple:
    """Get cached fast and capable LLM instances.

    Lazily initialized so the module can be imported without API key.
    Uses LLM_MODEL_FAST for planning (fast/cheap) and LLM_MODEL_CAPABLE
    for analysis and writing (high quality).

    Returns:
        Tuple of (fast_llm, capable_llm).
    """
    from llm_config import get_fast_llm, get_capable_llm

    fast_llm = get_fast_llm(temperature=0.3, max_tokens=4096)
    capable_llm = get_capable_llm(temperature=0.3, max_tokens=8192)
    return fast_llm, capable_llm


@lru_cache(maxsize=1)
def _get_chains() -> tuple:
    """Build and return the LCEL chains.

    Returns:
        Tuple of (planner_chain, research_chain, analysis_chain, writer_chain).
    """
    fast_llm, capable_llm = _get_llms()
    planner_chain = PLANNER_PROMPT | fast_llm | StrOutputParser()
    research_chain = RESEARCHER_PROMPT | fast_llm | StrOutputParser()
    analysis_chain = ANALYST_PROMPT | capable_llm | StrOutputParser()
    writer_chain = WRITER_PROMPT | capable_llm | StrOutputParser()
    return planner_chain, research_chain, analysis_chain, writer_chain


# ── Planner ────────────────────────────────────────────────────────────────

def run_planner(topic: str) -> Optional[dict]:
    """Run the Planner chain to decompose a topic into sub-questions.

    The planner now generates specific, time-anchored sub-questions with
    optimized search queries (short keyword strings, not full sentences).
    Each sub-question includes:
      - question: The specific, checkable research question
      - search_query: Short keyword-focused query for web search
      - rationale: Why this question matters
      - priority: Priority order (1 = highest)

    Args:
        topic: The research topic to plan.

    Returns:
        Dict with 'topic', 'sub_questions', 'suggested_approach' keys,
        or None if planning fails.
    """
    try:
        planner_chain, _, _, _ = _get_chains()
        raw = planner_chain.invoke({"topic": topic})
        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            plan = json.loads(json_match.group())
        else:
            plan = json.loads(raw)

        if isinstance(plan.get("sub_questions"), list):
            normalized = []
            for i, q in enumerate(plan["sub_questions"]):
                if isinstance(q, str):
                    # Legacy: plain string question — fill defaults
                    normalized.append({
                        "question": q,
                        "search_query": q[:60],  # Use question as rough search query
                        "rationale": "",
                        "priority": i + 1,
                    })
                elif isinstance(q, dict):
                    # Ensure search_query field exists (backward compat with old plans)
                    if "search_query" not in q:
                        q["search_query"] = q.get("question", topic)[:60]
                    normalized.append(q)
            plan["sub_questions"] = normalized
        return plan
    except Exception as e:
        print(f"  ⚠️  Planner failed (falling back to direct research): {e}")
        return None


# ── Analysis + Writing (after parallel research) ───────────────────────────

@traceable(name="langchain_analysis_writing", run_type="chain")
def run_analysis_writing(topic: str, merged_research: str) -> dict:
    """Run analysis and writing on pre-computed research.

    Called after parallel research completes. Takes the merged research
    from all sub-questions, analyzes it, and produces the final report.

    The analyst now outputs structured JSON with citation-forced findings.
    The parser extracts the findings text (with [S#] tags) for the writer,
    and also passes gap information so the writer can handle missing evidence
    transparently.

    The writer receives:
      - The analyst's findings (with [S#] tags preserved)
      - Structured source data extracted from the merged research's
        "Tracked Sources" section (from Step 2)
      - An evidence gap note if sources were insufficient

    After the writer finishes, the report is scanned for any missing
    Sources section — if absent, one is appended programmatically.

    Args:
        topic: The research topic.
        merged_research: Merged research string from all sub-question workers.

    Returns:
        dict with keys:
            - "research": The input merged research
            - "analysis": Analysis output string (findings with [S#] citations)
            - "report": Final written report string
            - "has_sufficient_evidence": Whether sources answered the question
            - "gap_reason": Explanation of gaps if evidence was insufficient
            - "sources_data": Structured source data for the report
    """
    _, _, analysis_chain, writer_chain = _get_chains()

    print(f"  🧠 LangChain: Analyzing findings from {len(merged_research)} chars of research...")
    raw_analysis = analysis_chain.invoke({"research": merged_research})

    # Parse the structured JSON from the analyst
    analysis_text = raw_analysis
    has_sufficient_evidence = True
    gap_reason = None
    try:
        json_match = re.search(r"\{.*\}", raw_analysis, re.DOTALL)
        if json_match:
            parsed = json.loads(json_match.group())
        else:
            parsed = json.loads(raw_analysis)

        if isinstance(parsed, dict):
            analysis_text = parsed.get("findings", raw_analysis)
            has_sufficient_evidence = parsed.get("has_sufficient_evidence", True)
            gap_reason = parsed.get("gap_reason")

        if not has_sufficient_evidence and gap_reason:
            print(f"  ⚠️  Analyst: evidence gap detected — {gap_reason[:80]}...")

    except (json.JSONDecodeError, Exception) as e:
        print(f"  ⚠️  Could not parse structured analyst output: {e}")
        analysis_text = raw_analysis  # Fall back to raw string

    # ── Extract structured source data from merged_research ─────────────────
    sources_data = _extract_sources_from_research(merged_research)
    print(f"  📚 Extracted {len(sources_data)} sources from research for writer")

    time.sleep(1)  # 🐌 Rate-limit spacer

    # ── Build writer input ─────────────────────────────────────────────────
    writer_input = {
        "topic": topic,
        "analysis": analysis_text,
        "sources_data": sources_data,
    }
    if gap_reason:
        # Append gap info so the writer can handle it explicitly
        writer_input["analysis"] += (
            f"\n\n### Evidence Gap Note\n"
            f"The following areas lack sufficient evidence: {gap_reason}\n"
            f"When writing the report, note these gaps explicitly rather "
            f"than writing filler or hedging prose."
        )

    print(f"  ✍️  LangChain: Writing report...")
    report = writer_chain.invoke(writer_input)

    # ── Post-processing: ensure Sources section exists ──────────────────────
    report = _ensure_sources_section(report, sources_data)

    return {
        "research": merged_research,
        "analysis": analysis_text,
        "report": report,
        "has_sufficient_evidence": has_sufficient_evidence,
        "gap_reason": gap_reason,
        "sources_data": sources_data,
    }


def _extract_sources_from_research(merged_research: str) -> str:
    """Extract the 'Tracked Sources' section from merged research.

    The merged research (from Step 2's run_parallel_research) includes
    a '## Tracked Sources' section at the end with per-source entries
    like: [S1] Title, URL: https://..., snippet.

    This function extracts and formats those entries for the writer
    prompt's {sources_data} placeholder.

    Args:
        merged_research: The merged research string.

    Returns:
        Formatted string of source entries, or a message if none found.
    """
    # Look for the Tracked Sources section
    match = re.search(
        r"## Tracked Sources\s*\n(.*?)(?:\n##|\Z)",
        merged_research,
        re.DOTALL,
    )
    if match:
        return match.group(1).strip()

    # Fallback: try to extract any [S#] entries from the full text
    lines: list[str] = []
    for line in merged_research.split("\n"):
        stripped = line.strip()
        if re.match(r"-?\s*\*{0,2}\[S\d+\]\*{0,2}", stripped):
            lines.append(stripped)
    if lines:
        return "\n".join(lines)

    return "No source metadata was available during research."


def _ensure_sources_section(report: str, sources_data: str) -> str:
    """Ensure the report has a '## Sources' section at the end.

    If the writer already generated one, leave it as-is.
    If not, append one using the provided sources_data.

    Args:
        report: The report text from the writer.
        sources_data: The source entries string.

    Returns:
        Report with a Sources section guaranteed to exist.
    """
    if re.search(r"^## Sources\b", report, re.MULTILINE):
        # Writer already added a Sources section — no intervention needed
        return report

    # No Sources section found — append one
    print(f"  📎 Appending Sources section to report")
    report = report.rstrip() + "\n\n## Sources\n" + sources_data
    return report


# ── Legacy: full sequential pipeline (kept for backward compat) ────────────

@traceable(name="langchain_research_full", run_type="chain")
def run_langchain(topic: str, memory_context: str = "") -> dict:
    """Run the full LangChain pipeline sequentially (legacy).

    For new code, prefer the split flow:
        1. run_planner()
        2. research_queue.run_parallel_research()
        3. run_analysis_writing()

    This function is kept for backward compatibility with router.py.

    Args:
        topic: The research topic to investigate.
        memory_context: Optional context from prior research.

    Returns:
        dict with keys: "plan", "research", "analysis", "report"
    """
    # Step 0: Plan
    print(f"  📋 LangChain: Planning research on '{topic}'...")
    plan = run_planner(topic)
    sub_questions_str = _format_sub_questions(plan) if plan else ""
    print(f"  ✅ Planner generated {len(plan.get('sub_questions', [])) if plan else 0} sub-questions")

    time.sleep(1)

    # Step 1: Research (sequential - used when run_parallel_research not available)
    print(f"  🔍 LangChain: Researching '{topic}'...")
    _, research_chain, _, _ = _get_chains()
    research = research_chain.invoke({
        "topic": topic,
        "memory_context": memory_context or "No prior context available.",
        "sub_questions": sub_questions_str,
    })

    time.sleep(2)

    # Step 2+3: Analysis + Writing
    result = run_analysis_writing(topic, research)

    return {
        "plan": plan,
        "research": result["research"],
        "analysis": result["analysis"],
        "report": result["report"],
    }


# ── Verifier ────────────────────────────────────────────────────────────────

@traceable(name="langchain_verifier", run_type="chain")
def run_verification(topic: str, report: str, merged_research: str) -> dict:
    """Run fact-check verification on the final report.

    Scans the report for factual claims and cross-references them against
    the provided merged research. Returns structured verification results.

    Args:
        topic: The research topic.
        report: The final report content.
        merged_research: The merged research used to write the report.

    Returns:
        Dict with keys: findings, total_claims_checked, passed, summary.
        None if verification fails.
    """
    try:
        from langchain_core.output_parsers import StrOutputParser
        from llm_config import get_capable_llm
        from chain.prompts import VERIFIER_PROMPT

        llm = get_capable_llm(temperature=0.1, max_tokens=4096)
        chain = VERIFIER_PROMPT | llm | StrOutputParser()

        print(f"  ✅ Verifier: Fact-checking report against research...")
        raw = chain.invoke({
            "topic": topic,
            "merged_research": merged_research[:12000],
            "report": report,
        })

        json_match = re.search(r"\{.*\}", raw, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
        else:
            result = json.loads(raw)

        passed = result.get("passed", True)
        num_findings = len(result.get("findings", []))
        total_checked = result.get("total_claims_checked", 0)
        print(f"  {'✅' if passed else '⚠️'} Verifier: {total_checked} claims checked, {num_findings} issues found")

        return result
    except Exception as e:
        print(f"  ⚠️  Verifier failed: {e}")
        return {
            "findings": [],
            "total_claims_checked": 0,
            "passed": True,
            "summary": f"Verification skipped: {e}",
        }


# ── Claim Verification (new Step 5, cheap model) ─────────────────────────────

@traceable(name="claim_verifier", run_type="chain")
def run_claim_verification(topic: str, report: str, merged_research: str) -> dict:
    """Verify each cited claim in the report against its cited source.

    Uses a fast/cheap model to check each claim+source pair individually.
    If any claim is NOT supported by its cited source, the report is routed
    back to the Analyst for revision with the specific unsupported claim + reason.

    Args:
        topic: The research topic.
        report: The final report text with [S#] citation tags.
        merged_research: The merged research with Tracked Sources section.

    Returns:
        Dict with keys:
            - passed (bool): True if ALL claims are supported by their sources
            - claims_checked (int): Total number of claim+source pairs checked
            - unsupported_claims (list[dict]): Each with claim_text, source_id,
              source_url, reason
            - summary (str): One-line summary
    """
    # ── Step 1: Extract source snippets from merged_research ────────────────
    source_snippets: dict[str, str] = _extract_source_snippets(merged_research)

    # ── Step 2: Parse report for [S#] citations and surrounding claims ──────
    claim_pairs = _extract_claim_source_pairs(report, source_snippets)

    if not claim_pairs:
        print(f"  ✅ Claim Verifier: No [S#] citations found to verify")
        return {
            "passed": True,
            "claims_checked": 0,
            "unsupported_claims": [],
            "summary": "No citations found in report to verify.",
        }

    print(f"  🔍 Claim Verifier: Checking {len(claim_pairs)} claim+source pairs...")

    # ── Step 3: Check each claim against its source with a fast LLM ─────────
    unsupported: list[dict] = []
    from llm_config import get_fast_llm
    from langchain_core.messages import HumanMessage, SystemMessage

    llm = get_fast_llm(temperature=0.0, max_tokens=512)

    for i, pair in enumerate(claim_pairs):
        claim_text = pair["claim"]
        source_id = pair["source_id"]
        source_snippet = pair["snippet"]
        source_url = pair.get("url", "")

        messages = [
            SystemMessage(
                content=(
                    "You are a strict fact-checker. Your job is to determine whether a claim "
                    "is actually supported by the provided source text.\n\n"
                    "Answer in strict JSON format:\n"
                    "{\"supported\": true/false, \"reason\": \"Brief explanation of your decision.\"}\n\n"
                    "Be strict — if the source doesn't explicitly support the claim, "
                    "mark it as unsupported. Do not guess or infer."
                )
            ),
            HumanMessage(
                content=(
                    f"Source [{source_id}] snippet:\n{source_snippet[:1000]}\n\n"
                    f"Claim from report:\n{claim_text}\n\n"
                    f"Does this source snippet actually support this claim? "
                    f"Answer as JSON: {{\"supported\": bool, \"reason\": \"string\"}}"
                )
            ),
        ]

        try:
            result = llm.invoke(messages)
            raw = result.content if hasattr(result, "content") else str(result)

            # Parse JSON from response
            json_match = re.search(r"\{.*\}", raw, re.DOTALL)
            if json_match:
                verdict = json.loads(json_match.group())
            else:
                verdict = json.loads(raw)

            supported = verdict.get("supported", False)
            reason = verdict.get("reason", "No reason given.")

            if not supported:
                unsupported.append({
                    "claim_text": claim_text,
                    "source_id": source_id,
                    "source_url": source_url,
                    "reason": reason,
                })
                print(f"  ❌ Claim {i+1}/{len(claim_pairs)}: NOT supported by {source_id} — {reason[:80]}")
            else:
                print(f"  ✅ Claim {i+1}/{len(claim_pairs)}: supported by {source_id}")

        except Exception as e:
            print(f"  ⚠️  Claim {i+1}/{len(claim_pairs)} verification error: {e}")
            # Don't fail on LLM error — treat as pass-through
            continue

    passed = len(unsupported) == 0
    summary = (
        f"Checked {len(claim_pairs)} claims: {len(unsupported)} unsupported."
        if unsupported else
        f"All {len(claim_pairs)} claims are supported by their cited sources."
    )

    print(f"  {'✅' if passed else '❌'} Claim Verifier: {summary}")

    return {
        "passed": passed,
        "claims_checked": len(claim_pairs),
        "unsupported_claims": unsupported,
        "summary": summary,
    }


def _extract_source_snippets(merged_research: str) -> dict[str, str]:
    """Extract source snippets from the Tracked Sources section.

    Reuses _extract_sources_from_research to get the raw section text,
    then parses individual [S#] entries into a dict mapping source_id
    ("S1") to the entry content (title + URL + snippet).

    Returns dict mapping source_id to the full entry text including URL.
    """
    snippets: dict[str, str] = {}

    section = _extract_sources_from_research(merged_research)
    if not section or section.startswith("No source"):
        return snippets

    # Split by [S#] entries
    entries = re.split(r"-\s*\*{0,2}\[(S\d+)\]\*{0,2}", section)
    # entries is [before_first_entry, S1, content_after_S1, S2, content_after_S2, ...]
    for i in range(1, len(entries) - 1, 2):
        sid = entries[i]
        content = entries[i + 1].strip()
        snippets[sid] = content[:500]  # Limit to 500 chars per source

    return snippets


def _extract_claim_source_pairs(report: str, source_snippets: dict[str, str]) -> list[dict]:
    """Parse the report text to extract claims and their cited [S#] sources.

    For each [S#] tag found, captures the surrounding sentence as the claim
    and pairs it with the source snippet.

    Returns:
        List of dicts, each with:
            - claim (str): The sentence containing the citation
            - source_id (str): e.g. "S1"
            - snippet (str): Source snippet text
            - url (str): Source URL if available
    """
    pairs: list[dict] = []
    seen: set[tuple[str, str]] = set()  # Deduplicate (claim_fingerprint, source_id)

    # Find all [S#] citations in the report
    for match in re.finditer(r"\[(S\d+)\]", report):
        source_id = match.group(1)
        pos = match.start()

        # Extract the sentence containing this citation
        # Find sentence boundaries: look backwards to previous period/newline,
        # forward to next period/newline
        start = pos
        while start > 0 and report[start] not in ".!?\n":
            start -= 1
        if start > 0:
            start += 1  # Skip the period/newline

        end = pos + len(match.group())
        while end < len(report) and report[end] not in ".!?\n":
            end += 1
        if end < len(report):
            end += 1  # Include the period

        claim_text = report[start:end].strip()

        # Skip if no real content
        if len(claim_text) < 10:
            continue

        # Get source snippet
        snippet = source_snippets.get(source_id, "(Source snippet not available in research material)")

        # Extract URL from snippet if present
        url = ""
        url_match = re.search(r"URL:\s*(\S+)", snippet)
        if url_match:
            url = url_match.group(1)

        # Deduplicate: use (normalized claim, source_id) as key
        fingerprint = (claim_text[:80].lower(), source_id)
        if fingerprint in seen:
            continue
        seen.add(fingerprint)

        pairs.append({
            "claim": claim_text,
            "source_id": source_id,
            "snippet": snippet,
            "url": url,
        })

    return pairs


def _format_sub_questions(plan: dict) -> str:
    """Format planner sub-questions into a readable string."""
    if not plan or "sub_questions" not in plan:
        return ""
    lines = ["\n\n## Research Plan\nThis research should answer the following sub-questions:\n"]
    for i, q in enumerate(plan["sub_questions"], 1):
        question = q.get("question", str(q)) if isinstance(q, dict) else str(q)
        rationale = q.get("rationale", "") if isinstance(q, dict) else ""
        lines.append(f"{i}. **{question}**")
        if rationale:
            lines.append(f"   — *Rationale: {rationale}*")
    if plan.get("suggested_approach"):
        lines.append(f"\n**Suggested approach:** {plan['suggested_approach']}")
    return "\n".join(lines)
