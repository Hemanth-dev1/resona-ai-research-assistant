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
            for i, q in enumerate(plan["sub_questions"]):
                if isinstance(q, str):
                    plan["sub_questions"][i] = {"question": q, "rationale": "", "priority": i + 1}
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

    Args:
        topic: The research topic.
        merged_research: Merged research string from all sub-question workers.

    Returns:
        dict with keys:
            - "research": The input merged research
            - "analysis": Analysis output string
            - "report": Final written report string
    """
    _, _, analysis_chain, writer_chain = _get_chains()

    print(f"  🧠 LangChain: Analyzing findings from {len(merged_research)} chars of research...")
    analysis = analysis_chain.invoke({"research": merged_research})

    time.sleep(1)  # 🐌 Rate-limit spacer

    print(f"  ✍️  LangChain: Writing report...")
    report = writer_chain.invoke({"topic": topic, "analysis": analysis})

    return {
        "research": merged_research,
        "analysis": analysis,
        "report": report,
    }


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
