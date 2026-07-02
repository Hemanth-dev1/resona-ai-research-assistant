"""Unified pipeline orchestrator — always uses LangGraph.

Provides a single run_pipeline() entry point that coordinates:
1. Planning (decompose topic into sub-questions)
2. Analysis + Writing + Critic + Verification (via router.run_analysis)

The async parallel research step and SSE streaming are handled by server.py
for UI interactivity. The orchestrator focuses on the synchronous core pipeline.

Usage:
    from orchestrator import run_pipeline

    result = run_pipeline("AI safety")
    print(result["report"])
    print(result["critique_iterations"], result["verification_result"])
"""

import os
import time
from typing import Optional


def get_mode() -> str:
    """Return the orchestration mode — always 'langgraph'.

    Kept as a function for backward compatibility with imports.
    """
    return "langgraph"


def run_pipeline(
    topic: str,
    merged_research: str = "",
    memory_context: str = "",
    max_critic_iterations: int = 3,
) -> dict:
    """Run the analysis + writing + critic + verification pipeline via LangGraph.

    This is the synchronous core of the research pipeline. The async parallel
    research step (web search) and SSE streaming are handled by server.py.

    Args:
        topic: The research topic.
        merged_research: Pre-computed research from parallel queue.
            If empty, the graph will run planner + LLM research internally.
        memory_context: Optional context from ChromaDB memory.
        max_critic_iterations: Max critic loop iterations.

    Returns:
        Dict with keys:
            - report (str): The final report content
            - critique_iterations (int): Number of critic iterations
            - critique_score (int or None): Final critique score
            - critique_passed (bool or None): Whether critique passed
            - verification_result (dict): Verification findings
            - plan (dict or None): Planner output
            - sub_questions (list): Planner sub-questions
            - error (str or None): Error message if pipeline failed
            - duration_seconds (float): Total pipeline duration
    """
    start_time = time.time()
    print(f"\n🚀 Pipeline: '{topic}' | research={len(merged_research)} chars")

    # Step 1: Plan + Research (if no merged_research provided)
    plan = None
    sub_questions = []
    if not merged_research:
        from chain.chain import run_planner
        plan = run_planner(topic)
        sub_questions = plan.get("sub_questions", []) if plan else []
        print(f"  📋 Planner: {len(sub_questions)} sub-questions generated")

        # Fallback: run a synchronous web search for CLI mode
        print(f"  🔍 Running fallback web search for '{topic}'...")
        try:
            from duckduckgo_search import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(topic, max_results=8))
                if results:
                    merged_research = "\n\n".join(
                        f"- **{r['title']}**: {r['body']}" for r in results
                    )
                    print(f"  ✅ Web search: {len(results)} results, {len(merged_research)} chars")
                else:
                    merged_research = ""
        except Exception as e:
            print(f"  ⚠️  Web search failed: {e} — using LLM-only mode")
            merged_research = ""

    # Step 2: Analysis + Writing + Critic + Verification (always LangGraph)
    from router import run_analysis
    report, critique_iterations, verification_result = run_analysis(
        topic, merged_research
    )

    if report.startswith("❌"):
        duration = time.time() - start_time
        return {
            "report": "",
            "critique_iterations": 0,
            "critique_score": None,
            "critique_passed": None,
            "verification_result": {},
            "plan": plan,
            "sub_questions": sub_questions,
            "error": report,
            "duration_seconds": duration,
        }

    duration = time.time() - start_time
    print(f"  ⏱️  Pipeline completed in {duration:.1f}s")

    return {
        "report": report,
        "critique_iterations": critique_iterations,
        "critique_score": None,
        "critique_passed": None,
        "verification_result": verification_result,
        "plan": plan,
        "sub_questions": sub_questions,
        "error": None,
        "duration_seconds": duration,
    }
