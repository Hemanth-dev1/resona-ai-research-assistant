"""Unified pipeline orchestrator with mode selection.

Provides a single run_pipeline() entry point that coordinates:
1. Planning (decompose topic into sub-questions)
2. Analysis + Writing + Critic + Verification (via router.run_analysis)

The async parallel research step and SSE streaming are handled by server.py
for UI interactivity. The orchestrator focuses on the synchronous core pipeline.

Configured via ORCHESTRATION env var: langgraph (default), crewai, or langchain.

Usage:
    from orchestrator import run_pipeline, get_mode, OrchestrationMode

    result = run_pipeline("AI safety", mode="langgraph")
    print(result["report"])
    print(result["critique_iterations"], result["verification_result"])
"""

import os
import time
from enum import Enum
from typing import Optional


class OrchestrationMode(str, Enum):
    """Available pipeline orchestration modes."""

    LANGGRAPH = "langgraph"  # LangGraph StateGraph (recommended, default)
    CREWAI = "crewai"        # CrewAI agent pipeline
    LANGCHAIN = "langchain"  # LangChain LCEL chains


# Default mode if ORCHESTRATION env var is not set
_DEFAULT_MODE = OrchestrationMode.LANGGRAPH


def get_mode() -> OrchestrationMode:
    """Get the configured orchestration mode from the ORCHESTRATION env var.

    Returns:
        OrchestrationMode enum value.
        Defaults to LANGGRAPH if ORCHESTRATION is not set or invalid.
    """
    mode_str = os.getenv("ORCHESTRATION", "langgraph").lower().strip()
    try:
        return OrchestrationMode(mode_str)
    except ValueError:
        valid = ", ".join(m.value for m in OrchestrationMode)
        print(f"  ⚠️  Unknown ORCHESTRATION='{mode_str}'. Valid: {valid}. Falling back to langgraph.")
        return _DEFAULT_MODE


def get_available_modes() -> list[str]:
    """Get list of available orchestration modes.

    Returns:
        List of mode name strings.
    """
    return [m.value for m in OrchestrationMode]


def run_pipeline(
    topic: str,
    merged_research: str = "",
    mode: Optional[str] = None,
    memory_context: str = "",
    max_critic_iterations: int = 3,
) -> dict:
    """Run the analysis + writing + critic + verification pipeline.

    This is the synchronous core of the research pipeline. The async parallel
    research step (web search) and SSE streaming are handled by server.py.

    Args:
        topic: The research topic.
        merged_research: Pre-computed research from parallel queue.
            If empty, the graph will run planner + LLM research internally.
        mode: Orchestration mode ('langgraph', 'crewai', 'langchain').
            If None, uses get_mode() to read the ORCHESTRATION env var.
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
            - mode (str): The mode that was used
            - duration_seconds (float): Total pipeline duration
    """
    if mode is None:
        mode = get_mode().value

    start_time = time.time()
    print(f"\n🚀 Pipeline: '{topic}' | mode={mode} | research={len(merged_research)} chars")

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

    # Step 2: Analysis + Writing + Critic + Verification
    from router import run_analysis
    report, critique_iterations, verification_result = run_analysis(
        topic, merged_research, mode=mode
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
            "mode": mode,
            "duration_seconds": duration,
        }

    duration = time.time() - start_time
    print(f"  ⏱️  Pipeline completed in {duration:.1f}s")

    return {
        "report": report,
        "critique_iterations": critique_iterations,
        "critique_score": None,  # Available from graph but not returned by router
        "critique_passed": None,
        "verification_result": verification_result,
        "plan": plan,
        "sub_questions": sub_questions,
        "error": None,
        "mode": mode,
        "duration_seconds": duration,
    }
