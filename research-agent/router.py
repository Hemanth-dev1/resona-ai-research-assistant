"""Router that runs the research pipeline via LangGraph.

Always uses LangGraph StateGraph with conditional critic edges
(planner → analysis_writer → critic ↔ revise → verifier → END).

Uses LangSmith tracing via @traceable.
"""

import os
import time

from langsmith import traceable

from memory.chroma_store import save_report
from schemas.parser import parse_report


@traceable(name="resona_pipeline", run_type="chain")
def run_analysis(topic: str, merged_research: str) -> tuple:
    """Run analysis + writing on pre-computed research via LangGraph.

    Uses the LangGraph StateGraph with conditional critic edges
    (planner → analysis_writer → critic ↔ revise → verifier → END).

    Args:
        topic: The research topic.
        merged_research: Merged research from all sub-question workers.

    Returns:
        Tuple of (report_text, critique_iterations_count, verification_result_dict).
        If error, returns ("❌ error message", 0, {}).
    """
    start_time = time.time()

    from graph import run_pipeline_graph
    result = run_pipeline_graph(
        topic=topic,
        merged_research=merged_research,
        memory_context=os.getenv("MEMORY_CONTEXT", ""),
        mode="langchain",
        max_critic_iterations=int(os.getenv("RESONA_MAX_CRITIC_ITERATIONS", "3")),
    )
    if result.get("error"):
        return (f"❌ Graph pipeline error: {result['error']}", 0, {})
    report = result.get("report", "")
    critique_iterations = result.get("critique_iterations", 0)

    verification_result = {
        "findings": result.get("verification_findings", []),
        "total_claims_checked": result.get(
            "total_claims_checked", len(result.get("verification_findings", []))
        ),
        "passed": result.get("verification_passed", True),
        "summary": result.get("verification_summary", ""),
    }

    # Validate report structure with Pydantic model
    parsed = parse_report(report, topic)
    if parsed is None:
        print("  ⚠️  Report failed Pydantic validation — structure may be malformed")
    else:
        print(
            f"  ✅ Report validated: {len(parsed.sources)} sources,"
            f" {len(parsed.key_insights)} insights"
        )

    # Save to ChromaDB for future retrieval
    try:
        save_report(topic, report)
    except Exception as e:
        print(f"  ⚠️  ChromaDB save skipped: {e}")

    duration = time.time() - start_time
    print(f"  ⏱️  Analysis+writing completed in {duration:.1f}s")
    return (report, critique_iterations, verification_result)
