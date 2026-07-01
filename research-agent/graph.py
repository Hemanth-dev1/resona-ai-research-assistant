"""LangGraph StateGraph for the research pipeline.

Formalizes the pipeline with typed state and conditional edges:

    START → [planner?] → analysis_writer → critic
        → (passed? → verifier | revise → critic → ...)
            verifier → (passed? → END | revise → critic → ...)

The research step (web search) and planning UI events are handled outside
the graph by server.py. If the graph receives pre-computed merged_research
(and thus planning already happened), it skips directly to analysis_writer.
"""

from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from schemas.models import CritiqueResult


# ── Graph State ────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """Typed state that flows through graph nodes."""

    topic: str
    memory_context: str
    mode: str                       # 'langchain' | 'crewai'
    plan: Optional[dict]
    sub_questions: list
    merged_research: str
    report: str
    critique_score: Optional[int]
    critique_passed: Optional[bool]
    critique_iterations: int
    max_critic_iterations: int
    last_critique: Optional[CritiqueResult]  # avoids re-scoring in revise
    verification_passed: Optional[bool]       # verifier node result
    verification_summary: Optional[str]       # verifier summary text
    verification_findings: list               # list of finding dicts
    total_claims_checked: int                 # number of claims scanned
    verification_iterations: int              # how many times verifier has run
    max_verification_iterations: int          # max verifier cycles (default 1)
    strict_verification: bool                  # if True, failed verification routes back to revise
    error: Optional[str]


# ── Node Functions ─────────────────────────────────────────────────────────

def planner_node(state: ResearchState) -> dict:
    """Decompose topic into sub-questions.

    Only called if planning hasn't been done externally (merged_research is empty).
    """
    print(f"  📋 Graph: Planning research on '{state['topic']}'...")
    from chain.chain import run_planner
    plan = run_planner(state["topic"])
    sub_questions = plan.get("sub_questions", []) if plan else []
    print(f"  ✅ Graph: Planner generated {len(sub_questions)} sub-questions")
    return {"plan": plan, "sub_questions": sub_questions}


def analysis_writer_node(state: ResearchState) -> dict:
    """Analyze merged research and write the report.

    Uses the capable model via LangChain chain.
    (CrewAI mode for analysis+writing is not supported inside the graph —
     use 'langchain' mode for graph, and CrewAI for the full pipeline.)
    """
    topic = state["topic"]
    merged = state["merged_research"]
    if not merged:
        # No external research — run planner first, then use LLM-only research
        from chain.chain import run_planner
        plan = run_planner(topic)
        sub_qs = plan.get("sub_questions", []) if plan else []
        from chain.prompts import RESEARCHER_PROMPT
        from llm_config import get_fast_llm
        from langchain_core.output_parsers import StrOutputParser
        llm = get_fast_llm(temperature=0.3, max_tokens=4096)
        chain = RESEARCHER_PROMPT | llm | StrOutputParser()
        merged = chain.invoke({
            "topic": topic,
            "memory_context": state.get("memory_context", "") or "No prior context.",
            "sub_questions": "",
        })

    print(f"  🧠 Graph: Analyzing + writing report...")
    from chain.chain import run_analysis_writing
    result = run_analysis_writing(topic, merged)
    report = result.get("report", "")
    return {"report": report}


def critic_node(state: ResearchState) -> dict:
    """Score the report on quality dimensions.

    Uses the capable model. Sets score/passed and increments iteration.
    Stores the CritiqueResult so revise_node can use it without re-scoring.
    """
    topic = state["topic"]
    report = state["report"]
    iteration = state.get("critique_iterations", 0) + 1

    print(f"  📝 Graph: Critic iteration {iteration}/{state.get('max_critic_iterations', 3)}...")

    from critic import score_report
    critique: CritiqueResult = score_report(topic, report)
    critique.iteration = iteration

    print(f"  📊 Graph: Score {critique.overall_score}/10 | {'✅ Passed' if critique.passed else '❌ Needs revision'}")

    return {
        "critique_score": critique.overall_score,
        "critique_passed": critique.passed,
        "critique_iterations": iteration,
        "last_critique": critique,
    }


def revise_node(state: ResearchState) -> dict:
    """Revise the report based on the LAST critique (no re-scoring needed).

    Uses the CritiqueResult stored in `last_critique` from critic_node
    to avoid an extra LLM call.
    """
    topic = state["topic"]
    report = state["report"]
    iteration = state.get("critique_iterations", 0)
    last_critique: Optional[CritiqueResult] = state.get("last_critique")

    print(f"  🔄 Graph: Revising report (iteration {iteration})...")

    if last_critique:
        from critic import _revise_report
        revised = _revise_report(topic, report, last_critique)
    else:
        # Fallback: if no last critique available, call full loop
        from critic import run_critic_loop
        revised, _ = run_critic_loop(topic, report, max_iterations=1)

    return {"report": revised}


# ── Entry Router ───────────────────────────────────────────────────────────

def route_from_start(state: ResearchState) -> str:
    """Decide whether to run planner or skip to analysis_writer.

    If merged_research is already provided (pre-computed by server.py),
    skip the planner node since planning already happened.
    """
    if state.get("merged_research") and len(state.get("merged_research", "")) > 50:
        return "analysis_writer"
    return "planner"


def route_after_critic(state: ResearchState) -> str:
    """Decide next step after critique.

    Returns:
        "revise" — go back to revise the report
        END — report is good enough or max iterations reached → proceed to verifier
    """
    passed = state.get("critique_passed", False)
    iterations = state.get("critique_iterations", 0)
    max_iter = state.get("max_critic_iterations", 3)

    if passed:
        print(f"  ✅ Graph: Report accepted (score ≥ threshold)")
        return END  # Goes to verifier (edge: critic → verifier)

    if iterations >= max_iter:
        print(f"  ⚠️  Graph: Max iterations ({max_iter}) reached — proceeding to verifier")
        return END  # Goes to verifier

    print(f"  🔄 Graph: Routing to revise (iteration {iterations}/{max_iter})")
    return "revise"


# ── Verifier Node ──────────────────────────────────────────────────────────

def verifier_node(state: ResearchState) -> dict:
    """Fact-check the final report against the research material.

    Scans the report for factual claims and cross-references them against
    the merged research. If strict_verification is True and critical/high
    issues are found, routes back to revise (up to max_verification_iterations).
    """
    topic = state["topic"]
    report = state["report"]
    merged_research = state.get("merged_research", "")
    iteration = state.get("verification_iterations", 0) + 1
    max_iter = state.get("max_verification_iterations", 1)

    print(f"  ✅ Graph: Verifying report against research (attempt {iteration}/{max_iter})...")

    from chain.chain import run_verification
    result = run_verification(topic, report, merged_research)

    passed = result.get("passed", True)
    findings = result.get("findings", [])
    summary = result.get("summary", "Verification complete.")

    severity = ""
    if not passed:
        critical_count = sum(1 for f in findings if f.get("severity") == "critical")
        high_count = sum(1 for f in findings if f.get("severity") == "high")
        severity = f" ({critical_count} critical, {high_count} high)" if findings else ""
        print(f"  ⚠️  Graph: Verification found {len(findings)} issues{severity}")

    return {
        "verification_passed": passed,
        "verification_summary": summary,
        "verification_findings": findings,
        "verification_iterations": iteration,
        "total_claims_checked": result.get("total_claims_checked", len(findings)),
    }


def route_after_verifier(state: ResearchState) -> str:
    """Decide next step after verification.

    If strict_verification is True and verification failed, route back to
    revise for another iteration (up to max_verification_iterations).
    """
    passed = state.get("verification_passed", True)
    strict = state.get("strict_verification", False)
    iterations = state.get("verification_iterations", 0)
    max_iter = state.get("max_verification_iterations", 1)

    if not passed and strict and iterations < max_iter:
        print(f"  🔄 Graph: Verification failed ({iterations}/{max_iter}) — routing back to revise")
        return "revise"

    if not passed:
        print(f"  ⚠️  Graph: Verification flagged {'' if strict else '(non-strict mode) '}issues — complete")
    else:
        print(f"  ✅ Graph: Verification passed — complete")
    return END


# ── Build Graph ────────────────────────────────────────────────────────────

def build_research_graph() -> StateGraph:
    """Build and compile the research pipeline StateGraph."""
    builder = StateGraph(ResearchState)

    # Add nodes
    builder.add_node("planner", planner_node)
    builder.add_node("analysis_writer", analysis_writer_node)
    builder.add_node("critic", critic_node)
    builder.add_node("revise", revise_node)
    builder.add_node("verifier", verifier_node)

    # Entry: skip planner if research already pre-computed
    builder.add_conditional_edges(
        "__start__",
        route_from_start,
        {
            "planner": "planner",
            "analysis_writer": "analysis_writer",
        },
    )

    # planner → analysis_writer (only if planner ran)
    builder.add_edge("planner", "analysis_writer")

    # analysis_writer → critic
    builder.add_edge("analysis_writer", "critic")

    # Conditional: critic → revise or verifier (via END)
    builder.add_conditional_edges(
        "critic",
        route_after_critic,
        {"revise": "revise", END: "verifier"},
    )

    # Loop back: revise → critic for re-scoring
    builder.add_edge("revise", "critic")

    # Critic → verifier is handled via the conditional edge above (END → verifier)

    # Conditional: verifier → END or revise (strict mode)
    builder.add_conditional_edges(
        "verifier",
        route_after_verifier,
        {END: END, "revise": "revise"},
    )

    return builder.compile()


# ── Singleton ──────────────────────────────────────────────────────────────

_research_graph = None


def get_research_graph() -> StateGraph:
    """Get the compiled research graph (cached singleton)."""
    global _research_graph
    if _research_graph is None:
        _research_graph = build_research_graph()
    return _research_graph


# ── Convenience Runner ─────────────────────────────────────────────────────

def run_pipeline_graph(
    topic: str,
    merged_research: str = "",
    memory_context: str = "",
    mode: str = "langchain",
    max_critic_iterations: int = 3,
    strict_verification: bool = False,
) -> dict:
    """Run the research pipeline via LangGraph.

    Args:
        topic: The research topic.
        merged_research: Pre-computed research from parallel queue (optional).
            If empty, the graph will run planner + LLM research internally.
        memory_context: Optional ChromaDB context.
        mode: 'langchain' for analysis+writing (CrewAI not supported in graph).
        max_critic_iterations: Max critic loop iterations.
        strict_verification: If True, failed verification routes back to revise.

    Returns:
        Final state dict with keys: report, critique_iterations, critique_score,
        plan, sub_questions, verification_passed, verification_summary,
        verification_findings, error (if any).
    """
    graph = get_research_graph()

    initial_state: ResearchState = {
        "topic": topic,
        "memory_context": memory_context or "",
        "mode": mode,
        "plan": None,
        "sub_questions": [],
        "merged_research": merged_research or "",
        "report": "",
        "critique_score": None,
        "critique_passed": None,
        "critique_iterations": 0,
        "max_critic_iterations": max_critic_iterations,
        "last_critique": None,
        "verification_passed": None,
        "verification_summary": None,
        "verification_findings": [],
        "verification_iterations": 0,
        "max_verification_iterations": 1,
        "total_claims_checked": 0,
        "strict_verification": strict_verification,
        "error": None,
    }

    try:
        final_state = graph.invoke(initial_state)
        return {
            "report": final_state.get("report", ""),
            "critique_iterations": final_state.get("critique_iterations", 0),
            "critique_score": final_state.get("critique_score"),
            "critique_passed": final_state.get("critique_passed"),
            "plan": final_state.get("plan"),
            "sub_questions": final_state.get("sub_questions", []),
            "verification_passed": final_state.get("verification_passed"),
            "verification_summary": final_state.get("verification_summary"),
            "verification_findings": final_state.get("verification_findings", []),
            "total_claims_checked": final_state.get("total_claims_checked", 0),
            "error": None,
        }
    except Exception as e:
        import traceback
        tb = traceback.format_exc()
        print(f"❌ Graph pipeline failed: {e}\n{tb}")
        return {
            "report": "",
            "critique_iterations": 0,
            "critique_score": None,
            "critique_passed": None,
            "plan": None,
            "sub_questions": [],
            "error": str(e),
        }
