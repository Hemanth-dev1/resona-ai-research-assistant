"""LangGraph StateGraph for the research pipeline.

Formalizes the pipeline with typed state and conditional edges:

    START → [planner?] → analysis_writer → critic
        → (passed? → verifier | revise → critic → ...)
            verifier → (passed? → END | revise → critic → ...)

The research step (web search) and planning UI events are handled outside
the graph by server.py. If the graph receives pre-computed merged_research
(and thus planning already happened), it skips directly to analysis_writer.
"""

import re
from typing import Optional, TypedDict

from langgraph.graph import END, StateGraph

from schemas.models import CritiqueResult


# ── Graph State ────────────────────────────────────────────────────────────

class ResearchState(TypedDict):
    """Typed state that flows through graph nodes."""

    topic: str
    memory_context: str
    mode: str                       # 'langchain' (reserved for future providers)
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
    claim_verification_passed: Optional[bool] # claim verifier result (Step 5)
    claim_verification_summary: Optional[str]  # claim verifier summary
    unsupported_claims: list                  # list of unsupported claim dicts
    claim_verifier_iterations: int            # how many times claim verifier has run
    max_claim_verifier_iterations: int        # max claim verifier cycles (default 2)
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


def research_retry_node(state: ResearchState) -> dict:
    """When claim-verifier finds unsupported/fabricated claims, run targeted
    web searches for those claim topics to find REAL sources — instead of
    asking the Analyst to strip citations and write vague prose.

    This node:
    1. Extracts search queries from the unsupported claim text
    2. Runs DuckDuckGo web searches for each query
    3. Formats results as new Tracked Sources (with IDs continuing from
       the existing max source ID)
    4. Appends the new research to the existing merged_research
    5. Clears the unsupported claims so the next analysis_writer pass
       has fresh data

    Runs synchronously (DuckDuckGo calls are blocking but fast).
    """
    topic = state["topic"]
    merged = state["merged_research"]
    unsupported = state.get("unsupported_claims", [])

    if not unsupported:
        return {"merged_research": merged, "unsupported_claims": []}

    print(f"  🔍 Research Retry: Searching for sources on {len(unsupported)} unsupported claim(s)...")

    # Determine the starting source ID (continue from existing max)
    existing_ids = re.findall(r"\[S(\d+)\]", merged)
    next_id = max((int(x) for x in existing_ids), default=0) + 1

    new_sections: list[str] = []
    new_sources: list[str] = []

    for i, uc in enumerate(unsupported):
        claim_text = uc.get("claim_text", "")[:200]
        source_id = uc.get("source_id", "?")

        # Generate a focused search query from the claim text
        query = _claim_to_search_query(claim_text, topic)
        print(f"     [{source_id}] Searching: {query}")

        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=4))
        except ImportError:
            print(f"     ⚠️  duckduckgo_search not installed — skipping web search")
            continue
        except Exception as e:
            print(f"     ⚠️  Search failed for '{query}': {e}")
            continue

        if not results:
            print(f"     ⚠️  No results for '{query}'")
            continue

        # Format results with new source IDs
        for j, r in enumerate(results):
            sid = f"S{next_id}"
            title = r.get("title", "Untitled")
            url = r.get("href", "")
            body = r.get("body", "")
            new_sections.append(
                f"## Retry Finding: {title}\n"
                f"[{sid}] {title}\n"
                f"    URL: {url}\n"
                f"    {body[:500]}"
            )
            new_sources.append(
                f"- **[{sid}]** {title}\n"
                f"  URL: {url}\n"
                f"  {body[:200]}\n"
            )
            next_id += 1

    if new_sections:
        # Build the retry research block
        retry_block = (
            "\n\n---\n"
            "## Claim Retry Research\n"
            "The following sources were retrieved through targeted searches "
            "after claims in the previous report could not be verified.\n\n"
            + "\n\n---\n\n".join(new_sections)
            + "\n\n---\n\n"
            + "## Tracked Sources (Retry)\n\n"
            + "\n".join(new_sources)
            + "\n---\n"
        )
        merged += retry_block
        print(f"  ✅ Research Retry: Added {len(new_sections)} new sources for {len(unsupported)} claim(s)")
    else:
        print(f"  ⚠️  Research Retry: No new sources found — proceeding with existing research")
        # Append a note so the Analyst knows it tried
        merged += (
            "\n\n---\n"
            "## Claim Retry Research\n"
            "No additional sources could be retrieved for the following claims. "
            "If you cannot support them, set has_sufficient_evidence=false and "
            "explain in gap_reason what specific information is missing.\n"
            "---\n"
        )

    return {"merged_research": merged, "unsupported_claims": []}


def _claim_to_search_query(claim_text: str, topic: str) -> str:
    """Extract a focused search query from an unsupported claim.

    Strips citation tags, removes hedging language, and keeps key
    noun phrases.  Falls back to topic + first few keywords.
    """
    # Remove [S#] tags
    text = re.sub(r"\[S\d+\]", "", claim_text)
    # Remove URLs
    text = re.sub(r"https?://\S+", "", text)
    # Remove common hedge/filler phrases
    text = re.sub(r"\b(it is|the|a|an|this|that|these|those)\b", "", text, flags=re.IGNORECASE)
    # Take first 8 significant words
    words = [w for w in text.split() if len(w) > 2][:8]
    if not words:
        return topic
    return " ".join(words)


def analysis_writer_node(state: ResearchState) -> dict:
    """Analyze merged research and write the report.

    Uses the capable model via LangChain chain.
    (Only 'langchain' mode is supported inside the graph.)
    """
    topic = state["topic"]
    merged = state["merged_research"]

    if not merged or len(merged) < 50:
        # No external research — use a clearly distinct placeholder that will NOT
        # be mistaken for real source material.  The placeholder must:
        # 1. NOT use ## Tracked Sources (the header the Analyst is trained to scan for [S#] IDs)
        # 2. NOT use [S#] patterns that the LLM could mimic
        # 3. Explicitly instruct the Analyst to produce ZERO citation tags
        merged = (
            "⚠️ NO SOURCES AVAILABLE — ZERO SOURCES WERE RETRIEVED FOR THIS TOPIC.\n"
            "DO NOT USE ANY [S#] CITATION TAGS. DO NOT INVENT SOURCES.\n"
            "If the topic cannot be answered from available knowledge, "
            "set has_sufficient_evidence=false and explain the gap.\n"
            "Do not pad the report with repeated statements about missing data — "
            "state the insufficiency ONCE and stop.\n"
        )
        print(f"  ⚠️  Graph: No search results for '{topic}' — Analyst will produce zero-citation report")

    # ── DEBUG: Log the actual sources being passed to the Analyst ──────
    tracked_sources_match = re.search(
        r"## Tracked Sources\s*\n(.*?)(?:\n##|\Z)", merged, re.DOTALL
    )
    if tracked_sources_match:
        sources_raw = tracked_sources_match.group(1).strip()
        source_ids = re.findall(r"\[S(\d+)\]", sources_raw)
        if source_ids:
            print(f"  📚 Analyst receives {len(source_ids)} real sources: S{', S'.join(source_ids)}")
            # Print first 3 source entries for verification
            for line in sources_raw.split("\n")[:12]:
                stripped = line.strip()
                if stripped:
                    print(f"     {stripped[:120]}")
        else:
            print(f"  ⚠️  Tracked Sources section found but no [S#] IDs parsed")
            print(f"     First 200 chars: {sources_raw[:200]}")
    else:
        print(f"  ⚠️  No Tracked Sources section found in merged research ({len(merged)} chars)")
        print(f"     First 100 chars: {merged[:100]!r}")

    print(f"  🧠 Graph: Analyzing + writing report...")
    from chain.chain import run_analysis_writing
    result = run_analysis_writing(topic, merged)
    report = result.get("report", "")
    # Clear unsupported claims so they don't accumulate on subsequent retries
    return {"report": report, "unsupported_claims": []}


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


# ── Claim Verifier Node (Step 5) ────────────────────────────────────────────

def claim_verifier_node(state: ResearchState) -> dict:
    """Verify each cited claim against its source using a fast/cheap model.

    Parses [S#] tags from the report, extracts the surrounding claim text,
    and sends each claim+source pair to a fast LLM for verification.
    If ANY claims are unsupported, routes to research_retry_node which runs
    targeted web searches for those claims to find REAL sources — instead of
    asking the Analyst to strip citations and write vague prose.
    """
    topic = state["topic"]
    report = state["report"]
    merged_research = state.get("merged_research", "")
    iteration = state.get("claim_verifier_iterations", 0) + 1
    max_iter = state.get("max_claim_verifier_iterations", 2)

    print(f"  🔍 Graph: Claim verification (attempt {iteration}/{max_iter})...")

    from chain.chain import run_claim_verification
    result = run_claim_verification(topic, report, merged_research)

    passed = result.get("passed", True)
    unsupported = result.get("unsupported_claims", [])
    summary = result.get("summary", "Claim verification complete.")
    claims_checked = result.get("claims_checked", 0)

    if not passed:
        print(f"  ❌ Graph: {len(unsupported)} unsupported claim(s) found")
        for uc in unsupported:
            print(f"     - {uc.get('source_id', '?' )}: {uc.get('claim_text', '')[:80]}...")
            print(f"       Reason: {uc.get('reason', '')[:80]}")

    return {
        "claim_verification_passed": passed,
        "claim_verification_summary": summary,
        "unsupported_claims": unsupported,
        "claim_verifier_iterations": iteration,
        "total_claims_checked": claims_checked,
    }


def route_after_claim_verifier(state: ResearchState) -> str:
    """Decide next step after claim verification.

    If claims are unsupported and we haven't hit max iterations, route to
    research_retry_node which runs targeted web searches for the unsupported
    claims to find REAL sources (instead of asking the Analyst to strip
    citations and write vaguely).

    After max iterations (one retry cycle), route DIRECTLY to END so the
    HARD GATE in run_pipeline_graph() appends the "Could Not Verify" warning.
    We skip critic + verifier entirely because the report is known to have
    fabricated citations — running critic/verifier on it wastes time and
    tokens, and the user gets the partial report faster.
    """
    passed = state.get("claim_verification_passed", True)
    iterations = state.get("claim_verifier_iterations", 0)
    max_iter = state.get("max_claim_verifier_iterations", 2)
    unsupported = state.get("unsupported_claims", [])

    if not passed and iterations < max_iter and unsupported:
        print(f"  🔄 Graph: Routing to research_retry for {len(unsupported)} unsupported claim(s) (attempt {iterations}/{max_iter})")
        return "research_retry"

    if not passed:
        print(f"  ⚠️  Graph: Max claim verifier iterations ({max_iter}) reached — {len(unsupported)} unresolved claim(s). Routing directly to END (skipping critic+verifier)")
    else:
        print(f"  ✅ Graph: All cited claims verified — proceeding to critic")
        return "critic"

    # ── After retry exhausted: skip critic+verifier entirely, go straight to END
    #    so the HARD GATE in run_pipeline_graph() appends the warning and returns.
    return END


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
    max_iter = state.get("max_verification_iterations", 2)

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
    max_iter = state.get("max_verification_iterations", 2)

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
    builder.add_node("research_retry", research_retry_node)
    builder.add_node("claim_verifier", claim_verifier_node)
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

    # analysis_writer → claim_verifier (Step 5: cheap claim check BEFORE critic)
    builder.add_edge("analysis_writer", "claim_verifier")

    # Conditional: claim_verifier → research_retry (search for unsupported claims)
    #               or critic (proceed if all claims verified)
    #               or END (max retries exhausted — Could-Not-Verify)
    builder.add_conditional_edges(
        "claim_verifier",
        route_after_claim_verifier,
        {
            "research_retry": "research_retry",
            "critic": "critic",
            END: END,
        },
    )

    # research_retry → analysis_writer (re-analyze with new sources)
    builder.add_edge("research_retry", "analysis_writer")

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
    strict_verification: bool = True,
) -> dict:
    """Run the research pipeline via LangGraph.

    Args:
        topic: The research topic.
        merged_research: Pre-computed research from parallel queue (optional).
            If empty, the graph will run planner + LLM research internally.
        memory_context: Optional ChromaDB context.
        mode: 'langchain' for analysis+writing.
        strict_verification: If True, failed verification routes back to revise.

    Note: max_critic_iterations and max_verification_iterations are now
    hardcoded to 1 in the state dict.  All retry loops are capped at 1
    attempt to prevent fabrication exposure and latency from runaway
    loops on reports with no real source material.

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
        "max_critic_iterations": 1,
        "last_critique": None,
        "verification_passed": None,
        "verification_summary": None,
        "verification_findings": [],
        "verification_iterations": 0,
        "max_verification_iterations": 1,
        "total_claims_checked": 0,
        "strict_verification": strict_verification,
        "claim_verification_passed": None,
        "claim_verification_summary": None,
        "unsupported_claims": [],
        "claim_verifier_iterations": 0,
        "max_claim_verifier_iterations": 2,
        "error": None,
    }

    try:
        final_state = graph.invoke(initial_state)

        # ── HARD GATE: If claim_verifier still has unresolved unsupported
        #    claims after max iterations, append a transparent warning to
        #    the partial report instead of discarding it entirely.  The user
        #    gets the verified sections plus an explicit "could not verify"
        #    note for the problematic claims.
        unsupported = final_state.get("unsupported_claims", [])
        if unsupported:
            print(f"  🚫 HARD GATE: {len(unsupported)} unresolved unsupported claim(s) — appending warning")
            report = final_state.get("report", "") or ""
            warning = (
                "\n\n---\n"
                "## ⚠️ Could Not Verify\n"
                "The following claims could not be verified against reliable sources. "
                "The rest of the report above has passed fact-checking.\n"
                f"*Topic:* {topic}\n"
                f"*Reason:* Insufficient reliable sources to confirm these statements.\n\n"
            )
            for uc in unsupported:
                sid = uc.get("source_id", "?")
                claim = uc.get("claim_text", "")[:200]
                reason = uc.get("reason", "")[:200]
                warning += f"- **[{sid}]** \"{claim}\" — {reason}\n"
            warning += (
                "\n*These sections were removed from the verified report above. "
                "Try running a more specific search query for this topic.*\n"
            )
            report += warning
            return {
                "report": report,
                "critique_iterations": final_state.get("critique_iterations", 0),
                "critique_score": final_state.get("critique_score"),
                "critique_passed": final_state.get("critique_passed"),
                "plan": final_state.get("plan"),
                "sub_questions": final_state.get("sub_questions", []),
                "verification_passed": False,
                "verification_summary": warning.strip(),
                "verification_findings": unsupported,
                "total_claims_checked": final_state.get("total_claims_checked", 0),
                "error": None,  # Not an error — partial report is returned
            }

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
