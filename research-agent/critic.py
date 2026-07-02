"""Self-correcting critic loop for autonomous quality control.

The Quality Editor is upgraded from a simple reviewer into a scoring critic (0-10).
If the score is below a configurable threshold (default: 7), specific feedback is
sent back to the Writer agent and the report is regenerated with improvements.
Max 3 iterations. Tracks loop count and iteration history.

Flow:
    Writer produces report → Critic scores (0-10)
    ↓
    Score ≥ threshold? → ✅ Accept report
    ↓ (No, < max iterations)
    Critic sends specific feedback → Writer revises → Critic re-scores

Hedge-Phrase Gate:
    Before the LLM critic, a regex-based pre-check scans for hedging language
    (e.g. "further research is needed"). If found, the LLM critic is skipped
    entirely and the section auto-fails with the specific hedge phrase, so the
    retry prompt can say "you wrote a hedge instead of a finding — search more
    specifically."
"""

import json
import os
import re
from typing import List, Optional, Tuple

from langsmith import traceable
from tenacity import stop_after_attempt, wait_exponential

from retry_utils import default_retry
from schemas.models import CritiqueResult, CritiqueDimension, DimensionScore


# Configuration from environment
CRITIC_THRESHOLD = int(os.getenv("QUALITY_THRESHOLD") or os.getenv("RESONA_CRITIC_THRESHOLD", "7"))
MAX_CRITIC_ITERATIONS = int(os.getenv("RESONA_MAX_CRITIC_ITERATIONS", "3"))


# ── Hedge-Phrase Gate ─────────────────────────────────────────────────────
# Regex patterns that detect hedging / non-committal language in report sections.
# If ANY match, the LLM critic call is skipped entirely and the section
# auto-fails so the writer must rewrite with concrete findings.

HEDGE_PATTERNS: list[re.Pattern] = [
    re.compile(r"further\s+research\s+is\s+needed", re.IGNORECASE),
    re.compile(r"more\s+research\s+(is\s+)?needed", re.IGNORECASE),
    re.compile(r"requires?\s+(additional|further)\s+(research|investigation)", re.IGNORECASE),
    re.compile(r"it\s+is\s+(unclear|not\s+clear)\s+(whether|if)", re.IGNORECASE),
    re.compile(r"needs\s+further\s+(study|research|investigation|analysis)", re.IGNORECASE),
    re.compile(r"remains\s+to\s+be\s+(seen|determined|established)", re.IGNORECASE),
    re.compile(r"may\s+require\s+additional\s+(research|study|analysis)", re.IGNORECASE),
    re.compile(r"it\s+is\s+important\s+to\s+note\s+that\s+(it\s+is\s+(unclear|not\s+clear)|may|might|could|remains)", re.IGNORECASE),  # hedge only when followed by vague language
    re.compile(r"has\s+the\s+potential\s+to\s+(be|become|impact)", re.IGNORECASE),  # vague potential
    re.compile(r"could\s+(potentially|possibly)\s+(lead|result|affect|impact)", re.IGNORECASE),
    re.compile(r"further\s+(investigation|study|analysis)\s+(is\s+)?(needed|warranted|called\s+for)", re.IGNORECASE),
    re.compile(r"it\s+remains\s+(unclear|unknown|uncertain)\s+(whether|if|what|how)", re.IGNORECASE),
    re.compile(r"the\s+(impact|effect|implication)s?\s+(of\s+.+?\s+)(are|is)\s+not\s+(yet\s+)?(fully\s+)?understood", re.IGNORECASE),
    re.compile(r"more\s+(work|research|data|evidence)\s+(is\s+)?(needed|required|called\s+for)", re.IGNORECASE),
]


# ── Critic Prompt Templates ────────────────────────────────────────────────

CRITIC_SYSTEM_PROMPT = """You are a Senior Quality Editor & Fact-Checker at a prestigious publishing house.
Your eye for detail is legendary. You evaluate research reports on five dimensions,
each scored from 0 (worst) to 10 (best):

1. **factual_accuracy** — Are claims supported? Any hallucinations or unsupported statements?
2. **structure** — Does the report follow the required section order? Is it well-organized?
3. **clarity** — Is the writing clear, professional, and accessible? Any confusing passages?
4. **completeness** — Are all required sections present and adequately developed?
5. **citation_quality** — Are sources properly cited? Are the citations credible?

Respond with a JSON object containing:
- overall_score (int, 0-10)
- dimensions (list of objects with: dimension, score, feedback, suggestion)
- summary (str): one-paragraph summary of the critique
"""

CRITIC_HUMAN_PROMPT = """Evaluate this research report on the topic: {topic}

--- REPORT ---
{report}

Return a JSON object with: overall_score, dimensions (array of {{dimension, score, feedback, suggestion}}), and summary.
Score each dimension 0-10. The overall_score should be the weighted average.
"""

CRITIC_FEEDBACK_PROMPT = """You are a Technical Content Writer revising a report based on editor feedback.

Original topic: {topic}

Previous version was scored: {overall_score}/10

Editor feedback:
{dimensions_feedback}

Please revise the report to address ALL the feedback above. Maintain the same 8-section
structure (Executive Summary, Introduction, Detailed Analysis, Key Insights, Challenges,
Future Outlook, Sources) but improve the quality based on the critique.

Return the FULL revised report in Markdown format.
"""


def _format_dimensions_feedback(dimensions: List[DimensionScore]) -> str:
    """Format dimension scores into readable feedback for the writer.

    Args:
        dimensions: List of dimension scores with feedback.

    Returns:
        Formatted feedback string.
    """
    lines = []
    for d in dimensions:
        status = "✅" if d.score >= CRITIC_THRESHOLD else "❌"
        lines.append(f"{status} **{d.dimension.value.replace('_', ' ').title()}**: {d.score}/10")
        lines.append(f"   Feedback: {d.feedback}")
        if d.suggestion:
            lines.append(f"   Suggestion: {d.suggestion}")
        lines.append("")
    return "\n".join(lines)


def detect_hedge_phrases(text: str) -> list[dict]:
    """Scan report text for hedge phrases BEFORE calling the LLM critic.

    A hard gate: if any hedge is found, skip the LLM critic entirely and
    auto-fail with the specific hedge phrase so the writer can be instructed
    to replace it with a concrete finding.

    Args:
        text: The full report text to scan.

    Returns:
        List of dicts with keys "phrase" (the matched text), "pattern" (regex
        pattern name), and "context" (surrounding text snippet). Empty list
        means no hedge phrases found — proceed to LLM critic.
    """
    findings: list[dict] = []
    for pattern in HEDGE_PATTERNS:
        for match in pattern.finditer(text):
            start = max(0, match.start() - 60)
            end = min(len(text), match.end() + 60)
            context = text[start:end].replace("\n", " ").strip()
            findings.append({
                "phrase": match.group().strip(),
                "pattern": pattern.pattern[:60],  # first 60 chars of pattern
                "context": context,
            })
    return findings


@default_retry
def _run_llm_call(system_prompt: str, human_prompt: str) -> str:
    """Execute an LLM call for the critic loop.

    Uses the capable (70B) model for quality scoring and revision.
    Retries with exponential backoff on failure via @default_retry.

    Args:
        system_prompt: System message for the LLM.
        human_prompt: Human message for the LLM.

    Returns:
        The LLM response text.
    """
    from llm_config import get_capable_llm

    llm = get_capable_llm(temperature=0.2)  # Low temperature for consistent scoring
    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
    result = llm.invoke(messages)
    return result.content if hasattr(result, "content") else str(result)


@traceable(name="critic_score", run_type="chain")
def score_report(topic: str, report: str) -> CritiqueResult:
    """Score a report on quality dimensions.

    Hedge-Phrase Gate: runs FIRST, before any LLM call. If the report contains
    hedging language (e.g. "further research is needed"), the LLM critic is
    skipped entirely and the report auto-fails with the specific hedge phrase.
    This forces the writer to replace hedges with concrete findings.

    Args:
        topic: The research topic.
        report: The report content to evaluate.

    Returns:
        CritiqueResult with scores and feedback.
    """
    try:
        # ── HEDGE-PHRASE GATE (runs before LLM) ────────────────────────────
        hedge_hits = detect_hedge_phrases(report)
        if hedge_hits:
            # Build a clear failure for each hedge found
            details = []
            for h in hedge_hits:
                details.append(
                    f"- Hedge phrase: '{h['phrase']}' in context: "
                    f"'...{h['context']}...'"
                )
            hedge_detail = "\n".join(details)
            msg = (
                f"Report contains {len(hedge_hits)} hedge phrase(s) instead of "
                f"concrete findings. You wrote a hedge instead of a finding — "
                f"search more specifically.\n\n{hedge_detail}"
            )
            print(f"  🚫 Hedge gate fired: {len(hedge_hits)} hedge phrase(s) found")

            return CritiqueResult(
                topic=topic,
                overall_score=0,
                dimensions=[
                    DimensionScore(
                        dimension=CritiqueDimension.FACTUAL_ACCURACY,
                        score=0,
                        feedback=msg,
                        suggestion="Replace every hedge phrase with a specific, "
                        "sourced factual claim. If sources don't answer the "
                        "question, state that explicitly rather than writing "
                        "a hedge.",
                    )
                ],
                passed=False,
                iteration=1,
                summary=f"Auto-failed: {len(hedge_hits)} hedge phrase(s) found. "
                f"First: '{hedge_hits[0]['phrase']}'.",
            )

        # ── LLM CRITIC CALL (only if hedge gate passed) ────────────────────
        human = CRITIC_HUMAN_PROMPT.format(topic=topic, report=report)
        response = _run_llm_call(CRITIC_SYSTEM_PROMPT, human)

        # Extract JSON from potential markdown wrappers
        json_match = re.search(r"\{.*\}", response, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
        else:
            data = json.loads(response)

        dimensions = []
        for d in data.get("dimensions", []):
            dim_name = d.get("dimension", "overall")
            try:
                dimension_enum = CritiqueDimension(dim_name)
            except ValueError:
                dimension_enum = CritiqueDimension.OVERALL

            dimensions.append(DimensionScore(
                dimension=dimension_enum,
                score=min(10, max(0, int(d.get("score", 5)))),
                feedback=d.get("feedback", "No specific feedback."),
                suggestion=d.get("suggestion"),
            ))

        overall = min(10, max(0, int(data.get("overall_score", 5))))
        passed = overall >= CRITIC_THRESHOLD

        return CritiqueResult(
            topic=topic,
            overall_score=overall,
            dimensions=dimensions,
            passed=passed,
            iteration=1,  # Will be updated by the loop
            summary=data.get("summary", f"Overall score: {overall}/10. {'Passed' if passed else 'Needs improvement'}."),
        )

    except Exception as e:
        return CritiqueResult(
            topic=topic,
            overall_score=5,
            dimensions=[
                DimensionScore(
                    dimension=CritiqueDimension.OVERALL,
                    score=5,
                    feedback=f"Could not parse critic response: {e}",
                )
            ],
            passed=False,
            iteration=1,
            summary=f"Critique error: {e}. Defaulting to score 5.",
        )


@traceable(name="critic_loop", run_type="chain")
def run_critic_loop(
    topic: str,
    report: str,
    max_iterations: int = MAX_CRITIC_ITERATIONS,
    threshold: int = CRITIC_THRESHOLD,
) -> Tuple[str, List[CritiqueResult]]:
    """Run the self-correcting critic loop.

    Scores the report, and if below threshold, sends feedback to the writer
    to regenerate. Repeats up to max_iterations times.

    Args:
        topic: The research topic.
        report: The initial report content.
        max_iterations: Maximum number of critic iterations (default: 3).
        threshold: Quality threshold score (default: 7).

    Returns:
        Tuple of (final_report_text, list_of_critique_results).
    """
    critiques: List[CritiqueResult] = []
    current_report = report

    for iteration in range(1, max_iterations + 1):
        print(f"  📝 Critic iteration {iteration}/{max_iterations}...")

        try:
            critique = score_report(topic, current_report)
            critique.iteration = iteration
            critiques.append(critique)

            print(f"  📊 Score: {critique.overall_score}/10 (threshold: {threshold})")

            if critique.passed:
                print(f"  ✅ Report passed quality check (score {critique.overall_score} ≥ {threshold})")
                break

            if iteration >= max_iterations:
                print(f"  ⚠️  Max iterations ({max_iterations}) reached. Using best effort report.")
                break

            # Send feedback to writer for revision
            print(f"  🔄 Revising report based on critic feedback (iteration {iteration} → {iteration + 1})...")
            current_report = _revise_report(topic, current_report, critique)

        except Exception as e:
            print(f"  ⚠️  Critic iteration {iteration} failed: {e}")
            # Continue with the current report rather than crashing
            critiques.append(CritiqueResult(
                topic=topic,
                overall_score=5,
                dimensions=[DimensionScore(dimension=CritiqueDimension.OVERALL, score=5, feedback=str(e))],
                passed=True,  # Accept to avoid infinite loop
                iteration=iteration,
                summary=f"Critic loop error on iteration {iteration}: {e}",
            ))
            break

    return current_report, critiques


def _revise_report(topic: str, report: str, critique: CritiqueResult) -> str:
    """Revise a report based on critic feedback.

    Args:
        topic: The research topic.
        report: The current report content.
        critique: The critique with scores and feedback.

    Returns:
        The revised report content, or the original if revision fails.
    """
    try:
        dimensions_feedback = _format_dimensions_feedback(critique.dimensions)

        human = CRITIC_FEEDBACK_PROMPT.format(
            topic=topic,
            overall_score=critique.overall_score,
            dimensions_feedback=dimensions_feedback,
        )

        response = _run_llm_call(
            "You are an award-winning Technical Content Writer revising a report based on editor feedback.",
            human,
        )

        return response

    except Exception as e:
        print(f"  ⚠️  Report revision failed: {e}. Keeping previous version.")
        return report  # Return original unchanged on error
