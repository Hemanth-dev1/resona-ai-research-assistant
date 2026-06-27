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


@default_retry
def _run_llm_call(system_prompt: str, human_prompt: str) -> str:
    """Execute an LLM call for the critic loop.

    Retries with exponential backoff on failure via @default_retry.

    Args:
        system_prompt: System message for the LLM.
        human_prompt: Human message for the LLM.

    Returns:
        The LLM response text.
    """
    from llm_config import get_llm

    llm = get_llm(temperature=0.2)  # Low temperature for consistent scoring
    messages = [
        ("system", system_prompt),
        ("human", human_prompt),
    ]
    result = llm.invoke(messages)
    return result.content if hasattr(result, "content") else str(result)


@traceable(name="critic_score", run_type="chain")
def score_report(topic: str, report: str) -> CritiqueResult:
    """Score a report on quality dimensions.

    Args:
        topic: The research topic.
        report: The report content to evaluate.

    Returns:
        CritiqueResult with scores and feedback.
    """
    try:
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
