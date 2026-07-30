"""RAGAS evaluation for the RAG memory layer.

Runs RAGAS metrics (faithfulness, answer_relevancy, context_recall) on each
research run that uses the RAG layer (LangChain mode). Scores are stored
alongside the ChromaDB report metadata and exposed via /api/stats.

Metrics:
    - faithfulness: Are the claims in the answer supported by the retrieved context?
    - answer_relevancy: How relevant is the answer to the question?
    - context_recall: Was all relevant context retrieved from ChromaDB?

Usage:
    from ragas_eval import evaluate_rag, get_average_scores
    scores = evaluate_rag(query, answer, contexts)
    avg = get_average_scores()
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

from langsmith import traceable

# RAGAS scores storage path
RAGAS_SCORES_PATH = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "output", "ragas_scores.json"
)


def _load_scores() -> List[dict]:
    """Load historical RAGAS scores from disk.

    Returns:
        List of score dicts, each containing topic, timestamp, and metrics.
    """
    path = Path(RAGAS_SCORES_PATH)
    if not path.exists():
        return []
    try:
        with open(path, "r") as f:
            return json.load(f)
    except (json.JSONDecodeError, IOError):
        return []


def _save_scores(scores: List[dict]) -> None:
    """Save RAGAS scores to disk.

    Args:
        scores: List of score dicts to persist.
    """
    path = Path(RAGAS_SCORES_PATH)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(scores, f, indent=2)


def _compute_faithfulness(answer: str, contexts: List[str]) -> float:
    """Compute a simplified faithfulness score.

    Checks how many claims in the answer are supported by the retrieved context.
    This is a lightweight approximation — for full RAGAS, use the ragas library.

    Args:
        answer: The generated answer text.
        contexts: Retrieved context chunks.

    Returns:
        Faithfulness score between 0.0 and 1.0.
    """
    if not contexts or not answer:
        return 0.0

    try:
        from llm_config import get_llm

        llm = get_llm(temperature=0.1)

        context_text = "\n\n".join(contexts)
        prompt = [
            (
                "system",
                "You evaluate faithfulness. Given a context and an answer, "
                "score how well the answer is supported by the context from 0.0 to 1.0. "
                "Return ONLY a single float number. No explanation.",
            ),
            (
                "human",
                f"Context:\n{context_text}\n\n"
                f"Answer:\n{answer}\n\n"
                f"Faithfulness score (0.0-1.0):",
            ),
        ]

        result = llm.invoke(prompt)
        response = result.content if hasattr(result, "content") else str(result)

        # Extract float from response
        import re
        match = re.search(r"(\d+\.?\d*)", response.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5

    except Exception:
        return 0.5  # Default mid-range on error


def _compute_answer_relevancy(question: str, answer: str) -> float:
    """Compute a simplified answer relevancy score.

    Measures how relevant the answer is to the original question.

    Args:
        question: The original query/question.
        answer: The generated answer.

    Returns:
        Relevancy score between 0.0 and 1.0.
    """
    if not question or not answer:
        return 0.0

    try:
        from llm_config import get_llm

        llm = get_llm(temperature=0.1)

        prompt = [
            (
                "system",
                "You evaluate answer relevancy. Given a question and an answer, "
                "score how relevant and directly responsive the answer is from 0.0 to 1.0. "
                "Return ONLY a single float number. No explanation.",
            ),
            (
                "human",
                f"Question: {question}\n\n"
                f"Answer: {answer[:500]}...\n\n"
                f"Relevancy score (0.0-1.0):",
            ),
        ]

        result = llm.invoke(prompt)
        response = result.content if hasattr(result, "content") else str(result)

        import re
        match = re.search(r"(\d+\.?\d*)", response.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5

    except Exception:
        return 0.5


def _compute_context_recall(query: str, contexts: List[str]) -> float:
    """Compute a simplified context recall score.

    Measures whether all relevant information was retrieved from the context.

    Args:
        query: The search query.
        contexts: Retrieved context chunks.

    Returns:
        Recall score between 0.0 and 1.0.
    """
    if not query or not contexts:
        return 0.0

    try:
        from llm_config import get_llm

        llm = get_llm(temperature=0.1)

        context_text = "\n\n".join(contexts)
        prompt = [
            (
                "system",
                "You evaluate context recall. Given a query and retrieved context chunks, "
                "score how well the context covers the information needed to answer the query "
                "from 0.0 to 1.0. Return ONLY a single float number. No explanation.",
            ),
            (
                "human",
                f"Query: {query}\n\n"
                f"Retrieved Context:\n{context_text}\n\n"
                f"Context recall score (0.0-1.0):",
            ),
        ]

        result = llm.invoke(prompt)
        response = result.content if hasattr(result, "content") else str(result)

        import re
        match = re.search(r"(\d+\.?\d*)", response.strip())
        if match:
            score = float(match.group(1))
            return max(0.0, min(1.0, score))
        return 0.5

    except Exception:
        return 0.5


@traceable(name="ragas_evaluate", run_type="chain")
def evaluate_rag(
    query: str,
    answer: str,
    contexts: Optional[List[str]] = None,
    topic: Optional[str] = None,
) -> Dict[str, float]:
    """Run RAGAS evaluation metrics on a research run.

    Computes faithfulness, answer_relevancy, and context_recall scores.

    Args:
        query: The research topic/query.
        answer: The generated report content.
        contexts: Retrieved context chunks from ChromaDB. If None, skips
                  context-dependent metrics.
        topic: Optional topic name for storage.

    Returns:
        Dict with keys: faithfulness, answer_relevancy, context_recall,
        and overall (average of available metrics).
    """
    contexts = contexts or []
    has_context = len(contexts) > 0

    # Compute metrics
    scores = {"topic": topic or query}

    if has_context:
        scores["faithfulness"] = _compute_faithfulness(answer, contexts)
        scores["context_recall"] = _compute_context_recall(query, contexts)
    else:
        scores["faithfulness"] = 0.0
        scores["context_recall"] = 0.0

    scores["answer_relevancy"] = _compute_answer_relevancy(query, answer)

    # Overall = average of available metrics
    available = [v for k, v in scores.items() if k not in ("topic",) and isinstance(v, (int, float))]
    scores["overall"] = round(sum(available) / len(available), 4) if available else 0.0

    # Round for readability
    for k in scores:
        if isinstance(scores[k], float):
            scores[k] = round(scores[k], 4)

    # Store with timestamp
    scores["timestamp"] = datetime.now().isoformat()
    all_scores = _load_scores()
    all_scores.append(scores)
    _save_scores(all_scores)

    print(f"  📊 RAGAS: faithfulness={scores.get('faithfulness', 'N/A')}, "
          f"relevancy={scores['answer_relevancy']}, "
          f"recall={scores.get('context_recall', 'N/A')}, "
          f"overall={scores['overall']}")

    return scores


def get_average_scores() -> Dict[str, float]:
    """Get average RAGAS scores across all stored evaluations.

    Returns:
        Dict with average scores for each metric and total count.
    """
    all_scores = _load_scores()

    if not all_scores:
        return {
            "faithfulness": 0.0,
            "answer_relevancy": 0.0,
            "context_recall": 0.0,
            "overall": 0.0,
            "total_evaluations": 0,
        }

    totals = {"faithfulness": 0.0, "answer_relevancy": 0.0, "context_recall": 0.0, "overall": 0.0}
    count = len(all_scores)

    for s in all_scores:
        for key in totals:
            if key in s:
                totals[key] += s[key]

    averages = {k: round(v / count, 4) for k, v in totals.items()}
    averages["total_evaluations"] = count
    return averages
