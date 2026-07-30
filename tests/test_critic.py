"""Mocked tests for critic.py — hedge detection and scoring logic.

Tests run without any LLM calls or API keys.
"""

import unittest
from unittest.mock import patch, MagicMock
import os
import json


class TestHedgeDetection(unittest.TestCase):
    """Test the hedge-phrase gate that runs before the LLM critic."""

    def setUp(self):
        # Ensure env vars don't interfere
        self._env_patch = patch.dict(os.environ, {}, clear=True)
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_detect_common_hedges(self):
        """Common hedge phrases like 'further research is needed' are detected."""
        from critic import detect_hedge_phrases
        text = "This area requires further research is needed to draw conclusions."
        results = detect_hedge_phrases(text)
        self.assertGreater(len(results), 0)
        self.assertIn("further research is needed", results[0]["phrase"].lower())

    def test_no_hedges_returns_empty(self):
        """Report without hedge phrases returns empty list."""
        from critic import detect_hedge_phrases
        text = "The market reached $5B in 2025. Three major players control 80% share."
        results = detect_hedge_phrases(text)
        self.assertEqual(len(results), 0)

    def test_multiple_hedges_found(self):
        """Report with multiple hedge phrases returns all of them."""
        from critic import detect_hedge_phrases
        text = "This remains to be seen. Further investigation is needed. It is unclear whether."
        results = detect_hedge_phrases(text)
        self.assertGreaterEqual(len(results), 2)

    def test_hedge_context_includes_surrounding_text(self):
        """Each hedge result includes surrounding context."""
        from critic import detect_hedge_phrases
        text = "Some introductory text. This area requires further research is needed before we can conclude. More text after."
        results = detect_hedge_phrases(text)
        self.assertIn("context", results[0])
        self.assertGreater(len(results[0]["context"]), len(results[0]["phrase"]))

    def test_hedge_gate_skips_llm_and_returns_score_zero(self):
        """When hedge phrases found, score_report returns score 0 and passed=False."""
        from critic import score_report
        with patch("critic._run_llm_call") as mock_llm:
            text = "This topic requires further research is needed."
            result = score_report("test topic", text)
            self.assertFalse(result.passed)
            self.assertEqual(result.overall_score, 0)
            mock_llm.assert_not_called()


class TestCriticScoreBoundaries(unittest.TestCase):
    """Test the score/decision boundary around QUALITY_THRESHOLD."""

    def setUp(self):
        self._env_patch = patch.dict(os.environ, {"QUALITY_THRESHOLD": "7"}, clear=True)
        self._env_patch.start()

    def tearDown(self):
        self._env_patch.stop()

    def test_score_above_threshold_passes(self):
        """Score >= threshold returns passed=True."""
        from critic import score_report
        mock_result = json.dumps({
            "overall_score": 8,
            "dimensions": [
                {"dimension": "factual_accuracy", "score": 8, "feedback": "Good."}
            ],
            "summary": "Good report."
        })
        with patch("critic._run_llm_call", return_value=mock_result):
            result = score_report("test", "## Report\nContent.")
            self.assertTrue(result.passed)
            self.assertEqual(result.overall_score, 8)

    def test_score_below_threshold_fails(self):
        """Score < threshold returns passed=False."""
        from critic import score_report
        mock_result = json.dumps({
            "overall_score": 5,
            "dimensions": [
                {"dimension": "factual_accuracy", "score": 5, "feedback": "Needs work."}
            ],
            "summary": "Below threshold."
        })
        with patch("critic._run_llm_call", return_value=mock_result):
            result = score_report("test", "## Report\nWeak content.")
            self.assertFalse(result.passed)
            self.assertEqual(result.overall_score, 5)

    def test_score_exactly_at_threshold_passes(self):
        """Score exactly at threshold (7) should pass."""
        from critic import score_report
        mock_result = json.dumps({
            "overall_score": 7,
            "dimensions": [
                {"dimension": "factual_accuracy", "score": 7, "feedback": "Acceptable."}
            ],
            "summary": "At threshold."
        })
        with patch("critic._run_llm_call", return_value=mock_result):
            result = score_report("test", "## Report\nOK content.")
            self.assertTrue(result.passed)

    def test_critic_loop_accepts_after_revision(self):
        """run_critic_loop should accept on first pass if score is good enough."""
        from critic import run_critic_loop
        with patch("critic.score_report") as mock_score:
            from schemas.models import CritiqueResult, CritiqueDimension, DimensionScore
            mock_score.return_value = CritiqueResult(
                topic="test",
                overall_score=9,
                dimensions=[DimensionScore(dimension=CritiqueDimension.OVERALL, score=9, feedback="Great.")],
                passed=True,
                iteration=1,
                summary="Accepted."
            )
            final_report, critiques = run_critic_loop("test", "## Report\nGreat content.", max_iterations=3)
            self.assertEqual(len(critiques), 1)
            self.assertTrue(critiques[0].passed)
