"""Mocked tests for chain/chain.py — run_planner and analysis/writer chains.

Tests verify parsed output shape (sub_questions list, report sections)
without making any real LLM calls. Patches target the DEFINITION module,
not the import site (since imports are lazy inside function bodies).
"""

import json
import unittest
from unittest.mock import patch, MagicMock


class TestPlanner(unittest.TestCase):
    """Test the run_planner function with mocked LLM calls."""

    def test_run_planner_returns_dict_with_sub_questions(self):
        """run_planner returns a dict with a sub_questions list."""
        from chain.chain import run_planner

        mock_response = json.dumps({
            "topic": "Quantum computing",
            "sub_questions": [
                {
                    "question": "What are the latest breakthroughs in quantum computing?",
                    "search_query": "quantum computing breakthroughs 2025 2026",
                    "rationale": "Cover recent advances.",
                    "priority": 1,
                },
            ],
            "suggested_approach": "Research across 1 sub-topic.",
        })

        with patch("token_tracker.track_chain_invoke", return_value=mock_response):
            with patch("chain.chain._get_chains") as mock_chains:
                mock_chain = MagicMock()
                mock_chains.return_value = (mock_chain, None, None, None)
                result = run_planner("Quantum computing")

        self.assertIsNotNone(result)
        self.assertIn("sub_questions", result)
        self.assertIsInstance(result["sub_questions"], list)
        self.assertEqual(len(result["sub_questions"]), 1)
        for q in result["sub_questions"]:
            self.assertIn("question", q)
            self.assertIn("search_query", q)
            self.assertIn("priority", q)

    def test_run_planner_handles_legacy_string_questions(self):
        """run_planner normalizes legacy string-format sub_questions to dict format."""
        from chain.chain import run_planner

        mock_response = json.dumps({
            "topic": "Test",
            "sub_questions": ["Q1?", "Q2?", "Q3?"],
        })

        with patch("token_tracker.track_chain_invoke", return_value=mock_response):
            with patch("chain.chain._get_chains") as mock_chains:
                mock_chain = MagicMock()
                mock_chains.return_value = (mock_chain, None, None, None)
                result = run_planner("Test")

        self.assertEqual(len(result["sub_questions"]), 3)
        for q in result["sub_questions"]:
            self.assertIsInstance(q, dict)
            self.assertIn("question", q)
            self.assertIn("search_query", q)

    def test_run_planner_fallback_on_llm_failure(self):
        """When LLM fails, run_planner falls back to heuristic sub-questions."""
        from chain.chain import run_planner

        with patch("token_tracker.track_chain_invoke", side_effect=Exception("LLM down")):
            with patch("chain.chain._get_chains") as mock_chains:
                mock_chain = MagicMock()
                mock_chains.return_value = (mock_chain, None, None, None)
                result = run_planner("Artificial intelligence trends 2026")

        self.assertIsNotNone(result)
        self.assertIn("sub_questions", result)
        self.assertGreaterEqual(len(result["sub_questions"]), 1)
        for q in result["sub_questions"]:
            self.assertIn("question", q)
            self.assertIn("search_query", q)

    def test_empty_llm_response_triggers_fallback(self):
        """Empty planner response triggers fallback sub-questions."""
        from chain.chain import run_planner

        with patch("token_tracker.track_chain_invoke", return_value=""):
            with patch("chain.chain._get_chains") as mock_chains:
                mock_chain = MagicMock()
                mock_chains.return_value = (mock_chain, None, None, None)
                result = run_planner("Test topic")

        self.assertIsNotNone(result)
        self.assertGreater(len(result["sub_questions"]), 0)


class TestAnalysisWriting(unittest.TestCase):
    """Test the run_analysis_writing function output shape."""

    def test_run_analysis_writing_returns_report(self):
        """run_analysis_writing returns a dict with a report key."""
        from chain.chain import run_analysis_writing

        mock_analysis = json.dumps({
            "findings": "## Analyst Findings\n\nKey finding here.",
            "has_sufficient_evidence": True,
            "gap_reason": None,
        })
        mock_report = "## Executive Summary\n\nSummary.\n\n## Detailed Analysis\n\nContent.\n\n## Key Insights\n- Insight 1\n\n## Sources\n[S1] Real source"

        with patch("token_tracker.track_chain_invoke", side_effect=[mock_analysis, mock_report]):
            with patch("chain.chain._get_chains") as mock_chains:
                mock_chain = MagicMock()
                mock_chains.return_value = (None, None, mock_chain, mock_chain)
                result = run_analysis_writing(
                    "Test topic",
                    "## Tracked Sources\n\n[S1] Real source\n  URL: https://example.com\n  Snippet: content"
                )

        self.assertIn("report", result)
        self.assertIn("analysis", result)
        self.assertIn("sources_data", result)
        self.assertIsInstance(result["report"], str)
        self.assertGreater(len(result["report"]), 0)

    def test_ensure_sources_section_present(self):
        """Report gets a Sources section appended if missing."""
        from chain.chain import run_analysis_writing

        mock_analysis = json.dumps({
            "findings": "## Findings\n\nContent.",
            "has_sufficient_evidence": True,
        })
        mock_report = "## Executive Summary\n\nSummary."

        with patch("token_tracker.track_chain_invoke", side_effect=[mock_analysis, mock_report]):
            with patch("chain.chain._get_chains") as mock_chains:
                mock_chain = MagicMock()
                mock_chains.return_value = (None, None, mock_chain, mock_chain)
                result = run_analysis_writing(
                    "Test",
                    "## Tracked Sources\n\n[S1] Real source\n  URL: https://example.com\n  Snippet: content"
                )

        self.assertIn("## Sources", result["report"])


class TestVerifier(unittest.TestCase):
    """Test the run_verification function with mocked LLM."""

    def test_run_verification_returns_passed(self):
        """run_verification returns a dict with passed, findings, summary keys."""
        from chain.chain import run_verification

        mock_response = json.dumps({
            "passed": True,
            "findings": [],
            "total_claims_checked": 5,
            "summary": "All claims supported by sources.",
        })

        mock_llm = MagicMock()
        mock_llm.invoke.return_value.content = mock_response

        with patch("llm_config.get_fast_llm", return_value=mock_llm):
            with patch("chain.prompts.VERIFIER_PROMPT", "Verify this: {topic} {merged_research} {report}"):
                result = run_verification(
                    "test topic",
                    "## Report\nContent with [S1] citation.",
                    "## Tracked Sources\n\n[S1] Real source\n  URL: https://x.com\n  Snippet: content"
                )

        self.assertIn("passed", result)
        self.assertIn("findings", result)
        self.assertIn("summary", result)
        self.assertTrue(result["passed"])
