"""Mocked tests for graph.py — LangGraph state machine routing.

Tests verify node routing and critic loop termination without any LLM calls.
Import-oriented tests (planner_node, critic_node) use context-manager patches
on the import site inside the function, not on non-existent module attributes.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock


class TestGraphRouting(unittest.TestCase):
    """Test the graph's routing functions directly (no LLM calls needed).

    No setUp patches needed — routing functions are pure Python logic
    that don't import any chain/critic modules at module level.
    """

    def _make_state(self, **overrides):
        """Helper to build a ResearchState dict with defaults."""
        base = {
            "topic": "Test topic",
            "memory_context": "",
            "mode": "langchain",
            "plan": None,
            "sub_questions": [],
            "merged_research": "",
            "report": "## Report\n\nSome content.",
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
            "strict_verification": False,
            "claim_verification_passed": None,
            "claim_verification_summary": None,
            "unsupported_claims": [],
            "claim_verifier_iterations": 0,
            "max_claim_verifier_iterations": 2,
            "error": None,
        }
        base.update(overrides)
        return base

    def test_route_from_start_with_research_skips_planner(self):
        """When merged_research exists, route to analysis_writer (skip planner)."""
        from graph import route_from_start
        # String must be > 50 chars to trigger the skip-to-analysis_writer path
        state = self._make_state(merged_research="## Tracked Sources\n\n[S1] Real source content here, with extra length to exceed 50 chars threshold.")
        self.assertEqual(route_from_start(state), "analysis_writer")

    def test_route_from_start_without_research_goes_to_planner(self):
        """When merged_research is empty, route to planner."""
        from graph import route_from_start
        state = self._make_state(merged_research="")
        self.assertEqual(route_from_start(state), "planner")

    def test_route_after_critic_passed_goes_to_END(self):
        """When critique passes, route to END (which goes to verifier)."""
        from graph import route_after_critic
        state = self._make_state(critique_passed=True, critique_iterations=1)
        self.assertEqual(route_after_critic(state), "__end__")

    def test_route_after_critic_max_iterations_goes_to_END(self):
        """When max iterations reached, route to END even if not passed."""
        from graph import route_after_critic
        state = self._make_state(
            critique_passed=False,
            critique_iterations=3,
            max_critic_iterations=3,
        )
        self.assertEqual(route_after_critic(state), "__end__")

    def test_route_after_critic_needs_revision(self):
        """When not passed and iterations remain, route to revise."""
        from graph import route_after_critic
        state = self._make_state(
            critique_passed=False,
            critique_iterations=1,
            max_critic_iterations=3,
        )
        self.assertEqual(route_after_critic(state), "revise")

    def test_route_after_claim_verifier_all_ok_goes_to_critic(self):
        """When all claims verified, route to critic."""
        from graph import route_after_claim_verifier
        state = self._make_state(claim_verification_passed=True)
        self.assertEqual(route_after_claim_verifier(state), "critic")

    def test_route_after_claim_verifier_unsupported_retries(self):
        """When claims unsupported and under max attempts, route to research_retry."""
        from graph import route_after_claim_verifier
        state = self._make_state(
            claim_verification_passed=False,
            claim_verifier_iterations=1,
            max_claim_verifier_iterations=2,
            unsupported_claims=[{"claim_text": "fake claim", "source_id": "S1"}],
        )
        self.assertEqual(route_after_claim_verifier(state), "research_retry")

    def test_route_after_claim_verifier_maxed_out_goes_to_END(self):
        """When max retries exhausted, route to END (skips critic+verifier)."""
        from graph import route_after_claim_verifier
        state = self._make_state(
            claim_verification_passed=False,
            claim_verifier_iterations=2,
            max_claim_verifier_iterations=2,
            unsupported_claims=[{"claim_text": "fake", "source_id": "S1"}],
        )
        self.assertEqual(route_after_claim_verifier(state), "__end__")

    def test_route_after_verifier_passed_goes_to_END(self):
        """When verification passes, route to END."""
        from graph import route_after_verifier
        state = self._make_state(verification_passed=True)
        self.assertEqual(route_after_verifier(state), "__end__")

    def test_route_after_verifier_strict_failed_routes_to_revise(self):
        """When strict mode and verification fails, route to revise."""
        from graph import route_after_verifier
        state = self._make_state(
            verification_passed=False,
            strict_verification=True,
            verification_iterations=1,
            max_verification_iterations=2,
        )
        self.assertEqual(route_after_verifier(state), "revise")

    def test_planner_node_calls_run_planner(self):
        """planner_node calls chain.chain.run_planner via context-manager patch."""
        from graph import planner_node
        # Patch at the source module where the function imports from
        with patch("chain.chain.run_planner") as mock_planner:
            mock_planner.return_value = {
                "sub_questions": [
                    {"question": "Q1?", "search_query": "Q1 search", "priority": 1},
                ]
            }
            state = self._make_state()
            result = planner_node(state)
            self.assertEqual(len(result["sub_questions"]), 1)
            self.assertEqual(result["sub_questions"][0]["question"], "Q1?")
            mock_planner.assert_called_once()

    def test_critic_node_increments_iteration(self):
        """critic_node increments critique_iterations and stores critique."""
        from graph import critic_node
        from schemas.models import CritiqueResult, DimensionScore, CritiqueDimension
        with patch("critic.score_report") as mock_score:
            mock_score.return_value = CritiqueResult(
                topic="test",
                overall_score=8,
                dimensions=[],
                passed=True,
                iteration=1,
                summary="Good report.",
            )
            state = self._make_state(critique_iterations=0)
            result = critic_node(state)
            self.assertEqual(result["critique_iterations"], 1)
            self.assertEqual(result["critique_score"], 8)
            self.assertTrue(result["critique_passed"])
            mock_score.assert_called_once()


class TestCitationIntegrity(unittest.TestCase):
    """Regression tests for citation/source-ID integrity (fixes from cf300d2/5627192)."""

    def test_zero_sources_produces_no_citations(self):
        """When no sources are available, the report should NOT fabricate [S#] tags.

        This is the regression test for the fix that prevented citation fabrication
        in the no-source case (commits cf300d2, 5627192).
        """
        from chain.chain import _ensure_sources_section

        # Simulate the no-sources case merged_research from graph.py
        report = "\u26a0\ufe0f No sources were available for this topic."
        result = _ensure_sources_section(report, "No source metadata was available during research.")
        # Should NOT append a Sources section with fabricated IDs
        self.assertNotIn("## Sources", result)

    def test_ensure_sources_does_not_duplicate_existing_section(self):
        """If report already has a Sources section, leave it unchanged."""
        from chain.chain import _ensure_sources_section

        report = "## Summary\n\nContent.\n\n## Sources\n\n[S1] Real source"
        result = _ensure_sources_section(report, "[S1] Real source")
        # Should not add another ## Sources
        self.assertEqual(result.count("## Sources"), 1)

    def test_ensure_sources_appends_if_missing(self):
        """If report has real sources but no Sources section, append one."""
        from chain.chain import _ensure_sources_section

        report = "## Summary\n\nContent."
        sources = "[S1] Title: Example\n  URL: https://example.com\n  Snippet: content"
        result = _ensure_sources_section(report, sources)
        self.assertIn("## Sources", result)
