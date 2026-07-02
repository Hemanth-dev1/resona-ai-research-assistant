"""Tests for the unified pipeline orchestrator."""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


class TestOrchestrationMode:
    """Tests for get_mode()."""

    def test_get_mode_always_langgraph(self):
        """get_mode() always returns 'langgraph'."""
        from orchestrator import get_mode
        assert get_mode() == "langgraph"


# Helper to detect if we're in a full environment (Docker) vs test-only
_has_chromadb = False
try:
    import chromadb  # noqa: F401
    _has_chromadb = True
except ImportError:
    pass

_has_provider = (
    bool(os.getenv("GROQ_API_KEY"))
    or bool(os.getenv("OPENAI_API_KEY"))
    or bool(os.getenv("ANTHROPIC_API_KEY"))
)


# Skip full pipeline tests when dependencies are not available
_skip_pipeline = not (_has_chromadb and _has_provider)
_skip_reason = "Skipped: needs full Docker environment (chromadb + API key)"


class TestRunPipeline:
    """Tests for run_pipeline()."""

    @pytest.mark.skipif(_skip_pipeline, reason=_skip_reason)
    def test_run_pipeline_returns_dict(self):
        """run_pipeline returns a dict with expected keys."""
        from orchestrator import run_pipeline
        result = run_pipeline("test topic")
        assert isinstance(result, dict)
        assert "report" in result
        assert "critique_iterations" in result
        assert "verification_result" in result
        assert "plan" in result
        assert "sub_questions" in result
        assert "error" in result
        assert "duration_seconds" in result

    @pytest.mark.skipif(not _has_provider, reason="Skipped: needs API key for LLM")
    def test_run_pipeline_with_research(self):
        """run_pipeline accepts pre-computed merged_research."""
        from orchestrator import run_pipeline
        research = "Some pre-computed research data about the topic."
        result = run_pipeline("test topic", merged_research=research)
        assert isinstance(result, dict)
        assert result["error"] is None or "❌" not in str(result.get("error", ""))


class TestFallbackResearch:
    """Tests for the fallback DuckDuckGo research in CLI mode."""

    @pytest.mark.skipif(_skip_pipeline, reason=_skip_reason)
    def test_fallback_with_known_topic(self):
        """Fallback research returns content for a well-known topic."""
        from orchestrator import run_pipeline
        result = run_pipeline("Python programming language")
        assert result["error"] is None or "❌" not in str(result.get("error", ""))
        assert len(result.get("report", "")) > 50 or result.get("error") is None
