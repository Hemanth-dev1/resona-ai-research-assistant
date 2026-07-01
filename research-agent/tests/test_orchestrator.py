"""Tests for the unified pipeline orchestrator."""

import os
import sys
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest


class TestOrchestrationMode:
    """Tests for OrchestrationMode enum and get_mode()."""

    def test_get_mode_default(self):
        """get_mode() returns LANGGRAPH when ORCHESTRATION is unset."""
        if "ORCHESTRATION" in os.environ:
            del os.environ["ORCHESTRATION"]
        from orchestrator import get_mode, OrchestrationMode
        assert get_mode() == OrchestrationMode.LANGGRAPH

    def test_get_mode_langgraph(self):
        """get_mode() returns LANGGRAPH for valid 'langgraph' value."""
        os.environ["ORCHESTRATION"] = "langgraph"
        from orchestrator import get_mode, OrchestrationMode
        assert get_mode() == OrchestrationMode.LANGGRAPH

    def test_get_mode_crewai(self):
        """get_mode() returns CREWAI for valid 'crewai' value."""
        os.environ["ORCHESTRATION"] = "crewai"
        from orchestrator import get_mode, OrchestrationMode
        assert get_mode() == OrchestrationMode.CREWAI

    def test_get_mode_langchain(self):
        """get_mode() returns LANGCHAIN for valid 'langchain' value."""
        os.environ["ORCHESTRATION"] = "langchain"
        from orchestrator import get_mode, OrchestrationMode
        assert get_mode() == OrchestrationMode.LANGCHAIN

    def test_get_mode_invalid(self):
        """get_mode() falls back to LANGGRAPH for invalid values."""
        os.environ["ORCHESTRATION"] = "invalid_mode"
        from orchestrator import get_mode, OrchestrationMode
        assert get_mode() == OrchestrationMode.LANGGRAPH

    def test_get_available_modes(self):
        """get_available_modes() returns all three mode strings."""
        from orchestrator import get_available_modes, OrchestrationMode
        modes = get_available_modes()
        assert sorted(modes) == sorted([m.value for m in OrchestrationMode])
        assert len(modes) == 3
        assert "langgraph" in modes
        assert "crewai" in modes
        assert "langchain" in modes


# Helper to detect if we're in a full environment (Docker) vs test-only
_has_chromadb = False
try:
    import chromadb  # noqa: F401
    _has_chromadb = True
except ImportError:
    pass

_has_crewai = False
try:
    import crewai  # noqa: F401
    _has_crewai = True
except ImportError:
    pass

_has_provider = (
    bool(os.getenv("GROQ_API_KEY"))
    or bool(os.getenv("OPENAI_API_KEY"))
    or bool(os.getenv("ANTHROPIC_API_KEY"))
)


# Skip full pipeline tests when dependencies are not available
_skip_pipeline = not (_has_chromadb and _has_crewai and _has_provider)
_skip_reason = "Skipped: needs full Docker environment (chromadb + crewai + API key)"


class TestRunPipeline:
    """Tests for run_pipeline()."""

    @pytest.mark.skipif(_skip_pipeline, reason=_skip_reason)
    def test_run_pipeline_returns_dict(self):
        """run_pipeline returns a dict with expected keys."""
        os.environ["ORCHESTRATION"] = "langchain"
        from orchestrator import run_pipeline
        result = run_pipeline("test topic")
        assert isinstance(result, dict)
        assert "report" in result
        assert "critique_iterations" in result
        assert "verification_result" in result
        assert "plan" in result
        assert "sub_questions" in result
        assert "error" in result
        assert "mode" in result
        assert "duration_seconds" in result

    @pytest.mark.skipif(not _has_provider, reason="Skipped: needs API key for LLM")
    def test_run_pipeline_error_mode(self):
        """run_pipeline returns error for unknown mode."""
        from orchestrator import run_pipeline
        result = run_pipeline("test", mode="nonexistent")
        assert result["error"] is not None

    @pytest.mark.skipif(_skip_pipeline, reason=_skip_reason)
    def test_run_pipeline_with_research(self):
        """run_pipeline accepts pre-computed merged_research."""
        os.environ["ORCHESTRATION"] = "langchain"
        from orchestrator import run_pipeline
        research = "Some pre-computed research data about the topic."
        result = run_pipeline("test topic", merged_research=research)
        assert isinstance(result, dict)
        assert result["error"] is None or "❌" not in str(result.get("error", ""))

    @pytest.mark.skipif(_skip_pipeline, reason=_skip_reason)
    def test_run_pipeline_mode_langgraph(self):
        """run_pipeline works with langgraph mode."""
        os.environ["ORCHESTRATION"] = "langgraph"
        from orchestrator import run_pipeline
        result = run_pipeline("test", merged_research="Research data here.")
        assert isinstance(result, dict)
        assert "report" in result

    @pytest.mark.skipif(_skip_pipeline, reason=_skip_reason)
    def test_run_pipeline_mode_default(self):
        """run_pipeline uses get_mode() when mode is None."""
        os.environ["ORCHESTRATION"] = "langchain"
        from orchestrator import run_pipeline
        result = run_pipeline("test topic", merged_research="Research data.")
        assert result["mode"] == "langchain"


class TestFallbackResearch:
    """Tests for the fallback DuckDuckGo research in CLI mode."""

    @pytest.mark.skipif(_skip_pipeline, reason=_skip_reason)
    def test_fallback_with_known_topic(self):
        """Fallback research returns content for a well-known topic."""
        from orchestrator import run_pipeline
        result = run_pipeline("Python programming language")
        assert result["error"] is None or "❌" not in str(result.get("error", ""))
        assert len(result.get("report", "")) > 50 or result.get("error") is None
