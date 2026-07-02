"""Agent definitions for the research pipeline.

Uses Groq via OPENAI_API_BASE env var (set in main.py).
Search via SerperDevTool (falls back to DuckDuckGo if no SERPER_API_KEY).

Model routing:
- Researcher: fast/cheap model (LLM_MODEL_FAST, default: llama-3.1-8b-instant)
- Analyst: capable model (LLM_MODEL_CAPABLE, default: llama-3.3-70b-versatile)
- Writer: capable model (LLM_MODEL_CAPABLE, default: llama-3.3-70b-versatile)
"""

import os

from crewai import Agent, LLM


def get_search_tools() -> list:
    """Get search tools — Serper if key available, else DuckDuckGo."""
    tools = []

    # Always include web scraping
    from crewai_tools import ScrapeWebsiteTool
    tools.append(ScrapeWebsiteTool())

    # Serper (requires API key)
    if os.getenv("SERPER_API_KEY"):
        from crewai_tools import SerperDevTool
        tools.append(SerperDevTool())
        print("  🔍 Using SerperDevTool for web search")
        return tools

    # DuckDuckGo fallback (free)
    from tools import DuckDuckGoSearchTool, DuckDuckGoSearchResults
    tools.append(DuckDuckGoSearchTool())
    tools.append(DuckDuckGoSearchResults())
    print("  🔍 Using DuckDuckGo for web search")
    return tools


def make_agents():
    """Create the research agents with model routing.

    Planner and Researcher use the fast/cheap model.
    Analyst and Writer use the capable model for deep reasoning.

    Returns:
        tuple[Agent, Agent, Agent, Agent]: (planner, researcher, analyst, writer)
    """
    tools = get_search_tools()

    from llm_config import get_provider, get_fast_model_name, get_capable_model_name
    provider = get_provider().value
    fast_model = get_fast_model_name()
    capable_model = get_capable_model_name()

    # Use crewai.LLM class with caching disabled
    # (LiteLLM 1.90+ sends cache_breakpoint which Groq doesn't support)
    fast_llm = LLM(model=f"{provider}/{fast_model}", disable_cache=True, temperature=0.3)
    capable_llm = LLM(model=f"{provider}/{capable_model}", disable_cache=True, temperature=0.3)

    planner = Agent(
        role="Research Planner",
        goal="Decompose broad research topics into focused, actionable sub-questions.",
        backstory=(
            "You are a strategic research planner who excels at breaking down complex topics "
            "into well-structured, focused questions. You ensure every angle of the topic "
            "is covered before research begins."
        ),
        llm=fast_llm,
        allow_delegation=False,
        verbose=True,
    )

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Gather accurate, up-to-date information on the given topic from reliable web sources, guided by the research plan.",
        backstory=(
            "You are a meticulous researcher with expertise in investigative journalism. "
            "You find authoritative sources, cross-reference facts, and extract key insights. "
            "You always check for the most recent information and note conflicting viewpoints."
        ),
        tools=tools,
        llm=fast_llm,
        allow_delegation=False,
        verbose=True,
    )

    analyst = Agent(
        role="Data Analyst & Insight Specialist",
        goal="Analyze the research findings and identify patterns, trends, and key takeaways.",
        backstory=(
            "You are a sharp data analyst who excels at spotting patterns and extracting "
            "meaningful insights from raw information. You organize findings logically and "
            "highlight the most important conclusions for the reader."
        ),
        llm=capable_llm,
        allow_delegation=False,
        verbose=True,
    )

    writer = Agent(
        role="Technical Content Writer",
        goal="Write a polished, well-structured report that presents the findings clearly and professionally.",
        backstory=(
            "You are an award-winning technical writer who makes complex topics accessible. "
            "You produce clean, publication-ready reports with executive summaries, "
            "detailed analysis, and proper citations."
        ),
        llm=capable_llm,
        allow_delegation=False,
        verbose=True,
    )

    return planner, researcher, analyst, writer
