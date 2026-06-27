"""Agent definitions for the research pipeline.

Uses Groq via OPENAI_API_BASE env var (set in main.py).
Search via SerperDevTool (falls back to DuckDuckGo if no SERPER_API_KEY).
"""

import os

from crewai import Agent


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
    """Create the three research agents.

    Returns:
        tuple[Agent, Agent, Agent]: (researcher, analyst, writer)
    """
    tools = get_search_tools()

    researcher = Agent(
        role="Senior Research Analyst",
        goal="Gather accurate, up-to-date information on the given topic from reliable web sources.",
        backstory=(
            "You are a meticulous researcher with expertise in investigative journalism. "
            "You find authoritative sources, cross-reference facts, and extract key insights. "
            "You always check for the most recent information and note conflicting viewpoints."
        ),
        tools=tools,
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
        allow_delegation=False,
        verbose=True,
    )

    return researcher, analyst, writer
