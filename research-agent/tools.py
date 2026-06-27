"""Custom tools for web search without requiring external API keys.

Provides DuckDuckGo search as a free alternative to SerperDevTool.
"""

import json
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class DuckDuckGoSearchInput(BaseModel):
    """Input schema for DuckDuckGo search."""
    query: str = Field(description="The search query to look up")


class DuckDuckGoSearchTool(BaseTool):
    """Tool to search the web using DuckDuckGo (no API key required)."""

    name: str = "DuckDuckGo Search"
    description: str = (
        "Search the web for information using DuckDuckGo. "
        "Use this to find recent, authoritative information on any topic. "
        "Input should be a clear search query."
    )
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        """Execute the search and return formatted results."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=8))

            if not results:
                return f"No results found for query: {query}"

            formatted = [
                f"{i}. **{r.get('title', 'No title')}**\n"
                f"   {r.get('body', 'No description')}\n"
                f"   Source: {r.get('href', 'No URL')}"
                for i, r in enumerate(results, 1)
            ]
            return "\n\n".join(formatted)

        except Exception as e:
            return f"Search error: {str(e)}"


class DuckDuckGoSearchResults(BaseTool):
    """Tool to get raw search results as JSON for deeper analysis."""

    name: str = "DuckDuckGo Search Results"
    description: str = (
        "Get detailed search results from DuckDuckGo including titles, "
        "snippets, and URLs. Use when you need structured data."
    )
    args_schema: Type[BaseModel] = DuckDuckGoSearchInput

    def _run(self, query: str) -> str:
        """Execute the search and return JSON-formatted results."""
        try:
            from duckduckgo_search import DDGS

            with DDGS() as ddgs:
                results = list(ddgs.text(query, max_results=10))

            if not results:
                return json.dumps({"query": query, "results": [], "count": 0})

            return json.dumps({
                "query": query,
                "results": [
                    {
                        "title": r.get("title", ""),
                        "snippet": r.get("body", ""),
                        "url": r.get("href", ""),
                    }
                    for r in results
                ],
                "count": len(results),
            }, indent=2)

        except Exception as e:
            return json.dumps({"error": str(e)})
