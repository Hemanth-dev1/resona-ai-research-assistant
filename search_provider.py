"""Unified web search with a reliable primary provider and a free fallback.

Both research_queue.py and orchestrator.py previously called `ddgs`
directly and swallowed all errors into an empty result list. ddgs is an
unofficial DuckDuckGo scraper that gets rate-limited/blocked often,
especially from cloud IPs — that was the actual cause of "no sources
retrieved" failures, not a logic bug in the pipeline.

This module tries Tavily first (if TAVILY_API_KEY is set — reliable,
built for this use case), falls back to ddgs (free, no key needed, but
less reliable), and only returns an empty list if BOTH fail — logging
clearly which provider was used or why both failed, instead of a single
generic "Search error" line.
"""

import os

from tavily import TavilyClient
from ddgs import DDGS

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "").strip()


def web_search(query: str, max_results: int = 8) -> list[dict]:
    """Returns a list of {"url", "title", "snippet"} dicts. Empty list only
    if every configured provider fails."""
    if TAVILY_API_KEY:
        results = _search_tavily(query, max_results)
        if results:
            return results
        print(f"  \u26a0\ufe0f  Tavily returned no results for '{query}', falling back to ddgs")

    results = _search_ddgs(query, max_results)
    if not results:
        print(f"  \u26a0\ufe0f  All search providers failed for '{query}' — report will note insufficient sources")
    return results


def _search_tavily(query: str, max_results: int) -> list[dict]:
    try:
        client = TavilyClient(api_key=TAVILY_API_KEY)
        response = client.search(query, max_results=max_results)
        return [
            {
                "url": r.get("url", ""),
                "title": r.get("title", ""),
                "snippet": r.get("content", ""),
            }
            for r in response.get("results", [])
        ]
    except ImportError:
        print("  \u26a0\ufe0f  tavily-python not installed — run: pip install tavily-python")
        return []
    except Exception as e:
        print(f"  \u26a0\ufe0f  Tavily search error: {e}")
        return []


def _search_ddgs(query: str, max_results: int) -> list[dict]:
    try:
        with DDGS(timeout=20) as ddgs:
            results = list(ddgs.text(query, max_results=max_results))
        return [
            {"url": r.get("href", ""), "title": r.get("title", ""), "snippet": r.get("body", "")}
            for r in results
        ]
    except ImportError:
        print("  \u26a0\ufe0f  ddgs not installed")
        return []
    except Exception as e:
        print(f"  \u26a0\ufe0f  ddgs search error: {e}")
        return []
