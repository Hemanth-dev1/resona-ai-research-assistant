"""Mocked tests for search_provider.py — Tavily-first, ddgs-fallback behaviour.

All tests run without any network access or API keys.
"""

import sys
import unittest
from unittest.mock import patch, MagicMock


class TestWebSearch(unittest.TestCase):
    """Test the web_search function with mocked providers."""

    def setUp(self):
        # Clear only TAVILY_API_KEY — don't wipe entire env
        self._tavily_key_patch = patch.dict("os.environ", {"TAVILY_API_KEY": ""}, clear=False)
        self._tavily_key_patch.start()

    def tearDown(self):
        self._tavily_key_patch.stop()

    def test_tavily_returns_results(self):
        """When TAVILY_API_KEY is set and Tavily returns results, return them."""
        import search_provider
        with patch.object(search_provider, "TAVILY_API_KEY", "tvly-fake-key"):
            mock_client = MagicMock()
            mock_client.search.return_value = {
                "results": [
                    {"url": "https://example.com/1", "title": "Result 1", "content": "Snippet 1"},
                    {"url": "https://example.com/2", "title": "Result 2", "content": "Snippet 2"},
                ]
            }
            with patch("search_provider.TavilyClient", return_value=mock_client) as mock_patched:
                results = search_provider.web_search("test query", max_results=2)

        self.assertEqual(len(results), 2)
        mock_patched.assert_called_once()
        self.assertEqual(results[0]["url"], "https://example.com/1")
        self.assertEqual(results[0]["title"], "Result 1")
        self.assertEqual(results[0]["snippet"], "Snippet 1")

    def test_tavily_empty_falls_back_to_ddgs(self):
        """When Tavily returns empty, fall back to ddgs."""
        import search_provider
        with patch.object(search_provider, "TAVILY_API_KEY", "tvly-fake-key"):
            mock_client = MagicMock()
            mock_client.search.return_value = {"results": []}
            with patch("search_provider.TavilyClient", return_value=mock_client) as mock_tavily:
                with patch("search_provider.DDGS") as mock_ddgs:
                    mock_instance = MagicMock()
                    mock_instance.text.return_value = [
                        {"href": "https://ddgs.example.com", "title": "DDGS Result", "body": "DDGS snippet"}
                    ]
                    mock_ddgs.return_value.__enter__.return_value = mock_instance
                    results = search_provider.web_search("test query", max_results=2)

        self.assertEqual(len(results), 1)
        mock_tavily.assert_called_once()
        self.assertEqual(results[0]["url"], "https://ddgs.example.com")
        self.assertEqual(results[0]["title"], "DDGS Result")

    def test_no_tavily_key_uses_ddgs(self):
        """When TAVILY_API_KEY is empty, skip Tavily and use ddgs directly."""
        import search_provider
        with patch.object(search_provider, "TAVILY_API_KEY", ""):
            with patch("search_provider.DDGS") as mock_ddgs:
                mock_instance = MagicMock()
                mock_instance.text.return_value = [
                    {"href": "https://ddgs.example.com", "title": "DDGS", "body": "snippet"}
                ]
                mock_ddgs.return_value.__enter__.return_value = mock_instance
                results = search_provider.web_search("test", max_results=2)

        self.assertEqual(len(results), 1)
        mock_ddgs.assert_called_once()

    def test_both_providers_fail_returns_empty(self):
        """When both Tavily and ddgs fail, return empty list."""
        import search_provider
        with patch.object(search_provider, "TAVILY_API_KEY", "tvly-fake-key"):
            mock_client = MagicMock()
            mock_client.search.side_effect = Exception("Tavily down")
            with patch("search_provider.TavilyClient", return_value=mock_client) as mock_tavily:
                with patch("search_provider.DDGS") as mock_ddgs:
                    mock_instance = MagicMock()
                    mock_instance.text.side_effect = Exception("DDGS rate-limited")
                    mock_ddgs.return_value.__enter__.return_value = mock_instance
                    results = search_provider.web_search("test", max_results=2)

        self.assertEqual(len(results), 0)

    def test_tavily_import_error_falls_back(self):
        """When tavily-python is not installed, fall back to ddgs."""
        import search_provider
        with patch.object(search_provider, "TAVILY_API_KEY", "tvly-fake-key"):
            with patch("search_provider.TavilyClient", side_effect=ImportError("No module tavily")) as mock_tavily:
                with patch("search_provider.DDGS") as mock_ddgs:
                    mock_instance = MagicMock()
                    mock_instance.text.return_value = [
                        {"href": "https://ddgs.example.com", "title": "DDGS Fallback", "body": "snippet"}
                    ]
                    mock_ddgs.return_value.__enter__.return_value = mock_instance
                    results = search_provider.web_search("test", max_results=2)

        self.assertEqual(len(results), 1)
        mock_tavily.assert_called_once()
        self.assertEqual(results[0]["url"], "https://ddgs.example.com")

    def test_output_format_has_expected_keys(self):
        """Each result dict has url, title, snippet keys."""
        import search_provider
        with patch.object(search_provider, "TAVILY_API_KEY", ""):
            with patch("search_provider.DDGS") as mock_ddgs:
                mock_instance = MagicMock()
                mock_instance.text.return_value = [
                    {"href": "https://x.com", "title": "X", "body": "Body text"}
                ]
                mock_ddgs.return_value.__enter__.return_value = mock_instance
                results = search_provider.web_search("test", max_results=2)
                results = search_provider.web_search("test", max_results=2)

        for r in results:
            self.assertIn("url", r)
            self.assertIn("title", r)
            self.assertIn("snippet", r)
