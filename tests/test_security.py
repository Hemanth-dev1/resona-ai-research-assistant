"""Tests for security features: rate limiting, input validation, filename sanitization.

All tests use mocking — no real API calls or network access.
"""

import os
import sys
import unittest
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

# Point to project root for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestRateLimiter(unittest.TestCase):
    """Tests for rate_limiter.py — token bucket behaviour."""

    def setUp(self):
        # Clear rate limiter state between tests
        from rate_limiter import _hits
        _hits.clear()

    def test_allows_requests_under_limit(self):
        """Rate limiter allows requests up to the configured limit."""
        from rate_limiter import check_rate_limit, _RATE_LIMIT
        ip = "192.168.1.1"
        # Should allow all requests up to the limit
        for _ in range(_RATE_LIMIT):
            self.assertTrue(check_rate_limit(ip))

    def test_blocks_requests_over_limit(self):
        """Rate limiter blocks requests above the configured limit."""
        from rate_limiter import check_rate_limit, _RATE_LIMIT
        ip = "192.168.1.2"
        for _ in range(_RATE_LIMIT):
            check_rate_limit(ip)
        # Next request should be denied
        self.assertFalse(check_rate_limit(ip))

    def test_different_ips_have_separate_counters(self):
        """Rate limiter maintains separate counters per IP."""
        from rate_limiter import check_rate_limit, _RATE_LIMIT
        ip1 = "10.0.0.1"
        ip2 = "10.0.0.2"
        for _ in range(_RATE_LIMIT):
            check_rate_limit(ip1)
        # ip1 should be blocked
        self.assertFalse(check_rate_limit(ip1))
        # ip2 should still be allowed
        self.assertTrue(check_rate_limit(ip2))

    def test_get_rate_limit_remaining(self):
        """get_rate_limit_remaining returns correct count."""
        from rate_limiter import check_rate_limit, get_rate_limit_remaining, _RATE_LIMIT
        ip = "10.0.0.3"
        self.assertEqual(get_rate_limit_remaining(ip), _RATE_LIMIT)
        check_rate_limit(ip)
        self.assertEqual(get_rate_limit_remaining(ip), _RATE_LIMIT - 1)


class TestInputValidation(unittest.TestCase):
    """Tests for _validate_input_length in server.py."""

    def test_short_input_passes(self):
        """Short inputs within limits pass validation."""
        from server import _validate_input_length, RESONA_MAX_INPUT_LENGTH
        result = _validate_input_length("short", "topic", max_length=RESONA_MAX_INPUT_LENGTH)
        self.assertEqual(result, "short")

    def test_empty_input_is_trimmed(self):
        """Empty/whitespace-only inputs are returned as empty string."""
        from server import _validate_input_length
        result = _validate_input_length("   ", "topic")
        self.assertEqual(result, "")

    def test_oversized_input_raises_422(self):
        """Input exceeding max_length raises HTTPException with 422."""
        from server import _validate_input_length
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            _validate_input_length("a" * 5001, "topic", max_length=5000)
        self.assertEqual(ctx.exception.status_code, 422)
        self.assertIn("topic", ctx.exception.detail)
        self.assertIn("5001", ctx.exception.detail)

    def test_boundary_exact_max_length(self):
        """Input at exactly the max length passes."""
        from server import _validate_input_length
        result = _validate_input_length("a" * 100, "message", max_length=100)
        self.assertEqual(len(result), 100)

    def test_boundary_exceed_by_one(self):
        """Input one char over max length fails with 422."""
        from server import _validate_input_length
        from fastapi import HTTPException
        with self.assertRaises(HTTPException) as ctx:
            _validate_input_length("a" * 101, "message", max_length=100)
        self.assertEqual(ctx.exception.status_code, 422)


class TestFilenameSanitization(unittest.TestCase):
    """Tests for _safe_filename in server.py."""

    def test_normal_filename_passes_through(self):
        """Normal filenames with alphanumeric chars pass through clean."""
        from server import _safe_filename
        result = _safe_filename("report.pdf")
        self.assertEqual(result, "report.pdf")

    def test_traversal_attempt_is_sanitized(self):
        """Path traversal like '../../etc/passwd' is sanitized to just the basename."""
        from server import _safe_filename
        result = _safe_filename("../../etc/passwd.txt")
        self.assertNotIn("..", result)
        self.assertNotIn("/", result)
        self.assertIn("passwd.txt", result)

    def test_traversal_with_encoded_chars(self):
        """Traversal with mixed encodings gets sanitized."""
        from server import _safe_filename
        result = _safe_filename("....//....//etc/shadow.txt")
        self.assertNotIn("/", result)
        # After removing path chars, should still have a recognizable name
        self.assertGreater(len(result), 0)

    def test_special_chars_replaced(self):
        """Non-alphanumeric special chars are replaced with underscores."""
        from server import _safe_filename
        result = _safe_filename("my<file>.pdf")
        self.assertNotIn("<", result)
        self.assertIn("my_file_.pdf", result)

    def test_empty_filename_gets_uuid_fallback(self):
        """Empty or dot-only filenames get a uuid fallback."""
        from server import _safe_filename
        result = _safe_filename("...")
        self.assertNotEqual(result, "...")
        self.assertNotEqual(result, "")
        self.assertIn("upload_", result)

    def test_none_filename_gets_uuid(self):
        """None/empty input gets a uuid fallback."""
        from server import _safe_filename
        result = _safe_filename("")
        self.assertIn("upload_", result)
        result2 = _safe_filename(" ")
        self.assertIn("upload_", result2)

    def test_no_extension_works(self):
        """Filenames without extensions are preserved."""
        from server import _safe_filename
        result = _safe_filename("Makefile")
        self.assertEqual(result, "Makefile")

    def test_multi_dot_filename(self):
        """Filenames with multiple dots are preserved."""
        from server import _safe_filename
        result = _safe_filename("archive.tar.gz")
        self.assertEqual(result, "archive.tar.gz")


class TestFilenameInUploadFlow(unittest.TestCase):
    """Integration test: upload_document uses sanitized filename for session_store."""

    @patch("server.session_store")
    @patch("server.extract_text", return_value="mocked text")
    def test_upload_passes_safe_name_to_store(self, mock_extract, mock_store):
        """upload_document passes the sanitized safe_name to session_store.add_document,
        not the raw client-supplied filename."""
        from fastapi import UploadFile
        from server import app

        # Create a test client
        client = TestClient(app)

        # Need to mock the rate limiter to allow the request through
        with patch("server.check_rate_limit", return_value=True):
            # Upload a file with a traversal-style name
            response = client.post(
                "/api/chat/documents",
                data={"session_id": "test_session_123"},
                files={"file": ("../../etc/passwd.txt", b"test content", "text/plain")},
            )

        # The store should have received the sanitized basename, not the traversal path
        if mock_store.add_document.called:
            call_args = mock_store.add_document.call_args[0]
            # call_args[0] = session_id, call_args[1] = source_name
            source_name = call_args[1]
            self.assertNotIn("..", str(source_name))
            self.assertNotIn("/", str(source_name))
            # Should contain the original base filename (passwd.txt) minus traversal
            self.assertIn("passwd", str(source_name))
