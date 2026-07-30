"""Mocked tests for memory/session_store.py and chat/conversation_buffer.py.

Session lifecycle CRUD tests without requiring a real ChromaDB instance.
"""

import unittest
from unittest.mock import patch, MagicMock
import uuid


class TestSessionStore(unittest.TestCase):
    """Test session_store.py — document add, query, list, delete."""

    def setUp(self):
        self._store_patches = [
            patch("memory.session_store._get_client"),
            patch("memory.session_store._get_or_create_collection"),
        ]
        for p in self._store_patches:
            p.start()

        # Reset the in-memory set
        import memory.session_store as ss
        ss._sessions_with_content.clear()

    def tearDown(self):
        for p in self._store_patches:
            p.stop()

    def test_new_session_id_generates_uuid_string(self):
        """new_session_id returns a non-empty string."""
        from memory.session_store import new_session_id
        sid = new_session_id()
        self.assertIsInstance(sid, str)
        self.assertEqual(len(sid), 20)

    def test_add_document_returns_chunk_count(self):
        """add_document chunks text and returns the number of chunks."""
        from memory.session_store import add_document

        text = "Hello world. " * 50  # ~750 chars, should create ~1 chunk
        count = add_document("test_session", "test.txt", text, doc_type="upload")
        self.assertGreater(count, 0)

    def test_add_empty_text_returns_zero(self):
        """Adding empty text returns 0 chunks."""
        from memory.session_store import add_document
        count = add_document("test_session", "empty.txt", "", doc_type="upload")
        self.assertEqual(count, 0)

    def test_has_content_after_adding_document(self):
        """After adding a document, has_content returns True."""
        from memory.session_store import add_document, has_content

        session_id = "content_test_session"
        add_document(session_id, "doc.txt", "Some content here.", doc_type="upload")
        self.assertTrue(has_content(session_id))

    def test_has_content_false_for_empty_session(self):
        """A session with no content returns False."""
        from memory.session_store import has_content
        self.assertFalse(has_content("nonexistent_session"))


class TestConversationBuffer(unittest.TestCase):
    """Test conversation_buffer.py — session lifecycle."""

    def setUp(self):
        import chat.conversation_buffer as cb
        cb._sessions.clear()
        cb._created_at.clear()

    def test_create_session_adds_to_dict(self):
        """create_session initialises an empty session."""
        from chat.conversation_buffer import create_session, session_exists

        create_session("test_sid_1")
        self.assertTrue(session_exists("test_sid_1"))

    def test_session_does_not_exist_before_creation(self):
        """session_exists returns False for uncreated sessions."""
        from chat.conversation_buffer import session_exists
        self.assertFalse(session_exists("never_created"))

    def test_add_turn_appends_message(self):
        """add_turn adds a ChatMessage to the session history."""
        import chat.conversation_buffer as cb
        cb.create_session("turn_test")
        cb.add_turn("turn_test", "user", "Hello")
        cb.add_turn("turn_test", "assistant", "Hi there!")

        history = cb.get_full_history("turn_test")
        self.assertEqual(len(history), 2)
        self.assertEqual(history[0].role, "user")
        self.assertEqual(history[0].content, "Hello")
        self.assertEqual(history[1].role, "assistant")
        self.assertEqual(history[1].content, "Hi there!")

    def test_get_windowed_history_returns_last_N_turns(self):
        """get_windowed_history returns only the most recent messages."""
        import chat.conversation_buffer as cb
        cb.create_session("window_test")
        for i in range(10):
            cb.add_turn("window_test", "user", f"Message {i}")
            cb.add_turn("window_test", "assistant", f"Response {i}")

        windowed = cb.get_windowed_history("window_test")
        # WINDOW_TURNS = 6, so 6 * 2 = 12 messages max
        self.assertLessEqual(len(windowed), 12)
        # The last message should be the most recent
        self.assertIn("Response 9", windowed[-1].content)

    def test_clear_session_removes_all_data(self):
        """clear_session removes session from all storage."""
        import chat.conversation_buffer as cb
        cb.create_session("clear_test")
        cb.add_turn("clear_test", "user", "Test")
        cb.clear_session("clear_test")

        self.assertFalse(cb.session_exists("clear_test"))
        self.assertEqual(cb.get_full_history("clear_test"), [])

    def test_get_created_at_returns_iso_format(self):
        """get_created_at returns an ISO-formatted date string."""
        import chat.conversation_buffer as cb
        cb.create_session("date_test")
        created = cb.get_created_at("date_test")
        self.assertIsNotNone(created)
        self.assertIn("T", created)  # ISO format has 'T'

    def test_get_created_at_none_for_missing_session(self):
        """get_created_at returns None for non-existent sessions."""
        import chat.conversation_buffer as cb
        self.assertIsNone(cb.get_created_at("missing_session"))
