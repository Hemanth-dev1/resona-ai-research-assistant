"""Windowed conversation memory.

Two tiers, matching how a human would actually recall a conversation:

1. Short-term: the last WINDOW_TURNS message pairs are kept verbatim
   in-process and injected into every prompt as-is.
2. Long-term: every turn is also embedded into the session's ChromaDB
   collection (doc_type="turn"), so if the conversation runs long,
   older turns can still be retrieved by relevance instead of being
   silently forgotten once they fall out of the window.

NOTE: the in-memory dict is per-process. That's fine for a single-instance
Render deploy (matches this repo's current deployment). If this is ever
scaled to multiple instances, swap `_sessions` for Redis — the interface
below (get_history / add_turn / create_session) would not need to change.
"""

from datetime import datetime
from typing import Optional

from memory import session_store
from schemas.chat_models import ChatMessage

WINDOW_TURNS = 6  # 6 user+assistant pairs kept verbatim in-prompt

# session_id -> list[ChatMessage]  (full history, in arrival order)
_sessions: dict[str, list[ChatMessage]] = {}
_created_at: dict[str, str] = {}


def create_session(session_id: str) -> None:
    if session_id not in _sessions:
        _sessions[session_id] = []
        _created_at[session_id] = datetime.utcnow().isoformat()


def session_exists(session_id: str) -> bool:
    return session_id in _sessions


def get_full_history(session_id: str) -> list[ChatMessage]:
    return _sessions.get(session_id, [])


def get_windowed_history(session_id: str) -> list[ChatMessage]:
    """Last WINDOW_TURNS*2 messages, verbatim — what goes straight into the prompt."""
    history = _sessions.get(session_id, [])
    return history[-(WINDOW_TURNS * 2):]


def add_turn(session_id: str, role: str, content: str) -> None:
    create_session(session_id)
    msg = ChatMessage(role=role, content=content)
    _sessions[session_id].append(msg)

    # Persist every turn into the vector store too, so it survives beyond
    # the verbatim window and can be recalled by relevance on long chats.
    turn_index = len(_sessions[session_id])
    session_store.add_document(
        session_id=session_id,
        source_name=f"turn_{turn_index}_{role}",
        text=content,
        doc_type="turn",
    )


def get_created_at(session_id: str) -> Optional[str]:
    return _created_at.get(session_id)


def clear_session(session_id: str) -> None:
    _sessions.pop(session_id, None)
    _created_at.pop(session_id, None)
    session_store.delete_session(session_id)
