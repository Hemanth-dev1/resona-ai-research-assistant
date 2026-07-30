"""Per-session ChromaDB store for the chatbot.

Unlike chroma_store.py (one global collection of past research reports,
embedded with OpenAI), this module gives every chat session its own
isolated collection so one user's uploaded documents and conversation
history never leak into another user's session.

Uses ChromaDB's bundled local ONNX MiniLM embedding function — no
OPENAI_API_KEY dependency, which matters for a chat feature that should
work with any LLM_PROVIDER (Groq/Gemini/Anthropic) alone.
"""

import os
import uuid
from datetime import datetime
from typing import Optional

import chromadb
from chromadb.utils import embedding_functions

DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db_sessions")
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100
SESSION_TTL_HOURS = 24  # sessions older than this are eligible for cleanup

_client: Optional[chromadb.PersistentClient] = None
_embedding_fn = None
_sessions_with_content: set[str] = set()  # cheap in-memory gate, avoids a DB round-trip on every empty-session turn


def _get_client() -> chromadb.PersistentClient:
    global _client
    if _client is None:
        _client = chromadb.PersistentClient(path=DEFAULT_DB_PATH)
    return _client


def _get_embedding_fn():
    """Local MiniLM embedding function bundled with chromadb — no API key needed."""
    global _embedding_fn
    if _embedding_fn is None:
        _embedding_fn = embedding_functions.DefaultEmbeddingFunction()
    return _embedding_fn


def _collection_name(session_id: str) -> str:
    # Chroma collection names must be alnum/underscore/hyphen, 3-63 chars
    return f"chat_{session_id}"


def new_session_id() -> str:
    return uuid.uuid4().hex[:20]


def _get_or_create_collection(session_id: str):
    client = _get_client()
    name = _collection_name(session_id)
    try:
        return client.get_collection(name=name, embedding_function=_get_embedding_fn())
    except Exception:
        return client.create_collection(
            name=name,
            embedding_function=_get_embedding_fn(),
            metadata={"created_at": datetime.utcnow().isoformat()},
        )


def _get_collection_if_exists(session_id: str):
    client = _get_client()
    name = _collection_name(session_id)
    try:
        return client.get_collection(name=name)
    except Exception:
        return None


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    if not text:
        return []
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def add_document(session_id: str, source_name: str, text: str, doc_type: str = "upload") -> int:
    """Chunk + embed a document (or a conversation turn) into a session's collection.

    Args:
        session_id: Isolated chat session identifier.
        source_name: Filename or a label like "turn_12" for conversational memory.
        text: Raw text content.
        doc_type: "upload" (user-provided document) or "turn" (chat history,
            stored so long conversations can be retrieved rather than kept
            entirely in the prompt window).

    Returns:
        Number of chunks stored.
    """
    if not text or not text.strip():
        return 0

    collection = _get_or_create_collection(session_id)
    chunks = _chunk_text(text)
    if not chunks:
        return 0

    base_id = f"{source_name}_{uuid.uuid4().hex[:8]}"
    ids = [f"{base_id}_{i}" for i in range(len(chunks))]
    metadatas = [
        {
            "source": source_name,
            "doc_type": doc_type,
            "chunk_index": i,
            "timestamp": datetime.utcnow().isoformat(),
        }
        for i in range(len(chunks))
    ]

    collection.add(documents=chunks, ids=ids, metadatas=metadatas)
    _sessions_with_content.add(session_id)
    return len(chunks)


def has_content(session_id: str) -> bool:
    """Check whether the session has any stored ChromaDB content.

    If the in-memory gate was lost due to a server restart, fall back to
    checking the existing session collection count so an existing session is
    still recognized without creating a new empty collection.
    """
    if session_id in _sessions_with_content:
        return True

    try:
        collection = _get_collection_if_exists(session_id)
        if collection is None:
            return False
        if collection.count() > 0:
            _sessions_with_content.add(session_id)
            return True
    except Exception:
        pass
    return False


def query_session(session_id: str, query: str, n_results: int = 4, doc_type: Optional[str] = None) -> list[dict]:
    """Retrieve the most relevant chunks for a query within one session.

    Args:
        session_id: Session to search within.
        query: The user's current message.
        n_results: Max chunks to return.
        doc_type: Optionally restrict to "upload" or "turn".

    Returns:
        List of {"text": ..., "source": ..., "doc_type": ...} dicts, most relevant first.
    """
    try:
        collection = _get_or_create_collection(session_id)
        count = collection.count()
        if count == 0:
            return []

        where = {"doc_type": doc_type} if doc_type else None
        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
            where=where,
        )
        docs = results.get("documents", [[]])[0]
        metas = results.get("metadatas", [[]])[0]
        return [
            {"text": d, "source": m.get("source", "unknown"), "doc_type": m.get("doc_type", "upload")}
            for d, m in zip(docs, metas)
        ]
    except Exception as e:
        print(f"  ⚠️  session_store query error: {e}")
        return []


def list_documents(session_id: str) -> list[str]:
    """Unique source filenames uploaded into this session."""
    try:
        collection = _get_or_create_collection(session_id)
        count = collection.count()
        if count == 0:
            return []
        results = collection.get(where={"doc_type": "upload"}, limit=count)
        sources = sorted({m["source"] for m in results.get("metadatas", []) if "source" in m})
        return sources
    except Exception as e:
        print(f"  ⚠️  session_store list_documents error: {e}")
        return []


def delete_session(session_id: str) -> None:
    """Delete a session's entire collection (clears docs + memory)."""
    try:
        client = _get_client()
        client.delete_collection(name=_collection_name(session_id))
    except Exception:
        pass  # already gone / never existed
    _sessions_with_content.discard(session_id)
