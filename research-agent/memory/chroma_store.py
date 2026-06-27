"""Persistent ChromaDB vector store for cross-session research memory.

All generated reports are chunked and stored in a local ChromaDB instance.
Subsequent research on related topics automatically retrieves prior findings
as additional context for the LangChain pipeline.
"""

import os
from datetime import datetime
from typing import Any, Optional

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction


# Default paths
DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), "chroma_db")
COLLECTION_NAME = "resona_research"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


# Module-level cached resources (lazily initialized)
_client: Optional[chromadb.PersistentClient] = None
_collection: Optional[chromadb.Collection] = None


def _get_client(path: Optional[str] = None) -> chromadb.PersistentClient:
    """Get or create the persistent ChromaDB client (cached).

    Args:
        path: Path to the ChromaDB storage directory. Defaults to ./chroma_db.

    Returns:
        Cached PersistentClient instance.
    """
    global _client
    if _client is None:
        db_path = path or DEFAULT_DB_PATH
        _client = chromadb.PersistentClient(path=db_path)
    return _client


def _get_collection(client: Optional[chromadb.PersistentClient] = None) -> chromadb.Collection:
    """Get or create the research memory collection (cached).

    Uses SentenceTransformer 'all-MiniLM-L6-v2' for embeddings.
    The client and embedding function are cached to avoid reloading
    the sentence-transformer model on every call.

    Args:
        client: Optional ChromaDB client. Uses cached client if not provided.

    Returns:
        Cached ChromaDB Collection instance.
    """
    global _collection
    if _collection is None:
        if client is None:
            client = _get_client()
        embedding_fn = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
        _collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            embedding_function=embedding_fn,
        )
    return _collection


def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks.

    Args:
        text: The text to split into chunks.
        chunk_size: Maximum characters per chunk.
        overlap: Number of overlapping characters between chunks.

    Returns:
        List of text chunks.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk.strip())
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def save_report(topic: str, report: str, metadata: Optional[dict] = None) -> None:
    """Save a report into ChromaDB as chunked vectors for future retrieval.

    The report is split into overlapping chunks, each stored as a separate
    document with metadata about the topic and chunk position.

    Args:
        topic: The research topic.
        report: The full report content.
        metadata: Optional additional metadata to attach to each chunk.
    """
    if not report or not report.strip():
        return

    collection = _get_collection()
    chunks = _chunk_text(report)

    # Generate unique IDs and metadata for each chunk
    base_id = f"{topic.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d%H%M%S')}"
    ids = [f"{base_id}_{i}" for i in range(len(chunks))]

    base_meta = {
        "topic": topic,
        "timestamp": datetime.now().isoformat(),
    }
    if metadata:
        base_meta.update(metadata)

    metadatas = [
        {**base_meta, "chunk_index": i, "chunk_total": len(chunks)}
        for i in range(len(chunks))
    ]

    collection.add(
        documents=chunks,
        ids=ids,
        metadatas=metadatas,
    )

    print(f"  💾 ChromaDB: Stored {len(chunks)} chunks for '{topic}'")


def get_relevant_context(query: str, n_results: int = 3) -> str:
    """Query ChromaDB for context relevant to the given topic.

    Retrieves the top-n most similar chunks from past research reports.

    Args:
        query: The search query (typically the research topic).
        n_results: Maximum number of chunks to retrieve.

    Returns:
        Joined string of relevant context, or empty string if no context found.
    """
    try:
        collection = _get_collection()

        # Check if collection has any documents
        count = collection.count()
        if count == 0:
            return ""

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count),
        )

        documents = results.get("documents", [[]])[0]
        if not documents:
            return ""

        context = "\n\n---\n\n".join(documents)
        print(f"  📚 ChromaDB: Retrieved {len(documents)} relevant chunk(s) from memory")
        return context

    except Exception as e:
        print(f"  ⚠️  ChromaDB query error: {e}")
        return ""


def get_all_topics() -> list:
    """Get a list of unique research topics stored in ChromaDB.

    Returns:
        List of topic strings, sorted alphabetically.
    """
    try:
        collection = _get_collection()
        count = collection.count()
        if count == 0:
            return []

        # Get all metadata
        results = collection.get(limit=count)
        metadatas = results.get("metadatas", [])

        if not metadatas:
            return []

        # Extract unique topics
        topics = sorted(set(m["topic"] for m in metadatas if "topic" in m))
        return topics

    except Exception as e:
        print(f"  ⚠️  ChromaDB list topics error: {e}")
        return []
