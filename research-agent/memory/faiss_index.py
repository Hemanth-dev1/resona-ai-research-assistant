"""In-session FAISS index for web source document similarity search.

This index is in-memory only and resets each session. It allows the Writer
agent to retrieve and cite specific passages from web sources gathered during
the research phase.
"""

from typing import Optional

import numpy as np
from sentence_transformers import SentenceTransformer


# Module-level state (in-memory, resets per session)
_index = None
_documents: list[str] = []
_embedder: Optional[SentenceTransformer] = None


def _get_embedder() -> SentenceTransformer:
    """Get or initialize the sentence transformer model.

    Returns:
        SentenceTransformer instance for 'all-MiniLM-L6-v2'.
    """
    global _embedder
    if _embedder is None:
        _embedder = SentenceTransformer("all-MiniLM-L6-v2")
    return _embedder


def build_index(documents: list[str]) -> None:
    """Build a FAISS index from a list of documents.

    Each document is embedded using the sentence transformer, then added
    to a FAISS IndexFlatL2 index for fast similarity search.

    Args:
        documents: List of document strings to index.
    """
    global _index, _documents

    if not documents:
        print("  ⚠️  FAISS: No documents to index")
        return

    try:
        import faiss
    except ImportError:
        print("  ⚠️  FAISS not installed. Skipping index build.")
        return

    _documents = documents
    embedder = _get_embedder()

    # Embed all documents
    embeddings = embedder.encode(documents, show_progress_bar=False)
    dimension = embeddings.shape[1]

    # Build the index
    _index = faiss.IndexFlatL2(dimension)
    _index.add(np.array(embeddings).astype(np.float32))

    print(f"  📊 FAISS: Indexed {len(documents)} documents (dim={dimension})")


def search_index(query: str, k: int = 3) -> list[str]:
    """Search the FAISS index for documents similar to the query.

    Args:
        query: Search query string.
        k: Number of top results to return.

    Returns:
        List of matching document strings, or empty list if index is empty.
    """
    global _index, _documents

    if _index is None or not _documents:
        return []

    try:
        import faiss
    except ImportError:
        return []

    embedder = _get_embedder()
    query_embedding = embedder.encode([query], show_progress_bar=False)

    # Search the index
    k_actual = min(k, len(_documents))
    distances, indices = _index.search(np.array(query_embedding).astype(np.float32), k_actual)

    results = []
    for idx in indices[0]:
        if 0 <= idx < len(_documents):
            results.append(_documents[idx])

    return results
