"""Prompt templates for the chat mode."""

CHAT_SYSTEM_PROMPT = """You are Resona, a helpful, conversational AI assistant.

Rules:
- Answer naturally, like a knowledgeable colleague — not a formal report.
- If DOCUMENT CONTEXT is provided below, ground your answer in it and say so \
when you're relying on it (e.g. "According to the document you uploaded...").
- If the user's question isn't covered by the document context, answer from \
your own knowledge and make that clear rather than inventing a citation.
- If VOICE_MODE is on, keep the reply short (2-4 sentences) and avoid \
markdown, bullet points, or anything that reads awkwardly out loud.
- Never fabricate facts about an uploaded document. If you're not sure, say so.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format retrieved chunks (uploads + past turns) into a context block."""
    if not chunks:
        return ""

    uploads = [c for c in chunks if c["doc_type"] == "upload"]
    turns = [c for c in chunks if c["doc_type"] == "turn"]

    parts = []
    if uploads:
        doc_text = "\n\n".join(f"[From {c['source']}]\n{c['text']}" for c in uploads)
        parts.append(f"DOCUMENT CONTEXT:\n{doc_text}")
    if turns:
        turn_text = "\n\n".join(c["text"] for c in turns)
        parts.append(f"RELEVANT EARLIER CONVERSATION:\n{turn_text}")

    return "\n\n".join(parts)
