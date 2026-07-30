"""Pydantic models for the multi-session chatbot.

Kept separate from models.py (which types the deep-research pipeline's
outputs) since chat has a different lifecycle: many short turns instead
of one long structured report.
"""

from datetime import datetime
from typing import List, Literal, Optional

from pydantic import BaseModel, Field


class ChatMessage(BaseModel):
    role: Literal["user", "assistant"]
    content: str
    timestamp: str = Field(default_factory=lambda: datetime.utcnow().isoformat())


class ChatTurnRequest(BaseModel):
    session_id: str
    message: str
    voice: bool = False  # hint from client: response will be read aloud (kept concise)


class SourceChunk(BaseModel):
    text: str
    source: str
    doc_type: Literal["upload", "turn"]


class ChatTurnResponse(BaseModel):
    session_id: str
    reply: str
    sources: List[SourceChunk] = Field(default_factory=list)


class SessionInfo(BaseModel):
    session_id: str
    created_at: str
    message_count: int
    documents: List[str] = Field(default_factory=list)
