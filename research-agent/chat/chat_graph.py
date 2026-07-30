"""Chat turn handler.

Deliberately NOT a LangGraph state machine like graph.py. That graph exists
to manage a multi-minute critic/revise/verify loop; a chat turn is two
sequential steps (retrieve, then stream a reply) where the only thing that
matters is time-to-first-token. Wrapping that in LangGraph would add
overhead without adding correctness — so this is a plain async generator.

Flow per turn:
    1. Pull the last WINDOW_TURNS messages verbatim (short-term memory)
    2. Query the session's ChromaDB collection for relevant uploaded-doc
       chunks AND relevant older turns that fell outside the window
    3. Stream the LLM's response token-by-token
    4. Persist the turn (both sides) back into conversation_buffer
""""""Chat turn handler.

Deliberately NOT a LangGraph state machine like graph.py. That graph exists
to manage a multi-minute critic/revise/verify loop; a chat turn is two
sequential steps (retrieve, then stream a reply) where the only thing that
matters is time-to-first-token. Wrapping that in LangGraph would add
overhead without adding correctness — so this is a plain async generator.

Flow per turn:
    1. Pull the last WINDOW_TURNS messages verbatim (short-term memory)
    2. Query the session's ChromaDB collection for relevant uploaded-doc
       chunks AND relevant older turns that fell outside the window
       (skipped entirely if the session has no stored content yet — see
       session_store.has_content)
    3. Stream the LLM's response token-by-token
    4. Persist the turn (both sides) in the background — the client
       doesn't wait on the embedding calls to see the reply finish
"""

import asyncio
from typing import AsyncGenerator

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage

from chat import conversation_buffer
from chat.chat_prompts import CHAT_SYSTEM_PROMPT, build_context_block
from llm_config import get_fast_llm
from memory import session_store


async def stream_chat_response(
    session_id: str, user_message: str, voice: bool = False
) -> AsyncGenerator[dict, None]:
    """Yields {"type": "token", "text": ...} then a final {"type": "done", "sources": [...]}.

    On error, yields {"type": "error", "message": ...} instead of raising,
    so the SSE stream can surface a clean message to the UI rather than
    dying silently mid-response.
    """
    conversation_buffer.create_session(session_id)

    # 1. Retrieve context — skip the ChromaDB round-trip entirely if this
    #    session has never had anything stored (the common case for a
    #    fresh chat's first few messages). This was the main hit to
    #    time-to-first-token.
    retrieved = []
    if session_store.has_content(session_id):
        retrieved = session_store.query_session(session_id, user_message, n_results=5)
    context_block = build_context_block(retrieved)

    # 2. Build the prompt
    system_text = CHAT_SYSTEM_PROMPT
    if voice:
        system_text += "\nVOICE_MODE: on\n"
    if context_block:
        system_text += f"\n{context_block}\n"

    messages = [SystemMessage(content=system_text)]
    for msg in conversation_buffer.get_windowed_history(session_id):
        if msg.role == "user":
            messages.append(HumanMessage(content=msg.content))
        else:
            messages.append(AIMessage(content=msg.content))
    messages.append(HumanMessage(content=user_message))

    # 3. Stream
    llm = get_fast_llm(temperature=0.4, max_tokens=1024 if not voice else 300)
    full_reply = []
    try:
        async for chunk in llm.astream(messages):
            text = chunk.content if hasattr(chunk, "content") else str(chunk)
            if text:
                full_reply.append(text)
                yield {"type": "token", "text": text}
    except Exception as e:
        yield {"type": "error", "message": f"LLM error: {e}"}
        return

    reply_text = "".join(full_reply)

    # 4. Persist both sides of the turn in the BACKGROUND. The two embed
    #    calls inside add_turn (user msg + assistant msg) used to run
    #    synchronously here, adding their latency after every single
    #    token had already finished streaming — the user was staring at
    #    a finished-looking reply while we blocked on embeddings. Fire
    #    them off instead; they'll land before the *next* retrieval
    #    query (which itself only happens once has_content() is true).
    asyncio.create_task(_persist_turn(session_id, user_message, reply_text))

    upload_sources = [c for c in retrieved if c["doc_type"] == "upload"]
    yield {
        "type": "done",
        "sources": [{"text": c["text"][:200], "source": c["source"], "doc_type": c["doc_type"]} for c in upload_sources],
    }


async def _persist_turn(session_id: str, user_message: str, reply_text: str) -> None:
    conversation_buffer.add_turn(session_id, "user", user_message)
    conversation_buffer.add_turn(session_id, "assistant", reply_text)
