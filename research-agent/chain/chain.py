"""LangChain LCEL pipeline for the research agent.

Builds three sequential chains (research → analysis → writing) using
LangChain Expression Language (LCEL) with ChatGroq as the LLM backend.
"""

import os
import time
from functools import lru_cache
from typing import Optional

from langchain_core.output_parsers import StrOutputParser
from langchain_groq import ChatGroq
from langsmith import traceable

from chain.prompts import RESEARCHER_PROMPT, ANALYST_PROMPT, WRITER_PROMPT


@lru_cache(maxsize=1)
def _get_llm() -> ChatGroq:
    """Get (and cache) the ChatGroq LLM instance configured from environment variables.

    Lazily initialized on first call so the module can be imported without
    GROQ_API_KEY being set (important for CrewAI-only usage).

    Returns:
        ChatGroq instance configured with API key and model.

    Raises:
        ValueError: If GROQ_API_KEY is not set.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found in environment. "
            "Set it in .env file or export GROQ_API_KEY=your-key"
        )

    model_name = os.getenv("LANGCHAIN_MODEL_NAME", "meta-llama/llama-4-scout-17b-16e-instruct")
    return ChatGroq(
        api_key=api_key,
        model=model_name,
        temperature=0.3,
        max_tokens=4096,
    )


def _get_chain() -> tuple:
    """Build and return the three sequential LCEL chains.

    Lazily initialized so the module can be imported without GROQ_API_KEY.

    Returns:
        Tuple of (research_chain, analysis_chain, writer_chain).
    """
    llm = _get_llm()
    research_chain = RESEARCHER_PROMPT | llm | StrOutputParser()
    analysis_chain = ANALYST_PROMPT | llm | StrOutputParser()
    writer_chain = WRITER_PROMPT | llm | StrOutputParser()
    return research_chain, analysis_chain, writer_chain


@traceable(name="langchain_research", run_type="chain")
def run_langchain(topic: str, memory_context: str = "") -> dict:
    """Run the LangChain research pipeline on a given topic.

    Executes three sequential steps:
    1. Research: Searches and summarizes findings on the topic
    2. Analysis: Identifies patterns and key themes from research
    3. Writing: Produces a structured report from the analysis

    The LLM and chains are lazily initialized on first call, so this
    function can be imported even without GROQ_API_KEY being set.

    Args:
        topic: The research topic to investigate.
        memory_context: Optional context from prior research (ChromaDB).

    Returns:
        dict with keys:
            - "research": Raw research output string
            - "analysis": Analysis output string
            - "report": Final written report string
    """
    research_chain, analysis_chain, writer_chain = _get_chain()

    print(f"  🔍 LangChain: Researching '{topic}'...")
    research = research_chain.invoke({
        "topic": topic,
        "memory_context": memory_context or "No prior context available.",
    })

    time.sleep(2)  # 🐌 Rate-limit spacer: stay within Groq 6000 TPM free tier

    print(f"  🧠 LangChain: Analyzing findings...")
    analysis = analysis_chain.invoke({"research": research})

    time.sleep(2)  # 🐌 Rate-limit spacer

    print(f"  ✍️  LangChain: Writing report...")
    report = writer_chain.invoke({"topic": topic, "analysis": analysis})

    return {
        "research": research,
        "analysis": analysis,
        "report": report,
    }
