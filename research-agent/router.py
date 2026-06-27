"""Router that selects the appropriate research pipeline based on mode.

Dispatches between:
- "crewai": The existing CrewAI multi-agent pipeline (default)
- "langchain": The new LangChain LCEL chain with RAG memory via ChromaDB

Both modes benefit from:
- Retry logic via tenacity (exponential backoff)
- Self-correcting critic loop (score + feedback loop)
- LangSmith tracing via @traceable
- Pydantic-validated PipelineResult output
"""

import os
import sys
import time
from typing import Optional

from langsmith import traceable

from chain.chain import run_langchain
from memory.chroma_store import get_relevant_context, save_report
from retry_utils import safe_invoke
from schemas.parser import parse_report


@traceable(name="resona_pipeline", run_type="chain")
def run(topic: str, mode: str = "crewai") -> str:
    """Run the research pipeline in the specified mode.

    Args:
        topic: The research topic to investigate.
        mode: Either "crewai" (default) or "langchain".

    Returns:
        The generated report content as a string.

    Raises:
        ValueError: If mode is not "crewai" or "langchain".
    """
    start_time = time.time()

    if mode == "langchain":
        # Retrieve relevant context from past research
        memory_context = get_relevant_context(topic)
        if memory_context:
            print(f"  🧠 RAG context loaded from ChromaDB memory")

        # Run the LangChain pipeline (with retry on chain.invoke)
        result = safe_invoke(
            run_langchain, topic, memory_context=memory_context,
            error_message="LangChain pipeline failed",
        )

        if isinstance(result, dict) and not result.get("success", True):
            error_msg = result.get("error", "LangChain pipeline failed")
            return f"❌ Pipeline error: {error_msg}"

        report = result.get("report", "")

        # Run self-correcting critic loop
        from critic import run_critic_loop

        report, critiques = run_critic_loop(topic, report)
        print(f"  📝 Critic loop: {len(critiques)} iteration(s)")

        # Validate report structure with Pydantic model
        parsed = parse_report(report, topic)
        if parsed is None:
            print("  ⚠️  Report failed Pydantic validation — structure may be malformed")
        else:
            print(f"  ✅ Report validated: {len(parsed.sources)} sources, {len(parsed.key_insights)} insights")

        # Save to ChromaDB for future retrieval
        save_report(topic, report)

        duration = time.time() - start_time
        print(f"  ⏱️  Pipeline completed in {duration:.1f}s")
        return report

    elif mode == "crewai":
        # Import and run the existing CrewAI pipeline
        try:
            from agents import make_agents
            from tasks import make_tasks
        except ImportError:
            print("❌ Could not import CrewAI modules. Make sure agents.py and tasks.py exist.")
            sys.exit(1)

        from crewai import Crew, Process

        # Set up LLM from unified config
        from llm_config import setup_crewai_env

        setup_crewai_env()

        researcher, analyst, writer = make_agents()
        tasks = make_tasks(topic, researcher, analyst, writer)

        crew = Crew(
            agents=[researcher, analyst, writer],
            tasks=tasks,
            process=Process.sequential,
            verbose=True,
        )

        print(f"\n🚀 Starting research on: '{topic}'\n")

        # Run with retry logic
        result = safe_invoke(crew.kickoff, error_message="CrewAI pipeline failed")
        if isinstance(result, dict) and not result.get("success", True):
            error_msg = result.get("error", "CrewAI pipeline failed")
            return f"❌ Pipeline error: {error_msg}"

        report = result.raw if hasattr(result, "raw") else str(result)

        # Validate report structure with Pydantic model
        parsed = parse_report(report, topic)
        if parsed is None:
            print("  ⚠️  Report failed Pydantic validation — structure may be malformed")
        else:
            print(f"  ✅ Report validated: {len(parsed.sources)} sources, {len(parsed.key_insights)} insights")

        # Run self-correcting critic loop
        from critic import run_critic_loop

        report, critiques = run_critic_loop(topic, report)
        print(f"  📝 Critic loop: {len(critiques)} iteration(s)")

        # Save to ChromaDB for future retrieval
        save_report(topic, report)

        duration = time.time() - start_time
        print(f"  ⏱️  Pipeline completed in {duration:.1f}s")
        return report

    else:
        raise ValueError(f"Unknown mode: '{mode}'. Use 'crewai' or 'langchain'.")
