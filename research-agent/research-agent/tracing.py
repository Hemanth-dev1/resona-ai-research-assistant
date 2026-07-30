"""LangSmith tracing initialization for Resona.

Sets up OpenTelemetry-based tracing for both CrewAI and LangChain pipelines.
All agent steps, token counts, latencies, and costs are sent to LangSmith.

Requirements:
    - LANGSMITH_API_KEY in .env or environment
    - LANGSMITH_PROJECT (optional, defaults to "resona")
    - LANGCHAIN_TRACING_V2=true (optional, enables LangChain tracing)
"""

import os
import sys


def setup_tracing() -> bool:
    """Initialize LangSmith tracing with OpenTelemetry instrumentors.

    This should be called once at application startup before any agent runs.

    Sets up:
    1. OpenTelemetry TracerProvider with LangSmith OtelSpanProcessor
    2. CrewAI instrumentation for agent step tracing
    3. OpenAI instrumentation for LLM call tracing (works with Groq's OpenAI-compatible API)

    Returns:
        True if tracing was initialized successfully, False if LANGSMITH_API_KEY is missing.

    Prints status messages so users know tracing is active.
    """
    api_key = os.getenv("LANGSMITH_API_KEY")
    project = os.getenv("LANGSMITH_PROJECT", "resona-ai-research-assistant")

    if not api_key:
        print("  ℹ️  LangSmith tracing: DISABLED (set LANGSMITH_API_KEY in .env)")
        return False

    try:
        from langsmith.integrations.otel import OtelSpanProcessor
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider

        # Set up tracer provider with LangSmith span processor
        tracer_provider = TracerProvider()
        tracer_provider.add_span_processor(OtelSpanProcessor())
        trace.set_tracer_provider(tracer_provider)

        # Set LangSmith environment variables
        os.environ["LANGCHAIN_TRACING_V2"] = "true"
        os.environ["LANGSMITH_TRACING"] = "true"
        os.environ["LANGSMITH_PROJECT"] = project

        # Instrument OpenAI-compatible API (captures LLM calls — works with Groq)
        try:
            from opentelemetry.instrumentation.openai import OpenAIInstrumentor

            OpenAIInstrumentor().instrument(tracer_provider=tracer_provider)
            print(f"  📡 LangSmith: OpenAI instrumentation active (covers Groq LLM calls)")
        except ImportError:
            print("  ⚠️  LangSmith: opentelemetry-instrumentation-openai not available")

        print(f"  ✅ LangSmith tracing initialized → https://smith.langchain.com")
        return True

    except ImportError as e:
        print(f"  ⚠️  LangSmith tracing setup failed: {e}")
        print("     Install with: pip install langsmith opentelemetry-sdk")
        return False
    except Exception as e:
        print(f"  ⚠️  LangSmith tracing error: {e}")
        return False



