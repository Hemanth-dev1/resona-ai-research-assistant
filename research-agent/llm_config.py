"""Unified LLM configuration.

Provides a single get_llm() factory that supports multiple providers.
Configured via LLM_PROVIDER env var: groq (default), openai, or anthropic.

Usage:
    from llm_config import get_llm, LLMProvider
    llm = get_llm()
    llm = get_llm(provider="openai", model="gpt-4o", temperature=0.5)
"""

import os
from enum import Enum
from functools import lru_cache
from typing import Any, Optional


class LLMProvider(str, Enum):
    """Supported LLM providers.

    GROQ is the primary (default) provider.  GEMINI is the automatic
    fallback when Groq's combined daily token budget is near exhaustion.
    """

    GROQ = "groq"
    GEMINI = "gemini"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Default models per provider
PROVIDER_DEFAULT_MODELS = {
    LLMProvider.GROQ: "llama-3.1-8b-instant",
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20240620",
}

# Fast (cheap/light) models for Planner, Research, Verification nodes
PROVIDER_FAST_MODELS = {
    LLMProvider.GROQ: "llama-3.1-8b-instant",
    LLMProvider.GEMINI: "gemini-2.5-flash",
    LLMProvider.OPENAI: "gpt-4o-mini",
    LLMProvider.ANTHROPIC: "claude-3-5-haiku-20241022",
}

# Capable (high-quality) models for Analyst, Writer, Critic nodes
PROVIDER_CAPABLE_MODELS = {
    LLMProvider.GROQ: "llama-3.3-70b-versatile",
    LLMProvider.GEMINI: "gemini-2.5-flash",  # Established stable, best for high-volume fallback
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20240620",
}

# Base URLs per provider
PROVIDER_BASE_URLS = {
    LLMProvider.GROQ: "https://api.groq.com/openai/v1",
    LLMProvider.GEMINI: None,  # Uses Google AI default
    LLMProvider.OPENAI: None,  # Uses OpenAI default
    LLMProvider.ANTHROPIC: None,  # Uses Anthropic default
}

# API key env vars per provider
PROVIDER_API_KEYS = {
    LLMProvider.GROQ: "GROQ_API_KEY",
    LLMProvider.GEMINI: "GOOGLE_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
}


def get_provider() -> LLMProvider:
    """Get the configured LLM provider from environment.

    Returns:
        LLMProvider enum value. Defaults to GROQ if LLM_PROVIDER is not set.
    """
    provider_str = os.getenv("LLM_PROVIDER", "groq").lower().strip()
    try:
        return LLMProvider(provider_str)
    except ValueError:
        valid = ", ".join(p.value for p in LLMProvider)
        print(f"  ⚠️  Unknown LLM_PROVIDER='{provider_str}'. Valid: {valid}. Falling back to groq.")
        return LLMProvider.GROQ


def get_model_name(provider: Optional[LLMProvider] = None) -> str:
    """Get the configured model name for the given provider.

    Checks LLM_MODEL env var first, then falls back to provider default.

    Args:
        provider: The LLM provider. If None, uses the configured provider.

    Returns:
        Model name string.
    """
    if provider is None:
        provider = get_provider()

    return os.getenv("LLM_MODEL", PROVIDER_DEFAULT_MODELS[provider])


def get_fast_model_name(provider: Optional[LLMProvider] = None) -> str:
    """Get the fast/cheap model name for low-complexity tasks.

    Checks LLM_MODEL_FAST env var first, then falls back to PROVIDER_FAST_MODELS.
    Falls back to the default model name if LLM_MODEL_FAST is not set.

    Args:
        provider: The LLM provider. If None, uses the configured provider.

    Returns:
        Fast model name string (e.g., llama-3.1-8b-instant, gpt-4o-mini).
    """
    if provider is None:
        provider = get_provider()

    return os.getenv("LLM_MODEL_FAST", PROVIDER_FAST_MODELS.get(provider, PROVIDER_DEFAULT_MODELS[provider]))


def get_capable_model_name(provider: Optional[LLMProvider] = None) -> str:
    """Get the capable/high-quality model for complex reasoning tasks.

    Checks LLM_MODEL_CAPABLE env var first, then falls back to PROVIDER_CAPABLE_MODELS.

    Args:
        provider: The LLM provider. If None, uses the configured provider.

    Returns:
        Capable model name string (e.g., llama-3.3-70b-versatile, gpt-4o).
    """
    if provider is None:
        provider = get_provider()

    model = os.getenv("LLM_MODEL_CAPABLE", PROVIDER_CAPABLE_MODELS.get(provider))
    if not model:
        # Fall back to the default LLM_MODEL if no capable default is defined
        return os.getenv("LLM_MODEL", PROVIDER_DEFAULT_MODELS[provider])
    return model


def get_api_key(provider: Optional[LLMProvider] = None) -> Optional[str]:
    """Get the API key for the given provider.

    Args:
        provider: The LLM provider. If None, uses the configured provider.

    Returns:
        API key string, or None if not found.
    """
    if provider is None:
        provider = get_provider()

    env_var = PROVIDER_API_KEYS[provider]
    return os.getenv(env_var)


@lru_cache(maxsize=4)
def get_llm(
    provider: Optional[LLMProvider] = None,
    model: Optional[str] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Any:
    """Get a configured LLM instance for the specified provider.

    Args:
        provider: LLM provider. If None, uses LLM_PROVIDER env var.
        model: Model name. If None, uses LLM_MODEL env var or provider default.
        temperature: Sampling temperature (default: 0.3).
        max_tokens: Maximum tokens in response (default: 4096).

    Returns:
        An LLM instance (ChatGroq, ChatOpenAI, or ChatAnthropic).

    Raises:
        ValueError: If the provider requires an API key that is not set.
        ImportError: If the provider package is not installed.
    """
    provider = provider or get_provider()
    model = model or get_model_name(provider)
    api_key = get_api_key(provider)

    if not api_key:
        env_var = PROVIDER_API_KEYS[provider]
        raise ValueError(
            f"{env_var} not found in environment. "
            f"Set it in .env or export {env_var}=your-key"
        )

    if provider == LLMProvider.GROQ:
        try:
            from langchain_groq import ChatGroq

            # Also set CrewAI-compatible env vars
            os.environ["OPENAI_BASE_URL"] = PROVIDER_BASE_URLS[provider]
            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_MODEL_NAME"] = model

            llm = ChatGroq(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            raise ImportError("langchain-groq not installed. Run: pip install langchain-groq")

    elif provider == LLMProvider.GEMINI:
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI

            llm = ChatGoogleGenerativeAI(
                google_api_key=api_key,
                model=model,
                temperature=temperature,
                max_output_tokens=max_tokens if max_tokens else None,
            )
        except ImportError:
            raise ImportError(
                "langchain-google-genai not installed. Run: pip install langchain-google-genai"
            )

    elif provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI

            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_MODEL_NAME"] = model

            llm = ChatOpenAI(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            raise ImportError("langchain-openai not installed. Run: pip install langchain-openai")

    elif provider == LLMProvider.ANTHROPIC:
        try:
            from langchain_anthropic import ChatAnthropic

            llm = ChatAnthropic(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            raise ImportError(
                "langchain-anthropic not installed. Run: pip install langchain-anthropic"
            )

    else:
        raise ValueError(f"Unsupported provider: {provider}")

    return llm


def get_fast_llm(
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.3,
    max_tokens: int = 4096,
) -> Any:
    """Get a fast/cheap LLM for simple tasks (Planner, Research, Verification).

    Includes **runtime 429 catch-and-fallback**: if the primary provider
    returns a rate-limit error (429), the call is automatically retried on
    Gemini Flash without crashing the pipeline.

    Also includes **provider-level pre-check fallback**: if Groq's combined
    daily budget is near exhaustion (tracked in memory), routes to Gemini
    before making the call.

    Uses LLM_MODEL_FAST or the provider's fast model default.

    Args:
        provider: LLM provider. If None, uses LLM_PROVIDER env var.
        temperature: Sampling temperature (default: 0.3).
        max_tokens: Maximum tokens in response (default: 4096).

    Returns:
        An LLM instance configured with the fast model, with automatic
        fallback to Gemini on rate-limit errors.
    """
    provider = provider or get_provider()
    model = get_fast_model_name(provider)

    # ── Step 1: Provider-level pre-check (in-memory tracker) ───────────
    from token_tracker import check_provider_fallback
    fb_provider, fb_model = check_provider_fallback(
        "groq",
        fallback_model_name="gemini-2.5-flash",
        fallback_provider=LLMProvider.GEMINI,
    )
    if fb_provider is not None:
        model = get_fast_model_name(fb_provider)
        return get_llm(provider=fb_provider, model=model, temperature=temperature, max_tokens=max_tokens)

    # ── Step 2: Primary LLM ────────────────────────────────────────────
    primary = get_llm(provider=provider, model=model, temperature=temperature, max_tokens=max_tokens)

    # ── Step 3: Runtime 429 fallback — LangChain retries on Gemini ─────
    _gemini_key = get_api_key(LLMProvider.GEMINI)
    if _gemini_key:
        try:
            _gemini_model = get_fast_model_name(LLMProvider.GEMINI)
            _gemini_llm = get_llm(provider=LLMProvider.GEMINI, model=_gemini_model,
                                  temperature=temperature, max_tokens=max_tokens)
            print(f"  ⚡ get_fast_llm: adding runtime 429 fallback to {LLMProvider.GEMINI.value}/{_gemini_model}")
            return primary.with_fallbacks([_gemini_llm])
        except Exception:
            pass  # If Gemini setup fails, just use primary without fallback

    return primary


def get_capable_llm(
    provider: Optional[LLMProvider] = None,
    temperature: float = 0.3,
    max_tokens: int = 8192,
) -> Any:
    """Get a capable/high-quality LLM for complex reasoning (Analyst, Writer, Critic).

    Includes **runtime 429 catch-and-fallback**: if the primary LLM returns
    a rate-limit error, the call is automatically retried on Gemini Flash,
    then on Groq 8B, without crashing the pipeline.

    Also includes **provider-level pre-check**: the fallback chain is:
        1. Groq 70B (primary) — fastest, highest quality
        2. Gemini Flash (cross-provider runtime fallback) — when Groq 70B
           returns a 429
        3. Groq 8B (tertiary runtime fallback) — if both 70B and Gemini fail

    Uses LLM_MODEL_CAPABLE or the provider's capable model default.

    Args:
        provider: LLM provider. If None, uses LLM_PROVIDER env var.
        temperature: Sampling temperature (default: 0.3).
        max_tokens: Maximum tokens in response (default: 8192 for longer
            outputs; clamped to 4096 when falling back to 8B models).

    Returns:
        An LLM instance with automatic runtime fallback on rate-limit errors.
    """
    provider = provider or get_provider()
    configured_model = get_capable_model_name(provider)
    fast_model = get_fast_model_name(provider)

    # ── Step 1: Model-level soft guard (in-memory tracker) ─────────────
    from token_tracker import check_model_fallback
    model = check_model_fallback(configured_model, fallback_model=fast_model)

    # ── Max_tokens clamping for 8B models ─────────────────────────────
    _mt = max_tokens
    if model and "8b" in model.lower() and _mt > 6000:
        _mt = 4096

    # ── Step 2: Primary LLM ────────────────────────────────────────────
    primary = get_llm(provider=provider, model=model, temperature=temperature, max_tokens=_mt)

    # ── Step 3: Runtime 429 fallbacks ──────────────────────────────────
    _fallbacks = []

    # Fallback A: Gemini Flash (if API key is set)
    _gemini_key = get_api_key(LLMProvider.GEMINI)
    if _gemini_key:
        try:
            _gemini_model = get_capable_model_name(LLMProvider.GEMINI)
            _gemini_llm = get_llm(provider=LLMProvider.GEMINI, model=_gemini_model,
                                  temperature=temperature,
                                  max_tokens=min(_mt, 4096))
            _fallbacks.append(_gemini_llm)
        except Exception:
            pass

    # Fallback B: Groq 8B (same provider, cheaper model — only if primary is 70B)
    if "70b" in model.lower() or "70b" in configured_model.lower():
        try:
            _groq_8b_llm = get_llm(provider=LLMProvider.GROQ, model=fast_model,
                                   temperature=temperature,
                                   max_tokens=min(_mt, 4096))
            _fallbacks.append(_groq_8b_llm)
        except Exception:
            pass

    if _fallbacks:
        print(f"  ⚡ get_capable_llm: adding runtime 429 fallbacks ({len(_fallbacks)} tiers)")
        return primary.with_fallbacks(_fallbacks)

    return primary


