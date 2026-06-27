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
    """Supported LLM providers."""

    GROQ = "groq"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"


# Default models per provider
PROVIDER_DEFAULT_MODELS = {
    LLMProvider.GROQ: "llama-3.1-8b-instant",
    LLMProvider.OPENAI: "gpt-4o",
    LLMProvider.ANTHROPIC: "claude-3-5-sonnet-20240620",
}

# Base URLs per provider
PROVIDER_BASE_URLS = {
    LLMProvider.GROQ: "https://api.groq.com/openai/v1",
    LLMProvider.OPENAI: None,  # Uses OpenAI default
    LLMProvider.ANTHROPIC: None,  # Uses Anthropic default
}

# API key env vars per provider
PROVIDER_API_KEYS = {
    LLMProvider.GROQ: "GROQ_API_KEY",
    LLMProvider.OPENAI: "OPENAI_API_KEY",
    LLMProvider.ANTHROPIC: "ANTHROPIC_API_KEY",
}

# CrewAI-compatible env var mappings (set via os.environ)
CREWAI_ENV_MAP = {
    LLMProvider.GROQ: {
        "OPENAI_BASE_URL": "https://api.groq.com/openai/v1",
        "OPENAI_API_KEY": "GROQ_API_KEY",
        "OPENAI_MODEL_NAME": "LLM_MODEL",
    },
    LLMProvider.OPENAI: {
        "OPENAI_API_KEY": "OPENAI_API_KEY",
        "OPENAI_MODEL_NAME": "LLM_MODEL",
    },
    LLMProvider.ANTHROPIC: {
        "ANTHROPIC_API_KEY": "ANTHROPIC_API_KEY",
    },
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


@lru_cache(maxsize=1)
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

            return ChatGroq(
                api_key=api_key,
                model=model,
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except ImportError:
            raise ImportError("langchain-groq not installed. Run: pip install langchain-groq")

    elif provider == LLMProvider.OPENAI:
        try:
            from langchain_openai import ChatOpenAI

            os.environ["OPENAI_API_KEY"] = api_key
            os.environ["OPENAI_MODEL_NAME"] = model

            return ChatOpenAI(
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

            return ChatAnthropic(
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


def setup_crewai_env(provider: Optional[LLMProvider] = None) -> None:
    """Set environment variables for CrewAI to use the configured provider.

    CrewAI reads OpenAI-compatible env vars (OPENAI_API_KEY, OPENAI_BASE_URL,
    OPENAI_MODEL_NAME) regardless of the actual provider.

    Args:
        provider: The LLM provider. If None, uses the configured provider.
    """
    if provider is None:
        provider = get_provider()

    api_key = get_api_key(provider)
    model = get_model_name(provider)

    if provider == LLMProvider.GROQ:
        os.environ["OPENAI_BASE_URL"] = PROVIDER_BASE_URLS[LLMProvider.GROQ]
        os.environ["OPENAI_API_KEY"] = api_key or ""
        os.environ["OPENAI_MODEL_NAME"] = model

    elif provider == LLMProvider.OPENAI:
        os.environ.pop("OPENAI_BASE_URL", None)
        os.environ["OPENAI_API_KEY"] = api_key or ""
        os.environ["OPENAI_MODEL_NAME"] = model

    elif provider == LLMProvider.ANTHROPIC:
        # Anthropic uses its own env var, but CrewAI needs OpenAI-compatible
        # This uses the LiteLLM or custom approach
        os.environ["OPENAI_API_KEY"] = api_key or ""
        os.environ["OPENAI_MODEL_NAME"] = model
