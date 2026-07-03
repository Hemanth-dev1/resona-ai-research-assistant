"""Token usage tracking across LLM models with daily limits.

Tracks total tokens consumed per model, resets at midnight UTC.
Provides a soft guard that automatically falls back from expensive
models (like 70B) to cheaper ones when approaching daily limits.

Usage:
    from token_tracker import get_tracker

    tracker = get_tracker()
    tracker.record("llama-3.3-70b-versatile", prompt_tokens=100, completion_tokens=50)
    print(tracker.get_usage("llama-3.3-70b-versatile"))
    # → {"prompt": 100, "completion": 50, "total": 150}

    if tracker.should_fallback("llama-3.3-70b-versatile"):
        print("Using 8B model instead")
"""

import os
from datetime import datetime, timezone
from typing import Any, Optional


# ── Known daily limits per model (free-tier TPD) ───────────────────────────
MODEL_DAILY_LIMITS: dict[str, int] = {
    "llama-3.3-70b-versatile": 100_000,  # Groq
    "llama-3.1-8b-instant": 500_000,     # Groq
    "gemini-3.5-flash": 1_000_000,        # Google — 1M TPM, effectively very high daily
}

# ── Known daily limits per provider ────────────────────────────────────────
# Used for provider-level fallback (e.g. Groq exhausted → Gemini)
PROVIDER_DAILY_LIMITS: dict[str, int] = {
    "groq": 600_000,   # Combined 70B(100K) + 8B(500K) ≈ 600K TPD practical
    "google": 1_500_000,  # 1500 requests/day * avg tokens
}

# Soft guard fraction of the daily limit (default 90%)
SOFT_GUARD_FRACTION = float(os.getenv("TOKEN_SOFT_GUARD_PCT", "0.9"))


class TokenTracker:
    """Tracks token usage per model, auto-resets at midnight UTC."""

    def __init__(self):
        self._usage: dict[str, dict] = {}
        self._last_reset_date: Optional[str] = None

    # ── Lifecycle ──────────────────────────────────────────────────────────

    def _ensure_reset(self):
        """Reset all counters if the UTC date has changed."""
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        if self._last_reset_date is None:
            self._last_reset_date = today
        elif self._last_reset_date != today:
            old = self._last_reset_date
            self._usage = {}
            self._last_reset_date = today
            print(f"  🔄 Token tracker: reset counters ({old} → {today})")

    # ── Recording ──────────────────────────────────────────────────────────

    def record(self, model: str, prompt_tokens: int, completion_tokens: int):
        """Record token usage for a single LLM call.

        Args:
            model: Model name (e.g. 'llama-3.3-70b-versatile').
            prompt_tokens: Input tokens consumed.
            completion_tokens: Output tokens generated.
        """
        self._ensure_reset()
        entry = self._usage.setdefault(model, {"prompt": 0, "completion": 0, "total": 0})
        entry["prompt"] += prompt_tokens
        entry["completion"] += completion_tokens
        entry["total"] += prompt_tokens + completion_tokens

    # ── Queries ────────────────────────────────────────────────────────────

    def get_usage(self, model: str) -> dict:
        """Get cumulative usage for a model.

        Returns:
            Dict with keys 'prompt', 'completion', 'total'.
        """
        self._ensure_reset()
        return dict(self._usage.get(model, {"prompt": 0, "completion": 0, "total": 0}))

    @staticmethod
    def get_limit(model: str) -> int:
        """Get the known daily token limit for a model.

        Returns the limit if known, otherwise returns 100_000 as a safe default.
        """
        return MODEL_DAILY_LIMITS.get(model, 100_000)

    def should_fallback(self, model: str) -> bool:
        """Check whether the model has exceeded the soft-guard threshold.

        The soft-guard threshold is SOFT_GUARD_FRACTION (default 90 %) of the
        model's daily limit.  Returns True when usage ≥ threshold.

        Args:
            model: Model name to check.

        Returns:
            True if usage ≥ threshold (i.e. a fallback model should be used).
        """
        self._ensure_reset()
        usage = self.get_usage(model)
        limit = self.get_limit(model)
        threshold = int(limit * SOFT_GUARD_FRACTION)
        total = usage.get("total", 0)

        if total >= threshold:
            print(
                f"  ⚠️  Token soft guard: {model} at {total}/{limit} tokens "
                f"({(total/limit)*100:.1f}%) ≥ {threshold}-token threshold"
            )
            return True
        return False

    # ── Provider-level tracking ──────────────────────────────────────────

    def get_provider_usage(self, provider_key: str) -> int:
        """Sum total tokens across all models belonging to a provider.

        Args:
            provider_key: e.g. ``"groq"`` or ``"google"``.  Matches models
                whose name starts with a key phrase associated with the provider.

        Returns:
            Cumulative total tokens for all models of that provider.
        """
        self._ensure_reset()
        # Map provider keys to model-name substrings
        model_prefixes = {
            "groq": ["llama"],
            "google": ["gemini"],
            "openai": ["gpt"],
            "anthropic": ["claude"],
        }
        prefixes = model_prefixes.get(provider_key, [provider_key])
        total = 0
        for model, usage in self._usage.items():
            if any(p in model.lower() for p in prefixes):
                total += usage.get("total", 0)
        return total

    def get_provider_limit(self, provider_key: str) -> int:
        """Get the daily token limit for a provider.

        Returns the configured limit or a safe default (500K).
        """
        return PROVIDER_DAILY_LIMITS.get(provider_key, 500_000)

    def should_fallback_provider(self, provider_key: str) -> bool:
        """Check whether a provider has crossed its soft-guard threshold.

        Args:
            provider_key: e.g. ``"groq"``.

        Returns:
            True if the provider's combined usage is ≥ SOFT_GUARD_FRACTION
            of its daily limit.
        """
        self._ensure_reset()
        total = self.get_provider_usage(provider_key)
        limit = self.get_provider_limit(provider_key)
        threshold = int(limit * SOFT_GUARD_FRACTION)

        if total >= threshold:
            print(
                f"  ⚠️  Provider soft guard: {provider_key} at {total}/{limit} tokens "
                f"({(total/limit)*100:.1f}%) ≥ {threshold}-token threshold"
            )
            return True
        return False

    # ── Diagnostics ────────────────────────────────────────────────────────

    def log_status(self):
        """Print current token usage vs limits for all tracked models."""
        self._ensure_reset()
        if not self._usage:
            print("  📊 Token usage today: no usage recorded yet")
            return

        lines = ["  📊 Token usage today:"]
        for model, usage in sorted(self._usage.items()):
            total = usage.get("total", 0)
            limit = self.get_limit(model)
            pct = (total / limit) * 100 if limit > 0 else 0
            flag = " ⚠️  near limit" if total >= int(limit * SOFT_GUARD_FRACTION) else ""
            lines.append(f"    {model}: {total}/{limit} tokens ({pct:.1f}%){flag}")
        print("\n".join(lines))


# ── Singleton ──────────────────────────────────────────────────────────────

_tracker: Optional[TokenTracker] = None


def get_tracker() -> TokenTracker:
    """Get the global token tracker singleton."""
    global _tracker
    if _tracker is None:
        _tracker = TokenTracker()
    return _tracker


# ── Convenience wrappers ───────────────────────────────────────────────────

def record_tokens(model: str, prompt_tokens: int, completion_tokens: int):
    """Record token usage.  Safe to call from anywhere in the codebase."""
    get_tracker().record(model, prompt_tokens, completion_tokens)


def check_model_fallback(
    configured_model: str,
    fallback_model: str = "llama-3.1-8b-instant",
) -> str:
    """Return the fallback model if the configured model is above its soft guard.

    Args:
        configured_model: The primary (expensive) model to check.
        fallback_model: The cheaper fallback model.

    Returns:
        The fallback model name if the guard is triggered, otherwise the
        original configured model.
    """
    tracker = get_tracker()
    if tracker.should_fallback(configured_model):
        print(
            f"  ⚠️  Auto-fallback (model): {configured_model} → {fallback_model} "
            f"(soft guard triggered)"
        )
        return fallback_model
    return configured_model


def check_provider_fallback(
    primary_provider_key: str,
    fallback_model_name: str,
    fallback_provider: Any,  # LLMProvider enum value
) -> tuple[Any, Optional[str]]:
    """Return a fallback ``(provider, model_name)`` pair if the primary
    provider has exceeded its soft-guard threshold **and** the fallback
    provider has its API key configured.

    Used for cross-provider failover: when Groq's combined token budget
    is near exhaustion, new calls are routed to Gemini Flash instead.

    Args:
        primary_provider_key: Provider identifier e.g. ``"groq"``.
        fallback_model_name: Model to use on the fallback provider.
        fallback_provider: ``LLMProvider`` enum value for the fallback.

    Returns:
        ``(fallback_provider, fallback_model_name)`` if the guard
        triggered and the fallback API key is present.
        ``(None, None)`` if no fallback is needed or the fallback
        provider is unavailable (no API key set).
    """
    from llm_config import LLMProvider, get_api_key as _get_api_key

    tracker = get_tracker()
    if not tracker.should_fallback_provider(primary_provider_key):
        return None, None  # No fallback needed

    # Guard: verify the fallback provider has an API key configured
    from llm_config import PROVIDER_API_KEYS as _PROV_API_KEYS
    fallback_key = _get_api_key(fallback_provider)
    if not fallback_key:
        env_var = _PROV_API_KEYS.get(fallback_provider, "?")
        print(
            f"  ⚠️  Provider fallback: {primary_provider_key} exhausted, but "
            f"{fallback_provider.value} unavailable — no API key set. "
            f"Set {env_var} in .env"
        )
        return None, None

    print(
        f"  ⚠️  Provider fallback: {primary_provider_key} exhausted → "
        f"routing to {fallback_provider.value}/{fallback_model_name}"
    )
    return fallback_provider, fallback_model_name


# ── Extract tokens from a raw LLM response (AIMessage) ────────────────────

def record_from_response(model_name: str, response: Any) -> None:
    """Extract token usage from an AIMessage and record it.

    Call this immediately after ``llm.invoke(messages)`` returns, before
    extracting ``.content``.  Handles both the ``usage_metadata`` format
    (newer LangChain / Groq) and ``response_metadata → token_usage``
    (older format).

    Safe to call on string responses too — silently returns if the response
    is not an AIMessage with usage data.

    Args:
        model_name: Model name e.g. ``"llama-3.3-70b-versatile"``.
        response: The object returned by ``llm.invoke()``.
    """
    try:
        # Newer LangChain / Groq: response.usage_metadata
        usage = getattr(response, "usage_metadata", None)
        if usage:
            prompt = (
                usage.input_tokens
                if hasattr(usage, "input_tokens")
                else usage.get("input_tokens", 0)
            )
            completion = (
                usage.output_tokens
                if hasattr(usage, "output_tokens")
                else usage.get("output_tokens", 0)
            )
            if prompt or completion:
                record_tokens(model_name, prompt, completion)
            return

        # Older LangChain / Groq: response.response_metadata → token_usage
        meta = getattr(response, "response_metadata", None) or {}
        token_usage = meta.get("token_usage", {}) or {}
        prompt = token_usage.get("prompt_tokens", 0)
        completion = token_usage.get("completion_tokens", 0)
        if prompt or completion:
            record_tokens(model_name, prompt, completion)
    except Exception:
        pass  # Never let tracking failures propagate


# ── Callback handler for chain invocations ─────────────────────────────────

try:
    from langchain_core.callbacks import BaseCallbackHandler as _BaseHandler
except ImportError:
    # Fallback: define a minimal stub so the module works without langchain
    class _BaseHandler:
        pass


class TokenUsageHandler(_BaseHandler):
    """LangChain callback handler that records token usage from chain calls.

    Pass via ``config={"callbacks": [handler]}`` to any ``.invoke()`` call.
    LangChain propagates callbacks through ``RunnableSequence``, so an LLM
    call nested inside a ``prompt | llm | StrOutputParser()`` chain will
    fire ``on_llm_end`` with the real token counts.
    """

    def __init__(self, model_name: str):
        super().__init__()
        self.model_name = model_name

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        """LangChain callback — called when an LLM finishes generating."""
        try:
            llm_output = getattr(response, "llm_output", None) or {}
            token_usage = llm_output.get("token_usage", {})
            prompt = token_usage.get("prompt_tokens", 0)
            completion = token_usage.get("completion_tokens", 0)
            if prompt or completion:
                record_tokens(self.model_name, prompt, completion)
        except Exception:
            pass


def track_chain_invoke(model_name: str, chain: Any, inputs: dict) -> Any:
    """Invoke a LangChain expression chain with automatic token tracking.

    Usage:
        result = track_chain_invoke("llama-3.3-70b-versatile", my_chain, {"key": val})
    """
    handler = TokenUsageHandler(model_name)
    return chain.invoke(inputs, config={"callbacks": [handler]})
