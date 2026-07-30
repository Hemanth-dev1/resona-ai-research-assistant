"""Retry utilities using tenacity for production resilience.

Wraps every LLM call with exponential backoff retry logic.
Provides decorators for crew.kickoff(), chain.invoke(), and other
fallible operations so failures return graceful error responses
instead of crashing with a 500.
"""

import functools
import os
import sys
import time
from typing import Any, Callable, Optional, TypeVar

from tenacity import (
    before_sleep_log,
    retry,
    stop_after_attempt,
    wait_exponential,
)

F = TypeVar("F", bound=Callable[..., Any])

# Retry configuration from environment (with sensible defaults)
MAX_RETRIES = int(os.getenv("RESONA_MAX_RETRIES", "3"))
MIN_WAIT_SECONDS = float(os.getenv("RESONA_RETRY_MIN_WAIT", "1.0"))
MAX_WAIT_SECONDS = float(os.getenv("RESONA_RETRY_MAX_WAIT", "10.0"))


def default_retry_decorator(
    stop_max_attempt_number: int = MAX_RETRIES,
    min_wait: float = MIN_WAIT_SECONDS,
    max_wait: float = MAX_WAIT_SECONDS,
) -> Callable[[F], F]:
    """Create a tenacity retry decorator with exponential backoff.

    Args:
        stop_max_attempt_number: Maximum number of retry attempts (default: 3).
        min_wait: Minimum wait between retries in seconds (default: 1.0).
        max_wait: Maximum wait between retries in seconds (default: 10.0).

    Returns:
        A tenacity retry decorator configured with exponential backoff.
    """
    return retry(
        stop=stop_after_attempt(stop_max_attempt_number),
        wait=wait_exponential(multiplier=1.0, min=min_wait, max=max_wait),
        reraise=True,
    )


# Default retry decorator instance
default_retry = default_retry_decorator()


def retry_call(
    func: Callable[..., Any],
    *args: Any,
    max_retries: int = MAX_RETRIES,
    **kwargs: Any,
) -> Any:
    """Execute a callable with retry logic.

    Wraps the function call in tenacity retry with exponential backoff.
    Returns a graceful error dict if all retries are exhausted.

    Args:
        func: The callable to execute.
        *args: Positional arguments for the callable.
        max_retries: Maximum number of attempts (default: 3).
        **kwargs: Keyword arguments for the callable.

    Returns:
        The return value of the callable on success, or a dict with
        {"error": message, "success": False} if all retries fail.
    """
    last_exception = None

    for attempt in range(1, max_retries + 1):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            last_exception = e
            if attempt < max_retries:
                wait_time = min(1.0 * (2 ** (attempt - 1)), 10.0)
                print(f"  ⚠️  Retry {attempt}/{max_retries} after {wait_time:.1f}s: {e}")
                time.sleep(wait_time)
            else:
                print(f"  ❌ All {max_retries} retries exhausted: {e}")

    return {
        "error": str(last_exception),
        "success": False,
        "message": f"Operation failed after {max_retries} attempts.",
    }


def safe_invoke(
    func: Callable[..., Any],
    *args: Any,
    error_message: str = "Operation failed",
    **kwargs: Any,
) -> Any:
    """Safely invoke a callable with retry and graceful error handling.

    Args:
        func: The callable to invoke (e.g., crew.kickoff, chain.invoke).
        *args: Positional arguments for the callable.
        error_message: Prefix for error messages.
        **kwargs: Keyword arguments for the callable.

    Returns:
        The result on success, or a dict with error context on failure.
    """
    result = retry_call(func, *args, **kwargs)

    if isinstance(result, dict) and "error" in result and not result.get("success", True):
        return {
            "error": f"{error_message}: {result['error']}",
            "success": False,
        }

    return result
