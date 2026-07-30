"""Simple in-memory token bucket rate limiter.

No external dependencies — uses a dict-based sliding window.
Configurable via RESONA_RATE_LIMIT_PER_MINUTE env var (default: 10).
"""

import os
import time
from collections import defaultdict

_RATE_LIMIT = int(os.getenv("RESONA_RATE_LIMIT_PER_MINUTE", "10"))
_WINDOW = 60  # seconds

# ip -> list of timestamps
_hits: dict[str, list[float]] = defaultdict(list)


def check_rate_limit(client_ip: str) -> bool:
    """Check if the client has exceeded the rate limit.

    Returns True if the request is allowed, False if rate-limited.
    """
    now = time.time()
    cutoff = now - _WINDOW

    # Prune expired entries
    hits = [t for t in _hits[client_ip] if t > cutoff]
    _hits[client_ip] = hits

    if len(hits) >= _RATE_LIMIT:
        return False

    _hits[client_ip].append(now)
    return True


def get_rate_limit_remaining(client_ip: str) -> int:
    """Get how many requests the client can still make in the current window."""
    now = time.time()
    cutoff = now - _WINDOW
    hits = [t for t in _hits[client_ip] if t > cutoff]
    return max(0, _RATE_LIMIT - len(hits))
