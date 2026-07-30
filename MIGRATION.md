# Migration Note — Repo Flattening

**Commit `b8a5efc`** (Jul 30, 2026) removed nested `research-agent/` duplicate directories and flattened the repository to a single root.

## What changed

Previously, the repo had three nested near-duplicate copies of the project:

```
repo-root/
├── (authoritative files — .git, .github/, server.py, etc.)
├── research-agent/        <-- stale copy with its own .git
│   ├── research-agent/    <-- even older snapshot
│   └── ...
```

After `git rm -r research-agent/` at commit `b8a5efc`, all files that were under `research-agent/<path>` now live directly at `<path>` in the repo root.

## Effect on `git log --follow`

Because `git rm` was used (not `git mv`), `git log --follow <path>` will NOT trace history through the `research-agent/` prefix. For example:

```bash
# This will show history only back to b8a5efc:
git log --follow server.py

# The pre-flattening history still exists in the repo but without --follow
# working across the directory removal. To see the full history:
git log --all --full-history -- server.py
```

## File map

| Old path | New path |
|----------|----------|
| `research-agent/server.py` | `server.py` |
| `research-agent/chain/chain.py` | `chain/chain.py` |
| `research-agent/static/index.html` | `static/index.html` |
| ... (all other files follow the same pattern) | |

The authoritative copy was determined by checking which tree had `.git/`, `.github/workflows/ci.yml`, `chat/`, `ingestion/`, and `search_provider.py` — these only existed at the repo root, confirming it was the canonical source.
