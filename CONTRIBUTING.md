# Contributing to Resona

## Local Setup

```bash
cd research-agent
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
# Edit .env with your API keys
```

## Running Tests

```bash
cd research-agent
PYTHONPATH=. pytest tests/ -v
```

All tests should pass without live API keys (mocked tests don't require network access).

## Commit Conventions

This project uses conventional commit messages:

| Prefix | When to use |
|--------|-------------|
| `feat:` | New feature |
| `fix:` | Bug fix |
| `docs:` | Documentation changes |
| `chore:` | Maintenance, cleanup, deps |
| `test:` | Adding or updating tests |
| `perf:` | Performance improvement |
| `security:` | Security hardening |
| `refactor:` | Code restructuring |

## Code Style

- Python: follow PEP 8. Run `flake8 . --select=E9,F63,F7,F82` before committing.
- Tests: every new module should have a corresponding `tests/test_<module>.py` with mocked dependencies so tests run in CI without API keys.

## Pull Request Process

1. Ensure tests pass: `PYTHONPATH=. pytest tests/ -v`
2. Ensure Python files parse: `python -c "import ast,glob;[ast.parse(open(f).read()) for f in glob.glob('**/*.py') if 'venv' not in f]"`

