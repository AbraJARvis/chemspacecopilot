# Unit Tests

## Quick Start

```bash
# Run all unit tests
uv run pytest tests/unit/ -v

# Run specific test file
uv run pytest tests/unit/test_gtm_sampling.py -v

# Run with coverage
uv run pytest tests/unit/ --cov=src/cs_copilot --cov-report=html
```

## Quick Validation

```bash
# Test basic infrastructure (3 seconds)
uv run python test_simple.py

# Test single prompt robustness (15 seconds)
uv run python test_robustness_minimal.py
```

## Pre-commit Hooks

```bash
# Install hooks
pre-commit install

# Run manually
pre-commit run --all-files
```

Pre-commit runs:

- Black (code formatting on `src/` and `tests/` only)
- File checks (trailing whitespace, large files, merge conflicts)
- Unit tests (`tests/unit/`)

## Linting and Formatting

```bash
# Format code with Black
uv run black src/ tests/

# Run Ruff linter
uv run ruff check src/ tests/ --fix

# Sort imports
uv run isort src/ tests/
```

## CI/CD Integration

```yaml
- name: Run Tests
  run: |
    export USE_S3=false
    uv run pytest tests/unit/ -v --tb=short
  env:
    DEEPSEEK_API_KEY: ${{ secrets.DEEPSEEK_API_KEY }}
```
