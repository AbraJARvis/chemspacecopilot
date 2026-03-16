# Contributing

Contributions are welcome! Please follow the guidelines below.

## Workflow

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-feature`)
3. Make your changes
4. Ensure code passes `pre-commit run --all-files`
5. Run unit tests: `uv run pytest tests/unit/ -v`
6. Submit a pull request

## Code Style

- **Formatting**: Black with 100-character line length (auto-applied by pre-commit)
- **Imports**: isort with Black profile
- **Linting**: Ruff (E, W, F, I, B, C4 rules)
- **Python Version**: 3.11 only
- **Type Hints**: Encouraged but not strictly enforced
- **Docstrings**: Required for toolkit methods (visible to the LLM as tool descriptions)

```bash
# Format code
uv run black src/ tests/

# Lint
uv run ruff check src/ tests/ --fix

# Sort imports
uv run isort src/ tests/
```

## Agent Instructions

Agent instructions written in `prompts.py` should be:

- Clear and specific (LLMs follow them literally)
- Include examples where helpful
- Specify output format expectations
- Reference available tools explicitly

## Adding Agents

See the [Agents architecture page](architecture/agents.md#adding-a-new-agent).

## Adding Tools

See the [Tools architecture page](architecture/tools.md#adding-a-new-tool).
