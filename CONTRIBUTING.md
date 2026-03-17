# Contributing to ResearchClaw

Thank you for your interest in contributing to ResearchClaw! This guide will help you get started.

## Development Setup

### 1. Fork and Clone

```bash
# Fork via GitHub UI, then clone your fork
git clone https://github.com/YOUR_USERNAME/AutoResearchClaw.git
cd AutoResearchClaw
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies

```bash
# Install in editable mode with dev dependencies
pip install -e ".[dev]"

# Or install only runtime dependencies
pip install -e .
```

## Configuration

ResearchClaw supports multiple configuration file names:

| File | Purpose |
|------|---------|
| `config.researchclaw.yaml` | Primary config (recommended) |
| `config.arc.yaml` | ARC-specific settings |
| `config.yaml` | Legacy fallback |
| `config.researchclaw.example.yaml` | Template for new setups |

Copy the example and customize:

```bash
cp config.researchclaw.example.yaml config.researchclaw.yaml
```

## Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_your_module.py

# Run with verbose output
pytest -v
```

## Health Check

Verify your setup is correct:

```bash
researchclaw doctor
```

## Pull Request Guidelines

1. **Branch naming**: Use descriptive names like `fix/issue-number-description` or `feat/feature-name`
2. **Commit messages**: Use clear, descriptive commit messages
3. **PR description**: Include:
   - What the change does
   - Why it's needed
   - How to test it
4. **Tests**: Add tests for new features; ensure existing tests pass

## Code Style

- Follow PEP 8 for Python code
- Use type hints where appropriate
- Keep functions focused and small

## Getting Help

- Check [Issues](https://github.com/aiming-lab/AutoResearchClaw/issues) for existing discussions
- Open a new issue for bugs or feature requests
- Join the community in discussions

---

Happy researching! 🔬
