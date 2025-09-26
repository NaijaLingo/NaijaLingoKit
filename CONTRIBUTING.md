## Development and Publishing

### Build and publish with uv

```bash
uv pip install --upgrade pip setuptools wheel
uv pip install build twine

# bump version in pyproject.toml before publishing
uv run python -m build

# TestPyPI first
uv run twine upload --repository testpypi dist/*

# Install from TestPyPI to verify
uv pip install -i https://test.pypi.org/simple naijaligo-asr

# Publish to PyPI
uv run twine upload dist/*
```

This project uses PEP 621 (`pyproject.toml`) with setuptools. No `setup.py` is required.

