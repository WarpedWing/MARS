# MARS Tests

Automated tests for MARS using pytest.

## Running Tests

> **Note**: Pytest is configured in `pyproject.toml` to automatically set the Python path to `src/`,
so you don't need to set `PYTHONPATH` manually.

### Run all tests

```bash
uv run pytest
```

### Run with verbose output

```bash
uv run pytest -v
```

### Run specific test file

```bash
uv run pytest tests/test_skip_list.py -v
```

### Run specific test

```bash
uv run pytest tests/test_skip_list.py::TestSkipDetection::test_geoservices_detected_correctly -v
```

### Run with coverage

```bash
uv run pytest tests/ --cov=src/mars --cov-report=html
```

## Test Organization

- `test_skip_list.py` - Tests for database skip list functionality
  - Catalog loading
  - Skip database configuration
  - Skip detection logic
  - Wildcard pattern matching
  - Edge cases

## Adding New Tests

1. Create a new test file: `tests/test_<feature>.py`
2. Import necessary modules
3. Use pytest fixtures from `conftest.py`
4. Follow the naming convention: `test_<what_it_tests>()`

Example:

```python
def test_my_feature(processor):
    """Test that my feature works correctly."""
    result = processor.my_feature()
    assert result is True
```

## Dependencies

Tests require:

- pytest
- pytest-cov (for coverage reports)

These are included in the project's development dependencies.
