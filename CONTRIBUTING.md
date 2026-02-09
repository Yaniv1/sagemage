# Contributing to sagemage

Thank you for your interest in contributing to sagemage! This document provides guidelines and instructions for contributing.

## Getting Started

1. **Fork the repository** on GitHub
2. **Clone your fork** locally:
   ```bash
   git clone https://github.com/yourusername/sagemage.git
   cd sagemage
   ```
3. **Add upstream remote**:
   ```bash
   git remote add upstream https://github.com/original-owner/sagemage.git
   ```

## Setting Up Development Environment

```bash
# Create and activate virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install package and development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

## Code Style

We follow PEP 8 and use the following tools for code quality:

- **black**: Code formatting (100 character line length)
- **isort**: Import sorting
- **flake8**: Linting
- **mypy**: Type checking (optional annotations)

### Format your code:

```bash
black sagemage tests
isort sagemage tests
```

### Check code quality:

```bash
flake8 sagemage tests
mypy sagemage
```

## Testing

### Run tests:

```bash
pytest
```

### Run with coverage:

```bash
pytest --cov=sagemage --cov-report=html
```

### Run specific test:

```bash
pytest tests/test_sagemage.py::test_function_name
```

All new features should include tests. Aim for >80% code coverage.

## Commit Messages

Please write clear and descriptive commit messages:

- Use imperative mood ("Add feature" not "Added feature")
- Start with a capital letter
- Use concise, descriptive headlines
- Reference issues when relevant: "Fixes #123"

Example:
```
Add CONTRIBUTING.md and improve documentation

- Add comprehensive contributing guidelines
- Update README with development instructions
- Fixes #42
```

## Pull Requests

1. **Update your branch**:
   ```bash
   git fetch upstream
   git rebase upstream/main
   ```

2. **Push to your fork**:
   ```bash
   git push origin your-feature-branch
   ```

3. **Create Pull Request** on GitHub with:
   - Clear title and description
   - Reference to related issues
   - Changes made and motivation
   - Testing instructions if applicable

4. **Ensure all checks pass**:
   - Tests pass
   - Code style is correct
   - Coverage is maintained/improved

## Documentation

- Update docstrings for new functions/classes
- Follow Google-style docstrings
- Update relevant README sections
- Update CHANGELOG.md with your changes

Example docstring:
```python
def example_function(param1: str, param2: int) -> bool:
    """Brief description of function.
    
    Longer description explaining what the function does,
    parameter details, and return value.
    
    Args:
        param1: Description of param1
        param2: Description of param2
    
    Returns:
        Description of return value
    
    Raises:
        ValueError: When param1 is empty
    """
```

## Types of Contributions

### Bug Reports
- Use GitHub Issues with the "bug" label
- Include reproducible example
- Describe expected vs actual behavior
- Include Python version and OS

### Feature Requests
- Use GitHub Issues with the "enhancement" label
- Describe the feature and use case
- Explain motivation and potential implementation

### Documentation
- Improve README, docstrings, or examples
- Fix typos and clarify confusing sections
- Add usage examples

### Code
- Follow guidelines above
- Include tests and documentation
- Keep PRs focused on single feature/fix

## Reporting Issues

When reporting bugs, please include:
- Python version: `python --version`
- Package version: `pip show sagemage`
- Minimal reproducible example
- Expected and actual behavior
- Relevant error messages or tracebacks

## License

By contributing, you agree that your contributions will be licensed under the MIT License.

## Questions?

- Open an issue for discussion
- Check existing issues and discussions
- Review documentation and examples

Thank you for contributing to sagemage!
