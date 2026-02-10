# SAGEMAGE

**Systematic Agentic Modular Analytic Generative Engineering Framework**

A Python package for building systematic, data-driven solutions with support for parametric configuration, dataset management, and LLM-powered analysis.

## Overview

SAGEMAGE provides a modular framework that abstracts common engineering patterns into reusable components:

- **Modular** - Loosely coupled components for maximum flexibility
- **Systematic** - Clear patterns and conventions for consistency
- **Extensible** - Easy to customize and build upon
- **Type-Safe** - Full type hints for IDE support
- **Documented** - Comprehensive documentation and examples

## Quick Start

```python
from sagemage import Agent
import json

# Load configuration
with open("config.json") as f:
    config = json.load(f)

# Create and run agent
agent = Agent(config)
results = agent.run(project_dir="/path/to/project")

# Results as pandas DataFrame
print(results)
```

## Features

### Core Components

- **ParamSet** - Configuration management with inheritance and JSON support
- **Agent** - Orchestrates data processing and LLM interactions
- **Dataset** - Handles data loading, chunking, and processing
- **ApiClient** - Manages LLM prompts and responses

### Built-In Examples

- **FooFoo** - Complete demonstrated workflow with sample data and instructions

## Installation

```bash
# From PyPI
pip install sagemage

# From GitHub
pip install git+https://github.com/Yaniv1/sagemage.git

# Development
git clone https://github.com/Yaniv1/sagemage.git
cd sagemage
pip install -e .
```

## Documentation

- [User Manual](./manual.md) - Complete guide with examples
- [Architecture](./docs/architecture.md) - Technical design and structure
- [Contributing](./CONTRIBUTING.md) - How to contribute
- [Changes](./changes.md) - Version history

## Example: FooFoo Classification

Run the built-in example:

```python
from sagemage.examples.foofoo.py.foofoo import foofoo

# Create workspace and run agent
results = foofoo(
    base_dir="/my/workspace",
    api_key_value="sk-your-api-key"
)

# Reset and try again
results = foofoo(base_dir="/my/workspace", reset=True)
```

## License

MIT License - See LICENSE file for details

## Support

- GitHub: https://github.com/Yaniv1/sagemage
- Issues: https://github.com/Yaniv1/sagemage/issues
