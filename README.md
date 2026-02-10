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
- **Agent** - Orchestrates data processing and LLM interactions with multi-inlet support
- **Dataset** - Handles data loading, chunking, and processing with per-inlet configuration
- **ApiClient** - Manages LLM prompts and responses
- **Agent Memory (AM)** - Enables chaining agents via inlet/outlet connections

### Multi-Inlet Support

Agents can consume multiple input datasets with independent chunk sizes:

```json
{
  "my_agent": {
    "inlets": [
      {
        "uri": "data/objects.csv",
        "type": "FS",
        "dataset_name": "objects",
        "dataset_id_column": "id",
        "dataset_columns": ["name"],
        "chunk_size": 2
      },
      {
        "uri": "data/categories.csv",
        "type": "FS",
        "dataset_name": "categories",
        "dataset_id_column": "cat_id",
        "dataset_columns": ["category"],
        "chunk_size": -1
      }
    ],
    "outlets": [
      {
        "uri": "results/output.csv",
        "type": "FS",
        "dataset_name": "objects",
        "result_columns": ["id", "name", "category"]
      },
      {
        "uri": "my_agent",
        "type": "AM",
        "result_columns": ["id", "name", "category"]
      }
    ]
  }
}
```

Features:
- **Cartesian combinations** - Multiple inlets create combinations (e.g., 2 chunks × 1 chunk = 2 combinations)
- **Per-inlet configuration** - Each inlet specifies dataset name, ID column, and columns to import
- **Independent chunking** - `chunk_size: -1` means no chunking (treat entire dataset as one chunk)
- **Agent chaining** - AM outlets enable downstream agents to consume upstream results
- **Result routing** - Single results set distributed to multiple outlets with different column configurations

### Examples & Demo

- Use the simple demo dispatcher to run packaged examples:

```python
from sagemage.examples import demo, list_examples

# show available examples
print(list_examples())

# run the default example ("foofoo")
demo()

# run a named example into a custom folder
demo("foofoo", base_dir="/tmp/myfoo", api_key_value="sk-...", reset=True)
```

Examples are included under `sagemage/examples` and are packaged with the distribution.

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
