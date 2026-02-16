# SAGEMAGE Architecture

**Sagemage** - Systematic Agentic Modular Analytic Generative Engineering Framework

## Overview

SAGEMAGE is a Python package that provides a modular framework for building systematic, data-driven solutions with support for parametric configuration, dataset management, and LLM-powered analysis. It abstracts common engineering patterns into reusable components.

## Design Philosophy

1. **Modular** - Loosely coupled components that can be used independently
2. **Systematic** - Clear patterns and conventions for configuration and data handling
3. **Type-Safe** - Comprehensive type hints for IDE support and runtime safety
4. **Extensible** - Easy to extend with custom agents, handlers, and processors
5. **Documented** - Full api documentation with clear examples

## Package Structure

```
sagemage/
├── __init__.py              # Public API exports
├── py.typed                 # PEP 561 type hints marker
├── core.py                  # ParamSet, Agent classes
├── utils.py                 # Utility functions
├── dataset.py               # Dataset class
└── api_client.py            # ApiClient class
```

## Core Modules

### 1. Core Module (`core.py`)

**Responsibility:** Parameter management and agent configuration

#### Classes

##### `ParamSet`
Base class for managing configuration parameters with support for:
- Dictionary initialization
- JSON file loading
- Nested parameter parsing
- Parameter inheritance from baseline

**Key Methods:**
- `__init__(params_dict, baseline)` - Initialize with dict or JSON
- `get(keys=[])` - Retrieve parameters (all or specific keys)
- `set(nest=True, **kvpairs)` - Set parameters with optional JSON parsing

**Example:**
```python
from sagemage import ParamSet

# Load from file
params = ParamSet("config.json")

# Load from dict
params = ParamSet({"temperature": 0.7, "max_tokens": 1000})

# With baseline
defaults = ParamSet({"temperature": 1.0})
custom = ParamSet({"max_tokens": 500}, baseline=defaults)

# Get parameters
all_params = params.get()
temp = params.get(["temperature"])
```

##### `Agent`
Represents an agent node with configuration parameters and orchestrates LLM-powered data processing.

**Extends:** `ParamSet`

**Purpose:** 
- Inherit all `ParamSet` functionality
- Manage multiple input datasets (inlets) with independent chunking
- Process data through LLM APIs
- Route results to multiple outputs (outlets)
- Support agent chaining via agent memory (AM)

**Key Features:**

1. **Multi-Inlet Support**
   - Accept multiple input datasets with independent chunk sizes
   - Each inlet specifies its own dataset configuration (columns, ID, name)
   - Generate cartesian product combinations of chunks

2. **Cartesian Chunk Combinations**
   - If inlet1 has 3 chunks and inlet2 has 1 chunk → 3 combinations
   - Each combination includes data from all inlets in a single prompt
   - Enables cross-dataset analysis without explicit joins

3. **Outlet Routing**
   - Route single results set to multiple outlets
   - Each outlet can specify different result columns
   - Support for file system (FS), agent memory (AM), S3, and KV storage

4. **Agent Memory (AM)**
   - AM outlets store results for consumption by downstream agents
   - AM inlets retrieve results from upstream agents
   - Enables agent chaining and orchestrated workflows

**Configuration:**

```json
{
  "my_agent": {
    "inlets": [
      {
        "uri": "data/input.csv",
        "type": "FS",
        "dataset_name": "items",
        "dataset_id_column": "id",
        "dataset_columns": ["name", "description"],
        "chunk_size": 2
      }
    ],
    "outlets": [
      {
        "uri": "results/output.csv",
        "type": "FS",
        "dataset_name": "results",
        "result_columns": ["id", "name", "classification"]
      },
      {
        "uri": "my_agent",
        "type": "AM",
        "result_columns": ["id", "name", "classification"]
      }
    ],
    "instructions": "instructions/prompt.txt",
    "model": {
      "client": "openai.OpenAI",
      "version": "gpt-5.1",
      "temperature": 0.7
    },
    "api_client": {
      "max_tokens": 4096
    },
    "chunk_size": 10
  }
}
```

**Agent Lifecycle:**

1. **Load Inlets** - Load data from all inlet sources (FS, AM)
2. **Process Datasets** - Create textual representations and group data
3. **Chunk Each Inlet** - Apply inlet-specific chunk sizes
4. **Generate Combinations** - Create cartesian product of all chunks
5. **Process Each Combination**:
   - Build prompt with data from all inlets
   - Call LLM API
   - Parse response into structured results
6. **Route Results** - Save combined results to all outlets
7. **Agent Memory** - Store in agent memory for downstream consumption

**Attributes:**

- `agent_id` - Unique identifier for this agent
- `results` - DataFrame containing processed results (set after run())

**Methods:**

- `run(project_dir, input_df, agent_outputs, agents_dict)` - Execute agent and return results

**Example:**

```python
from sagemage import Agent, AgentSet
import json

# Single agent
config = {
  "name": "classifier",
  "inlets": [{"uri": "data/input.csv", "type": "FS"}],
  "outlets": [{"uri": "results.csv", "type": "FS"}]
}
agent = Agent(config)
results = agent.run(project_dir="/my/work")

# Agent set with chaining
agent_set = AgentSet("config.json")
results = agent_set.run_all(project_dir="/my/work")
```

---

### 2. Utils Module (`utils.py`)

**Responsibility:** General-purpose utility functions

#### Functions

##### `print_dict(my_dict: Dict[str, Any]) -> None`
Pretty-prints a dictionary with smart formatting

- Single values and short lists on one line
- Long collections on separate lines
- Graceful error handling

##### `save_to_path(obj: Any, path: str, append: bool = False) -> bool`
Saves objects to files with format auto-detection

**Supported Formats:**
- `.json` - JSON with pretty printing
- `.csv` - DataFrame to CSV
- `.xlsx`, `.xls` - DataFrame to Excel
- `.txt` - String to text file

**Returns:** `True` on success, exception object on failure

**Example:**
```python
from sagemage import save_to_path
import pandas as pd

df = pd.read_csv("data.csv")
save_to_path(df, "output/data.json")  # Auto-detect format from extension
save_to_path({"key": "value"}, "config.json")
save_to_path("text content", "output.txt", append=True)
```

##### `setattrs(r: Optional[Dict] = None, **kwargs) -> Dict[str, Any]`
Updates dictionary with keyword arguments

Convenience function for dictionary construction

**Example:**
```python
from sagemage import setattrs

config = setattrs(
    temperature=0.7,
    max_tokens=1000,
    model="gpt-4"
)
# Returns: {"temperature": 0.7, "max_tokens": 1000, "model": "gpt-4"}
```

##### `flatten_dataframe(df: DataFrame, combination_id="combination_id", cols_to_flatten=None) -> DataFrame`
Flattens DataFrames by expanding list columns into multiple rows

**Use Case:** Convert structured data with list values into flat format

**Example:**
```python
from sagemage import flatten_dataframe
import pandas as pd

# Input df:
# id  | items
# 1   | [A, B]
# 2   | [C]

# After flatten:
# id  | combination_id | items
# 1   | 0              | A
# 1   | 1              | B
# 2   | 0              | C

flat_df = flatten_dataframe(df, combination_id="combo_id")
```

## Packaging and Runtime Behavior

### Versioning (single source of truth)

SAGEMAGE stores the package version in `sagemage/version.txt`. Build-time metadata (setuptools/pyproject) and runtime `__version__` both read from that file so there is a single source of truth for version numbers. The `pyproject.toml` uses `dynamic = ["version"]` and `tool.setuptools.dynamic.version = {file = ["sagemage/version.txt"]}` so packaging reads the same value that runtime exposes.

### Lazy imports to avoid build-time dependency issues

Top-level module import avoids importing heavy runtime dependencies (like `pandas`) during package metadata extraction. The package `__init__` reads `version.txt` and exposes a lazy attribute import via `__getattr__`, which imports submodules on first access. This prevents setuptools/pip metadata preparation from failing when runtime dependencies are not yet installed in the isolated build environment.

### Examples distribution and demo dispatcher

Examples are included under `sagemage/examples` and are packaged with the distribution. Each example provides a canonical entrypoint `run(base_dir, api_key_value=None, reset=False)`.

To simplify usage, SAGEMAGE exposes a dispatcher:

```python
from sagemage.examples import demo, list_examples

print(list_examples())
demo("foofoo")  # copies packaged assets into ./foofoo then runs the example
```

`demo()` copies packaged example assets into a local workspace (default `./<example_name>`) and invokes the example's `run()` function. The optional `reset=True` argument forces rebuilding the workspace from the packaged assets.

### Agent runtime behavior

`Agent.run()` orchestrates the full execution lifecycle:

- Loads dataset using `Dataset` from the configured `input_file` and `dataset_columns`.
- Chunking based on `chunk_size` and `max_items`.
- Reads instructions and optional resources.
- Resolves LLM client params from agent config (`model` + `api_client` + top-level fallbacks).
- Constructs prompts via `ApiClient.construct_prompt`, performs LLM requests via `ApiClient.get_response`, and parses structured outputs via `ApiClient.parse_response`.
- Aggregates parsed results into a `pandas.DataFrame` and (optionally) saves per-chunk outputs under `output_path` when configured.

`ApiClient.get_response()` applies model-specific request argument mapping for OpenAI clients. For GPT-5-family model names (`gpt-5*`), SAGEMAGE sends `max_completion_tokens` while allowing agent configs to keep using `max_tokens`.

`model`/`api_client` resolution order inside `Agent.run()`:
1. `model` (string or object) initializes API client config.
2. `api_client` object overlays and overrides `model`.
3. `version` is accepted as alias for `model`.
4. Top-level sampling params fill only missing keys (`temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`).
5. `api_key` may come from top-level agent config or fallback inside `model`/`api_client`.

---

### 3. Dataset Module (`dataset.py`)

**Responsibility:** Load, process, organize, and save datasets

#### Class: `Dataset`

Manages complete dataset lifecycle with support for:
- Multi-format loading (CSV, XLSX, JSON)
- Automatic grouping and chunking
- Format conversion and flattening
- Organized output directory structure

**Key Attributes:**
- `df` - Main DataFrame
- `items` - Textual representations of rows
- `groups` - Grouped DataFrames with output paths
- `chunks` - Chunked DataFrames with output paths

**Initialization Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `input_path` | str | "" | Path to input file |
| `output_path` | str | "" | Base output directory |
| `id_column` | str | "id" | Name of ID column |
| `columns` | List[str] | [] | Specific columns to use |
| `capitalize` | bool | True | Capitalize column names |
| `sheet_name` | int | 0 | Excel sheet to load |
| `root` | str | "data" | JSON root node |
| `textual` | str | "ITEM" | Column for text repr |
| `output_format` | str | "json" | Output format |
| `group_columns` | List[str] | [] | Columns to group by |
| `chunk_size` | int | 1 | Rows per chunk |

**Key Methods:**

- `load(**kwargs)` - Load data from input file
- `group()` - Group data by specified columns
- `chunk()` - Split groups into chunks
- `save()` - Save groups and chunks to files
- `flatten(combination_id, cols_to_flatten)` - Flatten list columns

**Example:**
```python
from sagemage import Dataset

# Load and process
ds = Dataset(
    input_path="data.csv",
    output_path="output/",
    id_column="id",
    group_columns=["category"],
    chunk_size=10
)

# Data is automatically:
# 1. Loaded from input_path
# 2. Grouped by category
# 3. Split into chunks of 10 rows
# 4. Saved to output directory structure:
#    output/data_csv/category=A/...
#    output/data_csv/category=B/...
```

---

### 4. API Client Module (`api_client.py`)

**Responsibility:** Interact with LLM APIs (OpenAI compatible)

#### Class: `ApiClient`

Manages LLM interactions with:
- Prompt construction with data injection
- API calls with error handling
- Response parsing into structured data (JSON/CSV/TXT)
- Optional merging with input data

**Initialization Parameters:**

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `client` | class | None | API client class (e.g., openai.OpenAI) |
| `model` | str or dict | "gpt-4" | Model name or model config (supports `version` alias) |
| `api_key` | str | "" | API authentication key |
| `temperature` | float | 1 | Sampling temperature |
| `max_tokens` | int | 4096 | Max response tokens |
| `top_p` | float | 1 | Nucleus sampling |
| `frequency_penalty` | float | 0 | Repetition penalty |
| `presence_penalty` | float | 0 | Diversity penalty |

When `client` is a string, `ApiClient` resolves dotted import paths such as `openai.OpenAI` at runtime.
For GPT-5-family models, `max_tokens` is mapped to `max_completion_tokens` in outbound OpenAI requests.

**Key Methods:**

##### `construct_prompt(items, data, resources, text, mapping) -> str`
Builds prompts with placeholder substitution

**Parameters:**
- `items` - List of prompt sections (may include placeholders)
- `data` - Data to inject
- `resources` - Resource references
- `text` - Additional text
- `mapping` - Placeholder-to-variable mapping

**Example:**
```python
prompt = client.construct_prompt(
    items=[
        "Analyze the following data:\n",
        "{$DATA$}",
        "\nProvide JSON results."
    ],
    data=["row1: value1", "row2: value2"]
)
```

##### `get_response(prompt, **overrides) -> Dict`
Calls LLM API with error handling

**Returns Dictionary:**
```python
{
    "timestamp": datetime,
    "params": {model, temperature, max_tokens, ...},
    "response": response_object
}
```

##### `parse_response(response, format, json_node, id_column, ...) -> DataFrame`
Parses LLM response into structured DataFrame

**Supported Formats:**
- `"json"` - Extract JSON with optional merge to input
- `"csv"` - Parse CSV-formatted response
- `"txt"` - Keep raw model text in a `response` column

**Parameters:**
- `response` - Response dict from `get_response()`
- `format` - Response format ("json" or "csv")
- `json_node` - JSON key containing results
- `id_column` - ID column for merging
- `flatten` - Whether to flatten list columns
- `input_path` - Input file for data merging
- `output_path` - Save results to file

**Example:**
```python
client = ApiClient(
    client=openai.OpenAI,
    model="gpt-4",
    api_key="sk-..."
)

# Get response
prompt = "Classify these items: [item1, item2, ...]"
response = client.get_response(prompt)

# Parse response
results_df = client.parse_response(
    response,
    format="json",
    json_node="classifications",
    output_path="results.csv"
)
```

---

## Data Flow Architecture

### Typical Workflow

```
Input Data
    ↓
Dataset.load()          
    ↓ (parse format, create textual items)
DataFrame + Items
    ↓
Dataset.group()         
    ↓ (organize by group_columns)
Grouped DataFrames
    ↓
Dataset.chunk()         
    ↓ (split into chunks)
Chunked DataFrames
    ↓
Dataset.save()          
    ↓ (to output directory)
Organized Files
    ↓
    ├─── ApiClient.construct_prompt()
    │        ↓
    │    Prompt with Data Injected
    │        ↓
    │    ApiClient.get_response()
    │        ↓
    │    LLM Response
    │        ↓
    └─── ApiClient.parse_response()
             ↓
         Results DataFrame
             ↓
         outlet persistence
             ↓
         CSV + optional JSON/TXT files
```

## Public API

### Imports

All public classes and utilities are exported from the package root:

```python
from sagemage import (
    # Core
    ParamSet,
    Agent,
    
    # Data Management
    Dataset,
    
    # API Interactions
    ApiClient,
    
    # Utilities
    print_dict,
    save_to_path,
    setattrs,
    flatten_dataframe,
)
```

### Key Properties

- `__version__` - Package version
- `__author__` - Package author
- `__email__` - Author email
- `__license__` - License type (MIT)

## Dependencies

### Core Dependencies
- `pandas>=1.3.0` - Data manipulation
- `numpy>=1.20.0` - Numerical computing
- `openai>=1.0.0` - OpenAI API client

### Optional Dependencies
- `dev` - Development tools (pytest, black, flake8, mypy)
- `docs` - Documentation tools (sphinx, sphinx-rtd-theme)
- `api` - API support (openai)

## Design Patterns

### 1. Configuration Management (ParamSet)
- Parameter inheritance through baseline
- Automatic JSON parsing
- Quick access to configuration

### 2. Data Pipeline (Dataset)
- Load → Group → Chunk → Save
- Each step independent but builds on previous
- Extensible output formats

### 3. LLM Integration (ApiClient)
- Structured prompting with data injection
- Flexible response parsing
- Seamless data merging

### 4. Utility Functions
- Format-agnostic file handling
- DataFrame operations
- Dictionary utilities

## Extension Points

### Custom Agent Types
```python
from sagemage import Agent

class CustomAgent(Agent):
    def analyze(self, data):
        # Custom analysis logic
        pass
```

### Custom DataSet Processors
```python
from sagemage import Dataset

class CustomDataset(Dataset):
    def process(self):
        # Custom processing
        self.load()
        # ... custom logic
        self.save()
```

### Custom ApiClient Subclasses
```python
from sagemage import ApiClient

class CustomApiClient(ApiClient):
    def get_response(self, prompt, **overrides):
        # Custom API provider logic
        pass
```

## Error Handling

All modules include try-catch blocks for:
- File I/O operations
- JSON parsing
- API calls
- Data transformations

Errors are logged to stdout with context information.

## Type Safety

Full type hints throughout:
- Function parameters and return types
- Optional parameters with defaults
- Union types for flexible inputs
- Generic types for collections

## Performance Considerations

- **Chunking** - Large datasets split into manageable pieces
- **Grouping** - Organize data before processing
- **Lazy Loading** - Files only loaded on access
- **Memory Efficient** - Works with DataFrames (column-oriented)

## Best Practices

1. **Use ParamSet for Configuration** - Centralize settings
2. **Leverage Dataset for Batch Processing** - Handle grouping/chunking automatically
3. **Cache API Responses** - LLM calls are expensive
4. **Type Hints Everything** - Help your IDE and future developers
5. **Use Output Paths** - Organize results systematically

## Example: Complete Workflow

```python
from sagemage import ParamSet, Dataset, ApiClient, save_to_path
import openai

# 1. Load configuration
config = ParamSet("config.json")

# 2. Process dataset
dataset = Dataset(
    input_path="data.csv",
    output_path="output/",
    id_column="id",
    group_columns=["category"],
    chunk_size=10
)

# 3. Initialize API client
client = ApiClient(
    client=openai.OpenAI,
    model=config.model,
    api_key=config.api_key,
    temperature=config.get(["temperature"])["temperature"]
)

# 4. Process each group
for group_path, group_df in dataset.groups.items():
    # Build prompt
    items_text = "\n".join(dataset.items)
    prompt = client.construct_prompt(
        items=[f"Analyze these items:\n{items_text}"],
        data=[str(row) for _, row in group_df.iterrows()]
    )
    
    # Get response
    response = client.get_response(prompt)
    
    # Parse response
    results = client.parse_response(
        response,
        format="json",
        json_node="results",
        input_path="data.csv",
        output_path=group_path.replace(".json", "_results.json")
    )

print("✓ Processing complete!")
```

---

## Version History

### 0.1.0 (Current)
- Initial release
- Core ParamSet and Agent classes
- Full Dataset management
- ApiClient for LLM interactions
- Comprehensive utilities

## Future Enhancements

- [ ] Async API support
- [ ] Multi-provider LLM support (Anthropic, Google, etc.)
- [ ] Caching layer for API responses
- [ ] Workflow orchestration
- [ ] Advanced agent framework
- [ ] Real-time streaming responses
- [ ] Distributed dataset processing

---

## License

MIT License - See LICENSE file for details
