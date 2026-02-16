# SAGEMAGE User Manual

A comprehensive guide to using the SAGEMAGE framework for building systematic, data-driven agentic solutions.

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Core Concepts](#core-concepts)
4. [Working with ParamSet](#working-with-paramset)
5. [Creating and Running Agents](#creating-and-running-agents)
6. [Managing Datasets](#managing-datasets)
7. [API Integration](#api-integration)
8. [Built-In Examples](#built-in-examples)
9. [Advanced Usage](#advanced-usage)

The packaged examples demonstrate the complete workflow: configuration, input data, instructions, and agent execution. Use the unified `demo()` dispatcher to list and run examples.

### From PyPI

#### Listing examples

```python
from sagemage.examples import list_examples
print(list_examples())
```

#### Running an example (recommended)

```python
from sagemage.examples import demo

# Default (runs "foofoo" into ./foofoo)
demo()

# Run into a custom folder and provide an API key
demo("foofoo", base_dir="/my/work/foofoo_test", api_key_value="sk-your-key")

# Reset the example workspace before running
demo("foofoo", base_dir="/my/work/foofoo_test", reset=True)
```

The `demo()` helper will copy packaged example assets into the chosen `base_dir` (default: `./<example_name>`), then invoke the example's canonical `run()` entrypoint. This ensures examples are self-contained and repeatable.
```python
from sagemage import ParamSet

params = ParamSet({
    "temperature": 0.7,
    "max_tokens": 1000,
    "model": "gpt-4"
})
```

**From JSON File:**
```python
params = ParamSet("path/to/config.json")
```

**With Baseline/Defaults:**
```python
defaults = ParamSet({"temperature": 1.0, "max_tokens": 2000})
custom = ParamSet({"temperature": 0.5}, baseline=defaults)

# Get all params (merged with defaults)
merged = custom.get()
```

### Accessing Parameters

```python
# Get all parameters
all_params = params.get()

# Get specific parameters
subset = params.get(["temperature", "model"])

# Set parameters
params.set(temperature=0.9, verbose=True)

# Set from JSON string
params.set(nest=True, config='{"temperature": 0.8}')
```

## Creating and Running Agents

### Basic Agent Configuration

Create a configuration JSON:
```json
{
  "name": "classifier",
  "api_key": "settings/api_key.txt",
  "input_file": "inputs/data.csv",
  "dataset_name": "my_dataset",
  "dataset_id_column": "id",
  "dataset_columns": ["text"],
  "instructions": "instructions/prompt.txt",
  "model": {
    "client": "openai.OpenAI",
    "version": "gpt-5.1",
    "temperature": 0.7,
    "max_tokens": 4096
  },
  "api_client": {
    "top_p": 1,
    "presence_penalty": 0
  },
  "output_path": "results",
  "chunk_size": 10
}
```

### Running an Agent

```python
from sagemage import Agent
import json

# Load configuration
with open("config.json") as f:
    config = json.load(f)

# Create agent
agent = Agent(config)

# Run the agent
results = agent.run(project_dir="/path/to/project")

# Results is a pandas DataFrame
print(results)
print(results.head())
```

### Agent Lifecycle

1. **Configuration** - Define parameters including input files, API key, instructions
2. **Dataset Loading** - Agent loads data from `input_file`
3. **Chunking** - Data is split into chunks based on `chunk_size`
4. **Processing** - For each chunk:
   - Read instructions and resources
   - Build prompt with data
   - Call LLM API
   - Parse responses
   - Save per-chunk results
5. **Results** - Concatenated results returned as DataFrame

## Managing Datasets

### Creating a Dataset

```python
from sagemage import Dataset

dataset = Dataset(
    input_path="data/input.csv",
    output_path="data/output",
    id_column="id",
    columns=["name", "description"],
    chunk_size=5
)

# Access chunks
for chunk_name, chunk_data in dataset.chunks.items():
    print(f"Processing {chunk_name}")
    print(chunk_data)
```

### Supported Operations

- **Loading:** CSV files with pandas
- **Grouping:** By ID column
- **Chunking:** Fixed size batches
- **Selection:** Specific column subsets
- **Flattening:** Nested structures to tabular format

## API Integration

### ApiClient Usage

```python
from sagemage import ApiClient

client = ApiClient(api_key="sk-...")

# Construct a prompt
prompt = client.construct_prompt(
    items=["instruction text"],
    data=["item1", "item2", "item3"],
    resources=["<context>metadata</context>"]
)

# Get response from LLM
response = client.get_response(prompt)

# Parse response
results_df = client.parse_response(
    response,
    format="json",
    json_node="results"
)
```

### Agent-based LLM client params (recommended)

`Agent.run()` builds `ApiClient(...)` from agent params using this order:

1. Start from `model`:
   - If string, it becomes `{"model": "<value>"}`.
   - If object, all keys are used.
2. Merge `api_client` on top (overrides `model` keys).
3. If `version` exists and `model` is missing, `version` is renamed to `model`.
4. Top-level params (`temperature`, `max_tokens`, `top_p`, `frequency_penalty`, `presence_penalty`) are applied only if missing.
5. `api_key` can be provided at top level or inside `model`/`api_client` as fallback.

For OpenAI GPT-5-family models (`gpt-5*`), SAGEMAGE keeps `max_tokens` in config but sends it as `max_completion_tokens` in the API call.

```json
{
  "my_agent": {
    "api_key": "settings/api_key.txt",
    "model": {
      "client": "openai.OpenAI",
      "version": "gpt-5.1",
      "temperature": 0.2
    },
    "api_client": {
      "model": "gpt-5.1-mini",
      "max_tokens": 2048
    }
  }
}
```

In this example, effective values are `model="gpt-5.1-mini"`, `temperature=0.2`, `max_tokens=2048`.

### Response Parsing Formats

`ApiClient.parse_response()` supports:

- `format="json"`: parse JSON and extract `json_node`.
- `format="csv"`: parse line-based CSV-like text.
- `format="txt"`: store raw model text in a single `response` column.

```python
results_df = client.parse_response(response, format="txt")
```

## Built-In Examples

### Running the FooFoo Example

The foofoo example demonstrates the complete workflow: config, input data, instructions, and agent execution.

#### Method 1: Direct Function Call

```python
from sagemage.examples.foofoo.py.foofoo import foofoo

# Create and run in default location
results = foofoo()

# Run with custom location and API key
results = foofoo(
    base_dir="/my/work/foofoo_test",
    api_key_value="sk-your-key-here"
)

# Run with reset to start fresh after experiments
results = foofoo(
    base_dir="/my/work/foofoo_test",
    reset=True
)
```

#### Method 2: Module Execution

```bash
# From command line
python -m sagemage.examples.foofoo.py.foofoo
```

#### What the FooFoo Example Does

1. **Creates Local Workspace:**
   - `settings/foofoo.json` - Agent configuration
   - `settings/api_key.txt` - API key
   - `inputs/foofoo_input.csv` - Sample data (animals, objects, etc.)
   - `instructions/foofoo.txt` - Classification prompt
   - `results/` - Output directory

2. **Runs Classification Task:**
   - Reads items from CSV
   - Sends to LLM with classification instruction
   - Parses structured JSON response
   - Returns results as DataFrame

3. **Example Output:**
   ```
   | item_id | name              | type      |
   |---------|-------------------|-----------|
   | 1       | cat               | animal    |
   | 2       | dog               | animal    |
   | 3       | apple             | fruit     |
   | 4       | tree              | plant     |
   | 5       | astronaut         | person    |
   ```

#### Using the Reset Flag

```python
# First run - creates workspace
foofoo(base_dir="/my/work/test1")

# After experimenting, reset to clean state
foofoo(base_dir="/my/work/test1", reset=True)

# This removes config/inputs/instructions and rebuilds them
# Useful for restarting after modifications or failed runs
```

## Advanced Usage

### Custom Agent Configuration

```python
# Full configuration with all options
config = {
    "type": "GENERATIVE_ANALYZER",
    "name": "advanced_agent",
    "run": True,
    
    # Paths
    "project_dir": "./",
    "api_key": "settings/api_key.txt",
    "input_file": "inputs/data.csv",
    "instructions": "instructions/prompt.txt",
    "resources": "resources/context.json",
    "output_path": "results",
    
    # Dataset configuration
    "dataset_name": "dataset",
    "dataset_id_column": "id",
    "dataset_columns": ["text", "metadata"],
    "textual": "text",  # Primary text column
    
    # Processing
    "chunk_size": 10,
    "max_items": -1,  # -1 means no limit
    
    # Output
    "output_file": "results.csv",
    "output_extension": "json",
    "json_node": "results",
    "result_columns": ["id", "text", "classification"],

    # LLM client configuration
    "model": {
        "client": "openai.OpenAI",
        "version": "gpt-5.1",
        "temperature": 0.7
    },
    "api_client": {
        "max_tokens": 4096
    },
    
    # Metadata
    "verbose": True,
    "analyze": False
}

agent = Agent(params_dict=config)
results = agent.run(project_dir="/data/my_project")
```

### Utility Functions

```python
from sagemage import print_dict, flatten_dataframe, save_to_path

# Pretty print configuration
print_dict(agent.get())

# Flatten nested DataFrames
flat_df = flatten_dataframe(nested_df, sep="_")

# Save results with automatic path creation
save_to_path(results, "outputs/analysis/results.csv")
```

### Extending for Custom Workflows

```python
from sagemage import Agent, Dataset, ApiClient

class CustomAgent(Agent):
    def run_custom(self, project_dir):
        # Custom processing logic
        dataset = Dataset(
            input_path=self.get(["input_file"])[0],
            chunk_size=self.get(["chunk_size"])[0]
        )
        
        # Your custom business logic here
        return dataset

agent = CustomAgent(config)
results = agent.run_custom(project_dir)
```

## Inlets and Outlets

### Multi-Inlet Processing

Agents can consume multiple input datasets with independent chunking. Each inlet can specify its own dataset configuration.

```json
{
  "multi_input_agent": {
    "inlets": [
      {
        "uri": "data/objects.csv",
        "type": "FS",
        "dataset_name": "objects",
        "dataset_id_column": "id",
        "dataset_columns": ["name", "description"],
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
    ]
  }
}
```

**Inlet Configuration:**
- `uri` - Path to data file (relative to project_dir for FS type)
- `type` - Inlet type: `FS` (file system), `AM` (agent memory)
- `dataset_name` - Name for this dataset in prompts
- `dataset_id_column` - Column containing unique IDs
- `dataset_columns` - Columns to import from this file
- `chunk_size` - Chunk size for this inlet (-1 = no chunking)

**Cartesian Combinations:**
When multiple inlets are provided, chunks are combined as a cartesian product:
- Inlet 1: 3 chunks
- Inlet 2: 1 chunk (no chunking)
- Result: 3 combinations → 3 API calls with different data combinations

### Outlet Configuration

Outlets define how results are saved. Each outlet can specify different result columns.

```json
{
  "my_agent": {
    "outlets": [
      {
        "uri": "results/all_fields.csv",
        "type": "FS",
        "dataset_name": "results",
        "result_columns": ["id", "name", "category", "analysis"]
      },
      {
        "uri": "results/summary.csv",
        "type": "FS",
        "dataset_name": "summary",
        "result_columns": ["id", "analysis"]
      },
      {
        "uri": "my_agent",
        "type": "AM",
        "result_columns": ["id", "name", "category", "analysis"]
      }
    ]
  }
}
```

**Outlet Configuration:**
- `uri` - File path for FS type, or agent_id for AM type
- `type` - Outlet type: `FS` (file), `AM` (agent memory), `S3`, `KV`
- `dataset_name` - Optional name for identification
- `result_columns` - Columns to include in this outlet

**Outlet Types:**
- **FS** - Save to file system (CSV, JSON, JSONL)
- **AM** - Store in agent memory for downstream agents to consume
- **S3** - Save to S3 storage (future)
- **KV** - Save to key-value cache (future)

### Output File Extensions

Agent runs always produce CSV outputs, and can additionally save per-combination files based on `output_extension`:

- `"json"`: save parsed records as JSON.
- `"txt"`: save raw text response.

```json
{
  "my_agent": {
    "output_extension": "txt"
  }
}
```

### Agent Chaining with Agent Memory

Use AM (Agent Memory) inlets/outlets to chain agents together:

```json
{
  "_DEFAULT_": {
    "outlets": [{"uri": "output.csv", "type": "FS"}]
  },
  
  "classifier_agent": {
    "inlets": [{"uri": "data/input.csv", "type": "FS", "dataset_name": "items"}],
    "outlets": [
      {"uri": "classifier_results.csv", "type": "FS"},
      {"uri": "classifier_agent", "type": "AM"}
    ]
  },
  
  "analyzer_agent": {
    "inlets": [
      {"uri": "classifier_agent", "type": "AM", "chunk_size": 100}
    ],
    "outlets": [
      {"uri": "analysis_results.csv", "type": "FS"}
    ]
  }
}
```

**Running chained agents:**

```python
from sagemage import AgentSet

# Load configuration
agent_set = AgentSet("config.json")

# Run all agents in sequence
results = agent_set.run_all(project_dir="/my/workspace")

# Downstream agents automatically receive upstream results via AM inlets
```

Process:
1. `classifier_agent` processes input data
2. Results are stored in agent memory via AM outlet
3. `analyzer_agent` consumes results via AM inlet
4. `analyzer_agent` processes the received data
5. Final results saved to outlet

## Advanced Configuration

### Dynamic Output Paths

Output path is constructed as:
```
<project_dir>/<output_datasets['result_files']>/<outlet_uri>/<version>/
```

Example with _DEFAULT_:
```json
{
  "_DEFAULT_": {
    "project_dir": "./",
    "output_datasets": {
      "result_files": "result_files"
    },
    "output_version": "%Y-%m-%d-%H-%M-%S"
  }
}
```

Result files are saved to: `result_files/<outlet_uri>/<timestamp>/`

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'pandas'"
**Solution:** Install dependencies:
```bash
pip install sagemage[dev]
```

### Issue: API Key Not Found
**Solution:** Ensure api_key file exists and path is correct:
```python
# Verify path
with open("settings/api_key.txt") as f:
    key = f.read().strip()
```

### Issue: "No such file or directory: inputs/data.csv"
**Solution:** Verify input_file path is relative to project_dir:
```python
# If project_dir is "/my/work", input_file should be relative to that
agent.run(project_dir="/my/work")  # Looks for /my/work/inputs/data.csv
```

### Issue: Empty Results
**Solution:** Check if API is being called:
```python
# Verify API key is valid
# Check verbose output
config["verbose"] = True
agent = Agent(config)
results = agent.run()
```

## Resources

- [Architecture Documentation](./architecture.md) - Technical details on package structure
- [Contributing Guidelines](./CONTRIBUTING.md) - How to contribute
- [Changes Log](./changes.md) - Version history

## Support

For issues, questions, or contributions, visit:
https://github.com/Yaniv1/sagemage
