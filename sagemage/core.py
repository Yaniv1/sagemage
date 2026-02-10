"""Core SAGEMAGE classes for parameter management and agents."""

import json
import os
import datetime as dt
import itertools
from typing import Any, Dict, Optional

# local imports delayed inside methods to avoid circular imports at module import time


class ParamSet:
    """Manages sets of parameters with support for JSON file loading and nesting."""

    def __init__(self, params_dict: Dict[str, Any] = None, baseline: Optional["ParamSet"] = None):
        """
        Initialize ParamSet.

        Args:
            params_dict: Dictionary of parameters or path to JSON file
            baseline: Another ParamSet to use as baseline/defaults
        """
        if params_dict is None:
            params_dict = {}

        # Load from baseline if provided
        if baseline is not None:
            if isinstance(baseline, ParamSet):
                self.set(**baseline.get())
            else:
                raise TypeError("baseline must be a ParamSet object or None")

        # Load from JSON file if string path provided
        if isinstance(params_dict, str) and params_dict.endswith(".json"):
            if os.path.isfile(params_dict):
                self.source = params_dict
                try:
                    with open(params_dict, "r") as f:
                        params_dict = json.load(f)
                except Exception as e_get_json:
                    print(f"Error loading JSON: {e_get_json=} {params_dict=}")
                    params_dict = {}

        # Set parameters from dictionary
        if isinstance(params_dict, dict):
            self.set(**params_dict)

    def get(self, keys: list = None) -> Dict[str, Any]:
        """
        Get parameters.

        Args:
            keys: Optional list of specific keys to retrieve. If empty, returns all.

        Returns:
            Dictionary of parameters
        """
        if keys is None:
            keys = []

        if len(keys) == 0:
            return self.__dict__.copy()
        else:
            return {k: getattr(self, k) for k in keys if k in list(self.__dict__.keys())}

    def set(self, nest: bool = True, **kvpairs) -> None:
        """
        Set parameters.

        Args:
            nest: If True, attempt to parse JSON strings
            **kvpairs: Key-value pairs to set
        """
        for k, v in kvpairs.items():
            setattr(self, k, v)
            if nest and isinstance(v, str):
                # Try to load JSON file
                if v.endswith(".json") and os.path.isfile(v):
                    try:
                        with open(v, "r") as f:
                            setattr(self, k, json.load(f))
                    except Exception:
                        pass
                # Try to parse JSON string
                elif any([v.strip().startswith(c0) for c0 in ["{", "["]]):
                    if any([v.strip().endswith(c1) for c1 in ["}", "]"]]):
                        try:
                            setattr(self, k, json.loads(v))
                        except Exception as e_str_to_json:
                            print(f"Error parsing JSON string: {e_str_to_json=}\n{v}")


# Helper functions for inlet/outlet resolution
def resolve_inlet(inlet: Dict[str, Any], project_dir: str, agent_outputs: Dict[str, Any] = None, agents_dict: Dict[str, Any] = None) -> Optional[Any]:
    """
    Resolve an inlet to retrieve input data.
    
    Args:
        inlet: Dictionary with 'uri' and 'type' keys
        project_dir: Base directory for FS paths
        agent_outputs: Mapping of agent_id to outputs (for AM type)
        agents_dict: Mapping of agent_id to agent objects (unused, kept for compatibility)
    
    Returns:
        Data from the inlet (DataFrame for FS, agent memory, or KV; None if not found)
    """
    if agent_outputs is None:
        agent_outputs = {}
    
    inlet_type = inlet.get("type", "FS")
    inlet_uri = inlet.get("uri", "")
    
    if inlet_type == "FS":
        # File system input
        full_path = inlet_uri if os.path.isabs(inlet_uri) else os.path.join(project_dir, inlet_uri)
        if os.path.isfile(full_path):
            import pandas as pd
            try:
                return pd.read_csv(full_path)
            except Exception:
                try:
                    return pd.read_excel(full_path)
                except Exception:
                    return None
    elif inlet_type == "AM":
        # Agent memory input - retrieve from agent output by agent_id
        agent_id = inlet_uri
        agent_output = agent_outputs.get(agent_id)
        if agent_output is not None:
            return agent_output.results if hasattr(agent_output, 'results') else agent_output
        return None
    elif inlet_type == "KV":
        # Key-value cache input (placeholder for future implementation)
        return None
    elif inlet_type == "S3":
        # S3 storage input (placeholder for future implementation)
        return None
    
    return None


def save_outlet(outlet: Dict[str, Any], data: Any, output_path: str, version: str = None) -> bool:
    """
    Save data to an outlet.
    
    Args:
        outlet: Dictionary with 'uri' and 'type' keys
        data: Data to save (typically a DataFrame)
        output_path: Base output path for FS outlets
        version: Version string for timestamped outputs
    
    Returns:
        True if saved successfully, False otherwise
    """
    outlet_type = outlet.get("type", "FS")
    outlet_uri = outlet.get("uri", "")
    
    if outlet_type == "FS":
        # File system output - combine output_path with outlet_uri
        full_path = os.path.join(output_path, version if version else "", outlet_uri or "output.csv")
        full_path = full_path.rstrip(os.sep)  # Remove trailing separator
        os.makedirs(os.path.dirname(full_path), exist_ok=True)
        try:
            if full_path.endswith(".csv"):
                data.to_csv(full_path, index=False)
            elif full_path.endswith((".json", ".jsonl")):
                data.to_json(full_path, orient="records", lines=(full_path.endswith(".jsonl")))
            else:
                data.to_csv(full_path, index=False)
            return True
        except Exception as e:
            print(f"Error saving to outlet {outlet_uri}: {e}")
            return False
    elif outlet_type == "AM":
        # Agent memory - don't save, just return (will be stored in agent_outputs)
        return True
    elif outlet_type == "KV":
        # Key-value cache (placeholder)
        return True
    elif outlet_type == "S3":
        # S3 storage (placeholder)
        return True
    
    return False


class Agent(ParamSet):
    """Represents an agent node with configuration parameters."""

    def __init__(self, agent_id: str = None, params_dict: Dict[str, Any] = None, baseline: Optional[ParamSet] = None):
        """
        Initialize Agent.

        Args:
            agent_id: Unique identifier for this agent
            params_dict: Dictionary of agent parameters or path to JSON file
            baseline: Another ParamSet to use as baseline
        """
        if params_dict is None:
            params_dict = {}

        super().__init__(params_dict=params_dict, baseline=baseline)
        self.agent_id = agent_id
        # agent identifier (set by AgentSet or may be provided in config)
        self.agent_id = getattr(self, "agent_id", getattr(self, "name", None))

    def run(self, project_dir: Optional[str] = None, input_df: Optional[Any] = None, agent_outputs: Dict[str, Any] = None, agent_id: str = None, agents_dict: Dict[str, Any] = None):
        """Run the agent: load dataset, construct prompts, call API, and parse results.

        Args:
            project_dir: Base directory to resolve relative paths (defaults to agent.project_dir or current dir)
            input_df: Optional DataFrame to use as input (for backward compatibility with single input)
            agent_outputs: Mapping of agent_id to outputs (for agent memory resolution)
            agent_id: ID of this agent (used for storing outputs)
            agents_dict: Mapping of agent_id to agent objects (for resolving AM inlets by agent name)

        Returns:
            pandas.DataFrame: concatenated results from API parsing (may be empty)
        """
        # Import here to avoid circular imports
        from .dataset import Dataset
        from .api_client import ApiClient

        import pandas as pd

        if agent_outputs is None:
            agent_outputs = {}
        if agents_dict is None:
            agents_dict = {}

        # Resolve project directory
        project_dir = project_dir or getattr(self, "project_dir", os.getcwd())

        # Determine output_path from outlets and output_datasets
        output_datasets = getattr(self, "output_datasets", {})
        outlets = getattr(self, "outlets", None)
        if outlets and not isinstance(outlets, list):
            outlets = [outlets]

        # Find FS type outlet URI
        outlet_uri = None
        if outlets:
            for outlet in outlets:
                if outlet.get("type") == "FS":
                    outlet_uri = outlet.get("uri", "")
                    break

        # Construct output_path: project_dir / output_datasets['result_files'] / outlet_uri
        if outlet_uri:
            output_path = os.path.join(
                project_dir,
                output_datasets.get("result_files", "result_files"),
                outlet_uri
            )
        else:
            output_path = os.path.join(project_dir, "results")

        # Collect all datasets from inlets - use dict keyed by inlet uri
        datasets_to_process = {}
        
        # Check for new inlets parameter
        inlets = getattr(self, "inlets", None)
        if inlets:
            # Multiple inputs via inlets
            if not isinstance(inlets, list):
                inlets = [inlets]
            
            for inlet in inlets:
                inlet_data = resolve_inlet(inlet, project_dir, agent_outputs, agents_dict)
                if inlet_data is not None:
                    # Create a Dataset for each inlet using attributes from the inlet
                    dataset = Dataset(
                        input_path="",
                        output_path=output_path,
                        id_column=inlet.get("dataset_id_column", getattr(self, "dataset_id_column", "id")),
                        columns=inlet.get("dataset_columns", getattr(self, "dataset_columns", [])),
                        chunk_size=inlet.get("chunk_size", getattr(self, "chunk_size", 1)),
                    )
                    dataset.df = inlet_data.copy() if hasattr(inlet_data, 'copy') else inlet_data
                    dataset.inlet = inlet  # Store inlet for later reference
                    datasets_to_process[inlet.get("uri", "")] = dataset
        elif input_df is not None:
            # Single input from parameter (backward compatibility)
            dataset = Dataset(
                input_path="",
                output_path=output_path,
                id_column=getattr(self, "dataset_id_column", "id"),
                columns=getattr(self, "dataset_columns", []),
                chunk_size=getattr(self, "chunk_size", 1),
            )
            dataset.df = input_df.copy()
            datasets_to_process["input_df"] = dataset
        else:
            # Backward compatibility: check for input_file or input_from
            input_from = getattr(self, "input_from", None)
            if input_from and input_from in agent_outputs:
                # Get from agent memory
                input_df = agent_outputs[input_from]
                dataset = Dataset(
                    input_path="",
                    output_path=output_path,
                    id_column=getattr(self, "dataset_id_column", "id"),
                    columns=getattr(self, "dataset_columns", []),
                    chunk_size=getattr(self, "chunk_size", 1),
                )
                dataset.df = input_df.copy()
                datasets_to_process[input_from] = dataset
            else:
                # Input file
                input_file = getattr(self, "input_file", None)
                if not input_file:
                    raise ValueError("Agent missing required 'input_file', 'inlets', or 'input_from' parameter")

                # Build full input path
                full_input_path = input_file if os.path.isabs(input_file) else os.path.join(project_dir, input_file)

                # Prepare dataset
                dataset = Dataset(
                    input_path=full_input_path,
                    output_path=output_path,
                    id_column=getattr(self, "dataset_id_column", "id"),
                    columns=getattr(self, "dataset_columns", []),
                    chunk_size=getattr(self, "chunk_size", 1),
                )
                datasets_to_process["input_file"] = dataset
        
        # Ensure textual column exists for all datasets
        textual_col = getattr(self, "textual", "ITEM")
        for inlet_uri, dataset in datasets_to_process.items():
            if textual_col not in dataset.df.columns:
                # create textual representation
                cols = dataset.columns or [c for c in dataset.df.columns if c != dataset.id_column]
                items = [", ".join([str(row[c]) for c in cols]) for _, row in dataset.df.iterrows()]
                ids = list(dataset.df[dataset.id_column]) if dataset.id_column in dataset.df.columns else list(dataset.df.index)
                dataset.df[textual_col] = [str(ids[i]) + ". " + items[i] for i in range(len(items))]
            dataset.group()
            dataset.chunk()

        # Read instructions
        inst = ""
        if hasattr(self, "instructions") and self.instructions:
            inst_path = self.instructions if os.path.isabs(self.instructions) else os.path.join(project_dir, self.instructions)
            try:
                with open(inst_path, "r", encoding="utf-8") as f:
                    inst = f.read()
            except Exception:
                inst = str(self.instructions)

        # Read resources (expecting JSON mapping)
        resources = {}
        if hasattr(self, "resources") and self.resources:
            res_path = self.resources if os.path.isabs(self.resources) else os.path.join(project_dir, self.resources)
            try:
                with open(res_path, "r", encoding="utf-8") as f:
                    resources = json.load(f)
            except Exception:
                # if resources is not a file, try parsing as JSON string or leave empty
                try:
                    resources = json.loads(self.resources)
                except Exception:
                    resources = {}

        # Read API key (file or direct)
        api_key = getattr(self, "api_key", "")
        if api_key:
            key_path = api_key if os.path.isabs(api_key) else os.path.join(project_dir, api_key)
            if os.path.isfile(key_path):
                try:
                    with open(key_path, "r", encoding="utf-8") as f:
                        api_key = f.read().strip()
                except Exception:
                    pass

        # Initialize API client (attempt to use openai if available)
        client_cls = None
        try:
            import openai

            client_cls = openai.OpenAI
        except Exception:
            client_cls = None

        api_client = ApiClient(client=client_cls, api_key=api_key)

        results = pd.DataFrame()
        
        # Get outlets configuration
        outlets = getattr(self, "outlets", None)
        if outlets and not isinstance(outlets, list):
            outlets = [outlets]
        
        # decide whether to save results based on presence of outlets
        should_save = bool(outlets)
        
        # Get version for timestamped outputs
        version = dt.datetime.now().strftime(getattr(self, "output_version", "%Y-%m-%d-%H-%M-%S"))

        # Collect chunks from all datasets
        all_chunks = {}  # {inlet_uri: [(chunk_key, chunk_df), ...]}
        for inlet_uri, dataset in datasets_to_process.items():
            chunks_list = []
            chunks_dict = getattr(dataset, "chunks", {})
            if not chunks_dict:
                # If no chunks, treat entire dataset as single chunk
                chunks_list = [("full", dataset.df)]
            else:
                chunks_list = list(chunks_dict.items())
            all_chunks[inlet_uri] = chunks_list
        
        # Create combinations of chunks from all inlets (cartesian product)
        inlet_uris = list(all_chunks.keys())
        chunk_lists = [all_chunks[uri] for uri in inlet_uris]
        
        # Generate all combinations
        for combination in itertools.product(*chunk_lists):
            # combination is a tuple of (chunk_key, chunk_df) for each inlet
            # Build a dict of {inlet_uri: (chunk_key, chunk_df)}
            combo_dict = {inlet_uris[i]: combination[i] for i in range(len(inlet_uris))}
            
            # Construct prompt with data from all inlets
            textual_col = getattr(self, "textual", "ITEM")
            prompt_data = [inst]  # Start with instructions
            
            for inlet_uri, (chk_key, chv) in combo_dict.items():
                dataset = datasets_to_process[inlet_uri]
                inlet = getattr(dataset, "inlet", {})
                dataset_name = inlet.get("dataset_name", getattr(self, "dataset_name", "dataset"))
                
                # Extract data from chunk
                data = list(chv[textual_col].values) if textual_col in chv.columns else [str(r) for r in chv.values]
                prompt_data.append(f"<{dataset_name}>")
                for d in data:
                    prompt_data.append(f"   {d}")
                prompt_data.append(f"</{dataset_name}>")
            
            # Construct full prompt
            prompt = api_client.construct_prompt(
                items=prompt_data,
                data=[],  # Data already in items
                resources=[f"<{rk}>\n    {rv}\n</{rk}>" for rk, rv in resources.items()],
            )

            # Call API
            response = api_client.get_response(prompt=prompt)
            
            # Get result_columns from first FS outlet
            result_columns = None
            for outlet in outlets:
                if outlet.get("type") == "FS":
                    result_columns = outlet.get("result_columns", getattr(self, "result_columns", None))
                    break

            # Parse response
            df = api_client.parse_response(
                response,
                format=getattr(self, "output_extension", "json"),
                json_node=getattr(self, "json_node", "results"),
                id_column=getattr(self, "dataset_id_column", "id"),
                data_columns=result_columns,
            )

            # Save per-combination if requested
            if should_save:
                out_dir = os.path.join(output_path, version)
                os.makedirs(out_dir, exist_ok=True)
                combination_key = "_".join([str(combo_dict[uri][0]) for uri in inlet_uris])
                chunk_name = combination_key + "_" + getattr(self, "output_file", "results.csv")
                try:
                    df.to_csv(os.path.join(out_dir, chunk_name), index=False)
                except Exception:
                    pass

            results = pd.concat([results, df]) if not results.empty else df
        
        # Handle outlets - save to configured output locations
        if outlets:
            for outlet in outlets:
                outlet_type = outlet.get("type", "FS")
                if outlet_type in ["FS", "S3", "KV"]:
                    # Persistent storage - save the combined results
                    save_outlet(outlet, results, output_path, version)
                elif outlet_type == "AM":
                    # Agent memory - just mark that this is available
                    # Data will be retrieved by next agent via inlets with type="AM"
                    pass

        self.results = results
        return results


class AgentSet:
    """Represents a collection of Agents loaded from a JSON configuration.

    Example JSON structures supported:

    - Root mapping of agent_id -> agent_config
    - {"AGENTS": {agent_id: config, ...}, "_DEFAULT_": {...}}

    """

    def __init__(self, source: str, node: Optional[str] = None):
        """Load agents from `source` (path to JSON file or dict-like)."""
        if isinstance(source, str) and os.path.isfile(source):
            with open(source, "r", encoding="utf-8") as f:
                data = json.load(f)
        elif isinstance(source, dict):
            data = source
        else:
            raise ValueError("source must be a path to a JSON file or a dict")

        # navigate to node if provided
        if node:
            data = data.get(node, {}) if isinstance(data, dict) else {}

        # baseline defaults
        baseline_cfg = None
        if isinstance(data, dict) and "_DEFAULT_" in data:
            baseline_cfg = data.get("_DEFAULT_")

        # if the node contains AGENTS mapping, use that
        if isinstance(data, dict) and "AGENTS" in data:
            agents_map = data.get("AGENTS", {})
        else:
            # otherwise assume that top-level keys mapping to dicts are agents
            agents_map = {k: v for k, v in data.items() if isinstance(v, dict)}

        self.baseline = ParamSet(baseline_cfg) if baseline_cfg else None
        self.agents = {}
        for aid, cfg in agents_map.items():
            agent = Agent(cfg, baseline=self.baseline)
            agent.agent_id = aid
            self.agents[aid] = agent

    def list_ids(self):
        return list(self.agents.keys())

    def run_all(self, project_dir: Optional[str] = None, verbose: bool = True):
        results = {}
        # map of outputs by agent id (for agent memory inlets)
        agent_outputs = {}
        for agent_id, agent in self.agents.items():
            if verbose:
                print(f"== Running agent: {agent_id} ==")
            # print about if available
            about = getattr(agent, "about", None)
            if about:
                # about may be a path relative to project_dir
                about_path = about if os.path.isabs(about) else os.path.join(project_dir or getattr(agent, "project_dir", "."), about)
                try:
                    with open(about_path, "r", encoding="utf-8") as f:
                        print(f.read())
                except Exception:
                    try:
                        # fallback to raw string
                        print(str(about))
                    except Exception:
                        pass

            # determine if this agent consumes previous agent output
            # Check for new inlets first
            inlets = getattr(agent, "inlets", None)
            input_from = getattr(agent, "input_from", None)
            
            # Pass agent_outputs and agents_dict so inlets with type="AM" can access previous outputs
            res = agent.run(project_dir=project_dir, agent_outputs=agent_outputs, agent_id=agent_id, agents_dict=self.agents)

            # Store outputs for downstream agents (both old and new methods)
            agent_outputs[agent_id] = res
        return results
