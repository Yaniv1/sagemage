"""Core SAGEMAGE classes for parameter management and agents."""

import json
import os
import pandas as pd
import datetime as dt
import itertools
from typing import Any, Dict, Optional
import inspect

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
def resolve_inlet(inlet: Dict[str, Any], project_dir: str, agent_outputs: Dict[str, Any] = None, **context) -> Optional[Any]:
    """
    Resolve an inlet to retrieve input data.
    
    Args:
        inlet: Dictionary with 'uri' and 'type' keys
        project_dir: Base directory for FS paths
        agent_outputs: Mapping of agent_id to outputs (for AM type)
        context: global or local kv caches for KV type
    
    Returns:
        Data from the inlet (DataFrame for FS, agent memory, or KV; None if not found)
    """
    if agent_outputs is None:
        agent_outputs = {}
    
    inlet_type = inlet.get("type", "FS")
    inlet_uri = inlet.get("uri", "")
    
    if inlet_type == "FS":
        # File system input
        input_path = inlet_uri if os.path.isabs(inlet_uri) else os.path.join(project_dir, inlet_uri)        
        input_data = None
    elif inlet_type == "AM":
        # Agent memory input - retrieve from agent output by agent_id
        input_path = inlet_uri
        input_data = agent_outputs.get(inlet_uri) 
        print(f"Retrieved agent output for inlet_uri={inlet_uri}:\n{input_data}")
        
    elif inlet_type == "KV":
        # Key-value cache input (placeholder for future implementation)     
        input_path  = inlet_uri
        input_data = pd.DataFrame(context.get(inlet_uri))
        
    elif inlet_type == "S3":
        # S3 storage input (placeholder for future implementation)
        input_path  = inlet_uri
        input_data = None
    
    return input_path, input_data


def save_outlet(outlet: Dict[str, Any], data: Any, output_path: str) -> bool:
    """
    Save data to an outlet.
    
    Args:
        outlet: Dictionary with 'uri' and 'type' keys
        data: Data to save (typically a DataFrame)
        output_path: Base output path for FS outlets
    
    Returns:
        True if saved successfully, False otherwise
    """
    outlet_type = outlet.get("type", "FS")
    outlet_uri = outlet.get("uri", "")
    
    if outlet_type == "FS":
        # File system output - combine output_path with outlet_uri
        full_path = output_path        
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
        
        
        
        if not(hasattr(self,'verbose')):
            self.verbose = False
            
        if not(hasattr(self, "delta")):
            self.delta = False
            
        if self.verbose:
            print(f"Initialized agent {self.agent_id} with parameters:")
            for k,v in self.get().items():
                print(f"  {k}: {v}")
                
        
        

    def run(self, project_dir: Optional[str] = None, agent_outputs: Dict[str, Any] = None, agents_dict: Dict[str, Any] = None):
        """Run the agent: load dataset, construct prompts, call API, and parse results.

        Args:
            project_dir: Base directory to resolve relative paths (defaults to agent.project_dir or current dir)
            agent_outputs: Mapping of agent_id to outputs (for agent memory resolution)
            agent_id: ID of this agent (used for storing outputs)
            agents_dict: Mapping of agent_id to agent objects (for resolving AM inlets by agent name)

        Returns:
            pandas.DataFrame: concatenated results from API parsing (may be empty)
        """
        # Import here to avoid circular imports
        from .dataset import Dataset
        from .api_client import ApiClient
        from .utils import print_dict
        

        if agent_outputs is None:
            agent_outputs = {}
        if agents_dict is None:
            agents_dict = {}
            
        if self.verbose: 
            print("Agent Results:\n")
            print_dict(agent_outputs)

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

        # Get version for timestamped outputs
        version = dt.datetime.now().strftime(getattr(self, "output_version", "%Y-%m-%d-%H-%M-%S"))


        # Construct output_path: project_dir / output_datasets['result_files'] / outlet_uri
        output_path = os.path.join(
            project_dir,
            output_datasets.get("result_files", "result_files"),
            self.agent_id,
            version if version else ""                
        )
        
        if self.verbose: print(f"{output_path=}")
        
        # Ensure output_path exists
        os.makedirs(output_path, exist_ok=True)

        # Collect all datasets from inlets - use dict keyed by inlet uri
        datasets = {}
        
        # Ensure textual column exists for all datasets
        textual_col = getattr(self, "textual", "ITEM")
        
        # Check for new inlets parameter
        inlets = getattr(self, "inlets", [])
        
        if inlets:
                        
            for inlet in inlets:
                inlet_path, inlet_data = resolve_inlet(inlet, project_dir, agent_outputs=agent_outputs)
                # Create a Dataset for each inlet using attributes from the inlet
                if self.verbose:
                    print(f"{inlet_data=}")
                if inlet_data is not None:
                    if self.verbose: print(f"Resolved inlet {inlet_path} from agent memory with data:\n{inlet_data}") 
                dataset = Dataset(
                    name = inlet.get("dataset_name", inlet.get("uri", inlet_path)),
                    input_path=inlet_path,
                    input_data=inlet_data,
                    output_path=f"{output_path}/{inlet.get('dataset_name', inlet.get('uri', 'dataset'))}".replace('\\','/').replace('.','_'),
                    id_column=inlet.get("dataset_id_column","id"),
                    columns=inlet.get("dataset_columns", []),
                    chunk_size=inlet.get("chunk_size", 1),
                    verbose=self.verbose
                )
                dataset.inlet = inlet
                
                if textual_col not in dataset.df.columns:
                    # create textual representation
                    cols = dataset.columns or [c for c in dataset.df.columns if c != dataset.id_column]
                    items = [", ".join([str(row[c]) for c in cols]) for _, row in dataset.df.iterrows()]
                    ids = list(dataset.df[dataset.id_column]) if dataset.id_column in dataset.df.columns else list(dataset.df.index)
                    dataset.df[textual_col] = [str(ids[i]) + ". " + items[i] for i in range(len(items))]
                    dataset.group()
                    dataset.chunk()
            
                if self.verbose: print(f"{inlet_path}:\n",dataset.df.head())
                
                datasets.update({inlet_path:dataset})

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

        
        
        # Get outlets configuration
        outlets = getattr(self, "outlets", None)
        if outlets and not isinstance(outlets, list):
            outlets = [outlets]
        
        outlets_df = pd.DataFrame(outlets)
        for i,row in outlets_df.iterrows():
            if row["type"] == "FS":
                # If FS outlet has no URI, set it to a default based on agent_id
                outlets_df.loc[i,"output_path"] = os.path.join(output_path, f"{row.get('dataset_name',row.get('uri'))}").replace('\\','/')
                
        if self.verbose: print("Outlets:\n",outlets_df)
        
        # decide whether to save results based on presence of outlets
        should_save = bool(outlets)
        
        
        # Collect chunks from all datasets
        all_chunks = {}  # {inlet_uri: [(chunk_key, chunk_df), ...]}
        for inlet_uri, dataset in datasets.items():
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
        if self.verbose: print(chunk_lists)
        
        # Generate all combinations
        combinations = list(itertools.product(*chunk_lists))
        if self.verbose: print(combinations)
        
        results = pd.DataFrame()
        
        fs_outlet_path = outlets_df[outlets_df["type"] == "FS"]["output_path"].iloc[0] if not outlets_df[outlets_df["type"] == "FS"].empty else None
        prev_files = os.listdir(fs_outlet_path) if fs_outlet_path and os.path.exists(fs_outlet_path) else []     
        
        textual_col = getattr(self, "textual", "ITEM")
        prompt_items = [inst]  # Start with instructions  
        if self.verbose: print(f"{prompt_items=}")        
        
        prompt_resources = [f"<{rk}>\n    {rv}\n</{rk}>" for rk, rv in resources.items()]   
        if self.verbose: print(f"{prompt_resources=}")
                
        for combination in combinations:
            # combination is a tuple of (chunk_key, chunk_df) for each inlet
            # Build a dict of {inlet_uri: (chunk_key, chunk_df)}
            combo_dict = {inlet_uris[i]: combination[i] for i in range(len(inlet_uris))}
            combination_key = "_".join([str(combo_dict[uri][0]).split('/')[-1].split('.')[0] for uri in inlet_uris])
            
            comb_df = None
            if self.delta:           
                # try to load available results for specific chunk combination.     
                if len(prev_files) > 0 and f"{combination_key}.csv" in prev_files:
                    # chunck combination already analyzed.
                    comb_df = pd.read_csv(os.path.join(fs_outlet_path, f"{combination_key}.csv" ).replace('\\','/')) 
                    if self.verbose: print(f"Loaded existing results for combination {combination_key} from {fs_outlet_path}")
                    
            if comb_df is None:
                # Construct prompt with data from all inlets                
                prompt_data = []
                for inlet_uri, (chk_key, chv) in combo_dict.items():
                    dataset = datasets[inlet_uri]
                    inlet = getattr(dataset, "inlet", {})
                    dataset_name = inlet.get("dataset_name", inlet.get("uri"))
                    
                    # Extract data from chunk
                    data = list(chv[textual_col].values) if textual_col in chv.columns else [str(r) for r in chv.values]
                    prompt_data.append(f"<{dataset_name}>")
                    for d in data:
                        prompt_data.append(f"   {d}")
                    prompt_data.append(f"</{dataset_name}>")

                if self.verbose: print(f"{prompt_data=}")
                
                # Construct ApiClient              
                api_client = ApiClient(client=client_cls, api_key=api_key)
                # Construct full prompt with instructions, resources, and data  
                prompt = api_client.construct_prompt(
                    items=prompt_items,
                    data=prompt_data,
                    resources=prompt_resources,
                )
            
                if self.verbose: print("Prompt:\n",prompt)

                # Call API
                response = api_client.get_response(prompt=prompt)
                if self.verbose: 
                    print("Response:\n", response)
                    
                # save response to outlet. Use combination of chunk keys for filename            
                response_path = os.path.join(output_path, "RESPONSES", f"{combination_key}.json").replace('\\','/')
                os.makedirs(os.path.dirname(response_path), exist_ok=True)
                if self.verbose: print(f"Saving response to {response_path}")            
                with open(response_path, "w", encoding="utf-8") as f:
                    json.dump(response, f, indent=4, default=str)
                
                # Get result_columns from first FS outlet
                
                # Parse response
                comb_df = api_client.parse_response(
                    response,
                    format=getattr(self, "output_extension", "json"),
                    json_node=getattr(self, "json_node", "results"),
                    id_column=None,
                    data_columns=getattr(self, "result_columns", []),
                )
                
                # Handle outlets - save to configured output locations
            
                for i,row in outlets_df.iterrows():
                    outlet = dict(row)
                    outlet_type = outlet.get("type", "FS")
                    if outlet_type in ["FS", "S3", "KV"]:
                        # Persistent storage - save the combined results
                        comb_result_path = os.path.join(outlet["output_path"], 
                                                        f"{combination_key}.csv").replace('\\','/')
                        if self.verbose: print(f"Saving combination result to {comb_result_path} for outlet {outlet.get('dataset_name',outlet.get('uri'))}")
                        save_outlet(outlet, comb_df, comb_result_path)
                    elif outlet_type == "AM":
                        # Agent memory - just mark that this is available
                        # Data will be retrieved by next agent via inlets with type="AM"
                        pass     
            
            # add combination result to result master dataframe
            results = pd.concat([results, comb_df]) if not results.empty else comb_df

        results.reset_index(drop=True, inplace=True)
        # Handle outlets - save to configured output locations        
        for i,row in outlets_df.iterrows():
            outlet = dict(row)
            outlet_type = outlet.get("type", "FS")
            if outlet_type in ["FS", "S3", "KV"]:
                # Persistent storage - save the combined results
                result_path = os.path.join(outlet["output_path"], 
                                            f"{outlet.get('dataset_name',outlet.get('uri'))}.csv").replace('\\','/')
                save_outlet(outlet, results, result_path)
            elif outlet_type == "AM":
                # Agent memory - just mark that this is available
                # Data will be retrieved by next agent via inlets with type="AM"
                pass          
        
        if self.verbose: print("Final combined results:\n", results)
    
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
            # remove _DEFAULT_ from agent configs if present
            data.pop("_DEFAULT_", None)
        self.verbose = baseline_cfg.get('verbose',False)
           

        # if the node contains AGENTS mapping, use that
        if isinstance(data, dict) and "AGENTS" in data:
            agents_map = data.get("AGENTS", {})
        else:
            # otherwise assume that top-level keys mapping to dicts are agents
            agents_map = {k: v for k, v in data.items() if isinstance(v, dict)}

        if self.verbose: print(f"Loaded agents: {list(agents_map.keys())}")
        if self.verbose: print(f"Baseline config: {baseline_cfg}")
        self.baseline = ParamSet(baseline_cfg) if baseline_cfg else None
        self.agents = {}
        for aid, cfg in agents_map.items():
            agent = Agent(agent_id=aid, params_dict=cfg, baseline=self.baseline)
            agent.agent_id = aid
            self.agents[aid] = agent
            if self.verbose: print(f"agent: {agent.get()}\n")

    def list_ids(self):
        return list(self.agents.keys())

    def run_all(self, project_dir: Optional[str] = None):
        results = {}
        # map of outputs by agent id (for agent memory inlets)
        agent_outputs = {}
        for agent_id, agent in self.agents.items():
            if self.verbose:
                if self.verbose: print(f"== Running agent: {agent_id} ==")
            # print about if available
            about = getattr(agent, "about", None)
            if about:
                # about may be a path relative to project_dir
                about_path = about if os.path.isabs(about) else os.path.join(project_dir or getattr(agent, "project_dir", "."), about)
                try:
                    with open(about_path, "r", encoding="utf-8") as f:
                        if self.verbose: print(f.read())
                except Exception as err_about:
                    try:
                        # fallback to raw string
                        if self.verbose: print(str(about))
                    except Exception:
                        pass
                        
            # Pass agent_outputs and agents_dict so inlets with type="AM" can access previous outputs
            agent_results = agent.run(project_dir=project_dir, agent_outputs=agent_outputs, agents_dict=self.agents)

            # Store outputs for downstream agents (both old and new methods)
            agent_outputs[agent_id] = agent_results
            
            if self.verbose: print(f"Agent {agent_id} output:\n{agent_outputs[agent_id]}")
            
        return results
