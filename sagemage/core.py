"""Core SAGEMAGE classes for parameter management and agents."""

import json
import os
import datetime as dt
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


class Agent(ParamSet):
    """Represents an agent node with configuration parameters."""

    def __init__(self, params_dict: Dict[str, Any] = None, baseline: Optional[ParamSet] = None):
        """
        Initialize Agent.

        Args:
            params_dict: Dictionary of agent parameters or path to JSON file
            baseline: Another ParamSet to use as baseline
        """
        if params_dict is None:
            params_dict = {}

        super().__init__(params_dict=params_dict, baseline=baseline)

    def run(self, project_dir: Optional[str] = None):
        """Run the agent: load dataset, construct prompts, call API, and parse results.

        Args:
            project_dir: Base directory to resolve relative paths (defaults to agent.project_dir or current dir)
            save_results: If True, attempt to save per-chunk results to configured output paths

        Returns:
            pandas.DataFrame: concatenated results from API parsing (may be empty)
        """
        # Import here to avoid circular imports
        from .dataset import Dataset
        from .api_client import ApiClient

        import pandas as pd

        # Resolve project directory
        project_dir = project_dir or getattr(self, "project_dir", os.getcwd())

        # Input file
        input_file = getattr(self, "input_file", None)
        if not input_file:
            raise ValueError("Agent missing required 'input_file' parameter")

        # Build full input path
        full_input_path = input_file if os.path.isabs(input_file) else os.path.join(project_dir, input_file)

        # Prepare dataset
        dataset = Dataset(
            input_path=full_input_path,
            output_path=getattr(self, "output_path", os.path.join(project_dir, "results")),
            id_column=getattr(self, "dataset_id_column", "id"),
            columns=getattr(self, "dataset_columns", []),
            chunk_size=getattr(self, "chunk_size", 1),
        )

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

        # decide whether to save results based on presence of output_path
        should_save = bool(getattr(self, "output_path", None))

        # Iterate chunks and call API
        for chk, chv in getattr(dataset, "chunks", {}).items():
            # textual column default
            textual_col = getattr(self, "textual", "ITEM")
            data = list(chv[textual_col].values) if textual_col in chv.columns else [str(r) for r in chv.values]

            # Construct prompt
            dataset_name = getattr(self, "dataset_name", "dataset")
            prompt = api_client.construct_prompt(
                items=[inst],
                data=[f"<{dataset_name}>"] + [f"   {d}" for d in data] + [f"</{dataset_name}>"],
                resources=[f"<{rk}>\n    {rv}\n</{rk}>" for rk, rv in resources.items()],
            )

            # Call API
            response = api_client.get_response(prompt=prompt)

            # Parse response
            df = api_client.parse_response(
                response,
                format=getattr(self, "output_extension", "json"),
                json_node=getattr(self, "json_node", "results"),
                id_column=getattr(self, "dataset_id_column", "id"),
                data_columns=getattr(self, "result_columns", None),
            )

            # Save per-chunk if requested (determined by presence of output_path)
            if should_save:
                out_base = getattr(self, "output_path", os.path.join(project_dir, "results"))
                version = dt.datetime.now().strftime(getattr(self, "output_version", "%Y-%m-%d-%H-%M-%S"))
                out_dir = os.path.join(project_dir, out_base, version, getattr(self, "output_datasets", {}).get("result_files", "result_files"))
                os.makedirs(out_dir, exist_ok=True)
                chunk_name = os.path.basename(chk).split(".")[0] + "_" + getattr(self, "output_file", "results.csv")
                try:
                    df.to_csv(os.path.join(out_dir, chunk_name), index=False)
                except Exception:
                    pass

            results = pd.concat([results, df]) if not results.empty else df

        return results
