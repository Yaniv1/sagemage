"""Core SAGEMAGE classes for parameter management and agents."""

import json
import os
from typing import Any, Dict, Optional


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
