"""Sagemage - Systematic Agentic Modular Analytic Generative Engineering Framework."""

import re
import os


def _get_version(default="0.0.0"):
    """Extract version from version.txt (single source of truth)."""
    version_file = os.path.join(os.path.dirname(__file__), 'version.txt')
    try:
        with open(version_file, 'r', encoding='utf-8') as f:
            return f.read().strip()
    except (FileNotFoundError, IOError):
        # Fallback if version.txt not found (shouldn't happen)
        return default


__version__ = _get_version("0.1.1")
__author__ = "Yaniv Mordecai"
__email__ = "modelanalyzer@gmail.com"
__license__ = "MIT"

# Import core classes
from .api_client import ApiClient
from .core import Agent, ParamSet
from .dataset import Dataset
from .utils import flatten_dataframe, print_dict, save_to_path, setattrs

__all__ = [
    "ParamSet",
    "Agent",
    "Dataset",
    "ApiClient",
    "print_dict",
    "save_to_path",
    "setattrs",
    "flatten_dataframe",
]
