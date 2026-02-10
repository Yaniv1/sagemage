"""Sagemage - Systematic Agentic Modular Analytic Generative Engineering Framework."""

import re
import os


def _get_version():
    """Extract version from pyproject.toml"""
    pyproject_path = os.path.join(os.path.dirname(__file__), '..', 'pyproject.toml')
    try:
        with open(pyproject_path, 'r', encoding='utf-8') as f:
            content = f.read()
            match = re.search(r'^\s*version\s*=\s*["\']([^"\']+)["\']', content, re.MULTILINE)
            if match:
                return match.group(1)
    except Exception:
        pass
    return "0.0.0"


__version__ = _get_version()
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
