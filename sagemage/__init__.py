"""Sagemage - Systematic Agentic Modular Analytic Generative Engineering Framework."""

__version__ = "0.1.1
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
