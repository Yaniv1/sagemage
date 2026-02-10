"""Sagemage - Systematic Agentic Modular Analytic Generative Engineering Framework."""

import re
import os
import sys


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


# Lazy imports: only import when module is actually used, not during build
def __getattr__(name):
    """Lazy import of submodules to avoid dependency issues during package build."""
    import importlib
    
    _lazy_modules = {
        "ApiClient": ("api_client", "ApiClient"),
        "Agent": ("core", "Agent"),
        "ParamSet": ("core", "ParamSet"),
        "Dataset": ("dataset", "Dataset"),
        "flatten_dataframe": ("utils", "flatten_dataframe"),
        "print_dict": ("utils", "print_dict"),
        "save_to_path": ("utils", "save_to_path"),
        "setattrs": ("utils", "setattrs"),
    }
    
    if name in _lazy_modules:
        module_name, attr_name = _lazy_modules[name]
        module = importlib.import_module(f".{module_name}", package=__name__)
        return getattr(module, attr_name)
    
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


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
