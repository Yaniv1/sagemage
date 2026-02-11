"""Examples package for sagemage.

Provides a simple dispatcher `demo()` which copies the example resources
into a local working directory and invokes the example's run function.
"""
from __future__ import annotations

import os
import sys
import importlib
import shutil
from pathlib import Path
import importlib.resources as pkg_resources
from typing import Optional


def _copy_example_files(example_name: str, dest: Path, reset: bool = False) -> Path:
    """Copy example package files into `dest`.

    Returns the destination path where files were copied.
    """
    # Locate the example package resources inside the installed package
    pkg = f"sagemage.examples.{example_name}"
    try:
        example_files = pkg_resources.files(f"sagemage.examples").joinpath(example_name)
    except Exception:
        raise ImportError(f"Example '{example_name}' not found in sagemage.examples")

    # Use a temporary extracted path if resources are in a zip
    with pkg_resources.as_file(example_files) as src_dir:
        src = Path(src_dir)
        dest.mkdir(parents=True, exist_ok=True)
        
        # Copy contents into dest (allow existing dirs)
        for f,g,h in os.walk(src):
            for i in h:
                src_file = os.path.join(f, i)
                rel_path = os.path.relpath(src_file, src)
                dest_file = os.path.join(dest, rel_path)
                dest_file_dir = os.path.dirname(dest_file)
                os.makedirs(dest_file_dir, exist_ok=True)
                if reset or not os.path.exists(dest_file):
                    shutil.copy2(src_file, dest_file)
        
    return dest


def demo(example_name: str = "foofoo", base_dir: Optional[str] = None, api_key_value: Optional[str] = None, reset: bool = False , verbose: bool = False):
    """Run a packaged example by name.

    - Copies the example files into `base_dir` (defaults to `./<example_name>`)
    - Imports the example package and invokes its `run()` entrypoint with
      arguments `(base_dir, api_key_value, reset)`.
    """
    # default base_dir to a subfolder under cwd to avoid polluting cwd
    if base_dir is None:
        base = Path.cwd() / example_name
    else:
        base = Path(base_dir)
    base = base.resolve()

    # _copy_example_files(example_name, base, reset=reset)

    module_name = f"sagemage.examples.{example_name}"
    mod = importlib.import_module(module_name)

    # Prefer a canonical `run` entrypoint
    if hasattr(mod, "run"):
        entry = getattr(mod, "run")
    else:
        # fallback to function with example name
        entry = getattr(mod, example_name, None)
    if entry is None:
        raise AttributeError(f"Example module {module_name} has no callable entrypoint (run or {example_name})")

    return entry(str(base), api_key_value, reset, verbose)


def list_examples():
    """Return a list of available example names included in the package."""
    examples = []
    root = pkg_resources.files("sagemage.examples")
    try:
        with pkg_resources.as_file(root) as root_path:
            for p in root_path.iterdir():
                if p.is_dir() and not p.name.startswith("_"):
                    examples.append(p.name)
    except Exception:
        # Fallback to pkgutil inspection
        import pkgutil
        import importlib

        pkg = importlib.import_module("sagemage.examples")
        for finder, name, ispkg in pkgutil.iter_modules(pkg.__path__):
            if ispkg and not name.startswith("_"):
                examples.append(name)

    return sorted(set(examples))

