"""SAGEMAGE foofoo example - classification agent demo."""

import os
import sys
import json
import shutil
from pathlib import Path

from sagemage.core import AgentSet


def run(base_dir: str = None, api_key_value: str = None, reset: bool = False, verbose: bool = False):
    """Create a local foofoo example workspace and run the foofoo demo.

    Args:
        base_dir: path to create the local example (defaults to cwd)
        api_key_value: optional API key string to write into settings/api_key.txt
        reset: if True, rebuild the local directory from scratch before running

    Returns:
        The parsed results returned by the demo function
    """
    base = Path(base_dir or os.getcwd()).resolve()
    settings_dir = base / 'settings'
    inputs_dir = base / 'inputs'
    instr_dir = base / 'instructions'
    
    # if reset and os.path.isdir(base):
    #     shutil.rmtree(base)   

    for d in (settings_dir, inputs_dir, instr_dir):
        os.makedirs(d, exist_ok=True)
                
        for f,g,h in os.walk(Path(__file__).parent / d.name):
            for i in h:
                src_file = Path(f) / i
                dst_file = d / i
                if reset or not dst_file.exists():
                    shutil.copy(src_file, dst_file)
                    if verbose: print(f"Copied {src_file} to {dst_file}")
    
    # write files (do not overwrite existing non-empty files, unless reset=True)
    settings_file = settings_dir / 'foofoo.json'
    api_key_file = settings_dir / 'api_key.txt'

    # copy packaged mapping-style settings and about
    src_settings = Path(__file__).parent / 'settings' / 'foofoo.json'
    src_about = Path(__file__).parent / 'settings' / 'about.txt'
    
    # if not settings_file.exists():
    #     shutil.copy(src_settings, settings_file)
    # # copy about
    # if src_about.exists():
    #     shutil.copy(src_about, settings_dir / 'about.txt')

    if api_key_value is not None or not api_key_file.exists():
        api_key_file.write_text((api_key_value or "YOUR_API_KEY_HERE") + "\n", encoding='utf-8')

    if verbose: 
        for f,g,h in os.walk(base):
            for i in h:
                print(Path(f) / i)
    
    # run agents via AgentSet using the settings mapping
    aset = AgentSet(source=str(settings_file))
    if verbose: print(f"Running foofoo agents at {base}")
    results = aset.run_all(project_dir=str(base))

    if verbose: 
        print("Finished. Results preview:")
        for k, v in results.items():
            print(f"-- {k} --")
            try:
                print(v.head() if hasattr(v, 'head') else v)
            except Exception:
                print(str(v))

    return results


__all__ = ["run"]
