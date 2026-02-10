import os
import os
import sys
import json
import shutil
from pathlib import Path

from sagemage.core import AgentSet
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
    py_dir = base / 'py'

    # If reset is True, clean up the config/input directories first
    if reset:
        for d in (settings_dir, inputs_dir, instr_dir):
            if d.exists():
                shutil.rmtree(d)

    for d in (settings_dir, inputs_dir, instr_dir, py_dir):
        d.mkdir(parents=True, exist_ok=True)

    # write files (do not overwrite existing non-empty files, unless reset=True)
    settings_file = settings_dir / 'foofoo.json'
    api_key_file = settings_dir / 'api_key.txt'
    input_file = inputs_dir / 'foofoo_input.csv'
    instr_file = instr_dir / 'foofoo.txt'

    # copy packaged mapping-style settings and about
    src_settings = Path(__file__).parent / 'settings' / 'foofoo.json'
    src_about = Path(__file__).parent / 'settings' / 'about.txt'
    if not settings_file.exists():
        shutil.copy(src_settings, settings_file)
    # copy about
    if src_about.exists():
        shutil.copy(src_about, settings_dir / 'about.txt')

    if api_key_value is not None or not api_key_file.exists():
        api_key_file.write_text((api_key_value or "YOUR_API_KEY_HERE") + "\n", encoding='utf-8')

    if not input_file.exists():
        input_file.write_text("item_id,name\n1,cat\n2,dog\n3,apple\n4,tree\n5,astronaut\n6,ring\n7,car\n8,banana\n9,mug\n10,chair\n", encoding='utf-8')

    if not instr_file.exists():
        instr_file.write_text("classify each item, return the results as a json array with the following items: item_id, name, type\n", encoding='utf-8')

    # copy analyzer instruction if present in package
    pkg_analyze_inst = Path(__file__).parent / 'instructions' / 'analyze_prompt.txt'
    if pkg_analyze_inst.exists():
        shutil.copy(pkg_analyze_inst, instr_dir / 'analyze_prompt.txt')

    # run agents via AgentSet using the settings mapping
    aset = AgentSet(source=str(settings_file))
    print(f"Running foofoo agents at {base}")
    results = aset.run_all(project_dir=str(base))

    print("Finished. Results preview:")
    for k, v in results.items():
        print(f"-- {k} --")
        try:
            print(v.head() if hasattr(v, 'head') else v)
        except Exception:
            print(str(v))

    return results


__all__ = ["run"]
"""SAGEMAGE foofoo example - classification agent demo."""
