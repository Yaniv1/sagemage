import os
import sys
import json
import shutil

def foofoo(base_dir: str = None, api_key_value: str = None, reset: bool = False):
    """Create a local foofoo example folder structure and run the agent.

    Args:
        base_dir: path to create the local example (defaults to cwd)
        api_key_value: optional API key string to write into settings/api_key.txt
        reset: if True, rebuild the local directory from scratch before running

    Returns:
        The parsed results returned by Agent.run()
    """
    # Lazy import to avoid heavy dependencies if not running
    from pathlib import Path

    base = Path(base_dir or os.getcwd()).resolve()
    settings_dir = base / 'settings'
    inputs_dir = base / 'inputs'
    instr_dir = base / 'instructions'
    py_dir = base / 'py'

    # If reset is True, clean up the config/input directories first
    if reset:
        print(f"Resetting foofoo directory at {base}...")
        for d in (settings_dir, inputs_dir, instr_dir):
            if d.exists():
                shutil.rmtree(d)
                print(f"  Removed {d.name}/")

    for d in (settings_dir, inputs_dir, instr_dir, py_dir):
        d.mkdir(parents=True, exist_ok=True)

    #write files (do not overwrite existing non-empty files, unless reset=True)
    settings_file = settings_dir / 'foofoo.json'
    api_key_file = settings_dir / 'api_key.txt'
    input_file = inputs_dir / 'foofoo_input.csv'
    instr_file = instr_dir / 'foofoo.txt'

    if not settings_file.exists():
        settings_file.write_text(json.dumps({
            "type": "GENERATIVE_ANALYZER",
            "run": True,
            "delta": True,
            "project_dir": "./",
            "readme": "readme.txt",
            "name": "foofoo_agent",
            "api_key": "settings/api_key.txt",
            "input_file": "inputs/foofoo_input.csv",
            "dataset_name": "foofoo",
            "dataset_id_column": "item_id",
            "dataset_columns": ["name"],
            "result_columns": ["item_id", "name", "type"],
            "instructions": "instructions/foofoo.txt",
            "resources": "",
            "output_path": "results/test",
            "output_datasets": {
                "input_files": "input_files",
                "result_files": "result_files",
                "prompts": "prompts",
                "analysis": "analysis"
            },
            "output_version": "%Y-%m-%d-%H-%M-%S",
            "output_file": "test.csv",
            "chunk_size": 2,
            "max_items": -1,
            "execute": True,
            "analyze": False,
            "verbose": True,
            "textual": "name",
            "output_extension": "json",
            "json_node": "results"
        }, indent=2), encoding='utf-8')

    if api_key_value is not None or not api_key_file.exists():
        api_key_file.write_text((api_key_value or "YOUR_API_KEY_HERE") + "\n", encoding='utf-8')

    if not input_file.exists():
        input_file.write_text("item_id,name\n1,cat\n2,dog\n3,apple\n4,tree\n5,astronaut\n6,Lord of the Rings\n7,car\n8,banana\n9,mug\n10,chair\n", encoding='utf-8')

    if not instr_file.exists():
        instr_file.write_text("classify each item, return the results as a json array with the following items: item_id, name, type\n", encoding='utf-8')

    # Run the agent - import directly since we're inside the sagemage package
    from sagemage.core import Agent

    cfg = json.loads(settings_file.read_text(encoding='utf-8'))
    agent = Agent(cfg)

    print(f"Running foofoo agent at {base}")
    results = agent.run(project_dir=str(base))

    print("Finished. Results preview:")
    print(results.head() if hasattr(results, 'head') else results)

    return results


if __name__ == '__main__':
    foofoo()
