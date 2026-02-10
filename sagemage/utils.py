"""Utility functions for SAGEMAGE."""

import itertools
import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd


def print_dict(my_dict: Dict[str, Any]) -> None:
    """
    Pretty print a dictionary.

    Args:
        my_dict: Dictionary to print
    """
    for k, v in my_dict.items():
        try:
            if isinstance(v, str) or len(v) <= 1:
                print(k, "\t\t=\t\t", v)
            else:
                print(k, "\t\t=\n", v)
        except (TypeError, AttributeError):
            print(k, "\t\t=\t\t", v)


def save_to_path(obj: Any, path: str, append: bool = False) -> bool:
    """
    Save an object to a file.

    Supports JSON, CSV, XLSX/XLS, and TXT formats.

    Args:
        obj: Object to save (dict, DataFrame, or string)
        path: Path to save file to
        append: If True, append to file (for CSV and TXT)

    Returns:
        True if successful, False otherwise. Returns exception object on error.
    """
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    if obj is not None:
        fmode = "w"
        if append:
            if os.path.isfile(path):
                fmode = "a"

        try:
            path_lower = path.lower()
            if path_lower.endswith(".json"):
                with open(path, mode=fmode) as f:
                    json.dump(obj, f, indent=4, default=str)
            elif path_lower.endswith(".csv"):
                obj.to_csv(path, mode=fmode, index=False, encoding="utf-8")
            elif path_lower.endswith((".xlsx", ".xls")):
                obj.to_excel(path)
            elif path_lower.endswith(".txt"):
                with open(path, mode=fmode) as f:
                    f.write(obj)

            return os.path.isfile(path)

        except Exception as e:
            print(f"Error saving to {path}: {e}")
            return e

    return False


def setattrs(r: Optional[Dict[str, Any]] = None, **kwargs) -> Dict[str, Any]:
    """
    Update a dictionary with keyword arguments.

    Args:
        r: Dictionary to update (default: empty dict)
        **kwargs: Key-value pairs to add

    Returns:
        Updated dictionary
    """
    if r is None:
        r = {}
    for k, v in kwargs.items():
        r.update({k: v})
    return r


def flatten_dataframe(
    df: pd.DataFrame,
    combination_id: str = "combination_id",
    cols_to_flatten: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Flatten a DataFrame by expanding list columns.

    Args:
        df: DataFrame to flatten
        combination_id: Column name for combination ID
        cols_to_flatten: List of column names to flatten. If empty/None, flatten all list columns.

    Returns:
        Flattened DataFrame
    """
    if cols_to_flatten is None:
        cols_to_flatten = []

    # Identify list columns
    list_columns = [
        col for col in df.columns if df[col].apply(lambda x: isinstance(x, list)).any()
    ]
    list_columns = [
        col for col in list_columns if col in cols_to_flatten or cols_to_flatten == []
    ]
    non_list_columns = [col for col in df.columns if col not in list_columns]

    flattened_rows = []

    for _, row in df.iterrows():
        list_values = [
            row[col] if isinstance(row[col], list) else [row[col]] for col in list_columns
        ]
        product_list = list(itertools.product(*list_values))

        for cid in range(len(product_list)):
            combination = product_list[cid]
            new_row = {col: row[col] for col in non_list_columns}
            if combination_id:
                new_row.update({combination_id: cid})
            new_row.update({col: val for col, val in zip(list_columns, combination)})

            flattened_rows.append(new_row)

    flattened_df = pd.DataFrame(flattened_rows)
    return flattened_df
