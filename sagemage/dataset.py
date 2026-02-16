"""Dataset management for SAGEMAGE."""

import json
import os
import random
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import datetime as dt


from .utils import flatten_dataframe, save_to_path, setattrs


# ==========================================================
# 🔐 SAFE DATAFRAME CONVERTER
# ==========================================================

class DataFrameConverter:
    """
    Securely applies transformation rules to a pandas DataFrame.

    Supports:
        - Column-level transformations
        - Row-level transformations
        - Full DataFrame expressions
        - Optional filtering
        - Safe support for datetime via `dt`, numpy via `np`, and pandas via `pd`
    """

    SAFE_GLOBALS = {
        "__builtins__": {},
        "np": np,
        "pd": pd,
        "dt": dt,
        "abs": abs,
        "min": min,
        "max": max,
        "sum": sum,
        "len": len,
        "round": round,
        "float": float,
        "int": int,
        "str": str,
    }

    def __init__(self, conversions: Optional[List[Dict]] = None, verbose: bool = False):
        self.conversions = conversions or []
        self.verbose = verbose

    def _safe_eval(self, expr: str, local_vars: Dict[str, Any]):
        """Safely evaluate an expression with restricted globals."""
        try:
            return eval(expr, self.SAFE_GLOBALS, local_vars)
        except Exception as e:
            raise ValueError(f"Conversion expression failed: {expr}\n{str(e)}")

    def apply(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply conversion list to DataFrame."""
        df = df.copy()

        for conv in self.conversions:
            target = conv.get("target")
            source = conv.get("source")
            op = conv.get("op")
            scope = conv.get("scope", "col")  # col | row | df
            filt = conv.get("filter")

            if not target or not op:
                continue

            if self.verbose:
                print(f"Applying conversion -> {target} | scope={scope}")

            # Optional filtering
            if filt:
                df = df[df[source].isin(filt)]

            if scope == "col":
                if source not in df.columns:
                    continue
                df[target] = df[source].apply(
                    lambda v: self._safe_eval(op, {"v": v, "df": df})
                )

            elif scope == "row":
                df[target] = df.apply(
                    lambda row: self._safe_eval(
                        op,
                        {
                            "row": row,
                            "df": df,
                        },
                    ),
                    axis=1,
                )

            elif scope == "df":
                df = self._safe_eval(op, {"df": df})

        return df


class Dataset:
    """Manages loading, processing, and saving datasets."""

    def __init__(
        self,
        name: str = "dataset",
        input_path: str = "",
        input_data: Optional[pd.DataFrame] = None,
        output_path: str = "",
        id_column: str = "id",
        columns: Optional[List[str]] = None,
        **kwargs,
    ):
        """
        Initialize Dataset.

        Args:
            input_path: Path to input data file
            output_path: Path to save processed data
            id_column: Name of ID column
            columns: Specific columns to process
            **kwargs: Additional configuration
                - capitalize: Capitalize column names (default: True)
                - sheet_name: Excel sheet to load (default: 0)
                - root: JSON root node (default: "data")
                - textual: Column name for textual representation (default: "ITEM")
                - output_format: Output format "json" or other (default: "json")
                - group_columns: Columns to group by (default: [])
                - chunk_size: Size of chunks (default: 1)
        """
        if columns is None:
            columns = []

        self.defaults = setattrs(
            capitalize=True,
            sheet_name=0,
            root="data",
            textual="ITEM",
            output_format="json",
            group_columns=[],
            chunk_size=1,
            verbose=False
        )
        self.name = name
        self.input_path = input_path
        self.output_path = output_path
        self.id_column = id_column
        self.columns = columns

        for k, v in self.defaults.items():
            setattr(self, k, kwargs.get(k, v))
        
        self.df = input_data.copy() if input_data is not None else None
                   
        if type(self.input_path) == str and os.path.isfile(self.input_path) or self.df is not None:            
            self.load()       

        if self.output_path:
            if self.verbose: print(f"Saving dataset '{self.name}' groups and chunks  to {self.output_path}...")
            self.save()

    def load(self, **kwargs) -> None:
        """Load the input file into a DataFrame."""
        
        file = self.input_path
        columns = self.columns
        id_column = self.id_column
        
        
        if self.df is not None:
            df = self.df.copy()        
        else:
        
            if file == "":
                file = getattr(self, "file", "")
            if id_column == "":
                id_column = self.id_column

            sheet_name = self.sheet_name
            root = self.root

            # Load based on file format
            file_ext = file.split(".")[-1].lower()
            if file_ext == "csv":
                df = pd.read_csv(file)
            elif file_ext in ["xls", "xlsx"]:
                df = pd.read_excel(file, sheet_name=sheet_name)
            elif file_ext == "json":
                with open(file, "r") as json_file:
                    items_dict = json.load(json_file)
                    if root is not None:
                        items_dict = items_dict.get(root)
                    df = pd.DataFrame(items_dict)
            else:
                raise ValueError(f"Unsupported file format: {file_ext}")
        
        if df is not None:
            if self.verbose: print(f"Loaded data from {file}:\n{df.head()}")

        item_ids = []
        items = []

        # Determine columns
        if columns == []:
            columns = list(df.columns)
            if id_column in columns:
                columns.remove(id_column)
            self.columns = columns

        # Extract IDs
        if id_column != "" and id_column in list(df.columns):
            item_ids = list(df[id_column].values)
        else:
            item_ids = list(df.index)

        # Create textual representations
        items = [", ".join([str(row[c]) for c in columns]) for _, row in df.iterrows()]
        items = [str(item_ids[x]) + ". " + items[x] for x in range(len(items))]

        df[self.textual] = items

        if self.capitalize:
            df.columns = [c.upper() for c in df.columns]

        self.df = df
        self.items = items

        self.group()
        if self.verbose: print(f"Grouped data:\n{list(self.groups.keys())}")
        self.chunk()
        if self.verbose: print(f"Chunked data:\n{list(self.chunks.keys())}")

    def group(self) -> None:
        """Group DataFrame by specified columns."""
        groups = {}
        group_columns = self.group_columns
        group_columns = [c for c in group_columns if c in self.df.columns]

        if group_columns != []:
            df_groups = self.df.groupby(group_columns).groups

            for g, gids in df_groups.items():
                group_key = "/".join(
                    [f"{group_columns[c]}={g[c]}" for c in range(len(group_columns))]
                )
                group_path = f"{self.output_path}/{self.name.replace('.', '_')}/{group_key.replace('/', '_')}.{self.output_format}"
                group_data = self.df.iloc[gids]

                groups.update({group_path: group_data})

        else:
            group_path = f"{self.output_path}/{self.name.replace('.', '_')}.{self.output_format}"
            groups.update({group_path: self.df})

        self.groups = groups

    def chunk(self) -> None:
        """Chunk grouped data into smaller pieces."""
        ds_chunks = {}

        for g, gdf in self.groups.items():
            n_chunks = max(1,int(np.ceil(len(gdf) / self.chunk_size))) # Ensure at least 1 chunk
            chunks = [ch for ch in np.array_split(gdf, n_chunks)]
            for i, ch in enumerate(chunks):
                chunk_path = g.split("/")

                chunk_path[-1] = f"{'.'.join(chunk_path[-1].split('.')[:-1])}_chunk_{i:05d}.{chunk_path[-1].split('.')[-1]}"
                chunk_path = "/".join(chunk_path)

                ds_chunks.update({chunk_path: ch})

        self.chunks = ds_chunks

    def get_chunks(self, max_chunks: Any = None, sample: Any = False) -> List:
        """Return dataset chunks after applying optional max_chunks/sample restrictions.

        Rules:
        - max_chunks:
          - None/False/0/-1 => no restriction
          - number => keep first N chunks
          - [a, b] => keep chunks in inclusive index range [a, b]
        - sample (applied after max_chunks):
          - False/None/0 => no restriction
          - True/1 => randomly choose one chunk
          - number => randomly choose N chunks
        """
        chunks_dict = getattr(self, "chunks", {})
        chunks_list = list(chunks_dict.items()) if chunks_dict else [("full", self.df)]
        filtered = list(chunks_list)

        # 1) max_chunks restriction
        if max_chunks not in [None, False, 0, -1]:
            if isinstance(max_chunks, (list, tuple)) and len(max_chunks) == 2:
                try:
                    start = int(max_chunks[0])
                    end = int(max_chunks[1])
                    lo, hi = sorted([start, end])
                    lo = max(0, lo)
                    hi = max(0, hi)
                    filtered = filtered[lo:hi + 1]
                except Exception:
                    if self.verbose:
                        print(f"Invalid max_chunks range ignored: {max_chunks}")
            elif isinstance(max_chunks, (int, float)) and not isinstance(max_chunks, bool):
                n = int(max_chunks)
                if n > 0:
                    filtered = filtered[:n]

        # 2) sample restriction (drawn after max_chunks)
        sample_n = None
        if sample is True:
            sample_n = 1
        elif isinstance(sample, (int, float)) and not isinstance(sample, bool):
            sample_i = int(sample)
            if sample_i > 0:
                sample_n = sample_i

        if sample_n is not None and len(filtered) > 0:
            sample_n = min(sample_n, len(filtered))
            filtered = random.sample(filtered, sample_n)

        return filtered

    def save(self) -> None:
        """Save grouped and chunked data to files."""
        if self.verbose: print(self.groups.keys())
        for g, gdf in self.groups.items():
            data = gdf.copy()
            if self.output_format == "json":
                data = {
                    "data": [{c: row[c] for c in gdf.columns} for _, row in gdf.iterrows()]
                }
            save_to_path(data, g, append=False)

        if self.verbose: print(self.chunks.keys())
        for c, cdf in self.chunks.items():
            data = cdf.copy()
            if self.output_format == "json":
                data = {
                    "data": [{c: row[c] for c in cdf.columns} for _, row in cdf.iterrows()]
                }
            save_to_path(data, c, append=False)

    def flatten(
        self, combination_id: str = "combination_id", cols_to_flatten: Optional[List[str]] = None
    ) -> None:
        """
        Flatten list columns in grouped and chunked data.

        Args:
            combination_id: Column name for combination ID
            cols_to_flatten: Columns to flatten
        """
        if cols_to_flatten is None:
            cols_to_flatten = []

        for g, gdf in self.groups.items():
            self.groups[g] = flatten_dataframe(
                gdf,
                combination_id=getattr(self, "combination_id", "combination_id"),
                cols_to_flatten=cols_to_flatten,
            )

        for c, cdf in self.chunks.items():
            self.chunks[c] = flatten_dataframe(
                cdf,
                combination_id=getattr(self, "combination_id", "combination_id"),
                cols_to_flatten=cols_to_flatten,
            )
