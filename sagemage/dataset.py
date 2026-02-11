"""Dataset management for SAGEMAGE."""

import json
import os
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .utils import flatten_dataframe, save_to_path, setattrs


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
