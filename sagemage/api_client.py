"""API client for LLM interactions in SAGEMAGE."""

import datetime as dt
import json
import re
from typing import Any, Dict, List, Optional

import pandas as pd

from .utils import flatten_dataframe, setattrs


class ApiClient:
    """Client for interacting with LLM APIs like OpenAI."""

    def __init__(
        self,
        client=None,
        model: str = "gpt-4",
        api_key: str = "",
        **kwargs,
    ):
        """
        Initialize ApiClient.

        Args:
            client: API client class (default: openai.OpenAI)
            model: Model name (default: "gpt-4")
            api_key: API key for authentication
            **kwargs: Additional configuration
                - temperature: Sampling temperature (default: 1)
                - max_tokens: Maximum tokens in response (default: 4096)
                - top_p: Nucleus sampling parameter (default: 1)
                - frequency_penalty: Frequency penalty (default: 0)
                - presence_penalty: Presence penalty (default: 0)
        """
        self.defaults = setattrs(
            temperature=1,
            max_tokens=4096,
            top_p=1,
            frequency_penalty=0,
            presence_penalty=0,
        )

        self.api_key = api_key
        self.model = model

        # Initialize client if provided
        if client is not None:
            try:
                self.client = client(api_key=self.api_key)
            except Exception as e:
                print(f"Error initializing API client: {e}")
                self.client = None
        else:
            self.client = None

        for k, v in self.defaults.items():
            setattr(self, k, kwargs.get(k, v))

    def construct_prompt(
        self,
        items: Optional[List[str]] = None,
        data: Optional[List[str]] = None,
        resources: Optional[List[str]] = None,
        text: Optional[List[str]] = None,
        mapping: Optional[Dict[str, str]] = None,
    ) -> str:
        """
        Create a textual prompt from items and data.

        Args:
            items: List of prompt items/sections (may include placeholders)
            data: Data to insert into {$DATA$} placeholder
            resources: Resources to insert into {$RESOURCES$} placeholder
            text: Text to insert into {$TEXT$} placeholder
            mapping: Custom mapping of placeholders to variable names

        Returns:
            Constructed prompt string
        """
        if items is None:
            items = ["Analyze my data", "\n", "{$DATA$}"]
        if data is None:
            data = []
        if resources is None:
            resources = []
        if text is None:
            text = []
        if mapping is None:
            mapping = {"{$DATA$}": "data", "{$RESOURCES$}": "resources", "{$TEXT$}": "text"}

        r = ""
        for a in items:
            if isinstance(a, list):
                a = "\n".join(a)
            r += a

        for mk, mv in mapping.items():
            lmv = "\n".join(locals().get(mv, []))
            r = r.replace(mk, lmv)

        return r

    def get_response(self, prompt: str, **overrides) -> Dict[str, Any]:
        """
        Get response from LLM API.

        Args:
            prompt: Prompt to send to the model
            **overrides: Parameter overrides

        Returns:
            Dictionary with timestamp, params, and response
        """
        if self.client is None:
            return {
                "timestamp": dt.datetime.now(),
                "params": {},
                "response": "Error: API client not initialized",
            }

        try:
            # Import openai here to check type
            import openai

            if isinstance(self.client, openai.OpenAI):
                model_params_map = {
                    "DEFAULT": [
                        "model",
                        "temperature",
                        "max_tokens",
                        "top_p",
                        "frequency_penalty",
                        "presence_penalty",
                    ],
                    "gpt-4": {},
                    "gpt-4-turbo": {},
                    "gpt-3.5-turbo": {},
                }

                cli_args = {}

                # Map model-specific parameters
                for pk, pv in model_params_map.get(self.model, {}).items():
                    cli_args.update({pk: overrides.get(pv, overrides.get(pk, getattr(self, pv)))})

                # Add default parameters
                for pk in model_params_map["DEFAULT"]:
                    if pk not in list(model_params_map.get(self.model, {}).values()):
                        cli_args.update({pk: overrides.get(pk, getattr(self, pk))})

                messages = [{"role": "user", "content": prompt}]
                cli_args = setattrs(cli_args, messages=messages)

                response = self.client.chat.completions.create(**cli_args)
            else:
                cli_args = {"prompt": prompt}
                response = self.client.complete(**cli_args)

        except Exception as response_error:
            print(f"Response error: {response_error=}")
            response = f"Error: {response_error}"

        return {"timestamp": dt.datetime.now(), "params": cli_args, "response": response}

    def parse_response(
        self,
        response: Dict[str, Any],
        format: str = "json",
        json_node: str = "results",
        id_column: Optional[str] = None,
        data_columns: Optional[List[str]] = None,
        flatten: bool = False,
        flat_combination_id: str = "combination_id",
        cols_to_flatten: Optional[List[str]] = None,
        input_path: str = "",
        input_merge_mode: str = "inner",
        output_path: str = "",
    ) -> pd.DataFrame:
        """
        Parse LLM response into DataFrame.

        Args:
            response: Response dictionary from get_response()
            format: Response format ("json" or "csv")
            json_node: JSON node containing results
            id_column: Column name for IDs
            data_columns: Names of data columns
            flatten: Whether to flatten list columns
            flat_combination_id: Column name for combination ID
            cols_to_flatten: Specific columns to flatten
            input_path: Path to input file for merging
            input_merge_mode: Merge mode ("inner", "outer", "left", "right")
            output_path: Path to save output

        Returns:
            Parsed DataFrame
        """
        if cols_to_flatten is None:
            cols_to_flatten = []
        if data_columns is None:
            data_columns = []

        sdf = pd.DataFrame()

        try:
            if format.lower() == "csv":
                results = response.get("response").choices[0].message.content.split("\n")
                s_array = []
                for r in results:
                    s = r.split(". ")
                    item_id = s[0]
                    data = s[1].split(",")
                    s_array.append([item_id] + data)
                columns = [id_column] + data_columns
                sdf = pd.DataFrame(s_array, columns=columns)

            elif format.lower() == "json":
                results = response.get("response").choices[0].message.content
                results = re.sub(
                    r"^```json\s*|\s*```$",
                    "",
                    results.strip(),
                    flags=re.DOTALL | re.IGNORECASE,
                )
                print('Results:\n',results)

                try:
                    rdf = pd.DataFrame(json.loads(results)[json_node])
                    if id_column:
                        rdf[id_column] = rdf[id_column].apply(str)
                    sdf = pd.concat([sdf, rdf])
                except Exception as err_parse_response:
                    print(f"Error parsing response: {err_parse_response=}")
            
            elif format.lower() == "txt":
                sdf = pd.DataFrame({"response": [response.get("response").choices[0].message.content]})

        except Exception as e:
            print(f"Error in parse_response: {e}")
            return sdf

        df = sdf.copy()

        # Merge with input if provided
        if input_path:
            try:
                input_df = pd.read_csv(input_path)
                input_df[id_column] = input_df[id_column].apply(str)
                df = input_df.merge(sdf, how=input_merge_mode, on=[id_column]).fillna("")
            except Exception as e:
                print(f"Error merging with input: {e}")

        # Flatten if requested
        if flatten:
            df = flatten_dataframe(df, flat_combination_id, cols_to_flatten)

        # Save to file if output path provided
        if output_path:
            try:
                df.to_csv(output_path, index=False)
            except Exception as e:
                print(f"Error saving output: {e}")

        return df
