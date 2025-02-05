# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "datasets",
#     "lean-interact",
#     "rich",
# ]
# ///
"""
This module provides functions to type-check Lean formalizations
using sequential, parallel, and optimized batch methods.
"""

import concurrent.futures
import json
import os
import sys
from typing import Any

from datasets import load_dataset
from tqdm import tqdm

from lean_interact import AutoLeanServer, LeanREPLConfig

current_directory = os.path.dirname(os.path.abspath(__file__))
if current_directory not in sys.path:
    sys.path.insert(0, current_directory)

from utils import console, is_valid_lean

DEFAULT_TIMEOUT = 60


def type_check_sequential(dataset: list[dict[str, Any]], repl_config: LeanREPLConfig) -> list[bool]:
    """
    Type-checks each formalization sequentially.

    Args:
        dataset: A list of dictionaries with keys 'lean4_src_header' and 'lean4_formalization'.
        repl_config: Configuration for the Lean REPL.

    Returns:
        A list of booleans indicating if each formalization is valid.
    """
    server = AutoLeanServer(repl_config)
    successes = [False for _ in dataset]
    for i, row in enumerate(tqdm(dataset)):
        src_header = row["lean4_src_header"]
        formalization = row["lean4_formalization"]
        try:
            server_output = server.run_code(src_header + "\n" + formalization + " sorry", timeout=DEFAULT_TIMEOUT)
            successes[i] = is_valid_lean(server_output)
        except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError) as e:
            console.log(f"Error while type checking row {i}: {e}")
    return successes


def type_check_parallel(dataset: list[dict[str, Any]], repl_config: LeanREPLConfig) -> list[bool]:
    """
    Type-checks each formalization in parallel.

    Args:
        dataset: A list of dictionaries with keys 'lean4_src_header' and 'lean4_formalization'.
        repl_config: Configuration for the Lean REPL.

    Returns:
        A list of booleans indicating if each formalization is valid.
    """

    def process_row(row: dict[str, Any]) -> bool:
        server = AutoLeanServer(repl_config)
        src_header = row["lean4_src_header"]
        formalization = row["lean4_formalization"]
        try:
            server_output = server.run_code(src_header + "\n" + formalization + " sorry", timeout=DEFAULT_TIMEOUT)
            return is_valid_lean(server_output)
        except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError) as e:
            console.log(f"Error while type checking: {e}")
            return False

    with concurrent.futures.ThreadPoolExecutor() as executor:
        results = list(tqdm(executor.map(process_row, dataset), total=len(dataset)))
    return results


def type_check_parallel_optimized(dataset: list[dict[str, Any]], repl_config: LeanREPLConfig) -> list[bool]:
    """
    Optimized parallel type-checking by batching formalizations with a common source header.

    Args:
        dataset: A list of dictionaries with keys 'lean4_src_header' and 'lean4_formalization'.
        repl_config: Configuration for the Lean REPL.

    Returns:
        A list of booleans indicating if each formalization is valid.
    """
    # Group by common src header
    formalizations_idx = list(enumerate(dataset))
    formalizations_grouped: dict[str, list] = {}
    for idx, row in formalizations_idx:
        header = row["lean4_src_header"]
        formalizations_grouped.setdefault(header, []).append((idx, row["lean4_formalization"]))

    def type_check_group(src_header: str, group: list) -> list[int]:
        server = AutoLeanServer(repl_config)
        context_env = server.run_code(src_header, add_to_session_cache=True)["env"]
        valid_indices = []
        for idx, formalization in group:
            try:
                server_output = server.run_code(formalization + " sorry", env=context_env, timeout=DEFAULT_TIMEOUT)
                if is_valid_lean(server_output):
                    valid_indices.append(idx)
            except (TimeoutError, ConnectionAbortedError, json.JSONDecodeError) as e:
                console.log(f"Error in group with header '{src_header}': {e}")
        return valid_indices

    res: list[bool] = [False for _ in dataset]
    with concurrent.futures.ThreadPoolExecutor() as executor:
        for group in list(
            tqdm(
                executor.map(lambda x: type_check_group(*x), formalizations_grouped.items()),
                total=len(formalizations_grouped),
            )
        ):
            for idx in group:
                res[idx] = True

    return res


if __name__ == "__main__":
    proofnetsharp = load_dataset("PAug/ProofNetSharp", split="valid")
    config = LeanREPLConfig(lean_version="v4.15.0", require="mathlib")

    # Uncomment the desired method
    successes = type_check_sequential(proofnetsharp, config)
    # successes = type_check_parallel(proofnetsharp, config)
    # successes = type_check_parallel_optimized(proofnetsharp, config)

    assert len(successes) == len(proofnetsharp)

    if any(not success for success in successes):
        console.log("Failures:")
        for idx, success in enumerate(successes):
            if not success:
                console.log(proofnetsharp[idx])
    else:
        console.log("All formalizations are valid!")
