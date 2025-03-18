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

import json
from typing import Any

from datasets import load_dataset
from rich.console import Console
from tqdm import tqdm

from lean_interact import AutoLeanServer, LeanREPLConfig
from lean_interact.utils import is_valid_lean

console = Console()
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
            console.log(f"Error while type checking entry {i}: {e}")
    return successes


def type_check_sequential_optimized(dataset: list[dict[str, Any]], repl_config: LeanREPLConfig) -> list[bool]:
    """
    Optimized type-checking by batching formalizations with a common context.

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
    for idx, group in enumerate(tqdm(formalizations_grouped.items())):
        src_header, group = group
        valid_indices = type_check_group(src_header, group)
        for idx in valid_indices:
            res[idx] = True
    return res


if __name__ == "__main__":
    proofnetsharp = load_dataset("PAug/ProofNetSharp", split="valid")
    config = LeanREPLConfig(lean_version="v4.15.0", require="mathlib")

    # successes = type_check_sequential(proofnetsharp, config)
    successes = type_check_sequential_optimized(proofnetsharp, config)

    assert len(successes) == len(proofnetsharp)

    if any(not success for success in successes):
        console.log("Failures:")
        for idx, success in enumerate(successes):
            if not success:
                console.log(proofnetsharp[idx])
    else:
        console.log("All formalizations are well-typed!")
