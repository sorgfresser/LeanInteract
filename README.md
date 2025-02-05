# LeanInteract

**LeanInteract** is a Python package designed to seamlessly interact with Lean 4 through the [Lean REPL](https://github.com/leanprover-community/repl).

> [!NOTE]
> This tool is still experimental and has been primarily tested on Linux. Compatibility with macOS is not guaranteed. Windows is not supported at the moment. Please report any issues you encounter.

## Key Features

- **🚀 Ease of Use**: LeanInteract abstracts the complexities of Lean setup and interaction, enabling quick experimentation.
- **🔗 Interactivity**: Execute Lean code, files, tactics, and (partial) proofs directly from Python. Easily iterate through environment and proof states.
- **🔄 Up-to-date**: LeanInteract is a lightweight wrapper around [Lean REPL](https://github.com/leanprover-community/repl), ensuring it stays current with the latest Lean versions and features.
- **🔧 Compatibility**: Supports all Lean versions between `v4.7.0-rc1` and `v4.16.0-rc2`.
  - Ensures compatibility with various Lean projects and machine learning benchmarks.
  - Need older versions? Open an issue [here](https://github.com/augustepoiroux/repl).
- **📥 Automatic Setup**: Automatically downloads and builds Lean REPL versions for you. Versions are cached for fast reuse.
- **📦 Temporary Projects**: Easily instantiate temporary Lean environments with dependencies.
  - Useful for experimenting and interacting with benchmarks like [ProofNet#](https://huggingface.co/datasets/PAug/ProofNetSharp) and [MiniF2F](https://github.com/yangky11/miniF2F-lean4) without manual setup.

## Similar tools

We recommend checking out these tools:

- **[PyPantograph](https://github.com/lenianiva/PyPantograph)**: Based on Pantograph, offering more options for proof interactions than Lean REPL.
- **[LeanDojo](https://github.com/lean-dojo/LeanDojo)**: Parses Lean projects to create datasets and interact with theorems to prove them.
- **[leanclient](https://github.com/oOo0oOo/leanclient)**: Interact with the Lean LSP server.

LeanInteract is inspired by:

- **[pylean](https://github.com/zhangir-azerbayev/repl)**
- **[lean4_jupyter](https://github.com/utensil/lean4_jupyter)**

## Installation and Setup

Requirements:

- Python >= 3.10
- git
- [Lean 4](https://leanprover-community.github.io/get_started.html)

If the requirements are met, install the LeanInteract package:

```bash
pip install lean-interact
```

## Script examples

In the `examples` directory, you will find various scripts demonstrating how to use LeanInteract. We recommend [uv](https://github.com/astral-sh/uv) to run these scripts using `uv run <script>`.

- `type_check.py`: type check formalizations from the [ProofNet#](https://huggingface.co/datasets/PAug/ProofNetSharp) benchmark.
- `beq_plus.py`: run [BEq+](https://arxiv.org/abs/2406.07222) metric on the ProofNetVerif benchmark.

Soon to be added:

- `proof_generation.py`: use [DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152) and other proof generators to prove theorems from the [MiniF2F](https://github.com/yangky11/miniF2F-lean4) benchmark.
- `improving_autoformalization_using_type_checking.py`: an implementation of the sampling method used in [Improving Autoformalization using Type Checking](https://arxiv.org/abs/2406.07222).

## Usage

### Default Lean version (latest available)

```python
from lean_interact import LeanREPLConfig, LeanServer

config = LeanREPLConfig() # download and build Lean REPL
server = LeanServer(config) # start Lean REPL
server.run_code("theorem ex (n : Nat) : n = 5 → n = 5 := sorry")
```

<details>
<summary>Output</summary>

```json
{"sorries": [{"proofState": 0,
   "pos": {"line": 1, "column": 40},
   "goal": "n : Nat\n⊢ n = 5 → n = 5",
   "endPos": {"line": 1, "column": 45}}],
 "messages": [{"severity": "warning",
   "pos": {"line": 1, "column": 8},
   "endPos": {"line": 1, "column": 10},
   "data": "declaration uses 'sorry'"}],
 "env": 0}
```

</details>

You can then iterate on the proof state by executing tactics:

```python
server.run_tactic("intro h", proof_state=0)
```

<details>
<summary>Output</summary>

```json
{"proofState": 1, "goals": ["n : Nat\nh : n = 5\n⊢ n = 5"]}
```

</details>

```python
server.run_tactic("exact h", proof_state=1)
```

<details>
<summary>Output</summary>

```json
{"proofState": 2, "goals": []}
```

</details>

Or by running the entire proof:

```python
server.run_proof("intro h\nexact h", proof_state=0)
```

<details>
<summary>Output</summary>

```json
{"proofState": 3, "goals": []}
```

</details>

You can also iterate on the environment:

```python
server.run_code("theorem ex2 (x : Nat) : x = 5 → x = 5 := by\n  exact ex x", env=0)
```

<details>
<summary>Output</summary>

```json
{"env": 1}
```

</details>

> [!NOTE]
> The initial invocation of `LeanREPLConfig` might take some time as it downloads and builds Lean REPL. Future executions with identical parameters will be significantly quicker due to caching.

### Using a specific Lean version

```python
config = LeanREPLConfig(lean_version="v4.7.0")
```

### Using an existing Lean project directory

```python
config = LeanREPLConfig(project_dir="path/to/your/project")
```

You can then use `run_code`, `run_tactic` and `run_file` as usual:

```python
server = LeanServer(config)
server.run_file("file.lean")
```

> [!IMPORTANT]
> Ensure the project in `project_dir` has been *successfully* built with `lake build` before using the REPL.

### Using a temporary project with dependencies

```python
config = LeanREPLConfig(lean_version="v4.7.0", require=[LeanRequire(
    name="mathlib",
    git="https://github.com/leanprover-community/mathlib4.git",
    rev="v4.7.0"
)])
```

Mathlib being a frequent requirement, a shortcut is available:

```python
config = LeanREPLConfig(lean_version="v4.7.0", require="mathlib")
```

You can then use Mathlib:

```python
server = LeanServer(config_readme_mathlib)
server.run_code("""import Mathlib
theorem ex_mathlib (x : ℝ) (y : ℚ) :\n  ( Irrational x ) -> Irrational ( x + y ) := sorry""")
```

<details>
<summary>Output</summary>

```json
{"sorries": [{"proofState": 0,
   "pos": {"line": 4, "column": 26},
   "goal": "x : ℝ\ny : ℚ\n⊢ Irrational (x + ↑y)",
   "endPos": {"line": 4, "column": 31}}],
 "messages": [{"severity": "warning",
   "pos": {"line": 3, "column": 8},
   "endPos": {"line": 3, "column": 18},
   "data": "declaration uses 'sorry'"}],
 "env": 0}
```

</details>

> [!NOTE]
>
> - Mathlib is a large library and may take some time to download and build.
> - Mathlib, and other libraries, are not available for all Lean versions. An error will be raised if the Lean version you are using does not support Mathlib. You can always specify a different version with the `lean_version` parameter if you know a compatible version.
> - A separate cache is used for each unique set of dependencies.

## Advanced options

### LeanServer

Two versions of Lean servers are available:

- **`LeanServer`**: A wrapper around Lean REPL. Interact with it using `run_code`, `run_file`, and `run_tactic` methods.
- **`AutoLeanServer`**: An experimental subclass of `LeanServer` monitoring memory usage to limit *out of memory* crashes in multiprocessing contexts. Additionally, since Lean REPL retains all environment and proof states,  `AutoLeanServer` regularly clears the REPL's memory to prevent crashes. Use the `add_to_session_cache` attribute in various methods to prevent selected environment/proof states to be cleared.

> [!TIP]
>
> - To run multiple requests in parallel, we recommend using multiprocessing with one `AutoLeanServer` instance per process.
> - Make sure to instantiate `LeanREPLConfig` before starting the processes to avoid conflicts during Lean REPL's download and build.
> - While `AutoLeanServer` can help prevent crashes, it is not a complete solution. If you encounter crashes, consider reducing the number of parallel processes or increasing the memory available to your system.

### Helper Methods and Variables

- `clear_cache()`: Removes all Lean REPL versions and temporary projects in the package cache. This can help resolve some issues. If it does, please open an issue.

### Custom Lean REPL

To use a forked Lean REPL project, specify the git repository using the `repl_git` parameter in the `LeanREPLConfig`. Your fork should have a similar format to <https://github.com/augustepoiroux/repl>. For assistance, feel free to contact [us](mailto:auguste.poiroux@epfl.ch).
