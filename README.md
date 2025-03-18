# LeanInteract

**LeanInteract** is a Python package designed to seamlessly interact with Lean 4 through the [Lean REPL](https://github.com/leanprover-community/repl).

## Key Features

- **🔗 Interactivity**: Execute Lean code and files directly from Python, and iterate on environment states.
- **🚀 Ease of Use**: LeanInteract abstracts the complexities of Lean setup and interaction, enabling quick experimentation.
  - Automatically downloads and builds Lean REPL versions for you. Versions are cached for fast reuse.
- **🔧 Compatibility**: Supports all Lean versions between `v4.7.0-rc1` and `v4.18.0-rc1`.
  - Ensures compatibility with various Lean projects and machine learning benchmarks.
- **📦 Temporary Projects**: Easily instantiate temporary Lean environments with dependencies.
  - Useful for experimenting and interacting with benchmarks depending on [Mathlib](https://github.com/leanprover-community/mathlib4) like [ProofNet#](https://huggingface.co/datasets/PAug/ProofNetSharp) and [MiniF2F](https://github.com/yangky11/miniF2F-lean4) without manually setting up a Lean project.

## Installation and Setup

You can install the LeanInteract package using the following command:

```bash
pip install lean-interact
```

Requirements:

- Python >= 3.10
- git
- [Lean 4](https://leanprover-community.github.io/get_started.html)
  - Tip: the `install-lean` command can install Lean 4 for you after installing LeanInteract.

> [!NOTE]
> This tool is still experimental and has been primarily tested on Linux. Compatibility with macOS is not guaranteed. For Windows, use WSL.
> Please report any issues you encounter.

## Script examples

In the `examples` directory, you will find a few scripts demonstrating how to use LeanInteract. We recommend [uv](https://github.com/astral-sh/uv) to run these scripts (`uv run <script>.py`).

- `beq_plus.py`: run the autoformalization [BEq+](https://arxiv.org/abs/2406.07222) metric on the [ProofNetVerif](https://huggingface.co/datasets/PAug/ProofNetVerif) benchmark.
- `type_check.py`: optimize type checking of formal statements using environment states.

Soon to be added:

- `proof_generation_and_autoformalization.py`: use [DeepSeek-Prover-V1.5](https://arxiv.org/abs/2408.08152), [Goedel-Prover](https://goedel-lm.github.io/), and other models to prove theorems from the [MiniF2F](https://github.com/yangky11/miniF2F-lean4) and [ProofNet#](https://huggingface.co/datasets/PAug/ProofNetSharp) benchmarks.
- `statement_autoformalization_sampling.py`: an implementation of the sampling-based statement autoformalization method used in [Improving Autoformalization using Type Checking](https://arxiv.org/abs/2406.07222).

## Usage

### Default Lean version (latest available)

```python
from lean_interact import LeanREPLConfig, LeanServer

config = LeanREPLConfig(verbose=True) # download and build Lean REPL
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

Iterate on the environment state:

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

### Specific Lean version

```python
config = LeanREPLConfig(lean_version="v4.7.0")
```

### Existing Lean projects

```python
config = LeanREPLConfig(project=LocalProject("path/to/your/project"))
```

or

```python
config = LeanREPLConfig(project=GitProject("https://github.com/yangky11/lean4-example"))
```

You can then use `run_code` and `run_file` as usual:

```python
server = LeanServer(config)
server.run_file("file.lean")
```

> [!IMPORTANT]
> Ensure the project can be *successfully* built with `lake build` before using LeanInteract.

### Temporary project with dependencies

```python
config = LeanREPLConfig(lean_version="v4.7.0", project=TempRequireProject([LeanRequire(
    name="mathlib",
    git="https://github.com/leanprover-community/mathlib4.git",
    rev="v4.7.0"
)]))
```

Mathlib being a frequent requirement, a shortcut is available:

```python
config = LeanREPLConfig(lean_version="v4.7.0", project=TempRequireProject("mathlib"))
```

You can then use Mathlib as follows:

```python
server = LeanServer(config)
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
> - A separate cache is used for each unique set of dependencies.

### Fine-grained temporary project

For more control over the temporary project, you can use `TemporaryProject` to specify the content of the lakefile.

```python
config = LeanREPLConfig(lean_version="v4.18.0-rc1", project=TemporaryProject("""
import Lake
open Lake DSL

package "dummy" where
  version := v!"0.1.0"

@[default_target]
lean_exe "dummy" where
  root := `Main

require mathlib from git
  "https://github.com/leanprover-community/mathlib4.git" @ "v4.18.0-rc1"
"""))
```

### Tactic and proof modes (experimental)

```python
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

or by running the entire/partial proofs:

```python
server.run_proof("intro h\nexact h", proof_state=0)
```

<details>
<summary>Output</summary>

```json
{"proofState": 3, "goals": []}
```

</details>

## Helper Commands

- `install_lean`: Installs Lean 4 version manager `elan`.
- `clear_lean_cache`: Removes all Lean REPL versions and temporary projects in the package cache. This can help resolve some issues. If it does, please open an issue.

## Advanced options

### LeanServer

Two versions of Lean servers are available:

- **`LeanServer`**: A wrapper around Lean REPL. Interact with it using `run_code`, `run_file`, and `run_tactic` methods.
- **`AutoLeanServer`**: An experimental subclass of `LeanServer` automatically recovering from crashes and timeouts. It also monitors memory usage to limit *out of memory* crashes in multiprocessing contexts. Use the `add_to_session_cache` attribute available in various methods to prevent selected environment/proof states to be cleared.

> [!TIP]
>
> - To run multiple requests in parallel, we recommend using multiprocessing with one global `LeanREPLConfig` instance, and one `AutoLeanServer` instance per process.
> - Make sure to instantiate `LeanREPLConfig` before starting the processes to avoid conflicts during Lean REPL's download and build.
> - While `AutoLeanServer` can help prevent crashes, it is not a complete solution. If you encounter crashes, consider reducing the number of parallel processes or increasing the memory available to your system.

### Custom Lean REPL

To use a forked Lean REPL project, specify the git repository using the `repl_git` parameter in the `LeanREPLConfig`. Your fork should have a similar versioning format to <https://github.com/augustepoiroux/repl> (i.e. having a branch with commits for each Lean version). For assistance, feel free to contact [us](mailto:auguste.poiroux@epfl.ch).

## Similar tools

We recommend checking out these tools:

- **[PyPantograph](https://github.com/lenianiva/PyPantograph)**: Based on Pantograph, offering more options for proof interactions than Lean REPL.
- **[LeanDojo](https://github.com/lean-dojo/LeanDojo)**: Parses Lean projects to create datasets and interact with proof states.
- **[itp-interface](https://github.com/trishullab/itp-interface)**: A Python interface for interacting and extracting data from Lean 4 and Coq.
- **[leanclient](https://github.com/oOo0oOo/leanclient)**: Interact with the Lean LSP server.

LeanInteract is inspired by **[pylean](https://github.com/zhangir-azerbayev/repl)** and **[lean4_jupyter](https://github.com/utensil/lean4_jupyter)**.
