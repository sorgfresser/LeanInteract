import shutil

from lean_interact.server import AutoLeanServer, LeanREPLConfig, LeanRequire, LeanServer
from lean_interact.utils import clear_cache

__all__ = ["LeanREPLConfig", "LeanServer", "AutoLeanServer", "LeanRequire", "clear_cache"]

# check if lake is installed
if shutil.which("lake") is None:
    raise RuntimeError(
        "Lean 4 build system (`lake`) is not installed. You can find installation instructions here: https://leanprover-community.github.io/get_started.html"
    )
