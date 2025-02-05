import logging
import os
import resource
import shutil

from packaging.version import Version
from rich.logging import RichHandler

logger = logging.getLogger("document_autoformalization")
logger.setLevel("INFO")
handler = RichHandler(rich_tracebacks=True)
handler.setLevel("NOTSET")
handler.setFormatter(logging.Formatter("%(message)s", datefmt="[%X]"))
logger.handlers = []
logger.addHandler(handler)


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_CACHE_DIR = os.path.join(ROOT_DIR, "cache")
DEFAULT_REPL_GIT_URL = "https://github.com/augustepoiroux/repl"

os.makedirs(DEFAULT_CACHE_DIR, exist_ok=True)


def _limit_memory(max_mb: int):
    """Limit the memory usage of the current process."""
    try:
        resource.setrlimit(resource.RLIMIT_AS, (max_mb * 1024 * 1024, max_mb * 1024 * 1024))
    except ValueError:
        logger.warning(f"Failed to set memory limit to {max_mb} MB.")


def clear_cache():
    shutil.rmtree(DEFAULT_CACHE_DIR, ignore_errors=True)


def fetch_lean_versions(repl_repo_dir: str) -> list[str]:
    """
    Fetch the list of Lean versions available in the `versions` folder of the local REPL repository.
    """
    versions_dir = os.path.join(repl_repo_dir, "versions")
    if not os.path.isdir(versions_dir):
        return []
    return sorted(
        [filename[:-5] for filename in os.listdir(versions_dir) if filename.endswith(".diff")],
        key=lambda v: Version(v),
    )


def get_project_lean_version(project_dir: str) -> str | None:
    """
    Get the Lean version used in a project.
    """
    toolchain_file = os.path.join(project_dir, "lean-toolchain")
    if os.path.isfile(toolchain_file):
        with open(toolchain_file, "r", encoding="utf-8") as f:
            content = f.read().strip()
            if content:
                try:
                    return content.split(":")[-1]
                except Exception:
                    pass
    return None
