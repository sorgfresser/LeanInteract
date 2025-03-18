import gc
import hashlib
import json
import os
import shutil
import subprocess
from copy import deepcopy
from dataclasses import dataclass
from time import sleep
from typing import Literal

import pexpect
import psutil
from git import GitCommandError, Repo

from lean_interact.utils import (
    DEFAULT_CACHE_DIR,
    DEFAULT_REPL_GIT_URL,
    DEFAULT_REPL_VERSION,
    _limit_memory,
    get_project_lean_version,
    logger,
)

DEFAULT_TIMEOUT: int | None = None


@dataclass(frozen=True)
class LeanRequire:
    """Lean project dependency"""

    name: str
    git: str
    rev: str | None = None

    def __hash__(self):
        return hash((self.name, self.git, self.rev))


@dataclass(frozen=True)
class BaseProject:
    """Base class for Lean projects"""

    def _get_directory(self, cache_dir: str, lean_version: str | None = None) -> str:
        """Get the project directory."""
        raise NotImplementedError("Subclasses must implement this method")

    def _instantiate(self, cache_dir: str, lean_version: str, verbose: bool = True) -> None:
        """Instantiate the project."""
        raise NotImplementedError("Subclasses must implement this method")


@dataclass(frozen=True)
class LocalProject(BaseProject):
    """Use an existing local Lean project directory"""

    directory: str

    def _get_directory(self, cache_dir: str, lean_version: str | None = None) -> str:
        """Get the project directory."""
        return self.directory

    def _instantiate(self, cache_dir: str, lean_version: str, verbose: bool = True):
        """Instantiate the local project."""
        stdout = None if verbose else subprocess.DEVNULL
        stderr = None if verbose else subprocess.DEVNULL
        # check that the project is buildable
        subprocess.run(["lake", "build"], cwd=self.directory, check=True, stdout=stdout, stderr=stderr)


@dataclass(frozen=True)
class GitProject(BaseProject):
    """Use an online git repository with a Lean project"""

    url: str
    rev: str | None = None

    def _get_directory(self, cache_dir: str, lean_version: str | None = None) -> str:
        repo_name = "/".join(self.url.split("/")[-2:]).replace(".git", "")
        return os.path.join(cache_dir, "git_projects", repo_name, self.rev or "latest")

    def _instantiate(self, cache_dir: str, lean_version: str, verbose: bool = True):
        """Instantiate the git project."""
        stdout = None if verbose else subprocess.DEVNULL
        stderr = None if verbose else subprocess.DEVNULL

        project_dir = self._get_directory(cache_dir)

        # check if the git repository is already cloned
        repo = Repo(project_dir) if os.path.exists(project_dir) else Repo.clone_from(self.url, project_dir)

        if self.rev:
            repo.git.checkout(self.rev)
        else:
            repo.git.pull()

        repo.submodule_update(init=True, recursive=True)

        # check that the project is buildable
        subprocess.run(["lake", "build"], cwd=project_dir, check=True, stdout=stdout, stderr=stderr)


@dataclass(frozen=True)
class BaseTempProject(BaseProject):
    """Base class for temporary Lean projects"""

    def _get_directory(self, cache_dir: str, lean_version: str | None = None) -> str:
        if lean_version is None:
            raise ValueError("`lean_version` cannot be `None`")
        # create a unique hash to allow for caching
        hash_content = self._get_hash_content(lean_version)
        tmp_project_dir = os.path.join(cache_dir, "tmp_projects", lean_version, hash_content)
        os.makedirs(tmp_project_dir, exist_ok=True)
        return tmp_project_dir

    def _instantiate(self, cache_dir: str, lean_version: str, verbose: bool = True):
        """Instantiate the temporary project."""
        stdout = None if verbose else subprocess.DEVNULL
        stderr = None if verbose else subprocess.DEVNULL

        tmp_project_dir = self._get_directory(cache_dir, lean_version)

        # check if the Lean project is already built
        if not os.path.exists(os.path.join(tmp_project_dir, "lakefile.lean")):
            # clean the content of the folder in case of a previous aborted build
            shutil.rmtree(tmp_project_dir, ignore_errors=True)
            os.makedirs(tmp_project_dir, exist_ok=True)

            # initialize the Lean project
            cmd_init = ["lake", f"+{lean_version}", "init", "dummy", "exe.lean"]
            if lean_version.startswith("v4") and int(lean_version.split(".")[1]) <= 7:
                cmd_init = ["lake", f"+{lean_version}", "init", "dummy", "exe"]
            subprocess.run(cmd_init, cwd=tmp_project_dir, check=True, stdout=stdout, stderr=stderr)

            # Create or modify the lakefile
            self._modify_lakefile(tmp_project_dir, lean_version)

            logger.info("Preparing Lean environment with dependencies (may take a while the first time)...")
            subprocess.run(["lake", "update"], cwd=tmp_project_dir, check=True, stdout=stdout, stderr=stderr)
            # in case mathlib is used as a dependency, we try to get the cache
            subprocess.run(["lake", "exe", "cache", "get"], cwd=tmp_project_dir, stdout=stdout, stderr=stderr)
            subprocess.run(["lake", "build"], cwd=tmp_project_dir, check=True, stdout=stdout, stderr=stderr)

    def _get_hash_content(self, lean_version: str) -> str:
        """Return a unique hash for the project content."""
        raise NotImplementedError("Subclasses must implement this method")

    def _modify_lakefile(self, project_dir: str, lean_version: str) -> None:
        """Modify the lakefile according to project needs."""
        raise NotImplementedError("Subclasses must implement this method")


@dataclass(frozen=True)
class TemporaryProject(BaseTempProject):
    """Use custom lakefile content to create a temporary Lean project"""

    content: str

    def _get_hash_content(self, lean_version: str) -> str:
        """Return a unique hash based on the content."""
        return hashlib.sha256(self.content.encode()).hexdigest()

    def _modify_lakefile(self, project_dir: str, lean_version: str) -> None:
        """Write the content to the lakefile."""
        with open(os.path.join(project_dir, "lakefile.lean"), "w") as f:
            f.write(self.content)


@dataclass(frozen=True)
class TempRequireProject(BaseTempProject):
    """
    Set up a temporary project with dependencies.
    As Mathlib is a common dependency, you can just set `require="mathlib"` and a compatible version of mathlib will be used.
    This feature has been developed mostly to be able to run benchmarks using Mathlib as a dependency
    (such as [ProofNet#](https://huggingface.co/datasets/PAug/ProofNetSharp) or
    [MiniF2F](https://github.com/yangky11/miniF2F-lean4)) without having to manually set up a Lean project.
    """

    require: Literal["mathlib"] | list[LeanRequire]

    def _normalize_require(self, lean_version: str) -> list[LeanRequire]:
        """Normalize the require field to always be a list."""
        require = self.require
        if require == "mathlib":
            return [LeanRequire("mathlib", "https://github.com/leanprover-community/mathlib4.git", lean_version)]
        assert isinstance(require, list)
        return sorted(require, key=lambda x: x.name)

    def _get_hash_content(self, lean_version: str) -> str:
        """Return a unique hash based on dependencies."""
        require = self._normalize_require(lean_version)
        return hashlib.sha256(str(require).encode()).hexdigest()

    def _modify_lakefile(self, project_dir: str, lean_version: str) -> None:
        """Add requirements to the lakefile."""
        require = self._normalize_require(lean_version)
        with open(os.path.join(project_dir, "lakefile.lean"), "a") as f:
            for req in require:
                f.write(f'\n\nrequire {req.name} from git\n  "{req.git}"' + (f' @ "{req.rev}"' if req.rev else ""))


class LeanREPLConfig:
    def __init__(
        self,
        lean_version: str | None = None,
        project: BaseProject | None = None,
        repl_rev: str = DEFAULT_REPL_VERSION,
        repl_git: str = DEFAULT_REPL_GIT_URL,
        cache_dir: str = DEFAULT_CACHE_DIR,
        max_memory: int = 32 * 1024,
        verbose: bool = False,
    ):
        """
        Configuration class used to specify the Lean environment we wish to use.

        Args:
            lean_version: Lean version you want to use with the REPL.
                If `None`, `LeanREPLConfig` will try to infer `lean_version` from `project`, otherwise
                the latest version available in the Lean REPL repository will be used.
            project: Source configuration for the Lean project. Use one of:
                - LocalProject: A directory where a local Lean project is stored.
                - GitProject: A git repository with a Lean project.
                - TemporaryProject: A temporary Lean project with a custom lakefile that will be created.
                - TempRequireProject: A temporary Lean project with dependencies that will be created.
            repl_rev:
                The REPL version you want to use. It is not recommended to change this value unless you know what you are doing.
            repl_git:
                The git repository of the Lean REPL. It is not recommended to change this value unless you know what you are doing.
            cache_dir:
                The directory where the Lean REPL and temporary Lean projects with dependencies will be cached.
                Default is inside the package directory.
            max_memory:
                The maximum memory usage in MB for the Lean server. Setting this value too low may lead to timeouts.
                Default is 32GB.
            verbose:
                Whether to print additional information during the setup process.
        """
        self.lean_version = lean_version
        self.project = project
        self.repl_git = repl_git
        self.repl_rev = repl_rev
        self.cache_dir = cache_dir
        self.max_memory = max_memory

        self.verbose = verbose
        self._stdout = None if self.verbose else subprocess.DEVNULL
        self._stderr = None if self.verbose else subprocess.DEVNULL

        self.repo_name = "/".join(self.repl_git.split("/")[-2:]).replace(".git", "")
        self.cache_clean_repl_dir = os.path.join(self.cache_dir, self.repo_name, "repl_clean_copy")

        # check if lake is installed
        if shutil.which("lake") is None:
            raise RuntimeError(
                "Lean 4 build system (`lake`) is not installed. You can try to run `install-lean` or find installation instructions here: https://leanprover-community.github.io/get_started.html"
            )

        self._setup_repl()

        assert isinstance(self.lean_version, str)

        if self.project is None:
            self._working_dir = self._cache_repl_dir
        else:
            self.project._instantiate(
                cache_dir=self.cache_dir,
                lean_version=self.lean_version,
                verbose=self.verbose,
            )
            self._working_dir = self.project._get_directory(cache_dir=self.cache_dir, lean_version=self.lean_version)

    def _setup_repl(self) -> None:
        assert isinstance(self.repl_rev, str)

        # check if the repl is already cloned
        if not os.path.exists(self.cache_clean_repl_dir):
            os.makedirs(self.cache_clean_repl_dir, exist_ok=True)
            Repo.clone_from(self.repl_git, self.cache_clean_repl_dir)

        repo = Repo(self.cache_clean_repl_dir)
        try:
            repo.git.checkout(self.repl_rev)
        except GitCommandError:
            repo.remote().pull()
            try:
                repo.git.checkout(self.repl_rev)
            except GitCommandError:
                raise ValueError(f"Lean REPL version `{self.repl_rev}` is not available.")

        # check if the Lean version is available in the repository
        lean_versions_sha = self._get_available_lean_versions_sha()
        lean_versions_sha_dict = dict(lean_versions_sha)
        if not lean_versions_sha:
            raise ValueError("No Lean versions are available in the Lean REPL repository.")
        if self.lean_version is None:
            if (
                self.project is None
                or isinstance(self.project, TemporaryProject)
                or isinstance(self.project, TempRequireProject)
            ):
                self.lean_version = lean_versions_sha[-1][0]
            elif isinstance(self.project, LocalProject) or isinstance(self.project, GitProject):
                # get the Lean version from the project
                inferred_ver = get_project_lean_version(self.project._get_directory(self.cache_dir))
                self.lean_version = inferred_ver if inferred_ver else lean_versions_sha[-1][0]
        if self.lean_version not in lean_versions_sha_dict:
            raise ValueError(
                f"Lean version `{self.lean_version}` is required but not available in the Lean REPL repository."
            )

        # check if the repl revision is already in the cache
        self._cache_repl_dir = os.path.join(self.cache_dir, self.repo_name, f"repl_{self.repl_rev}_{self.lean_version}")
        if not os.path.exists(self._cache_repl_dir):
            # copy the repository to the version directory and checkout the required revision
            os.makedirs(self._cache_repl_dir, exist_ok=True)
            shutil.copytree(self.cache_clean_repl_dir, self._cache_repl_dir, dirs_exist_ok=True)
            cached_repo = Repo(self._cache_repl_dir)
            cached_repo.git.checkout(lean_versions_sha_dict[self.lean_version])

        # check that the lean version is correct
        assert self.lean_version == get_project_lean_version(self._cache_repl_dir), (
            f"An error occured while preparing the Lean REPL. The requested Lean version `{self.lean_version}` "
            f"does not match the fetched Lean version in the repository `{get_project_lean_version(self._cache_repl_dir)}`."
            f"Please open an issue on GitHub if you think this is a bug."
        )

        # build the repl
        subprocess.run(
            ["lake", "build"], cwd=self._cache_repl_dir, check=True, stdout=self._stdout, stderr=self._stderr
        )

    def _get_available_lean_versions_sha(self) -> list[tuple[str, str]]:
        """
        Get the available Lean versions for the selected REPL.
        """
        repo = Repo(self.cache_clean_repl_dir)
        return [
            (str(commit.message.strip()), commit.hexsha) for commit in repo.iter_commits(f"{self.repl_rev}...master")
        ]

    def get_available_lean_versions(self) -> list[str]:
        """
        Get the available Lean versions for the selected REPL.
        """
        return [commit[0] for commit in self._get_available_lean_versions_sha()]

    def is_setup(self) -> bool:
        return hasattr(self, "_working_dir")


class LeanServer:
    config: LeanREPLConfig
    _proc: pexpect.spawn | None

    def __init__(self, config: LeanREPLConfig):
        """
        This class is a Python wrapper for the Lean REPL. Please refer to the \
        [Lean REPL documentation](https://github.com/leanprover-community/repl) to learn more about the Lean REPL commands.

        \u26a0 Multiprocessing: instantiate one config before starting multiprocessing. Then instantiate one `LeanServer`
        per process by passing the config instance to the constructor. This will ensure that the REPL is already set up
        for your specific environment and avoid concurrency conflicts.

        Args:
            config: The configuration for the Lean server.
        """
        self.config = config
        assert self.config.is_setup(), "The Lean environment has not been set up properly."
        self._proc = None
        self.start()

    @property
    def lean_version(self) -> str | None:
        return self.config.lean_version

    def start(self) -> None:
        self._proc = pexpect.spawn(
            "/bin/bash",
            cwd=self.config._working_dir,
            encoding="utf-8",
            codec_errors="replace",
            echo=False,
            preexec_fn=lambda: _limit_memory(self.config.max_memory),
        )
        # `stty -icanon` is required to handle arbitrary long inputs in the Lean REPL
        self._proc.sendline("stty -icanon")
        self._proc.sendline(f"lake env {self.config._cache_repl_dir}/.lake/build/bin/repl")

    def is_alive(self) -> bool:
        return hasattr(self, "_proc") and self._proc is not None and self._proc.isalive()

    def kill(self) -> None:
        if hasattr(self, "_proc") and self._proc:
            self._proc.terminate(force=True)
            del self._proc
        gc.collect()

    def restart(self) -> None:
        self.kill()
        self.start()

    def __del__(self):
        self.kill()

    def _execute_cmd_in_repl(self, json_query: str, verbose: bool, timeout: float | None) -> str:
        """Send JSON queries to the Lean REPL and wait for the standard delimiter."""
        assert self._proc is not None
        if verbose:
            logger.info(f"Sending query: {json_query}")
        self._proc.sendline(json_query)
        self._proc.sendline()
        _ = self._proc.expect_exact("\r\n\r\n", timeout=timeout)
        return self._proc.before or ""

    def _parse_repl_output(self, raw_output: str, verbose: bool) -> dict:
        """Clean up raw REPL output and parse JSON response."""
        output = raw_output.replace("\r\n", "\n")
        output = output[output.find('{"') :] if '{"' in output else ""
        if verbose:
            logger.info(f"Server raw output: `{raw_output}")
            logger.info(f"Server cleaned output: `{output}`")
        try:
            return json.loads(output)
        except json.JSONDecodeError as e:
            raise json.JSONDecodeError(
                msg=f"Could not parse the Lean server output: `{repr(output)}`.",
                doc=e.doc,
                pos=e.pos,
            ) from e

    def _process_request(
        self, dict_query: dict, verbose: bool = False, timeout: float | None = DEFAULT_TIMEOUT
    ) -> dict:
        if not self.is_alive():
            raise ChildProcessError("The Lean server is not running.")

        json_query = json.dumps(dict_query, ensure_ascii=False)
        try:
            raw_output = self._execute_cmd_in_repl(json_query, verbose, timeout)
        except pexpect.exceptions.TIMEOUT as e:
            self.kill()
            raise TimeoutError(f"The Lean server did not respond in time ({timeout=}) and is now killed.") from e
        except pexpect.exceptions.EOF as e:
            self.kill()
            raise ConnectionAbortedError(
                "The Lean server closed unexpectedly. Possible reasons (not exhaustive):\n"
                "- Not enough memory and/or compute available\n"
                "- Your cached Lean REPL is corrupted. In this case, clear the cache"
                " using the `clear_cache` (`from pyleanrepl import clear_cache`) method."
            ) from e

        return self._parse_repl_output(raw_output, verbose)

    def run_file(
        self,
        path: str,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        extra_repl_args: dict | None = None,
    ) -> dict:
        if not path:
            raise ValueError("`path` cannot be `None` or empty")
        if not isinstance(path, str):
            raise ValueError("`path` must be a string")
        return self._process_request(
            dict_query=dict(path=path) | (extra_repl_args or {}), timeout=timeout, verbose=verbose
        )

    def run_code(
        self,
        code: str,
        env: int | None = None,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        extra_repl_args: dict | None = None,
    ) -> dict:
        """
        Run a short Lean code snippet and return the Lean REPL output.

        Args:
            code: The Lean code to run.
            env: The environment to use.
            verbose: Whether to print additional information during the verification process.
            timeout: The timeout for the request.
            optimize_prefix_env_reuse: Whether to optimize the environment reuse by checking if the current\
                code is a prefix of a cached code.
                Used for performance optimization purposes and only if `env` parameter is `None`.
        Returns:
            The output of the Lean server.
        """

        if not code:
            raise ValueError("`code` cannot be `None` or empty")

        if not isinstance(code, str):
            raise ValueError("`code` must be a string")

        command = dict(cmd=code) | (dict(env=env) if env is not None else {}) | (extra_repl_args or {})
        return self._process_request(dict_query=command, timeout=timeout, verbose=verbose)

    def run_tactic(
        self,
        tactic: str,
        proof_state: int,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        extra_repl_args: dict | None = None,
    ) -> dict:
        """
        Run a tactic in the Lean REPL.

        Args:
            tactic: The tactic to run.
            proof_state: The proof state to apply the tactic to.
            verbose: Whether to print additional information during the verification process.
            timeout: The timeout for the request.
        Returns:
            The output of the Lean server.
        """
        if not tactic:
            raise ValueError("`tactic` cannot be `None` or empty")

        if not isinstance(tactic, str):
            raise ValueError("`tactic` must be a string")

        if proof_state is None:
            raise ValueError("`proof_state` cannot be `None`")

        command = dict(tactic=tactic, proofState=proof_state) | (extra_repl_args or {})
        return self._process_request(dict_query=command, timeout=timeout, verbose=verbose)

    def run_proof(
        self,
        proof: str,
        proof_state: int,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        extra_repl_args: dict | None = None,
    ):
        """
        EXPERIMENTAL

        Run a (partial) proof in the Lean REPL.

        Args:
            proof: The (partial) proof to run.
            proof_state: The proof state to apply the proof to.
            verbose: Whether to print additional information during the verification process.
            timeout: The timeout for the request.
        Returns:
            The output of the Lean server.
        """
        if not proof:
            raise ValueError("`proof` cannot be `None` or empty")

        if not isinstance(proof, str):
            raise ValueError("`proof` must be a string")

        if proof_state is None:
            raise ValueError("`proof_state` cannot be `None`")

        return self.run_tactic(
            tactic=f"(\n{proof}\n)",
            proof_state=proof_state,
            verbose=verbose,
            timeout=timeout,
            extra_repl_args=extra_repl_args,
        )

    def pickle(
        self,
        path: str,
        env: int | None = None,
        proof_state: int | None = None,
        timeout: float | None = DEFAULT_TIMEOUT,
        verbose: bool = False,
    ) -> None:
        """
        Pickle the environment or proof state to a file.
        Only one of `env` or `proof_state` can be provided.

        Args:
            path: The path to the .olean file to save the environment or proof state.
            env: The environment to pickle.
            proof_state: The proof state to pickle.
        """
        if not path:
            raise ValueError("`path` cannot be `None` or empty")
        if not isinstance(path, str):
            raise ValueError("`path` must be a string")

        if env is None and proof_state is None:
            raise ValueError("Either `env` or `proof_state` must be provided")
        if env is not None and proof_state is not None:
            raise ValueError("Only one of `env` or `proof_state` can be provided")

        os.makedirs(os.path.dirname(path), exist_ok=True)

        if env is not None:
            self._process_request(dict_query=dict(pickleTo=path, env=env), verbose=verbose, timeout=timeout)
        else:
            self._process_request(
                dict_query=dict(pickleTo=path, proofState=proof_state), verbose=verbose, timeout=timeout
            )

    def unpickle(
        self,
        path: str,
        is_proof_state: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        verbose: bool = False,
    ) -> int:
        """
        Unpickle an environment or proof state from a file.

        Args:
            path: The path to the .olean file to load the environment or proof state from.
            is_proof_state: Whether to unpickle a proof state instead of an environment.
        Returns:
            The environment or proof state index.
        """
        if not path:
            raise ValueError("`path` cannot be `None` or empty")
        if not isinstance(path, str):
            raise ValueError("`path` must be a string")

        if is_proof_state:
            return self._process_request(
                dict_query=dict(unpickleProofStateFrom=path), verbose=verbose, timeout=timeout
            )["proofState"]
        else:
            return self._process_request(dict_query=dict(unpickleEnvFrom=path), verbose=verbose, timeout=timeout)["env"]


@dataclass
class _SessionState:
    session_id: int
    repl_id: int
    pickle_file: str
    is_proof_state: bool


class AutoLeanServer(LeanServer):
    def __init__(
        self,
        config: LeanREPLConfig,
        max_total_memory: float = 0.8,
        max_restart_attempts: int = 5,
    ):
        """
        This class is a Python wrapper for the Lean REPL. `AutoLeanServer` differs from `LeanServer` by automatically \
        restarting when it runs out of memory to clear Lean environment states. \
        It also automatically recovers from timeouts (). \
        An exponential backoff strategy is used to restart the server, making this class slightly more friendly for multiprocessing
        than `LeanServer` when multiple instances are competing for resources. \
        Please refer to the original [Lean REPL documentation](https://github.com/leanprover-community/repl) to learn more about the \
        Lean REPL commands.

        A session cache is implemented to keep user-selected environment / proof states across these automatic restarts. \
        Use the `add_to_session_cache` parameter in the different class methods to add the command to \
        the session cache. `AutoLeanServer` works best when only a few states are cached simultaneously. \
        You can use `remove_from_session_cache` and `clear_session_cache` to clear the session cache. \
        Cached state indices are negative integers starting from -1 to not conflict with the positive integers used by the Lean REPL.

        **Note:** the session cache is specific to each `AutoLeanServer` instance and is cleared when the instance is deleted. \
        If you want truly persistent states, you can use the `pickle` and `unpickle` methods to save and load states to disk.

        \u26a0 Multiprocessing: instantiate the config before starting multiprocessing. Then instantiate one `LeanServer`
        per process by passing the config instance to the constructor. This will ensure that the REPL is already set up
        for your specific environment and avoid concurrency conflicts.

        Args:
            config: The configuration for the Lean server.
            max_total_memory: The maximum proportion of total memory usage before restarting the Lean server. Default is 0.8 (80%).
            max_restart_attempts: The maximum number of restart attempts before raising a `MemoryError`. Default is 5.
        """
        self._state_counter = 0
        self._restart_persistent_session_cache: dict[int, _SessionState] = {}
        self._max_total_memory = max_total_memory
        self._max_restart_attempts = max_restart_attempts
        self._restart_counter = 0
        super().__init__(config=config)

    def _get_repl_state_id(self, state_id: int | None) -> int | None:
        if state_id is None:
            return None
        if state_id >= 0:
            return state_id
        return self._restart_persistent_session_cache[state_id].repl_id

    def _reload_session_cache(self, verbose: bool = False) -> None:
        """
        Reload the session cache. This method should be called only after a restart of the Lean REPL.
        """
        for state_data in self._restart_persistent_session_cache.values():
            state_data.repl_id = self.unpickle(
                path=state_data.pickle_file,
                is_proof_state=state_data.is_proof_state,
                verbose=verbose,
                add_to_session_cache=False,
            )

    def restart(self, verbose: bool = False) -> None:
        super().restart()
        self._reload_session_cache(verbose=verbose)

    def remove_from_session_cache(self, session_state_id: int) -> None:
        """
        Remove an environment from the session cache.

        Args:
            env_id: The environment id to remove.
        """
        if (state_cache := self._restart_persistent_session_cache.pop(session_state_id, None)) is not None:
            os.remove(state_cache.pickle_file)

    def clear_session_cache(self, force: bool = False) -> None:
        """
        Clear the session cache.

        Args:
            force: Whether to directly clear the session cache. \
                `force=False` will only clear the session cache the next time the server runs out of memory while \
                still allowing you to add new content in the session cache in the meantime.
        """
        states_data = list(self._restart_persistent_session_cache.values())
        for state_data in states_data:
            self.remove_from_session_cache(session_state_id=state_data.session_id)
        self._restart_persistent_session_cache = {}
        if force:
            self.restart()

    def __del__(self):
        # delete the session cache
        for state_data in self._restart_persistent_session_cache.values():
            try:
                os.remove(state_data.pickle_file)
            except FileNotFoundError:
                pass

        super().__del__()

    def _store_session_cache(
        self, hash_key: str, repl_id: int, is_proof_state: bool = False, verbose: bool = False
    ) -> int:
        self._state_counter -= 1
        process_id = os.getpid()  # use process id to avoid conflicts in multiprocessing
        pickle_file = os.path.join(
            self.config._working_dir,
            f"session_cache/{hashlib.sha256(hash_key.encode()).hexdigest()}_{process_id}.olean",
        )
        if is_proof_state:
            self.pickle(path=pickle_file, proof_state=repl_id, verbose=verbose)
        else:
            self.pickle(path=pickle_file, env=repl_id, verbose=verbose)

        self._restart_persistent_session_cache[self._state_counter] = _SessionState(
            session_id=self._state_counter,
            repl_id=repl_id,
            pickle_file=pickle_file,
            is_proof_state=is_proof_state,
        )
        return self._state_counter

    def run_file(
        self,
        path: str,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        extra_repl_args: dict | None = None,
        add_to_session_cache: bool = False,
    ) -> dict:
        res = super().run_file(path=path, verbose=verbose, timeout=timeout, extra_repl_args=extra_repl_args)
        if add_to_session_cache:
            try:
                res["env"] = self._store_session_cache(
                    hash_key=f"path_{extra_repl_args}", repl_id=res["env"], is_proof_state=False, verbose=verbose
                )
            except (ValueError, KeyError) as e:
                raise ValueError("Failed to add the environment to the session cache.") from e
        return res

    def run_code(
        self,
        code: str,
        env: int | None = None,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        extra_repl_args: dict | None = None,
        add_to_session_cache: bool = False,
    ) -> dict:
        res = super().run_code(code=code, env=env, verbose=verbose, timeout=timeout, extra_repl_args=extra_repl_args)
        if add_to_session_cache:
            try:
                res["env"] = self._store_session_cache(
                    hash_key=f"code_{extra_repl_args}_env_{env}",
                    repl_id=res["env"],
                    is_proof_state=False,
                    verbose=verbose,
                )
            except (ValueError, KeyError) as e:
                raise ValueError("Failed to add the environment to the session cache.") from e
        return res

    def run_tactic(
        self,
        tactic: str,
        proof_state: int,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        extra_repl_args: dict | None = None,
        add_to_session_cache: bool = False,
    ) -> dict:
        res = super().run_tactic(
            tactic=tactic, proof_state=proof_state, verbose=verbose, timeout=timeout, extra_repl_args=extra_repl_args
        )
        if add_to_session_cache:
            try:
                res["proofState"] = self._store_session_cache(
                    hash_key=f"tactic_{extra_repl_args}_proofstate_{proof_state}",
                    repl_id=res["proofState"],
                    is_proof_state=True,
                    verbose=verbose,
                )
            except (ValueError, KeyError) as e:
                raise ValueError("Failed to add the proof state to the session cache.") from e
        return res

    def run_proof(
        self,
        proof: str,
        proof_state: int,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        extra_repl_args: dict | None = None,
        add_to_session_cache: bool = False,
    ) -> dict:
        res = super().run_proof(
            proof=proof, proof_state=proof_state, verbose=verbose, timeout=timeout, extra_repl_args=extra_repl_args
        )
        if add_to_session_cache:
            try:
                res["proofState"] = self._store_session_cache(
                    hash_key=f"proof_{extra_repl_args}_proofstate_{proof_state}",
                    repl_id=res["proofState"],
                    is_proof_state=True,
                    verbose=verbose,
                )
            except (ValueError, KeyError) as e:
                raise ValueError("Failed to add the proof state to the session cache.") from e
        return res

    def _process_request(
        self,
        dict_query: dict,
        verbose: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
    ) -> dict:
        if psutil.virtual_memory().percent >= 100 * self._max_total_memory:
            self.kill()
            if self._restart_counter >= self._max_restart_attempts:
                raise MemoryError(
                    f"Memory usage is too high. We attempted to restart the Lean server {self._max_restart_attempts} times without success."
                )
            if verbose:
                logger.info("Memory usage is too high. Reloading the Lean server...")
            sleep(2**self._restart_counter)
            self._restart_counter += 1
            return self._process_request(dict_query=dict_query, verbose=verbose, timeout=timeout)

        if not self.is_alive():
            self.start()
            self._reload_session_cache(verbose=verbose)

        # Replace the negative environment / proof state ids with the corresponding REPL ids
        if dict_query.get("env", 0) < 0:
            dict_query = deepcopy(dict_query)
            dict_query["env"] = self._get_repl_state_id(dict_query["env"])
        if dict_query.get("proofState", 0) < 0:
            dict_query = deepcopy(dict_query)
            dict_query["proofState"] = self._get_repl_state_id(dict_query["proofState"])

        res = super()._process_request(dict_query=dict_query, verbose=verbose, timeout=timeout)

        self._restart_counter = 0
        return res

    def unpickle(
        self,
        path: str,
        is_proof_state: bool = False,
        timeout: float | None = DEFAULT_TIMEOUT,
        verbose: bool = False,
        add_to_session_cache: bool = False,
    ) -> int:
        res = super().unpickle(path=path, is_proof_state=is_proof_state, timeout=timeout, verbose=verbose)
        if add_to_session_cache:
            try:
                res = self._store_session_cache(path, repl_id=res, is_proof_state=is_proof_state, verbose=verbose)
            except (ValueError, KeyError) as e:
                raise ValueError("Failed to add the environment to the session cache.") from e
        return res
