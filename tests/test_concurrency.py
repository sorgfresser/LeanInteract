import asyncio
import multiprocessing as mp
import os
import shutil
import tempfile
import time
import unittest
from unittest import mock

from filelock import FileLock, Timeout

from lean_interact import AutoLeanServer, LeanREPLConfig
from lean_interact.config import BaseTempProject
from lean_interact.interface import Command, CommandResponse, Message
from lean_interact.server import LeanServer
from lean_interact.utils import DEFAULT_REPL_GIT_URL


# Helper function for testing config creation in a separate process
def create_config_process(cache_dir, lean_version="v4.18.0", verbose=False):
    """Function that attempts to create a LeanREPLConfig instance."""
    try:
        # This should timeout if another process is already holding the lock
        LeanREPLConfig(lean_version=lean_version, cache_dir=str(cache_dir), verbose=verbose)
        return True
    except Exception:
        return False


# Function to access/modify a cache file with locking
def access_cache_file(file_path, process_id, timeout=1):
    """Function to simulate a process accessing and modifying a cache file."""
    try:
        lock_path = f"{file_path}.lock"
        with FileLock(lock_path, timeout=timeout):
            # Read the file
            try:
                with open(file_path, "r", encoding="utf-8") as f:
                    content = f.read()

                # Modify the file (simulate updating cache)
                with open(file_path, "w", encoding="utf-8") as f:
                    f.write(f"{content} - Modified by process {process_id}")
                time.sleep(timeout)
                return True
            except Exception as e:
                print(f"Error in access_cache_file: {e}")
                return False
    except Timeout:
        return False


class TestFileLocks(unittest.TestCase):
    """Tests for file locking functionality."""

    def setUp(self):
        self.temp_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Clean up temporary directory
        if os.path.exists(self.temp_dir):
            shutil.rmtree(self.temp_dir)

    def test_config_repl_setup_lock(self):
        """Test that the REPL setup lock prevents concurrent access."""
        # Create a temporary cache directory
        cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)

        # Prepare the lock file path that would be used by LeanREPLConfig
        repo_name = "/".join(DEFAULT_REPL_GIT_URL.split("/")[-2:]).replace(".git", "")
        cache_clean_repl_dir = os.path.join(cache_dir, repo_name, "repl_clean_copy")
        os.makedirs(os.path.dirname(cache_clean_repl_dir), exist_ok=True)
        lock_file = f"{cache_clean_repl_dir}.lock"

        # Acquire the lock that would be used by LeanREPLConfig
        with FileLock(lock_file, timeout=0.1):
            # Start a process that tries to create a LeanREPLConfig, which should try to acquire the same lock
            ctx = mp.get_context("spawn")

            p = ctx.Process(target=create_config_process, args=(cache_dir,))
            p.start()
            p.join(timeout=30)

            # The process should have exited with a timeout exception
            self.assertNotEqual(p.exitcode, 0, "Process should have failed to acquire the lock")

    def test_temporary_project_lock(self):
        """Test that the temporary project lock prevents concurrent access."""

        # Create a temp project instance with a simple hash
        class SimpleTestProject(BaseTempProject):
            def _get_hash_content(self, lean_version: str) -> str:
                """Return a fixed hash."""
                return "test_hash"

            def _modify_lakefile(self, project_dir: str, lean_version: str) -> None:
                """Do nothing."""
                pass

        test_project = SimpleTestProject()

        # Get the project directory and lock path
        cache_dir = os.path.join(self.temp_dir, "cache")
        os.makedirs(cache_dir, exist_ok=True)
        project_dir = test_project._get_directory(str(cache_dir), "v4.18.0")
        lock_file = f"{project_dir}.lock"

        # Hold the lock to simulate another process using it
        with FileLock(lock_file, timeout=0.1):
            # Try to instantiate the project which should try to acquire the same lock
            with self.assertRaises(Exception):
                # Use a short timeout to avoid test hanging
                with mock.patch("lean_interact.config.FileLock", return_value=FileLock(lock_file, timeout=0.1)):
                    test_project._instantiate(str(cache_dir), "v4.18.0", verbose=False)


def _worker(idx, result_queue, config):
    try:
        # Each process gets its own server instance
        server = AutoLeanServer(config)
        # Each process runs a simple Lean command
        cmd = Command(cmd=f"def x{idx} := {idx}")
        result = server.run(cmd)
        # Return result type and env id
        if isinstance(result, CommandResponse):
            result_queue.put((idx, True, result.env))
        else:
            result_queue.put((idx, False, str(result)))
    except Exception as e:
        result_queue.put((idx, False, str(e)))


class TestAutoLeanServerConcurrency(unittest.TestCase):
    """Realistic concurrent use-cases for AutoLeanServer and LeanREPLConfig (no mocks)."""

    @classmethod
    def setUpClass(cls):
        # Use a dedicated cache dir for concurrency tests
        cls.cache_dir = tempfile.mkdtemp(prefix="lean_concurrency_cache_")
        # Pre-instantiate config to avoid race in REPL setup
        cls.config = LeanREPLConfig(cache_dir=cls.cache_dir, verbose=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.cache_dir, ignore_errors=True)

    def test_concurrent_autoleanserver(self):
        """Test concurrent use of AutoLeanServer."""
        num_proc = 4
        ctx = mp.get_context("spawn")
        result_queue = ctx.Queue()
        procs = [ctx.Process(target=_worker, args=(i, result_queue, self.config)) for i in range(num_proc)]
        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=60)
        results = [result_queue.get(timeout=5) for _ in range(num_proc)]
        self.assertTrue(all(success for (_, success, _) in results), f"Failures: {results}")
        env_ids = [env for (_, _, env) in results]
        self.assertEqual(len(env_ids), num_proc)


class TestAsyncRun(unittest.TestCase):
    """Tests for the async_run methods of LeanServer and AutoLeanServer, including concurrency."""

    @classmethod
    def setUpClass(cls):
        cls.config = LeanREPLConfig(verbose=True)

    def test_async_run_eval_leansrv(self):
        server = LeanServer(self.config)
        cmd = Command(cmd="#eval 1 + 1")
        result = asyncio.run(server.async_run(cmd))
        # Should be a CommandResponse and contain a message with '2'
        self.assertIsInstance(result, CommandResponse)
        assert isinstance(result, CommandResponse)
        self.assertTrue(any("2" in m.data for m in result.messages))

    def test_async_run_eval_autoleansrv(self):
        server = AutoLeanServer(self.config)
        cmd = Command(cmd="#eval 2 + 2")
        result = asyncio.run(server.async_run(cmd))
        self.assertIsInstance(result, CommandResponse)
        assert isinstance(result, CommandResponse)
        self.assertTrue(any("4" in m.data for m in result.messages))

    def test_async_run_concurrent_multiple_servers(self):
        n = 30
        config = LeanREPLConfig(verbose=True)
        servers = [AutoLeanServer(config) for _ in range(n)]
        cmds = [Command(cmd=f"#eval {i} * {i}") for i in range(n)]

        async def run_all():
            tasks = [srv.async_run(cmd) for srv, cmd in zip(servers, cmds)]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())

        # Collect expected outputs
        expected_outputs = {str(i * i) for i in range(n)}
        actual_outputs = set()
        for result in results:
            self.assertIsInstance(result, CommandResponse)
            assert isinstance(result, CommandResponse)
            for m in result.messages:
                actual_outputs.add(m.data)
        self.assertEqual(actual_outputs, expected_outputs)

    def test_async_run_concurrent_autoleansrv(self):
        n = 20
        server = AutoLeanServer(self.config)
        cmds = [Command(cmd=f"#eval {i} + {i}") for i in range(n)]

        async def run_all():
            tasks = [server.async_run(cmd) for cmd in cmds]
            return await asyncio.gather(*tasks)

        results = asyncio.run(run_all())

        # Collect expected outputs
        expected_outputs = {str(i + i) for i in range(n)}
        actual_outputs = set()
        for result in results:
            self.assertIsInstance(result, CommandResponse)
            assert isinstance(result, CommandResponse)
            # Find all numbers in all messages
            for m in result.messages:
                actual_outputs.add(m.data)
        self.assertEqual(actual_outputs, expected_outputs)


if __name__ == "__main__":
    unittest.main()
