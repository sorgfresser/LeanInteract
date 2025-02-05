import time
import unittest
import unittest.mock
from queue import Queue
from threading import Thread

import pexpect
import psutil

from lean_interact.server import (
    DEFAULT_TIMEOUT,
    AutoLeanServer,
    LeanREPLConfig,
    LeanRequire,
    LeanServer,
    _SessionState,
)


class TestLeanServer(unittest.TestCase):
    maxDiff = None

    @classmethod
    def setUpClass(cls):
        # Pre-run configs for all available versions to get the cache
        lean_versions = LeanREPLConfig().get_available_lean_versions()
        for version in ["v4.7.0", "v4.14.0", lean_versions[-1]]:
            LeanREPLConfig(lean_version=version)

        # prepare Mathlib for the last version
        LeanREPLConfig(lean_version=lean_versions[-1], require="mathlib")

    def test_init_with_lean_version(self):
        lean_versions = LeanREPLConfig().get_available_lean_versions()
        for version in ["v4.7.0", "v4.14.0", lean_versions[-1]]:
            server = AutoLeanServer(config=LeanREPLConfig(lean_version=version))
            self.assertEqual(server.lean_version, version)

    def test_init_with_require(self):
        lean_versions = LeanREPLConfig().get_available_lean_versions()
        require = [
            LeanRequire(
                name="mathlib",
                git="https://github.com/leanprover-community/mathlib4.git",
                rev=lean_versions[-1],
            )
        ]
        server = AutoLeanServer(LeanREPLConfig(require="mathlib"))
        self.assertEqual(server.config.require, require)

    def test_init_with_project_dir(self):
        project_dir = "/tmp/path/to/project"
        with self.assertRaises(FileNotFoundError):
            AutoLeanServer(LeanREPLConfig(project_dir=project_dir, lean_version="v4.7.0"))

    def test_run_code_simple(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        result = server.run_code("def x := 42")
        self.assertDictEqual(result, {"env": 0})

    def test_run_code_with_env(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        result1 = server.run_code("def x := 1", add_to_session_cache=True)
        self.assertDictEqual(result1, {"env": -1})
        env_id = result1.get("env")
        result2 = server.run_code("def y := x + 1", env=env_id)
        self.assertDictEqual(result2, {"env": 1})

    def test_run_tactic(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        result = server.run_code("theorem zero_eq_zero : 0 = 0 := sorry", add_to_session_cache=True)
        self.assertDictEqual(
            result,
            {
                "env": -1,
                "messages": [
                    {
                        "data": "declaration uses 'sorry'",
                        "endPos": {"column": 20, "line": 1},
                        "pos": {"column": 8, "line": 1},
                        "severity": "warning",
                    }
                ],
                "sorries": [
                    {
                        "endPos": {"column": 37, "line": 1},
                        "goal": "⊢ 0 = 0",
                        "pos": {"column": 32, "line": 1},
                        "proofState": 0,
                    }
                ],
            },
        )
        tactic_result = server.run_tactic("rfl", proof_state=0)
        self.assertDictEqual(tactic_result, {"proofState": 1, "goals": []})

    def test_run_file_nonexistent(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        output = server.run_file("nonexistent_file.lean")
        self.assertDictEqual(
            output, {"message": "no such file or directory (error code: 2)\n  file: nonexistent_file.lean"}
        )

    def test_is_alive(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        self.assertTrue(server.is_alive())
        server.kill()
        self.assertFalse(server.is_alive())

    def test_restart(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        old_proc = server._proc
        server.restart()
        self.assertNotEqual(server._proc, old_proc)
        self.assertTrue(server.is_alive())

    def test_clear_session_cache(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        server.run_code("def x := 1", add_to_session_cache=True)
        server.clear_session_cache()
        self.assertEqual(len(server._restart_persistent_session_cache), 0)

    def test_init_with_invalid_rev(self):
        with self.assertRaises(Exception):
            AutoLeanServer(config=LeanREPLConfig(lean_version="invalid_rev"))

    def test_extremely_long_command(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        result = server.run_code("def " + "a" * 10000 + " : 1 + 1 = 2 := sorry", add_to_session_cache=True)
        self.assertDictEqual(
            result,
            {
                "env": -1,
                "sorries": [
                    {
                        "proofState": 0,
                        "pos": {"line": 1, "column": 10020},
                        "goal": "⊢ 1 + 1 = 2",
                        "endPos": {"line": 1, "column": 10025},
                    }
                ],
                "messages": [
                    {
                        "severity": "warning",
                        "pos": {"line": 1, "column": 4},
                        "endPos": {"line": 1, "column": 10004},
                        "data": "declaration uses 'sorry'",
                    }
                ],
            },
        )
        result = server.run_tactic("rfl", proof_state=0)
        self.assertDictEqual(result, {"proofState": 1, "goals": []})

    def test_lean_version(self):
        server = AutoLeanServer(config=LeanREPLConfig(lean_version="v4.14.0"))
        result = server.run_code("#eval Lean.versionString")
        self.assertDictEqual(
            result,
            {
                "env": 0,
                "messages": [
                    {
                        "data": '"4.14.0"',
                        "endPos": {"column": 5, "line": 1},
                        "pos": {"column": 0, "line": 1},
                        "severity": "info",
                    }
                ],
            },
        )

    def test_mathlib(self):
        server = AutoLeanServer(config=LeanREPLConfig(require="mathlib"))
        result = server.run_code("import Mathlib", add_to_session_cache=True)
        self.assertDictEqual(result, {"env": -1})
        result = server.run_code(
            "theorem exercise_1_1a\n  (x : ℝ) (y : ℚ) (n : ℕ) (h : Odd n) :\n  ( Irrational x ) -> Irrational ( x + y ) := sorry",
            env=-1,
            add_to_session_cache=True,
        )
        self.assertDictEqual(
            result,
            {
                "sorries": [
                    {
                        "proofState": 0,
                        "pos": {"column": 46, "line": 3},
                        "endPos": {"column": 51, "line": 3},
                        "goal": "x : ℝ\ny : ℚ\nn : ℕ\nh : Odd n\n⊢ Irrational x → Irrational (x + ↑y)",
                    }
                ],
                "messages": [
                    {
                        "data": "declaration uses 'sorry'",
                        "endPos": {"column": 21, "line": 1},
                        "pos": {"column": 8, "line": 1},
                        "severity": "warning",
                    }
                ],
                "env": -2,
            },
        )
        result = server.run_tactic("exact?", proof_state=0)
        self.assertDictEqual(
            result,
            {
                "proofState": 1,
                "goals": [],
                "messages": [
                    {
                        "data": "Try this: exact fun a => (fun {q} {x} => irrational_add_rat_iff.mpr) a",
                        "endPos": {"column": 0, "line": 0},
                        "pos": {"column": 0, "line": 0},
                        "severity": "info",
                    }
                ],
            },
        )

    def test_restart_with_env(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        result = server.run_code("def x := 1", add_to_session_cache=True)
        env_id = result.get("env")
        self.assertEqual(env_id, -1)
        server.restart()
        result = server.run_code("noncomputable def y := x + 1", env=env_id)
        self.assertDictEqual(result, {"env": 1})
        self.assertEqual(list(server._restart_persistent_session_cache.keys()), [env_id])

    def test_process_request_memory_restart(self):
        server = AutoLeanServer(config=LeanREPLConfig(), max_total_memory=0.01, max_restart_attempts=2)
        # Mock psutil.virtual_memory().percent to be high
        with unittest.mock.patch("psutil.virtual_memory") as mock_virtual_memory:
            mock_virtual_memory.return_value.percent = 99.0
            with unittest.mock.patch("time.sleep", return_value=None):
                with self.assertRaises(MemoryError):
                    server._process_request({}, verbose=False)
        self.assertFalse(server.is_alive())

    @unittest.mock.patch("lean_interact.server.LeanServer._process_request")
    def test_process_request_with_negative_env_id(self, mock_super):
        server = AutoLeanServer(config=LeanREPLConfig())
        # Prepare restart_persistent_session_cache
        server._restart_persistent_session_cache[-1] = _SessionState(-1, 10, "", False)
        request = {"env": -1}
        with unittest.mock.patch.object(server, "_get_repl_state_id", return_value=10):
            mock_super.return_value = {"env": 10}
            result = server._process_request(request)
            mock_super.assert_called_with(dict_query={"env": 10}, verbose=False, timeout=DEFAULT_TIMEOUT)
            self.assertEqual(result, {"env": 10})

    @unittest.mock.patch("lean_interact.server.LeanServer._process_request")
    def test_process_request_with_negative_proof_state_id(self, mock_super):
        server = AutoLeanServer(config=LeanREPLConfig())
        # Prepare restart_persistent_session_cache
        server._restart_persistent_session_cache[-2] = _SessionState(-2, 20, "", True)
        request = {"proofState": -2}
        with unittest.mock.patch.object(server, "_get_repl_state_id", return_value=20):
            mock_super.return_value = {"proofState": 20}
            result = server._process_request(request)
            mock_super.assert_called_with(dict_query={"proofState": 20}, verbose=False, timeout=DEFAULT_TIMEOUT)
            self.assertEqual(result, {"proofState": 20})

    @unittest.mock.patch("lean_interact.server.LeanServer._process_request", return_value={})
    @unittest.mock.patch("lean_interact.server.psutil.virtual_memory")
    def test_process_request_server_restart(self, mock_virtual_memory, mock_process_request):
        server = AutoLeanServer(config=LeanREPLConfig())
        server.kill()
        self.assertFalse(server.is_alive())
        mock_virtual_memory.return_value.percent = 0.0
        server._process_request({})
        self.assertTrue(server.is_alive())

    @unittest.mock.patch("lean_interact.server.LeanServer._process_request")
    def test_process_request_timeout_recovery(self, mock_process_request):
        # Simulate a timeout exception
        mock_process_request.side_effect = TimeoutError("Simulated timeout")

        server = AutoLeanServer(config=LeanREPLConfig())
        with self.assertRaises(TimeoutError):
            server._process_request({"cmd": "test"}, timeout=1)

        # Verify that the server did not attempt to restart
        self.assertTrue(server.is_alive())
        mock_process_request.assert_called_once()

    # @unittest.mock.patch("lean_interact.server.LeanServer._process_request")
    # def test_process_request_eof_recovery(self, mock_process_request):
    #     # Simulate a ConnectionAbortedError exception indicating server crash
    #     mock_process_request.side_effect = ConnectionAbortedError("Simulated server crash")

    #     max_restart_attempts = 2
    #     server = AutoLeanServer(config=LeanREPLConfig(), max_restart_attempts=max_restart_attempts)
    #     with self.assertRaises(ConnectionAbortedError):
    #         server._process_request({"cmd": "test"})

    #     # Verify that the server attempted to restart max_restart_attempts times
    #     self.assertFalse(server.is_alive())
    #     self.assertEqual(mock_process_request.call_count, max_restart_attempts + 1)

    @unittest.mock.patch("lean_interact.server.psutil.virtual_memory")
    def test_process_request_memory_overload_recovery(self, mock_virtual_memory):
        # Simulate high memory usage
        mock_virtual_memory.return_value.percent = 95.0

        max_restart_attempts = 2
        server = AutoLeanServer(
            config=LeanREPLConfig(), max_total_memory=0.8, max_restart_attempts=max_restart_attempts
        )
        with self.assertRaises(MemoryError):
            server._process_request({"cmd": "test"})

        # Verify that the server is not alive after exceeding max restart attempts
        self.assertFalse(server.is_alive())

    def test_autoleanserver_recovery_after_timeout(self):
        server = AutoLeanServer(config=LeanREPLConfig())

        # Mock the expect_exact method to raise a timeout exception
        def raise_timeout_error(*args, **kwargs):
            raise pexpect.exceptions.TIMEOUT("")

        with unittest.mock.patch.object(server._proc, "expect_exact", side_effect=raise_timeout_error):
            with self.assertRaises(TimeoutError):
                server.run_code("def x := y")

        # Send a new command to verify auto-recovery
        result = server.run_code("def z := 3")
        self.assertDictEqual(result, {"env": 0})

    def test_leanserver_killed_after_timeout(self):
        server = LeanServer(config=LeanREPLConfig())

        # Mock the expect_exact method to raise a timeout exception
        def raise_timeout_error(*args, **kwargs):
            raise pexpect.exceptions.TIMEOUT("")

        with unittest.mock.patch.object(server._proc, "expect_exact", side_effect=raise_timeout_error):
            with self.assertRaises(TimeoutError):
                server.run_code("def a := b")

        # Ensure the server is killed after the timeout
        # self.assertFalse(server.is_alive())
        with self.assertRaises(ChildProcessError):
            server.run_code("def z := 3")

    def test_run_proof(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        result = server.run_code("theorem test_run_proof : (x : Nat) -> x = x := sorry", add_to_session_cache=True)
        self.assertEqual(result.get("env"), -1)

        proof_result = server.run_proof("intro x\nrfl", proof_state=0)
        self.assertDictEqual(proof_result, {"proofState": 1, "goals": []})

    def test_run_proof_equivalence(self):
        server = AutoLeanServer(config=LeanREPLConfig())
        result = server.run_code("theorem test_run_proof_seq : (x : Nat) -> x = x := sorry", add_to_session_cache=True)
        self.assertEqual(result.get("env"), -1)

        step1 = server.run_tactic("intro x", proof_state=0)
        step2 = server.run_tactic("rfl", proof_state=step1["proofState"])
        self.assertDictEqual(step2, {"proofState": 2, "goals": []})

    def test_bug_increasing_memory(self):
        mem_limit = 512
        server = AutoLeanServer(config=LeanREPLConfig(max_memory=mem_limit))

        # Get initial memory usage
        assert server._proc is not None
        server_process = psutil.Process(server._proc.pid)
        start_mem = server_process.memory_info().rss / (1024 * 1024)  # Convert to MB

        # Run code in separate thread to allow memory monitoring
        result_queue = Queue()

        def run_code_thread():
            try:
                # execute a known fast infinite memory-increasing code
                result = server.run_code(
                    "theorem dummy {x : ∀ α, X α} {ι : Type _} {x₁ : ι → ∀ α, X α} {x₂ : ι → ∀ α, X α} (x₃ : ι → ∀ α, X α) {x₄ : ι → ∀ α, X α} {x₅ : ι → ∀ α, X α} {x₆ : ι → ∀ α, X α} {x₇ : ι → ∀ α, X α} {x₈ : ι → ∀ α, X α} {x₉ : ι → ∀ α, X α} {x₀ : ι → ∀ α, X α} {x₁₀ : ι → ∀ α, X α} {x₁₁ : ι → ∀ α, X α} {x₁₂ : ι → ∀ α, X α} (x₁₃ : ι → ∀ α, X α) (x₁₄ : ι → ∀ α, X α) (x₁₅ : ι → ∀ α, X α) {x₁₆ : ι → ∀ α, X α} {x₁₇ : ι → ∀ α, X α} {x₁₈ : ι → ∀ α, X α} {x₁₉ : ι → ∀ α, X α} {x₂₀ : ι → ∀ α, X α} (x₂₁ : ι → ∀ α, X α) (x₂₂ : ι → ∀ α, X α) (x₂₃ : ι → ∀ α, X α) (x₂₄ : ι → ∀ α, X α) (x₂₅ : ι → ∀ α, X α) (x₂₆ : ι → ∀ α, X α) (x₂₇ : ι → ∀ α, X α) (x₂₈ : ι → ∀ α, X α) (x₂₉ : ι → ∀ α, X α) (x₃₀ : ι → ∀ α, X α) {x₃₁ : ι → ∀ α, X α} {x₃₂ : ι → ∀ α, X α} (x₃₃ : ι → ∀ α, X α) (x₃₄ : ι → ∀ α, X α) (x₃₅ : ι → ∀ α, X α) (x₃ sorry",
                    timeout=10,
                )
                result_queue.put(("success", result))
            except TimeoutError as e:
                result_queue.put(("timeout", e))
            except Exception as e:
                result_queue.put(("error", e))

        # Start code execution thread
        thread = Thread(target=run_code_thread)
        thread.start()

        # Monitor memory usage
        max_mem = start_mem
        while thread.is_alive():
            current_mem = server_process.memory_info().rss / (1024 * 1024)
            max_mem = max(max_mem, current_mem)
            if current_mem > mem_limit:
                server.kill()
                raise MemoryError(f"Memory usage exceeded limit: {current_mem:.1f}MB > {mem_limit}MB")
            time.sleep(1)

        # Get result
        status, result = result_queue.get()
        if status == "error":
            raise result

        # Assert memory stayed within limits
        self.assertLess(max_mem, mem_limit, f"Memory usage peaked at {max_mem:.1f}MB, exceeding {mem_limit}MB limit")


if __name__ == "__main__":
    unittest.main()
