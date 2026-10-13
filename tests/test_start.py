"""Boundary tests for the standalone launcher; no installs or APIs are used."""

from __future__ import annotations

from contextlib import contextmanager
import io
import json
import queue
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import start


class FakeProcess:
    def __init__(self, output: str = "", returncode: int = 0):
        self.stdout = io.StringIO(output)
        self.returncode = returncode

    def wait(self):
        return self.returncode


class InterruptingQueue:
    def __init__(self):
        self.events = []
        self.interrupted = False

    def put(self, event):
        self.events.append(event)

    def get(self, timeout=None):
        if not self.interrupted:
            self.interrupted = True
            raise KeyboardInterrupt
        return self.events.pop(0)

    def get_nowait(self):
        if not self.events:
            raise queue.Empty
        return self.events.pop(0)


class InterruptibleProcess(FakeProcess):
    def __init__(self):
        super().__init__("API_KEY=secret\n")
        self.terminated = False
        self.killed = False

    def poll(self):
        return -15 if self.terminated else None

    def terminate(self):
        self.terminated = True

    def kill(self):
        self.killed = True
        self.terminated = True

    def wait(self, timeout=None):
        return self.poll()


class StartTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()
        self.venv = self.root / "env with spaces"
        self.interpreter = self.venv / "Scripts" / "python.exe"
        self.setup_dir = self.root / "setup"
        self.setup_dir.mkdir()

    def _args(self, *extra: str) -> list[str]:
        return ["--venv", str(self.venv), *extra]

    @contextmanager
    def _interactive(self):
        with mock.patch.object(start.sys.stdin, "isatty", return_value=True), \
             mock.patch.object(start.sys.stdout, "isatty", return_value=True), \
             mock.patch.object(start, "_private_setup_dir", return_value=self.setup_dir):
            yield

    def _healthy_probe(self, command, **kwargs):
        return subprocess.CompletedProcess(command, 0, stdout="")

    def test_non_tty_refuses_before_setup_or_install(self):
        with mock.patch.object(start.sys.stdin, "isatty", return_value=False), \
             mock.patch.object(start, "read_preferences", return_value={}), \
             mock.patch.object(start.install, "ensure_venv") as ensure, \
             mock.patch.object(start.install, "install_dependencies") as install_deps, \
             mock.patch.object(start.subprocess, "run") as run, \
             mock.patch("builtins.print") as printer:
            code = start.main(self._args())
        self.assertEqual(code, 2)
        ensure.assert_not_called()
        install_deps.assert_not_called()
        run.assert_not_called()
        self.assertIn("需要交互式终端", str(printer.call_args))

    def test_no_venv_delegates_default_active_or_repository_selection(self):
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("version_info = 3.12.13\n", encoding="utf-8")
        with self._interactive(), \
             mock.patch.object(start, "read_preferences", return_value={}), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv) as choose, \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=self._healthy_probe), \
             mock.patch.object(start.install, "install_dependencies") as install_deps:
            code = start.main([])
        self.assertEqual(code, 0)
        choose.assert_called_once_with(None)
        install_deps.assert_not_called()

    def test_omitted_lang_stays_out_of_wizard_argv_and_uses_saved_language_for_startup(self):
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("version_info = 3.12.13\n", encoding="utf-8")
        with self._interactive(), \
             mock.patch.object(start, "read_preferences", return_value={"language": "en"}), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv), \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=self._healthy_probe) as run, \
             mock.patch.object(start.install, "install_dependencies") as install_deps, \
             mock.patch("builtins.print") as printer:
            code = start.main(self._args())
        self.assertEqual(code, 0)
        install_deps.assert_not_called()
        wizard = run.call_args.args[0]
        self.assertNotIn("--lang", wizard)
        self.assertIn("Preparing the target Python 3.12 environment. Please wait.", str(printer.mock_calls))

    def test_existing_healthy_core_skips_installation_and_passes_arguments(self):
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("version_info = 3.12.13\n", encoding="utf-8")
        with self._interactive(), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv), \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=self._healthy_probe) as run, \
             mock.patch.object(start.install, "install_dependencies") as install_deps:
            code = start.main(self._args("--config", "study dir/project.toml", "--lang", "en"))
        self.assertEqual(code, 0)
        install_deps.assert_not_called()
        wizard = run.call_args.args[0]
        self.assertEqual(wizard[:4], [str(self.interpreter), "-m", "lumina.wizard", "--setup-log"])
        self.assertIn(("--config", str((Path.cwd() / "study dir/project.toml").resolve())),
                      tuple(zip(wizard, wizard[1:])))
        self.assertEqual(wizard[-2:], ["--lang", "en"])
        self.assertEqual(run.call_args.kwargs["cwd"], str(start.REPO_ROOT))
        self.assertIsNone(run.call_args.kwargs["stdin"])
        self.assertIsNone(run.call_args.kwargs["stdout"])
        self.assertIsNone(run.call_args.kwargs["stderr"])
        self.assertTrue(any("tomlkit" in str(call) for call in run.call_args_list))
        self.assertTrue((self.setup_dir / "startup.log").is_file())
        self.assertTrue((self.setup_dir / "installation.log").is_file())
        self.assertTrue((self.setup_dir / "errors.log").is_file())
        self.assertFalse((self.setup_dir / "events.jsonl").exists())

    def test_existing_core_only_attempts_default_marker_install_and_records_fallback(self):
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("version_info = 3.12.13\n", encoding="utf-8")

        def probe(command, **kwargs):
            result = self._healthy_probe(command, **kwargs)
            if command[-1:] == [start.install.MARKER_PROBE]:
                return subprocess.CompletedProcess(command, 1, stdout="marker missing")
            return result

        with self._interactive(), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv), \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=probe), \
             mock.patch.object(start.install, "install_dependencies", return_value=False) as install_deps, \
             mock.patch.object(start, "_launch_wizard", return_value=0):
            code = start.main(self._args())

        self.assertEqual(code, 0)
        self.assertEqual(install_deps.call_args.args[:3], (self.interpreter, start.REPO_ROOT, False))
        receipt = json.loads((self.venv / start.INSTALL_PROFILE_RECEIPT).read_text(encoding="utf-8"))
        self.assertEqual(receipt["profile"], "pypdf-fallback")

    def test_recorded_fallback_skips_repeated_marker_install_with_warning(self):
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("version_info = 3.12.13\n", encoding="utf-8")
        (self.venv / start.INSTALL_PROFILE_RECEIPT).write_text(
            json.dumps({"profile": "pypdf-fallback"}), encoding="utf-8"
        )
        with self._interactive(), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv), \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=self._healthy_probe), \
             mock.patch.object(start.install, "install_dependencies") as install_deps, \
             mock.patch.object(start, "_launch_wizard", return_value=0), \
             mock.patch("builtins.print") as printer:
            code = start.main(self._args("--lang", "en"))

        self.assertEqual(code, 0)
        install_deps.assert_not_called()
        self.assertIn("pypdf fallback", str(printer.mock_calls))

    def test_recorded_marker_skips_install_when_marker_probe_is_healthy(self):
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("version_info = 3.12.13\n", encoding="utf-8")
        (self.venv / start.INSTALL_PROFILE_RECEIPT).write_text(
            json.dumps({"profile": "marker"}), encoding="utf-8"
        )
        with self._interactive(), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv), \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=self._healthy_probe), \
             mock.patch.object(start.install, "install_dependencies") as install_deps, \
             mock.patch.object(start, "_launch_wizard", return_value=0):
            code = start.main(self._args())

        self.assertEqual(code, 0)
        install_deps.assert_not_called()

    def test_new_environment_uses_default_marker_profile(self):
        self.venv.mkdir()
        with self._interactive(), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv), \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=self._healthy_probe), \
             mock.patch.object(start.install, "install_dependencies") as install_deps:
            code = start.main(self._args())
        self.assertEqual(code, 0)
        self.assertEqual(install_deps.call_args.args[:3], (self.interpreter, start.REPO_ROOT, False))
        self.assertIsNotNone(install_deps.call_args.kwargs["runner"])

    def test_missing_pip_bootstraps_only_target(self):
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("version_info = 3.12.13\n", encoding="utf-8")
        probe_results = iter([
            subprocess.CompletedProcess([], 1, stdout="no pip"),
            subprocess.CompletedProcess([], 0, stdout="pip 24"),
            subprocess.CompletedProcess([], 0, stdout=""),
            *[subprocess.CompletedProcess([], 0, stdout="") for _ in start._core_modules()],
            subprocess.CompletedProcess([], 0, stdout=""),
            subprocess.CompletedProcess([], 0, stdout=""),
        ])
        popen_commands = []

        def popen(command, **kwargs):
            popen_commands.append(command)
            return FakeProcess("bootstrapped\n")

        with self._interactive(), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv), \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=lambda *a, **k: next(probe_results)), \
             mock.patch.object(start.subprocess, "Popen", side_effect=popen), \
             mock.patch.object(start.install, "install_dependencies"):
            code = start.main(self._args())
        self.assertEqual(code, 0)
        self.assertEqual(popen_commands[0], [str(self.interpreter), "-m", "ensurepip", "--upgrade"])

    def test_install_failure_does_not_launch_wizard(self):
        self.venv.mkdir()
        (self.venv / "pyvenv.cfg").write_text("version_info = 3.12.13\n", encoding="utf-8")
        responses = iter([
            subprocess.CompletedProcess([], 0, stdout=""),
            subprocess.CompletedProcess([], 1, stdout="broken dependencies"),
            subprocess.CompletedProcess([], 1, stdout="broken dependencies"),
        ])
        with self._interactive(), \
             mock.patch.object(start.install, "choose_venv_path", return_value=self.venv), \
             mock.patch.object(start.install, "ensure_venv", return_value=self.interpreter), \
             mock.patch.object(start.subprocess, "run", side_effect=lambda *a, **k: next(responses)), \
             mock.patch.object(start.install, "install_dependencies") as install_deps, \
             mock.patch.object(start, "_launch_wizard") as launch:
            code = start.main(self._args())
        self.assertEqual(code, 1)
        install_deps.assert_called_once()
        launch.assert_not_called()

    def test_stream_runner_preserves_output_and_exit_code_without_shell(self):
        log = self.root / "installation.log"
        with mock.patch.object(start.subprocess, "Popen", return_value=FakeProcess("API_KEY=secret\n", 7)) as popen:
            result = start._stream_process([self.interpreter, "-m", "pip", "install", "-r", "requirements.txt"], log)
        self.assertEqual(result.returncode, 7)
        self.assertIn("[REDACTED]", log.read_text(encoding="utf-8"))
        self.assertNotIn("secret", log.read_text(encoding="utf-8"))
        self.assertEqual(popen.call_args.args[0], [str(self.interpreter), "-m", "pip", "install", "-r", "requirements.txt"])
        self.assertFalse(popen.call_args.kwargs["shell"])
        self.assertEqual(popen.call_args.kwargs["encoding"], "utf-8")
        self.assertEqual(popen.call_args.kwargs["env"]["PYTHONIOENCODING"], "utf-8")

    def test_stream_runner_waits_with_notice_when_output_is_silent(self):
        class DelayedOutput(io.StringIO):
            def readline(self):
                if self.tell() == 0:
                    import time
                    time.sleep(0.03)
                    self.write("done\n")
                    self.seek(0)
                return super().readline()

        process = FakeProcess()
        process.stdout = DelayedOutput()
        with mock.patch.object(start.subprocess, "Popen", return_value=process), \
             mock.patch("builtins.print") as printer:
            result = start._stream_process(
                [self.interpreter, "-m", "pip", "install"],
                self.root / "installation.log",
                heartbeat_seconds=0.01,
            )
        self.assertEqual(result.returncode, 0)
        self.assertIn("输出可能会在下载阶段暂时静默", str(printer.mock_calls))

    def test_ctrl_c_terminates_child_and_keeps_redacted_output(self):
        process = InterruptibleProcess()
        events = InterruptingQueue()
        log = self.root / "installation.log"
        with mock.patch.object(start.subprocess, "Popen", return_value=process), \
             mock.patch.object(start.queue, "Queue", return_value=events):
            with self.assertRaises(KeyboardInterrupt):
                start._stream_process([self.interpreter, "-m", "pip", "install"], log)
        text = log.read_text(encoding="utf-8")
        self.assertTrue(process.terminated)
        self.assertFalse(process.killed)
        self.assertIn("[REDACTED]", text)
        self.assertNotIn("secret", text)
        self.assertIn("child terminated", text)


if __name__ == "__main__":
    unittest.main()
