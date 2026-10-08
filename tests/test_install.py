"""Installer tests: every package boundary is a stubbed runner, so no network, no installs."""

from __future__ import annotations

import subprocess
import sys
import types
import unittest
import venv
from contextlib import redirect_stderr, redirect_stdout
from io import StringIO
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import install
from install import PDF_WARNING

REPO_PIPELINE = install.REPO_ROOT / "run_pipeline.py"


class FakeRunner:
    """Records argv lists and replays scripted return codes keyed by substring."""

    def __init__(self, failures=()):
        self.commands = []
        self.kwargs = []
        self.failures = tuple(failures)

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        self.kwargs.append(kwargs)
        joined = " ".join(str(part) for part in command)
        code = 1 if any(marker in joined for marker in self.failures) else 0
        return types.SimpleNamespace(returncode=code, args=list(command), stdout=b"", stderr=b"")

    def argv_list(self) -> list[str]:
        return [" ".join(str(part) for part in command) for command in self.commands]

    @property
    def calls(self):
        return list(zip(self.commands, self.kwargs))


def capture(func, *args, **kwargs) -> tuple[str, str, object]:
    """Run func with captured streams; returns stdout, stderr and the result."""
    out, err = StringIO(), StringIO()
    with redirect_stdout(out), redirect_stderr(err):
        result = func(*args, **kwargs)
    return out.getvalue(), err.getvalue(), result


def failure(callable_, *args) -> install.InstallError:
    """Return the InstallError raised by the call; fail the test if it is not raised."""
    try:
        callable_(*args)
    except install.InstallError as exc:
        return exc
    raise AssertionError(f"{callable_} did not raise InstallError")


def make_env(root: Path, version: str = "3.12.13") -> Path:
    root.mkdir(parents=True, exist_ok=True)
    # uv writes version_info; the test for the stdlib shape uses a real EnvBuilder instead.
    (root / "pyvenv.cfg").write_text(
        f"home = C:/Python312\nimplementation = CPython\nversion_info = {version}\n",
        encoding="utf-8",
    )
    interpreter = install.interpreter_path(root)
    interpreter.parent.mkdir(parents=True, exist_ok=True)
    interpreter.write_bytes(b"")
    return interpreter


def make_repo(root: Path, with_pdf: bool = True) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    (root / install.CORE_REQUIREMENTS).write_text("pypdf>=6.1,<7\n", encoding="utf-8")
    if with_pdf:
        (root / install.PDF_REQUIREMENTS).write_text(
            f"-r {install.CORE_REQUIREMENTS}\nmarker-pdf==2.0.0\n", encoding="utf-8"
        )
    return root


class TempDirTest(unittest.TestCase):
    def setUp(self):
        import tempfile

        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name).resolve()


class VenvPathSelectionTests(TempDirTest):
    def test_explicit_path_wins(self):
        chosen = install.choose_venv_path(
            self.root / "custom", running_prefix="C:/active", running_base_prefix="C:/base"
        )
        self.assertEqual(chosen, (self.root / "custom").resolve())

    def test_active_venv_used_when_running_inside_one(self):
        chosen = install.choose_venv_path(
            None, running_prefix=str(self.root / "env"), running_base_prefix="C:/base"
        )
        self.assertEqual(chosen, (self.root / "env").resolve())

    def test_repository_venv_used_outside_any_venv(self):
        repo = make_repo(self.root / "repo")
        chosen = install.choose_venv_path(
            None, running_prefix="C:/base", running_base_prefix="C:/base", repo_root=repo
        )
        self.assertEqual(chosen, repo / ".venv")


class EnsureVenvTests(TempDirTest):
    def test_creates_environment_only_when_absent(self):
        target = self.root / "fresh"

        def create(path):
            scripts = install.interpreter_path(Path(path)).parent
            scripts.mkdir(parents=True, exist_ok=True)
            install.interpreter_path(Path(path)).write_bytes(b"")

        with mock.patch.object(install.venv, "EnvBuilder") as builder:
            builder.return_value = mock.MagicMock(create=create)
            interpreter = install.ensure_venv(target)
        builder.assert_called_once_with(with_pip=True)
        self.assertEqual(interpreter, install.interpreter_path(target))
        self.assertTrue(interpreter.is_file())

    def test_reuses_existing_valid_environment_without_creating(self):
        target = self.root / "existing"
        expected = make_env(target)
        with mock.patch.object(install.venv, "EnvBuilder") as builder:
            interpreter = install.ensure_venv(target)
        builder.assert_not_called()
        self.assertEqual(interpreter, expected)

    def test_rejects_existing_environment_with_wrong_python_version(self):
        make_env(self.root / "old", version="3.11.9")
        with mock.patch.object(install.venv, "EnvBuilder"):
            caught = failure(install.ensure_venv, self.root / "old")
        self.assertIn("not Python 3.12", str(caught))

    def test_rejects_nonempty_non_venv_directory(self):
        target = self.root / "my-stuff"
        target.mkdir()
        (target / "notes.txt").write_text("keep me", encoding="utf-8")
        with mock.patch.object(install.venv, "EnvBuilder") as builder:
            caught = failure(install.ensure_venv, target)
        builder.assert_not_called()
        self.assertIn("refusing to overwrite", str(caught))

    def test_rejects_root_home_and_repository_root(self):
        for target in (Path(Path.home().anchor), Path.home().resolve(), install.REPO_ROOT):
            with self.subTest(target=target):
                with mock.patch.object(install.venv, "EnvBuilder") as builder:
                    failure(install.ensure_venv, target.resolve())
                builder.assert_not_called()

    def test_accepts_real_stdlib_venv_shape(self):
        """A real EnvBuilder writes 'version = 3.12.x'; uv writes 'version_info'. Accept both."""
        target = self.root / "stdlib-env"
        venv.EnvBuilder(with_pip=False).create(str(target))
        keys = {line.split("=", 1)[0].strip()
                for line in (target / "pyvenv.cfg").read_text(encoding="utf-8").splitlines()
                if "=" in line}
        self.assertNotIn("version_info", keys, "stdlib venv should not write version_info")
        with mock.patch.object(install.venv, "EnvBuilder") as builder:
            interpreter = install.ensure_venv(target)
        builder.assert_not_called()
        self.assertTrue(interpreter.is_file())
        self.assertEqual(interpreter, install.interpreter_path(target))

    def test_rejects_real_venv_from_another_python_version(self):
        target = self.root / "wrong"
        target.mkdir(parents=True)
        (target / "pyvenv.cfg").write_text("version = 3.11.9\n", encoding="utf-8")
        caught = failure(install.ensure_venv, target)
        self.assertIn("not Python 3.12", str(caught))


class InstallDependencyTests(TempDirTest):
    def setUp(self):
        super().setUp()
        self.repo = make_repo(self.root / "repo")
        self.interpreter = make_env(self.root / "env")

    def test_default_run_installs_pdf_profile_and_uses_target_interpreter(self):
        runner = FakeRunner()
        _, _, marker_ok = capture(install.install_dependencies, self.interpreter, self.repo,
                                 False, runner)
        self.assertTrue(marker_ok)
        interpreter = str(self.interpreter)
        argv = runner.argv_list()
        self.assertEqual(argv[0], f"{interpreter} -m pip install -r {self.repo / install.PDF_REQUIREMENTS}")
        # The core file is pulled in by requirements-pdf.txt, so it is not installed twice.
        self.assertNotIn(f"{interpreter} -m pip install -r {self.repo / install.CORE_REQUIREMENTS}", argv)
        for command in runner.commands:
            self.assertIsInstance(command, list)
            self.assertEqual(str(command[0]), interpreter)
            self.assertIn(str(command[1]), {"-m", "-c"})
        for _, kwargs in runner.calls:
            # cwd must be the repo so the child can import the local, uninstalled lumina package.
            self.assertEqual(kwargs["cwd"], str(install.REPO_ROOT))
        self.assertIn(f"{interpreter} -m pip check", argv)
        for module in install.CORE_PROBES:
            self.assertIn(f"{interpreter} -c import {module}", argv)
        self.assertIn(f"{interpreter} -c {install.MARKER_PROBE}", argv)

    def test_marker_probe_uses_published_module_level_api(self):
        self.assertIn("from marker.models import create_model_dict", install.MARKER_PROBE)
        self.assertIn("from marker.output import text_from_rendered", install.MARKER_PROBE)
        self.assertNotIn("PdfConverter.create_model_dict", install.MARKER_PROBE)
        # Importing must not instantiate: that would download and load model weights.
        self.assertNotIn("create_model_dict(", install.MARKER_PROBE)
        self.assertNotIn("text_from_rendered(", install.MARKER_PROBE)
        compile(install.MARKER_PROBE, "<marker probe>", "exec")

    def test_marker_failure_falls_back_to_core_and_reports_degraded_mode(self):
        runner = FakeRunner(failures=(install.PDF_REQUIREMENTS,))
        _, err, marker_ok = capture(install.install_dependencies, self.interpreter, self.repo,
                                    False, runner)
        interpreter = str(self.interpreter)
        argv = runner.argv_list()
        self.assertFalse(marker_ok)
        self.assertIn(f"{interpreter} -m pip install -r {self.repo / install.CORE_REQUIREMENTS}", argv)
        self.assertNotIn(f"{interpreter} -c {install.MARKER_PROBE}", argv)
        self.assertIn(PDF_WARNING, err)
        self.assertIn(PDF_WARNING, err)

    def test_marker_import_failure_degrades_without_reinstalling_core(self):
        runner = FakeRunner(failures=("marker.converters.pdf",))
        _, err, marker_ok = capture(install.install_dependencies, self.interpreter, self.repo,
                                    False, runner)
        interpreter = str(self.interpreter)
        self.assertFalse(marker_ok)
        # requirements-pdf.txt already carries -r requirements.txt, so core needs no second install.
        self.assertNotIn(
            f"{interpreter} -m pip install -r {self.repo / install.CORE_REQUIREMENTS}",
            runner.argv_list(),
        )
        self.assertIn(f"{interpreter} -c {install.MARKER_PROBE}", runner.argv_list())
        self.assertIn(PDF_WARNING, err)

    def test_skip_marker_installs_core_only(self):
        runner = FakeRunner()
        _, _, marker_ok = capture(install.install_dependencies, self.interpreter, self.repo,
                                 True, runner)
        interpreter = str(self.interpreter)
        argv = runner.argv_list()
        self.assertFalse(marker_ok)
        self.assertNotIn(f"{interpreter} -m pip install -r {self.repo / install.PDF_REQUIREMENTS}", argv)
        self.assertIn(f"{interpreter} -m pip install -r {self.repo / install.CORE_REQUIREMENTS}", argv)

    def test_missing_pdf_requirements_falls_back_to_core(self):
        repo = make_repo(self.root / "bare", with_pdf=False)
        runner = FakeRunner()
        _, err, marker_ok = capture(install.install_dependencies, self.interpreter, repo, False,
                                    runner)
        interpreter = str(self.interpreter)
        self.assertFalse(marker_ok)
        self.assertIn(
            f"{interpreter} -m pip install -r {repo / install.CORE_REQUIREMENTS}", runner.argv_list()
        )
        self.assertIn(install.PDF_REQUIREMENTS, err)

    def test_core_failure_is_fatal(self):
        runner = FakeRunner(failures=(install.CORE_REQUIREMENTS,))
        error = failure(install.install_dependencies, self.interpreter, self.repo, True, runner)
        self.assertIn(install.CORE_REQUIREMENTS, str(error))

    def test_pip_check_failure_is_fatal(self):
        runner = FakeRunner(failures=("pip check",))
        error = failure(install.install_dependencies, self.interpreter, self.repo, True, runner)
        self.assertIn("pip check", str(error))

    def test_core_import_probe_failure_is_fatal(self):
        runner = FakeRunner(failures=("import pypdf",))
        error = failure(install.install_dependencies, self.interpreter, self.repo, True, runner)
        self.assertIn("pypdf", str(error))


class MainTests(TempDirTest):
    def test_reports_nonzero_on_install_error(self):
        with mock.patch.object(install, "_install", side_effect=install.InstallError("boom")):
            _, err, code = capture(install.main, [])
        self.assertEqual(code, 1)
        self.assertIn("boom", err)

    def test_runs_against_requested_environment_and_reports_interpreter(self):
        interpreter = make_env(self.root / "env")
        args = mock.Mock(venv=self.root / "env", skip_marker=True)
        with mock.patch.object(install, "install_dependencies", return_value=False) as dependency:
            out, _, code = capture(install._install, args)
        self.assertEqual(code, 0)
        dependency.assert_called_once_with(interpreter, install.REPO_ROOT, True)
        self.assertIn(str(interpreter), out)
        self.assertIn("run_pipeline.py", out)
        self.assertIn("pypdf fallback", out)

    def test_run_hint_uses_absolute_paths_and_study_config(self):
        interpreter = self.root / "env dir" / "Scripts" / "python.exe"
        hint = install.run_hint(interpreter)
        self.assertIn(str(REPO_PIPELINE), hint)
        self.assertIn(str(install.REPO_ROOT / install.EXAMPLE_CONFIG), hint)
        self.assertNotIn("--domain", hint)
        self.assertNotIn("--stage", hint)
        if sys.platform == "win32":
            self.assertTrue(hint.startswith("& "), hint)
        # A quoted path is what lets the PowerShell '&' form survive spaces in the venv directory.
        self.assertIn(f'"{interpreter}"', hint)

    def test_reports_nonzero_on_os_error(self):
        with mock.patch.object(install, "_install", side_effect=OSError("no such file")):
            _, err, code = capture(install.main, [])
        self.assertEqual(code, 1)
        self.assertIn("no such file", err)

    def test_missing_os_interpreter_is_reported_not_traced(self):
        with mock.patch.object(venv, "EnvBuilder") as builder:
            builder.return_value = mock.MagicMock()  # creates nothing
            _, err, code = capture(install.main, ["--venv", str(self.root / "never-created")])
        self.assertEqual(code, 1)
        self.assertIn("no interpreter", err)


class SubprocessContractTests(unittest.TestCase):
    def test_run_never_uses_a_shell(self):
        with mock.patch.object(install.subprocess, "run") as run:
            install._run(["python", "-c", "pass"])
        run.assert_called_once_with(["python", "-c", "pass"], cwd=str(install.REPO_ROOT))
        self.assertIs(install.subprocess.run, subprocess.run)


if __name__ == "__main__":
    unittest.main()
