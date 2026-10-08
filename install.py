"""Bounded setup: create or reuse a Python 3.12 venv and install LUMINA dependencies.

Stdlib only. Every package operation runs as ``<target interpreter> -m pip`` so an
activated system interpreter is never touched, and no shell is involved.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import venv
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent
CORE_REQUIREMENTS = "requirements.txt"
PDF_REQUIREMENTS = "requirements-pdf.txt"
PDF_WARNING = "marker-pdf is not verified as ready by this setup"
EXAMPLE_CONFIG = "configs/my-study/project.toml"
CORE_PROBES = ("pypdf", "lumina.configuration", "lumina.agent.runtime", "lumina.cross_validation", "tiktoken")
# Published marker API (https://pypi.org/project/marker-pdf/): create_model_dict lives in
# marker.models and text_from_rendered in marker.output, not on PdfConverter. Import only;
# calling create_model_dict() would download and load model weights.
MARKER_PROBE = (
    "from marker.converters.pdf import PdfConverter;"
    "from marker.models import create_model_dict;"
    "from marker.output import text_from_rendered;"
    "assert callable(create_model_dict) and callable(text_from_rendered) and callable(PdfConverter)"
)


class InstallError(RuntimeError):
    """Fatal setup problem reported to the caller as a nonzero exit."""


def choose_venv_path(explicit=None, running_prefix=sys.prefix, running_base_prefix=sys.base_prefix,
                     repo_root=REPO_ROOT) -> Path:
    """Explicit --venv wins, then the active venv, then <repo>/.venv."""
    if explicit is not None:
        return Path(explicit).expanduser().resolve()
    if running_prefix != running_base_prefix:
        return Path(running_prefix).resolve()
    return (Path(repo_root) / ".venv").resolve()


def _unsafe_target_reason(target: Path, repo_root: Path = REPO_ROOT) -> str | None:
    """Refuse targets where a venv would clobber unrelated user data."""
    if target.parent == target:
        return "the filesystem root"
    if target == Path.home().resolve():
        return "the home directory"
    if target == Path(repo_root).resolve():
        return "the repository root"
    return None


def interpreter_path(target: Path) -> Path:
    return target / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def _venv_version(cfg: Path) -> str:
    """Read the version from pyvenv.cfg; stdlib venv writes 'version', uv writes 'version_info'."""
    values = {}
    for line in cfg.read_text(encoding="utf-8", errors="replace").splitlines():
        key, separator, value = line.partition("=")
        if separator:
            values.setdefault(key.strip(), value.strip())
    return values.get("version_info") or values.get("version") or ""


def _check_pyvenv_cfg(cfg: Path, target: Path) -> None:
    version = _venv_version(cfg)
    parts = version.split(".")
    if len(parts) < 2 or parts[0] != "3" or parts[1] != "12":
        raise InstallError(
            f"existing environment {target} is not Python 3.12 "
            f"(pyvenv.cfg reports {version or 'no version'}). "
            "Point --venv at a 3.12 environment or remove the old one."
        )


def ensure_venv(target: Path) -> Path:
    """Create the venv when absent/empty, reuse it when valid; return its interpreter."""
    reason = _unsafe_target_reason(target)
    if reason:
        raise InstallError(f"refusing to use {reason} as the virtual environment directory")
    cfg = target / "pyvenv.cfg"
    if cfg.is_file():
        _check_pyvenv_cfg(cfg, target)
    elif target.exists() and any(target.iterdir()):
        raise InstallError(
            f"{target} is not empty and is not a virtual environment; refusing to overwrite it. "
            "Choose another --venv path or clear the directory yourself."
        )
    else:
        print(f"Creating virtual environment (Python 3.12) at {target}")
        venv.EnvBuilder(with_pip=True).create(str(target))
    interpreter = interpreter_path(target)
    if not interpreter.is_file():
        raise InstallError(f"virtual environment at {target} has no interpreter at {interpreter}")
    return interpreter


def _run(command, runner=None):
    command = [str(part) for part in command]
    print("+", " ".join(command), flush=True)
    # cwd=REPO_ROOT lets the child import the local lumina package from a checkout that was
    # never pip-installed, regardless of the directory install.py was launched from.
    return (runner or subprocess.run)(command, cwd=str(REPO_ROOT))


def _install_profile(interpreter: Path, repo_root: Path, skip_marker: bool, runner) -> bool:
    """Install the pdf profile by default; fall back to core so pypdf stays usable."""
    pdf = repo_root / PDF_REQUIREMENTS
    if skip_marker:
        print("Lightweight profile requested (--skip-marker): installing the core requirements only.")
        return False
    if not pdf.is_file():
        print(f"WARNING: {PDF_REQUIREMENTS} not found; falling back to the core profile.", file=sys.stderr)
        return False
    if _run([interpreter, "-m", "pip", "install", "-r", pdf], runner).returncode == 0:
        return True
    print(f"WARNING: marker-pdf installation failed ({PDF_REQUIREMENTS});"
          " falling back to the core profile.", file=sys.stderr)
    return False


def _verify_environment(interpreter: Path, runner) -> None:
    """Fail loudly on a broken dependency set or an unusable core package."""
    if _run([interpreter, "-m", "pip", "check"], runner).returncode != 0:
        raise InstallError("pip check reported a broken environment")
    for module in CORE_PROBES:
        if _run([interpreter, "-c", f"import {module}"], runner).returncode != 0:
            raise InstallError(f"import probe failed for {module}")


def install_dependencies(interpreter: Path, repo_root: Path, skip_marker: bool,
                        runner=None) -> bool:
    """Install requirements, degrading honestly from marker-pdf to the pypdf fallback.

    Returns True only when marker-pdf is installed and importable.
    """
    marker_ok = _install_profile(interpreter, repo_root, skip_marker, runner)
    if not marker_ok:
        core = repo_root / CORE_REQUIREMENTS
        if _run([interpreter, "-m", "pip", "install", "-r", core], runner).returncode != 0:
            raise InstallError(f"core requirements installation failed ({CORE_REQUIREMENTS})")
    _verify_environment(interpreter, runner)
    if marker_ok and _run([interpreter, "-c", MARKER_PROBE], runner).returncode != 0:
        marker_ok = False
    if not marker_ok:
        print(f"NOTE: {PDF_WARNING}. The pypdf fallback is ready. "
              'Existing Marker packages are not removed; set pdf.backend="pypdf", '
              'pdf.fallback="none" for text-only processing.', file=sys.stderr)
    return marker_ok


def run_hint(interpreter: Path, repo_root: Path = REPO_ROOT,
             config: str = EXAMPLE_CONFIG) -> str:
    """Printable no-activation run command; PowerShell needs '&' to run a quoted path."""
    pipeline = repo_root / "run_pipeline.py"
    invocation = f'"{interpreter}" "{pipeline}" --config "{repo_root / config}"'
    return f"& {invocation}" if sys.platform == "win32" else invocation


def _install(args) -> int:
    if sys.version_info[:2] != (3, 12):
        raise InstallError(f"Python 3.12 is required to run install.py "
                          f"(running {sys.version.split()[0]})")
    target = choose_venv_path(args.venv)
    interpreter = ensure_venv(target)
    print(f"Target interpreter: {interpreter}")
    marker_ok = install_dependencies(interpreter, REPO_ROOT, args.skip_marker)
    print()
    print(f"PDF packages: {'marker-pdf ready' if marker_ok else 'pypdf fallback ready; Marker not verified'}")
    if marker_ok:
        print('Marker package/API ready. OCR/layout weights are prepared on first PDF use and count toward max_runtime.')
    print("Run the pipeline with (no activation needed; point --config at your own study):")
    print(f"  {run_hint(interpreter)}")
    return 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="Install LUMINA dependencies into a Python 3.12 venv.")
    parser.add_argument("--venv", type=Path, default=None,
                        help="venv directory (default: the active venv, else <repo>/.venv)")
    parser.add_argument("--skip-marker", action="store_true",
                        help="lightweight profile: install the core requirements and skip marker-pdf "
                             "(faster and smaller, but still needs package access)")
    args = parser.parse_args(argv)
    try:
        return _install(args)
    except (InstallError, OSError) as exc:
        print(f"error: {exc or exc.__class__.__name__}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
