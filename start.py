"""Interactive, stdlib-only launcher for the LUMINA configuration wizard."""

from __future__ import annotations

import argparse
from datetime import UTC, datetime
import json
import os
import queue
import subprocess
import sys
import threading
import uuid
from pathlib import Path

import install
from lumina.ui_support import read_preferences, redact as _shared_redact


REPO_ROOT = Path(__file__).resolve().parent
PYTHON_VERSION = (3, 12)
PREFERENCES_PATH = REPO_ROOT / ".lumina-ui.json"
INSTALL_HEARTBEAT_SECONDS = 5.0
INSTALL_PROFILE_RECEIPT = ".lumina-install-profile.json"


class StartupError(RuntimeError):
    """A setup failure that must prevent wizard startup."""


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Start the interactive LUMINA wizard.")
    parser.add_argument(
        "--venv",
        type=Path,
        help="Python 3.12 virtual environment (default: active venv, otherwise <repo>/.venv)",
    )
    parser.add_argument("--config", type=Path, help="optional study configuration path")
    parser.add_argument("--lang", choices=("zh", "en"), default=None)
    return parser


def _redact(value: object) -> str:
    return str(_shared_redact(str(value)))


def _write_log(path: Path, value: object) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(_redact(value))


def _private_setup_dir() -> Path:
    parent = REPO_ROOT / "runs" / "_setup"
    parent.mkdir(parents=True, exist_ok=True)
    setup_dir = parent / f"{datetime.now(UTC):%Y%m%dT%H%M%SZ}-{uuid.uuid4()}"
    setup_dir.mkdir(mode=0o700)
    try:
        setup_dir.chmod(0o700)
    except OSError:
        pass
    return setup_dir


def _is_tty() -> bool:
    return bool(sys.stdin.isatty() and sys.stdout.isatty())


def _startup_language(explicit: str | None) -> str:
    if explicit in {"zh", "en"}:
        return explicit
    preference = read_preferences(PREFERENCES_PATH)
    language = preference.get("language")
    return language if language in {"zh", "en"} else "zh"


def _say(language: str, zh: str, en: str) -> None:
    print(zh if language == "zh" else en, flush=True)


def _resolve_config(config: Path | None, original_cwd: Path) -> Path | None:
    if config is None:
        return None
    config = config.expanduser()
    return (config if config.is_absolute() else original_cwd / config).resolve()


def _display_command(command: list[object]) -> str:
    return " ".join(str(part) for part in command)


def _wait_notice(language: str) -> str:
    return ("安装仍在进行中，请继续等待；输出可能会在下载阶段暂时静默。"
            if language == "zh" else
            "Installation is still running; download stages may be temporarily quiet.")


def _record_output(line: str, log_path: Path) -> None:
    print(_redact(line), end="", flush=True)
    _write_log(log_path, line)


def _drain_output(events: queue.Queue, log_path: Path) -> None:
    while True:
        try:
            kind, value = events.get_nowait()
        except queue.Empty:
            return
        if kind == "line":
            _record_output(value, log_path)
        elif kind == "error":
            _write_log(log_path, f"output reader error: {type(value).__name__}: {value}\n")


def _terminate_process(process: subprocess.Popen, log_path: Path) -> None:
    if process.poll() is None:
        try:
            process.terminate()
            _write_log(log_path, "child installer terminate requested\n")
        except OSError as exc:
            _write_log(log_path, f"child installer terminate failed: {type(exc).__name__}: {exc}\n")
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            _write_log(log_path, "child installer did not terminate; killing child\n")
            process.kill()
            process.wait()
    _write_log(log_path, f"child installer exit code: {process.poll()}\n")


def _stream_process(command: list[object], log_path: Path, *, cwd: Path | str = REPO_ROOT,
                    language: str = "zh", heartbeat_seconds: float = INSTALL_HEARTBEAT_SECONDS) -> subprocess.CompletedProcess:
    command = [str(part) for part in command]
    _write_log(log_path, f"$ {_display_command(command)}\n")
    environment = os.environ.copy()
    environment["PYTHONIOENCODING"] = "utf-8"
    try:
        process = subprocess.Popen(
            command,
            cwd=str(cwd),
            env=environment,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            bufsize=1,
            shell=False,
        )
    except OSError as exc:
        _write_log(log_path, f"{type(exc).__name__}: {exc}\n")
        raise

    events: queue.Queue = queue.Queue()

    def read_output() -> None:
        try:
            if process.stdout is not None:
                for line in iter(process.stdout.readline, ""):
                    events.put(("line", line))
        except OSError as exc:
            events.put(("error", exc))
        finally:
            events.put(("done", None))

    reader = threading.Thread(target=read_output, name="lumina-installer-output", daemon=True)
    reader.start()
    try:
        while True:
            try:
                kind, value = events.get(timeout=heartbeat_seconds)
            except queue.Empty:
                notice = _wait_notice(language)
                print(notice, flush=True)
                _write_log(log_path, f"{notice}\n")
                continue
            if kind == "line":
                _record_output(value, log_path)
            elif kind == "error":
                raise value
            elif kind == "done":
                break
        reader.join()
        returncode = process.wait()
        _write_log(log_path, f"[exit code {returncode}]\n")
        return subprocess.CompletedProcess(command, returncode)
    except BaseException as exc:
        _terminate_process(process, log_path)
        reader.join(timeout=5)
        _drain_output(events, log_path)
        if isinstance(exc, KeyboardInterrupt):
            _write_log(log_path, "installer interrupted; child terminated and output retained\n")
        else:
            _write_log(log_path, f"installer aborted: {type(exc).__name__}: {_redact(exc)}\n")
        raise
    finally:
        if process.stdout is not None:
            process.stdout.close()


def _probe(command: list[object], startup_log: Path) -> subprocess.CompletedProcess:
    command = [str(part) for part in command]
    result = subprocess.run(
        command,
        cwd=str(REPO_ROOT),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        shell=False,
    )
    output = getattr(result, "stdout", "")
    if output:
        _write_log(startup_log, output)
    return result


def _core_modules() -> tuple[str, ...]:
    return tuple(dict.fromkeys((*install.CORE_PROBES, "tomlkit")))


def _core_is_healthy(interpreter: Path, startup_log: Path) -> bool:
    result = _probe([interpreter, "-m", "pip", "check"], startup_log)
    if result.returncode != 0:
        _write_log(startup_log, "core validation failed: pip check\n")
        return False
    for module in _core_modules():
        result = _probe([interpreter, "-c", f"import {module}"], startup_log)
        if result.returncode != 0:
            _write_log(startup_log, f"core validation failed: import {module}\n")
            return False
    return True


def _marker_is_healthy(interpreter: Path, startup_log: Path) -> bool:
    result = _probe([interpreter, "-c", install.MARKER_PROBE], startup_log)
    if result.returncode != 0:
        _write_log(startup_log, "marker validation failed\n")
        return False
    return True


def _install_profile(target: Path) -> str | None:
    try:
        data = json.loads((target / INSTALL_PROFILE_RECEIPT).read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return None
    profile = data.get("profile") if isinstance(data, dict) else None
    return profile if isinstance(profile, str) and profile in {"marker", "pypdf-fallback"} else None


def _save_install_profile(target: Path, marker_ready: bool) -> None:
    (target / INSTALL_PROFILE_RECEIPT).write_text(
        json.dumps({"profile": "marker" if marker_ready else "pypdf-fallback"}) + "\n",
        encoding="utf-8",
    )


def _ensure_pip(interpreter: Path, startup_log: Path, installation_log: Path, language: str) -> None:
    if _probe([interpreter, "-m", "pip", "--version"], startup_log).returncode == 0:
        return
    _say(language, "目标环境缺少 pip，正在仅在目标环境中引导安装。请稍候。",
         "pip is missing; bootstrapping it only in the target environment. Please wait.")
    _write_log(startup_log, "pip missing; running target interpreter -m ensurepip\n")
    result = _stream_process([interpreter, "-m", "ensurepip", "--upgrade"], installation_log, language=language)
    if result.returncode != 0 or _probe([interpreter, "-m", "pip", "--version"], startup_log).returncode != 0:
        raise StartupError(f"target environment pip bootstrap failed (exit code {result.returncode})")


def _launch_wizard(interpreter: Path, setup_dir: Path, config: Path | None, lang: str | None,
                   startup_log: Path) -> int:
    command: list[object] = [interpreter, "-m", "lumina.wizard", "--setup-log", setup_dir]
    if config is not None:
        command.extend(("--config", config))
    if lang is not None:
        command.extend(("--lang", lang))
    _write_log(startup_log, f"launching wizard: {_display_command(command)}\n")
    try:
        result = subprocess.run(
            [str(part) for part in command],
            cwd=str(REPO_ROOT),
            stdin=None,
            stdout=None,
            stderr=None,
            shell=False,
        )
    except OSError as exc:
        _write_log(setup_dir / "errors.log", f"{type(exc).__name__}: {exc}\n")
        print(f"error: {_redact(exc)}", file=sys.stderr)
        return 1
    _write_log(startup_log, f"wizard exit code: {result.returncode}\n")
    if result.returncode:
        _write_log(setup_dir / "errors.log", f"wizard exited with code {result.returncode}\n")
    return result.returncode


def _noninteractive_message(language: str) -> int:
    message = (
        "start.py 需要交互式终端，已在环境准备前停止。\n"
        "请改用旧入口：python run_pipeline.py --config PATH"
        if language == "zh"
        else "start.py requires an interactive terminal and stopped before environment setup.\n"
        "Use the legacy command instead: python run_pipeline.py --config PATH"
    )
    print(message, file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    language = _startup_language(args.lang)
    if not _is_tty():
        return _noninteractive_message(language)
    if sys.version_info[:2] != PYTHON_VERSION:
        message = "start.py 需要 Python 3.12。" if language == "zh" else "Python 3.12 is required to run start.py."
        print(message, file=sys.stderr)
        return 1

    setup_dir = _private_setup_dir()
    startup_log = setup_dir / "startup.log"
    installation_log = setup_dir / "installation.log"
    errors_log = setup_dir / "errors.log"
    for log_path in (installation_log, errors_log):
        _write_log(log_path, "")
    original_cwd = Path.cwd().resolve()
    _write_log(startup_log, f"launcher started; cwd={original_cwd}\n")

    try:
        target = Path(install.choose_venv_path(args.venv))
        existing_environment = (target / "pyvenv.cfg").is_file()
        _write_log(startup_log, f"target environment: {target}\n")
        _say(language, "正在准备目标 Python 3.12 环境，请稍候。",
             "Preparing the target Python 3.12 environment. Please wait.")
        interpreter = Path(install.ensure_venv(target))
        _write_log(startup_log, f"target interpreter: {interpreter}\n")
        _ensure_pip(interpreter, startup_log, installation_log, language)

        install_required = True
        if existing_environment and _core_is_healthy(interpreter, startup_log):
            install_required = False
            profile = _install_profile(target)
            if profile == "pypdf-fallback":
                _say(language, "目标环境已记录为 pypdf 后备，跳过重复安装；如需重试 Marker，请删除环境内的安装记录。",
                     "This environment is recorded as the pypdf fallback; skipping repeated installation. Remove the venv-local profile receipt to retry Marker.")
                _write_log(startup_log, "recorded pypdf fallback; dependency installation skipped\n")
            elif _marker_is_healthy(interpreter, startup_log):
                _say(language, "目标环境健康，跳过依赖安装。",
                     "Target environment is healthy; skipping dependency installation.")
                _write_log(startup_log, "healthy existing environment; dependency installation skipped\n")
            else:
                install_required = True
        if install_required:
            _say(language, "正在安装缺少的依赖，请稍候；安装器会报告进度。",
                 "Installing missing dependencies. Please wait; the installer reports progress.")
            marker_ready = install.install_dependencies(
                interpreter,
                REPO_ROOT,
                False,
                runner=lambda command, **kwargs: _stream_process(
                    command,
                    installation_log,
                    cwd=kwargs.get("cwd", REPO_ROOT),
                    language=language,
                ),
            )
            if not _core_is_healthy(interpreter, startup_log):
                raise StartupError("core validation failed after dependency installation")
            _save_install_profile(target, marker_ready)

        return _launch_wizard(
            interpreter,
            setup_dir,
            _resolve_config(args.config, original_cwd),
            args.lang,
            startup_log,
        )
    except KeyboardInterrupt:
        _write_log(errors_log, "launcher interrupted; child processes were terminated where owned\n")
        print("启动已中断，子进程已终止。" if language == "zh" else
              "Startup interrupted; owned child processes were terminated.", file=sys.stderr)
        return 130
    except (install.InstallError, OSError, StartupError) as exc:
        message = f"{type(exc).__name__}: {_redact(exc)}\n"
        _write_log(errors_log, message)
        print(f"error: {_redact(exc)}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
