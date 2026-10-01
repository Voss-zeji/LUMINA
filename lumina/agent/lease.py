"""OS-owned run lock: a stale heartbeat never permits takeover of a live worker."""
from __future__ import annotations

import ctypes
import json
import os
import time
import uuid
from pathlib import Path


def process_identity() -> str:
    if os.name == "nt":
        from ctypes import wintypes
        kernel = ctypes.WinDLL("kernel32", use_last_error=True)
        kernel.GetCurrentProcess.restype = wintypes.HANDLE
        kernel.GetProcessTimes.argtypes = [wintypes.HANDLE, *(ctypes.POINTER(wintypes.FILETIME),) * 4]
        creation, exit_time, system, user = (wintypes.FILETIME() for _ in range(4))
        if not kernel.GetProcessTimes(kernel.GetCurrentProcess(), ctypes.byref(creation), ctypes.byref(exit_time),
                                      ctypes.byref(system), ctypes.byref(user)):
            raise OSError(ctypes.get_last_error(), "cannot identify process creation time")
        return str((creation.dwHighDateTime << 32) | creation.dwLowDateTime)
    stat = Path(f"/proc/{os.getpid()}/stat")
    if stat.exists():
        return stat.read_text().rsplit(")", 1)[1].split()[19]
    return f"{os.getpid()}-{uuid.uuid4().hex}"  # OS file lock still owns exclusion on non-/proc platforms.


class RunLease:
    def __init__(self, run_dir: str | Path):
        self.path = Path(run_dir) / ".worker.lock"
        self.handle = None
        self.last_heartbeat = 0.
        self.owner = dict(pid=os.getpid(), process_identity=process_identity(), owner_id=uuid.uuid4().hex)

    def __enter__(self):
        self.handle = os.fdopen(os.open(self.path, os.O_RDWR | os.O_CREAT, 0o600), "r+b")
        try:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self.handle.close()
            self.handle = None
            raise RuntimeError("a worker owns this run; heartbeat expiry does not authorize takeover") from exc
        self.handle.seek(0)
        self.handle.write(b"#")
        self.handle.flush()
        self.touch(force=True)
        return self

    def touch(self, force=False):
        if self.handle is None:
            raise RuntimeError("worker lease is not held")
        now = time.time()
        if not force and now - self.last_heartbeat < 5:
            return
        self.last_heartbeat = now
        self.handle.seek(1)
        self.handle.write(json.dumps(self.owner | {"heartbeat": now}).encode("utf-8"))
        self.handle.truncate()
        self.handle.flush()

    def __exit__(self, *_):
        if self.handle is not None:
            self.handle.seek(0)
            if os.name == "nt":
                import msvcrt
                msvcrt.locking(self.handle.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.handle, fcntl.LOCK_UN)
            self.handle.close()
            self.handle = None
