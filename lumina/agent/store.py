"""Durable SQLite ledger for one LUMINA agent run (G3).

The database is the authority for run state, tasks, request attempts, gates
and events.  state.json / events.jsonl / errors.jsonl are exported views only,
so a damaged ledger is never silently replaced by an empty run.

Standard library only, and free of any import from the rest of LUMINA: the
control layer must stay usable when the research core is broken.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
import tempfile
import time
import uuid
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path

DB_NAME = "ledger.sqlite"
SCHEMA_VERSION = 1
SUCCESS_STATUS = "succeeded"

MAIN_STATES = (
    "INIT", "PREFLIGHT", "INPUT_READY", "SMOKE", "HUMAN_GATE_SMOKE",
    "BATCH", "FINAL_QC", "ARCHIVE", "DONE",
)
# Interruptions remember where to resume; they are never silently forgotten.
INTERRUPT_STATES = ("PAUSED", "RECOVERABLE_ERROR", "HUMAN_GATE")
FATAL_ERROR = "FATAL_ERROR"
# DONE finished the work; FATAL_ERROR must be resolved by a human, never resumed.
TERMINAL_STATES = frozenset({"DONE", FATAL_ERROR})
STATES = MAIN_STATES + INTERRUPT_STATES + (FATAL_ERROR,)


class StoreError(RuntimeError):
    """Any condition that must stop the run instead of being worked around."""


_SCHEMA_SQL = """
CREATE TABLE run (
    singleton INTEGER PRIMARY KEY CHECK (singleton = 1),
    state TEXT NOT NULL,
    resume_state TEXT,
    pause_requested INTEGER NOT NULL DEFAULT 0 CHECK (pause_requested IN (0, 1)),
    started_at REAL NOT NULL,
    updated_at REAL NOT NULL,
    spec_hash TEXT NOT NULL
);
CREATE TABLE tasks (
    task_id TEXT PRIMARY KEY,
    stage TEXT NOT NULL,
    payload TEXT NOT NULL,
    status TEXT NOT NULL,
    artifacts TEXT NOT NULL DEFAULT '[]',
    error TEXT,
    updated_at REAL NOT NULL
);
CREATE TABLE attempts (
    attempt_id TEXT PRIMARY KEY,
    task_id TEXT,
    request_hash TEXT,
    kind TEXT,
    status TEXT,
    reserved_tokens INTEGER,
    reserved_cost REAL,
    actual_tokens INTEGER,
    actual_cost REAL,
    raw_path TEXT,
    receipt TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE gates (
    gate_id TEXT PRIMARY KEY,
    kind TEXT NOT NULL,
    target TEXT NOT NULL,
    reason TEXT NOT NULL,
    status TEXT NOT NULL,
    decision TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE TABLE events (
    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
    at REAL NOT NULL,
    kind TEXT NOT NULL,
    payload TEXT NOT NULL
);
"""


@lru_cache(maxsize=1)
def _expected_schema() -> dict[str, set[str]]:
    """Table and column names the current schema promises, from one probe DB."""
    probe = sqlite3.connect(":memory:")
    try:
        probe.executescript(_SCHEMA_SQL)
        names = [row[0] for row in probe.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")]
        return {name: {c[1] for c in probe.execute(f"PRAGMA table_info({name})")} for name in names}
    finally:
        probe.close()


def _dumps(value: object) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_error_event(kind: str, payload: dict) -> bool:
    return kind.startswith("error") or bool(payload.get("error"))


def _next_states(state: str, resume_state: str | None) -> frozenset[str]:
    if state in TERMINAL_STATES:
        return frozenset()
    if state in INTERRUPT_STATES:
        allowed = set(INTERRUPT_STATES)
        if resume_state:
            allowed.add(resume_state)
        return frozenset(allowed)
    chain = {name: MAIN_STATES[index + 1] for index, name in enumerate(MAIN_STATES[:-1])}
    return frozenset({chain[state], FATAL_ERROR}) | set(INTERRUPT_STATES)


class Store:
    """The run's canonical ledger.  Create it once, then reopen it forever."""

    def __init__(self, run_dir: str | Path, create: bool = False):
        self.run_dir = Path(run_dir)
        self.path = self.run_dir / DB_NAME
        # O_EXCL decides ownership of initialization atomically: only the invocation that
        # created the file may write the schema, so a damaged or half-written ledger is
        # never adopted or reset.
        fresh = False
        if create:
            self.run_dir.mkdir(parents=True, exist_ok=True)
            try:
                os.close(os.open(self.path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o644))
                fresh = True
            except FileExistsError:
                pass
        elif not self.path.exists():
            raise StoreError(f"no ledger at {self.path}; create the run explicitly")
        # isolation_level=None: transaction() owns BEGIN/COMMIT/ROLLBACK.
        try:
            self._conn = sqlite3.connect(self.path, isolation_level=None, timeout=30.0)
        except sqlite3.Error as exc:
            raise StoreError(f"cannot open ledger {self.path}: {exc}") from exc
        self._conn.row_factory = sqlite3.Row
        try:
            if fresh:
                self._conn.executescript(_SCHEMA_SQL)
                self._conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            self._verify()
        except (sqlite3.Error, StoreError) as exc:
            self._conn.close()
            # Only ever remove a file this call created and never populated.
            if fresh and self.path.stat().st_size == 0:
                self.path.unlink(missing_ok=True)
            if isinstance(exc, StoreError):
                raise
            raise StoreError(f"ledger {self.path} is unusable: {exc}") from exc

    # -- plumbing ---------------------------------------------------------

    def _verify(self) -> None:
        report = self._conn.execute("PRAGMA integrity_check").fetchone()[0]
        if report != "ok":
            raise StoreError(f"ledger {self.path} failed integrity_check: {report}")
        version = self._conn.execute("PRAGMA user_version").fetchone()[0]
        if version != SCHEMA_VERSION:
            raise StoreError(f"ledger {self.path} has schema version {version}, expected {SCHEMA_VERSION}")
        present = {row[0] for row in self._conn.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'")}
        for table, columns in _expected_schema().items():
            if table not in present:
                raise StoreError(f"ledger {self.path} is missing table {table}")
            have = {row[1] for row in self._conn.execute(f"PRAGMA table_info({table})")}
            if have != columns:
                raise StoreError(
                    f"ledger {self.path} table {table} columns {sorted(have)} != {sorted(columns)}")

    def check_integrity(self) -> str:
        """Re-run the SQLite integrity check; raises StoreError if damaged."""
        report = self._conn.execute("PRAGMA integrity_check").fetchone()[0]
        if report != "ok":
            raise StoreError(f"ledger {self.path} failed integrity_check: {report}")
        return report

    def close(self) -> None:
        # sqlite3.Connection.close() is idempotent, so double close and __exit__ are safe.
        self._conn.close()

    def __enter__(self) -> "Store":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    @contextmanager
    def transaction(self):
        """Yield the row-mapping connection under an immediate write lock."""
        if self._conn.in_transaction:
            raise StoreError("nested transaction on the run ledger")
        self._conn.execute("BEGIN IMMEDIATE")
        try:
            yield self._conn
        except BaseException:
            self._conn.rollback()
            raise
        self._conn.commit()

    def _run(self, conn: sqlite3.Connection | None = None) -> dict:
        conn = conn or self._conn
        row = conn.execute("SELECT * FROM run WHERE singleton = 1").fetchone()
        if row is None:
            raise StoreError("run row missing; the ledger was never initialized")
        if row["state"] not in STATES or row["resume_state"] is not None and row["resume_state"] not in MAIN_STATES[:-1]:
            raise StoreError("run state or resume target is corrupt")
        if row["state"] in INTERRUPT_STATES and row["resume_state"] is None:
            raise StoreError("interrupted run has no saved resume target")
        if not isinstance(row["spec_hash"], str) or not row["spec_hash"]:
            raise StoreError("run specification identity is corrupt")
        for field in ("started_at", "updated_at"):
            if not isinstance(row[field], (int, float)) or not math.isfinite(row[field]) or row[field] <= 0:
                raise StoreError(f"run {field} is corrupt")
        return dict(row)

    @staticmethod
    def _event(conn, kind: str, payload: dict) -> None:
        conn.execute("INSERT INTO events(at, kind, payload) VALUES(?,?,?)",
                     (time.time(), kind, _dumps(payload)))

    # -- run row ----------------------------------------------------------

    def initialize(self, spec_hash: str) -> dict:
        if not spec_hash:
            raise StoreError("initialize requires a spec_hash")
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM run WHERE singleton = 1").fetchone()
            if row is not None:
                if row["spec_hash"] != spec_hash:
                    raise StoreError("run already initialized with a different spec_hash")
                return dict(row)
            now = time.time()
            conn.execute("INSERT INTO run(singleton, state, resume_state, pause_requested, started_at,"
                         " updated_at, spec_hash) VALUES(1, 'INIT', NULL, 0, ?, ?, ?)",
                         (now, now, spec_hash))
            self._event(conn, "state", {"from": None, "to": "INIT", "reason": "initialize",
                                        "stage": None, "task_id": None, "resume_state": None})
        return self._run()

    def set_state(self, state: str, reason: str, stage: str | None = None,
                  task_id: str | None = None) -> dict:
        if state not in STATES:
            raise StoreError(f"unknown state {state}")
        with self.transaction() as conn:
            current = self._run(conn)
            previous, resume = current["state"], current["resume_state"]
            if previous in TERMINAL_STATES:
                # DONE is finished work; FATAL_ERROR needs a human decision, not a resume.
                raise StoreError(f"{previous} is terminal; no transition out of it is allowed")
            if state not in _next_states(previous, resume):
                raise StoreError(f"illegal transition {previous} -> {state}")
            now = time.time()
            if state in INTERRUPT_STATES:
                # Keep the first interruption's target: PAUSED -> HUMAN_GATE must not resume to PAUSED.
                keep = resume if previous in INTERRUPT_STATES else previous
                conn.execute("UPDATE run SET state=?, resume_state=?, updated_at=? WHERE singleton=1",
                             (state, keep, now))
            else:
                keep = None
                conn.execute("UPDATE run SET state=?, resume_state=NULL, updated_at=? WHERE singleton=1",
                             (state, now))
            self._event(conn, "state", {"from": previous, "to": state, "reason": reason, "stage": stage,
                                        "task_id": task_id, "resume_state": keep})
        return self._run()

    def request_pause(self) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE run SET pause_requested = 1, updated_at = ? WHERE singleton = 1",
                         (time.time(),))
            self._event(conn, "pause_requested", {"requested": True})

    def clear_pause(self) -> None:
        with self.transaction() as conn:
            conn.execute("UPDATE run SET pause_requested = 0, updated_at = ? WHERE singleton = 1",
                         (time.time(),))
            self._event(conn, "pause_cleared", {"requested": False})

    def paused_requested(self) -> bool:
        return bool(self._run()["pause_requested"])

    # -- tasks ------------------------------------------------------------

    def put_task(self, task_id: str, stage: str, payload: dict, status: str = "pending") -> dict:
        if not task_id or not stage:
            raise StoreError("task_id and stage are required")
        encoded = _dumps(payload)
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is not None:
                if row["stage"] != stage or row["payload"] != encoded:
                    raise StoreError(f"task {task_id} already exists with a different identity")
                return self._task(row)
            conn.execute("INSERT INTO tasks(task_id, stage, payload, status, artifacts, error, updated_at)"
                         " VALUES(?,?,?,?,'[]',NULL,?)", (task_id, stage, encoded, status, time.time()))
        return self.task(task_id)

    def finish_task(self, task_id: str, status: str, artifacts: list, error=None) -> dict:
        recorded = [self._artifact(item) for item in (artifacts or [])]
        text = None if error is None else (error if isinstance(error, str) else _dumps(error))
        with self.transaction() as conn:
            row = conn.execute("SELECT status, artifacts FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
            if row is None:
                raise StoreError(f"unknown task {task_id}")
            conn.execute("UPDATE tasks SET status=?, artifacts=?, error=?, updated_at=? WHERE task_id=?",
                         (status, _dumps(recorded), text, time.time(), task_id))
            # Every completion is an audit record: a succeeded row alone cannot show which
            # bytes were ever published, and cache recovery depends on that lineage.
            self._event(conn, "task_finished", {
                "task_id": task_id, "status": status, "error": text,
                "previous_status": row["status"], "previous_artifacts": json.loads(row["artifacts"]),
                "artifacts": recorded,
            })
        return self.task(task_id)

    def task(self, task_id: str) -> dict | None:
        row = self._conn.execute("SELECT * FROM tasks WHERE task_id = ?", (task_id,)).fetchone()
        return None if row is None else self._task(row)

    def tasks(self, stage: str | None = None) -> list[dict]:
        if stage is None:
            rows = self._conn.execute("SELECT * FROM tasks ORDER BY task_id").fetchall()
        else:
            rows = self._conn.execute("SELECT * FROM tasks WHERE stage = ? ORDER BY task_id",
                                      (stage,)).fetchall()
        return [self._task(row) for row in rows]

    @staticmethod
    def _task(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["payload"] = json.loads(record["payload"])
        record["artifacts"] = json.loads(record["artifacts"])
        return record

    def _artifact(self, path: str | Path) -> dict:
        root = self.run_dir.resolve()
        target = Path(path).resolve()
        if target == root or not target.is_relative_to(root):
            raise StoreError(f"artifact {path} is outside the run directory")
        if not target.is_file():
            raise StoreError(f"artifact {path} is not an existing file")
        return {"path": target.relative_to(root).as_posix(), "sha256": _sha256(target),
                "bytes": target.stat().st_size}

    def valid_task(self, task_id: str) -> bool:
        """Re-check the artifacts themselves; a ledger row alone proves nothing."""
        record = self.task(task_id)
        # A succeeded task with nothing to re-check proves nothing: an artifact-backed
        # stage must hand over at least one file whose hash still matches.
        if record is None or record["status"] != SUCCESS_STATUS or not record["artifacts"]:
            return False
        root = self.run_dir.resolve()
        for item in record["artifacts"]:
            try:
                target = (root / item["path"]).resolve()
                if not target.is_relative_to(root) or not target.is_file():
                    return False
                if target.stat().st_size == 0 or _sha256(target) != item["sha256"]:
                    return False
            except (OSError, KeyError, TypeError):
                return False
        return True

    # -- events and gates -------------------------------------------------

    def event(self, kind: str, payload: dict) -> None:
        with self.transaction() as conn:
            self._event(conn, kind, payload)

    def create_gate(self, kind: str, target: str, reason: str) -> str:
        if not reason or not reason.strip():
            raise StoreError("a gate needs a reason")
        gate_id = uuid.uuid4().hex
        with self.transaction() as conn:
            now = time.time()
            conn.execute("INSERT INTO gates(gate_id, kind, target, reason, status, decision, created_at,"
                         " updated_at) VALUES(?,?,?,?,'pending',NULL,?,?)",
                         (gate_id, kind, target, reason, now, now))
            self._event(conn, "gate_created",
                        {"gate_id": gate_id, "kind": kind, "target": target, "reason": reason})
        return gate_id

    def gates(self) -> list[dict]:
        records = []
        for row in self._conn.execute("SELECT * FROM gates ORDER BY created_at, gate_id"):
            record = dict(row)
            record["decision"] = json.loads(record["decision"]) if record["decision"] else None
            records.append(record)
        return records

    def approve_gate(self, gate_id: str, decision: dict) -> dict:
        # Callers own the authorization; the store only keeps approvals reasoned and single-shot.
        if not isinstance(decision, dict) or not str(decision.get("reason") or "").strip():
            raise StoreError("an approval needs a nonempty reason")
        with self.transaction() as conn:
            row = conn.execute("SELECT * FROM gates WHERE gate_id = ?", (gate_id,)).fetchone()
            if row is None:
                raise StoreError(f"unknown gate {gate_id}")
            if row["status"] != "pending":
                raise StoreError(f"gate {gate_id} is already {row['status']} and cannot be re-approved")
            conn.execute("UPDATE gates SET status='approved', decision=?, updated_at=? WHERE gate_id=?",
                         (_dumps(decision), time.time(), gate_id))
            self._event(conn, "gate_approved", {"gate_id": gate_id, "decision": decision})
        return next(g for g in self.gates() if g["gate_id"] == gate_id)

    # -- views ------------------------------------------------------------

    def snapshot(self) -> dict:
        def counts(table: str) -> dict:
            rows = self._conn.execute(f"SELECT status, count(*) AS n FROM {table} GROUP BY status").fetchall()
            return {row["status"]: row["n"] for row in rows}

        money = self._conn.execute(
            "SELECT count(*) AS n, coalesce(sum(reserved_tokens), 0) AS tokens,"
            " coalesce(sum(reserved_cost), 0) AS cost, coalesce(sum(actual_tokens), 0) AS actual_tokens,"
            " coalesce(sum(actual_cost), 0) AS actual_cost FROM attempts").fetchone()
        gate_counts = counts("gates")
        task_counts = counts("tasks")
        return {
            "run": self._run(),
            "tasks": {"total": sum(task_counts.values()), "by_status": task_counts},
            "attempts": {
                "total": money["n"],
                "by_status": counts("attempts"),
                "reserved_tokens": money["tokens"], "reserved_cost": money["cost"],
                "actual_tokens": money["actual_tokens"], "actual_cost": money["actual_cost"],
            },
            "gates": {
                "total": sum(gate_counts.values()), "by_status": gate_counts,
                "pending": [g["gate_id"] for g in self.gates() if g["status"] == "pending"],
            },
        }

    def _events(self) -> list[dict]:
        return [{"event_id": row["event_id"], "at": row["at"], "kind": row["kind"],
                 "payload": json.loads(row["payload"])}
                for row in self._conn.execute("SELECT * FROM events ORDER BY event_id")]

    def export_views(self) -> dict:
        """Rewrite the human-readable views from the ledger, each file atomically."""
        self.run_dir.mkdir(parents=True, exist_ok=True)
        events = self._events()
        errors = [e for e in events if _is_error_event(e["kind"], e["payload"])]

        def write(name: str, text: str) -> None:
            with tempfile.NamedTemporaryFile("w", dir=self.run_dir, suffix=".part", delete=False,
                                             encoding="utf-8") as handle:
                handle.write(text)
                temp = Path(handle.name)
            try:
                os.replace(temp, self.run_dir / name)
            except OSError:
                temp.unlink(missing_ok=True)
                raise

        write("state.json", json.dumps(self.snapshot(), ensure_ascii=False, indent=2, default=str))
        write("events.jsonl", "".join(_dumps(e) + "\n" for e in events))
        write("errors.jsonl", "".join(_dumps(e) + "\n" for e in errors))
        return {"events": len(events), "errors": len(errors)}
