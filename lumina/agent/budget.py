"""Budget reservation and attempt accounting for one LUMINA run (G4).

Every real HTTP attempt gets a reservation before it is dispatched.  The
SQLite ledger is the only accounting authority, so nothing here trusts
in-memory state, and a resumed run never resets a budget.  Money that cannot
be measured stays held: unknown usage is never recorded as zero.

Two columns per attempt carry the accounting.  The reserved pair always holds
the conservative maximum taken from the frozen pricing bounds; the actual pair
is filled in only when the provider reports usage.  One derived classification
decides whether a reservation is still held or settled, so a settled attempt
is counted exactly once, never as reservation AND actual:

  settled -> status rejected, or received with a known actual.
  held    -> reserved, dispatched, unknown, or received whose usage is
             unknown and therefore keeps the conservative hold.

totals()["calls"] counts real HTTP attempts and is derived from the dispatch
evidence stored in the receipt, not from the status alone: an attempt released
before dispatch never spent a call, while a dispatched 429 did.

A settlement that breaks its own reservation also writes a durable stop marker
into that receipt.  Over-budget bookkeeping is derived from the frozen pricing
bounds, so usage above them cannot be trusted to stay inside the run budget;
the ledger therefore stops spending until a new run, while the real actuals stay
recorded uncapped.

A parent-created immutable import receipt may arrive as status "imported": the
response was already paid for in another run, so it is reused for zero new HTTP
calls, blocks any fresh reservation of the same request, and its copied usage is
reported as provenance only, never charged to this run's budget again.

Standard library only, and free of any import from the rest of LUMINA.
"""
from __future__ import annotations

import hashlib
import json
import math
import time
import uuid

from .store import Store, StoreError

MAX_ATTEMPTS_PER_REQUEST = 3
KINDS = frozenset({"chat", "embedding", "advisor"})
USAGE_FIELDS = ("input_tokens", "output_tokens")
# A request is reusable only when a durable response exists for it, either paid
# for in this run or copied from a closed one.
CACHED_STATUSES = ("received", "imported")


class BudgetError(RuntimeError):
    """Base class for conditions the control layer must not work around."""


class BudgetExceeded(BudgetError):
    """A reservation or a settlement would break a frozen limit."""


class UnknownRequest(BudgetError):
    """The outcome of a prior request is unknown; replay is not safe."""


class AttemptLimit(BudgetError):
    """The per task/request automatic attempt ceiling is already reached."""


def _count(value, name: str, strict: bool = False) -> int:
    """A strict, nonnegative integer: bools, floats, inf and NaN are refused."""
    if type(value) is not int:
        raise BudgetExceeded(f"{name} must be an int, got {type(value).__name__}")
    if value < 0:
        raise BudgetExceeded(f"{name} must be nonnegative")
    if strict and value == 0:
        raise BudgetExceeded(f"{name} must be positive")
    return value


def _rate(value, name: str) -> float:
    """A finite nonnegative money rate; fractional dollars per million are normal."""
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise BudgetExceeded(f"{name} must be a finite number")
    if value < 0:
        raise BudgetExceeded(f"{name} must be nonnegative")
    return float(value)


def _known_usage(usage):
    """Return (input, output) only when both counters are known integers."""
    if not isinstance(usage, dict):
        return None
    values = []
    for field in USAGE_FIELDS:
        value = usage.get(field)
        if type(value) is not int or value < 0:
            return None
        values.append(value)
    return values[0], values[1]


def _fingerprint(value) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _status_list(statuses) -> str:
    """Quote a fixed status tuple for an IN (...) clause."""
    return ",".join("'" + name + "'" for name in statuses)


def _pricing_for(spec: dict, model: str) -> dict:
    """Frozen rates and token bounds for a model, validated like contracts.py."""
    pricing = spec.get("pricing") if isinstance(spec, dict) else None
    if not isinstance(pricing, dict) or not isinstance(pricing.get(model), dict):
        raise BudgetExceeded(f"no frozen pricing for model {model}")
    entry = pricing[model]
    for field in ("input_per_million", "output_per_million"):
        _rate(entry.get(field), f"pricing {model}.{field}")
    for field in ("max_input_tokens", "max_output_tokens"):
        _count(entry.get(field), f"pricing {model}.{field}")
    return entry


def _receipt_of(row) -> dict:
    """Decode the canonical receipt, refusing a damaged one outright.

    The receipt carries the dispatch evidence and the stop marker, so quietly
    degrading a malformed one to {} would erase real calls from the accounting
    and ignore a settlement that stopped the run.  That is exactly the silent
    underpayment this ledger exists to prevent, so the damage is raised instead
    and every caller stops before spending.
    """
    if not row["receipt"]:
        raise StoreError(f"attempt {row['attempt_id']} has no canonical receipt")
    try:
        value = json.loads(row["receipt"])
    except (TypeError, ValueError) as exc:
        raise StoreError(
            f"attempt {row['attempt_id']} has an unreadable receipt: {exc}") from exc
    if not isinstance(value, dict):
        raise StoreError(f"attempt {row['attempt_id']} has a receipt that is not an object")
    if not isinstance(value.get("model"), str) or not value["model"].strip():
        raise StoreError(f"attempt {row['attempt_id']} has no receipted model")
    status = row["status"]
    if status in {"dispatched", "unknown", "received"}:
        dispatched_at = value.get("dispatched_at")
        if isinstance(dispatched_at, bool) or not isinstance(dispatched_at, (int, float)) or not math.isfinite(dispatched_at) or dispatched_at <= 0:
            raise StoreError(f"attempt {row['attempt_id']} lost its dispatch evidence")
    if status == "imported" and not isinstance(value.get("imported_from"), dict):
        raise StoreError(f"attempt {row['attempt_id']} lost import provenance")
    return value


def _read_receipt(row) -> dict:
    """Best-effort decode for caller-supplied receipts, as cached() checks.

    cached() has no transaction to roll back, so it reports the damage as an
    UnknownRequest, which is the control layer's "this may already have been
    paid, do not resend" signal.
    """
    if not row["receipt"]:
        return {}
    try:
        value = json.loads(row["receipt"])
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}


def _dispatched(receipt: dict) -> bool:
    """Dispatch evidence, which survives a later status change to rejected."""
    return receipt.get("dispatched_at") is not None


def _totals(conn) -> dict:
    """Aggregate settled and held accounting from every attempt row."""
    rows = conn.execute(
        "SELECT attempt_id, status, reserved_tokens, reserved_cost, actual_tokens, actual_cost,"
        " receipt"
        " FROM attempts").fetchall()
    known_tokens = held_tokens = 0
    known_cost = held_cost = 0.0
    calls = attempts = imports = unknown_usage = unknown_requests = 0
    for row in rows:
        attempts += 1
        status = row["status"]
        receipt = _receipt_of(row)
        if status == "imported":
            # Provenance copied from another run's immutable receipt: it was
            # already paid there, so it costs no call and no budget here.  The
            # copied usage is history, not this run's spend, so charging it again
            # would invent cost out of a source run that is already closed.
            imports += 1
            continue
        if _dispatched(receipt):
            calls += 1
        if status == "rejected":
            # Confirmed not executed: nothing was spent, whatever the status was.
            known_tokens += row["actual_tokens"] or 0
            known_cost += row["actual_cost"] or 0.0
        elif row["actual_tokens"] is not None:
            # Known tokens.  A reconciled or settled request may still have
            # unknown money, in which case the conservative hold stays for cost.
            known_tokens += row["actual_tokens"]
            if row["actual_cost"] is None:
                held_cost += row["reserved_cost"] or 0.0
            else:
                known_cost += row["actual_cost"]
        else:
            # Usage unknown: the whole conservative reservation stays held.
            held_tokens += row["reserved_tokens"] or 0
            held_cost += row["reserved_cost"] or 0.0
            if status == "received":
                unknown_usage += 1
        if status == "unknown":
            unknown_requests += 1
    return {
        "attempts": attempts, "calls": calls, "known_tokens": known_tokens,
        "held_tokens": held_tokens, "known_cost": known_cost, "held_cost": held_cost,
        "unknown_usage": unknown_usage, "unknown_requests": unknown_requests,
        "imports": imports,
    }


class Budget:
    """Frozen budget enforcement and attempt ledger on top of a run's Store."""

    def __init__(self, store: Store, spec: dict):
        budget = spec.get("budget") if isinstance(spec, dict) else None
        if not isinstance(budget, dict):
            raise BudgetError("specification carries no budget")
        for field in ("max_calls", "max_tokens"):
            _count(budget.get(field), f"budget.{field}", strict=True)
        for field in ("max_cost", "max_runtime"):
            value = budget.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise BudgetError(f"budget.{field} must be a finite number")
            if value <= 0:
                raise BudgetError(f"budget.{field} must be positive")
        self.store = store
        self.spec = spec
        self.budget = {**budget,
                       "max_calls": budget["max_calls"], "max_tokens": budget["max_tokens"],
                       "max_cost": float(budget["max_cost"]),
                       "max_runtime": float(budget["max_runtime"])}

    # -- internal helpers -------------------------------------------------

    def _started_at(self, conn) -> float:
        row = conn.execute("SELECT started_at FROM run WHERE singleton = 1").fetchone()
        if row is None:
            raise StoreError("run row missing; the ledger was never initialized")
        return float(row["started_at"])

    def _open_slots(self, conn) -> int:
        """Undispatched reservations still holding a max_calls slot."""
        row = conn.execute("SELECT COUNT(*) AS n FROM attempts WHERE status = 'reserved'").fetchone()
        return int(row["n"])

    @staticmethod
    def _stop_reason(conn):
        """Why this ledger is closed to new spending, or None.

        A settlement that broke its own reservation proves the frozen pricing
        bound was too low, so the actual cost cannot be trusted to stay inside
        the run budget.  Such an attempt is recorded uncapped and closes the
        ledger: nothing else may spend until a human reconciles it and starts a
        new run.  Reading the durable receipt also survives a reopen.
        """
        for row in conn.execute("SELECT attempt_id, status, receipt FROM attempts"):
            reason = _receipt_of(row).get("stop_reason")
            if reason:
                return f"attempt {row['attempt_id']} stopped this run: {reason}"
        return None

    def _enforce(self, conn, extra_calls: int = 0, extra_tokens: int = 0,
                 extra_cost: float = 0.0, attempt_id: str = None) -> dict:
        """Check the four frozen limits against the ledger plus a pending delta.

        An undispatched reservation still occupies a max_calls slot, so two
        concurrent reservations cannot both pass a max_calls of one; the slot
        returns when the reservation is cancelled or reclaimed.
        """
        totals = _totals(conn)
        runtime = time.time() - self._started_at(conn)
        pending_calls = totals["calls"] + self._open_slots(conn)
        pending_tokens = totals["known_tokens"] + totals["held_tokens"]
        pending_cost = totals["known_cost"] + totals["held_cost"]
        where = f" for attempt {attempt_id}" if attempt_id else ""
        if pending_calls + extra_calls > self.budget["max_calls"]:
            raise BudgetExceeded(
                f"max_calls {self.budget['max_calls']} reached{where}: "
                f"{pending_calls} counted + {extra_calls} requested")
        if pending_tokens + extra_tokens > self.budget["max_tokens"]:
            raise BudgetExceeded(
                f"max_tokens {self.budget['max_tokens']} reached{where}: "
                f"{pending_tokens} counted + {extra_tokens} requested")
        if pending_cost + extra_cost > math.nextafter(self.budget["max_cost"], math.inf):
            raise BudgetExceeded(
                f"max_cost {self.budget['max_cost']} reached{where}: "
                f"{pending_cost} counted + {extra_cost} requested")
        if runtime > self.budget["max_runtime"]:
            raise BudgetExceeded(
                f"max_runtime {self.budget['max_runtime']}s exceeded{where}: {runtime:.1f}s used")
        return totals

    def _attempt(self, conn, attempt_id: str):
        row = conn.execute("SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
        if row is None:
            raise UnknownRequest(f"unknown attempt {attempt_id}")
        return row

    @staticmethod
    def _write(conn, attempt_id: str, now: float, **columns) -> None:
        assignments = ["updated_at = ?"]
        values = [now]
        for key, value in columns.items():
            assignments.append(key + " = ?")
            values.append(value)
        values.append(attempt_id)
        conn.execute("UPDATE attempts SET " + ", ".join(assignments) + " WHERE attempt_id = ?", values)

    @staticmethod
    def _encode(receipt: dict) -> str:
        return json.dumps(receipt, ensure_ascii=False, sort_keys=True, default=str)

    @staticmethod
    def _record_event(conn, kind: str, payload: dict) -> None:
        # The store's own event table, so this audit trail is exported with the rest.
        conn.execute("INSERT INTO events(at, kind, payload) VALUES(?,?,?)",
                     (time.time(), kind,
                      json.dumps(payload, ensure_ascii=False, sort_keys=True, default=str)))

    # -- public API -------------------------------------------------------

    def reserve(self, task_id: str, request_hash: str, kind: str, model: str,
                input_tokens: int, output_tokens: int) -> str:
        """Reserve the frozen conservative maximum for one real attempt."""
        if not task_id or not request_hash:
            raise BudgetError("task_id and request_hash are required")
        if kind not in KINDS:
            raise BudgetError(f"unknown request kind {kind!r}")
        requested_in = _count(input_tokens, "input_tokens")
        requested_out = _count(output_tokens, "output_tokens")
        price = _pricing_for(self.spec, model)
        # The frozen bounds are the most the provider may charge.  A request
        # above them is refused rather than quietly reserved at a larger size.
        if requested_in > price["max_input_tokens"]:
            raise BudgetExceeded(
                f"input_tokens {requested_in} exceeds the frozen max_input_tokens "
                f"{price['max_input_tokens']} for {model}")
        if requested_out > price["max_output_tokens"]:
            raise BudgetExceeded(
                f"output_tokens {requested_out} exceeds the frozen max_output_tokens "
                f"{price['max_output_tokens']} for {model}")
        reserved_tokens = price["max_input_tokens"] + price["max_output_tokens"]
        reserved_cost = (price["max_input_tokens"] * price["input_per_million"]
                         + price["max_output_tokens"] * price["output_per_million"]) / 1000000.0
        attempt_id = uuid.uuid4().hex
        with self.store.transaction() as conn:
            prior = conn.execute(
                "SELECT status, COUNT(*) AS n FROM attempts WHERE task_id = ? AND request_hash = ?"
                " GROUP BY status", (task_id, request_hash)).fetchall()
            states = {row["status"]: row["n"] for row in prior}
            # Any live attempt for this pair forbids a second reservation: a
            # still-reserved slot may dispatch at any moment, and dispatched,
            # unknown, received and imported outcomes must never be replayed.
            for status in ("reserved", "dispatched", "unknown", "received", "imported"):
                if states.get(status):
                    raise UnknownRequest(
                        f"task {task_id} request {request_hash} already has "
                        f"{states[status]} {status} attempt(s)")
            used = sum(states.values())
            if used >= MAX_ATTEMPTS_PER_REQUEST:
                raise AttemptLimit(
                    f"task {task_id} request {request_hash} used {used} attempts, "
                    f"ceiling {MAX_ATTEMPTS_PER_REQUEST}")
            self._enforce(conn, extra_calls=1, extra_tokens=reserved_tokens,
                          extra_cost=reserved_cost, attempt_id=attempt_id)
            stopped = self._stop_reason(conn)
            if stopped:
                raise BudgetExceeded(
                    f"{stopped}; {task_id}/{request_hash} cannot be reserved until a new run")
            now = time.time()
            receipt = {"model": model,
                       "requested": {"input_tokens": requested_in, "output_tokens": requested_out}}
            conn.execute(
                "INSERT INTO attempts(attempt_id, task_id, request_hash, kind, status,"
                " reserved_tokens, reserved_cost, actual_tokens, actual_cost, raw_path, receipt,"
                " created_at, updated_at) VALUES(?,?,?,?,'reserved',?,?,NULL,NULL,NULL,?,?,?)",
                (attempt_id, task_id, request_hash, kind, reserved_tokens, reserved_cost,
                 self._encode(receipt), now, now))
            self._record_event(conn, "budget_reserved",
                               {"attempt_id": attempt_id, "task_id": task_id,
                                "request_hash": request_hash, "kind": kind, "model": model,
                                "reserved_tokens": reserved_tokens,
                                "reserved_cost": reserved_cost})
        return attempt_id

    def dispatch(self, attempt_id: str) -> dict:
        """reserved -> dispatched in one transaction, right before real HTTP."""
        with self.store.transaction() as conn:
            row = self._attempt(conn, attempt_id)
            if row["status"] != "reserved":
                raise StoreError(f"attempt {attempt_id} is {row['status']}, not reserved")
            # The reservation already holds this call slot, so dispatch spends it
            # rather than adding one; a sole reservation may still dispatch at one.
            self._enforce(conn, attempt_id=attempt_id)
            receipt = _receipt_of(row)
            now = time.time()
            receipt["dispatched_at"] = now
            self._write(conn, attempt_id, now, status="dispatched", receipt=self._encode(receipt))
            self._record_event(conn, "budget_dispatched",
                               {"attempt_id": attempt_id, "task_id": row["task_id"],
                                "request_hash": row["request_hash"]})
        return self.attempt(attempt_id)

    def receive(self, attempt_id: str, response: dict, usage) -> dict:
        """Durably record the full response and usage, then report any overshoot.

        The receipt is committed and only then is any budget error raised, so a
        paid response is never lost because the run tripped a ceiling, and usage
        is recorded uncapped rather than clamped to the reservation.
        """
        if not isinstance(response, dict):
            raise BudgetError("response must be a dict to be receipted")
        if usage is not None and not isinstance(usage, dict):
            raise BudgetError("usage must be a dict or None")
        known = _known_usage(usage)
        detail = None
        with self.store.transaction() as conn:
            row = self._attempt(conn, attempt_id)
            if row["status"] not in ("dispatched", "unknown"):
                raise StoreError(f"cannot receive an attempt in status {row['status']}")
            receipt = _receipt_of(row)
            price = _pricing_for(self.spec, receipt.get("model", ""))
            actual_tokens = actual_cost = None
            if known is not None:
                actual_tokens = known[0] + known[1]
                actual_cost = (known[0] * price["input_per_million"]
                               + known[1] * price["output_per_million"]) / 1000000.0
            receipt.update({"response": response, "response_hash": _fingerprint(response),
                            "usage": usage, "received_at": time.time()})
            self._write(conn, attempt_id, time.time(), status="received",
                        actual_tokens=actual_tokens, actual_cost=actual_cost,
                        receipt=self._encode(receipt))
            self._record_event(conn, "budget_received",
                              {"attempt_id": attempt_id, "actual_tokens": actual_tokens,
                               "actual_cost": actual_cost, "unknown_usage": known is None})
            over_reservation = bool(actual_tokens is not None and (
                actual_tokens > row["reserved_tokens"]
                or actual_cost > math.nextafter(row["reserved_cost"] or 0.0, math.inf)))
            over_budget = False
            if not over_reservation:
                # Totals already include this settlement, so ask with no delta.
                # A raise here would roll back the receipt, so only record it.
                try:
                    self._enforce(conn, attempt_id=attempt_id)
                except BudgetExceeded:
                    over_budget = True
            if over_reservation or over_budget:
                detail = (
                    f"actual usage {actual_tokens} tokens / {actual_cost} exceeds the "
                    f"reservation {row['reserved_tokens']} / {row['reserved_cost']}"
                    if over_reservation else
                    f"the ledger is now over a frozen limit after settling {attempt_id}")
                # Durable close marker, so the stop survives a reopen and stops
                # every later reservation without depending on totals arithmetic.
                receipt["stop_reason"] = detail
                self._write(conn, attempt_id, time.time(), receipt=self._encode(receipt))
                self._record_event(conn, "error_budget_overshoot",
                                   {"attempt_id": attempt_id, "detail": detail,
                                    "actual_tokens": actual_tokens,
                                    "actual_cost": actual_cost})
        # The paid response is durable now, so stop the caller before more spending.
        if detail is not None:
            raise BudgetExceeded(
                f"attempt {attempt_id} settled over budget: {detail}. "
                f"Recorded, not capped; stop the run and reconcile.")
        return self.attempt(attempt_id)

    def fail(self, attempt_id: str, error: str, confirmed_not_executed: bool = False) -> dict:
        """Record a failure.  Only a confirmed-not-executed attempt becomes rejected.

        The caller redacts the error string; the ledger keeps it verbatim so a
        human can adjudicate an unknown request later.
        """
        text = str(error if error is not None else "")
        with self.store.transaction() as conn:
            row = self._attempt(conn, attempt_id)
            receipt = _receipt_of(row)
            now = time.time()
            receipt["error"] = text
            receipt["failed_at"] = now
            if confirmed_not_executed:
                if row["status"] not in ("reserved", "dispatched"):
                    raise StoreError(f"cannot reject an attempt in status {row['status']}")
                receipt["confirmed_not_executed"] = True
                self._write(conn, attempt_id, now, status="rejected",
                            actual_tokens=0, actual_cost=0.0, receipt=self._encode(receipt))
            else:
                if row["status"] != "dispatched":
                    raise StoreError(f"cannot fail an attempt in status {row['status']}")
                # Outcome unknown: the hold stays and the request is not replayable.
                self._write(conn, attempt_id, now, status="unknown",
                            receipt=self._encode(receipt))
            self._record_event(conn, "budget_failed",
                              {"attempt_id": attempt_id, "error": text,
                               "confirmed_not_executed": confirmed_not_executed})
        return self.attempt(attempt_id)

    def cached(self, task_id: str, request_hash: str) -> dict:
        """Return a verified receipt, or None.  A corrupt receipt raises UnknownRequest.

        An imported receipt counts: the request was paid for in its source run,
        so reusing it here must still cost zero new HTTP calls.
        """
        row = self.store._conn.execute(
            "SELECT * FROM attempts WHERE task_id = ? AND request_hash = ?"
            f" AND status IN ({_status_list(CACHED_STATUSES)}) ORDER BY created_at LIMIT 1",
            (task_id, request_hash)).fetchone()
        if row is None:
            return None
        receipt = _read_receipt(row)
        if (_fingerprint(receipt.get("response")) != receipt.get("response_hash")):
            raise UnknownRequest(f"receipt fingerprint mismatch for {task_id}/{request_hash}")
        return {"response": receipt.get("response"), "usage": receipt.get("usage"),
                "status": row["status"], "attempt_id": row["attempt_id"],
                "actual_tokens": row["actual_tokens"], "actual_cost": row["actual_cost"]}

    def recover_unfinished(self) -> dict:
        """After a crash: dispatched -> unknown; reserved -> released (0 calls spent)."""
        marked_unknown, released = [], []
        with self.store.transaction() as conn:
            now = time.time()
            rows = conn.execute(
                "SELECT attempt_id, status, receipt FROM attempts"
                " WHERE status IN ('dispatched', 'reserved')").fetchall()
            for row in rows:
                receipt = _receipt_of(row)
                if row["status"] == "dispatched":
                    # The HTTP call may already have been paid for, so hold it.
                    receipt["recovered_at"] = now
                    self._write(conn, row["attempt_id"], now, status="unknown",
                                receipt=self._encode(receipt))
                    self._record_event(conn, "budget_recovered_unknown",
                                       {"attempt_id": row["attempt_id"]})
                    marked_unknown.append(row["attempt_id"])
                else:
                    # Still reserved means the dispatch transaction never
                    # committed, so this request provably never left the process.
                    receipt["released_at"] = now
                    self._write(conn, row["attempt_id"], now, status="rejected",
                                actual_tokens=0, actual_cost=0.0, receipt=self._encode(receipt))
                    self._record_event(conn, "budget_released",
                                       {"attempt_id": row["attempt_id"],
                                        "reason": "never dispatched"})
                    released.append(row["attempt_id"])
        return {"marked_unknown": marked_unknown, "released": released}

    def totals(self) -> dict:
        """Current accounting against the frozen limits.  Budgets never reset."""
        with self.store.transaction() as conn:
            totals = _totals(conn)
            totals["open_slots"] = self._open_slots(conn)
            totals["runtime"] = time.time() - self._started_at(conn)
            totals["stop_reason"] = self._stop_reason(conn)
        return totals

    def over_budget(self) -> bool:
        """True when the ledger already exceeds a frozen limit or is stopped."""
        totals = self.totals()
        return (totals["calls"] + totals["open_slots"] > self.budget["max_calls"]
                or totals["known_tokens"] + totals["held_tokens"] > self.budget["max_tokens"]
                or totals["known_cost"] + totals["held_cost"] > math.nextafter(self.budget["max_cost"], math.inf)
                or totals["stop_reason"] is not None)

    def reconcile(self, attempt_id: str, outcome: str, reason: str, usage=None) -> dict:
        """Explicit human outcome for an unresolved request.

        not_executed releases the hold and lets the task be retried within the
        persisted attempt ceiling.  executed_unknown keeps holding and stays
        unreplayable, because a paid request must not be silently re-sent.
        """
        if outcome not in ("not_executed", "executed_unknown"):
            raise BudgetError("outcome must be not_executed or executed_unknown")
        if not str(reason or "").strip():
            raise BudgetError("reconcile needs a reason")
        known = _known_usage(usage)
        with self.store.transaction() as conn:
            row = self._attempt(conn, attempt_id)
            if row["status"] != "unknown":
                raise StoreError(
                    f"only unknown attempts can be reconciled; {attempt_id} is {row['status']}")
            receipt = _receipt_of(row)
            now = time.time()
            receipt["reconciled"] = {"outcome": outcome, "reason": reason, "at": now,
                                     "usage": usage}
            if outcome == "not_executed":
                self._write(conn, attempt_id, now, status="rejected",
                            actual_tokens=0, actual_cost=0.0, receipt=self._encode(receipt))
            else:
                # The hold stays; a known usage settles tokens but keeps money held.
                self._write(conn, attempt_id, now, status="unknown",
                            actual_tokens=None if known is None else known[0] + known[1],
                            actual_cost=None, receipt=self._encode(receipt))
            self._record_event(conn, "budget_reconciled",
                              {"attempt_id": attempt_id, "outcome": outcome, "reason": reason})
        return self.attempt(attempt_id)

    def attempt(self, attempt_id: str) -> dict:
        row = self.store._conn.execute(
            "SELECT * FROM attempts WHERE attempt_id = ?", (attempt_id,)).fetchone()
        if row is None:
            return None
        record = dict(row)
        # Same rule as _receipt_of: a damaged receipt is reported, never masked.
        record["receipt"] = _read_receipt(row)
        return record
