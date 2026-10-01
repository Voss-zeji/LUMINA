"""G4 budget ledger tests: reservation, settlement, ceilings, recovery, no replay."""
from __future__ import annotations

import json
import hashlib
import sys
import tempfile
import threading
import time
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumina.agent.budget import (
    AttemptLimit,
    Budget,
    BudgetError,
    BudgetExceeded,
    UnknownRequest,
)
from lumina.agent.store import Store, StoreError

RATE_IN = 0.25
RATE_OUT = 1.5
MAX_IN = 100
MAX_OUT = 200


def fingerprint(value) -> str:
    """The same canonical fingerprint the ledger stores in a receipt."""
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, default=str).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def make_spec(**budget_overrides) -> dict:
    budget = {"max_calls": 10, "max_tokens": 100000, "max_cost": 5.0, "max_runtime": 3600.0}
    budget.update(budget_overrides)
    model = {"input_per_million": RATE_IN, "output_per_million": RATE_OUT,
             "max_input_tokens": MAX_IN, "max_output_tokens": MAX_OUT, "basis": "test"}
    return {"budget": budget, "pricing": {"vendor/chat-a": dict(model),
                                          "vendor/chat-b": dict(model)}}


class BudgetCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        # LIFO: the stores close before TemporaryDirectory removes the ledger.
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run"
        self.store = Store(self.run_dir, create=True)
        self.addCleanup(self.store.close)
        self.store.initialize("spec-abc")
        self.spec = make_spec()
        self.budget = Budget(self.store, self.spec)

    def reopen(self, spec=None) -> Budget:
        store = Store(self.run_dir)
        # The previous store is already registered for cleanup from setUp;
        # Store.close() is idempotent, so re-registering the new one is safe.
        self.addCleanup(store.close)
        self.store = store
        self.budget = Budget(store, spec or self.spec)
        return self.budget

    def on_connection(self, spec, fn):
        """Run fn(budget) against a separate Store, in the calling thread.

        sqlite3 refuses cross-thread use by default, so both the connection and
        its close must happen inside the thread that opens it.
        """
        store = Store(self.run_dir)
        try:
            return fn(Budget(store, spec or self.spec))
        finally:
            store.close()

    def in_thread(self, target):
        """Run target() on a worker thread and return anything it produced."""
        box = {}

        def run():
            try:
                box["value"] = target()
            except Exception as exc:  # noqa: BLE001 - the caller asserts on the type
                box["error"] = exc

        thread = threading.Thread(target=run)
        thread.start()
        thread.join(timeout=30)
        self.assertFalse(thread.is_alive(), "the worker thread must finish")
        if "error" in box:
            raise box["error"]
        return box.get("value")

    def reserve(self, task_id="t1", request_hash="r1", **kwargs) -> str:
        return self.budget.reserve(task_id, request_hash, kwargs.pop("kind", "chat"),
                                  kwargs.pop("model", "vendor/chat-a"),
                                  kwargs.pop("input_tokens", 10),
                                  kwargs.pop("output_tokens", 20))

    def settle(self, attempt_id, response=None, usage=None) -> dict:
        self.budget.dispatch(attempt_id)
        return self.budget.receive(attempt_id, response if response is not None else {"ok": True},
                                   usage)

    def write_attempt(self, receipt, **columns) -> str:
        """Insert an attempt row directly, as the parent's import writer does."""
        attempt_id = columns.pop("attempt_id", "imported-attempt-1")
        now = time.time()
        row = {"task_id": "t1", "request_hash": "r1", "kind": "chat",
               "status": "imported", "reserved_tokens": 0, "reserved_cost": 0.0,
               "actual_tokens": 0, "actual_cost": 0.0, "raw_path": None, **columns}
        with self.store.transaction() as conn:
            conn.execute(
                "INSERT INTO attempts(attempt_id, task_id, request_hash, kind, status,"
                " reserved_tokens, reserved_cost, actual_tokens, actual_cost, raw_path,"
                " receipt, created_at, updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (attempt_id, row["task_id"], row["request_hash"], row["kind"],
                 row["status"], row["reserved_tokens"], row["reserved_cost"],
                 row["actual_tokens"], row["actual_cost"], row["raw_path"],
                 json.dumps(receipt), now, now))
        return attempt_id

    def import_receipt(self, response=None, usage=None, **extra) -> dict:
        """A parent-created immutable import receipt from an already paid run."""
        payload = {"ok": "imported"} if response is None else response
        provenance = {"run_id": "run-20260901", "attempt_id": "src-attempt-7",
                      "spec_hash": "spec-src"}
        return {"model": "vendor/chat-a", "response": payload,
                "response_hash": fingerprint(payload),
                "original_usage": usage if usage is not None
                else {"input_tokens": 900, "output_tokens": 900},
                "imported_from": provenance, **extra}

    def overwrite_receipt(self, attempt_id, payload):
        """Damage a canonical receipt the way disk corruption would."""
        with self.store.transaction() as conn:
            conn.execute("UPDATE attempts SET receipt = ? WHERE attempt_id = ?",
                         (payload, attempt_id))


class TestReservation(BudgetCase):
    def test_fractional_money_rates_are_accepted(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.budget.receive(attempt_id, {"ok": True},
                            {"input_tokens": 10, "output_tokens": 20})
        expected = (10 * RATE_IN + 20 * RATE_OUT) / 1e6
        self.assertAlmostEqual(self.budget.attempt(attempt_id)["actual_cost"], expected)
        self.assertAlmostEqual(self.budget.totals()["known_cost"], expected)

    def test_reserve_holds_the_frozen_conservative_maximum(self):
        attempt_id = self.reserve(input_tokens=5, output_tokens=5)
        record = self.budget.attempt(attempt_id)
        self.assertEqual(record["status"], "reserved")
        self.assertEqual(record["reserved_tokens"], MAX_IN + MAX_OUT)
        self.assertAlmostEqual(record["reserved_cost"],
                               (MAX_IN * RATE_IN + MAX_OUT * RATE_OUT) / 1e6)
        self.assertEqual(self.budget.totals()["held_tokens"], MAX_IN + MAX_OUT)
        self.assertEqual(self.budget.totals()["calls"], 0,
                         "an undispatched reservation is not yet a real call")

    def test_request_above_frozen_bounds_is_rejected(self):
        with self.assertRaises(BudgetExceeded):
            self.reserve(input_tokens=MAX_IN + 1)
        with self.assertRaises(BudgetExceeded):
            self.reserve(output_tokens=MAX_OUT + 1)
        self.assertEqual(self.budget.totals()["attempts"], 0,
                         "a refused reservation must not be recorded")

    def test_requests_exactly_at_the_bound_are_allowed(self):
        attempt_id = self.reserve(input_tokens=MAX_IN, output_tokens=MAX_OUT)
        self.assertEqual(self.budget.attempt(attempt_id)["reserved_tokens"], MAX_IN + MAX_OUT)

    def test_embedding_may_reserve_the_output_bound_with_zero_actual_output(self):
        attempt_id = self.reserve(kind="embedding", input_tokens=50, output_tokens=0)
        self.budget.dispatch(attempt_id)
        self.budget.receive(attempt_id, {"embedding": [0.1]},
                            {"input_tokens": 50, "output_tokens": 0})
        record = self.budget.attempt(attempt_id)
        self.assertEqual(record["actual_tokens"], 50)
        self.assertAlmostEqual(record["actual_cost"], 50 * RATE_IN / 1e6)

    def test_bad_token_arguments_rejected(self):
        for bad in (-1, True, False, 1.5, float("inf"), float("nan"), "10", None):
            with self.subTest(bad=bad), self.assertRaises(BudgetExceeded):
                self.reserve(input_tokens=bad)
            with self.subTest(bad=bad), self.assertRaises(BudgetExceeded):
                self.reserve(output_tokens=bad)

    def test_unknown_kind_and_unpriced_model_rejected(self):
        with self.assertRaises(BudgetError):
            self.budget.reserve("t1", "r1", "telepathy", "vendor/chat-a", 1, 1)
        with self.assertRaises(BudgetExceeded):
            self.budget.reserve("t1", "r1", "chat", "vendor/missing", 1, 1)

    def test_every_kind_is_accepted(self):
        for kind in ("chat", "embedding", "advisor"):
            attempt_id = self.reserve(task_id="t-" + kind, request_hash="r-" + kind, kind=kind)
            self.assertEqual(self.budget.attempt(attempt_id)["kind"], kind)

    def test_invalid_pricing_rates_are_refused(self):
        for bad in (float("inf"), float("nan"), -0.5, True, "1.0"):
            with self.subTest(bad=bad):
                spec = make_spec()
                spec["pricing"]["vendor/chat-a"]["input_per_million"] = bad
                budget = Budget(self.store, spec)
                with self.assertRaises(BudgetExceeded):
                    budget.reserve("t1", "r1", "chat", "vendor/chat-a", 1, 1)


class TestLiveReservationBlocksReuse(BudgetCase):
    def test_second_reservation_while_first_is_still_reserved_is_refused(self):
        self.reserve()
        with self.assertRaises(UnknownRequest):
            self.reserve()
        self.assertEqual(self.budget.totals()["attempts"], 1)

    def test_released_reservation_frees_the_pair_again(self):
        attempt_id = self.reserve()
        self.budget.recover_unfinished()
        self.assertEqual(self.budget.attempt(attempt_id)["status"], "rejected")
        fresh = self.reserve()
        self.assertEqual(self.budget.attempt(fresh)["status"], "reserved")


class TestSettlement(BudgetCase):
    def test_known_usage_settles_once_not_reserved_plus_actual(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, usage={"input_tokens": 10, "output_tokens": 20})
        totals = self.budget.totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["known_tokens"], 30)
        self.assertEqual(totals["held_tokens"], 0, "settlement releases the hold")
        self.assertEqual(totals["unknown_usage"], 0)

    def test_unknown_usage_keeps_the_reservation_and_is_never_zero(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, usage=None)
        record = self.budget.attempt(attempt_id)
        self.assertEqual(record["status"], "received")
        self.assertIsNone(record["actual_tokens"])
        self.assertIsNone(record["actual_cost"])
        totals = self.budget.totals()
        self.assertEqual(totals["held_tokens"], MAX_IN + MAX_OUT)
        self.assertEqual(totals["unknown_usage"], 1)
        self.assertEqual(totals["known_tokens"], 0)
        self.assertEqual(totals["known_cost"], 0.0)

    def test_partial_or_bogus_usage_is_not_treated_as_known(self):
        for usage in ({"input_tokens": 10}, {"output_tokens": 20}, {},
                      {"input_tokens": -1, "output_tokens": 5},
                      {"input_tokens": 1.0, "output_tokens": 2},
                      {"input_tokens": True, "output_tokens": 2}):
            with self.subTest(usage=usage):
                self.setUp()
                attempt_id = self.reserve()
                self.settle(attempt_id, usage=usage)
                self.assertIsNone(self.budget.attempt(attempt_id)["actual_tokens"])
                self.assertEqual(self.budget.totals()["held_tokens"], MAX_IN + MAX_OUT)

    def test_overshoot_is_recorded_then_raises_without_losing_the_response(self):
        attempt_id = self.reserve(input_tokens=50, output_tokens=50)
        self.budget.dispatch(attempt_id)
        response = {"choices": [{"text": "expensive"}]}
        with self.assertRaises(BudgetExceeded):
            self.budget.receive(attempt_id, response,
                                {"input_tokens": MAX_IN + 5, "output_tokens": MAX_OUT + 5})
        record = self.budget.attempt(attempt_id)
        self.assertEqual(record["status"], "received",
                         "the paid response survives the ceiling error")
        self.assertEqual(record["actual_tokens"], MAX_IN + MAX_OUT + 10)
        self.assertGreater(record["actual_cost"], record["reserved_cost"],
                           "usage is recorded uncapped")
        self.assertEqual(record["receipt"]["response"], response)
        totals = self.budget.totals()
        self.assertEqual(totals["known_tokens"], MAX_IN + MAX_OUT + 10)
        self.assertEqual(totals["held_tokens"], 0)
        self.assertTrue(self.budget.over_budget())

    def test_overshoot_then_new_reservation_is_refused_until_a_new_run(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        with self.assertRaises(BudgetExceeded):
            self.budget.receive(attempt_id, {"ok": True},
                                {"input_tokens": MAX_IN + 1, "output_tokens": MAX_OUT + 1})
        with self.assertRaises(BudgetExceeded):
            self.reserve(request_hash="r-next")
        self.assertEqual(self.budget.totals()["attempts"], 1)

    def test_cached_response_is_still_reusable_after_an_overshoot(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        response = {"ok": "paid"}
        with self.assertRaises(BudgetExceeded):
            self.budget.receive(attempt_id, response,
                                {"input_tokens": MAX_IN + 1, "output_tokens": MAX_OUT + 1})
        reopened = self.reopen()
        cached = reopened.cached("t1", "r1")
        self.assertEqual(cached["response"], response)
        self.assertTrue(reopened.over_budget())

    def test_confirmed_failed_attempt_counts_a_call_with_zero_actual(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.budget.fail(attempt_id, "429 rate limited", confirmed_not_executed=True)
        totals = self.budget.totals()
        self.assertEqual(totals["calls"], 1, "a dispatched 429 spent a real call")
        self.assertEqual(totals["known_tokens"], 0)
        self.assertEqual(totals["held_tokens"], 0)


class TestDispatchEvidence(BudgetCase):
    def test_undispatched_release_spends_no_call(self):
        self.reserve()
        self.budget.recover_unfinished()
        totals = self.budget.totals()
        self.assertEqual(totals["calls"], 0)
        self.assertEqual(totals["open_slots"], 0)

    def test_dispatched_429_then_rejected_still_counts_one_call(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.budget.fail(attempt_id, "429", confirmed_not_executed=True)
        self.assertEqual(self.budget.attempt(attempt_id)["status"], "rejected")
        self.assertEqual(self.budget.totals()["calls"], 1)

    def test_reserve_then_dispatch_allowed_at_max_calls_one(self):
        budget = Budget(self.store, make_spec(max_calls=1))
        attempt_id = budget.reserve("t1", "r1", "chat", "vendor/chat-a", 10, 20)
        budget.dispatch(attempt_id)
        self.assertEqual(budget.totals()["calls"], 1)
        with self.assertRaises(BudgetExceeded):
            budget.reserve("t2", "r2", "chat", "vendor/chat-a", 10, 20)

    def test_open_slot_blocks_a_second_reservation_at_max_calls_one(self):
        budget = Budget(self.store, make_spec(max_calls=1))
        budget.reserve("t1", "r1", "chat", "vendor/chat-a", 10, 20)
        with self.assertRaises(BudgetExceeded):
            budget.reserve("t2", "r2", "chat", "vendor/chat-a", 10, 20)

    def test_settlement_that_pushes_the_run_over_ceiling_raises(self):
        budget = Budget(self.store, make_spec(max_tokens=300, max_calls=5))
        attempt_id = budget.reserve("t1", "r1", "chat", "vendor/chat-a", 10, 20)
        budget.dispatch(attempt_id)
        # Actual equals the reservation, but the ceiling check still sees 300 > 300? No:
        # exactly at the ceiling is allowed, so overshoot must exceed it.
        budget.receive(attempt_id, {"ok": True},
                       {"input_tokens": 100, "output_tokens": 200})
        self.assertEqual(budget.totals()["known_tokens"], 300)
        self.assertFalse(budget.over_budget())

    def test_conservative_holds_make_the_run_ceiling_unreachable_by_accumulation(self):
        # A settlement can never exceed its own reservation without tripping the
        # over-reservation stop, and a second hold is refused up front, so the
        # run ceiling is never crossed by accumulation.
        budget = Budget(self.store, make_spec(max_tokens=350, max_calls=5, max_cost=1000.0))
        first = budget.reserve("t1", "r1", "chat", "vendor/chat-a", 10, 20)
        budget.dispatch(first)
        budget.receive(first, {"ok": True}, {"input_tokens": 100, "output_tokens": 200})
        self.assertEqual(budget.totals()["known_tokens"], 300)
        with self.assertRaises(BudgetExceeded):
            budget.reserve("t2", "r2", "chat", "vendor/chat-a", 10, 20)
        self.assertFalse(budget.over_budget(), "the refused hold never overbooked the run")


class TestReplayAndAttempts(BudgetCase):
    def test_unconfirmed_failure_holds_reservation_and_blocks_replay(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.budget.fail(attempt_id, "socket timeout")
        record = self.budget.attempt(attempt_id)
        self.assertEqual(record["status"], "unknown")
        self.assertEqual(record["receipt"]["error"], "socket timeout",
                         "the failure reason is durable for adjudication")
        totals = self.budget.totals()
        self.assertEqual(totals["unknown_requests"], 1)
        self.assertEqual(totals["held_tokens"], MAX_IN + MAX_OUT)
        with self.assertRaises(UnknownRequest):
            self.reserve()

    def test_failure_reason_reaches_the_event_log(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.budget.fail(attempt_id, "gateway timeout")
        kinds = [event["kind"] for event in self.store._events()]
        self.assertIn("budget_failed", kinds)
        payload = [e["payload"] for e in self.store._events() if e["kind"] == "budget_failed"][-1]
        self.assertEqual(payload["error"], "gateway timeout")

    def test_received_success_blocks_a_new_reservation(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, usage={"input_tokens": 10, "output_tokens": 20})
        with self.assertRaises(UnknownRequest):
            self.reserve()

    def test_three_confirmed_failures_then_attempt_limit(self):
        for index in range(3):
            attempt_id = self.reserve()
            self.budget.dispatch(attempt_id)
            self.budget.fail(attempt_id, f"429 attempt {index}", confirmed_not_executed=True)
        with self.assertRaises(AttemptLimit):
            self.reserve()

    def test_attempt_ceiling_survives_reopen(self):
        for _ in range(3):
            attempt_id = self.reserve()
            self.budget.dispatch(attempt_id)
            self.budget.fail(attempt_id, "429", confirmed_not_executed=True)
        self.reopen()
        with self.assertRaises(AttemptLimit):
            self.reserve()

    def test_distinct_requests_have_independent_attempt_ceilings(self):
        for index in range(3):
            self.reserve(request_hash=f"other-{index}")
        # Those three spent nothing against r1, which keeps its own ceiling.
        attempt_id = self.reserve()
        self.assertEqual(self.budget.attempt(attempt_id)["status"], "reserved",
                         "a different request_hash has its own three attempts")
        self.budget.recover_unfinished()
        for _ in range(2):
            attempt_id = self.reserve()
            self.budget.dispatch(attempt_id)
            self.budget.fail(attempt_id, "429", confirmed_not_executed=True)
        with self.assertRaises(AttemptLimit):
            self.reserve()

    def test_calls_accumulate_across_restarts_without_reset(self):
        budget = Budget(self.store, make_spec(max_calls=3))
        for index in range(3):
            attempt_id = budget.reserve(f"t{index}", f"r{index}", "chat",
                                        "vendor/chat-a", 10, 20)
            budget.dispatch(attempt_id)
            budget.receive(attempt_id, {"ok": index},
                           {"input_tokens": 10, "output_tokens": 20})
        budget = self.reopen(make_spec(max_calls=3))
        self.assertEqual(budget.totals()["calls"], 3)
        with self.assertRaises(BudgetExceeded):
            budget.reserve("t-new", "r-new", "chat", "vendor/chat-a", 1, 1)


class TestCachedReuse(BudgetCase):
    def test_response_persists_and_is_reused_across_reopen(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, response={"choices": [{"text": "hi"}]},
                    usage={"input_tokens": 10, "output_tokens": 20})
        self.reopen()
        cached = self.budget.cached("t1", "r1")
        self.assertEqual(cached["response"], {"choices": [{"text": "hi"}]})
        self.assertEqual(cached["usage"], {"input_tokens": 10, "output_tokens": 20})
        self.assertEqual(cached["status"], "received")
        self.assertEqual(self.budget.totals()["calls"], 1, "reuse costs no new call")

    def test_cached_misses_when_nothing_received(self):
        self.assertIsNone(self.budget.cached("t-absent", "r-absent"))

    def test_corrupt_receipt_raises_and_never_pays_again(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, response={"a": 1},
                    usage={"input_tokens": 1, "output_tokens": 2})
        with self.store.transaction() as conn:
            conn.execute(
                "UPDATE attempts SET receipt = ? WHERE attempt_id = ?",
                (json.dumps({"response": {"a": 2}, "response_hash": "0" * 64,
                             "usage": None}), attempt_id))
        with self.assertRaises(UnknownRequest):
            self.budget.cached("t1", "r1")
        with self.assertRaises(UnknownRequest):
            self.reserve()

    def test_unparsable_receipt_raises(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, usage={"input_tokens": 1, "output_tokens": 2})
        with self.store.transaction() as conn:
            conn.execute("UPDATE attempts SET receipt = 'not json' WHERE attempt_id = ?",
                         (attempt_id,))
        with self.assertRaises(UnknownRequest):
            self.budget.cached("t1", "r1")

    def test_receipt_holds_the_full_response_durably(self):
        attempt_id = self.reserve()
        response = {"choices": [{"message": {"content": "x" * 500}}], "meta": {"nested": [1, 2]}}
        self.settle(attempt_id, response=response)
        self.reopen()
        self.assertEqual(self.budget.attempt(attempt_id)["receipt"]["response"], response)


class TestImportedCache(BudgetCase):
    """A parent-created immutable import receipt from an already paid run."""

    def test_imported_response_is_reused_with_zero_new_calls(self):
        self.write_attempt(self.import_receipt({"choices": [{"text": "hi"}]}))
        self.reopen()
        cached = self.budget.cached("t1", "r1")
        self.assertEqual(cached["status"], "imported")
        self.assertEqual(cached["response"], {"choices": [{"text": "hi"}]})
        self.assertEqual(cached["actual_tokens"], 0)
        self.assertEqual(cached["actual_cost"], 0.0)
        self.assertEqual(self.budget.totals()["calls"], 0, "an import spent no HTTP call")

    def test_import_counts_separately_and_never_charges_copied_usage(self):
        self.write_attempt(self.import_receipt())
        totals = self.budget.totals()
        self.assertEqual(totals["imports"], 1)
        self.assertEqual(totals["known_tokens"], 0,
                         "historical usage is provenance, not this run's spend")
        self.assertEqual(totals["held_tokens"], 0)
        self.assertEqual(totals["known_cost"], 0.0)
        self.assertEqual(totals["held_cost"], 0.0)
        self.assertFalse(self.budget.over_budget(),
                         "an import cannot consume the target run's budget")

    def test_import_does_not_reset_or_duplicate_a_real_call(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, usage={"input_tokens": 10, "output_tokens": 20})
        self.write_attempt(self.import_receipt(), attempt_id="imported-2", request_hash="r2")
        totals = self.budget.totals()
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(totals["imports"], 1)
        self.assertEqual(totals["known_tokens"], 30)

    def test_reserve_refuses_while_an_import_is_the_predecessor(self):
        self.write_attempt(self.import_receipt())
        with self.assertRaises(UnknownRequest):
            self.reserve()
        self.assertEqual(self.budget.totals()["attempts"], 1)

    def test_corrupt_import_receipt_raises_and_never_pays_again(self):
        receipt = self.import_receipt()
        attempt_id = self.write_attempt(receipt)
        tampered = dict(receipt, response={"tampered": True})
        self.overwrite_receipt(attempt_id, json.dumps(tampered))
        with self.assertRaises(UnknownRequest):
            self.budget.cached("t1", "r1")
        with self.assertRaises(UnknownRequest):
            self.reserve()

    def test_unparsable_import_receipt_raises(self):
        attempt_id = self.write_attempt(self.import_receipt())
        self.overwrite_receipt(attempt_id, "not json")
        with self.assertRaises(UnknownRequest):
            self.budget.cached("t1", "r1")
        with self.assertRaises(UnknownRequest):
            self.reserve()

    def test_cached_still_misses_when_only_an_unrelated_import_exists(self):
        self.write_attempt(self.import_receipt(), attempt_id="imported-3")
        self.assertIsNone(self.budget.cached("t1", "r-absent"))


class TestCorruptCanonicalReceipt(BudgetCase):
    """A damaged receipt must never degrade into silently erased accounting."""

    def test_totals_refuses_to_hide_a_corrupt_reserved_receipt(self):
        attempt_id = self.reserve()
        self.overwrite_receipt(attempt_id, "not json")
        with self.assertRaises(StoreError):
            self.budget.totals()

    def test_totals_refuses_to_hide_a_corrupt_dispatched_receipt(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.overwrite_receipt(attempt_id, "not json")
        with self.assertRaises(StoreError):
            self.budget.totals()

    def test_totals_refuses_to_hide_a_corrupt_rejected_receipt(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.budget.fail(attempt_id, "429", confirmed_not_executed=True)
        self.overwrite_receipt(attempt_id, "[1, 2, 3]")
        with self.assertRaises(StoreError):
            self.budget.totals()

    def test_stripped_receipt_cannot_erase_paid_call_accounting(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, usage={"input_tokens": 10, "output_tokens": 20})
        self.overwrite_receipt(attempt_id, "{}")
        with self.assertRaises(StoreError):
            self.budget.totals()
        self.overwrite_receipt(attempt_id, "not json")
        with self.assertRaises(StoreError):
            self.budget.totals()

    def test_lost_dispatch_evidence_cannot_reduce_max_calls(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, usage={"input_tokens": 10, "output_tokens": 20})
        receipt = self.budget.attempt(attempt_id)["receipt"]
        receipt.pop("dispatched_at")
        self.overwrite_receipt(attempt_id, json.dumps(receipt))
        with self.assertRaises(StoreError):
            self.budget.reserve("new-task", "new-request", "chat", "vendor/chat-a", 10, 20)

    def test_reserve_refuses_when_a_prior_receipt_is_corrupt(self):
        first = self.reserve()
        self.overwrite_receipt(first, "not json")
        with self.assertRaises(StoreError):
            self.reserve(request_hash="r-next")

    def test_corrupt_stop_marker_is_not_ignored(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        oversized = {"input_tokens": MAX_IN + 1, "output_tokens": MAX_OUT + 1}
        with self.assertRaises(BudgetExceeded):
            self.budget.receive(attempt_id, {"ok": True}, oversized)
        self.overwrite_receipt(attempt_id, "not json")
        with self.assertRaises(StoreError):
            self.budget.over_budget()
        with self.assertRaises(StoreError):
            self.reserve(request_hash="r-next")

    def test_dispatch_refuses_a_corrupt_receipt_without_spending_a_call(self):
        attempt_id = self.reserve()
        self.overwrite_receipt(attempt_id, "not json")
        with self.assertRaises(StoreError):
            self.budget.dispatch(attempt_id)
        with self.store.transaction() as conn:
            row = conn.execute("SELECT status FROM attempts WHERE attempt_id = ?",
                               (attempt_id,)).fetchone()
        self.assertEqual(row["status"], "reserved",
                         "the refusal rolled back, so no call was spent")


class TestRecovery(BudgetCase):
    def test_reserved_is_released_without_consuming_a_call(self):
        self.reserve()
        report = self.budget.recover_unfinished()
        self.assertEqual(len(report["released"]), 1)
        totals = self.budget.totals()
        self.assertEqual(totals["calls"], 0)
        self.assertEqual(totals["held_tokens"], 0)
        self.assertEqual(totals["open_slots"], 0)

    def test_dispatched_becomes_unknown_and_holds(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        report = self.budget.recover_unfinished()
        self.assertEqual(report["marked_unknown"], [attempt_id])
        totals = self.budget.totals()
        self.assertEqual(totals["unknown_requests"], 1)
        self.assertEqual(totals["held_tokens"], MAX_IN + MAX_OUT)
        self.assertEqual(totals["calls"], 1, "a dispatched call stays counted")
        with self.assertRaises(UnknownRequest):
            self.reserve()

    def test_received_attempts_are_untouched_by_recovery(self):
        attempt_id = self.reserve()
        self.settle(attempt_id, usage={"input_tokens": 10, "output_tokens": 20})
        self.budget.recover_unfinished()
        self.assertEqual(self.budget.attempt(attempt_id)["status"], "received")

    def test_recovery_survives_reopen(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.reopen()
        self.budget.recover_unfinished()
        self.assertEqual(self.budget.attempt(attempt_id)["status"], "unknown")

    def test_recovery_leaves_no_attempt_in_a_transient_status(self):
        for index in range(3):
            attempt_id = self.reserve(task_id=f"t{index}", request_hash=f"r{index}")
            if index % 2:
                self.budget.dispatch(attempt_id)
        self.budget.recover_unfinished()
        rows = self.store._conn.execute("SELECT status FROM attempts").fetchall()
        self.assertTrue(all(row["status"] not in ("reserved", "dispatched") for row in rows))


class TestReconcile(BudgetCase):
    def make_unknown(self) -> str:
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        self.budget.fail(attempt_id, "timeout")
        return attempt_id

    def test_not_executed_outcome_releases_the_hold(self):
        attempt_id = self.make_unknown()
        self.budget.reconcile(attempt_id, "not_executed", "provider logs show no request")
        totals = self.budget.totals()
        self.assertEqual(totals["held_tokens"], 0)
        self.assertEqual(self.budget.attempt(attempt_id)["status"], "rejected")

    def test_not_executed_allows_a_retry_inside_the_attempt_ceiling(self):
        attempt_id = self.make_unknown()
        self.budget.reconcile(attempt_id, "not_executed", "confirmed never sent")
        retry = self.reserve()
        self.assertEqual(self.budget.attempt(retry)["status"], "reserved")

    def test_executed_unknown_keeps_holding_and_refuses_fresh_retry(self):
        attempt_id = self.make_unknown()
        self.budget.reconcile(attempt_id, "executed_unknown", "billed, no response")
        totals = self.budget.totals()
        self.assertEqual(totals["held_tokens"], MAX_IN + MAX_OUT)
        self.assertEqual(self.budget.attempt(attempt_id)["status"], "unknown")
        with self.assertRaises(UnknownRequest):
            self.reserve()

    def test_executed_unknown_with_known_usage_settles_tokens_but_holds_money(self):
        attempt_id = self.make_unknown()
        self.budget.reconcile(attempt_id, "executed_unknown", "provider billed 40 tokens",
                              usage={"input_tokens": 20, "output_tokens": 20})
        totals = self.budget.totals()
        self.assertEqual(totals["known_tokens"], 40)
        self.assertEqual(totals["held_tokens"], 0, "the token hold is settled")
        self.assertGreater(totals["held_cost"], 0.0, "money stays held, never zeroed")

    def test_reconcile_records_the_reason_and_outcome_for_audit(self):
        attempt_id = self.make_unknown()
        self.budget.reconcile(attempt_id, "executed_unknown", "billed per provider log")
        entry = self.budget.attempt(attempt_id)["receipt"]["reconciled"]
        self.assertEqual(entry["outcome"], "executed_unknown")
        self.assertEqual(entry["reason"], "billed per provider log")
        kinds = [event["kind"] for event in self.store._events()]
        self.assertIn("budget_reconciled", kinds)

    def test_reason_and_outcome_are_required(self):
        attempt_id = self.make_unknown()
        with self.assertRaises(BudgetError):
            self.budget.reconcile(attempt_id, "executed_unknown", "  ")
        with self.assertRaises(BudgetError):
            self.budget.reconcile(attempt_id, "maybe", "reason")

    def test_only_unknown_attempts_are_reconcilable(self):
        attempt_id = self.reserve()
        self.budget.dispatch(attempt_id)
        with self.assertRaises(StoreError):
            self.budget.reconcile(attempt_id, "not_executed", "premature")

    def test_repeated_reconciliation_is_refused(self):
        attempt_id = self.make_unknown()
        self.budget.reconcile(attempt_id, "not_executed", "never sent")
        with self.assertRaises(StoreError):
            self.budget.reconcile(attempt_id, "executed_unknown", "changed my mind")


class TestCeilings(BudgetCase):
    def test_tiny_cost_limit_is_not_weakened_by_absolute_epsilon(self):
        spec = make_spec(max_cost=1e-16)
        spec["pricing"]["vendor/chat-a"]["input_per_million"] = 1e-10
        spec["pricing"]["vendor/chat-a"]["output_per_million"] = 1e-10
        budget = Budget(self.store, spec)
        with self.assertRaises(BudgetExceeded):
            budget.reserve("t-tiny", "r-tiny", "chat", "vendor/chat-a", 10, 20)

    def test_each_dimension_stops_independently(self):
        for overrides in ({"max_calls": 1}, {"max_tokens": 300},
                          {"max_cost": 0.0005}):
            with self.subTest(overrides=overrides):
                self.setUp()
                self.spec = make_spec(**overrides)
                self.budget = Budget(self.store, self.spec)
                self.reserve()
                with self.assertRaises(BudgetExceeded):
                    self.reserve(request_hash="r2")

    def test_token_ceiling_below_one_hold_refuses_the_first_reservation(self):
        budget = Budget(self.store, make_spec(max_tokens=299, max_calls=5))
        with self.assertRaises(BudgetExceeded):
            budget.reserve("t1", "r1", "chat", "vendor/chat-a", 10, 20)
        self.assertEqual(budget.totals()["attempts"], 0)

    def test_expired_runtime_stops_a_dispatch_without_spending_a_call(self):
        budget = Budget(self.store, make_spec(max_runtime=1.0))
        attempt_id = budget.reserve("t1", "r1", "chat", "vendor/chat-a", 10, 20)
        # The run row was created in setUp; backdate it so the wallclock, which a
        # resume must keep, is now past the frozen runtime limit.
        with self.store.transaction() as conn:
            conn.execute("UPDATE run SET started_at = ? WHERE singleton = 1",
                         (time.time() - 3600.0,))
        with self.assertRaises(BudgetExceeded):
            budget.dispatch(attempt_id)
        totals = budget.totals()
        self.assertEqual(totals["calls"], 0, "a refused dispatch spends no call")
        self.assertEqual(totals["open_slots"], 1, "the reservation still holds its slot")

    def test_budget_is_never_reset_by_a_resume(self):
        spec = make_spec(max_calls=2)
        budget = Budget(self.store, spec)
        first = budget.reserve("t1", "r1", "chat", "vendor/chat-a", 10, 20)
        budget.dispatch(first)
        budget.receive(first, {"ok": True}, {"input_tokens": 10, "output_tokens": 20})
        second = budget.reserve("t2", "r2", "chat", "vendor/chat-a", 10, 20)
        budget.dispatch(second)
        budget.receive(second, {"ok": True}, {"input_tokens": 10, "output_tokens": 20})
        budget = self.reopen(spec)
        self.assertEqual(budget.totals()["calls"], 2)
        with self.assertRaises(BudgetExceeded):
            budget.reserve("t-new", "r-new", "chat", "vendor/chat-a", 1, 1)

    def test_runtime_uses_the_run_wallclock_start_not_process_time(self):
        budget = Budget(self.store, make_spec(max_runtime=1e9))
        started = self.store._run()["started_at"]
        self.assertLess(abs(budget.totals()["runtime"] - (time.time() - started)), 5.0)

    def test_invalid_specification_budget_is_refused(self):
        for bad in ({"max_calls": 0}, {"max_calls": True}, {"max_cost": float("inf")},
                    {"max_runtime": -1}, {"max_tokens": 1.5}, {"max_cost": 0}):
            with self.subTest(bad=bad), self.assertRaises(BudgetError):
                Budget(self.store, make_spec(**bad))
        with self.assertRaises(BudgetError):
            Budget(self.store, {"pricing": {}})

    def test_every_totals_key_is_present(self):
        totals = self.budget.totals()
        for key in ("attempts", "calls", "known_tokens", "held_tokens", "known_cost",
                    "held_cost", "unknown_usage", "unknown_requests", "imports", "runtime"):
            self.assertIn(key, totals)
        self.assertEqual(totals["imports"], 0, "a fresh run has imported nothing")


class TestConcurrentCeiling(BudgetCase):
    def test_two_connections_cannot_both_reserve_at_the_call_ceiling(self):
        spec = make_spec(max_calls=1)
        self.reserve()
        with self.assertRaises(BudgetExceeded):
            self.on_connection(
                spec, lambda b: b.reserve("t2", "r2", "chat", "vendor/chat-a", 10, 20))

    def test_concurrent_reservations_share_one_token_ceiling(self):
        # 450 tokens fits exactly one 300-token conservative hold, so whichever
        # worker wins the write lock, the loser must be refused for tokens.
        spec = make_spec(max_tokens=450, max_calls=10)
        granted, refused = [], []
        barrier = threading.Barrier(2)

        def worker(task_id):
            try:
                barrier.wait(timeout=5)
                granted.append(self.on_connection(
                    spec, lambda b: b.reserve(task_id, "r-" + task_id, "chat",
                                              "vendor/chat-a", 10, 20)))
            except Exception as exc:  # the assertions below inspect the type
                refused.append(exc)

        threads = [threading.Thread(target=worker, args=(task_id,))
                   for task_id in ("t1", "t2")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(len(granted) + len(refused), 2, "every worker must answer")
        self.assertEqual(len(granted), 1, "450 tokens admits exactly one 300-token hold")
        self.assertTrue(all(isinstance(exc, BudgetExceeded) for exc in refused))
        self.assertEqual(Budget(self.store, spec).totals()["held_tokens"], 300 * len(granted))

    def test_three_workers_at_two_slots_grant_exactly_two(self):
        # Three 300-token holds against a 600-token ceiling: exactly two winners
        # may hold at once, and the third must be refused rather than overbook.
        spec = make_spec(max_tokens=600, max_calls=10)
        granted, refused = [], []
        barrier = threading.Barrier(3)

        def worker(task_id):
            try:
                barrier.wait(timeout=5)
                granted.append(self.on_connection(
                    spec, lambda b: b.reserve(task_id, "r-" + task_id, "chat",
                                              "vendor/chat-a", 10, 20)))
            except Exception as exc:  # the assertions below inspect the type
                refused.append(exc)

        threads = [threading.Thread(target=worker, args=(f"t{index}",))
                   for index in range(3)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(timeout=15)
        self.assertEqual(len(granted), 2, "600 tokens admits exactly two 300-token holds")
        self.assertEqual(len(refused), 1)
        self.assertIsInstance(refused[0], BudgetExceeded)
        self.assertEqual(Budget(self.store, spec).totals()["held_tokens"], 600)


if __name__ == "__main__":
    unittest.main()
