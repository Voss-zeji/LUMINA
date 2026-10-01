"""G3 durable ledger tests: state machine, artifacts, corruption, pause, approvals."""
from __future__ import annotations

import json
import hashlib
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from lumina.agent.store import DB_NAME, Store, StoreError


class StoreCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.run_dir = Path(self._tmp.name) / "run"
        self.store = Store(self.run_dir, create=True)
        self.addCleanup(self.store.close)
        self.store.initialize("spec-abc")

    def artifact(self, name: str = "outputs/result.json", text: str = '{"a": 1}') -> Path:
        path = self.run_dir / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        return path

    def reopen(self) -> Store:
        self.store.close()
        store = Store(self.run_dir)
        self.store = store
        self.addCleanup(store.close)
        return store


class TestTransactions(StoreCase):
    def test_context_manager_closes_store(self):
        with Store(self.run_dir) as store:
            self.assertEqual(store.snapshot()["run"]["state"], "INIT")
        with self.assertRaises(sqlite3.ProgrammingError):
            store.transaction().__enter__()

    def test_rollback_discards_partial_writes(self):
        with self.assertRaises(StoreError):
            with self.store.transaction() as conn:
                conn.execute("UPDATE run SET state='BATCH' WHERE singleton=1")
                self.store.put_task("t1", "smoke", {"q": 1})
                raise StoreError("simulated crash")
        self.assertEqual(self.store.snapshot()["run"]["state"], "INIT")
        self.assertIsNone(self.store.task("t1"))

    def test_reopen_keeps_committed_state(self):
        self.store.put_task("t1", "smoke", {"q": 1})
        self.store.set_state("PREFLIGHT", "start")
        self.reopen()
        self.assertEqual(self.store.snapshot()["run"]["state"], "PREFLIGHT")
        self.assertEqual(self.store.task("t1")["status"], "pending")
        with self.store.transaction() as conn:
            self.assertIsNotNone(conn.execute("SELECT 1").fetchone())
            with self.assertRaises(StoreError):
                with self.store.transaction():
                    pass


class TestStateMachine(StoreCase):
    def test_corrupt_state_is_not_treated_as_a_new_or_successful_run(self):
        with self.store.transaction() as conn:
            conn.execute("UPDATE run SET state='UNKNOWN_CORRUPT' WHERE singleton=1")
        with self.assertRaises(StoreError):
            self.store.snapshot()

    def test_full_chain(self):
        chain = ["PREFLIGHT", "INPUT_READY", "SMOKE", "HUMAN_GATE_SMOKE", "BATCH",
                 "FINAL_QC", "ARCHIVE", "DONE"]
        for state in chain:
            self.store.set_state(state, f"enter {state}")
        self.assertEqual(self.store.snapshot()["run"]["state"], "DONE")
        with self.assertRaises(StoreError):
            self.store.set_state("BATCH", "after done")

    def test_illegal_skips_and_backwards_refused(self):
        with self.assertRaises(StoreError):
            self.store.set_state("BATCH", "skip ahead")
        self.store.set_state("PREFLIGHT", "ok")
        with self.assertRaises(StoreError):
            self.store.set_state("INIT", "backwards")
        with self.assertRaises(StoreError):
            self.store.set_state("SMOKE", "not next")

    def test_interruptions_resume_only_to_saved_state(self):
        self.store.set_state("PREFLIGHT", "ok")
        self.store.set_state("INPUT_READY", "inputs frozen")
        self.store.set_state("PAUSED", "user pause", stage="prepare")
        self.assertEqual(self.store.snapshot()["run"]["resume_state"], "INPUT_READY")
        with self.assertRaises(StoreError):
            self.store.set_state("SMOKE", "skip while paused")
        self.store.set_state("HUMAN_GATE", "unknown request")
        self.assertEqual(self.store.snapshot()["run"]["resume_state"], "INPUT_READY")
        resumed = self.store.set_state("INPUT_READY", "resumed")
        self.assertEqual(resumed["state"], "INPUT_READY")
        self.assertIsNone(resumed["resume_state"])

    def test_fatal_error_is_terminal(self):
        self.store.set_state("PREFLIGHT", "ok")
        self.store.set_state("FATAL_ERROR", "ledger unreadable")
        run = self.store.snapshot()["run"]
        self.assertEqual(run["state"], "FATAL_ERROR")
        self.assertIsNone(run["resume_state"])
        for target in ("PREFLIGHT", "SMOKE", "PAUSED", "RECOVERABLE_ERROR", "HUMAN_GATE", "DONE"):
            with self.assertRaises(StoreError, msg=target):
                self.store.set_state(target, "resume from fatal")
        self.assertEqual(self.store.snapshot()["run"]["state"], "FATAL_ERROR")

    def test_done_cannot_be_re_entered_from_active_state(self):
        self.store.set_state("PREFLIGHT", "ok")
        with self.assertRaises(StoreError):
            self.store.set_state("DONE", "skip ahead")

    def test_transition_events_record_reason_and_scope(self):
        self.store.set_state("PREFLIGHT", "start", stage="prepare", task_id="p1")
        events = self.store._events()
        self.assertEqual(events[-1]["kind"], "state")
        self.assertEqual(events[-1]["payload"]["reason"], "start")
        self.assertEqual(events[-1]["payload"]["stage"], "prepare")
        self.assertEqual(events[-1]["payload"]["task_id"], "p1")


class TestTasksAndArtifacts(StoreCase):
    def test_identity_is_immutable_but_repeat_is_idempotent(self):
        self.store.put_task("t1", "smoke", {"question": "q1"})
        again = self.store.put_task("t1", "smoke", {"question": "q1"})
        self.assertEqual(again["status"], "pending")
        with self.assertRaises(StoreError):
            self.store.put_task("t1", "batch", {"question": "q1"})
        with self.assertRaises(StoreError):
            self.store.put_task("t1", "smoke", {"question": "q2"})

    def test_valid_task_survives_reopen_but_not_tampering(self):
        path = self.artifact()
        self.store.put_task("t1", "smoke", {"question": "q1"})
        self.store.finish_task("t1", "succeeded", [path])
        self.assertTrue(self.reopen().valid_task("t1"))
        self.assertEqual(self.store.task("t1")["artifacts"][0]["path"], "outputs/result.json")

        path.write_text('{"a": 2}', encoding="utf-8")
        self.assertFalse(self.store.valid_task("t1"))
        path.unlink()
        self.assertFalse(self.store.valid_task("t1"))

    def test_empty_or_missing_artifacts_and_failed_tasks_are_not_valid(self):
        empty = self.artifact("outputs/empty.txt", "")
        self.store.put_task("t-empty", "smoke", {})
        self.store.finish_task("t-empty", "succeeded", [empty])
        self.assertFalse(self.store.valid_task("t-empty"))

        self.store.put_task("t-fail", "smoke", {})
        self.store.finish_task("t-fail", "failed", [], error="parse error")
        self.assertFalse(self.store.valid_task("t-fail"))
        self.assertIsNone(self.store.task("nope"))

    def test_succeeded_without_artifacts_is_not_valid(self):
        self.store.put_task("t-bare", "smoke", {})
        self.store.finish_task("t-bare", "succeeded", [])
        self.assertEqual(self.store.task("t-bare")["status"], "succeeded")
        self.assertFalse(self.store.valid_task("t-bare"))

    def test_every_completion_records_artifact_lineage(self):
        first = self.artifact("outputs/v1.json")
        self.store.put_task("t1", "examiner", {"q": 1})
        self.store.finish_task("t1", "succeeded", [first])
        second = self.artifact("outputs/v2.json", '{"b": 2}')
        self.store.finish_task("t1", "succeeded", [second])

        finished = [e["payload"] for e in self.store._events() if e["kind"] == "task_finished"]
        self.assertEqual(len(finished), 2)
        self.assertEqual(finished[0]["previous_status"], "pending")
        self.assertEqual(finished[0]["previous_artifacts"], [])
        self.assertEqual(finished[0]["artifacts"][0]["path"], "outputs/v1.json")
        self.assertEqual(finished[0]["artifacts"][0]["sha256"], hashlib.sha256(b'{"a": 1}').hexdigest())
        self.assertEqual(finished[1]["previous_artifacts"][0]["path"], "outputs/v1.json")
        self.assertEqual(finished[1]["artifacts"][0]["path"], "outputs/v2.json")
        self.assertEqual(finished[1]["artifacts"][0]["sha256"], self.store.task("t1")["artifacts"][0]["sha256"])
        self.assertIsNone(finished[0]["error"])

    def test_path_escape_refused(self):
        outside = Path(self._tmp.name) / "outside.json"
        outside.write_text("{}", encoding="utf-8")
        self.store.put_task("t1", "smoke", {})
        with self.assertRaises(StoreError):
            self.store.finish_task("t1", "succeeded", [outside])
        self.assertEqual(self.store.task("t1")["artifacts"], [])

    def test_escaping_artifact_recorded_directly_fails_validation(self):
        path = self.artifact()
        self.store.put_task("t1", "smoke", {})
        self.store.finish_task("t1", "succeeded", [path])
        with self.store.transaction() as conn:
            conn.execute("UPDATE tasks SET artifacts=? WHERE task_id='t1'",
                         (json.dumps([{"path": "../outside.json", "sha256": "0" * 64}]),))
        self.assertFalse(self.store.valid_task("t1"))

    def test_tasks_filter_by_stage(self):
        self.store.put_task("a", "smoke", {})
        self.store.put_task("b", "batch", {})
        self.assertEqual([t["task_id"] for t in self.store.tasks("batch")], ["b"])
        self.assertEqual(len(self.store.tasks()), 2)


class TestDamagedLedger(StoreCase):
    def test_missing_ledger_fails_without_create(self):
        with self.assertRaises(StoreError):
            Store(self.run_dir / "absent")

    def test_corrupt_file_fails_and_is_not_replaced(self):
        self.store.close()
        self.store.path.write_bytes(b"this is not a database" * 100)
        with self.assertRaises(StoreError):
            Store(self.run_dir)
        with self.assertRaises(StoreError):
            Store(self.run_dir, create=True)
        self.assertEqual(self.store.path.read_bytes()[:9], b"this is n")

    def test_schema_version_mismatch_fails(self):
        self.store.close()
        conn = sqlite3.connect(self.store.path)
        try:
            conn.execute("PRAGMA user_version = 99")
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(StoreError):
            Store(self.run_dir)

    def test_missing_table_fails(self):
        self.store.close()
        conn = sqlite3.connect(self.store.path)
        try:
            conn.execute("DROP TABLE gates")
            conn.commit()
        finally:
            conn.close()
        with self.assertRaises(StoreError):
            Store(self.run_dir)

    def test_integrity_check_surfaces_page_damage(self):
        self.store.put_task("t1", "smoke", {"question": "q1"})
        self.assertEqual(self.store.check_integrity(), "ok")
        self.store.close()
        path = Path(self.store.path)
        page_size = int.from_bytes(path.read_bytes()[16:18], "big")
        with path.open("r+b") as handle:
            handle.seek(page_size)  # second page: wreck its b-tree header
            handle.write(b"\xff" * 12)
        with self.assertRaises(StoreError):
            Store(self.run_dir)

    def test_create_does_not_adopt_existing_empty_file(self):
        self.store.close()
        self.store.path.write_bytes(b"")
        with self.assertRaises(StoreError):
            Store(self.run_dir, create=True)
        self.assertEqual(self.store.path.stat().st_size, 0)

    def test_create_does_not_adopt_unrelated_sqlite_file(self):
        self.store.close()
        self.store.path.unlink()
        conn = sqlite3.connect(self.store.path)
        try:
            conn.execute("CREATE TABLE somebody_elses_data (x INTEGER)")
            conn.execute("INSERT INTO somebody_elses_data VALUES (1)")
            conn.commit()
        finally:
            conn.close()
        before = self.store.path.read_bytes()
        with self.assertRaises(StoreError):
            Store(self.run_dir, create=True)
        self.assertEqual(self.store.path.read_bytes(), before)
        conn = sqlite3.connect(self.store.path)
        try:
            self.assertTrue(conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='somebody_elses_data'").fetchone())
            self.assertIsNone(conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='run'").fetchone())
        finally:
            conn.close()

    def test_existing_valid_ledger_reopens_under_create(self):
        self.store.put_task("t1", "smoke", {"question": "q1"})
        self.reopen()
        again = Store(self.run_dir, create=True)
        self.addCleanup(again.close)
        self.assertEqual(again.snapshot()["tasks"]["by_status"], {"pending": 1})
        self.assertEqual(again.snapshot()["run"]["spec_hash"], "spec-abc")

    def test_concurrent_creates_produce_one_ledger(self):
        target = Path(self._tmp.name) / "race"
        winner = Store(target, create=True)
        self.addCleanup(winner.close)
        winner.initialize("spec-abc")
        # A second creator arriving late must adopt, not reinitialize.
        loser = Store(target, create=True)
        self.addCleanup(loser.close)
        self.assertEqual(loser.snapshot()["run"]["spec_hash"], "spec-abc")
        loser.put_task("t1", "smoke", {})
        self.assertEqual(winner.snapshot()["tasks"]["by_status"], {"pending": 1})

    def test_create_makes_nested_run_directory(self):
        target = Path(self._tmp.name) / "nested" / "run"
        store = Store(target, create=True)
        self.addCleanup(store.close)
        store.initialize("spec-abc")
        self.assertTrue((target / DB_NAME).is_file())


class TestPause(StoreCase):
    def test_pause_flag_survives_reopen(self):
        self.assertFalse(self.store.paused_requested())
        self.store.request_pause()
        self.assertTrue(self.reopen().paused_requested())
        self.store.clear_pause()
        self.assertFalse(self.store.paused_requested())


class TestGates(StoreCase):
    def test_approval_is_single_shot_and_reasoned(self):
        gate_id = self.store.create_gate("smoke", "run-1", "smoke report ready")
        with self.assertRaises(StoreError):
            self.store.approve_gate(gate_id, {"reason": "  "})
        approved = self.store.approve_gate(gate_id, {"reason": "reviewed", "by": "voss"})
        self.assertEqual(approved["status"], "approved")
        self.assertEqual(approved["decision"]["by"], "voss")
        with self.assertRaises(StoreError):
            self.store.approve_gate(gate_id, {"reason": "changed my mind"})
        with self.assertRaises(StoreError):
            self.store.approve_gate("no-such-gate", {"reason": "typo"})
        with self.assertRaises(StoreError):
            self.store.create_gate("smoke", "run-1", "")
        self.assertIn("gate_approved", [e["kind"] for e in self.store._events()])


class TestViews(StoreCase):
    def test_export_reflects_canonical_db(self):
        self.store.put_task("t1", "smoke", {})
        self.store.finish_task("t1", "failed", [], error="boom")
        self.store.event("error_parse", {"task_id": "t1", "error": "boom"})
        (self.run_dir / "state.json").write_text("stale", encoding="utf-8")
        self.store.export_views()
        state = json.loads((self.run_dir / "state.json").read_text(encoding="utf-8"))
        self.assertEqual(state["tasks"]["by_status"], {"failed": 1})
        events = (self.run_dir / "events.jsonl").read_text(encoding="utf-8").splitlines()
        errors = (self.run_dir / "errors.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(errors), 2)
        self.assertEqual(len(events), len(errors) + 1)
        self.assertEqual(json.loads(errors[-1])["kind"], "error_parse")
        self.assertEqual(list(self.run_dir.glob("*.part")), [])


if __name__ == "__main__":
    unittest.main()
