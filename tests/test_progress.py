"""Focused tests for the read-only live progress layer."""
from __future__ import annotations

import io
import json
import shutil
import tempfile
import time
import unittest
from pathlib import Path

from lumina.agent.store import Store
from lumina.progress import ProgressMonitor, accepted_round_data, format_progress, format_round_summary


class ProgressTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.run = Path(self.temp.name) / "run"
        self.run.mkdir()
        (self.run / "reports").mkdir()
        (self.run / "outputs" / "R01" / "ensemble" / "MiniCross01").mkdir(parents=True)
        spec = {
            "domain": "aqua", "papers": [{"paper_uid": "p1"}, {"paper_uid": "p2"}],
            "models": [{"model": "m1"}, {"model": "m2"}],
            "questions": [{"index": 1}, {"index": 2}], "rounds": [1],
            "smoke_paper_uids": ["p1"], "run": {"min_cross_scores": [1]},
        }
        (self.run / "manifest.json").write_text(json.dumps({"specification": spec}), encoding="utf-8")
        with Store(self.run, create=True) as store:
            store.initialize("progress-spec")

    def _artifact_task(self, store, stage, payload, name):
        path = self.run / "outputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(name, encoding="utf-8")
        task_id = name.replace("/", "_")
        store.put_task(task_id, stage, payload)
        store.finish_task(task_id, "succeeded", [path])
        return task_id

    def test_monitor_is_read_only_and_isolates_trial_scope(self):
        with Store(self.run) as store:
            store.set_state("PREFLIGHT", "test")
            store.set_state("INPUT_READY", "test")
            store.set_state("SMOKE", "test")
            store.event("stage_started", {"stage": "examiner", "round_index": 1,
                                           "scope": ["p1"], "phase": "trial"})
            p1_task = self._artifact_task(store, "examiner", {"paper_uid": "p1", "question": 1, "round_index": 1}, "p1-q1")
            p2_task = self._artifact_task(store, "examiner", {"paper_uid": "p2", "question": 1, "round_index": 1}, "p2-q1")
            store.event("cache_hit", {"task_id": p1_task, "stage": "examiner"})
            store.event("cache_hit", {"task_id": p2_task, "stage": "examiner"})
            before = store.snapshot()
        monitor = ProgressMonitor(self.run, language="zh", stream=io.StringIO())
        result = monitor.poll()
        with Store(self.run) as reopened:
            after = reopened.snapshot()
        self.assertEqual(before["run"]["state"], after["run"]["state"])
        self.assertEqual(result["snapshot"]["phase"], "trial")
        self.assertEqual(result["snapshot"]["scope"], ["p1"])
        self.assertEqual(result["snapshot"]["extraction"]["expected"], 4)
        self.assertEqual(result["snapshot"]["extraction"]["completed"], 1)
        self.assertEqual(result["snapshot"]["cache_reuse"], 1)
        self.assertEqual(result["snapshot"]["events"][0]["kind"], "state")

    def test_cost_uses_canonical_accounting_for_imports_and_unknown_cost(self):
        with Store(self.run) as store:
            now = time.time()
            store._conn.executemany(
                "INSERT INTO attempts(attempt_id, task_id, request_hash, kind, status, reserved_tokens, "
                "reserved_cost, actual_tokens, actual_cost, raw_path, receipt, created_at, updated_at) "
                "VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                [
                    ("unknown", "task", "hash", "chat", "received", 10, None, 3, None, None,
                     json.dumps({"model": "vendor/model", "dispatched_at": now}), now, now),
                    ("imported", "task2", "hash2", "chat", "imported", 10, 5.0, 3, 1.0, None,
                     json.dumps({"model": "vendor/model", "imported_from": {"run": "source"}}), now, now),
                ])
        snapshot = ProgressMonitor(self.run).snapshot()
        self.assertIsNone(snapshot["cost"]["total_cost"])
        self.assertEqual(snapshot["cost"]["unpriced_requests"], 1)
        self.assertFalse(snapshot["cost"]["cost_complete"])

    def test_long_wait_heartbeat_without_event_growth(self):
        stream = io.StringIO()
        monitor = ProgressMonitor(self.run, language="en", stream=stream)
        self.assertTrue(monitor.emit(now=0.0))
        self.assertFalse(monitor.emit(now=1.0))
        self.assertTrue(monitor.emit(now=16.0))
        self.assertEqual(len(stream.getvalue().splitlines()), 2)

    def test_malformed_presentation_data_is_a_display_gap(self):
        path = self.run / "outputs" / "R01" / "ensemble" / "MiniCross01" / "aqua_Ensemble_Result_Q01_MiniCross01.xlsx"
        path.write_bytes(b"not an xlsx")
        frozen = self.run / "checkpoints" / "stages" / "trial-ensemble" / path.name
        frozen.parent.mkdir(parents=True)
        shutil.copyfile(path, frozen)
        spec = json.loads((self.run / "manifest.json").read_text(encoding="utf-8"))["specification"]
        data = accepted_round_data(self.run, spec, 1,
                                   frozen_artifacts=[str(frozen.relative_to(self.run))])
        self.assertIsNone(data["MiniCross01"]["rows"])
        self.assertIn("error", data["MiniCross01"])

    def test_round_output_includes_paths_and_at_most_five_previews(self):
        import pandas as pd

        path = self.run / "outputs" / "R01" / "ensemble" / "MiniCross01" / "aqua_Ensemble_Result_Q01_MiniCross01.xlsx"
        pd.DataFrame([{"item": str(i)} for i in range(7)]).to_excel(path, index=False)
        frozen = self.run / "checkpoints" / "stages" / "trial-ensemble" / path.name
        frozen.parent.mkdir(parents=True)
        shutil.copyfile(path, frozen)
        with Store(self.run) as store:
            store.event("round_finished", {"round_index": 1, "scope": ["p1"], "phase": "trial",
                                            "paths": ["outputs/R01/ensemble/MiniCross01/result.xlsx"],
                                            "frozen_artifacts": {"ensemble": [str(frozen.relative_to(self.run))]},
                                            "budget_start": {"calls": 1}, "budget_delta": {"calls": 2,
                                                "known_tokens": 30, "known_cost": 0.12, "unpriced_requests": 1},
                                            "new_requests": 2, "new_usage": {"known_tokens": 30,
                                                "known_cost": 0.12, "unpriced_requests": 1},
                                            "min_consensus_models": 2,
                                            "counters": {"extraction": 4, "cross": 2, "success": 5,
                                                         "failed": 1, "unresolved": 1, "scoped_cache_events": 2}})
        monitor = ProgressMonitor(self.run)
        snapshot = monitor.snapshot()
        self.assertEqual(snapshot["rounds"][0]["accepted"]["MiniCross01"]["rows"], 7)
        self.assertLessEqual(len(snapshot["rounds"][0]["accepted"]["MiniCross01"]["previews"]), 5)
        pd.DataFrame([{"item": "batch-overwrite"}]).to_excel(path, index=False)
        updated = monitor.snapshot()
        self.assertEqual(updated["rounds"][0]["accepted"]["MiniCross01"]["rows"], 7)
        rendered = format_round_summary(snapshot, 1, language="en")
        self.assertIn("round finished R01", rendered)
        self.assertIn("outputs/R01/ensemble/MiniCross01/result.xlsx", rendered)
        self.assertIn("accepted rows=7", rendered)
        self.assertIn("new requests=2", rendered)
        self.assertIn("success=5 failed=1 unresolved=1 cache=2", rendered)
        self.assertIn("min_consensus_models=2", rendered)

    def test_formatting_keeps_cost_subtotal_and_unknown_cross_denominator(self):
        snapshot = {"available": True, "state": "SMOKE", "phase": "trial", "round": 1,
                    "scope": ["p1"], "current": {"stage": "cross"},
                    "extraction": {"completed": 1, "expected": 4},
                    "cross": {"candidates": None, "verifier_pairs": None, "completed_pairs": None},
                    "cache_reuse": 2, "waiting": {"kind": "model_response"},
                    "cost": {"known_cost_subtotal": 0.25, "unpriced_requests": 1}}
        line = format_progress(snapshot, language="en")
        self.assertIn("known cost subtotal 0.25", line)
        self.assertIn("cross pairs ?", line)
        self.assertIn("unpriced requests=1", line)


if __name__ == "__main__":
    unittest.main()
