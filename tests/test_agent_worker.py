from __future__ import annotations

import sys
import tempfile
import time
import unittest
from pathlib import Path

from lumina.agent.contracts import ResearchSpecification, create_run
from lumina.agent.store import Store
from lumina.agent.worker import supervise
from test_agent_contracts import config_for, request_for


class WorkerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.config = config_for()
        self.paper = self.root / "paper.md"
        self.paper.write_text("China. Flux is 2.", encoding="utf-8")

    def run_dir(self, seconds):
        request = request_for(self.paper)
        request["budget"]["max_runtime"] = seconds
        spec = ResearchSpecification.from_config(request, self.config)
        run = create_run(spec, self.root / "runs")
        with Store(run, create=True) as store:
            store.initialize(spec.fingerprint)
            for state in ("PREFLIGHT", "INPUT_READY", "SMOKE"):
                store.set_state(state, "fixture")
        return run

    def test_blocked_child_is_terminated_and_dispatched_attempt_stays_unknown(self):
        run = self.run_dir(2)
        marker = self.root / "child-marker.json"
        script = (
            "import os,sys,time,json; from pathlib import Path; "
            "from lumina.agent.store import Store; from lumina.agent.budget import Budget; "
            "from lumina.agent.lease import RunLease; run=Path(sys.argv[1]);\n"
            "with RunLease(run), Store(run) as store:\n"
            " spec=json.loads((run/'manifest.json').read_text())['specification']\n"
            " budget=Budget(store,spec)\n"
            " attempt=budget.reserve('fixture-task','request','chat','mock/Model_A',10000,1000)\n"
            " budget.dispatch(attempt)\n"
            " Path(sys.argv[2]).write_text(json.dumps({'pid':os.getpid(),'attempt':attempt}))\n"
            " time.sleep(30)\n"
        )
        start = time.monotonic()
        result = supervise(run, self.root / "unused.py", worker_command=[sys.executable, "-c", script, str(run), str(marker)])
        self.assertLess(time.monotonic() - start, 8)
        self.assertTrue(marker.is_file())
        self.assertEqual(result["state"], "HUMAN_GATE")
        self.assertEqual(result["budget"]["calls"], 1)
        self.assertEqual(result["budget"]["unknown_requests"], 1)
        self.assertGreater(result["budget"]["held_cost"], 0)
        with Store(run) as store:
            self.assertEqual({g["kind"] for g in store.gates()}, {"unknown_request", "budget"})
            self.assertEqual(store._conn.execute("SELECT status FROM attempts").fetchone()["status"], "unknown")

    def test_expired_deadline_admits_no_worker(self):
        run = self.run_dir(1)
        with Store(run) as store, store.transaction() as conn:
            conn.execute("UPDATE run SET started_at=started_at-10")
        marker = self.root / "should-not-exist"
        result = supervise(run, self.root / "unused.py", worker_command=[sys.executable, "-c", "from pathlib import Path; import sys; Path(sys.argv[1]).touch()", str(marker)])
        self.assertEqual(result["state"], "HUMAN_GATE")
        self.assertFalse(marker.exists())
        self.assertEqual(result["budget"]["calls"], 0)

    def test_missing_worker_receipt_is_not_success(self):
        run = self.run_dir(30)
        with self.assertRaisesRegex(RuntimeError, "control receipt"):
            supervise(run, self.root / "unused.py", worker_command=[sys.executable, "-c", "raise SystemExit(3)"])


if __name__ == "__main__":
    unittest.main()
