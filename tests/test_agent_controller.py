from __future__ import annotations

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from lumina import prompts
from lumina.agent.controller import approve, begin, execute
from lumina.agent.lease import RunLease
from lumina.agent.store import Store
from test_agent_contracts import config_for, request_for


class ControllerTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def request(self, domain, count=1):
        papers = []
        for index in range(count):
            paper = self.root / f"{domain}-{index}.md"
            paper.write_text(f"# Introduction\nChina. Fish. Flux is 2. Table 2. Paper {index}.", encoding="utf-8")
            papers.append(str(paper))
        request = request_for(papers[0], domain)
        request["papers"] = papers
        return request

    def client(self, domain):
        client = Mock()

        def create(**payload):
            messages = payload["messages"]
            if messages[0]["content"] == prompts.message_system_ragQuery.strip():
                content = dict(existing_flag=1, direct_quote="China")
            else:
                question = [q.strip("\n") for q in prompts.questions_for_domain(domain)].index(messages[-1]["content"]) + 1
                item = "Study_location" if question == 1 else ("Specie" if domain == "aqua" and question == 2 else "Flux-1" if domain == "aqua" else "Forest_smoldering")
                answer = dict(value="China" if question == 1 else "fish" if item == "Specie" else 2,
                              evidence="China", confidence_lv=95)
                if domain == "aqua" and question == 3:
                    answer["unit"] = "mg m-2 h-1"
                content = {item: answer}
            response = dict(choices=[dict(message=dict(content=json.dumps(content)), finish_reason="stop")],
                            usage=dict(prompt_tokens=12, completion_tokens=5, total_tokens=17))
            return SimpleNamespace(model_dump=lambda **_: response)

        client.chat.completions.create.side_effect = create
        return client

    def embedding(self):
        response = Mock()
        response.json.return_value = dict(data=[dict(embedding=[1., 0.])], usage=dict(prompt_tokens=3, total_tokens=3))
        return response

    def test_dry_run_creates_plan_and_dispatches_nothing(self):
        with patch("lumina.agent.runtime.OpenAI") as sdk, patch("lumina.agent.runtime.requests.post") as post:
            run = begin(self.request("aqua"), config_for(), self.root / "runs", dry_run=True)
            with Store(run) as store:
                self.assertEqual(store.snapshot()["run"]["state"], "INPUT_READY")
                self.assertEqual(len(store.tasks("examiner")), 6)
            sdk.assert_not_called()
            post.assert_not_called()
        self.assertTrue((run / "reports" / "plan_report.json").is_file())

    def test_both_domains_smoke_gate_batch_reuse_and_final_qc(self):
        for domain in ("aqua", "wildfire"):
            with self.subTest(domain=domain):
                config = config_for(domain)
                client = self.client(domain)
                with patch("lumina.agent.runtime.OpenAI", return_value=client), \
                     patch("lumina.agent.runtime.requests.post", return_value=self.embedding()) as post, \
                     patch("lumina.agent.runtime.RunContext.sleep"):
                    run = begin(self.request(domain, count=2), config, self.root / "runs", dry_run=True)
                    first = execute(run, config)
                    self.assertEqual(first["state"], "HUMAN_GATE_SMOKE", first)
                    counts = (client.chat.completions.create.call_count, post.call_count)
                    again = execute(run, config)
                    self.assertEqual(again["state"], "HUMAN_GATE_SMOKE")
                    self.assertEqual((client.chat.completions.create.call_count, post.call_count), counts)
                    with Store(run) as store:
                        gate = next(g for g in store.gates() if g["kind"] == "smoke")
                    approve(run, gate["gate_id"], "Reviewed synthetic smoke", config=config)
                    final = execute(run, config)
                    self.assertEqual(final["state"], "DONE", final)
                    # Exactly one paper's worth of new requests; smoke tasks were retained.
                    self.assertEqual(client.chat.completions.create.call_count, counts[0] * 2)
                    # New paper chunks cost one request; the identical evidence query is shared.
                    self.assertEqual(post.call_count, counts[1] + 1)
                    saved = (client.chat.completions.create.call_count, post.call_count)
                    self.assertEqual(execute(run, config)["state"], "DONE")
                    self.assertEqual((client.chat.completions.create.call_count, post.call_count), saved)
                report = json.loads((run / "reports" / "final_report.json").read_text(encoding="utf-8"))
                self.assertEqual(report["scientific_validation"], "NOT_EVALUATED")
                self.assertEqual(report["extraction"]["completed"], report["extraction"]["expected"])
                self.assertTrue(report["final_qc"]["passed"])

    def test_os_lock_cannot_be_reclaimed_by_expired_heartbeat(self):
        run = self.root / "lease"
        run.mkdir()
        script = "from lumina.agent.lease import RunLease; from pathlib import Path; import sys;\ntry:\n with RunLease(Path(sys.argv[1])): pass\nexcept RuntimeError: sys.exit(23)"
        with RunLease(run) as lease:
            lease.owner["heartbeat"] = -999999
            result = subprocess.run([sys.executable, "-c", script, str(run)], capture_output=True, timeout=10)
            self.assertEqual(result.returncode, 23, result.stderr.decode())
        result = subprocess.run([sys.executable, "-c", script, str(run)], capture_output=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr.decode())

    def test_human_decision_survives_interruption_before_accounting(self):
        config = config_for()
        client = self.client("aqua")
        successful = client.chat.completions.create.side_effect
        client.chat.completions.create.side_effect = requests.Timeout("provider outcome unknown")
        run = begin(self.request("aqua"), config, self.root / "runs", dry_run=True)
        with patch("lumina.agent.runtime.OpenAI", return_value=client), patch("lumina.agent.runtime.RunContext.sleep"):
            self.assertEqual(execute(run, config)["state"], "HUMAN_GATE")
        with Store(run) as store:
            gate = next(g for g in store.gates() if g["kind"] == "unknown_request")
        with patch("lumina.agent.controller.Budget.reconcile", side_effect=RuntimeError("simulated interruption")):
            with self.assertRaises(RuntimeError):
                approve(run, gate["gate_id"], "Provider log confirms non-execution", outcome="not_executed")
        client.chat.completions.create.side_effect = successful
        with patch("lumina.agent.runtime.OpenAI", return_value=client), \
             patch("lumina.agent.runtime.requests.post", return_value=self.embedding()), \
             patch("lumina.agent.runtime.RunContext.sleep"):
            result = execute(run, config)
            self.assertEqual(result["state"], "HUMAN_GATE_SMOKE", result)
        with Store(run) as store:
            row = store._conn.execute("SELECT status,receipt FROM attempts WHERE attempt_id=?", (gate["target"],)).fetchone()
            self.assertEqual(row["status"], "rejected")
            self.assertEqual(json.loads(row["receipt"])["reconciled"]["outcome"], "not_executed")

    def test_hard_process_exit_keeps_dispatch_unknown_and_os_lock_recoverable(self):
        config = config_for()
        run = begin(self.request("aqua"), config, self.root / "runs", dry_run=True)
        configfile = self.root / "synthetic_config.json"
        configfile.write_text(json.dumps(vars(config)), encoding="utf-8")
        with Store(run) as store:
            store.set_state("SMOKE", "simulate worker dispatch boundary")
        script = (
            "import json,os,sys; from pathlib import Path; from types import SimpleNamespace; "
            "from lumina.agent.runtime import RunContext; from lumina.agent.lease import RunLease; "
            "run=Path(sys.argv[1]); cfg=SimpleNamespace(**json.loads(Path(sys.argv[2]).read_text()));\n"
            "with RunLease(run):\n"
            " rt=RunContext(run,cfg)\n"
            " task=next(t for t in rt.store.tasks('examiner') if t['payload']['question']==1)\n"
            " attempt=rt.budget.reserve(task['task_id'],'hard-crash-request','chat',task['payload']['source_model'],10000,1000)\n"
            " rt.budget.dispatch(attempt)\n"
            " os._exit(37)\n"
        )
        child = subprocess.run([sys.executable, "-c", script, str(run), str(configfile)], capture_output=True, timeout=15)
        self.assertEqual(child.returncode, 37, child.stderr.decode())
        with patch("lumina.agent.runtime.OpenAI") as sdk, patch("lumina.agent.runtime.requests.post") as http:
            self.assertEqual(execute(run, config)["state"], "HUMAN_GATE")
            sdk.assert_not_called()
            http.assert_not_called()
        with Store(run) as store:
            row = store._conn.execute("SELECT status,actual_cost FROM attempts").fetchone()
            self.assertEqual(row["status"], "unknown")
            self.assertIsNone(row["actual_cost"])
            self.assertEqual(len(store.gates()), 1)

    def test_soft_pause_resume_preserves_start_time_and_does_not_dispatch_while_paused(self):
        config = config_for()
        run = begin(self.request("aqua"), config, self.root / "runs", dry_run=True)
        with Store(run) as store:
            started_at = store.snapshot()["run"]["started_at"]
            store.request_pause()
        with patch("lumina.agent.runtime.OpenAI") as sdk:
            self.assertEqual(execute(run, config)["state"], "PAUSED")
            sdk.assert_not_called()
        with patch("lumina.agent.runtime.OpenAI", return_value=self.client("aqua")), \
             patch("lumina.agent.runtime.requests.post", return_value=self.embedding()), \
             patch("lumina.agent.runtime.RunContext.sleep"):
            self.assertEqual(execute(run, config)["state"], "HUMAN_GATE_SMOKE")
        with Store(run) as store:
            self.assertEqual(store.snapshot()["run"]["started_at"], started_at)

    def test_corrupt_ledger_is_fatal_and_never_recreated(self):
        config = config_for()
        run = begin(self.request("aqua"), config, self.root / "runs", dry_run=True)
        corrupt = b"corrupt canonical authority"
        (run / "ledger.sqlite").write_bytes(corrupt)
        with patch("lumina.agent.runtime.OpenAI") as sdk:
            self.assertEqual(execute(run, config)["state"], "FATAL_ERROR")
            sdk.assert_not_called()
        self.assertEqual((run / "ledger.sqlite").read_bytes(), corrupt)
        self.assertTrue((run / "reports" / "fatal_report.json").is_file())


if __name__ == "__main__":
    unittest.main()
