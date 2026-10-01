from __future__ import annotations

import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

from lumina import examiner
from lumina.agent.budget import Budget
from lumina.agent.controller import begin, execute
from lumina.agent.contracts import TaskKey
from lumina.agent.imports import import_task
from lumina.agent.runtime import RunContext
from lumina.agent.store import Store
from test_agent_contracts import config_for, request_for
import test_agent_controller as controller_tests


class ImportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        paper = self.root / "paper.md"
        paper.write_text("China. Fish. Flux is 2. Table 2.", encoding="utf-8")
        self.config = config_for()
        self.request = request_for(paper)
        self.source = begin(self.request, self.config, self.root / "runs", dry_run=True, run_id="source")
        self.target = begin(self.request, self.config, self.root / "runs", dry_run=True, run_id="target")
        helper = controller_tests.ControllerTests()
        client = helper.client("aqua")
        with patch("lumina.agent.runtime.OpenAI", return_value=client), \
             patch("lumina.agent.runtime.requests.post", return_value=helper.embedding()), \
             patch("lumina.agent.runtime.RunContext.sleep"):
            self.assertEqual(execute(self.source, self.config)["state"], "HUMAN_GATE_SMOKE")
        with Store(self.source) as store:
            self.task = next(t for t in store.tasks("examiner") if t["payload"].get("question") == 1)
            self.source_totals = Budget(store, self._spec(self.source)).totals()

    def _spec(self, directory):
        import json
        return json.loads((directory / "manifest.json").read_text(encoding="utf-8"))["specification"]

    def test_explicit_import_keeps_provenance_and_replays_without_new_payment(self):
        result = import_task(self.source, self.target, self.task["task_id"], reason="reuse verified synthetic task")
        key = replace(TaskKey(**self.task["payload"]), run_id=self.target.name)
        self.assertEqual(result["tasks"], [key.task_id])
        context = RunContext(self.target, self.config)
        self.addCleanup(context.close)
        self.assertTrue(context.valid(key))
        model = next(m for m in self.config.FULL_LLM_POOL.values() if m["model"] == key.source_model)
        paper = next((self.target / "inputs").glob("*.md"))
        from lumina import prompts
        with patch("lumina.agent.runtime.OpenAI") as sdk, context.task_scope(key):
            answer, _ = examiner.run_llm_prompt_mode(model, self.config.LLM_SETTINGS,
                paper.read_text(encoding="utf-8"), key.paper_uid, key.question,
                prompts.questions_for_domain("aqua")[key.question - 1], "test", runtime=context)
            self.assertIn("China", answer)
            sdk.assert_not_called()
        self.assertEqual(context.budget.totals()["calls"], 0)
        self.assertEqual(context.budget.totals()["known_cost"], 0)
        self.assertEqual(context.budget.totals()["imports"], 1)
        with Store(self.source) as store:
            after = Budget(store, self._spec(self.source)).totals()
        self.assertEqual(after["calls"], self.source_totals["calls"])
        self.assertEqual(after["known_cost"], self.source_totals["known_cost"])

    def test_signature_drift_and_changed_source_are_refused(self):
        changed_config = config_for()
        changed_config.RUN["temperature"] = .2
        changed = begin(self.request, changed_config, self.root / "runs", dry_run=True, run_id="different")
        with self.assertRaisesRegex(ValueError, "signatures"):
            import_task(self.source, changed, self.task["task_id"], reason="test")
        artifact = self.source / self.task["artifacts"][0]["path"]
        artifact.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "artifacts"):
            import_task(self.source, self.target, self.task["task_id"], reason="test")


if __name__ == "__main__":
    unittest.main()
