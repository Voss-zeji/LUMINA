from __future__ import annotations

import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import run_pipeline
from lumina.agent.cli import main
from lumina.agent.controller import begin
from test_agent_contracts import config_for, request_for


class CLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paper = self.root / "paper.md"
        self.paper.write_text("China. Flux is 2.", encoding="utf-8")
        self.config = config_for()

    def invoke(self, args):
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(args)
        return code, json.loads(output.getvalue())

    def test_run_dry_status_pause_and_safe_paths_without_network(self):
        specfile = self.root / "spec.json"
        specfile.write_text(json.dumps(request_for(self.paper)), encoding="utf-8")
        with patch.object(run_pipeline, "load_config", return_value=self.config), \
             patch("lumina.agent.runtime.OpenAI") as sdk, \
             patch("lumina.agent.runtime.requests.post") as post:
            code, result = self.invoke(["run", "--spec", str(specfile), "--dry-run", "--run-id", "test",
                                       "--runs-dir", str(self.root / "runs")])
            self.assertEqual(code, 0)
            self.assertEqual(result["state"], "INPUT_READY")
            sdk.assert_not_called()
            post.assert_not_called()
        common = ["--run-id", "test", "--runs-dir", str(self.root / "runs")]
        code, result = self.invoke(["status", *common])
        self.assertEqual(code, 0)
        self.assertEqual(result["budget"]["calls"], 0)
        code, result = self.invoke(["pause", *common])
        self.assertEqual(code, 0)
        self.assertTrue(result["run"]["pause_requested"])
        code, result = self.invoke(["status", "--run-id", "../escape", "--runs-dir", str(self.root / "runs")])
        self.assertEqual(code, 1)
        self.assertEqual(result["status"], "failed")
        self.assertNotIn("SECRET-DO-NOT-PERSIST", json.dumps(result))

    def test_legacy_entry_and_new_subcommands_have_separate_parsers(self):
        with patch.object(run_pipeline, "load_config", return_value=self.config), patch.object(run_pipeline, "run") as old:
            run_pipeline.main(["--domain", "aqua", "--stage", "all"])
            old.assert_called_once_with("aqua", "all", self.config)
        with patch("lumina.agent.cli.main", return_value=0) as new, self.assertRaises(SystemExit) as exit:
            run_pipeline.main(["status", "--run-id", "example"])
        self.assertEqual(exit.exception.code, 0)
        new.assert_called_once_with(["status", "--run-id", "example"])

    def test_corrupt_authority_returns_nonzero_without_reset(self):
        run = begin(request_for(self.paper), self.config, self.root / "runs", dry_run=True, run_id="broken")
        (run / "ledger.sqlite").write_bytes(b"corrupt authority")
        code, result = self.invoke(["status", "--run-id", run.name, "--runs-dir", str(self.root / "runs")])
        self.assertEqual(code, 1)
        self.assertIn("ledger", result["error"])
        self.assertEqual((run / "ledger.sqlite").read_bytes(), b"corrupt authority")


if __name__ == "__main__":
    unittest.main()
