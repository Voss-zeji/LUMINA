from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from test_agent_contracts import config_for, request_for

MOCK_CONFIG = '''
from types import SimpleNamespace
from unittest.mock import Mock
from lumina import prompts
from lumina.agent import runtime
import json

def reply(**payload):
    messages = payload['messages']
    if messages[0]['content'] == prompts.message_system_ragQuery.strip():
        content = dict(existing_flag=1, direct_quote='China')
    else:
        q = [s.strip('\\n') for s in prompts.questions_for_domain('aqua')].index(messages[-1]['content']) + 1
        item = 'Study_location' if q == 1 else 'Specie' if q == 2 else 'Flux-1'
        answer = dict(value='China' if q == 1 else 'fish' if q == 2 else 2, evidence='China', confidence_lv=95)
        if q == 3: answer['unit'] = 'mg m-2 h-1'
        content = {item: answer}
    result = dict(choices=[dict(message=dict(content=json.dumps(content)), finish_reason='stop')],
                  usage=dict(prompt_tokens=12, completion_tokens=5, total_tokens=17))
    return SimpleNamespace(model_dump=lambda **_: result)
client = Mock()
client.chat.completions.create.side_effect = reply
runtime.OpenAI = lambda **_: client
embedding = Mock()
embedding.json.return_value = dict(data=[dict(embedding=[1., 0.])], usage=dict(prompt_tokens=3, total_tokens=3))
runtime.requests.post = lambda *args, **kwargs: embedding
runtime.RunContext.sleep = lambda *args, **kwargs: None
'''


class CLIProcessTests(unittest.TestCase):
    def test_all_control_commands_through_real_worker_processes_are_offline(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            paper = root / "paper.md"
            paper.write_text("China. Fish. Flux is 2.", encoding="utf-8")
            config = root / "synthetic.py"
            config.write_text("\n".join(f"{key} = {value!r}" for key, value in vars(config_for()).items()) + MOCK_CONFIG,
                              encoding="utf-8")
            spec = root / "spec.json"
            request = request_for(paper)
            request["budget"]["max_runtime"] = 300
            spec.write_text(json.dumps(request), encoding="utf-8")
            runs = root / "runs"
            project = Path(__file__).resolve().parents[1]
            environment = dict(os.environ, LUMINA_TEST_OFFLINE="1", PYTHONPATH=str(project / "tests"))

            def cli(args, expected):
                child = subprocess.run([sys.executable, str(project / "run_pipeline.py"), *args],
                                       cwd=project, env=environment, capture_output=True, text=True,
                                       encoding="utf-8", timeout=40)
                self.assertEqual(child.returncode, expected, child.stderr + child.stdout)
                self.assertNotIn("SECRET-DO-NOT-PERSIST", child.stdout + child.stderr)
                return json.loads(child.stdout)

            common = ["--runs-dir", str(runs), "--run-id", "cli-test"]
            initial = cli(["run", *common, "--spec", str(spec), "--config", str(config), "--dry-run"], 0)
            self.assertEqual(initial["state"], "INPUT_READY")
            cli(["pause", *common], 0)
            self.assertEqual(cli(["resume", *common, "--config", str(config)], 2)["state"], "PAUSED")
            self.assertEqual(cli(["resume", *common, "--config", str(config)], 2)["state"], "HUMAN_GATE_SMOKE")
            status = cli(["status", *common], 0)
            gate = status["gates"]["pending"][0]
            cli(["approve", *common, "--gate-id", gate, "--reason", "reviewed synthetic smoke"], 0)
            self.assertEqual(cli(["resume", *common, "--config", str(config)], 0)["state"], "DONE")
            report = cli(["report", *common], 0)
            self.assertEqual(report["outcome"], "success")
            self.assertEqual(report["scientific_validation"], "NOT_EVALUATED")
            self.assertEqual(cli(["evaluate", *common], 0)["status"], "NOT_EVALUATED")
