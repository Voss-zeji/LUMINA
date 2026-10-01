from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import Mock, patch

import requests

from lumina.agent.contracts import ResearchSpecification, create_run
from lumina.agent.runtime import ControlStop, RunContext
from lumina.agent.store import Store
from lumina import prompts
from lumina.common import fingerprint
from test_agent_contracts import config_for, request_for


class RuntimeTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        paper = self.root / "paper.md"
        paper.write_text("Article text.", encoding="utf-8")
        self.config = config_for()
        self.spec = ResearchSpecification.from_config(request_for(paper), self.config)
        self.run = create_run(self.spec, self.root / "runs", "r")
        with Store(self.run, create=True) as store:
            store.initialize(self.spec.fingerprint)
            store.set_state("PREFLIGHT", "test")
            store.set_state("INPUT_READY", "test")
            store.set_state("SMOKE", "test")
        self.runtime = RunContext(self.run, self.config)
        self.addCleanup(self.runtime.close)
        self.model = self.config.FULL_LLM_POOL["a"]
        self.messages = [dict(role="system", content=prompts.message_system_v2.format(domain="test")),
                         dict(role="user", content=prompts.message_system_v2_output.format(content="Article text.")),
                         dict(role="user", content=prompts.questions_for_domain("aqua")[0].strip("\n"))]
        self.key = self.runtime.key("examiner", paper_uid=self.spec.data["papers"][0]["paper_uid"],
                                    question=1, source_model=self.model["model"])

    def client(self, finish_reason="stop", usage=True):
        envelope = dict(choices=[dict(message=dict(content='{"value": 2}'), finish_reason=finish_reason)],
                        usage=dict(prompt_tokens=12, completion_tokens=5, total_tokens=17) if usage else None)
        client = Mock()
        client.chat.completions.create.return_value = SimpleNamespace(model_dump=lambda **_: envelope)
        return client

    def test_disables_hidden_retries_and_replays_response_after_output_loss(self):
        client = self.client()
        with patch("lumina.agent.runtime.OpenAI", return_value=client) as sdk:
            with self.runtime.task_scope(self.key):
                self.assertEqual(self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01),
                                 ('{"value": 2}', 17))
            self.assertEqual(sdk.call_args.kwargs["max_retries"], 0)
            self.assertEqual(client.chat.completions.create.call_args.kwargs["max_tokens"], 1000)
            # No published CSV: resume reparses the canonical saved response.
            self.runtime.close()
            self.runtime = RunContext(self.run, self.config)
            self.addCleanup(self.runtime.close)
            with self.runtime.task_scope(self.key):
                self.assertEqual(self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01)[1], 17)
            self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(self.runtime.budget.totals()["calls"], 1)

    def test_timeout_holds_budget_and_requires_human_without_second_call(self):
        client = self.client()
        client.chat.completions.create.side_effect = requests.Timeout("request state unknown")
        with patch("lumina.agent.runtime.OpenAI", return_value=client):
            for _ in range(2):
                with self.assertRaises(ControlStop), self.runtime.task_scope(self.key):
                    self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01)
            self.assertEqual(client.chat.completions.create.call_count, 1)
        totals = self.runtime.budget.totals()
        self.assertGreater(totals["held_cost"], 0)
        self.assertEqual(totals["unknown_requests"], 1)
        self.assertEqual(len(self.runtime.store.gates()), 1)

    def test_truncated_answer_is_durable_but_never_accepted_or_automatically_requeried(self):
        client = self.client(finish_reason="length")
        with patch("lumina.agent.runtime.OpenAI", return_value=client):
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "truncated"), self.runtime.task_scope(self.key):
                    self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01)
            self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertNotEqual(self.runtime.store.task(self.key.task_id)["status"], "succeeded")

    def test_pause_and_out_of_scope_requests_never_reach_transport(self):
        with patch("lumina.agent.runtime.OpenAI") as sdk:
            with self.assertRaisesRegex(RuntimeError, "task scope"):
                self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01)
            self.runtime.store.request_pause()
            with self.assertRaises(ControlStop), self.runtime.task_scope(self.key):
                self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01)
            sdk.assert_not_called()
        self.assertEqual(self.runtime.budget.totals()["calls"], 0)

    def test_embedding_is_single_attempt_and_raw_vector_reused(self):
        response = Mock()
        response.json.return_value = dict(data=[dict(embedding=[1., 0.])], usage=dict(prompt_tokens=3, total_tokens=3))
        key = self.runtime.key("embeddings", paper_uid=self.key.paper_uid, source_model="mock/embedding")
        with patch("lumina.agent.runtime.requests.post", return_value=response) as post:
            for _ in range(2):
                with self.runtime.task_scope(key):
                    self.assertEqual(self.runtime.embedding("chunk", self.config.EMBEDDING_MODEL, self.config.LLM_SETTINGS), [1., 0.])
            self.assertEqual(post.call_count, 1)
        self.assertEqual(self.runtime.budget.totals()["calls"], 1)

    def test_identical_embedding_request_is_shared_without_changing_origin_artifact(self):
        response = Mock()
        response.json.return_value = dict(data=[dict(embedding=[1., 0.])], usage=dict(prompt_tokens=3, total_tokens=3))
        key = self.runtime.key("embeddings", paper_uid=self.key.paper_uid, source_model="mock/embedding")
        other = self.runtime.key("evidence_embedding", paper_uid=self.key.paper_uid, question=1,
                                 source_model=self.model["model"], candidate="candidate-two")
        with patch("lumina.agent.runtime.requests.post", return_value=response) as post:
            with self.runtime.task_scope(key):
                self.runtime.embedding("shared evidence", self.config.EMBEDDING_MODEL, self.config.LLM_SETTINGS)
            original = replace(key, chunk=fingerprint("shared evidence"))
            artifact = self.runtime.store.task(original.task_id)["artifacts"][0]
            with self.runtime.task_scope(other):
                self.runtime.embedding("shared evidence", self.config.EMBEDDING_MODEL, self.config.LLM_SETTINGS)
            self.assertEqual(post.call_count, 1)
        self.assertTrue(self.runtime.valid(original))
        self.assertTrue(self.runtime.valid(other))
        self.assertEqual(self.runtime.store.task(original.task_id)["artifacts"][0], artifact)
        self.assertEqual(self.runtime.budget.totals()["calls"], 1)

    def test_redacted_receipt_and_configuration_drift(self):
        client = self.client()
        client.chat.completions.create.return_value = SimpleNamespace(model_dump=lambda **_: dict(
            choices=[dict(message=dict(content='SECRET-DO-NOT-PERSIST'), finish_reason="stop")], usage=None))
        with patch("lumina.agent.runtime.OpenAI", return_value=client), self.runtime.task_scope(self.key):
            self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01)
        self.assertNotIn(b"SECRET-DO-NOT-PERSIST", (self.run / "ledger.sqlite").read_bytes())
        self.assertGreater(self.runtime.budget.totals()["held_tokens"], 0)
        self.config.RUN["temperature"] = .9
        with self.assertRaisesRegex(ValueError, "specification"):
            RunContext(self.run, self.config)

    def test_wrong_task_roles_and_changed_payload_never_dispatch_second_request(self):
        client = self.client()
        with patch("lumina.agent.runtime.OpenAI", return_value=client):
            with self.runtime.task_scope(self.key):
                with self.assertRaisesRegex(ValueError, "role"):
                    self.runtime.chat(self.config.FULL_LLM_POOL["b"], self.config.LLM_SETTINGS, self.messages, .01)
                with self.assertRaisesRegex(ValueError, "temperature"):
                    self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .9)
                self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01)
                with self.assertRaisesRegex(ValueError, "three messages"):
                    self.runtime.chat(self.model, self.config.LLM_SETTINGS, [{"role": "user", "content": "new science"}], .01)
            self.assertEqual(client.chat.completions.create.call_count, 1)

    def test_non_json_embedding_body_is_retained_and_not_requeried(self):
        response = Mock()
        response.json.side_effect = ValueError("not JSON")
        response.text = "malformed paid response"
        key = self.runtime.key("embeddings", paper_uid=self.key.paper_uid, source_model="mock/embedding")
        with patch("lumina.agent.runtime.requests.post", return_value=response) as post:
            for _ in range(2):
                with self.assertRaisesRegex(ValueError, "data item"), self.runtime.task_scope(key):
                    self.runtime.embedding("chunk", self.config.EMBEDDING_MODEL, self.config.LLM_SETTINGS)
            self.assertEqual(post.call_count, 1)
        self.assertIn(b"malformed paid response", (self.run / "ledger.sqlite").read_bytes())
        self.assertGreater(self.runtime.budget.totals()["held_cost"], 0)

    def test_wrong_question_prompt_is_refused_before_transport(self):
        wrong = [dict(message) for message in self.messages]
        wrong[-1]["content"] = prompts.questions_for_domain("aqua")[2].strip("\n")
        with patch("lumina.agent.runtime.OpenAI") as sdk, self.runtime.task_scope(self.key):
            with self.assertRaisesRegex(ValueError, "frozen paper and question"):
                self.runtime.chat(self.model, self.config.LLM_SETTINGS, wrong, .01)
            sdk.assert_not_called()

    def test_known_auth_rejection_releases_hold_but_does_not_blindly_retry(self):
        response = Mock(status_code=401)
        response.json.return_value = {"error": {"code": "invalid_api_key"}}
        client = self.client()
        client.chat.completions.create.side_effect = requests.HTTPError("credentials rejected", response=response)
        with patch("lumina.agent.runtime.OpenAI", return_value=client), self.assertRaises(ControlStop), self.runtime.task_scope(self.key):
            self.runtime.chat(self.model, self.config.LLM_SETTINGS, self.messages, .01)
        totals = self.runtime.budget.totals()
        self.assertEqual(totals["unknown_requests"], 0)
        self.assertEqual(totals["held_cost"], 0)
        self.assertEqual(totals["calls"], 1)
        self.assertEqual(client.chat.completions.create.call_count, 1)
        self.assertEqual(self.runtime.store.gates()[0]["kind"], "confirmed_rejection")

    def test_reentering_a_succeeded_task_keeps_authoritative_success(self):
        output = self.run / "outputs" / "proof.json"
        output.write_text("{}", encoding="utf-8")
        self.runtime.completed(self.key, [output])
        with self.runtime.task_scope(self.key):
            self.assertEqual(self.runtime.store.task(self.key.task_id)["status"], "succeeded")
        self.assertTrue(self.runtime.valid(self.key))

    def test_pause_after_publication_keeps_the_completed_artifacts(self):
        output = self.run / "outputs" / "proof.json"
        output.write_text("{}", encoding="utf-8")
        with self.assertRaises(ControlStop), self.runtime.task_scope(self.key):
            self.runtime.completed(self.key, [output])
            raise ControlStop("pause during rate-limit wait after publication")
        self.assertTrue(self.runtime.valid(self.key))
        self.assertEqual(self.runtime.store.task(self.key.task_id)["status"], "succeeded")


if __name__ == "__main__":
    unittest.main()
