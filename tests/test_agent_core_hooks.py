"""G3/G4 control hooks inside the scientific core.

The ledger, budget and control layer are real; only the HTTP boundary and the
cooperative wait are faked.  Nothing in this file may reach a provider.
"""
from __future__ import annotations

import json
import sys
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))

from lumina import composite, cross_validation as cv, examiner, llm, prompts
from lumina.agent.contracts import ResearchSpecification, create_run
from lumina.agent.runtime import ControlStop, RunContext
from lumina.agent.store import Store
from test_agent_contracts import config_for

ANSWERS = {
    1: dict(Study_location=dict(value="China", evidence="China", confidence_lv=95)),
    2: dict(Specie=dict(value="fish", evidence="China", confidence_lv=90)),
    3: dict(Flux_1=dict(value=2, evidence="China", confidence_lv=80, unit="mg m-2 h-1")),
}


def request_with_room(path):
    """The shared fixture budget covers one task; a whole domain run needs room."""
    return dict(domain="aqua", papers=[str(path)],
                budget=dict(max_calls=1000, max_tokens=10000000, max_cost=1000.0, max_runtime=3600),
                pricing={model: dict(input_per_million=1.0, output_per_million=2.0,
                                     max_input_tokens=10000, max_output_tokens=1000,
                                     basis="synthetic fixture; not a real price")
                         for model in ("mock/Model_A", "mock/Model_B", "mock/embedding")})


class CoreHookCase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        paper = self.root / "001_original.md"
        paper.write_text("China. 2020. Fish. Flux is 2.", encoding="utf-8")
        self.config = config_for()
        self.spec = ResearchSpecification.from_config(request_with_room(paper), self.config)
        self.run_dir = create_run(self.spec, self.root / "runs", "r")
        with Store(self.run_dir, create=True) as store:
            store.initialize(self.spec.fingerprint)
            for state in ("PREFLIGHT", "INPUT_READY", "SMOKE"):
                store.set_state(state, "test fixture")
        self.runtime = RunContext(self.run_dir, self.config)
        self.addCleanup(self.runtime.close)
        outputs = self.run_dir / "outputs"
        # Outputs live inside the run directory: the ledger refuses escaping artifacts.
        self.domain_cfg = dict(
            markdown_dir=str(self.run_dir / "inputs"),
            examiner_output=str(outputs / "examiner"),
            composite_dir=str(outputs / "composite"),
            embedding_dir=str(outputs / "embedding"),
            crosser_dir=str(outputs / "crosser"),
            domain_knowledge="test", composite_prefix="C", questions=[1, 2, 3])

    # -- fake HTTP boundary -------------------------------------------------

    def envelope(self, content):
        return dict(choices=[dict(message=dict(content=content), finish_reason="stop")],
                    usage=dict(prompt_tokens=12, completion_tokens=5, total_tokens=17))

    def sdk_completion(self, content):
        """A legacy SDK object: attributes, not the runtime's model_dump envelope."""
        message = SimpleNamespace(content=content)
        choice = SimpleNamespace(message=message)
        return SimpleNamespace(choices=[choice],
                               usage=SimpleNamespace(total_tokens=17))

    def client(self, reply):
        client = Mock()
        client.chat.completions.create.side_effect = (
            lambda **_: SimpleNamespace(model_dump=lambda **_: self.envelope(
                reply(client.chat.completions.create.call_args.kwargs["messages"]))))
        return client

    def post(self, vector=(1.0, 0.0)):
        response = Mock()
        response.json.return_value = dict(data=[dict(embedding=list(vector))],
                                         usage=dict(prompt_tokens=3, total_tokens=3))
        return response

    def fake_transport(self, client=None, vector=(1.0, 0.0)):
        """Fake only the provider boundary; every ledger decision stays real."""
        stack = ExitStack()
        self.addCleanup(stack.close)
        client = client if client is not None else self.client(self.answer_for)
        post = self.post(vector)
        stack.enter_context(patch("lumina.agent.runtime.OpenAI", return_value=client))
        stack.enter_context(patch("lumina.agent.runtime.requests.post", return_value=post))
        # Cooperative waiting keeps its control checks but spends no wall clock.
        stack.enter_context(patch.object(RunContext, "sleep", lambda self, seconds: self.check()))
        return client, post

    def answer_for(self, messages):
        """Answer like a working provider: the real questions, answered correctly."""
        if messages[0]["content"] == prompts.message_system_ragQuery.strip():
            return json.dumps(dict(existing_flag=1, direct_quote="China"))
        asked = [q.strip("\n") for q in prompts.questions_for_domain("aqua")]
        return json.dumps(ANSWERS[asked.index(messages[-1]["content"]) + 1])

    def run_examiner(self, reply=None, **kwargs):
        client, _ = self.fake_transport(self.client(reply or self.answer_for), **kwargs)
        examiner.run_examiner_for_domain("aqua", self.domain_cfg, self.config.FULL_LLM_POOL,
                                         self.config.LLM_SETTINGS, self.config.RUN, runtime=self.runtime)
        return client

    def examiner_outputs(self):
        return sorted(Path(self.domain_cfg["examiner_output"]).rglob("*.csv"))


class TransportDelegation(CoreHookCase):
    def test_paid_functions_delegate_before_touching_a_provider(self):
        model = self.config.FULL_LLM_POOL["a"]
        settings = self.config.LLM_SETTINGS
        runtime = Mock()
        runtime.chat.return_value = ("ok", 3)
        runtime.embedding.return_value = [1.0]
        messages = [{"role": "user", "content": "extract"}]
        with patch.object(llm, "OpenAI") as sdk, patch.object(llm.requests, "post") as post:
            self.assertEqual(llm.single_chat(model, settings, messages, .5, runtime=runtime), ("ok", 3))
            self.assertEqual(llm.embedding_response("chunk", self.config.EMBEDDING_MODEL, settings,
                                                   runtime=runtime), [1.0])
            self.assertEqual(llm.llm_requery(model, settings, "system", "question", .5,
                                             runtime=runtime), ("ok", 3))
            sdk.assert_not_called()
            post.assert_not_called()
        runtime.chat.assert_any_call(model, settings, messages, .5)
        runtime.chat.assert_any_call(model, settings,
                                     [{"role": "system", "content": "system"},
                                      {"role": "user", "content": "question"}], .5)
        runtime.embedding.assert_called_once_with("chunk", self.config.EMBEDDING_MODEL, settings)

    def test_legacy_path_is_unchanged_without_a_runtime(self):
        model = self.config.FULL_LLM_POOL["a"]
        settings = self.config.LLM_SETTINGS
        client = Mock()
        client.chat.completions.create.return_value = self.sdk_completion('{"value": 1}')
        with patch.object(llm, "OpenAI", return_value=client):
            content, tokens = llm.single_chat(model, settings, [{"role": "user", "content": "x"}])
        self.assertEqual((content, tokens), ('{"value": 1}', 17))
        with patch.object(llm.requests, "post", return_value=self.post()):
            self.assertEqual(llm.embedding_response("chunk", self.config.EMBEDDING_MODEL, settings),
                             [1.0, 0.0])

    def test_rate_limit_sleep_reaches_the_cooperative_control_layer(self):
        from lumina.utils import sleep_for_rate_limit
        runtime = Mock()
        slow = dict(self.config.FULL_LLM_POOL["a"], limit_token=10)
        sleep_for_rate_limit(slow, runtime=runtime)
        runtime.sleep.assert_called_once_with(59.0)
        with patch("lumina.utils.time.sleep") as nap:
            sleep_for_rate_limit(slow)
            nap.assert_called_once_with(59.0)


class ExaminerHooks(CoreHookCase):
    def task_for(self, artifact: Path):
        return next(t for t in self.runtime.store.tasks("examiner")
                    if any(a["path"].endswith(artifact.name) for a in t["artifacts"]))

    def test_successful_task_is_registered_with_both_artifacts(self):
        self.run_examiner()
        tasks = self.runtime.store.tasks("examiner")
        self.assertEqual(len(tasks), 6)
        models = [m["model"] for m in self.config.FULL_LLM_POOL.values()]
        for task in tasks:
            self.assertEqual(task["status"], "succeeded")
            self.assertEqual(sorted(Path(a["path"]).name.rsplit(".", 1)[-1]
                                    for a in task["artifacts"]), ["csv", "json"])
            payload = task["payload"]
            self.assertEqual(payload["stage"], "examiner")
            self.assertEqual(payload["round_index"], 1)
            self.assertIn(payload["source_model"], models)
            self.assertTrue(self.runtime.store.valid_task(task["task_id"]))

    def test_unchanged_output_skips_the_api_and_records_a_cache_hit(self):
        client = self.run_examiner()
        paid = client.chat.completions.create.call_count
        self.assertEqual(paid, 6)
        self.run_examiner()
        self.assertEqual(client.chat.completions.create.call_count, paid)
        self.assertEqual(len([e for e in self.runtime.store._events() if e["kind"] == "cache_hit"]), 6)

    def test_edited_artifact_is_rebuilt_from_the_cached_paid_response(self):
        client = self.run_examiner()
        target = self.examiner_outputs()[0]
        task_id = self.task_for(target)["task_id"]
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        self.assertFalse(self.runtime.store.valid_task(task_id))
        self.run_examiner()
        # Repaired from the stored paid response: no new request, and the answer is not
        # taken from the edited file.
        self.assertEqual(client.chat.completions.create.call_count, 6)
        self.assertTrue(self.runtime.store.valid_task(task_id))
        frame = pd.read_csv(target, sep="\t", keep_default_na=False, dtype={"paper_index": str})
        self.assertEqual(set(frame["item"]), {"Study_location"})
        self.assertEqual(set(frame["value"]), {"China"})

    def test_deleted_artifact_is_regenerated_from_the_cached_paid_response(self):
        client = self.run_examiner()
        self.examiner_outputs()[0].unlink()
        self.run_examiner()
        self.assertEqual(client.chat.completions.create.call_count, 6)
        self.assertEqual(len(self.examiner_outputs()), 6)

    def test_parse_failure_records_a_failed_task_and_keeps_the_raw_sidecar(self):
        with self.assertRaises(RuntimeError):
            self.run_examiner(reply=lambda messages: "not json at all")
        tasks = self.runtime.store.tasks("examiner")
        self.assertEqual(len(tasks), 6)
        for task in tasks:
            self.assertEqual(task["status"], "failed")
            self.assertFalse(self.runtime.store.valid_task(task["task_id"]))
            self.assertTrue(task["error"])
        self.assertFalse(self.examiner_outputs())
        sidecars = list(Path(self.domain_cfg["examiner_output"]).rglob("*_invalid.txt"))
        self.assertEqual(len(sidecars), 6)
        self.assertIn("not json at all", sidecars[0].read_text(encoding="utf-8"))

    def test_control_stop_is_never_swallowed_into_a_failed_task(self):
        self.fake_transport()
        with patch.object(examiner, "run_llm_prompt_mode", side_effect=ControlStop("paused")):
            with self.assertRaises(ControlStop):
                examiner.run_examiner_for_domain("aqua", self.domain_cfg, self.config.FULL_LLM_POOL,
                                                 self.config.LLM_SETTINGS, self.config.RUN,
                                                 runtime=self.runtime)
        self.assertEqual([t["status"] for t in self.runtime.store.tasks("examiner")], ["unresolved"])

    def test_a_pause_request_stops_before_any_paid_request(self):
        client, _ = self.fake_transport()
        self.runtime.store.request_pause()
        with self.assertRaises(ControlStop):
            examiner.run_examiner_for_domain("aqua", self.domain_cfg, self.config.FULL_LLM_POOL,
                                             self.config.LLM_SETTINGS, self.config.RUN, runtime=self.runtime)
        client.chat.completions.create.assert_not_called()
        self.assertEqual(self.runtime.budget.totals()["calls"], 0)


class EmbeddingHooks(CoreHookCase):
    def embed(self):
        _, post = self.fake_transport()
        cv.generate_embeddings_for_domain(self.domain_cfg, self.config.EMBEDDING_MODEL,
                                         self.config.LLM_SETTINGS, self.config.RUN, runtime=self.runtime)
        return post

    def matrix(self):
        return next(Path(self.domain_cfg["embedding_dir"]).rglob("*.npy"))

    def test_embedding_task_covers_the_matrix_and_its_metadata(self):
        self.embed()
        # The runtime derives one chunk key per chunk; the stage task is the one
        # without a chunk hash.
        tasks = self.runtime.store.tasks("embeddings")
        overall = [t for t in tasks if not t["payload"]["chunk"]]
        self.assertEqual(len(overall), 1)
        self.assertTrue(all(t["payload"]["chunk"] for t in tasks if t is not overall[0]))
        task = overall[0]
        self.assertEqual(task["status"], "succeeded")
        self.assertEqual(sorted(Path(a["path"]).name.rsplit(".", 1)[-1] for a in task["artifacts"]),
                         ["json", "npy"])
        self.assertEqual(task["payload"]["source_model"], self.config.EMBEDDING_MODEL["model"])
        self.assertEqual(task["payload"]["round_index"], 1)
        self.assertTrue(self.runtime.store.valid_task(task["task_id"]))

    def test_unchanged_matrix_skips_the_api_with_a_cache_hit(self):
        self.embed()
        self.assertEqual(self.embed().call_count, 0)
        self.assertTrue([e for e in self.runtime.store._events() if e["kind"] == "cache_hit"])

    def test_corrupt_matrix_rebuilds_from_cached_chunk_receipts_without_new_api(self):
        self.embed()
        task_id = self.embedding_stage_task()["task_id"]
        self.matrix().write_bytes(b"")
        self.assertFalse(self.runtime.store.valid_task(task_id))
        self.assertEqual(self.embed().call_count, 0)
        rebuilt = cv.load_embeddings(str(self.matrix()))
        self.assertIsNotNone(rebuilt)
        self.assertTrue(np.isfinite(rebuilt).all())
        self.assertTrue(self.runtime.store.valid_task(task_id))

    def embedding_stage_task(self):
        return next(t for t in self.runtime.store.tasks("embeddings") if not t["payload"]["chunk"])


class CrossHooks(CoreHookCase):
    def setUp(self):
        super().setUp()
        self.run_examiner()
        composite.create_baseline_composite(
            self.domain_cfg, [1, 2, 3], models=["Model_A", "Model_B"],
            round_index=self.config.RUN["round_index"], domain="aqua",
            expected_tasks=examiner.expected_tasks("aqua", self.domain_cfg, self.config.FULL_LLM_POOL,
                                                  self.config.LLM_SETTINGS, self.config.RUN))

    def cross(self, reply=None, **kwargs):
        client, post = self.fake_transport(self.client(reply or self.answer_for), **kwargs)
        cv.cross_validate_domain("aqua", self.domain_cfg, self.config.FULL_LLM_POOL,
                                 self.config.LLM_SETTINGS, self.config.EMBEDDING_MODEL,
                                 self.config.RUN, ["Model_A", "Model_B"], runtime=self.runtime)
        return client, post

    def votes(self):
        return sorted(Path(self.domain_cfg["crosser_dir"]).rglob("Candidate_*.csv"))

    def test_verifier_and_evidence_tasks_are_scoped_with_identity_fields(self):
        self.cross()
        self.assertEqual({t["stage"] for t in self.runtime.store.tasks()},
                         {"examiner", "embeddings", "evidence_embedding", "cross"})
        cross = self.runtime.store.tasks("cross")
        self.assertEqual(len(cross), len(self.votes()))
        full_ids = [m["model"] for m in self.config.FULL_LLM_POOL.values()]
        for task in cross:
            self.assertEqual(task["status"], "succeeded")
            payload = task["payload"]
            self.assertTrue(payload["candidate"])
            # Identity is the full model ID on both sides, never the short CSV name.
            self.assertIn(payload["source_model"], full_ids)
            self.assertIn(payload["verifier_model"], full_ids)
            self.assertNotEqual(payload["source_model"], payload["verifier_model"])
            self.assertEqual(payload["round_index"], 1)
            self.assertIn(payload["question"], (1, 2, 3))
            self.assertTrue(self.runtime.store.valid_task(task["task_id"]))
        evidence = self.runtime.store.tasks("evidence_embedding")
        self.assertTrue(evidence)
        for task in evidence:
            self.assertEqual(task["status"], "succeeded")
            self.assertTrue(task["artifacts"])
            self.assertIn(task["payload"]["source_model"], full_ids)

    def test_all_valid_verifiers_skip_evidence_and_paid_queries(self):
        self.cross()
        client, post = self.cross()
        self.assertEqual(client.chat.completions.create.call_count, 0)
        self.assertEqual(post.call_count, 0)

    def test_edited_vote_is_regenerated_from_the_cached_response(self):
        self.cross()
        target = self.votes()[0]
        target.write_text(target.read_text(encoding="utf-8") + "\n", encoding="utf-8")
        client, post = self.cross()
        self.assertEqual(client.chat.completions.create.call_count, 0)
        self.assertEqual(post.call_count, 0)
        frame = pd.read_csv(target, sep="\t", dtype={"direct_quote": str})
        self.assertEqual(int(frame.existing_flag.iloc[0]), 1)

    def test_unparseable_vote_fails_the_task_and_writes_minus_one_never_a_success(self):
        with self.assertRaises(RuntimeError):
            self.cross(reply=lambda messages: "not json at all")
        tasks = self.runtime.store.tasks("cross")
        self.assertEqual(len(tasks), len(self.votes()))
        for task in tasks:
            self.assertEqual(task["status"], "failed")
            self.assertEqual(task["artifacts"], [])
            self.assertTrue(task["error"])
        frame = pd.read_csv(self.votes()[0], sep="\t", dtype={"direct_quote": str})
        self.assertEqual(int(frame.existing_flag.iloc[0]), -1)

    def test_control_stop_propagates_and_never_becomes_a_negative_vote(self):
        self.fake_transport()
        with patch.object(cv, "llm_requery", side_effect=ControlStop("budget")):
            with self.assertRaises(ControlStop):
                cv.cross_validate_domain("aqua", self.domain_cfg, self.config.FULL_LLM_POOL,
                                         self.config.LLM_SETTINGS, self.config.EMBEDDING_MODEL,
                                         self.config.RUN, ["Model_A", "Model_B"], runtime=self.runtime)
        self.assertFalse(self.votes())
        self.assertFalse([t for t in self.runtime.store.tasks("cross") if t["status"] == "succeeded"])


if __name__ == "__main__":
    unittest.main()
