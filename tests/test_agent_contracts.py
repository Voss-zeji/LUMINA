from __future__ import annotations

import copy
import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lumina.agent.contracts import (
    ResearchSpecification, TaskKey, create_run, validate_manifest, verify_inputs,
)


def config_for(domain="aqua"):
    return SimpleNamespace(
        DOMAINS={domain: dict(domain_knowledge="test", questions=list(range(1, 4 if domain == "aqua" else 5)))},
        FULL_LLM_POOL={"a": dict(model="mock/Model_A", source="mock"),
                       "b": dict(model="mock/Model_B", source="mock")},
        SELECTED_KEYS=["a", "b"],
        LLM_SETTINGS={"mock": dict(key="SECRET-DO-NOT-PERSIST", url="https://example.invalid/v1", supports_json_mode=True)},
        EMBEDDING_MODEL=dict(model="mock/embedding", source="mock", url="https://example.invalid/embeddings"),
        RUN=dict(round_index=1, temperature=.01, chunk_size=2048, overlap_percent=20,
                 text_extension=1, min_cross_scores=[1]),
    )


def request_for(path, domain="aqua"):
    return dict(domain=domain, papers=[str(path)],
                budget=dict(max_calls=100, max_tokens=100000, max_cost=2., max_runtime=600),
                pricing={model: dict(input_per_million=1., output_per_million=2.,
                                    max_input_tokens=10000, max_output_tokens=1000,
                                    basis="synthetic fixture; not a real price")
                         for model in ["mock/Model_A", "mock/Model_B", "mock/embedding"]})


class ContractTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paper = self.root / "001_original.md"
        self.paper.write_text("# Introduction\nChina. Flux is 2.\n", encoding="utf-8")
        self.config = config_for()

    def spec(self, path=None):
        return ResearchSpecification.from_config(request_for(path or self.paper), self.config)

    def test_renaming_preserves_science_and_specification_identity(self):
        first = self.spec()
        renamed = self.paper.rename(self.root / "different_name.md")
        second = self.spec(renamed)
        self.assertEqual(first.fingerprint, second.fingerprint)
        self.assertEqual(first.scientific_hash, second.scientific_hash)
        self.assertEqual(first.data["papers"][0]["paper_uid"], second.data["papers"][0]["paper_uid"])
        self.assertNotEqual(first.data["papers"][0]["display_name"], second.data["papers"][0]["display_name"])

    def test_content_prompts_models_thresholds_and_budget_are_frozen(self):
        first = self.spec()
        self.paper.write_text("Changed article", encoding="utf-8")
        self.assertNotEqual(first.fingerprint, self.spec().fingerprint)
        self.paper.write_text("# Introduction\nChina. Flux is 2.\n", encoding="utf-8")
        self.config.RUN["temperature"] = .2
        self.assertNotEqual(first.scientific_hash, self.spec().scientific_hash)
        self.config = config_for()
        request = request_for(self.paper)
        request["budget"]["max_cost"] = 3
        changed = ResearchSpecification.from_config(request, self.config)
        self.assertNotEqual(first.fingerprint, changed.fingerprint)
        self.assertEqual(first.scientific_hash, changed.scientific_hash)
        data = first.data
        data["run"]["temperature"] = 999
        self.assertEqual(first.data["run"]["temperature"], .01)

    def test_model_display_alias_and_selection_order_do_not_change_scientific_identity(self):
        first = self.spec()
        self.config.FULL_LLM_POOL["renamed_alias"] = self.config.FULL_LLM_POOL.pop("a")
        self.config.SELECTED_KEYS = ["b", "renamed_alias"]
        self.assertEqual(first.fingerprint, self.spec().fingerprint)

    def test_secrets_are_excluded_and_endpoint_credentials_are_rejected(self):
        self.assertNotIn("SECRET-DO-NOT-PERSIST", self.spec().to_json())
        accidental = request_for(self.paper)
        accidental["pricing"]["mock/Model_A"]["basis"] = "credential SECRET-DO-NOT-PERSIST"
        with self.assertRaisesRegex(ValueError, "credential"):
            ResearchSpecification.from_config(accidental, self.config)
        for endpoint in ["https://u:pw@example.invalid/v1", "https://example.invalid/v1?api_key=secret"]:
            self.config.LLM_SETTINGS["mock"]["url"] = endpoint
            with self.subTest(endpoint=endpoint), self.assertRaisesRegex(ValueError, "endpoint"):
                self.spec()
        request = request_for(self.paper)
        request["api_key"] = "secret"
        with self.assertRaises(ValueError):
            ResearchSpecification.from_config(request, config_for())

    def test_duplicates_unsupported_input_and_ambiguous_models_fail(self):
        duplicate = self.root / "same.md"
        duplicate.write_bytes(self.paper.read_bytes())
        request = request_for(self.paper)
        request["papers"].append(str(duplicate))
        with self.assertRaisesRegex(ValueError, "duplicate"):
            ResearchSpecification.from_config(request, self.config)
        for uid in ("CON", "NUL", "1"):
            request = request_for(self.paper)
            request["papers"] = [dict(path=str(self.paper), paper_uid=uid)]
            with self.subTest(uid=uid), self.assertRaisesRegex(ValueError, "canonical"):
                ResearchSpecification.from_config(request, self.config)
        self.config.FULL_LLM_POOL["b"]["model"] = "other/Model_A"
        with self.assertRaisesRegex(ValueError, "model"):
            self.spec()

    def test_malformed_limits_and_scientific_settings_are_rejected(self):
        for field, invalid in [("max_calls", True), ("max_tokens", 0), ("max_cost", float("inf")),
                               ("max_runtime", -1)]:
            request = request_for(self.paper)
            request["budget"][field] = invalid
            with self.subTest(field=field), self.assertRaises(ValueError):
                ResearchSpecification.from_config(request, self.config)
        for key, value in [("chunk_size", 0), ("overlap_percent", 100), ("temperature", float("nan")),
                           ("min_cross_scores", [2])]:
            config = copy.deepcopy(self.config)
            config.RUN[key] = value
            with self.subTest(key=key), self.assertRaises(ValueError):
                ResearchSpecification.from_config(request_for(self.paper), config)

    def test_isolated_snapshots_detect_tampering_without_touching_original(self):
        spec = self.spec()
        first = create_run(spec, self.root / "runs", "first")
        second = create_run(spec, self.root / "runs", "second")
        self.assertNotEqual(first, second)
        verify_inputs(first)
        validate_manifest(first)
        frozen = next((first / "inputs").iterdir())
        self.assertEqual(frozen.stem, spec.data["papers"][0]["paper_uid"])
        frozen.write_text("tampered", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "input"):
            verify_inputs(first)
        verify_inputs(second)
        self.assertEqual(self.paper.read_text(encoding="utf-8"), "# Introduction\nChina. Flux is 2.\n")
        with self.assertRaises(FileExistsError):
            create_run(spec, self.root / "runs", "first")

    def test_manifest_changes_and_path_escape_are_rejected(self):
        run = create_run(self.spec(), self.root / "runs", "test")
        manifest_path = run / "manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["specification"]["run"]["temperature"] = .5
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "fingerprint"):
            validate_manifest(run)
        for run_id in ["../escape", "nested/id", "CON", ".", ""]:
            with self.subTest(run_id=run_id), self.assertRaises(ValueError):
                create_run(self.spec(), self.root / "runs", run_id)

    def test_task_identity_has_no_smoke_batch_phase(self):
        task = TaskKey(run_id="r", stage="examiner", paper_uid="p123", question=1,
                       source_model="mock/Model_A", round_index=1)
        self.assertEqual(task.task_id, TaskKey(**task.to_dict()).task_id)
        self.assertNotEqual(task.task_id, TaskKey(**(task.to_dict() | {"round_index": 2})).task_id)
        for invalid in [{"question": True}, {"question": 1.0}, {"round_index": 1.0}, {"round_index": True}]:
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                TaskKey(**(task.to_dict() | invalid))

    def test_smoke_uses_the_user_selected_paper_and_freezes_that_choice(self):
        other = self.root / "second.md"
        other.write_text("A different paper", encoding="utf-8")
        request = request_for(self.paper)
        request["papers"].append(str(other))
        original = ResearchSpecification.from_config(request, self.config)
        first_uid = next(p["paper_uid"] for p in original.data["papers"] if p["display_name"] == self.paper.name)
        self.assertEqual(original.data["smoke_paper_uids"], [first_uid])
        request["smoke_papers"] = [str(other)]
        selected = ResearchSpecification.from_config(request, self.config)
        self.assertNotEqual(original.fingerprint, selected.fingerprint)
        self.assertEqual(original.scientific_hash, selected.scientific_hash)
        request["smoke_papers"] = ["pUnknown"]
        with self.assertRaisesRegex(ValueError, "frozen input"):
            ResearchSpecification.from_config(request, self.config)


if __name__ == "__main__":
    unittest.main()
