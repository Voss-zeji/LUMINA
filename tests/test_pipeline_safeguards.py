from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import sys
import types

import numpy as np
import pandas as pd

if "langchain.text_splitter" not in sys.modules:
    text_splitter = types.ModuleType("langchain.text_splitter")
    text_splitter.MarkdownTextSplitter = object
    langchain = types.ModuleType("langchain")
    langchain.text_splitter = text_splitter
    sys.modules["langchain"] = langchain
    sys.modules["langchain.text_splitter"] = text_splitter

if "sklearn.metrics.pairwise" not in sys.modules:
    pairwise = types.ModuleType("sklearn.metrics.pairwise")
    pairwise.cosine_similarity = lambda *_args, **_kwargs: None
    metrics = types.ModuleType("sklearn.metrics")
    metrics.pairwise = pairwise
    sklearn = types.ModuleType("sklearn")
    sklearn.metrics = metrics
    sys.modules["sklearn"] = sklearn
    sys.modules["sklearn.metrics"] = metrics
    sys.modules["sklearn.metrics.pairwise"] = pairwise

if "json5" not in sys.modules:
    json5 = types.ModuleType("json5")
    json5.loads = lambda _content: {"invalid_syntax": {"value": None}}
    sys.modules["json5"] = json5

if "tiktoken" not in sys.modules:
    tiktoken = types.ModuleType("tiktoken")
    tiktoken.encoding_for_model = lambda _model: SimpleNamespace(encode=lambda _text: [])
    sys.modules["tiktoken"] = tiktoken

if "openai" not in sys.modules:
    openai = types.ModuleType("openai")
    openai.OpenAI = object
    sys.modules["openai"] = openai

import run_pipeline
from lumina import cross_validation, examiner, llm, preparation


class ChatCapabilityTests(unittest.TestCase):
    def test_omits_json_mode_for_a_provider_that_does_not_support_it(self) -> None:
        calls: dict[str, dict] = {}

        class FakeOpenAI:
            def __init__(self, **kwargs) -> None:
                calls["client"] = kwargs
                self.chat = SimpleNamespace(completions=self)

            def create(self, **kwargs):
                calls["request"] = kwargs
                return SimpleNamespace(
                    choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))],
                    usage=SimpleNamespace(total_tokens=7),
                )

        settings = {"provider": {"key": "key", "url": "https://example.test", "supports_json_mode": False}}
        with patch.object(llm, "OpenAI", FakeOpenAI):
            content, tokens = llm.single_chat({"model": "test-model", "source": "provider"}, settings, [])

        self.assertEqual((content, tokens), ("{}", 7))
        self.assertNotIn("response_format", calls["request"])

    def test_cross_request_error_reaches_the_stage_that_records_it(self) -> None:
        with patch.object(llm, "single_chat", side_effect=RuntimeError("provider unavailable")):
            with self.assertRaisesRegex(RuntimeError, "provider unavailable"):
                llm.llm_requery({}, {}, "system", "prompt")


class PipelineValidationTests(unittest.TestCase):
    def test_rejects_cross_score_threshold_above_independent_verifier_count(self) -> None:
        with self.assertRaisesRegex(SystemExit, "at most 1"):
            run_pipeline.validate_min_cross_scores(["model-a", "model-b"], {"min_cross_scores": [2]})

    def test_accepts_threshold_reachable_by_independent_verifiers(self) -> None:
        run_pipeline.validate_min_cross_scores(
            ["model-a", "model-b", "model-c"], {"min_cross_scores": [1, 2]}
        )


class PreparationTests(unittest.TestCase):
    def test_converted_markdown_is_written_to_configured_markdown_directory(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            pdf_dir = root / "pdfs"
            markdown_dir = root / "markdown"
            pdf_dir.mkdir()
            calls = []

            def convert(pdf_source: str, markdown_target: str) -> list[str]:
                calls.append((pdf_source, markdown_target))
                output = Path(markdown_target) / "01_paper.md"
                output.parent.mkdir(parents=True, exist_ok=True)
                output.write_text("body", encoding="utf-8")
                return [str(output)]

            with patch.object(preparation, "convert_pdfs_to_markdown", side_effect=convert):
                markdowns = preparation.ensure_markdowns(
                    {"pdf_dir": str(pdf_dir), "markdown_dir": str(markdown_dir)}
                )

            self.assertEqual(calls, [(str(pdf_dir), str(markdown_dir))])
            self.assertEqual(markdowns, [str(markdown_dir / "01_paper.md")])


class ExaminerFailureTests(unittest.TestCase):
    def test_failed_request_is_recorded_without_creating_a_resumable_csv(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_dir = root / "markdown"
            markdown_dir.mkdir()
            (markdown_dir / "01_paper.md").write_text("paper body", encoding="utf-8")
            output_dir = root / "examiner"
            domain_cfg = {
                "markdown_dir": str(markdown_dir),
                "examiner_output": str(output_dir),
                "domain_knowledge": "test domain",
            }
            model = {"model": "provider/model-a", "source": "provider"}
            run_cfg = {"round_index": 1, "temperature": 0.01}

            with (
                patch.object(examiner, "single_chat", side_effect=RuntimeError("provider unavailable")),
                patch.object(examiner, "sleep_for_rate_limit"),
            ):
                examiner.run_examiner_for_domain(
                    "aqua", domain_cfg, {"model-a": model}, {"provider": {}}, run_cfg
                )

            self.assertEqual(list(output_dir.rglob("*.csv")), [])
            failures = list(output_dir.rglob("*_invalid.txt"))
            self.assertEqual(len(failures), 3)
            self.assertTrue(all("LLM_ERROR: provider unavailable" in path.read_text(encoding="utf-8") for path in failures))


class CrossValidationTests(unittest.TestCase):
    def test_excludes_the_model_that_created_the_evidence_from_verification(self) -> None:
        models = {
            "a": {"model": "provider/model-a"},
            "b": {"model": "provider/model-b"},
            "c": {"model": "provider/model-c"},
        }

        verifiers = cross_validation.independent_verifiers(
            models, ["model-a", "model-b"], "model-a"
        )

        self.assertEqual([item["model"] for item in verifiers], ["provider/model-b"])

    def test_rejects_embedding_cache_with_wrong_chunk_count(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_dir = root / "markdown"
            markdown_dir.mkdir()
            (markdown_dir / "01_paper.md").write_text("paper body", encoding="utf-8")
            embedding_dir = root / "embeddings"
            domain_cfg = {"markdown_dir": str(markdown_dir), "embedding_dir": str(embedding_dir)}
            run_cfg = {"chunk_size": 2048, "overlap_percent": 20}
            cache = Path(cross_validation.embedding_file(domain_cfg, "01", 2048, 20))
            cache.parent.mkdir(parents=True)
            np.save(cache, np.array([[0.0, 0.0]]))

            class TwoChunkSplitter:
                def split_text(self, _text: str) -> list[str]:
                    return ["one", "two"]

            with (
                patch.object(cross_validation, "_splitter", return_value=TwoChunkSplitter()),
                patch.object(cross_validation, "embedding_response", return_value=[1.0, 2.0]),
            ):
                cross_validation.generate_embeddings_for_domain(domain_cfg, {}, {}, run_cfg)

            self.assertEqual(np.load(cache).shape, (2, 2))

    def test_rebuilds_a_corrupt_embedding_cache(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_dir = root / "markdown"
            markdown_dir.mkdir()
            (markdown_dir / "01_paper.md").write_text("paper body", encoding="utf-8")
            embedding_dir = root / "embeddings"
            domain_cfg = {"markdown_dir": str(markdown_dir), "embedding_dir": str(embedding_dir)}
            run_cfg = {"chunk_size": 2048, "overlap_percent": 20}
            cache = Path(cross_validation.embedding_file(domain_cfg, "01", 2048, 20))
            cache.parent.mkdir(parents=True)
            cache.write_text("not an npy file", encoding="utf-8")

            class OneChunkSplitter:
                def split_text(self, _text: str) -> list[str]:
                    return ["one"]

            with (
                patch.object(cross_validation, "_splitter", return_value=OneChunkSplitter()),
                patch.object(cross_validation, "embedding_response", return_value=[1.0, 2.0]),
            ):
                cross_validation.generate_embeddings_for_domain(domain_cfg, {}, {}, run_cfg)

            self.assertEqual(np.load(cache).shape, (1, 2))

    def test_keeps_the_raw_error_sidecar_for_a_failed_cross_request(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            markdown_dir = root / "markdown"
            markdown_dir.mkdir()
            (markdown_dir / "01_paper.md").write_text("paper body", encoding="utf-8")
            domain_cfg = {
                "markdown_dir": str(markdown_dir),
                "embedding_dir": str(root / "embeddings"),
                "composite_dir": str(root / "composite"),
                "composite_prefix": "Composite",
                "crosser_dir": str(root / "crosser"),
            }
            composite = Path(domain_cfg["composite_dir"]) / "Composite_Q01.xlsx"
            composite.parent.mkdir()
            composite.touch()
            cache = Path(cross_validation.embedding_file(domain_cfg, "01", 2048, 20))
            cache.parent.mkdir(parents=True)
            np.save(cache, np.array([[1.0, 2.0]]))
            composite_rows = pd.DataFrame(
                [{"model": "model-a", "evidence": "evidence", "paper_index": 1, "item": "item"}]
            )
            run_cfg = {"chunk_size": 2048, "overlap_percent": 20, "text_extension": 1, "temperature": 0.01}
            models = {
                "a": {"model": "provider/model-a"},
                "b": {"model": "provider/model-b"},
            }

            with (
                patch.object(cross_validation.pd, "read_excel", return_value=composite_rows),
                patch.object(cross_validation, "_chunks_for_paper", return_value=["context"]),
                patch.object(cross_validation, "embedding_response", return_value=[1.0, 2.0]),
                patch.object(cross_validation, "cosine_similarity", return_value=np.array([[1.0]])),
                patch.object(cross_validation, "llm_requery", side_effect=RuntimeError("provider unavailable")),
                patch.object(cross_validation, "sleep_for_rate_limit"),
            ):
                cross_validation.cross_validate_domain(
                    "aqua", domain_cfg, models, {}, {}, run_cfg, ["model-a", "model-b"]
                )

            failures = list((root / "crosser").rglob("*_invalid.txt"))
            self.assertEqual(len(failures), 1)
            self.assertIn("LLM_ERROR: provider unavailable", failures[0].read_text(encoding="utf-8"))

    def test_does_not_treat_a_failed_verifier_as_a_negative_vote(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            crosser_dir = Path(temp_dir) / "crosser"
            output = crosser_dir / "Paper_01" / "Q01" / "failed.csv"
            output.parent.mkdir(parents=True)
            pd.DataFrame(
                [
                    {
                        "paper_index": 1,
                        "question_index": 1,
                        "item_raw_index": 0,
                        "input_model": "model-a",
                        "output_model": "model-b",
                        "existing_flag": -1,
                    }
                ]
            ).to_csv(output, sep="\t", index=False)

            result = cross_validation.aggregate_cross_scores(
                {"crosser_dir": str(crosser_dir), "composite_dir": str(Path(temp_dir) / "composite"), "composite_prefix": "C"},
                ["model-a", "model-b"],
            )

            self.assertTrue(result.empty)
