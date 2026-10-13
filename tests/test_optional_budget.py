from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from lumina import prompts
from lumina.agent.budget import Budget, BudgetExceeded, UnknownRequest
from lumina.agent.contracts import ResearchSpecification, create_run
from lumina.agent.report import write_report
from lumina.agent.runtime import RunContext
from lumina.agent.store import Store
from lumina.configuration import load_config


def config_for(root: Path):
    paper = root / "paper.md"
    paper.write_text("# Paper\nA short fixture.", encoding="utf-8")
    return paper, SimpleNamespace(
        DOMAINS={"generic": dict(domain_knowledge="test", questions=[1, 2, 3],
                                  question_set=prompts.definition("aqua"))},
        FULL_LLM_POOL={"model": dict(model="vendor/model", source="vendor")},
        SELECTED_KEYS=["model"],
        LLM_SETTINGS={"vendor": dict(key="", url="https://example.invalid/v1")},
        EMBEDDING_MODEL=dict(model="vendor/embedding", source="vendor",
                             url="https://example.invalid/embeddings"),
        RUN=dict(round_index=1, temperature=.01, chunk_size=128, overlap_percent=20,
                 text_extension=1, min_cross_scores=[]),
    )


def request(paper: Path, *, budget=None, pricing=None):
    return dict(domain="generic", papers=[str(paper)],
                budget={} if budget is None else budget,
                pricing={} if pricing is None else pricing)


def unknown_pricing():
    return {"vendor/model": {}, "vendor/embedding": {}}


def known_pricing():
    row = dict(input_per_million=1.0, output_per_million=2.0,
               max_input_tokens=100, max_output_tokens=20, basis="fixture")
    return {"vendor/model": dict(row), "vendor/embedding": dict(row, max_output_tokens=0)}


class OptionalBudgetTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.paper, self.config = config_for(self.root)

    def spec(self, *, budget=None, pricing=None):
        return ResearchSpecification.from_config(
            request(self.paper, budget=budget, pricing=pricing), self.config).data

    def ledger(self, spec):
        run = self.root / "run"
        run.mkdir()
        store = Store(run, create=True)
        store.initialize("spec")
        self.addCleanup(store.close)
        return store, Budget(store, spec)

    def test_all_caps_none_are_frozen_as_null(self):
        spec = self.spec()
        self.assertEqual(spec["budget"], {
            "max_calls": None, "max_cost": None, "max_runtime": None, "max_tokens": None,
        })
        self.assertEqual(spec["request_limits"], {
            "vendor/model": {"max_input_tokens": None, "max_output_tokens": 4096},
            "vendor/embedding": {"max_input_tokens": None, "max_output_tokens": 0},
        })

    def test_partial_cap_is_enforced_without_other_caps(self):
        spec = self.spec(budget={"max_calls": 1}, pricing=known_pricing())
        _, budget = self.ledger(spec)
        attempt = budget.reserve("task", "request", "chat", "vendor/model", 10, 5)
        budget.dispatch(attempt)
        budget.receive(attempt, {"ok": True}, {"input_tokens": 2, "output_tokens": 1})
        with self.assertRaises(BudgetExceeded):
            budget.reserve("other", "request", "chat", "vendor/model", 10, 5)

    def test_unknown_price_keeps_known_usage_and_marks_cost_incomplete(self):
        spec = self.spec(pricing=unknown_pricing())
        _, budget = self.ledger(spec)
        attempt = budget.reserve("task", "request", "chat", "vendor/model", 10, 5)
        budget.dispatch(attempt)
        budget.receive(attempt, {"ok": True}, {"input_tokens": 2, "output_tokens": 1})
        totals = budget.totals()
        self.assertEqual(totals["known_tokens"], 3)
        self.assertEqual(totals["unpriced_requests"], 1)
        self.assertFalse(totals["cost_complete"])
        self.assertIsNone(totals["total_cost"])
        self.assertIsNone(budget.attempt(attempt)["actual_cost"])

    def test_finite_caps_still_have_old_reservation_semantics(self):
        spec = self.spec(budget=dict(max_calls=2, max_tokens=120, max_cost=1,
                                     max_runtime=60), pricing=known_pricing())
        _, budget = self.ledger(spec)
        attempt = budget.reserve("task", "request", "chat", "vendor/model", 10, 5)
        self.assertEqual(budget.attempt(attempt)["reserved_tokens"], 120)

    def test_unknown_request_cannot_be_replayed(self):
        spec = self.spec(pricing=unknown_pricing())
        _, budget = self.ledger(spec)
        attempt = budget.reserve("task", "request", "chat", "vendor/model", 10, 5)
        budget.dispatch(attempt)
        budget.fail(attempt, "socket timeout")
        with self.assertRaises(UnknownRequest):
            budget.reserve("task", "request", "chat", "vendor/model", 10, 5)

    def test_amount_cap_rejects_unknown_price_preflight(self):
        with self.assertRaisesRegex(ValueError, "complete input/output pricing"):
            self.spec(budget={"max_cost": 1}, pricing=unknown_pricing())

    def test_known_prices_require_basis_and_comparable_currency(self):
        price = known_pricing()
        price['vendor/model']['basis'] = ''
        with self.assertRaisesRegex(ValueError, 'basis'):
            self.spec(pricing=price)
        price = known_pricing()
        price['vendor/model']['currency'] = 'USD'
        price['vendor/embedding']['currency'] = 'CNY'
        with self.assertRaisesRegex(ValueError, 'currency'):
            self.spec(budget={'max_cost': 1}, pricing=price)

    def test_request_budget_falsey_non_tables_are_rejected(self):
        for malformed in ([], False):
            with self.subTest(malformed=malformed), self.assertRaisesRegex(ValueError, "budget"):
                self.spec(budget=malformed, pricing=unknown_pricing())

    def test_pricing_base_url_is_bound_to_the_frozen_endpoint(self):
        pricing = known_pricing()
        pricing["vendor/model"]["base_url"] = "https://EXAMPLE.invalid/v1/"
        pricing["vendor/embedding"]["base_url"] = "https://example.invalid/embeddings"
        spec = self.spec(pricing=pricing)
        self.assertEqual(spec["pricing"]["vendor/model"]["base_url"],
                         "https://example.invalid/v1")
        pricing["vendor/model"]["base_url"] = "https://proxy.invalid/v1"
        with self.assertRaisesRegex(ValueError, "base_url"):
            self.spec(pricing=pricing)

    def test_runtime_uses_legacy_pricing_bounds_for_old_snapshots(self):
        runtime = object.__new__(RunContext)
        runtime.spec = {"pricing": {
            "vendor/model": {"max_input_tokens": 123, "max_output_tokens": 456},
        }}
        self.assertEqual(runtime._request_limits({"model": "vendor/model"}, "chat"), (123, 456))

    def test_legacy_caps_and_pricing_bounds_remain_compatible(self):
        spec = self.spec(budget={"max_calls": 5, "max_tokens": 1000, "max_cost": 1,
                                 "max_runtime": 60}, pricing=known_pricing())
        self.assertEqual(spec["request_limits"]["vendor/model"],
                         {"max_input_tokens": 100, "max_output_tokens": 20})
        _, budget = self.ledger(spec)
        attempt = budget.reserve("task", "request", "chat", "vendor/model", 1, 1)
        self.assertEqual(budget.attempt(attempt)["reserved_tokens"], 120)

    def test_report_reads_extended_totals_without_dispatching(self):
        spec = self.spec(pricing=unknown_pricing())
        run = create_run(ResearchSpecification(json.dumps(spec, ensure_ascii=False, sort_keys=True)),
                         self.root / "runs", "report")
        with Store(run, create=True) as store:
            store.initialize("spec")
            budget = Budget(store, spec)
            attempt = budget.reserve("task", "request", "chat", "vendor/model", 10, 5)
            budget.dispatch(attempt)
            budget.receive(attempt, {"ok": True}, {"input_tokens": 2, "output_tokens": 1})
            report = write_report(SimpleNamespace(run_dir=run, spec=spec, store=store, budget=budget),
                                  "status")
        self.assertEqual(report["budget"]["unpriced_requests"], 1)
        self.assertFalse(report["budget"]["cost_complete"])
        self.assertIsNone(report["budget"]["total_cost"])

    def test_toml_accepts_omitted_budget_pricing_and_language(self):
        folder = self.root / "study"
        folder.mkdir()
        shutil.copyfile(Path(__file__).parents[1] / "configs/generic/questions.toml",
                        folder / "questions.toml")
        (folder / "paper.md").write_text("# Paper\nBiomass is 2 kg.", encoding="utf-8")
        (folder / "papers.txt").write_text("paper.md\n", encoding="utf-8")
        (folder / "secrets.local.toml").write_text(
            "[providers.chat]\nkey = \"offline\"\n[providers.embedding]\nkey = \"offline\"\n",
            encoding="utf-8")
        (folder / "project.toml").write_text('''
selected_models = ["a", "b"]

[project]
domain = "generic"
domain_knowledge = "terrestrial ecology"
run_id = "optional-budget"
runs_dir = "runs"
language = "en"

[models.a]
model = "vendor/model-a"
source = "chat"

[models.b]
model = "vendor/model-b"
source = "chat"

[providers.chat]
url = "https://example.invalid/v1"
supports_json_mode = true

[providers.embedding]
url = "https://example.invalid/embeddings"

[embedding]
model = "vendor/embedding"
source = "embedding"
url = "https://example.invalid/embeddings"

[run]
round_index = 1
temperature = 0.01
chunk_size = 128
overlap_percent = 20
text_extension = 1
min_cross_scores = [1]
''', encoding="utf-8")
        config = load_config(str(folder / "project.toml"))
        self.assertEqual(config.PROJECT["language"], "en")
        self.assertEqual(config.REQUEST["budget"], {
            "max_calls": None, "max_tokens": None, "max_cost": None, "max_runtime": None,
        })
        spec = ResearchSpecification.from_config(config.REQUEST, config).data
        self.assertEqual(spec["request_limits"]["vendor/model-a"],
                         {"max_input_tokens": None, "max_output_tokens": 4096})


if __name__ == "__main__":
    unittest.main()
