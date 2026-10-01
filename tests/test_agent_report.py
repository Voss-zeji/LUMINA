from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

# The sibling fixture module lives beside this file; every test drives the public controller.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from lumina import prompts
from lumina.agent.budget import Budget
from lumina.agent.contracts import validate_manifest
from lumina.agent.controller import approve, begin, execute
from lumina.agent.report import write_report
from lumina.agent.store import Store, StoreError
from test_agent_contracts import config_for, request_for  # noqa: E402 - sibling fixture module


def _cross_row(path):
    """The whole header plus the single vote row, so a rewrite keeps every column."""
    lines = [line.split("\t") for line in path.read_text(encoding="utf-8").splitlines() if line]
    return lines[0], list(lines[1])


def _rewrite_cross(path, **changes):
    header, row = _cross_row(path)
    for column, value in changes.items():
        row[header.index(column)] = value
    path.write_text("\t".join(header) + "\n" + "\t".join(row) + "\n", encoding="utf-8")


class ReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.papers, self.rounds = 1, ()

    # ---- synthetic run fixture: the real controller, mocked HTTP only ---------

    def request(self, domain="aqua", count=1):
        papers = []
        for index in range(count):
            paper = self.root / f"{domain}-{index}.md"
            paper.write_text(f"# Introduction\nChina. Flux is 2. Table 2. Paper {index}.", encoding="utf-8")
            papers.append(str(paper))
        request = request_for(papers[0], domain)
        request["papers"] = papers
        return request

    def client(self, domain="aqua", *, flags=None):
        client = Mock()

        def create(**payload):
            messages = payload["messages"]
            if messages[0]["content"] == prompts.message_system_ragQuery.strip():
                flag, quote = (flags or (1, "China"))
                content = dict(existing_flag=flag, direct_quote=quote if flag == 1 else None)
            else:
                question = [q.strip("\n") for q in prompts.questions_for_domain(domain)].index(messages[-1]["content"]) + 1
                item = "Study_location" if question == 1 else ("Specie" if domain == "aqua" and question == 2 else "Flux-1")
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

    def finish(self, run, domain="aqua", *, flags=None):
        """Drive one complete, mocked aqua run and return its directory."""
        config = config_for(domain)
        client = self.client(domain, flags=flags)
        request = self.request(domain, count=self.papers)
        if self.rounds:
            request["rounds"] = list(self.rounds)
        with patch("lumina.agent.runtime.OpenAI", return_value=client), \
             patch("lumina.agent.runtime.requests.post", return_value=self.embedding()) as post, \
             patch("lumina.agent.runtime.RunContext.sleep"):
            run = begin(request, config, self.root / "runs", dry_run=True)
            execute(run, config)
            with Store(run) as store:
                gate = next(g for g in store.gates() if g["kind"] == "smoke")
            approve(run, gate["gate_id"], "Reviewed synthetic smoke", config=config)
            self.assertEqual(execute(run, config)["state"], "DONE")
        self.assertGreater(post.call_count, 0)
        return run

    def report(self, run, **kwargs):
        """The credential-free CLI context: no config, no credentials."""
        with Store(run) as store:
            spec = validate_manifest(run, store.snapshot()["run"]["spec_hash"])["specification"]
            context = SimpleNamespace(run_dir=run, spec=spec, store=store,
                                      budget=Budget(store, spec))
            return write_report(context, "status_report", **kwargs)

    def qc(self, run):
        """The archived checkpoint the controller wrote, exactly as the CLI reads it."""
        return json.loads((run / "checkpoints" / "final_qc.json").read_text(encoding="utf-8"))

    # ---- truthful coverage of a complete run ---------------------------------

    def test_complete_run_reports_real_votes_and_no_secret(self):
        run = self.finish(self.root)
        report = self.report(run, final_qc=self.qc(run))
        self.assertEqual(report["outcome"], "success", report["diagnostics"])
        self.assertEqual(report["scientific_validation"], "NOT_EVALUATED")
        extraction = report["extraction"]
        self.assertEqual(extraction["completed"], extraction["expected"])
        self.assertEqual(extraction["by_classification"], {"completed_with_value": extraction["expected"]})
        verification = report["verification"]
        self.assertEqual(verification["completed_positive"], verification["expected_votes"])
        self.assertEqual(verification["completed_negative"], 0)
        self.assertEqual(verification["failed_votes"] + verification["pending_votes"], 0)
        self.assertEqual(verification["self_votes"], 0)
        self.assertEqual(verification["stale_votes"], 0)
        self.assertEqual(verification["incomplete_candidates"], 0)
        self.assertGreater(verification["eligible_candidates"], 0)
        self.assertEqual(verification["without_evidence"], 0)
        self.assertTrue(all(c["verified"] for c in verification["candidates"]))
        thresholds = report["thresholds"]["thresholds"]
        self.assertEqual(set(thresholds), {"full", "MiniCross01"})
        self.assertTrue(all(row["result_kind"] == "populated" for rows in thresholds.values()
                            for row in rows))
        self.assertNotIn("SECRET-DO-NOT-PERSIST", json.dumps(report))
        self.assertEqual(report["archived_qc"]["status"], "PASSED")
        self.assertEqual(report["archived_qc"]["by_status"], {"verified": len(report["final_qc"]["outputs"])})
        self.assertEqual(report["final_qc"]["passed"], True)
        self.assertTrue(report["outputs"])
        self.assertTrue(all(a["sha256"] and a["path"].startswith("outputs/") for a in report["outputs"]))
        self.assertEqual(report["fingerprints"]["specification"], report["specification_hash"])
        self.assertTrue(report["fingerprints"]["control_code"])
        self.assertEqual(report["errors"], [], report["errors"])
        self.assertEqual(report["calls"]["retries"], 0)

    def test_negative_votes_are_counted_not_confused_with_errors(self):
        run = self.finish(self.root, flags=(0, None))
        report = self.report(run)
        verification = report["verification"]
        self.assertEqual(verification["completed_negative"], verification["expected_votes"])
        self.assertEqual(verification["completed_positive"], 0)
        self.assertEqual(verification["invalid_artifacts"], 0)

    def test_edited_valid_vote_is_not_counted_as_a_real_negative(self):
        run = self.finish(self.root)
        vote = next((run / "outputs" / "R01" / "cross").rglob("Candidate_*==Output_*.csv"))
        _rewrite_cross(vote, existing_flag="0", direct_quote="")
        report = self.report(run)
        self.assertEqual(report["verification"]["completed_negative"], 0)
        self.assertEqual(report["verification"]["changed_artifacts"], 1)
        self.assertLess(report["verification"]["completed_positive"], report["verification"]["expected_votes"])

    def test_threshold_keeps_missing_votes_in_the_eligible_denominator(self):
        run = self.finish(self.root)
        vote = next((run / "outputs" / "R01" / "cross").rglob("Candidate_*==Output_*.csv"))
        vote.unlink()
        report = self.report(run)
        row = next(r for r in report["thresholds"]["thresholds"]["MiniCross01"] if r["incomplete_candidates"])
        self.assertEqual(row["eligible_candidates"], 2)
        self.assertEqual(row["verifiable_candidates"], 1)
        self.assertEqual(row["pass_rate"], .5)
        self.assertEqual(row["possible_upper_rate"], 1)
        self.assertEqual(row["rate_status"], "lower_bound_incomplete")

    # ---- extraction denominator and classifications ---------------------------

    def test_missing_output_and_tampered_output_are_not_completed_work(self):
        run = self.finish(self.root)
        outputs = sorted((run / "outputs" / "R01" / "examiner").rglob("*_R01.csv"))
        outputs[0].unlink()
        outputs[1].write_text("item\tvalue\tevidence\tconfidence_lv\nA\t1\tx\t9\n", encoding="utf-8")
        report = self.report(run)
        classes = report["extraction"]["by_classification"]
        self.assertEqual(classes.get("missing_output"), 1)
        self.assertEqual(classes.get("changed_or_unreceipted_output"), 1)
        self.assertLess(report["extraction"]["completed"], report["extraction"]["expected"])
        self.assertEqual(report["outcome"], "incomplete")
        self.assertTrue(report["extraction"]["missing_outputs"])

    def test_valid_no_value_answer_still_counts_as_completed(self):
        config = config_for()
        client = self.client()
        original = client.chat.completions.create.side_effect

        def create(**payload):
            messages = payload["messages"]
            if messages[0]["content"] != prompts.message_system_ragQuery.strip() and \
                    [q.strip("\n") for q in prompts.questions_for_domain("aqua")].index(messages[-1]["content"]) + 1 == 1:
                response = dict(choices=[dict(message=dict(content=json.dumps(
                    {"Study_location": dict(value=None, evidence=None, confidence_lv=80)})), finish_reason="stop")],
                    usage=dict(prompt_tokens=12, completion_tokens=5, total_tokens=17))
                return SimpleNamespace(model_dump=lambda **_: response)
            return original(**payload)

        client.chat.completions.create.side_effect = create
        with patch("lumina.agent.runtime.OpenAI", return_value=client), \
             patch("lumina.agent.runtime.requests.post", return_value=self.embedding()), \
             patch("lumina.agent.runtime.RunContext.sleep"):
            run = begin(self.request(), config, self.root / "runs", dry_run=True)
            execute(run, config)
            with Store(run) as store:
                gate = next(g for g in store.gates() if g["kind"] == "smoke")
            approve(run, gate["gate_id"], "Reviewed synthetic smoke", config=config)
            execute(run, config)
        report = self.report(run)
        classes = report["extraction"]["by_classification"]
        self.assertIn("completed_no_value", classes)
        self.assertEqual(report["extraction"]["completed"], report["extraction"]["expected"])
        self.assertEqual(report["extraction"]["answers"]["no_value"], 2)
        self.assertEqual(report["extraction"]["answers"]["with_value_no_evidence"], 0)

    # ---- verification denominator and excluded artifacts ----------------------

    def test_self_outdated_and_unselected_votes_never_count(self):
        run = self.finish(self.root)
        votes = sorted((run / "outputs" / "R01" / "cross").rglob("Candidate_*==Output_*.csv"))
        self.assertTrue(votes)
        # Outdated fingerprint: correct row, stale signature.
        _rewrite_cross(votes[0], verification_fingerprint="0" * 64)
        # A self vote and an unselected verifier never enter the denominator.
        header, row = _cross_row(votes[1])
        _rewrite_cross(votes[1], output_model=row[header.index("input_model")])
        _rewrite_cross(votes[2], output_model="Model_Z")
        report = self.report(run)
        verification = report["verification"]
        self.assertEqual(verification["stale_votes"], 1)
        self.assertGreaterEqual(verification["self_votes"], 1)
        self.assertGreaterEqual(verification["not_selected_artifacts"], 1)
        counted = verification["completed_positive"] + verification["completed_negative"]
        self.assertLess(counted, verification["expected_votes"])
        self.assertEqual(verification["failed_votes"] + verification["pending_votes"]
                         + verification["stale_votes"] + verification["missing_artifacts"],
                         verification["expected_votes"] - counted)

    def test_missing_vote_artifact_counts_pending_not_zero(self):
        run = self.finish(self.root)
        votes = sorted((run / "outputs" / "R01" / "cross").rglob("Candidate_*==Output_*.csv"))
        votes[0].unlink()
        report = self.report(run)
        verification = report["verification"]
        self.assertEqual(verification["completed_positive"] + verification["completed_negative"],
                         verification["expected_votes"] - 1)
        self.assertEqual(verification["pending_votes"], 0)
        self.assertEqual(verification["missing_artifacts"], 1)
        self.assertEqual(verification["stale_votes"] + verification["failed_votes"], 0)
        self.assertEqual(verification["incomplete_candidates"], 1)

    # ---- thresholds, disagreement and archive --------------------------------

    def test_threshold_fraction_over_all_eligible_candidates(self):
        run = self.finish(self.root)
        report = self.report(run)
        thresholds = report["thresholds"]["thresholds"]
        for label, rows in thresholds.items():
            for row in rows:
                if row["eligible_candidates"]:
                    self.assertEqual(row["pass_rate"], row["passing"] / row["eligible_candidates"])
                    self.assertLessEqual(row["passing"], row["verifiable_candidates"])
                else:
                    self.assertIsNone(row["pass_rate"])
        full = thresholds["full"]
        self.assertEqual(thresholds["MiniCross01"][0]["round_index"], full[0]["round_index"])

    def test_incomplete_verification_distinguishes_legitimate_empty(self):
        run = self.finish(self.root)
        votes = sorted((run / "outputs" / "R01" / "cross").rglob("Candidate_*==Output_*.csv"))
        votes[0].unlink()
        report = self.report(run)
        kinds = {row["result_kind"] for rows in report["thresholds"]["thresholds"].values()
                 for row in rows}
        self.assertTrue(kinds <= {"populated", "incomplete_verification", "missing"})
        incomplete = [row for rows in report["thresholds"]["thresholds"].values() for row in rows
                      if row["result_kind"] == "incomplete_verification"]
        if incomplete:
            self.assertTrue(all(row["incomplete_candidates"] for row in incomplete))

    def test_conflicting_extractions_are_preserved_without_unit_conversion(self):
        run = self.finish(self.root)
        composites = sorted((run / "outputs" / "R01" / "composite").glob("*.xlsx"))
        report = self.report(run)
        self.assertIn("conflict_count", report["disagreement"])
        for conflict in report["disagreement"]["conflicts"]:
            self.assertGreaterEqual(len({a["normalized"] for a in conflict["answers"].values()}), 2)
            self.assertNotIn("winner", conflict)
        self.assertTrue(composites)

    def test_archived_output_change_is_reported_not_trusted(self):
        run = self.finish(self.root)
        qc = json.loads((run / "checkpoints" / "final_qc.json").read_text(encoding="utf-8"))
        target = run / qc["outputs"][0]["path"]
        target.write_text(target.read_text(encoding="utf-8") + "tampered", encoding="utf-8")
        report = self.report(run, final_qc=qc)
        archived = report["archived_qc"]
        self.assertEqual(archived["claimed_passed"], True)
        self.assertEqual(archived["by_status"].get("changed"), 1)
        self.assertNotEqual(archived["status"], "PASSED")
        self.assertEqual(report["outcome"], "incomplete")
        self.assertEqual(report["final_qc"]["passed"], True)

    def test_corrupt_ledger_raises_and_missing_artifact_does_not(self):
        run = self.finish(self.root)
        (run / "outputs" / "R01" / "composite" / "LUMINA_Q01.xlsx").unlink()
        report = self.report(run)
        self.assertTrue(any(d["section"] == "composite" for d in report["diagnostics"]))
        # An artifact gap is a diagnostic; a damaged authority is the one thing that raises.
        self.assertEqual(report["verification"]["eligible_candidates"], 4)
        self.assertEqual(report["outcome"], "incomplete")
        # Corrupt the authority itself; the report must still refuse to certify it.
        with self.assertRaises(StoreError):
            with Store(run) as store:
                store._conn.execute("UPDATE run SET state='NOT_A_STATE' WHERE singleton=1")
                self.report(run)

    def test_report_never_dispatches_http(self):
        run = self.finish(self.root)
        with patch("lumina.agent.runtime.OpenAI") as sdk, patch("lumina.agent.runtime.requests.post") as post:
            self.report(run)
        sdk.assert_not_called()
        post.assert_not_called()

    # ---- multiple rounds, multiple papers and a narrowed scope --------------

    def test_two_rounds_are_kept_apart_in_every_denominator(self):
        self.rounds = (1, 2)
        run = self.finish(self.root)
        report = self.report(run, final_qc=self.qc(run))
        self.assertEqual(report["rounds"], [1, 2])
        self.assertEqual(report["extraction"]["expected"], 3 * 2 * 2)
        self.assertEqual(report["extraction"]["completed"], report["extraction"]["expected"])
        rounds = {c["round_index"] for c in report["verification"]["candidates"]}
        self.assertEqual(rounds, {1, 2})
        threshold_rows = {(row["round_index"], row["question"])
                          for row in report["thresholds"]["thresholds"]["full"]}
        self.assertEqual(threshold_rows, {(r, q) for r in (1, 2) for q in (1, 2, 3)})
        self.assertTrue(all("R01" in row["file"] or "R02" in row["file"]
                            for row in report["thresholds"]["thresholds"]["full"]))
        self.assertEqual(report["outcome"], "success")

    def test_scope_limit_changes_only_the_denominator_not_the_artifacts(self):
        self.papers = 2
        run = self.finish(self.root)
        full = self.report(run, final_qc=self.qc(run))
        one_uid = next(p["paper_uid"] for p in full["inputs"])
        scoped = self.report(run, paper_uids=[one_uid], final_qc=self.qc(run))
        self.assertEqual(scoped["paper_count"], 1)
        self.assertEqual(scoped["extraction"]["expected"], 3 * 2 * 1 * 1)
        self.assertEqual(scoped["extraction"]["completed"], scoped["extraction"]["expected"])
        self.assertEqual({c["paper_uid"] for c in scoped["verification"]["candidates"]}, {one_uid})
        self.assertEqual(scoped["outcome"], "success")

    def test_wildfire_text_and_numeric_questions_both_report(self):
        run = self.finish(self.root, domain="wildfire")
        report = self.report(run, final_qc=self.qc(run))
        self.assertEqual(report["extraction"]["expected"], 4 * 2 * 1 * 1)
        self.assertEqual(report["extraction"]["completed"], report["extraction"]["expected"])
        self.assertEqual(report["verification"]["completed_positive"], report["verification"]["expected_votes"])
        self.assertEqual(len(report["thresholds"]["thresholds"]["full"]), 4)
        self.assertEqual(report["outcome"], "success", report["diagnostics"])


if __name__ == "__main__":
    unittest.main()
