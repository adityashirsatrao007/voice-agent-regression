"""End-to-end tests for the three commands, through main() and its exit codes.

Every test chdirs into a temporary directory first: `run` writes into
``results/``, `regress` reads ``baseline.json``, and neither may touch the
repository while the suite runs.
"""

import contextlib
import io
import json
import os
import tempfile
import unittest
from pathlib import Path

from vareg.cli import main
from vareg.metrics import GATED_METRICS
from vareg.scenarios import load_matrix

# Observed with the shipped prompts: v2 passes everything, v1 fails exactly
# these three. If a prompt edit moves this set, the test says so by name.
V1_FAILURES = {"booking_missing_phone", "booking_missing_slot", "cancellation_policy"}


@contextlib.contextmanager
def isolated_cwd():
    original = Path.cwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)
        try:
            yield Path(tmp)
        finally:
            os.chdir(original)


def invoke(argv):
    """Run main() capturing both streams; returns (code, stdout, stderr)."""
    out, err = io.StringIO(), io.StringIO()
    with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        code = main(argv)
    return code, out.getvalue(), err.getvalue()


def read_report(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


class ListScenariosTests(unittest.TestCase):
    def test_one_table_row_per_scenario(self) -> None:
        code, out, _ = invoke(["list-scenarios"])
        self.assertEqual(code, 0)
        matrix = load_matrix()
        rows = [line for line in out.splitlines() if line.startswith("| ")][2:]
        self.assertEqual(len(rows), len(matrix))
        for scenario in matrix:
            self.assertIn(f"| {scenario.id} |", out)

    def test_matrix_header_is_present(self) -> None:
        _, out, _ = invoke(["list-scenarios"])
        self.assertIn(f"## scenario matrix ({len(load_matrix())} scenarios)", out)
        self.assertIn("| id | tags | turns | title |", out)

    def test_missing_scenario_directory_exits_2(self) -> None:
        code, _, err = invoke(["list-scenarios", "--dir", "/nonexistent/scenarios"])
        self.assertEqual(code, 2)
        self.assertIn("error:", err)


class RunCommandTests(unittest.TestCase):
    def test_run_writes_json_and_markdown_reports(self) -> None:
        with isolated_cwd():
            code, out, _ = invoke(["run", "--quiet"])
            self.assertEqual(code, 0)
            self.assertIn("wrote results/report.json", out)
            self.assertIn("wrote results/report.md", out)
            report = read_report(Path("results/report.json"))
            self.assertEqual(report["backend"], "mock")
            self.assertEqual(len(report["scenarios"]), len(load_matrix()))
            self.assertTrue(Path("results/report.md").is_file())

    def test_stdout_report_is_byte_identical_to_report_md(self) -> None:
        with isolated_cwd():
            invoke(["run", "--quiet"])
            _, out, _ = invoke(["run"])
            report_md = Path("results/report.md").read_text(encoding="utf-8")
            self.assertIn(report_md, out)

    def test_quiet_keeps_the_paths_but_drops_the_table(self) -> None:
        with isolated_cwd():
            _, out, _ = invoke(["run", "--quiet"])
            self.assertNotIn("| scenario | result |", out)
            self.assertIn("wrote results/report.json", out)

    def test_default_prompt_run_scores_perfectly(self) -> None:
        with isolated_cwd():
            code, _, _ = invoke(["run", "--quiet"])
            self.assertEqual(code, 0)
            metrics = read_report(Path("results/report.json"))["metrics"]
            for name in GATED_METRICS:
                self.assertEqual(metrics[name], 1.0, name)

    def test_v1_prompt_fails_exactly_three_scenarios(self) -> None:
        with isolated_cwd():
            code, _, _ = invoke(["run", "--prompts", "v1", "--quiet", "--out", "results/v1"])
            self.assertEqual(code, 0)  # run is report-only; the gate lives in regress
            report = read_report(Path("results/v1/report.json"))
            failed = {row["id"] for row in report["scenarios"] if not row["passed"]}
            self.assertEqual(failed, V1_FAILURES)
            self.assertLess(report["metrics"]["scenario_pass_rate"], 1.0)
            self.assertLess(report["metrics"]["tool_accuracy"], 1.0)

    def test_mock_run_reports_no_tokens_and_no_cost(self) -> None:
        with isolated_cwd():
            invoke(["run", "--quiet"])
            usage = read_report(Path("results/report.json"))["usage"]
            self.assertIsNone(usage["input_tokens"])
            self.assertIsNone(usage["output_tokens"])
            self.assertIsNone(usage["cost_usd"])
            self.assertIn("offline", usage["note"])

    def test_latency_block_counts_real_decisions(self) -> None:
        with isolated_cwd():
            invoke(["run", "--quiet"])
            latency = read_report(Path("results/report.json"))["latency"]
            self.assertGreater(latency["steps"], 0)
            self.assertGreaterEqual(latency["mean_ms"], 0.0)
            self.assertGreaterEqual(latency["p95_ms"], 0.0)

    def test_scenario_filter_runs_only_what_was_asked_for(self) -> None:
        with isolated_cwd():
            code, _, _ = invoke(["run", "--scenarios", "refusal_otp", "--quiet"])
            self.assertEqual(code, 0)
            report = read_report(Path("results/report.json"))
            self.assertEqual([row["id"] for row in report["scenarios"]], ["refusal_otp"])

    def test_unknown_scenario_id_exits_2_with_a_hint(self) -> None:
        with isolated_cwd():
            code, _, err = invoke(["run", "--scenarios", "no_such_scenario"])
            self.assertEqual(code, 2)
            self.assertIn("unknown scenario id", err)
            self.assertIn("list-scenarios", err)

    def test_unknown_prompt_version_exits_2_and_names_what_is_known(self) -> None:
        with isolated_cwd():
            code, _, err = invoke(["run", "--prompts", "v9"])
            self.assertEqual(code, 2)
            self.assertIn("unknown prompt version", err)
            self.assertIn("v1", err)
            self.assertIn("v2", err)

    def test_unknown_backend_is_rejected_by_the_parser(self) -> None:
        with isolated_cwd():
            with self.assertRaises(SystemExit) as caught:
                invoke(["run", "--backend", "gpt5"])
            self.assertEqual(caught.exception.code, 2)

    def test_write_baseline_captures_the_run_it_just_made(self) -> None:
        with isolated_cwd() as tmp:
            code, out, _ = invoke(["run", "--quiet", "--write-baseline"])
            self.assertEqual(code, 0)
            self.assertIn("wrote baseline.json", out)
            baseline = read_report(tmp / "baseline.json")
            report = read_report(Path("results/report.json"))
            self.assertEqual(baseline["metrics"], report["metrics"])
            self.assertEqual(
                [row["id"] for row in baseline["scenarios"]],
                [row["id"] for row in report["scenarios"]],
            )


class RegressCommandTests(unittest.TestCase):
    def _make_pair(self, cwd: Path) -> None:
        """Baseline from the default prompt, deliberately degraded candidate from v1."""
        self.assertEqual(invoke(["run", "--quiet", "--write-baseline"])[0], 0)
        self.assertEqual(invoke(["run", "--prompts", "v1", "--quiet", "--out", "results/v1"])[0], 0)

    def test_missing_baseline_exits_2(self) -> None:
        with isolated_cwd():
            code, _, err = invoke(["regress", "--candidate", "nowhere.json"])
            self.assertEqual(code, 2)
            self.assertIn("not found", err)

    def test_missing_candidate_file_exits_2(self) -> None:
        with isolated_cwd():
            invoke(["run", "--quiet", "--write-baseline"])
            code, _, err = invoke(["regress", "--candidate", "missing.json"])
            self.assertEqual(code, 2)
            self.assertIn("not found", err)

    def test_fresh_run_against_its_own_baseline_passes(self) -> None:
        with isolated_cwd():
            invoke(["run", "--quiet", "--write-baseline"])
            code, out, _ = invoke(["regress"])
            self.assertEqual(code, 0)
            self.assertIn("GATE PASS - no gated metric regressed", out)
            self.assertIn("baseline: baseline.json", out)

    def test_degraded_candidate_fails_with_exit_1(self) -> None:
        with isolated_cwd() as cwd:
            self._make_pair(cwd)
            code, out, _ = invoke(["regress", "--candidate", "results/v1/report.json"])
            self.assertEqual(code, 1)
            self.assertIn("GATE FAIL", out)
            for metric in ("tool_accuracy", "scenario_pass_rate", "arg_field_exact"):
                self.assertIn(f"| {metric} |", out)
            self.assertIn("candidate: results/v1/report.json", out)

    def test_threshold_can_absorb_a_small_drop(self) -> None:
        with isolated_cwd() as cwd:
            self._make_pair(cwd)
            # Worst observed v1 drop is scenario_pass_rate 1.0 -> 0.8125.
            code, out, _ = invoke(
                ["regress", "--candidate", "results/v1/report.json", "--threshold", "0.5"]
            )
            self.assertEqual(code, 0)
            self.assertIn("GATE PASS", out)

    def test_update_baseline_accepts_the_candidate_next_time(self) -> None:
        with isolated_cwd() as cwd:
            self._make_pair(cwd)
            code, _, _ = invoke(
                ["regress", "--candidate", "results/v1/report.json", "--update-baseline"]
            )
            self.assertEqual(code, 1)
            code, out, _ = invoke(["regress", "--candidate", "results/v1/report.json"])
            self.assertEqual(code, 0)
            self.assertIn("GATE PASS", out)

    def test_scenarios_dropped_from_the_candidate_fail_the_gate(self) -> None:
        with isolated_cwd() as cwd:
            self._make_pair(cwd)
            # A candidate that only ran one of the sixteen baseline scenarios:
            # running fewer scenarios must never look like an improvement.
            code, _, _ = invoke(
                ["run", "--quiet", "--scenarios", "refusal_otp", "--out", "results/partial"]
            )
            self.assertEqual(code, 0)
            code, out, _ = invoke(["regress", "--candidate", "results/partial/report.json"])
            self.assertEqual(code, 1)
            self.assertIn("scenario_set", out)


class ParserTests(unittest.TestCase):
    def test_no_arguments_is_a_usage_error(self) -> None:
        with self.assertRaises(SystemExit) as caught:
            invoke([])
        self.assertEqual(caught.exception.code, 2)

    def test_run_help_lists_the_backend_choices(self) -> None:
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            with self.assertRaises(SystemExit) as caught:
                main(["run", "--help"])
        self.assertEqual(caught.exception.code, 0)
        self.assertIn("{mock,llm,langgraph}", buf.getvalue())


if __name__ == "__main__":
    unittest.main()
