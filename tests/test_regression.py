import unittest

from vareg.metrics import GATED_METRICS
from vareg.regression import (
    RegressionError,
    compare,
    gate,
    load_run,
    render,
    verdict,
)


def run(metrics=None, ids=("booking_full_args", "refusal_otp"), backend="mock"):
    values = {name: 1.0 for name in GATED_METRICS}
    if metrics:
        values.update(metrics)
    return {
        "backend": backend,
        "metrics": values,
        "scenarios": [{"id": scenario_id} for scenario_id in ids],
    }


class GateTests(unittest.TestCase):
    def test_identical_runs_pass(self) -> None:
        self.assertEqual(gate(run(), run()), 0)

    def test_an_improvement_passes(self) -> None:
        baseline = run({"tool_accuracy": 0.5})
        candidate = run({"tool_accuracy": 1.0})
        self.assertEqual(gate(baseline, candidate), 0)

    def test_every_gated_metric_fails_when_it_drops(self) -> None:
        # The core regression test: break any single metric's comparison and
        # its subTest here goes red.
        for metric in GATED_METRICS:
            with self.subTest(metric=metric):
                baseline = run()
                candidate = run({metric: 0.5})
                self.assertEqual(gate(baseline, candidate), 1, metric)

    def test_a_small_drop_below_the_threshold_passes(self) -> None:
        baseline = run({"token_f1": 1.0})
        candidate = run({"token_f1": 0.9})
        self.assertEqual(gate(baseline, candidate, threshold=0.2), 0)
        self.assertEqual(gate(baseline, candidate, threshold=0.05), 1)

    def test_missing_metric_is_a_regression(self) -> None:
        candidate = run()
        del candidate["metrics"]["arg_field_exact"]
        self.assertEqual(gate(run(), candidate), 1)

    def test_non_numeric_metric_is_a_regression(self) -> None:
        candidate = run({"tool_accuracy": "1.0"})
        findings = compare(run(), candidate)
        by_name = {f.metric: f for f in findings}
        self.assertEqual(by_name["tool_accuracy"].status, "missing")

    def test_dropping_a_scenario_cannot_look_like_a_pass(self) -> None:
        findings = compare(run(ids=("a", "b")), run(ids=("a",)))
        self.assertEqual(findings[-1].status, "changed")
        self.assertEqual(gate(run(ids=("a", "b")), run(ids=("a",))), 1)

    def test_adding_a_scenario_is_also_flagged(self) -> None:
        findings = compare(run(ids=("a",)), run(ids=("a", "b")))
        self.assertIn("unexpected", findings[-1].metric)

    def test_delta_sign_convention(self) -> None:
        findings = compare(run({"keyword_recall": 1.0}), run({"keyword_recall": 0.75}))
        by_name = {f.metric: f for f in findings}
        self.assertAlmostEqual(by_name["keyword_recall"].delta, -0.25)
        self.assertEqual(by_name["keyword_recall"].status, "drop")

    def test_findings_follow_the_gated_metric_order(self) -> None:
        metrics = [f.metric for f in compare(run(), run())]
        self.assertEqual(metrics, list(GATED_METRICS))

    def test_verdict_text_for_both_outcomes(self) -> None:
        self.assertEqual(verdict(compare(run(), run())), "GATE PASS - no gated metric regressed")
        failing = compare(run(), run({"scenario_pass_rate": 0.5}))
        text = verdict(failing)
        self.assertIn("GATE FAIL", text)
        self.assertIn("scenario_pass_rate", text)

    def test_render_has_the_table_and_the_verdict(self) -> None:
        findings = compare(run(), run({"token_f1": 0.4}))
        output = render(findings, 0.0, verdict(findings))
        self.assertIn("| metric | baseline | candidate | delta | status |", output)
        self.assertIn("| token_f1 |", output)
        self.assertIn("GATE FAIL", output)

    def test_zero_baseline_with_a_zero_candidate_passes(self) -> None:
        baseline = run({"clarification_accuracy": 0.0})
        candidate = run({"clarification_accuracy": 0.0})
        self.assertEqual(gate(baseline, candidate), 0)


class LoadRunTests(unittest.TestCase):
    def test_missing_file(self) -> None:
        with self.assertRaisesRegex(RegressionError, "not found"):
            load_run(__import__("pathlib").Path("/nonexistent/report.json"))

    def test_invalid_json(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "report.json"
            bad.write_text("{oops", encoding="utf-8")
            with self.assertRaisesRegex(RegressionError, "not valid JSON"):
                load_run(bad)

    def test_file_without_metrics(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "report.json"
            path.write_text('{"backend": "mock"}', encoding="utf-8")
            with self.assertRaisesRegex(RegressionError, "no 'metrics'"):
                load_run(path)


if __name__ == "__main__":
    unittest.main()
