import json
import tempfile
import unittest
from pathlib import Path

from vareg.scenarios import (
    SCENARIO_DIR,
    ScenarioError,
    load_matrix,
    load_scenario,
    parse_scenario,
)

MINIMAL = {
    "id": "sample",
    "title": "t",
    "tags": ["booking"],
    "max_steps": 3,
    "turns": [{"user": "hello", "expect": {"tools": []}}],
}

# The categories the resume line claims are covered; asserted against the
# committed matrix so a scenario cannot quietly disappear.
REQUIRED_TAGS = {
    "booking",
    "validation",
    "auth",
    "escalation",
    "wrong-tool",
    "clarification",
    "code-mixed",
    "refusal",
    "no-tool",
    "tool-error",
    "multi-turn",
    "injection",
    "budget",
}


class MatrixTests(unittest.TestCase):
    def test_matrix_loads(self) -> None:
        scenarios = load_matrix()
        self.assertGreaterEqual(len(scenarios), 14)

    def test_ids_are_unique_and_sorted(self) -> None:
        ids = [s.id for s in load_matrix()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids, sorted(ids))

    def test_every_scenario_file_parses(self) -> None:
        for path in sorted(SCENARIO_DIR.glob("*.json")):
            with self.subTest(path=path.name):
                scenario = load_scenario(path)
                self.assertEqual(scenario.id, path.stem)
                self.assertTrue(scenario.title)
                self.assertTrue(scenario.tags)

    def test_coverage_of_the_required_categories(self) -> None:
        tags = {tag for scenario in load_matrix() for tag in scenario.tags}
        missing = REQUIRED_TAGS - tags
        self.assertEqual(missing, set(), f"scenario matrix lost coverage: {sorted(missing)}")

    def test_gold_pins_at_least_one_argument_somewhere(self) -> None:
        pinned = [
            scenario
            for scenario in load_matrix()
            for turn in scenario.turns
            if turn.expect.args
        ]
        self.assertGreaterEqual(len(pinned), 8)

    def test_at_least_one_multi_turn_scenario(self) -> None:
        multi = [s for s in load_matrix() if len(s.turns) > 1]
        self.assertGreaterEqual(len(multi), 2)

    def test_step_budget_is_declared_everywhere(self) -> None:
        for scenario in load_matrix():
            with self.subTest(scenario=scenario.id):
                self.assertGreaterEqual(scenario.max_steps, 1)


class SchemaRejectionTests(unittest.TestCase):
    def test_missing_required_key(self) -> None:
        raw = dict(MINIMAL)
        del raw["max_steps"]
        with self.assertRaisesRegex(ScenarioError, "max_steps"):
            parse_scenario(raw)

    def test_id_must_match_file_name(self) -> None:
        with self.assertRaisesRegex(ScenarioError, "does not match file name"):
            parse_scenario(MINIMAL, source=Path("/tmp/other.json"))

    def test_zero_step_budget_rejected(self) -> None:
        raw = {**MINIMAL, "max_steps": 0}
        with self.assertRaisesRegex(ScenarioError, "positive integer"):
            parse_scenario(raw)

    def test_empty_turns_rejected(self) -> None:
        with self.assertRaisesRegex(ScenarioError, "non-empty"):
            parse_scenario({**MINIMAL, "turns": []})

    def test_empty_user_turn_rejected(self) -> None:
        raw = {**MINIMAL, "turns": [{"user": "   ", "expect": {"tools": []}}]}
        with self.assertRaisesRegex(ScenarioError, "'user' is empty"):
            parse_scenario(raw)

    def test_expect_tools_is_mandatory(self) -> None:
        raw = {**MINIMAL, "turns": [{"user": "hi", "expect": {"keywords": ["x"]}}]}
        with self.assertRaisesRegex(ScenarioError, "expect.tools"):
            parse_scenario(raw)

    def test_unknown_expectation_key_rejected(self) -> None:
        raw = {**MINIMAL, "turns": [{"user": "hi", "expect": {"tools": [], "expected_tool": "x"}}]}
        with self.assertRaisesRegex(ScenarioError, "unknown expectation keys"):
            parse_scenario(raw)

    def test_args_must_name_an_expected_tool(self) -> None:
        raw = {
            **MINIMAL,
            "turns": [
                {"user": "hi", "expect": {"tools": [], "args": {"book_appointment": {"date": "x"}}}}
            ],
        }
        with self.assertRaisesRegex(ScenarioError, "never calls it"):
            parse_scenario(raw)

    def test_forbidden_tool_cannot_also_be_expected(self) -> None:
        raw = {
            **MINIMAL,
            "turns": [
                {"user": "hi", "expect": {"tools": ["book_appointment"], "forbid": ["book_appointment"]}}
            ],
        }
        with self.assertRaisesRegex(ScenarioError, "both expected and forbidden"):
            parse_scenario(raw)

    def test_duplicate_ids_cannot_exist_because_id_must_equal_the_stem(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            (Path(tmp) / "sample.json").write_text(json.dumps(MINIMAL), encoding="utf-8")
            # A second file claiming the same id would need the same file name.
            (Path(tmp) / "other.json").write_text(json.dumps(MINIMAL), encoding="utf-8")
            with self.assertRaisesRegex(ScenarioError, "does not match file name"):
                load_matrix(Path(tmp))

    def test_missing_directory_reported(self) -> None:
        with self.assertRaisesRegex(ScenarioError, "not found"):
            load_matrix(Path("/nonexistent/scenarios"))

    def test_invalid_json_reported_with_the_path(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "broken.json"
            bad.write_text("{not json", encoding="utf-8")
            with self.assertRaisesRegex(ScenarioError, "not valid JSON"):
                load_scenario(bad)


if __name__ == "__main__":
    unittest.main()
