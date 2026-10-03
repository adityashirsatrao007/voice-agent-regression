import tempfile
import unittest
from pathlib import Path

from vareg.prompts import (
    PROMPT_DIR,
    PromptError,
    load_manifest,
    load_spec,
    parse_faqs,
    parse_front_matter,
)


class ManifestTests(unittest.TestCase):
    def test_manifest_lists_both_versions(self) -> None:
        manifest = load_manifest()
        versions = [entry["version"] for entry in manifest["versions"]]
        self.assertEqual(versions, ["v1", "v2"])

    def test_default_is_v2(self) -> None:
        self.assertEqual(load_manifest()["default"], "v2")

    def test_every_version_carries_date_and_change_note(self) -> None:
        for entry in load_manifest()["versions"]:
            with self.subTest(version=entry.get("version")):
                self.assertTrue(entry.get("date"))
                self.assertTrue(entry.get("change"))
                self.assertEqual(len(entry["date"]), 10)  # ISO shape, YYYY-MM-DD

    def test_change_notes_are_distinct(self) -> None:
        notes = [entry["change"] for entry in load_manifest()["versions"]]
        self.assertEqual(len(set(notes)), len(notes))


class FrontMatterTests(unittest.TestCase):
    def test_v1_does_not_license_cancel(self) -> None:
        spec = load_spec("v1")
        self.assertNotIn("cancel_booking", spec.tools)

    def test_v2_licenses_cancel(self) -> None:
        spec = load_spec("v2")
        self.assertIn("cancel_booking", spec.tools)

    def test_v1_guesses_and_v2_asks(self) -> None:
        self.assertFalse(load_spec("v1").clarify_on_missing_slots)
        self.assertTrue(load_spec("v2").clarify_on_missing_slots)

    def test_front_matter_is_stripped_from_the_body(self) -> None:
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                body = load_spec(version).body
                self.assertNotIn("clarify_on_missing_slots", body)
                self.assertNotIn("tools:", body)
                self.assertIn("Sarva", body)

    def test_version_matches_the_manifest_entry(self) -> None:
        spec = load_spec("v2")
        self.assertEqual(spec.version, "v2")
        self.assertEqual(spec.date, "2026-10-03")

    def test_unknown_version_is_rejected_with_the_known_list(self) -> None:
        with self.assertRaisesRegex(PromptError, "v1, v2"):
            load_spec("v9")

    def test_unclosed_front_matter_is_rejected(self) -> None:
        with self.assertRaisesRegex(PromptError, "not closed"):
            parse_front_matter("---\nversion: v9\nbody text")

    def test_missing_front_matter_is_rejected(self) -> None:
        with self.assertRaisesRegex(PromptError, "must start"):
            parse_front_matter("just prose")

    def test_line_without_a_colon_is_rejected(self) -> None:
        with self.assertRaisesRegex(PromptError, "key: value"):
            parse_front_matter("---\nthis is not a pair\n---\nbody")

    def test_missing_file_is_reported(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            Path(tmp, "manifest.json").write_text(
                '{"default": "v1", "versions": [{"version": "v1", "date": "2026-10-03", "change": "c"}]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(PromptError, "prompt file not found"):
                load_spec("v1", Path(tmp))


class FaqTests(unittest.TestCase):
    def test_both_prompts_carry_the_hours_answer(self) -> None:
        for version in ("v1", "v2"):
            with self.subTest(version=version):
                answer = load_spec(version).faq_answer("What are your working hours?")
                self.assertIsNotNone(answer)
                self.assertIn("10:00", answer)
                self.assertIn("18:00", answer)

    def test_unrelated_question_gets_no_answer(self) -> None:
        self.assertIsNone(load_spec("v2").faq_answer("cancel my booking BK-123456"))

    def test_triggers_are_matched_case_insensitively(self) -> None:
        spec = load_spec("v2")
        self.assertIsNotNone(spec.faq_answer("WHAT ARE THE TIMINGS?"))

    def test_parse_faqs_requires_an_arrow(self) -> None:
        with self.assertRaisesRegex(PromptError, "no '=>'"):
            parse_faqs("FAQ: hours -> open all day")

    def test_parse_faqs_requires_both_halves(self) -> None:
        with self.assertRaisesRegex(PromptError, "trigger and an answer"):
            parse_faqs("FAQ: => ")


class PromptDiffTests(unittest.TestCase):
    """The two prompt files must differ where the README says they do."""

    def test_declared_tool_sets_differ_by_exactly_one_tool(self) -> None:
        v1, v2 = load_spec("v1"), load_spec("v2")
        self.assertEqual(set(v2.tools) - set(v1.tools), {"cancel_booking"})
        self.assertEqual(set(v1.tools) - set(v2.tools), set())

    def test_default_spec_resolves_from_the_manifest(self) -> None:
        self.assertEqual(load_spec().version, load_manifest()["default"])

    def test_prompt_files_are_present_on_disk(self) -> None:
        for version in ("v1", "v2"):
            self.assertTrue((PROMPT_DIR / f"{version}.md").is_file())


if __name__ == "__main__":
    unittest.main()
