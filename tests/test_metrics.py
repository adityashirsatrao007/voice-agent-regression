import unittest

from vareg.metrics import (
    GATED_METRICS,
    TurnScore,
    ScenarioScore,
    aggregate,
    keyword_hit,
    keyword_counts,
    latency_stats,
    normalize,
    percentile,
    ratio,
    token_f1,
)


class NormaliseTests(unittest.TestCase):
    def test_case_and_punctuation_are_dropped(self) -> None:
        self.assertEqual(normalize("  Booked!  consultation,  "), "booked consultation")

    def test_devanagari_matras_survive(self) -> None:
        # \w-based normalisers split मराठी into two tokens; this one must not.
        self.assertEqual(normalize("मराठी बोलता"), "मराठी बोलता")

    def test_danda_is_dropped_but_matra_is_not(self) -> None:
        self.assertEqual(normalize("नमस्ते।"), "नमस्ते")

    def test_empty_string_stays_empty(self) -> None:
        self.assertEqual(normalize(""), "")


class TokenF1Tests(unittest.TestCase):
    def test_identical_text_scores_one(self) -> None:
        self.assertEqual(token_f1("book a consultation", "book a consultation"), 1.0)

    def test_disjoint_text_scores_zero(self) -> None:
        self.assertEqual(token_f1("cancel booking", "working hours"), 0.0)

    def test_partial_overlap_is_hand_computable(self) -> None:
        # overlap 3 of 5 predicted, 3 of 3 gold -> P=0.6 R=1.0 F1=0.75
        self.assertAlmostEqual(token_f1("book a consultation for two", "book a consultation"), 0.75)

    def test_extra_gold_word_is_hand_computable(self) -> None:
        # overlap 2; P=1.0, R=2/3 -> F1 = 2*(1)*(2/3)/(1+2/3) = 0.8
        self.assertAlmostEqual(token_f1("cancel booking", "cancel my booking"), 0.8)

    def test_empty_prediction_scores_zero(self) -> None:
        self.assertEqual(token_f1("", "book a consultation"), 0.0)

    def test_empty_gold_scores_zero(self) -> None:
        self.assertEqual(token_f1("anything", ""), 0.0)

    def test_repeated_tokens_do_not_double_count(self) -> None:
        # Counter intersection caps the overlap at gold's multiplicity.
        self.assertAlmostEqual(token_f1("refund refund refund", "refund refund"), 0.8)

    def test_punctuation_differences_still_match(self) -> None:
        self.assertEqual(token_f1("Confirmation BK-100001.", "confirmation bk-100001"), 1.0)


class KeywordTests(unittest.TestCase):
    def test_whole_phrase_match(self) -> None:
        self.assertTrue(keyword_hit("Booked consultation today", "booked"))

    def test_substring_does_not_match_longer_token(self) -> None:
        # "book" must not satisfy a gold keyword of "booking" by prefix alone.
        self.assertFalse(keyword_hit("the booking is confirmed", "booked"))

    def test_case_insensitive(self) -> None:
        self.assertTrue(keyword_hit("I said no", "NO"))

    def test_punctuation_inside_keyword_is_ignored(self) -> None:
        self.assertTrue(keyword_hit("Confirmation BK-100001.", "BK-100001"))

    def test_missing_keyword(self) -> None:
        self.assertFalse(keyword_hit("nothing here", "sunday"))

    def test_empty_keyword_never_matches(self) -> None:
        self.assertFalse(keyword_hit("anything", ""))

    def test_counts(self) -> None:
        self.assertEqual(keyword_counts("hours are 10:00 to 18:00", ["10:00", "18:00"]), (2, 2))
        self.assertEqual(keyword_counts("hours", ["10:00", "18:00"]), (0, 2))


class ArithmeticTests(unittest.TestCase):
    def test_ratio_of_empty_denominator_is_one(self) -> None:
        self.assertEqual(ratio(0, 0), 1.0)

    def test_ratio(self) -> None:
        self.assertAlmostEqual(ratio(3, 4), 0.75)

    def test_percentile_nearest_rank(self) -> None:
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.95), 5.0)
        self.assertEqual(percentile([1, 2, 3, 4, 5], 0.5), 3.0)

    def test_percentile_of_empty_is_zero(self) -> None:
        self.assertEqual(percentile([], 0.95), 0.0)

    def test_latency_stats(self) -> None:
        stats = latency_stats([10.0, 20.0, 30.0])
        self.assertEqual(stats, {"steps": 3, "mean_ms": 20.0, "p95_ms": 30.0})

    def test_latency_stats_empty(self) -> None:
        self.assertEqual(latency_stats([]), {"steps": 0, "mean_ms": 0.0, "p95_ms": 0.0})


def _turn(**overrides) -> TurnScore:
    base = dict(
        observed_tools=("book_appointment",),
        expected_tools=("book_appointment",),
        tools_ok=True,
        args_matched=2,
        args_total=2,
        forbid_ok=True,
        forbid_args_ok=True,
        clarify_matched=0,
        clarify_total=0,
        keyword_matched=1,
        keyword_total=1,
        token_f1=1.0,
        budget_ok=True,
    )
    base.update(overrides)
    return TurnScore(**base)


class AggregateTests(unittest.TestCase):
    """Hand-computed: the numbers below would change if aggregation broke."""

    def test_hand_computed_aggregate(self) -> None:
        scores = [
            ScenarioScore("a", True, [_turn()], steps=2, latency_ms=1.0),
            ScenarioScore(
                "b",
                False,
                [
                    _turn(
                        tools_ok=False,
                        args_matched=1,
                        args_total=2,
                        keyword_matched=0,
                        keyword_total=1,
                        clarify_matched=1,
                        clarify_total=2,
                        token_f1=0.5,
                    )
                ],
                steps=3,
                latency_ms=2.0,
            ),
        ]
        metrics = aggregate(scores)
        self.assertEqual(metrics["tool_accuracy"], 0.5)
        self.assertEqual(metrics["arg_field_exact"], 0.75)
        self.assertEqual(metrics["token_f1"], 0.75)
        self.assertEqual(metrics["keyword_recall"], 0.5)
        self.assertEqual(metrics["clarification_accuracy"], 0.5)
        self.assertEqual(metrics["scenario_pass_rate"], 0.5)

    def test_gold_only_denominators_survive_a_silent_backend(self) -> None:
        # No tools called at all: metrics must fall, not vacuously pass.
        silent = _turn(
            observed_tools=(), tools_ok=False, args_matched=0, keyword_matched=0,
            clarify_matched=0, token_f1=0.0,
        )
        metrics = aggregate([ScenarioScore("a", False, [silent], steps=1, latency_ms=0.0)])
        self.assertEqual(metrics["tool_accuracy"], 0.0)
        self.assertEqual(metrics["arg_field_exact"], 0.0)
        self.assertEqual(metrics["token_f1"], 0.0)
        self.assertEqual(metrics["keyword_recall"], 0.0)
        self.assertEqual(metrics["scenario_pass_rate"], 0.0)

    def test_every_gated_metric_is_reported(self) -> None:
        metrics = aggregate([ScenarioScore("a", True, [_turn()], steps=1, latency_ms=0.0)])
        self.assertEqual(set(GATED_METRICS), set(metrics))


if __name__ == "__main__":
    unittest.main()
