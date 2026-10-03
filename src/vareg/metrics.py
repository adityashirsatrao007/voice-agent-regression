"""Scoring: what counts as a pass, and how each metric is computed.

`token_f1` and `normalize` are adapted from `indic-rag-evals`
(`src/rageval/metrics.py`, same author): the SQuAD bag-of-tokens F1 and the
Devanagari-safe normaliser are copied nearly verbatim, with `is_word_char`
inlined because this repo does not carry the BM25 tokenizer it lives in. What is
*new* here is everything about trajectories — tool sequences, field-level
argument checks, forbidden calls, clarification coverage and the step budget.

Denominators come from the gold files, never from the agent's output, so a
backend that returns nothing scores 0.0 instead of vacuously passing.
"""

from __future__ import annotations

import math
import unicodedata
from collections import Counter
from dataclasses import dataclass, field
from typing import Iterable, Sequence

from .agent import Conversation, TurnRecord
from .scenarios import Expectation, Scenario

# Higher is better, all in [0, 1]. These are the metrics `regress` gates on.
GATED_METRICS: tuple[str, ...] = (
    "tool_accuracy",
    "arg_field_exact",
    "token_f1",
    "keyword_recall",
    "clarification_accuracy",
    "scenario_pass_rate",
)


def _is_word_char(ch: str) -> bool:
    """Letters, digits and combining marks — copied from rageval.bm25.

    Python's ``\\w`` and ``str.isalnum`` miss combining marks, so Devanagari
    matras would otherwise split ``मराठी`` into two tokens and every Hindi
    answer would score below its real overlap.
    """
    return ch.isalnum() or unicodedata.category(ch).startswith("M")


def normalize(text: str) -> str:
    """NFC-fold, lowercase, drop punctuation, collapse whitespace."""
    parts: list[str] = []
    buffer: list[str] = []
    for char in unicodedata.normalize("NFC", text).lower():
        if _is_word_char(char):
            buffer.append(char)
        elif buffer:
            parts.append("".join(buffer))
            buffer = []
    if buffer:
        parts.append("".join(buffer))
    return " ".join(parts)


def token_f1(prediction: str, gold: str) -> float:
    """SQuAD-style bag-of-tokens F1 over normalised text (0.0 - 1.0)."""
    predicted = normalize(prediction).split()
    reference = normalize(gold).split()
    if not predicted or not reference:
        return 0.0
    common = Counter(predicted) & Counter(reference)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(predicted)
    recall = overlap / len(reference)
    return 2.0 * precision * recall / (precision + recall)


def keyword_hit(text: str, keyword: str) -> bool:
    """Whole-phrase containment over normalised text.

    Padding with spaces means ``book`` does not silently match ``booking``,
    while punctuation and case differences (``BK-100001`` vs ``bk 100001``)
    stop being a problem.
    """
    needle = normalize(keyword)
    if not needle:
        return False
    haystack = f" {normalize(text)} "
    return f" {needle} " in haystack


def keyword_counts(text: str, keywords: Sequence[str]) -> tuple[int, int]:
    matched = sum(1 for kw in keywords if keyword_hit(text, kw))
    return matched, len(keywords)


def ratio(numerator: int | float, denominator: int | float) -> float:
    """1.0 for an empty denominator: nothing was demanded, nothing failed."""
    if denominator == 0:
        return 1.0
    return float(numerator) / float(denominator)


@dataclass
class TurnScore:
    observed_tools: tuple[str, ...]
    expected_tools: tuple[str, ...]
    tools_ok: bool
    args_matched: int
    args_total: int
    forbid_ok: bool
    forbid_args_ok: bool
    clarify_matched: int
    clarify_total: int
    keyword_matched: int
    keyword_total: int
    token_f1: float | None
    budget_ok: bool
    problems: list[str] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return not self.problems


@dataclass
class ScenarioScore:
    scenario_id: str
    passed: bool
    turns: list[TurnScore]
    steps: int
    latency_ms: float
    problems: list[str] = field(default_factory=list)


def score_turn(expect: Expectation, turn: TurnRecord) -> TurnScore:
    observed = tuple(call.name for call in turn.tool_calls)
    problems: list[str] = []

    tools_ok = observed == tuple(expect.tools)
    if not tools_ok:
        problems.append(f"tools {list(observed)} != expected {list(expect.tools)}")

    args_matched = args_total = 0
    for tool_name, gold_fields in expect.args.items():
        actual = next((c.args for c in turn.tool_calls if c.name == tool_name), None)
        for field_name, gold_value in gold_fields.items():
            args_total += 1
            got = actual.get(field_name) if actual is not None else None
            if actual is None:
                problems.append(f"{tool_name}.{field_name}: tool was never called")
            elif got != gold_value:
                problems.append(f"{tool_name}.{field_name}: {got!r} != {gold_value!r}")
            else:
                args_matched += 1

    forbid_ok = True
    for tool_name in expect.forbid:
        if tool_name in observed:
            forbid_ok = False
            problems.append(f"forbidden tool called: {tool_name}")

    forbid_args_ok = True
    for tool_name, banned_fields in expect.forbid_args.items():
        for call in turn.tool_calls:
            if call.name != tool_name:
                continue
            for field_name in banned_fields:
                if field_name in call.args:
                    forbid_args_ok = False
                    problems.append(f"invented argument {tool_name}.{field_name}")

    text = turn.final_text
    clarify_matched, clarify_total = keyword_counts(text, expect.clarify)
    if clarify_matched != clarify_total:
        problems.append(f"clarification missing {sorted(set(expect.clarify))} in {text!r}")
    keyword_matched, keyword_total = keyword_counts(text, expect.keywords)
    if keyword_matched != keyword_total:
        problems.append(f"missing answer keywords {sorted(set(expect.keywords))} in {text!r}")

    f1 = None
    if expect.gold_answer:
        # Graded, not gated: gold_answer feeds the token_f1 metric. Pass/fail
        # stays on structure and keywords so a rephrased-but-correct answer is
        # still a pass while a wrong tool call never is.
        f1 = token_f1(text, expect.gold_answer)

    budget_ok = not turn.budget_exceeded
    if not budget_ok:
        problems.append(f"exceeded step budget ({len(turn.steps)} steps)")

    return TurnScore(
        observed_tools=observed,
        expected_tools=tuple(expect.tools),
        tools_ok=tools_ok,
        args_matched=args_matched,
        args_total=args_total,
        forbid_ok=forbid_ok,
        forbid_args_ok=forbid_args_ok,
        clarify_matched=clarify_matched,
        clarify_total=clarify_total,
        keyword_matched=keyword_matched,
        keyword_total=keyword_total,
        token_f1=f1,
        budget_ok=budget_ok,
        problems=problems,
    )


def score_scenario(scenario: Scenario, conversation: Conversation) -> ScenarioScore:
    scores = [
        score_turn(turn.expect, record)
        for turn, record in zip(scenario.turns, conversation.turns)
    ]
    problems = [f"{scenario.id} turn {i}: {p}" for i, s in enumerate(scores) for p in s.problems]
    steps = sum(len(record.steps) for record in conversation.turns)
    return ScenarioScore(
        scenario_id=scenario.id,
        passed=not problems,
        turns=scores,
        steps=steps,
        latency_ms=sum(record.latency_ms for record in conversation.turns),
        problems=problems,
    )


def aggregate(scores: Sequence[ScenarioScore]) -> dict[str, float]:
    turns = [turn for score in scores for turn in score.turns]
    f1s = [turn.token_f1 for turn in turns if turn.token_f1 is not None]
    return {
        "tool_accuracy": round(ratio(sum(t.tools_ok for t in turns), len(turns)), 4),
        "arg_field_exact": round(
            ratio(sum(t.args_matched for t in turns), sum(t.args_total for t in turns)), 4
        ),
        "token_f1": round(ratio(sum(f1s), len(f1s)), 4),
        "keyword_recall": round(
            ratio(sum(t.keyword_matched for t in turns), sum(t.keyword_total for t in turns)), 4
        ),
        "clarification_accuracy": round(
            ratio(sum(t.clarify_matched for t in turns), sum(t.clarify_total for t in turns)), 4
        ),
        "scenario_pass_rate": round(
            ratio(sum(1 for s in scores if s.passed), len(scores)), 4
        ),
    }


def percentile(values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile: values[ceil(q*n) - 1] on the sorted list."""
    if not values:
        return 0.0
    ordered = sorted(values)
    index = max(1, math.ceil(q * len(ordered))) - 1
    return float(ordered[min(index, len(ordered) - 1)])


def latency_stats(values: Iterable[float]) -> dict[str, float]:
    collected = list(values)
    if not collected:
        return {"steps": 0, "mean_ms": 0.0, "p95_ms": 0.0}
    return {
        "steps": len(collected),
        "mean_ms": round(sum(collected) / len(collected), 2),
        "p95_ms": round(percentile(collected, 0.95), 2),
    }
