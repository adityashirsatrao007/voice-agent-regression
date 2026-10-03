"""The regression gate: compare a candidate run against a committed baseline.

This is the command CI fails on. Rules, kept deliberately boring:

* every gated metric must exist in both runs and be numeric;
* a candidate value below ``baseline - threshold`` is a regression;
* the scenario id set must match, so "passed fewer scenarios by not running
  them" cannot look like a pass;
* wall-clock latency is reported but never gated — a busy CI runner is not a
  quality signal, and gating on it would make the suite flaky by construction.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

from .metrics import GATED_METRICS


class RegressionError(ValueError):
    """Baseline/candidate files that cannot be compared."""


@dataclass(frozen=True)
class Finding:
    metric: str
    baseline: float | None
    candidate: float | None
    delta: float | None
    status: str  # "ok" | "drop" | "missing" | "changed"

    @property
    def is_regression(self) -> bool:
        return self.status in {"drop", "missing", "changed"}


def load_run(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise RegressionError(f"run file not found: {path}")
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RegressionError(f"{path} is not valid JSON: {exc}") from exc
    if not isinstance(data.get("metrics"), dict):
        raise RegressionError(f"{path} has no 'metrics' object - is it a report.json?")
    return data


def _as_float(value: Any) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return float(value)


def compare(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    threshold: float = 0.0,
) -> list[Finding]:
    findings: list[Finding] = []
    base_metrics = baseline.get("metrics") or {}
    cand_metrics = candidate.get("metrics") or {}
    for metric in GATED_METRICS:
        before = _as_float(base_metrics.get(metric))
        after = _as_float(cand_metrics.get(metric))
        if before is None or after is None:
            findings.append(Finding(metric, before, after, None, "missing"))
            continue
        delta = round(after - before, 6)
        status = "drop" if delta < -threshold else "ok"
        findings.append(Finding(metric, before, after, delta, status))

    base_ids = {row.get("id") for row in baseline.get("scenarios") or []}
    cand_ids = {row.get("id") for row in candidate.get("scenarios") or []}
    if base_ids != cand_ids:
        missing = sorted(str(i) for i in base_ids - cand_ids)
        extra = sorted(str(i) for i in cand_ids - base_ids)
        detail = []
        if missing:
            detail.append(f"missing {missing}")
        if extra:
            detail.append(f"unexpected {extra}")
        findings.append(Finding("scenario_set: " + "; ".join(detail), None, None, None, "changed"))
    return findings


def render(findings: Sequence[Finding], threshold: float, verdict: str) -> str:
    lines = [
        f"## regression gate (threshold {threshold:g} on every gated metric)",
        "",
        "| metric | baseline | candidate | delta | status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for finding in findings:
        before = "-" if finding.baseline is None else f"{finding.baseline:.4f}"
        after = "-" if finding.candidate is None else f"{finding.candidate:.4f}"
        delta = "-" if finding.delta is None else f"{finding.delta:+.4f}"
        lines.append(f"| {finding.metric} | {before} | {after} | {delta} | {finding.status} |")
    lines += ["", verdict]
    return "\n".join(lines)


def verdict(findings: Sequence[Finding]) -> str:
    regressions = [f for f in findings if f.is_regression]
    if not regressions:
        return "GATE PASS - no gated metric regressed"
    names = ", ".join(f.metric for f in regressions)
    return f"GATE FAIL - regressed: {names}"


def gate(baseline: dict[str, Any], candidate: dict[str, Any], threshold: float = 0.0) -> int:
    """0 when the candidate is at least as good as the baseline, else 1."""
    findings = compare(baseline, candidate, threshold)
    return 1 if any(f.is_regression for f in findings) else 0
