"""Run assembly: score a matrix run, emit the report dict and its rendering.

The report dict is what gets written to ``results/report.json`` and what
``regress`` compares against ``baseline.json``, so its shape is the contract
between `run` and `regress`.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

from .agent import Agent, Conversation
from .backends.base import Backend
from .cost import cost_usd
from .metrics import GATED_METRICS, aggregate, latency_stats, score_scenario
from .prompts import PromptSpec
from .registry import ToolRegistry
from .scenarios import Scenario


def usage_block(backend: Backend) -> dict[str, Any]:
    """Token/cost accounting, or an explicit reason why there is none."""
    if backend.usage is None:
        return {
            "model": None,
            "input_tokens": None,
            "output_tokens": None,
            "cost_usd": None,
            "note": "no metered call was made - this backend runs offline, so "
            "there is no token count and no cost to report",
        }
    input_tokens = int(backend.usage.get("input_tokens") or 0)
    output_tokens = int(backend.usage.get("output_tokens") or 0)
    model = getattr(backend, "model", None)
    amount, note = cost_usd(str(model), input_tokens, output_tokens)
    return {
        "model": model,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": amount,
        "note": note,
    }


def run_matrix(
    scenarios: Sequence[Scenario],
    backend: Backend,
    registry: ToolRegistry,
    spec: PromptSpec,
) -> tuple[dict[str, Any], list[Conversation]]:
    conversations: list[Conversation] = []
    scores = []
    rows = []
    for scenario in scenarios:
        conversation = Agent(backend, registry, spec).run(scenario)
        conversations.append(conversation)
        score = score_scenario(scenario, conversation)
        scores.append(score)
        rows.append(
            {
                "id": scenario.id,
                "tags": list(scenario.tags),
                "turns": len(scenario.turns),
                "steps": score.steps,
                "ms": round(score.latency_ms, 2),
                "passed": score.passed,
                "tools": [name for t in score.turns for name in t.observed_tools],
                "problems": score.problems,
            }
        )
    all_steps = [step.latency_ms for conv in conversations for step in conv.steps]
    report = {
        "backend": backend.name,
        "prompt": {"version": spec.version, "date": spec.date, "change": spec.change},
        "scenarios": rows,
        "metrics": aggregate(scores),
        "latency": latency_stats(all_steps),
        "usage": usage_block(backend),
    }
    return report, conversations


def _status(passed: bool) -> str:
    return "PASS" if passed else "FAIL"


def render_report(report: dict[str, Any]) -> str:
    lines = [
        f"## voice-agent regression run - backend {report['backend']}, "
        f"prompt {report['prompt']['version']}",
        "",
        "| scenario | result | turns | tools | ms |",
        "| --- | --- | ---: | --- | ---: |",
    ]
    for row in report["scenarios"]:
        tools = ", ".join(row["tools"]) if row["tools"] else "-"
        lines.append(
            f"| {row['id']} | {_status(row['passed'])} | {row['turns']} | {tools} | {row['ms']:.2f} |"
        )
    metrics = report["metrics"]
    lines += ["", "### metrics", "", "| metric | value |", "| --- | ---: |"]
    for name in GATED_METRICS:
        lines.append(f"| {name} | {metrics[name]:.4f} |")
    latency = report["latency"]
    lines += [
        "",
        f"latency: {latency['mean_ms']:.2f} ms mean / {latency['p95_ms']:.2f} ms p95 "
        f"over {latency['steps']} model decisions "
        "(wall clock on this machine, excluded from the gate)",
    ]
    usage = report["usage"]
    if usage["cost_usd"] is None:
        cost_line = "not applicable"
    else:
        cost_line = f"${usage['cost_usd']:.6f}"
    lines.append(
        f"usage: input_tokens={usage['input_tokens']} output_tokens={usage['output_tokens']} "
        f"cost={cost_line} - {usage['note']}"
    )
    failures = [row for row in report["scenarios"] if not row["passed"]]
    if failures:
        lines += ["", "### failures", ""]
        for row in failures:
            for problem in row["problems"]:
                lines.append(f"- {row['id']}: {problem}")
    return "\n".join(lines)


def write_report(report: dict[str, Any], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    json_path = out_dir / "report.json"
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    md_path = out_dir / "report.md"
    md_path.write_text(render_report(report) + "\n", encoding="utf-8")
    return [json_path, md_path]
