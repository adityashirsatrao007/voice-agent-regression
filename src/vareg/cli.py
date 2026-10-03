"""Command line: ``python3 -m vareg {run,regress,list-scenarios}``.

Exit codes are the contract CI depends on:

0  success (and, for ``regress``, the candidate is not worse than the baseline)
1  ``regress`` found a regression
2  the harness could not run at all (bad scenario, unknown prompt, missing backend)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Sequence

from .backends import BACKEND_NAMES, make_backend
from .backends.base import BackendError
from .prompts import PromptError, load_spec
from .regression import RegressionError, compare, load_run, render, verdict
from .report import render_report, run_matrix, write_report
from .registry import DEFAULT_REGISTRY
from .scenarios import SCENARIO_DIR, ScenarioError, load_matrix

BASELINE_PATH = Path("baseline.json")
DEFAULT_CANDIDATE_DIR = Path("results/candidate")


def _add_run_options(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--backend", default="mock", choices=BACKEND_NAMES)
    parser.add_argument("--prompts", default=None, help="prompt version (default: manifest default)")
    parser.add_argument("--scenarios", default=None, help="comma-separated scenario ids")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vareg",
        description="Voice-agent regression harness: run the scenario matrix, gate a candidate run.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the full matrix and write the report")
    _add_run_options(run)
    run.add_argument("--out", default="results", help="output directory (default: results)")
    run.add_argument(
        "--write-baseline",
        action="store_true",
        help=f"also write this run to {BASELINE_PATH}",
    )
    run.add_argument("--quiet", action="store_true", help="suppress the stdout report")

    reg = sub.add_parser("regress", help="gate a candidate run against the baseline")
    reg.add_argument("--baseline", default=str(BASELINE_PATH), help="baseline report json")
    reg.add_argument(
        "--candidate",
        default=None,
        help="candidate report json; omit to run the matrix fresh first",
    )
    reg.add_argument("--threshold", type=float, default=0.0, help="allowed drop per metric")
    reg.add_argument("--backend", default="mock", choices=BACKEND_NAMES)
    reg.add_argument("--prompts", default=None, help="prompt version (default: manifest default)")
    reg.add_argument("--scenarios", default=None, help="comma-separated scenario ids")
    reg.add_argument("--out", default=str(DEFAULT_CANDIDATE_DIR), help="where a fresh run is saved")
    reg.add_argument(
        "--update-baseline",
        action="store_true",
        help="replace the baseline with the candidate after comparing",
    )

    listing = sub.add_parser("list-scenarios", help="print the scenario matrix")
    listing.add_argument("--dir", default=str(SCENARIO_DIR), help="scenario directory")

    return parser


def _select_scenarios(selected: str | None):
    scenarios = load_matrix()
    if not selected:
        return scenarios
    wanted = [name.strip() for name in selected.split(",") if name.strip()]
    by_id = {scenario.id: scenario for scenario in scenarios}
    unknown = [name for name in wanted if name not in by_id]
    if unknown:
        raise ScenarioError(
            f"unknown scenario id(s): {', '.join(unknown)}; try `python3 -m vareg list-scenarios`"
        )
    return [by_id[name] for name in wanted]


def _run(args: argparse.Namespace):
    spec = load_spec(args.prompts)
    scenarios = _select_scenarios(getattr(args, "scenarios", None))
    backend = make_backend(args.backend, spec, DEFAULT_REGISTRY)
    return run_matrix(scenarios, backend, DEFAULT_REGISTRY, spec)


def _cmd_run(args: argparse.Namespace) -> int:
    report, _ = _run(args)
    if not args.quiet:
        print(render_report(report))
    out_dir = Path(args.out)
    for path in write_report(report, out_dir):
        print(f"wrote {path}")
    if args.write_baseline:
        BASELINE_PATH.write_text(
            json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"wrote {BASELINE_PATH}")
    return 0


def _cmd_regress(args: argparse.Namespace) -> int:
    if args.candidate:
        candidate = load_run(Path(args.candidate))
        source = args.candidate
    else:
        report, _ = _run(args)
        candidate = report
        for path in write_report(report, Path(args.out)):
            print(f"wrote {path}")
        source = f"a fresh {report['backend']} run (prompt {report['prompt']['version']})"
    baseline = load_run(Path(args.baseline))
    findings = compare(baseline, candidate, args.threshold)
    print(render(findings, args.threshold, verdict(findings)))
    print(f"baseline: {args.baseline}")
    print(f"candidate: {source}")
    if args.update_baseline:
        Path(args.baseline).write_text(
            json.dumps(candidate, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"updated {args.baseline}")
    return 1 if any(f.is_regression for f in findings) else 0


def _cmd_list_scenarios(args: argparse.Namespace) -> int:
    scenarios = load_matrix(Path(args.dir))
    print(f"## scenario matrix ({len(scenarios)} scenarios)")
    print()
    print("| id | tags | turns | title |")
    print("| --- | --- | ---: | --- |")
    for scenario in scenarios:
        print(
            f"| {scenario.id} | {', '.join(scenario.tags)} | {len(scenario.turns)} | "
            f"{scenario.title} |"
        )
    return 0


COMMANDS = {
    "run": _cmd_run,
    "regress": _cmd_regress,
    "list-scenarios": _cmd_list_scenarios,
}


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, "command", None):
        parser.print_help()
        return 2
    try:
        return COMMANDS[args.command](args)
    except (BackendError, ScenarioError, PromptError, RegressionError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
