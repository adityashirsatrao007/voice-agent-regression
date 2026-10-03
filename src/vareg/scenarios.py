"""Scenario matrix loading and schema validation.

The matrix is the spec: each scenario carries the caller's turns and the gold
expectation for every turn (tool sequence, pinned argument fields, forbidden
tools, forbidden argument fields, slots that must be asked for, answer keywords
and, where a full answer is checkable, a reference answer for token-F1).

Field-level pinning is deliberate. Gold states only the fields the caller's own
words fully determine — a free-text field like `summary` is validated by the
schema but never asserted, because scoring prose against prose would just
measure wording agreement with ourselves.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

SCENARIO_DIR = Path(__file__).resolve().parents[2] / "scenarios"

ALLOWED_EXPECT_KEYS = {
    "tools",
    "args",
    "forbid",
    "forbid_args",
    "clarify",
    "keywords",
    "gold_answer",
}


class ScenarioError(ValueError):
    """A scenario file that does not satisfy the matrix schema."""


@dataclass(frozen=True)
class Expectation:
    tools: tuple[str, ...]
    args: dict[str, dict[str, Any]] = field(default_factory=dict)
    forbid: tuple[str, ...] = ()
    forbid_args: dict[str, tuple[str, ...]] = field(default_factory=dict)
    clarify: tuple[str, ...] = ()
    keywords: tuple[str, ...] = ()
    gold_answer: str = ""


@dataclass(frozen=True)
class Turn:
    user: str
    expect: Expectation


@dataclass(frozen=True)
class Scenario:
    id: str
    title: str
    tags: tuple[str, ...]
    max_steps: int
    turns: tuple[Turn, ...]
    source: Path | None = None


def _expectation(raw: dict[str, Any], where: str) -> Expectation:
    if not isinstance(raw, dict):
        raise ScenarioError(f"{where}: 'expect' must be an object")
    unknown = set(raw) - ALLOWED_EXPECT_KEYS
    if unknown:
        raise ScenarioError(f"{where}: unknown expectation keys: {sorted(unknown)}")
    if "tools" not in raw:
        raise ScenarioError(f"{where}: 'expect.tools' is required (may be an empty list)")
    tools = tuple(raw["tools"])
    args = {k: dict(v) for k, v in (raw.get("args") or {}).items()}
    for tool_name, fields in args.items():
        if tool_name not in tools:
            raise ScenarioError(
                f"{where}: expect.args pins {tool_name!r} but expect.tools never calls it"
            )
        if not isinstance(fields, dict):
            raise ScenarioError(f"{where}: expect.args[{tool_name!r}] must be an object")
    forbid = tuple(raw.get("forbid") or ())
    clash = set(forbid) & set(tools)
    if clash:
        raise ScenarioError(f"{where}: {sorted(clash)} is both expected and forbidden")
    # forbid_args is checked against the *observed* calls, so it is legal for a
    # tool the gold expects never to be called: that is exactly the
    # hallucinated-argument case (booking_missing_phone).
    forbid_args = {k: tuple(v) for k, v in (raw.get("forbid_args") or {}).items()}
    return Expectation(
        tools=tools,
        args=args,
        forbid=forbid,
        forbid_args=forbid_args,
        clarify=tuple(raw.get("clarify") or ()),
        keywords=tuple(raw.get("keywords") or ()),
        gold_answer=str(raw.get("gold_answer") or ""),
    )


def parse_scenario(raw: dict[str, Any], source: Path | None = None) -> Scenario:
    where = str(source) if source else "<inline>"
    for key in ("id", "title", "tags", "max_steps", "turns"):
        if key not in raw:
            raise ScenarioError(f"{where}: missing required key {key!r}")
    if source is not None and raw["id"] != source.stem:
        raise ScenarioError(f"{where}: id {raw['id']!r} does not match file name {source.stem!r}")
    if not isinstance(raw["max_steps"], int) or raw["max_steps"] < 1:
        raise ScenarioError(f"{where}: max_steps must be a positive integer")
    turns_raw = raw["turns"]
    if not isinstance(turns_raw, list) or not turns_raw:
        raise ScenarioError(f"{where}: turns must be a non-empty list")
    turns: list[Turn] = []
    for index, turn in enumerate(turns_raw):
        loc = f"{where} turn {index}"
        if not isinstance(turn, dict) or "user" not in turn:
            raise ScenarioError(f"{loc}: needs a 'user' string")
        if not str(turn["user"]).strip():
            raise ScenarioError(f"{loc}: 'user' is empty")
        turns.append(Turn(user=str(turn["user"]), expect=_expectation(turn.get("expect"), loc)))
    tags = tuple(raw["tags"])
    if not tags:
        raise ScenarioError(f"{where}: tags must not be empty")
    return Scenario(
        id=str(raw["id"]),
        title=str(raw["title"]),
        tags=tags,
        max_steps=int(raw["max_steps"]),
        turns=tuple(turns),
        source=source,
    )


def load_scenario(path: Path) -> Scenario:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ScenarioError(f"{path}: not valid JSON: {exc}") from exc
    return parse_scenario(raw, source=path)


def load_matrix(scenario_dir: Path = SCENARIO_DIR) -> list[Scenario]:
    if not scenario_dir.is_dir():
        raise ScenarioError(f"scenario directory not found: {scenario_dir}")
    paths = sorted(scenario_dir.glob("*.json"))
    if not paths:
        raise ScenarioError(f"no scenarios in {scenario_dir}")
    # Duplicate ids cannot reach here: parse_scenario already requires id == file
    # stem, and file names are unique within a directory.
    return [load_scenario(p) for p in paths]
