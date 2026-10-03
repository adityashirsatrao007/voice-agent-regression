"""Conversation runner: assembles messages, records turns, keeps the context.

The harness scores *trajectories*, not prose in isolation. Each turn keeps the
backend's full step trace (decision latency, tool calls, tool payloads) so a
scenario can assert an ordered tool sequence and the exact arguments that were
parsed out of the caller's words.

Message roles follow the function-calling convention the live backend speaks:
system, user, assistant (optionally with tool_calls), tool (one payload per
call, JSON-encoded so mock and live runs carry the same shape).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from .backends.base import Backend, TurnOutcome
from .prompts import PromptSpec
from .registry import ToolContext, ToolRegistry
from .scenarios import Scenario


@dataclass
class TurnRecord:
    index: int
    user: str
    outcome: TurnOutcome

    @property
    def steps(self):
        return self.outcome.steps

    @property
    def tool_calls(self):
        return self.outcome.tool_calls

    @property
    def final_text(self) -> str:
        return self.outcome.final_text

    @property
    def budget_exceeded(self) -> bool:
        return self.outcome.budget_exceeded

    @property
    def latency_ms(self) -> float:
        return sum(step.latency_ms for step in self.outcome.steps)


@dataclass
class Conversation:
    scenario_id: str
    backend: str
    prompt_version: str
    turns: list[TurnRecord] = field(default_factory=list)
    messages: list[dict[str, Any]] = field(default_factory=list)

    @property
    def steps(self):
        return [step for turn in self.turns for step in turn.steps]


class Agent:
    """Runs one scenario against one backend."""

    def __init__(self, backend: Backend, registry: ToolRegistry, spec: PromptSpec) -> None:
        self.backend = backend
        self.registry = registry
        self.spec = spec
        self.tools = registry.allowed(spec.tools)

    def run(self, scenario: Scenario) -> Conversation:
        ctx = ToolContext()
        messages: list[dict[str, Any]] = [{"role": "system", "content": self.spec.body}]
        conversation = Conversation(
            scenario_id=scenario.id,
            backend=self.backend.name,
            prompt_version=self.spec.version,
            messages=messages,
        )
        for index, turn in enumerate(scenario.turns):
            messages.append({"role": "user", "content": turn.user})
            outcome = self.backend.run_turn(messages, self.tools, ctx, scenario.max_steps)
            conversation.turns.append(TurnRecord(index=index, user=turn.user, outcome=outcome))
        return conversation


def tool_trace(conversation: Conversation) -> list[dict[str, Any]]:
    """Flat ``(turn, tool, args, ok)`` view used by the report and tests."""
    trace = []
    for turn in conversation.turns:
        for call in turn.tool_calls:
            trace.append(
                {
                    "turn": turn.index,
                    "tool": call.name,
                    "args": call.args,
                    "ok": call.ok,
                    "payload": json.dumps(call.payload, ensure_ascii=False),
                }
            )
    return trace
