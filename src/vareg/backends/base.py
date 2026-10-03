"""Backend contract and the shared turn loop.

A backend owns one *turn*: give it the message list, the tools the prompt
licenses and the conversation context, and it returns the messages it appended
plus a step trace. `run_turn` below is the function-calling loop every backend
except the LangGraph one reuses — system / user / assistant(tool_calls) /
tool(result) roles, arguments validated before execution, and a hard step
budget so an agent that loops forever fails the scenario instead of hanging the
suite.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Iterable, Sequence

from ..prompts import PromptSpec
from ..registry import Tool, ToolContext

if TYPE_CHECKING:  # pragma: no cover - typing only
    from ..registry import ToolRegistry


class BackendError(RuntimeError):
    """A backend that cannot start: missing package, missing key, bad config."""


@dataclass(frozen=True)
class ToolCall:
    name: str
    args: dict[str, Any]
    id: str


@dataclass(frozen=True)
class Decision:
    """What the model wants to do next: say something, or call tools."""

    content: str = ""
    tool_calls: tuple[ToolCall, ...] = ()


@dataclass
class ToolCallResult:
    name: str
    args: dict[str, Any]
    ok: bool
    payload: dict[str, Any]


@dataclass
class StepRecord:
    index: int
    latency_ms: float
    message: dict[str, Any]
    results: list[ToolCallResult] = field(default_factory=list)


@dataclass
class TurnOutcome:
    final_text: str
    steps: list[StepRecord]
    budget_exceeded: bool

    @property
    def tool_calls(self) -> list[ToolCallResult]:
        return [r for step in self.steps for r in step.results]


def assistant_message(decision: Decision) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": decision.content}
    if decision.tool_calls:
        message["tool_calls"] = [
            {"id": c.id, "name": c.name, "args": c.args} for c in decision.tool_calls
        ]
    return message


class Backend:
    name = "base"

    def __init__(self, spec: PromptSpec, registry: "ToolRegistry") -> None:
        self.spec = spec
        self.registry = registry
        # Token/cost accounting is only ever populated by a backend that really
        # called a metered API. mock and langgraph leave it None so the report
        # can say "nothing billed" instead of printing a made-up number.
        self.usage: dict[str, int] | None = None

    def decide(self, messages: list[dict[str, Any]], tools: Sequence[Tool], ctx: ToolContext) -> Decision:
        raise NotImplementedError

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Tool],
        ctx: ToolContext,
        max_steps: int,
    ) -> TurnOutcome:
        """Append one turn to *messages* and trace every decision in it."""
        steps: list[StepRecord] = []
        final_text = ""
        budget_exceeded = False
        for index in range(max_steps):
            started = time.perf_counter()
            decision = self.decide(messages, tools, ctx)
            latency_ms = (time.perf_counter() - started) * 1000.0
            message = assistant_message(decision)
            messages.append(message)
            step = StepRecord(index=index, latency_ms=latency_ms, message=message)
            steps.append(step)
            if not decision.tool_calls:
                final_text = decision.content
                break
            for call in decision.tool_calls:
                payload = self.execute(call, tools, ctx)
                step.results.append(
                    ToolCallResult(name=call.name, args=dict(call.args), ok=bool(payload.get("ok")), payload=payload)
                )
                messages.append(
                    {
                        "role": "tool",
                        "name": call.name,
                        "tool_call_id": call.id,
                        "content": json.dumps(payload, ensure_ascii=False),
                    }
                )
        else:
            # Ran out of budget without a final message: the scenario should fail.
            budget_exceeded = True
        return TurnOutcome(final_text=final_text, steps=steps, budget_exceeded=budget_exceeded)

    def execute(self, call: ToolCall, tools: Sequence[Tool], ctx: ToolContext) -> dict[str, Any]:
        """Run a call, but only for tools the prompt actually licensed.

        Order matters: an unregistered tool is reported as such (that is a
        backend bug), while a registered tool the prompt did not name gets
        ``not_licensed`` — the enforcement half of the prompt's ``tools:`` list,
        so a policy reaching for ``cancel_booking`` under v1 cannot succeed.
        """
        if self.registry.get(call.name) is None:
            return {
                "ok": False,
                "code": "unknown_tool",
                "error": f"unknown_tool: {call.name}",
            }
        if call.name not in {tool.name for tool in tools}:
            return {
                "ok": False,
                "code": "not_licensed",
                "error": f"not_licensed: {call.name} is not in the prompt's tool list",
            }
        return self.registry.execute(call.name, call.args, ctx)


def licensed(tools: Iterable[Tool]) -> dict[str, Tool]:
    return {tool.name: tool for tool in tools}
