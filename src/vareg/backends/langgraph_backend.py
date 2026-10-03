"""LangGraph backend: the same policy and tools, with LangGraph owning the loop.

What is exercised when ``langgraph`` is installed:

* ``create_react_agent`` (langgraph.prebuilt) compiling a graph of
  ``__start__ -> agent -> tools -> agent -> __end__``;
* ``ToolNode`` executing this repo's tool functions and returning
  ``ToolMessage`` payloads;
* ``recursion_limit`` as the step budget — a policy that never stops raising
  ``GraphRecursionError``, which this backend reports as a blown budget;
* message conversion both ways (our ``system/user/assistant/tool`` dicts to
  LangChain messages and back), so the harness can score the trajectory.

What is *not* exercised: no language model. The "model" node is a
``BaseChatModel`` subclass wrapping the deterministic mock policy, because a
live LLM would make the regression suite non-reproducible. ToolNode also builds
its own pydantic schema from each function's signature — a second, independent
validation layer over the registry's.

Requires ``langgraph`` (and therefore ``langchain-core``). Without it this
module still imports, and constructing the backend raises an actionable error.
"""

from __future__ import annotations

import json
import time
from typing import Any, Sequence

from ..prompts import PromptSpec
from ..registry import Tool, ToolContext, ToolRegistry
from .base import Backend, BackendError, StepRecord, ToolCallResult, TurnOutcome
from .mock import MockPolicy

_INSTALL_HINT = (
    "the 'langgraph' package is required for --backend langgraph: "
    "python3 -m venv .venv && .venv/bin/pip install langgraph "
    "(system pip is PEP-668 locked, so install into the venv and run with .venv/bin/python)"
)

try:  # pragma: no cover - exercised by the langgraph CI job
    from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
    from langchain_core.outputs import ChatGeneration, ChatResult
    from langgraph.errors import GraphRecursionError
    from langgraph.prebuilt import create_react_agent
except ImportError as exc:  # pragma: no cover - environment dependent
    AIMessage = HumanMessage = SystemMessage = ToolMessage = None  # type: ignore[assignment]
    ChatGeneration = ChatResult = GraphRecursionError = create_react_agent = None  # type: ignore[assignment]
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


class Toolset:
    """The registry's tools as bound methods with real signatures.

    LangChain derives a tool schema from each function's signature and takes
    its description from the docstring, so the signatures below are not
    decorative: they are what the graph advertises to the model. The docstrings
    mirror ``registry.TOOLS[*].description`` (a test asserts they match) and the
    body re-validates through the registry before touching any state, so the
    auth guard and the slot constraints hold on this path too.
    """

    def __init__(self, registry: ToolRegistry, ctx: ToolContext) -> None:
        self.registry = registry
        self.ctx = ctx

    def _run(self, name: str, args: dict[str, Any]) -> str:
        result = self.registry.execute(name, args, self.ctx)
        return json.dumps(result, ensure_ascii=False)

    def methods_for(self, names: Sequence[str]) -> list[Any]:
        return [getattr(self, name) for name in names]

    def book_appointment(self, service: str, date: str, time: str, party_size: int, phone: str) -> str:
        """Book a service-desk appointment once every slot is known."""
        return self._run(
            "book_appointment",
            {"service": service, "date": date, "time": time, "party_size": party_size, "phone": phone},
        )

    def cancel_booking(self, booking_id: str, reason: str) -> str:
        """Cancel an existing booking by its BK- reference."""
        return self._run("cancel_booking", {"booking_id": booking_id, "reason": reason})

    def verify_identity(self, phone: str) -> str:
        """Verify the caller by the mobile number registered with their account."""
        return self._run("verify_identity", {"phone": phone})

    def lookup_account(self, account_id: str) -> str:
        """Read the registered holder of an account. Requires verify_identity first."""
        return self._run("lookup_account", {"account_id": account_id})

    def get_balance(self, account_id: str) -> str:
        """Read the current balance of an account. Requires verify_identity first."""
        return self._run("get_balance", {"account_id": account_id})

    def transfer_to_human(self, reason: str, summary: str) -> str:
        """Hand the conversation to a human agent with a reason and a summary."""
        return self._run("transfer_to_human", {"reason": reason, "summary": summary})


def _build_policy_model(policy: MockPolicy, ctx: ToolContext, tools: Sequence[Tool]):
    """Define the BaseChatModel stand-in, once imports are known to exist.

    `bind_tools` is implemented here because `BaseChatModel.bind_tools` raises
    NotImplementedError and `create_react_agent` calls it unconditionally.
    """

    from langchain_core.language_models.chat_models import BaseChatModel

    class PolicyModel(BaseChatModel):
        """Deterministic model node: mock policy in, AIMessage out."""

        spec: Any = None
        registry: Any = None
        ctx: Any = None
        tools: Any = None
        policy: Any = None
        latencies: list[float] = []

        @property
        def _llm_type(self) -> str:
            return "vareg-mock-policy"

        def bind_tools(self, tools: Sequence[Any], **kwargs: Any):  # noqa: D102 - required by create_react_agent
            return self.bind(tools=list(tools), **kwargs)

        def _generate(self, messages, stop=None, run_manager=None, **kwargs):
            started = time.perf_counter()
            decision = self.policy.decide(
                [_message_to_dict(m) for m in messages], self.tools, self.ctx
            )
            self.latencies.append((time.perf_counter() - started) * 1000.0)
            tool_calls = [
                {"id": c.id, "name": c.name, "args": c.args, "type": "tool_call"}
                for c in decision.tool_calls
            ]
            message = AIMessage(content=decision.content, tool_calls=tool_calls)
            return ChatResult(generations=[ChatGeneration(message=message)])

    # pydantic treats a mutable class attribute as a shared default; give each
    # instance its own list.
    return PolicyModel(
        spec=policy.spec, registry=policy.registry, ctx=ctx, tools=list(tools),
        policy=policy, latencies=[],
    )


def _message_to_dict(message) -> dict[str, Any]:
    if isinstance(message, ToolMessage):
        return {
            "role": "tool",
            "content": message.content if isinstance(message.content, str) else str(message.content),
            "tool_call_id": message.tool_call_id,
            "name": message.name or "",
        }
    if isinstance(message, AIMessage):
        out: dict[str, Any] = {"role": "assistant", "content": message.content or ""}
        if message.tool_calls:
            out["tool_calls"] = [
                {"id": c["id"], "name": c["name"], "args": c["args"]} for c in message.tool_calls
            ]
        return out
    if isinstance(message, SystemMessage):
        return {"role": "system", "content": message.content or ""}
    return {"role": "user", "content": message.content or ""}


def _to_langchain(messages: Sequence[dict[str, Any]]):
    converted = []
    for message in messages:
        role, content = message.get("role"), message.get("content") or ""
        if role == "system":
            converted.append(SystemMessage(content=content))
        elif role == "assistant":
            tool_calls = [
                {"id": c["id"], "name": c["name"], "args": c["args"], "type": "tool_call"}
                for c in message.get("tool_calls") or []
            ]
            converted.append(AIMessage(content=content, tool_calls=tool_calls))
        elif role == "tool":
            converted.append(
                ToolMessage(
                    content=content,
                    tool_call_id=message.get("tool_call_id", ""),
                    name=message.get("name", ""),
                )
            )
        else:
            converted.append(HumanMessage(content=content))
    return converted


class LangGraphBackend(Backend):
    name = "langgraph"

    def __init__(self, spec: PromptSpec, registry: ToolRegistry) -> None:
        super().__init__(spec, registry)
        if _IMPORT_ERROR is not None:
            raise BackendError(_INSTALL_HINT) from _IMPORT_ERROR
        self.policy = MockPolicy(spec, registry)

    def decide(self, messages, tools, ctx):  # pragma: no cover - graph owns the loop
        raise BackendError("langgraph backend runs whole turns, not single decisions")

    def run_turn(
        self,
        messages: list[dict[str, Any]],
        tools: Sequence[Tool],
        ctx: ToolContext,
        max_steps: int,
    ) -> TurnOutcome:
        model = _build_policy_model(self.policy, ctx, tools)
        graph = create_react_agent(model, Toolset(self.registry, ctx).methods_for([t.name for t in tools]))
        before = len(messages)
        try:
            state = graph.invoke(
                {"messages": _to_langchain(messages)},
                # Two graph nodes per decision (agent, tools) plus start/end.
                config={"recursion_limit": 2 * max_steps + 3},
            )
        except GraphRecursionError:
            # The policy never produced a final answer inside the budget.
            return TurnOutcome(final_text="", steps=[], budget_exceeded=True)

        steps, call_args, call_names = [], {}, {}
        final_text = ""
        current: StepRecord | None = None
        latencies = list(model.latencies)
        latency_index = 0
        for message in state["messages"][before:]:
            if isinstance(message, AIMessage):
                latency = latencies[latency_index] if latency_index < len(latencies) else 0.0
                latency_index += 1
                current = StepRecord(
                    index=len(steps), latency_ms=latency, message=_message_to_dict(message)
                )
                steps.append(current)
                for call in message.tool_calls:
                    call_args[call["id"]] = call["args"]
                    call_names[call["id"]] = call["name"]
                if not message.tool_calls:
                    final_text = message.content or ""
            elif isinstance(message, ToolMessage) and current is not None:
                payload = _payload(message.content)
                name = message.name or call_names.get(message.tool_call_id, "")
                current.results.append(
                    ToolCallResult(
                        name=name,
                        args=dict(call_args.get(message.tool_call_id, {})),
                        ok=bool(payload.get("ok")),
                        payload=payload,
                    )
                )
            messages.append(_message_to_dict(message))
        return TurnOutcome(final_text=final_text, steps=steps, budget_exceeded=False)


def _payload(content: Any) -> dict[str, Any]:
    if isinstance(content, dict):
        return content
    try:
        parsed = json.loads(str(content))
    except (TypeError, ValueError):
        return {"ok": False, "error": f"tool_error: {content}"}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": str(parsed)}

