"""Tool registry: argument validation and the scripted offline backend.

Every tool declares its parameters once, in JSON-schema-ish form. The same
declaration feeds four things: validation before execution, the `tools` array
sent to a live model, the slot constraints the mock policy reads when it decides
whether a caller-supplied value is bookable, and the gold assertions in the
scenario files (which pin field names, not prose).

Handlers are scripted and deterministic: no network, no clock, no randomness.
That is the point — a regression run has to be comparable across machines.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date
from typing import Any, Callable, Iterable

# Booking references increment per conversation so a second booking in the same
# scenario gets a different id (BK-100002), which is what makes context-retention
# assertions possible without inventing a random id generator.
_FIRST_BOOKING_REF = 100000

# Scripted account directory. Values are fixtures, not real customer data.
ACCOUNT_NAMES = {
    "ACC-1234": "A. Sharma",
    "ACC-5678": "R. Iyer",
    "ACC-9999": "M. Nair",
}

PATTERNS = {
    "date": re.compile(r"^\d{4}-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])$"),
    "time": re.compile(r"^([01]\d|2[0-3]):[0-5]\d$"),
    "phone": re.compile(r"^[6-9]\d{9}$"),
    "account_id": re.compile(r"^ACC-\d{4}$"),
    "booking_id": re.compile(r"^BK-\d{6}$"),
}


@dataclass(frozen=True)
class Param:
    """One declared argument of a tool."""

    type: str  # "string" | "integer"
    required: bool = True
    description: str = ""
    enum: tuple[str, ...] | None = None
    pattern: str | None = None
    minimum: int | None = None
    maximum: int | None = None
    # Human phrase used in clarification turns, e.g. for party_size. Falls back
    # to a generic constraint description when absent.
    hint: str = ""

    def describe(self) -> dict[str, Any]:
        out: dict[str, Any] = {"type": self.type, "description": self.description}
        if self.enum is not None:
            out["enum"] = list(self.enum)
        if self.pattern is not None:
            out["pattern"] = self.pattern
        if self.minimum is not None:
            out["minimum"] = self.minimum
        if self.maximum is not None:
            out["maximum"] = self.maximum
        return out


@dataclass(frozen=True)
class Tool:
    name: str
    description: str
    params: dict[str, Param]
    handler: Callable[[dict[str, Any], "ToolContext"], dict[str, Any]]


@dataclass
class ToolContext:
    """Conversation state shared by the policy and the tool handlers.

    Lives for one scenario run. `authenticated` is the auth guard the account
    tools read; `chain` is the policy's deferred work (verify, then read);
    `pending_slots` is how turn two of a booking reuses turn one's arguments.
    """

    authenticated: bool = False
    phone: str | None = None
    account_id: str | None = None
    pending_intent: str | None = None
    pending_slots: dict[str, Any] = field(default_factory=dict)
    chain: list[tuple[str, dict[str, Any]]] = field(default_factory=list)
    last_call: dict[str, Any] = field(default_factory=dict)
    booking_seq: int = 0
    call_seq: int = 0


def _require_auth(ctx: ToolContext) -> dict[str, Any] | None:
    if not ctx.authenticated:
        return {
            "ok": False,
            "code": "auth_required",
            "error": "auth_required: verify the caller's phone number before reading an account",
        }
    return None


def _book(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    slot = date.fromisoformat(args["date"])
    # Scripted backend failure: the desk does not take bookings on Sundays.
    # 2026-10-18 (the date the tool_error_sunday scenario asks for) is a Sunday.
    if slot.weekday() == 6:
        return {
            "ok": False,
            "code": "slot_unavailable",
            "error": "slot_unavailable: bookings are not available on Sunday",
        }
    ctx.booking_seq += 1
    return {"ok": True, "output": f"BK-{_FIRST_BOOKING_REF + ctx.booking_seq:06d}"}


def _cancel(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return {"ok": True, "output": f"{args['booking_id']} cancelled"}


def _verify(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    ctx.authenticated = True
    ctx.phone = args["phone"]
    return {"ok": True, "output": "identity verified"}


def _lookup(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    blocked = _require_auth(ctx)
    if blocked:
        return blocked
    name = ACCOUNT_NAMES.get(args["account_id"])
    if name is None:
        return {
            "ok": False,
            "code": "account_not_found",
            "error": f"account_not_found: no account {args['account_id']}",
        }
    return {"ok": True, "output": name}


def _balance(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    blocked = _require_auth(ctx)
    if blocked:
        return blocked
    account_id = args["account_id"]
    if account_id not in ACCOUNT_NAMES:
        return {
            "ok": False,
            "code": "account_not_found",
            "error": f"account_not_found: no account {account_id}",
        }
    # Deliberately arithmetic instead of hashed: a reviewer can check the
    # balance in a scenario's gold answer by hand (ACC-1234 -> 1234*37+111).
    amount = int(account_id[-4:]) * 37 + 111
    return {"ok": True, "output": f"Rs. {amount:,}"}


def _transfer(args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
    return {"ok": True, "output": "queued for a human agent"}


BOOKING_PARAMS = {
    "service": Param("string", description="Type of appointment", enum=("consultation", "followup", "demo")),
    "date": Param("string", description="ISO date, YYYY-MM-DD", pattern="date", hint="must be a real date in YYYY-MM-DD form"),
    "time": Param("string", description="24h local time, HH:MM", pattern="time", hint="must be a 24-hour time such as 19:00"),
    "party_size": Param("integer", description="Number of people", minimum=1, maximum=12, hint="must be between 1 and 12"),
    "phone": Param("string", description="10-digit Indian mobile number", pattern="phone", hint="must be a 10-digit mobile number starting with 6-9"),
}

TOOLS: tuple[Tool, ...] = (
    Tool(
        name="book_appointment",
        description="Book a service-desk appointment once every slot is known.",
        params=BOOKING_PARAMS,
        handler=_book,
    ),
    Tool(
        name="cancel_booking",
        description="Cancel an existing booking by its BK- reference.",
        params={
            "booking_id": Param("string", description="Booking reference", pattern="booking_id", hint="must look like BK-123456"),
            "reason": Param(
                "string",
                description="Why the caller is cancelling",
                enum=("schedule_conflict", "billing", "technical", "complaint", "other"),
            ),
        },
        handler=_cancel,
    ),
    Tool(
        name="verify_identity",
        description="Verify the caller by the mobile number registered with their account.",
        params={"phone": Param("string", description="10-digit Indian mobile number", pattern="phone", hint="must be a 10-digit mobile number starting with 6-9")},
        handler=_verify,
    ),
    Tool(
        name="lookup_account",
        description="Read the registered holder of an account. Requires verify_identity first.",
        params={"account_id": Param("string", description="Account id", pattern="account_id", hint="must look like ACC-1234")},
        handler=_lookup,
    ),
    Tool(
        name="get_balance",
        description="Read the current balance of an account. Requires verify_identity first.",
        params={"account_id": Param("string", description="Account id", pattern="account_id", hint="must look like ACC-1234")},
        handler=_balance,
    ),
    Tool(
        name="transfer_to_human",
        description="Hand the conversation to a human agent with a reason and a summary.",
        params={
            "reason": Param("string", description="Reason category", enum=("schedule_conflict", "billing", "technical", "complaint", "other")),
            "summary": Param("string", description="Short transcript summary for the human"),
        },
        handler=_transfer,
    ),
)


def _check_value(param: Param, value: Any) -> str | None:
    """Return a human phrase describing why *value* is not acceptable, else None."""
    if param.type == "integer":
        if isinstance(value, bool) or not isinstance(value, int):
            return "must be an integer"
    elif not isinstance(value, str):
        return "must be a string"
    if param.enum is not None and value not in param.enum:
        return "must be one of " + ", ".join(param.enum)
    if param.pattern is not None:
        pattern = PATTERNS[param.pattern]
        if not isinstance(value, str) or not pattern.match(value):
            if param.hint:
                return param.hint
            return f"must match {param.pattern}"
    if isinstance(value, int) and not isinstance(value, bool):
        if param.minimum is not None and value < param.minimum:
            return param.hint or f"must be >= {param.minimum}"
        if param.maximum is not None and value > param.maximum:
            return param.hint or f"must be <= {param.maximum}"
    return None


class ToolRegistry:
    def __init__(self, tools: Iterable[Tool] = TOOLS) -> None:
        self._tools: dict[str, Tool] = {}
        for tool in tools:
            if tool.name in self._tools:
                raise ValueError(f"duplicate tool name: {tool.name}")
            self._tools[tool.name] = tool

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(self._tools)

    def get(self, name: str) -> Tool | None:
        return self._tools.get(name)

    def require(self, name: str) -> Tool:
        tool = self._tools.get(name)
        if tool is None:
            raise KeyError(name)
        return tool

    def allowed(self, names: Iterable[str]) -> list[Tool]:
        """Tools in *names*, in the order the prompt declared them.

        A name that no registered tool implements raises, because a prompt that
        promises a tool the registry does not have is an authoring bug that
        would otherwise show up as a mysterious scenario failure.
        """
        out = []
        for name in names:
            out.append(self.require(name.strip()))
        return out

    def check(self, tool_name: str, field_name: str, value: Any) -> str | None:
        """Constraint check for one field — used by the policy before it calls."""
        param = self.require(tool_name).params.get(field_name)
        if param is None:
            return f"unknown argument '{field_name}'"
        return _check_value(param, value)

    def validate(self, tool_name: str, args: dict[str, Any]) -> list[str]:
        """All problems with *args*, empty list when the call is well formed."""
        tool = self._tools.get(tool_name)
        if tool is None:
            return [f"unknown tool '{tool_name}'"]
        errors: list[str] = []
        for key in args:
            if key not in tool.params:
                errors.append(f"unknown argument '{key}' for tool '{tool_name}'")
        for name, param in tool.params.items():
            if name not in args:
                if param.required:
                    errors.append(f"missing required argument '{name}'")
                continue
            problem = _check_value(param, args[name])
            if problem:
                errors.append(f"argument '{name}' {problem}")
        return errors

    def execute(self, name: str, args: dict[str, Any], ctx: ToolContext) -> dict[str, Any]:
        """Validate then run. Always returns a payload, never raises."""
        tool = self._tools.get(name)
        if tool is None:
            return {"ok": False, "code": "unknown_tool", "error": f"unknown_tool: {name}"}
        errors = self.validate(name, args)
        if errors:
            return {
                "ok": False,
                "code": "validation",
                "error": "validation: " + "; ".join(errors),
            }
        try:
            return tool.handler(args, ctx)
        except Exception as exc:  # scripted handlers should not throw, but a guard
            # beats a traceback in the middle of a regression run.
            return {"ok": False, "code": "handler_error", "error": f"handler_error: {exc}"}

    def json_schema(self, tools: Iterable[Tool] | None = None) -> list[dict[str, Any]]:
        """OpenAI-style `tools` array for the live chat.completions backend."""
        out = []
        for tool in tools if tools is not None else self._tools.values():
            out.append(
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": {
                            "type": "object",
                            "properties": {k: p.describe() for k, p in tool.params.items()},
                            "required": [k for k, p in tool.params.items() if p.required],
                        },
                    },
                }
            )
        return out


DEFAULT_REGISTRY = ToolRegistry()
