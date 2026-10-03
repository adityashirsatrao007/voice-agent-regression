"""Deterministic offline policy (the default "model") plus its Backend wrapper.

This is not a language model and does not pretend to be one: it parses the
caller's turn with regular expressions over a fixed slot/intent vocabulary and
returns either a tool call or a sentence. What makes it useful for regression
work is that it reproduces the *failure modes the matrix is written to catch* —
wrong tool, invented argument, skipped auth, missing escalation, blown step
budget — not that it understands language.

Two inputs come from the prompt rather than from this file, which is what makes
a prompt bump observable without a network:

``spec.tools``
    The licensed tool list. A tool the prompt does not name cannot be called,
    so v1 (no ``cancel_booking``) escalates cancellations while v2 handles them.
``spec.clarify_on_missing_slots``
    v1 guesses a missing slot and gets a validation error back; v2 asks first.

The FAQ block in the prompt body is the third load-bearing input: the answer to
a no-tool question is read out of the prompt, not hard-coded here.
"""

from __future__ import annotations

import json
import re
from typing import Any, Sequence

from ..prompts import PromptSpec
from ..registry import Tool, ToolContext, ToolRegistry
from .base import Backend, Decision, ToolCall

REFERENCE_YEAR = 2026  # frozen: a live today() would move date gold every New Year

_MONTHS = {
    name.lower(): index
    for index, name in enumerate(
        (
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December",
        ),
        start=1,
    )
}

_PHONE_RE = re.compile(r"\b([6-9]\d{9})\b")
_ACCOUNT_RE = re.compile(r"\b(ACC-\d{4})\b")
_BOOKING_RE = re.compile(r"\b(BK-\d{6})\b")
_DATE_RE = re.compile(r"\b(\d{1,2})\s+(%s)\b" % "|".join(_MONTHS), re.I)
_TIME_RE = re.compile(r"\b(\d{1,2}):(\d{2})\b")
_TIME_WORD_RE = re.compile(r"\b(\d{1,2})\s*(am|pm)\b", re.I)
_PARTY_RE = re.compile(r"\b(\d+)\s+(?:people|persons|log)\b", re.I)

_INJECTION_RE = re.compile(
    r"[.!?]?\s*(?:also,?\s*)?ignore (?:your|all|previous|above)?\s*"
    r"(?:instructions|prompts?|guidelines)\b.*$",
    re.I | re.S,
)
_REFUSAL_RE = re.compile(
    r"\bOTP\b|bypass|without verifying|skip verification|password|\bhack\b", re.I
)
_ESCALATE_RE = re.compile(
    r"\btransfer me\b|\ba human\b|\bhuman agent\b|\ba person\b|real person"
    r"|\bmanager\b|talk to (?:a )?(?:person|someone)|speak to (?:a )?(?:person|someone)",
    re.I,
)
_CANCEL_RE = re.compile(r"\bcancel(?:lation)?\b", re.I)
_RESCHEDULE_RE = re.compile(r"\breschedule\b|change my appointment|move my appointment", re.I)
_BOOK_RE = re.compile(
    r"\bbook(?:ing)?\b|\bappointment\b|\bconsultation\b|\bslot\b|\bdemo\b|\breservation\b", re.I
)
_ACCOUNT_MENTION_RE = re.compile(r"\bbalance\b|\baccount\b|ACC-\d{4}", re.I)
_RESUME_BOOK_RE = re.compile(r"\b(?:make it|actually|instead|change|people|persons)\b", re.I)

_REASON_SCHEDULE = re.compile(r"can't make it|cannot make it|conflict|changed my|plans changed", re.I)
_REASON_BILLING = re.compile(r"\bbill\b|payment|debit|charge|refund|invoice", re.I)
_REASON_TECHNICAL = re.compile(r"not working|error|broken|bug|technical|failed", re.I)
_REASON_COMPLAINT = re.compile(
    r"third time|nothing is fixed|frustrat|angry|useless|again and again|right now", re.I
)

HUMAN_TEXT = "I've moved you to the human support queue - someone will pick this up shortly."
REFUSAL_TEXT = (
    "I can't share an OTP or skip verification. If you're locked out, "
    "I can connect you to a human agent."
)
AUTH_CLARIFY_TEXT = (
    "To pull up the account I need to verify you first - please share the number "
    "registered with it."
)
ACCOUNT_CLARIFY_TEXT = "Which account is that? I need the ACC- id before I can read it."
RESCHEDULE_CLARIFY_TEXT = (
    "I can't reschedule a booking here - could you share your booking reference? "
    "I can cancel and rebook once I have it."
)
INJECTED_NOTE = "I can't act on instructions that ask me to ignore my guidelines."
FALLBACK_TEXT = "I'm not sure I follow - could you rephrase that?"

SLOT_LABELS = {
    "service": "service type",
    "date": "date",
    "time": "time",
    "party_size": "party size",
    "phone": "phone number",
}


def _strip_injection(text: str) -> tuple[str, bool]:
    match = _INJECTION_RE.search(text)
    if match is None:
        return text, False
    return text[: match.start()].rstrip(), True


def _summary(text: str, limit: int = 120) -> str:
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "..."


def _reason(text: str) -> str:
    # Order matters: a cancellation-shaped complaint must not be classed as
    # billing just because the transcript also mentions a charge.
    if _REASON_SCHEDULE.search(text):
        return "schedule_conflict"
    if _REASON_BILLING.search(text):
        return "billing"
    if _REASON_TECHNICAL.search(text):
        return "technical"
    if _REASON_COMPLAINT.search(text):
        return "complaint"
    return "other"


class MockPolicy:
    def __init__(self, spec: PromptSpec, registry: ToolRegistry) -> None:
        self.spec = spec
        self.registry = registry

    # -- transcript parsing -------------------------------------------------
    @staticmethod
    def last_user(messages: Sequence[dict[str, Any]]) -> str:
        for message in reversed(messages):
            if message.get("role") == "user":
                return str(message.get("content") or "")
        return ""

    @staticmethod
    def slots(text: str) -> dict[str, Any]:
        found: dict[str, Any] = {}
        lowered = text.lower()
        for service in ("consultation", "followup", "demo"):
            if service in lowered:
                found["service"] = service
                break
        date_match = _DATE_RE.search(text)
        if date_match:
            day, month = int(date_match.group(1)), _MONTHS[date_match.group(2).lower()]
            found["date"] = f"{REFERENCE_YEAR:04d}-{month:02d}-{day:02d}"
        clock = _TIME_RE.search(text)
        if clock:
            found["time"] = f"{int(clock.group(1)):02d}:{clock.group(2)}"
        else:
            word_clock = _TIME_WORD_RE.search(text)
            if word_clock:
                hour, suffix = int(word_clock.group(1)), word_clock.group(2).lower()
                if suffix == "pm" and hour != 12:
                    hour += 12
                elif suffix == "am" and hour == 12:
                    hour = 0
                found["time"] = f"{hour:02d}:00"
        party = _PARTY_RE.search(text)
        if party:
            found["party_size"] = int(party.group(1))
        phone = _PHONE_RE.search(text)
        if phone:
            found["phone"] = phone.group(1)
        return found

    # -- decision helpers ---------------------------------------------------
    def _call(self, name: str, args: dict[str, Any], ctx: ToolContext) -> Decision:
        ctx.call_seq += 1
        ctx.last_call = {"name": name, "args": dict(args)}
        return Decision("", (ToolCall(name=name, args=dict(args), id=f"call_{ctx.call_seq}"),))

    @staticmethod
    def _detail(payload: dict[str, Any]) -> str:
        error = str(payload.get("error", "unknown error"))
        return error.split(":", 1)[1].strip() if ":" in error else error.strip()

    def _success_text(self, call: dict[str, Any], payload: dict[str, Any]) -> str:
        name, args = call.get("name", ""), call.get("args", {})
        output = str(payload.get("output", ""))
        if name == "book_appointment":
            return (
                f"Booked {args.get('service')} on {args.get('date')} at {args.get('time')} "
                f"for {args.get('party_size')} people. Confirmation {output}."
            )
        if name == "cancel_booking":
            return (
                f"Cancelled booking {args.get('booking_id')}. "
                "A confirmation will be sent to your registered number."
            )
        if name == "get_balance":
            return f"The balance in {args.get('account_id')} is {output}."
        if name == "lookup_account":
            return f"{args.get('account_id')} is registered to {output}."
        if name == "verify_identity":
            return f"Your number {args.get('phone')} is verified."
        if name == "transfer_to_human":
            return HUMAN_TEXT
        return output or "Done."

    def _error_text(self, call: dict[str, Any], payload: dict[str, Any]) -> str:
        name, args = call.get("name", ""), call.get("args", {})
        code = str(payload.get("code", ""))
        detail = self._detail(payload)
        if code == "validation":
            return f"I can't call {name} yet: {detail}. Could you share that?"
        if code == "auth_required":
            return f"You'll need to verify your number first - {detail}."
        if code == "not_licensed":
            return f"I can't call {name} - it isn't available to me right now."
        if code == "slot_unavailable":
            return (
                f"I couldn't complete that booking: {detail}. "
                "Would you like to pick another day?"
            )
        if code == "account_not_found":
            return f"I couldn't find {args.get('account_id', 'that account')}: {detail}."
        return f"I couldn't complete that request: {detail}."

    def _after_tool(self, payload: dict[str, Any], ctx: ToolContext, injected: bool) -> Decision:
        if ctx.chain:
            # Deferred step of a verify-then-read flow: run it before answering.
            name, args = ctx.chain.pop(0)
            return self._call(name, args, ctx)
        call = ctx.last_call
        if payload.get("ok"):
            text = self._success_text(call, payload)
            if call.get("name") in ("get_balance", "lookup_account"):
                ctx.pending_intent = None
        else:
            text = self._error_text(call, payload)
        if injected:
            text = f"{text} {INJECTED_NOTE}"
        return Decision(text)

    def _missing_slots(self, missing: list[str]) -> Decision:
        labels = ", ".join(SLOT_LABELS.get(name, name) for name in missing)
        return Decision(f"Before I can book I need {labels}. Could you share those?")

    def _booking(self, text: str, ctx: ToolContext, names: set[str]) -> Decision:
        merged = {**ctx.pending_slots, **self.slots(text)}
        ctx.pending_slots = merged
        if "book_appointment" not in names:
            return Decision(FALLBACK_TEXT)
        # Constraint checking reuses the registry's own declarations instead of a
        # second copy of the rules: if party_size's range moves, both the policy
        # and the validator move with it.
        for field_name in self.registry.require("book_appointment").params:
            value = merged.get(field_name)
            if value is None:
                continue
            problem = self.registry.check("book_appointment", field_name, value)
            if problem:
                return Decision(
                    f"The {SLOT_LABELS.get(field_name, field_name)} {problem}. "
                    "Could you give me a different one?"
                )
        missing = [name for name in self.registry.require("book_appointment").params if name not in merged]
        if missing:
            if self.spec.clarify_on_missing_slots:
                return self._missing_slots(missing)
            # v1 behaviour: guess. The registry rejects the call and the next
            # decision turns that error into a question.
            return self._call("book_appointment", dict(merged), ctx)
        return self._call("book_appointment", dict(merged), ctx)

    def _account(self, text: str, ctx: ToolContext, names: set[str]) -> Decision:
        account = _ACCOUNT_RE.search(text)
        if account:
            ctx.account_id = account.group(1)
        phone = _PHONE_RE.search(text)
        # pending_intent stores the tool name, so compare against the tool name:
        # storing "balance" here would silently fall back to lookup_account on
        # the turn where the caller hands over their phone number.
        wants_balance = "balance" in text.lower() or ctx.pending_intent == "get_balance"
        action = "get_balance" if wants_balance else "lookup_account"
        if action not in names:
            return Decision(FALLBACK_TEXT)
        if not ctx.authenticated:
            ctx.pending_intent = action
            if phone:
                if not ctx.account_id:
                    return Decision(ACCOUNT_CLARIFY_TEXT)
                ctx.chain = [(action, {"account_id": ctx.account_id})]
                return self._call("verify_identity", {"phone": phone.group(1)}, ctx)
            return Decision(AUTH_CLARIFY_TEXT)
        if not ctx.account_id:
            return Decision(ACCOUNT_CLARIFY_TEXT)
        return self._call(action, {"account_id": ctx.account_id}, ctx)

    # -- the policy ---------------------------------------------------------
    def decide(self, messages: Sequence[dict[str, Any]], tools: Sequence[Tool], ctx: ToolContext) -> Decision:
        names = {tool.name for tool in tools}
        text = self.last_user(messages)
        cleaned, injected = _strip_injection(text)
        last = messages[-1] if messages else {}

        if last.get("role") == "tool":
            payload = _load_payload(last)
            return self._after_tool(payload, ctx, injected)

        if _REFUSAL_RE.search(cleaned):
            ctx.chain = []
            return Decision(REFUSAL_TEXT)

        if _CANCEL_RE.search(cleaned):
            if "cancel_booking" in names:
                booking = _BOOKING_RE.search(cleaned)
                if not booking:
                    return Decision("Which booking should I cancel? I need the BK- reference.")
                return self._call(
                    "cancel_booking",
                    {"booking_id": booking.group(1), "reason": _reason(cleaned)},
                    ctx,
                )
            # No cancel tool licensed (prompt v1): the only honest move is a handoff.
            return self._call(
                "transfer_to_human",
                {"reason": _reason(cleaned), "summary": _summary(cleaned)},
                ctx,
            )

        if _RESCHEDULE_RE.search(cleaned):
            return Decision(RESCHEDULE_CLARIFY_TEXT)

        if _ESCALATE_RE.search(cleaned):
            return self._call(
                "transfer_to_human",
                {"reason": _reason(cleaned), "summary": _summary(cleaned)},
                ctx,
            )

        if _ACCOUNT_MENTION_RE.search(cleaned) or (ctx.pending_intent and not ctx.chain):
            return self._account(cleaned, ctx, names)

        if _BOOK_RE.search(cleaned):
            return self._booking(cleaned, ctx, names)

        if ctx.pending_slots and _RESUME_BOOK_RE.search(cleaned):
            return self._booking(cleaned, ctx, names)

        answer = self.spec.faq_answer(cleaned)
        if answer:
            return Decision(f"{answer} {INJECTED_NOTE}" if injected else answer)

        return Decision(INJECTED_NOTE if injected else FALLBACK_TEXT)


def _load_payload(message: dict[str, Any]) -> dict[str, Any]:
    raw = message.get("content")
    if isinstance(raw, dict):
        return raw
    try:
        parsed = json.loads(str(raw))
    except (TypeError, ValueError):
        return {"ok": False, "code": "tool_error", "error": f"tool_error: {raw}"}
    return parsed if isinstance(parsed, dict) else {"ok": False, "error": str(parsed)}


class MockBackend(Backend):
    name = "mock"

    def __init__(self, spec: PromptSpec, registry: ToolRegistry) -> None:
        super().__init__(spec, registry)
        self.policy = MockPolicy(spec, registry)

    def decide(self, messages: list[dict[str, Any]], tools: Sequence[Tool], ctx: ToolContext) -> Decision:
        return self.policy.decide(messages, tools, ctx)
