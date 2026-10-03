import json
import unittest

from vareg.agent import Agent, tool_trace
from vareg.backends.base import Backend, Decision, ToolCall
from vareg.backends.mock import MockBackend
from vareg.prompts import load_spec
from vareg.registry import DEFAULT_REGISTRY, ToolContext
from vareg.scenarios import load_matrix


def scenario(scenario_id: str):
    return next(s for s in load_matrix() if s.id == scenario_id)


class ScriptedBackend(Backend):
    """Plays back a fixed list of decisions — used to test the loop itself."""

    name = "scripted"

    def __init__(self, spec, registry, script):
        super().__init__(spec, registry)
        self.script = list(script)

    def decide(self, messages, tools, ctx):
        return self.script.pop(0) if self.script else Decision("done")


class LoopingBackend(Backend):
    """Never answers: the step budget has to stop it."""

    name = "looping"

    def __init__(self, spec, registry):
        super().__init__(spec, registry)
        self.calls = 0

    def decide(self, messages, tools, ctx):
        self.calls += 1
        return Decision(
            "",
            (
                ToolCall(
                    name="transfer_to_human",
                    args={"reason": "other", "summary": "still thinking"},
                    id=f"call_{self.calls}",
                ),
            ),
        )


class LoopMessageTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec()
        self.scn = scenario("booking_full_args")

    def test_roles_follow_the_function_calling_order(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(self.scn)
        roles = [m["role"] for m in conversation.messages]
        self.assertEqual(
            roles, ["system", "user", "assistant", "tool", "assistant"]
        )

    def test_assistant_message_carries_the_tool_call(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(self.scn)
        assistant = conversation.messages[2]
        self.assertEqual(assistant["tool_calls"][0]["name"], "book_appointment")
        self.assertEqual(assistant["tool_calls"][0]["args"]["party_size"], 2)
        self.assertEqual(assistant["content"], "")

    def test_tool_payload_is_json_and_successful(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(self.scn)
        payload = json.loads(conversation.messages[3]["content"])
        self.assertTrue(payload["ok"])
        self.assertEqual(payload["output"], "BK-100001")

    def test_tool_call_id_matches_the_tool_message(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(self.scn)
        self.assertEqual(
            conversation.messages[2]["tool_calls"][0]["id"],
            conversation.messages[3]["tool_call_id"],
        )

    def test_final_text_is_the_last_assistant_message(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(self.scn)
        self.assertEqual(conversation.turns[0].final_text, conversation.messages[-1]["content"])
        self.assertIn("Confirmation BK-100001", conversation.turns[0].final_text)

    def test_multi_turn_history_accumulates(self) -> None:
        scn = scenario("account_auth_multiturn")
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(scn)
        users = [m for m in conversation.messages if m["role"] == "user"]
        self.assertEqual(len(users), 3)
        self.assertEqual(len(conversation.turns), 3)

    def test_conversation_metadata(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(self.scn)
        self.assertEqual(conversation.scenario_id, "booking_full_args")
        self.assertEqual(conversation.backend, "mock")
        self.assertEqual(conversation.prompt_version, self.spec.version)

    def test_agent_only_exposes_licensed_tools(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        self.assertEqual(tuple(t.name for t in agent.tools), self.spec.tools)

    def test_tool_trace_is_flat(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        trace = tool_trace(agent.run(self.scn))
        self.assertEqual(len(trace), 1)
        self.assertEqual(trace[0]["tool"], "book_appointment")
        self.assertTrue(trace[0]["ok"])

    def test_step_latency_is_recorded(self) -> None:
        agent = Agent(MockBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(self.scn)
        for step in conversation.steps:
            self.assertGreaterEqual(step.latency_ms, 0.0)
        self.assertGreater(conversation.turns[0].latency_ms, 0.0)


class BudgetTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec()
        self.scn = scenario("booking_full_args")

    def test_looping_backend_is_stopped_by_the_budget(self) -> None:
        agent = Agent(LoopingBackend(self.spec, DEFAULT_REGISTRY), DEFAULT_REGISTRY, self.spec)
        turn = agent.run(self.scn).turns[0]
        self.assertEqual(len(turn.steps), self.scn.max_steps)
        self.assertTrue(turn.budget_exceeded)
        self.assertEqual(turn.final_text, "")

    def test_finishing_exactly_on_the_last_allowed_step_is_not_a_breach(self) -> None:
        script = [
            Decision("", (ToolCall("book_appointment", {"service": "demo"}, "call_1"),)),
            Decision("", (ToolCall("book_appointment", {"service": "demo"}, "call_2"),)),
            Decision("answered on the final allowed step"),
        ]
        agent = Agent(ScriptedBackend(self.spec, DEFAULT_REGISTRY, script), DEFAULT_REGISTRY, self.spec)
        turn = agent.run(self.scn).turns[0]
        self.assertEqual(len(turn.steps), 3)
        self.assertFalse(turn.budget_exceeded)
        self.assertEqual(turn.final_text, "answered on the final allowed step")


class LicensingTests(unittest.TestCase):
    def setUp(self) -> None:
        self.spec = load_spec()
        self.scn = scenario("booking_full_args")

    def test_executing_a_tool_the_prompt_does_not_license_is_refused(self) -> None:
        spec = load_spec("v1")  # v1 does not name cancel_booking
        script = [
            Decision(
                "",
                (ToolCall("cancel_booking", {"booking_id": "BK-123456", "reason": "other"}, "call_1"),),
            ),
            Decision("I could not do that."),
        ]
        agent = Agent(ScriptedBackend(spec, DEFAULT_REGISTRY, script), DEFAULT_REGISTRY, spec)
        conversation = agent.run(scenario("cancellation_policy"))
        payload = json.loads(conversation.messages[3]["content"])
        self.assertFalse(payload["ok"])
        self.assertEqual(payload["code"], "not_licensed")

    def test_unknown_tool_from_a_backend_is_reported_structurally(self) -> None:
        script = [Decision("", (ToolCall("time_travel", {}, "call_1"),)), Decision("nope")]
        agent = Agent(ScriptedBackend(self.spec, DEFAULT_REGISTRY, script), DEFAULT_REGISTRY, self.spec)
        conversation = agent.run(self.scn)
        payload = json.loads(conversation.messages[3]["content"])
        self.assertEqual(payload["code"], "unknown_tool")

    def test_tool_errors_reach_the_model_as_messages(self) -> None:
        spec = load_spec()
        script = [
            Decision("", (ToolCall("get_balance", {"account_id": "ACC-1234"}, "call_1"),)),
            Decision("you need to verify first"),
        ]
        agent = Agent(ScriptedBackend(spec, DEFAULT_REGISTRY, script), DEFAULT_REGISTRY, spec)
        conversation = agent.run(self.scn)
        payload = json.loads(conversation.messages[3]["content"])
        self.assertEqual(payload["code"], "auth_required")


if __name__ == "__main__":
    unittest.main()
