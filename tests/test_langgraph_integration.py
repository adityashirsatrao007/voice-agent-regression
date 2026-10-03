"""Integration tests that only run where ``langgraph`` is installed.

The whole file skips (rather than fakes success) on a machine without the
package, so `PYTHONPATH=src python3 -m unittest discover -s tests` stays green
on the system interpreter while the dedicated CI job exercises these paths.

What is genuinely under test here: the graph compiles, ToolNode executes this
repo's tools (including the auth guard), the message conversion round-trips so
the harness scores the same trajectory it would score on the mock backend, and
``recursion_limit`` is a working step budget.
"""

import inspect
import json
import unittest

from vareg.agent import Agent
from vareg.backends import langgraph_available, make_backend
from vareg.backends.base import Decision, ToolCall
from vareg.backends.langgraph_backend import (
    GraphRecursionError,
    LangGraphBackend,
    Toolset,
    _build_policy_model,
    _message_to_dict,
    _to_langchain,
)
from vareg.metrics import GATED_METRICS
from vareg.prompts import load_spec
from vareg.registry import DEFAULT_REGISTRY, ToolContext
from vareg.report import run_matrix, usage_block
from vareg.scenarios import load_matrix

requires_langgraph = unittest.skipUnless(
    langgraph_available(),
    "langgraph is not installed in this interpreter (see .venv in the README)",
)


class LoopPolicy:
    """A policy that never answers, so only the graph's budget can stop it."""

    spec = None
    registry = DEFAULT_REGISTRY

    def decide(self, messages, tools, ctx) -> Decision:
        return Decision(
            content="",
            tool_calls=(ToolCall(name="get_balance", args={"account_id": "ACC-1234"}, id="call-loop"),),
        )


def _graph_for(backend: LangGraphBackend, ctx: ToolContext, tools):
    model = _build_policy_model(backend.policy, ctx, tools)
    bound = Toolset(backend.registry, ctx).methods_for([tool.name for tool in tools])
    from langgraph.prebuilt import create_react_agent

    return create_react_agent(model, bound)


@requires_langgraph
class GraphConstructionTests(unittest.TestCase):
    def test_graph_nodes_and_edges_match_the_documented_shape(self) -> None:
        spec = load_spec()
        backend = LangGraphBackend(spec, DEFAULT_REGISTRY)
        tools = DEFAULT_REGISTRY.allowed(spec.tools)
        graph = _graph_for(backend, ToolContext(), tools)
        drawn = graph.get_graph()
        self.assertEqual(set(drawn.nodes), {"__start__", "agent", "tools", "__end__"})
        self.assertEqual(
            {(edge.source, edge.target) for edge in drawn.edges},
            {
                ("__start__", "agent"),
                ("agent", "tools"),
                ("tools", "agent"),
                ("agent", "__end__"),
            },
        )

    def test_only_prompt_licensed_tools_are_bound(self) -> None:
        v1 = load_spec("v1")
        v1_tools = DEFAULT_REGISTRY.allowed(v1.tools)
        bound = [
            fn.__name__
            for fn in Toolset(DEFAULT_REGISTRY, ToolContext()).methods_for([t.name for t in v1_tools])
        ]
        self.assertEqual(bound, [tool.name for tool in v1_tools])
        # v1's tool list omits cancel_booking: the graph cannot reach it either.
        self.assertNotIn("cancel_booking", bound)
        self.assertIn("cancel_booking", load_spec("v2").tools)

    def test_toolset_docstrings_are_the_registry_descriptions(self) -> None:
        # LangChain takes each tool's description from the docstring, so drift
        # here would silently change what the model is told about the tools.
        for name in DEFAULT_REGISTRY.names:
            tool = DEFAULT_REGISTRY.require(name)
            with self.subTest(tool=tool.name):
                method = getattr(Toolset, tool.name)
                self.assertEqual(inspect.getdoc(method), tool.description)


@requires_langgraph
class GraphRunTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.spec = load_spec()
        cls.scenarios = load_matrix()

    def test_full_matrix_passes_with_every_metric_at_one(self) -> None:
        backend = LangGraphBackend(self.spec, DEFAULT_REGISTRY)
        report, _ = run_matrix(self.scenarios, backend, DEFAULT_REGISTRY, self.spec)
        self.assertEqual(report["backend"], "langgraph")
        failed = [row["id"] for row in report["scenarios"] if not row["passed"]]
        self.assertEqual(failed, [])
        for name in GATED_METRICS:
            self.assertEqual(report["metrics"][name], 1.0, name)
        self.assertGreater(report["latency"]["steps"], 0)
        self.assertIsNone(report["usage"]["cost_usd"])

    def test_scores_identically_to_the_mock_loop(self) -> None:
        mock_report, _ = run_matrix(
            self.scenarios, make_backend("mock", self.spec, DEFAULT_REGISTRY),
            DEFAULT_REGISTRY, self.spec,
        )
        graph_report, _ = run_matrix(
            self.scenarios, LangGraphBackend(self.spec, DEFAULT_REGISTRY),
            DEFAULT_REGISTRY, self.spec,
        )
        self.assertEqual(mock_report["metrics"], graph_report["metrics"])

    def test_trajectory_matches_the_mock_backend_turn_by_turn(self) -> None:
        scenario = next(s for s in self.scenarios if s.id == "account_auth_multiturn")
        traces = {}
        for name in ("mock", "langgraph"):
            backend = make_backend(name, self.spec, DEFAULT_REGISTRY)
            conversation = Agent(backend, DEFAULT_REGISTRY, self.spec).run(scenario)
            traces[name] = [
                (
                    [(call.name, call.args) for call in turn.tool_calls],
                    turn.final_text,
                    turn.budget_exceeded,
                )
                for turn in conversation.turns
            ]
        self.assertEqual(traces["mock"], traces["langgraph"])
        # Hand-checked gold sequence: turn 0 refuses to read anything before
        # verification, turn 1 verifies then reads, turn 2 reads the holder.
        self.assertEqual(traces["mock"][0][0], [])
        self.assertEqual(
            [name for calls, _, _ in traces["mock"] for name, _ in calls],
            ["verify_identity", "get_balance", "lookup_account"],
        )
        self.assertEqual(
            traces["mock"][1][1], "The balance in ACC-1234 is Rs. 45,769."
        )

    def test_step_latencies_are_recorded_for_every_decision(self) -> None:
        scenario = next(s for s in self.scenarios if s.id == "booking_full_args")
        backend = LangGraphBackend(self.spec, DEFAULT_REGISTRY)
        conversation = Agent(backend, DEFAULT_REGISTRY, self.spec).run(scenario)
        self.assertGreater(len(conversation.steps), 0)
        for step in conversation.steps:
            self.assertGreaterEqual(step.latency_ms, 0.0)
        self.assertIsNone(usage_block(backend)["cost_usd"])

    def test_looping_policy_becomes_a_blown_budget_not_a_hang(self) -> None:
        backend = LangGraphBackend(self.spec, DEFAULT_REGISTRY)
        backend.policy = LoopPolicy()
        tools = DEFAULT_REGISTRY.allowed(self.spec.tools)
        messages = [
            {"role": "system", "content": self.spec.body},
            {"role": "user", "content": "what is my balance?"},
        ]
        outcome = backend.run_turn(messages, tools, ToolContext(), max_steps=2)
        self.assertTrue(outcome.budget_exceeded)
        self.assertEqual(outcome.final_text, "")

    def test_raw_graph_raises_graph_recursion_error_at_the_limit(self) -> None:
        tools = DEFAULT_REGISTRY.allowed(self.spec.tools)
        ctx = ToolContext()
        # Build the graph around the looping policy directly.
        model = _build_policy_model(LoopPolicy(), ctx, tools)
        from langgraph.prebuilt import create_react_agent

        graph = create_react_agent(
            model, Toolset(DEFAULT_REGISTRY, ctx).methods_for([tool.name for tool in tools])
        )
        with self.assertRaises(GraphRecursionError):
            graph.invoke(
                {"messages": _to_langchain([{"role": "user", "content": "hello"}])},
                config={"recursion_limit": 5},
            )


@requires_langgraph
class MessageConversionTests(unittest.TestCase):
    def test_every_role_round_trips(self) -> None:
        messages = [
            {"role": "system", "content": "you are a voice agent"},
            {"role": "user", "content": "namaste, balance batao"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "get_balance", "args": {"account_id": "ACC-1234"}}
                ],
            },
            {"role": "tool", "content": '{"ok": true}', "tool_call_id": "call_1", "name": "get_balance"},
        ]
        back = [_message_to_dict(message) for message in _to_langchain(messages)]
        self.assertEqual(back, messages)

    def test_plain_final_answer_round_trips(self) -> None:
        messages = [{"role": "assistant", "content": "We are open 9 to 6, Monday to Saturday."}]
        back = [_message_to_dict(message) for message in _to_langchain(messages)]
        self.assertEqual(back, messages)

    def test_unknown_role_falls_back_to_user(self) -> None:
        converted = _to_langchain([{"role": "narrator", "content": "hm"}])
        self.assertEqual(_message_to_dict(converted[0]), {"role": "user", "content": "hm"})


@requires_langgraph
class ToolsetContractTests(unittest.TestCase):
    def test_auth_guard_holds_on_the_graph_path(self) -> None:
        ctx = ToolContext()
        toolset = Toolset(DEFAULT_REGISTRY, ctx)
        blocked = json.loads(toolset.lookup_account("ACC-1234"))
        self.assertFalse(blocked["ok"])
        self.assertEqual(blocked["code"], "auth_required")
        self.assertTrue(json.loads(toolset.verify_identity("9876543210"))["ok"])
        allowed = json.loads(toolset.lookup_account("ACC-1234"))
        self.assertEqual(allowed, {"ok": True, "output": "A. Sharma"})

    def test_toolset_revalidates_arguments_before_running(self) -> None:
        toolset = Toolset(DEFAULT_REGISTRY, ToolContext())
        rejected = json.loads(toolset.get_balance(account_id="ACC-12345"))
        self.assertFalse(rejected["ok"])
        self.assertEqual(rejected["code"], "validation")
        self.assertIn("account_id", rejected["error"])

    def test_methods_for_returns_callable_bound_methods(self) -> None:
        toolset = Toolset(DEFAULT_REGISTRY, ToolContext())
        methods = toolset.methods_for(["verify_identity"])
        self.assertEqual(len(methods), 1)
        self.assertEqual(methods[0].__name__, "verify_identity")
        with self.assertRaises(AttributeError):
            toolset.methods_for(["no_such_tool"])


if __name__ == "__main__":
    unittest.main()
