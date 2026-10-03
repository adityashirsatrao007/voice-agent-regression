import os
import unittest
from unittest import mock

from vareg.backends import BACKEND_NAMES, langgraph_available, make_backend
from vareg.backends.base import BackendError
from vareg.backends.llm import LLMBackend, _api_message, _load_dotenv
from vareg.backends.mock import MockBackend
from vareg.cost import PRICES, cost_usd, price_for
from vareg.prompts import load_spec
from vareg.registry import DEFAULT_REGISTRY
from vareg.report import usage_block

CLEAN_ENV = {"LLM_API_KEY": "", "SARVAM_API_KEY": "", "LLM_MODEL": "", "LLM_BASE_URL": ""}


class FactoryTests(unittest.TestCase):
    def test_known_backends(self) -> None:
        self.assertEqual(BACKEND_NAMES, ("mock", "llm", "langgraph"))
        backend = make_backend("mock", load_spec(), DEFAULT_REGISTRY)
        self.assertIsInstance(backend, MockBackend)

    def test_unknown_backend_is_actionable(self) -> None:
        with self.assertRaisesRegex(BackendError, "choose from"):
            make_backend("gpt4", load_spec(), DEFAULT_REGISTRY)

    def test_langgraph_available_is_boolean(self) -> None:
        self.assertIn(langgraph_available(), (True, False))


class GuardTests(unittest.TestCase):
    """A missing key or package must produce one line, never a traceback."""

    @mock.patch.dict(os.environ, CLEAN_ENV)
    @mock.patch("vareg.backends.llm._load_dotenv")
    def test_llm_without_a_key_explains_what_to_do(self, _loader) -> None:
        with self.assertRaises(BackendError) as caught:
            LLMBackend(load_spec(), DEFAULT_REGISTRY)
        self.assertIn(".env.example", str(caught.exception))
        self.assertIn("--backend mock", str(caught.exception))

    @mock.patch.dict(os.environ, {**CLEAN_ENV, "LLM_API_KEY": "test-key"})
    @mock.patch("vareg.backends.llm._load_dotenv")
    def test_llm_without_a_model_is_rejected(self, _loader) -> None:
        with self.assertRaisesRegex(BackendError, "LLM_MODEL"):
            LLMBackend(load_spec(), DEFAULT_REGISTRY)

    @mock.patch.dict(os.environ, {**CLEAN_ENV, "LLM_API_KEY": "test-key", "LLM_MODEL": "m"})
    @mock.patch("vareg.backends.llm._load_dotenv")
    def test_llm_builds_with_key_and_model(self, _loader) -> None:
        backend = LLMBackend(load_spec(), DEFAULT_REGISTRY)
        self.assertEqual(backend.name, "llm")
        self.assertEqual(backend.usage, {"input_tokens": 0, "output_tokens": 0})

    def test_langgraph_backend_without_the_package_is_actionable(self) -> None:
        if langgraph_available():
            self.skipTest("langgraph is installed here; the guard is covered in CI-less envs")
        with self.assertRaisesRegex(BackendError, "pip install langgraph"):
            make_backend("langgraph", load_spec(), DEFAULT_REGISTRY)

    def test_env_loader_fills_only_unset_keys(self) -> None:
        import tempfile
        from pathlib import Path

        with tempfile.TemporaryDirectory() as tmp:
            env_file = Path(tmp) / ".env"
            env_file.write_text("# comment\nFOO_VAREG_TEST=bar\n\n", encoding="utf-8")
            with mock.patch.dict(os.environ, {}, clear=False):
                os.environ.pop("FOO_VAREG_TEST", None)
                _load_dotenv(env_file)
                self.assertEqual(os.environ["FOO_VAREG_TEST"], "bar")
                os.environ["FOO_VAREG_TEST"] = "kept"
                _load_dotenv(env_file)
                self.assertEqual(os.environ["FOO_VAREG_TEST"], "kept")
                os.environ.pop("FOO_VAREG_TEST")


class WireFormatTests(unittest.TestCase):
    def test_plain_user_message(self) -> None:
        self.assertEqual(
            _api_message({"role": "user", "content": "hello"}),
            {"role": "user", "content": "hello"},
        )

    def test_assistant_tool_calls_are_wrapped_as_functions(self) -> None:
        wire = _api_message(
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "name": "get_balance", "args": {"account_id": "ACC-1234"}}
                ],
            }
        )
        call = wire["tool_calls"][0]
        self.assertEqual(call["type"], "function")
        self.assertEqual(call["function"]["name"], "get_balance")
        self.assertIn("ACC-1234", call["function"]["arguments"])

    def test_tool_message_keeps_its_call_id(self) -> None:
        wire = _api_message(
            {"role": "tool", "content": '{"ok": true}', "tool_call_id": "call_1"}
        )
        self.assertEqual(wire["tool_call_id"], "call_1")
        self.assertEqual(wire["role"], "tool")


class UsageAndCostTests(unittest.TestCase):
    def test_offline_backend_reports_no_usage(self) -> None:
        backend = make_backend("mock", load_spec(), DEFAULT_REGISTRY)
        block = usage_block(backend)
        self.assertIsNone(block["cost_usd"])
        self.assertIsNone(block["input_tokens"])
        self.assertIn("offline", block["note"])

    def test_unpriced_model_reports_no_cost(self) -> None:
        amount, note = cost_usd("gpt-4o-mini", 1000, 500)
        self.assertIsNone(amount)
        self.assertIn("verify", note)

    def test_unknown_model_reports_no_cost(self) -> None:
        amount, note = cost_usd("definitely-not-listed", 1000, 500)
        self.assertIsNone(amount)
        self.assertIn("PRICES", note)

    def test_cost_arithmetic_once_a_price_is_filled_in(self) -> None:
        # 1M input at 0.30 + 0.5M output at 0.60 = 0.30 + 0.30 = 0.60 USD.
        entry = PRICES["gpt-4o-mini"]
        original = (entry["input_per_mtok"], entry["output_per_mtok"])
        try:
            entry["input_per_mtok"] = 0.30
            entry["output_per_mtok"] = 0.60
            amount, _ = cost_usd("gpt-4o-mini", 1_000_000, 500_000)
            self.assertAlmostEqual(amount, 0.60, places=6)
        finally:
            entry["input_per_mtok"], entry["output_per_mtok"] = original

    def test_shipped_table_is_deliberately_unpriced(self) -> None:
        for model, entry in PRICES.items():
            with self.subTest(model=model):
                self.assertIsNone(entry["input_per_mtok"])
                self.assertIsNone(entry["output_per_mtok"])

    def test_price_for_lookup(self) -> None:
        self.assertIsNone(price_for("nope"))
        self.assertIsNotNone(price_for("gpt-4o-mini"))


if __name__ == "__main__":
    unittest.main()
