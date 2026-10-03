import json
import unittest

from vareg.registry import (
    DEFAULT_REGISTRY,
    ACCOUNT_NAMES,
    ToolContext,
    ToolRegistry,
)


class ValidationTests(unittest.TestCase):
    def test_complete_booking_call_is_valid(self) -> None:
        errors = DEFAULT_REGISTRY.validate(
            "book_appointment",
            {"service": "consultation", "date": "2026-10-12", "time": "19:00",
             "party_size": 2, "phone": "9876543210"},
        )
        self.assertEqual(errors, [])

    def test_missing_required_argument_is_named(self) -> None:
        errors = DEFAULT_REGISTRY.validate("book_appointment", {"service": "consultation"})
        self.assertIn("missing required argument 'date'", errors)

    def test_unknown_argument_is_rejected(self) -> None:
        # Hallucinated-argument defence: the model cannot invent a field.
        errors = DEFAULT_REGISTRY.validate(
            "verify_identity", {"phone": "9876543210", "otp": "123456"}
        )
        self.assertIn("unknown argument 'otp' for tool 'verify_identity'", errors)

    def test_wrong_type_is_rejected(self) -> None:
        errors = DEFAULT_REGISTRY.validate(
            "book_appointment",
            {"service": "consultation", "date": "2026-10-12", "time": "19:00",
             "party_size": "two", "phone": "9876543210"},
        )
        self.assertIn("argument 'party_size' must be an integer", errors)

    def test_bool_is_not_accepted_as_integer(self) -> None:
        errors = DEFAULT_REGISTRY.validate(
            "book_appointment",
            {"service": "consultation", "date": "2026-10-12", "time": "19:00",
             "party_size": True, "phone": "9876543210"},
        )
        self.assertIn("argument 'party_size' must be an integer", errors)

    def test_enum_violation_lists_the_options(self) -> None:
        errors = DEFAULT_REGISTRY.validate(
            "book_appointment",
            {"service": "haircut", "date": "2026-10-12", "time": "19:00",
             "party_size": 2, "phone": "9876543210"},
        )
        self.assertIn("argument 'service' must be one of consultation, followup, demo", errors)

    def test_slot_range_is_enforced_low(self) -> None:
        errors = DEFAULT_REGISTRY.validate(
            "book_appointment",
            {"service": "consultation", "date": "2026-10-12", "time": "19:00",
             "party_size": 0, "phone": "9876543210"},
        )
        self.assertIn("argument 'party_size' must be between 1 and 12", errors)

    def test_slot_range_is_enforced_high(self) -> None:
        errors = DEFAULT_REGISTRY.validate(
            "book_appointment",
            {"service": "consultation", "date": "2026-10-12", "time": "19:00",
             "party_size": 40, "phone": "9876543210"},
        )
        self.assertIn("argument 'party_size' must be between 1 and 12", errors)

    def test_phone_pattern(self) -> None:
        errors = DEFAULT_REGISTRY.validate("verify_identity", {"phone": "0123456789"})
        self.assertEqual(
            errors, ["argument 'phone' must be a 10-digit mobile number starting with 6-9"]
        )

    def test_booking_reference_pattern(self) -> None:
        errors = DEFAULT_REGISTRY.validate(
            "cancel_booking", {"booking_id": "ABC", "reason": "other"}
        )
        self.assertEqual(errors, ["argument 'booking_id' must look like BK-123456"])

    def test_unknown_tool(self) -> None:
        self.assertEqual(DEFAULT_REGISTRY.validate("nope", {}), ["unknown tool 'nope'"])

    def test_check_reports_a_single_field(self) -> None:
        self.assertIsNone(DEFAULT_REGISTRY.check("book_appointment", "party_size", 4))
        self.assertEqual(
            DEFAULT_REGISTRY.check("book_appointment", "party_size", 40),
            "must be between 1 and 12",
        )
        self.assertEqual(
            DEFAULT_REGISTRY.check("book_appointment", "colour", "red"), "unknown argument 'colour'"
        )


class ExecutionTests(unittest.TestCase):
    def test_successful_booking_returns_a_reference(self) -> None:
        ctx = ToolContext()
        result = DEFAULT_REGISTRY.execute(
            "book_appointment",
            {"service": "consultation", "date": "2026-10-12", "time": "19:00",
             "party_size": 2, "phone": "9876543210"},
            ctx,
        )
        self.assertTrue(result["ok"])
        self.assertEqual(result["output"], "BK-100001")

    def test_second_booking_increments(self) -> None:
        ctx = ToolContext()
        args = {"service": "demo", "date": "2026-10-13", "time": "11:00",
                "party_size": 1, "phone": "9876543210"}
        DEFAULT_REGISTRY.execute("book_appointment", args, ctx)
        second = DEFAULT_REGISTRY.execute("book_appointment", args, ctx)
        self.assertEqual(second["output"], "BK-100002")

    def test_sunday_slot_fails_scripted_backend(self) -> None:
        # 2026-10-18 is a Sunday.
        result = DEFAULT_REGISTRY.execute(
            "book_appointment",
            {"service": "followup", "date": "2026-10-18", "time": "11:00",
             "party_size": 2, "phone": "9876543210"},
            ToolContext(),
        )
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "slot_unavailable")
        self.assertIn("Sunday", result["error"])

    def test_weekday_slot_succeeds(self) -> None:
        # 2026-10-12 is a Monday.
        result = DEFAULT_REGISTRY.execute(
            "book_appointment",
            {"service": "followup", "date": "2026-10-12", "time": "11:00",
             "party_size": 2, "phone": "9876543210"},
            ToolContext(),
        )
        self.assertTrue(result["ok"])

    def test_validation_failure_is_a_payload_not_an_exception(self) -> None:
        result = DEFAULT_REGISTRY.execute("book_appointment", {"service": "consultation"}, ToolContext())
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "validation")
        self.assertIn("validation:", result["error"])

    def test_unknown_tool_is_a_payload(self) -> None:
        result = DEFAULT_REGISTRY.execute("nope", {}, ToolContext())
        self.assertEqual(result["code"], "unknown_tool")

    def test_auth_guard_blocks_balance_reads(self) -> None:
        result = DEFAULT_REGISTRY.execute("get_balance", {"account_id": "ACC-1234"}, ToolContext())
        self.assertFalse(result["ok"])
        self.assertEqual(result["code"], "auth_required")

    def test_auth_guard_blocks_account_lookup(self) -> None:
        result = DEFAULT_REGISTRY.execute("lookup_account", {"account_id": "ACC-1234"}, ToolContext())
        self.assertEqual(result["code"], "auth_required")

    def test_verify_identity_opens_the_guard(self) -> None:
        ctx = ToolContext()
        verified = DEFAULT_REGISTRY.execute("verify_identity", {"phone": "9876543210"}, ctx)
        self.assertTrue(verified["ok"])
        self.assertTrue(ctx.authenticated)
        balance = DEFAULT_REGISTRY.execute("get_balance", {"account_id": "ACC-1234"}, ctx)
        self.assertTrue(balance["ok"])

    def test_balance_is_the_documented_arithmetic(self) -> None:
        # ACC-1234 -> 1234*37+111 = 45769; ACC-5678 -> 5678*37+111 = 210197.
        ctx = ToolContext()
        ctx.authenticated = True
        first = DEFAULT_REGISTRY.execute("get_balance", {"account_id": "ACC-1234"}, ctx)
        second = DEFAULT_REGISTRY.execute("get_balance", {"account_id": "ACC-5678"}, ctx)
        self.assertEqual(first["output"], "Rs. 45,769")
        self.assertEqual(second["output"], "Rs. 210,197")

    def test_unknown_account_is_reported_not_invented(self) -> None:
        ctx = ToolContext()
        ctx.authenticated = True
        result = DEFAULT_REGISTRY.execute("get_balance", {"account_id": "ACC-0000"}, ctx)
        self.assertEqual(result["code"], "account_not_found")

    def test_lookup_returns_the_scripted_holder(self) -> None:
        ctx = ToolContext()
        ctx.authenticated = True
        result = DEFAULT_REGISTRY.execute("lookup_account", {"account_id": "ACC-1234"}, ctx)
        self.assertEqual(result["output"], ACCOUNT_NAMES["ACC-1234"])

    def test_cancel_and_transfer_succeed(self) -> None:
        cancelled = DEFAULT_REGISTRY.execute(
            "cancel_booking", {"booking_id": "BK-123456", "reason": "schedule_conflict"}, ToolContext()
        )
        transferred = DEFAULT_REGISTRY.execute(
            "transfer_to_human", {"reason": "complaint", "summary": "angry caller"}, ToolContext()
        )
        self.assertEqual(cancelled["output"], "BK-123456 cancelled")
        self.assertTrue(transferred["ok"])


class RegistryShapeTests(unittest.TestCase):
    def test_duplicate_tool_names_are_refused(self) -> None:
        tools = list(DEFAULT_REGISTRY.allowed(["get_balance"])) * 2
        with self.assertRaises(ValueError):
            ToolRegistry(tools)

    def test_allowed_raises_on_a_tool_the_registry_lacks(self) -> None:
        with self.assertRaises(KeyError):
            DEFAULT_REGISTRY.allowed(["book_appointment", "time_travel"])

    def test_json_schema_shape(self) -> None:
        schema = DEFAULT_REGISTRY.json_schema()
        entry = next(e for e in schema if e["function"]["name"] == "book_appointment")
        self.assertEqual(entry["type"], "function")
        properties = entry["function"]["parameters"]["properties"]
        self.assertEqual(entry["function"]["parameters"]["required"], list(properties))
        self.assertEqual(properties["party_size"]["minimum"], 1)
        self.assertEqual(properties["party_size"]["maximum"], 12)
        self.assertEqual(properties["service"]["enum"], ["consultation", "followup", "demo"])

    def test_json_schema_is_serialisable(self) -> None:
        json.dumps(DEFAULT_REGISTRY.json_schema())

    def test_json_schema_of_a_subset(self) -> None:
        subset = DEFAULT_REGISTRY.allowed(["cancel_booking"])
        names = [e["function"]["name"] for e in DEFAULT_REGISTRY.json_schema(subset)]
        self.assertEqual(names, ["cancel_booking"])


if __name__ == "__main__":
    unittest.main()
