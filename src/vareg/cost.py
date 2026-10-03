"""Model pricing, deliberately unfilled.

A cost column is only honest if the price came from somewhere. No provider
pricing page was consulted while building this repo, so every entry ships as
``None``: the report prints ``not priced`` instead of a number nobody verified.
Fill an entry from the provider's own pricing page (with the date you read it)
and the same code path starts reporting a real cost for that model.

Costs are only ever computed when the `llm` backend actually ran and returned
usage. The offline backends print a note, never a figure.
"""

from __future__ import annotations

from typing import Any

PRICES: dict[str, dict[str, Any]] = {
    "gpt-4o-mini": {
        "input_per_mtok": None,
        "output_per_mtok": None,
        "note": "[verify] fill from the provider's pricing page before use",
    },
    "sarvam-m": {
        "input_per_mtok": None,
        "output_per_mtok": None,
        "note": "[verify] fill from the provider's pricing page before use",
    },
}


def price_for(model: str) -> dict[str, Any] | None:
    return PRICES.get(model)


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> tuple[float | None, str]:
    """``(cost, note)``. ``cost`` is None whenever the price is unknown."""
    entry = price_for(model)
    if entry is None:
        return None, f"model {model!r} has no entry in vareg.cost.PRICES"
    input_rate = entry.get("input_per_mtok")
    output_rate = entry.get("output_per_mtok")
    if input_rate is None or output_rate is None:
        return None, str(entry.get("note") or "price not filled in")
    cost = (input_tokens / 1_000_000.0) * float(input_rate) + (
        output_tokens / 1_000_000.0
    ) * float(output_rate)
    return round(cost, 6), f"list price as recorded in vareg.cost.PRICES for {model}"
