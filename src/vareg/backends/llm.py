"""OpenAI-compatible chat.completions backend with function calling.

Wired, guarded and *not exercised while building this repo*: no API key was
available, so nothing in the README reports a number from this path. The
guards exist so that running it without a key gives one actionable line rather
than a traceback:

    BackendError: LLM_API_KEY (or SARVAM_API_KEY) is not set. Copy .env.example
    to .env and fill it in, or run --backend mock for an offline run.

Endpoint shape (POST ``{base}/chat/completions``, ``tools`` array,
``choices[0].message.tool_calls[].function.arguments`` as a JSON string) follows
the OpenAI surface every compatible provider reimplements. It is marked
``[verify]`` in the constants below — check it against the provider's current
docs before trusting a run.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Sequence

from ..prompts import PromptSpec
from ..registry import Tool, ToolContext, ToolRegistry
from .base import Backend, BackendError, Decision, ToolCall

# [verify] against the provider's current API docs.
DEFAULT_BASE_URL = "https://api.sarvam.ai/v1"


def _load_dotenv(path: Path) -> None:
    """Fill *unset* ``os.environ`` entries from a local ``KEY=VALUE`` file.

    stdlib only, because a zero-dependency core is a stated requirement; an
    explicit ``export`` still wins because populated keys are never overwritten.
    """
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip().strip("'\"")


def _api_message(message: dict[str, Any]) -> dict[str, Any]:
    """Convert an internal message to the chat.completions wire shape."""
    out: dict[str, Any] = {"role": message["role"], "content": message.get("content") or ""}
    if message["role"] == "tool":
        out["tool_call_id"] = message.get("tool_call_id", "")
    if message["role"] == "assistant" and message.get("tool_calls"):
        out["tool_calls"] = [
            {
                "id": call["id"],
                "type": "function",
                "function": {"name": call["name"], "arguments": json.dumps(call["args"])},
            }
            for call in message["tool_calls"]
        ]
    return out


class LLMBackend(Backend):
    name = "llm"

    def __init__(
        self,
        spec: PromptSpec,
        registry: ToolRegistry,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout: float = 60.0,
    ) -> None:
        super().__init__(spec, registry)
        _load_dotenv(Path(".env"))
        self.api_key = (
            api_key or os.environ.get("LLM_API_KEY") or os.environ.get("SARVAM_API_KEY") or ""
        ).strip()
        if not self.api_key:
            raise BackendError(
                "LLM_API_KEY (or SARVAM_API_KEY) is not set. Copy .env.example to .env "
                "and fill it in, or run --backend mock for an offline run."
            )
        try:
            import requests  # optional dependency: only this backend needs it
        except ImportError as exc:  # pragma: no cover - environment dependent
            raise BackendError(
                "the 'requests' package is required for --backend llm: pip install requests"
            ) from exc
        self._http = requests
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
        self.model = (model or os.environ.get("LLM_MODEL") or "").strip()
        if not self.model:
            raise BackendError("LLM_MODEL is not set. Copy .env.example to .env and set it.")
        self.timeout = timeout
        self.usage = {"input_tokens": 0, "output_tokens": 0}

    def decide(self, messages: list[dict[str, Any]], tools: Sequence[Tool], ctx: ToolContext) -> Decision:
        payload = {
            "model": self.model,
            "messages": [_api_message(m) for m in messages],
            "tools": self.registry.json_schema(tools),
            "tool_choice": "auto",
        }
        try:
            response = self._http.post(
                f"{self.base_url}/chat/completions",
                json=payload,
                headers={"Authorization": f"Bearer {self.api_key}"},
                timeout=self.timeout,
            )
        except self._http.RequestException as exc:  # pragma: no cover - network
            raise BackendError(f"chat.completions request failed: {exc}") from exc
        if response.status_code != 200:
            raise BackendError(
                f"chat.completions returned HTTP {response.status_code}: {response.text[:300]}"
            )
        try:
            data = response.json()
            choice = data["choices"][0]["message"]
        except (ValueError, KeyError, IndexError) as exc:
            raise BackendError(f"unexpected chat.completions response: {exc}") from exc

        usage = data.get("usage") or {}
        if self.usage is not None:
            self.usage["input_tokens"] += int(usage.get("prompt_tokens") or 0)
            self.usage["output_tokens"] += int(usage.get("completion_tokens") or 0)

        calls: list[ToolCall] = []
        for index, raw in enumerate(choice.get("tool_calls") or [], start=1):
            function = raw.get("function") or {}
            arguments = function.get("arguments") or "{}"
            try:
                args = json.loads(arguments) if isinstance(arguments, str) else dict(arguments)
            except json.JSONDecodeError:
                # A model emitting non-JSON arguments must fail visibly at the
                # validator, not silently become an empty call.
                args = {"_malformed_arguments": str(arguments)}
            if not isinstance(args, dict):
                args = {"_malformed_arguments": str(arguments)}
            calls.append(
                ToolCall(
                    name=str(function.get("name") or ""),
                    args=args,
                    id=str(raw.get("id") or f"call_{index}"),
                )
            )
        return Decision(content=str(choice.get("content") or ""), tool_calls=tuple(calls))
