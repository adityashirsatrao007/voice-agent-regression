"""Backend factory.

Imports are lazy on purpose: `import vareg` must work with no `requests` and
no `langgraph` installed, because the default (mock) path is stdlib-only.
"""

from __future__ import annotations

from typing import Sequence

from ..prompts import PromptSpec
from ..registry import ToolRegistry
from .base import Backend, BackendError

BACKEND_NAMES: tuple[str, ...] = ("mock", "llm", "langgraph")


def make_backend(name: str, spec: PromptSpec, registry: ToolRegistry) -> Backend:
    if name == "mock":
        from .mock import MockBackend

        return MockBackend(spec, registry)
    if name == "llm":
        from .llm import LLMBackend

        return LLMBackend(spec, registry)
    if name == "langgraph":
        from .langgraph_backend import LangGraphBackend

        return LangGraphBackend(spec, registry)
    raise BackendError(f"unknown backend {name!r}; choose from {', '.join(BACKEND_NAMES)}")


def langgraph_available() -> bool:
    """Cheap probe used by `list-scenarios --backends` and the test suite."""
    try:
        import langgraph  # noqa: F401
    except ImportError:
        return False
    return True
