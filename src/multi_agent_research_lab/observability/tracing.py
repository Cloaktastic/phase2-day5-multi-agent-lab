"""Tracing hooks.

This file intentionally avoids binding to one provider. Students can plug in LangSmith,
Langfuse, OpenTelemetry, or simple JSON traces.
"""

from collections.abc import Iterator
from contextlib import contextmanager
from time import perf_counter
from typing import Any

import os
from dotenv import load_dotenv
load_dotenv()

from langfuse.langchain import CallbackHandler

def get_langfuse_handler():
    load_dotenv()
    if os.getenv("LANGFUSE_PUBLIC_KEY"):
        return CallbackHandler()
    return None

@contextmanager
def trace_span(name: str, attributes: dict[str, Any] | None = None) -> Iterator[dict[str, Any]]:
    """Minimal span context used by the skeleton.

    TODO(student): Replace or augment with LangSmith/Langfuse provider spans.
    """

    started = perf_counter()
    span: dict[str, Any] = {"name": name, "attributes": attributes or {}, "duration_seconds": None}
    try:
        yield span
    finally:
        span["duration_seconds"] = perf_counter() - started

