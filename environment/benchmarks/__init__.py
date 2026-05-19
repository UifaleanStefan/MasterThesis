"""
Stage 3 benchmark adapter registry.

Public surface:

    from environment.benchmarks import get_adapter, ADAPTERS
    adapter = get_adapter("hotpotqa")           # instantiated, lazy-load on first iter
    for doc in adapter.iter_documents(limit=3): ...

All adapters satisfy the ``BenchmarkAdapter`` Protocol defined in
``environment/benchmarks/base.py``.
"""

from __future__ import annotations

from typing import Callable

from .base import (
    BenchmarkAdapter,
    document_fingerprint,
    file_fingerprint,
    validate_document,
)
from .cuad import CUADAdapter
from .financebench import FinanceBenchAdapter
from .hotpotqa import HotpotQAAdapter
from .longmemeval import LongMemEvalAdapter
from .narrativeqa import NarrativeQAAdapter
from .qasper import QASPERAdapter

# Map short name → constructor. Names match `scripts/prefetch_benchmarks.py`
# fetcher keys so the prefetch manifest, verification report, and adapter
# all share one taxonomy.
ADAPTERS: dict[str, Callable[[], BenchmarkAdapter]] = {
    "hotpotqa":     HotpotQAAdapter,
    "qasper":       QASPERAdapter,
    "cuad":         CUADAdapter,
    "narrativeqa":  NarrativeQAAdapter,
    "financebench": FinanceBenchAdapter,
    "longmemeval":  LongMemEvalAdapter,
}


def get_adapter(name: str, **kwargs) -> BenchmarkAdapter:
    """Instantiate the adapter for ``name``.

    Raises ``ValueError`` for unknown names. ``**kwargs`` are forwarded
    to the adapter constructor (e.g. CUAD's ``include_impossible``).
    """
    if name not in ADAPTERS:
        raise ValueError(
            f"Unknown benchmark adapter: {name!r}. "
            f"Available: {sorted(ADAPTERS)}"
        )
    return ADAPTERS[name](**kwargs)


__all__ = [
    "ADAPTERS",
    "BenchmarkAdapter",
    "get_adapter",
    "document_fingerprint",
    "file_fingerprint",
    "validate_document",
]
