"""Shared pytest fixtures for thesis invariant tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from memory.event import Event

# --- Skip data-dependent benchmark tests when the corpora aren't on disk -----
# The Stage-3 benchmark adapters need data/benchmarks/ (multi-GB, gitignored,
# fetched via scripts/prefetch_benchmarks.py). On a fresh checkout or in CI that
# directory is absent, so the test_benchmark_* modules can't run. They should be
# SKIPPED there (not failed) and run normally on a developer machine where the
# data is present.
_BENCH_DATA = Path(__file__).resolve().parent.parent / "data" / "benchmarks"


def _benchmark_data_present() -> bool:
    return _BENCH_DATA.is_dir() and any(_BENCH_DATA.iterdir())


def pytest_collection_modifyitems(config, items):
    if _benchmark_data_present():
        return
    skip = pytest.mark.skip(
        reason="benchmark corpora absent (data/benchmarks not prefetched); "
               "run `python scripts/prefetch_benchmarks.py` to enable these tests"
    )
    for item in items:
        if "test_benchmark" in Path(str(item.fspath)).name:
            item.add_marker(skip)


@pytest.fixture
def sample_events() -> list[Event]:
    """A small, varied sequence of events covering colors, doors, and signs."""
    return [
        Event(step=0, observation="you are in a room. you see a red key.", action="pickup"),
        Event(step=1, observation="you are in a room. you see a sign: blue key opens north door.",
              action="move_north", is_hint=True),
        Event(step=2, observation="you are in a room. you see a blue key.", action="pickup"),
        Event(step=3, observation="you are in a room. you see a blue door requires blue key.",
              action="use_door"),
        Event(step=4, observation="you are in a room. you see the goal.", action="move_east"),
    ]


@pytest.fixture
def query_observation() -> str:
    """A retrieval query that should match the door-related hints in sample_events."""
    return "you see a blue door"
