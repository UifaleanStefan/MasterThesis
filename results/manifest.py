"""
Reproducibility manifest builder for result JSONs.

Every result JSON written by a run_*.py script should embed a `manifest` block
so that downstream consumers (thesis writeup, dashboard, future re-runs) can
identify exactly which code, embedding backend, and environment produced the
numbers.

Usage:
    from results.manifest import build_manifest

    payload = {
        "experiment": "graphmemory_v4_cmaes",
        "manifest": build_manifest(seed=42, extra={"n_eval_episodes": 200}),
        "results": {...},
    }
"""

from __future__ import annotations

import datetime as _dt
import os
import platform
import subprocess
import sys
from typing import Any


def _git_sha(short: bool = True) -> str:
    """Return the current git commit SHA, or 'unknown' if not in a repo."""
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "--short" if short else "HEAD", "HEAD"],
            stderr=subprocess.DEVNULL,
        )
        return out.decode().strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def _git_dirty() -> bool:
    """Return True if the working tree has uncommitted changes."""
    try:
        out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        )
        return bool(out.decode().strip())
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False


def _detect_embedding_backend() -> str:
    """
    Detect which embedding backend is active.

    Calls into ``memory.embedding.get_active_backend()`` so the manifest
    reflects what the embed_observation function actually does, not just
    what the EMBEDDING_BACKEND env var says. Important when the env var
    is unset — the default has changed from "tfidf" to
    "sentence-transformers" since Phase 3 / S1.

    Falls back to the raw env var (or "unknown") if the embedding module
    can't be imported (avoids a circular dependency at install time).
    """
    try:
        from memory.embedding import get_active_backend
        return get_active_backend()
    except Exception:
        return os.environ.get("EMBEDDING_BACKEND", "unknown").lower()


def _package_version(name: str) -> str:
    try:
        import importlib.metadata as _md
        return _md.version(name)
    except Exception:
        return "unknown"


def build_manifest(
    seed: int | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """
    Build a reproducibility manifest dict.

    Parameters
    ----------
    seed : int, optional
        The top-level seed used for the experiment. Recorded so downstream
        consumers can verify reproductions hit the same value.
    extra : dict, optional
        Caller-provided fields that are merged into the manifest (e.g., the
        candidate-evaluation episode count, the optimizer's sigma, etc.).

    Returns
    -------
    dict
        A JSON-serializable dict suitable for embedding under a "manifest" key.
    """
    manifest: dict[str, Any] = {
        "git_sha": _git_sha(short=True),
        "git_dirty": _git_dirty(),
        "timestamp_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "python_version": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy_version": _package_version("numpy"),
        "scipy_version": _package_version("scipy"),
        "embedding_backend": _detect_embedding_backend(),
    }
    if seed is not None:
        manifest["seed"] = seed
    if extra:
        manifest.update(extra)
    return manifest
