"""
Layer 0 — Snapshot tests for the six Stage 3 benchmark adapters.

For each adapter, lock the first 3 documents (deterministic seed=42, no
shuffle) to a committed fixture under ``tests/fixtures/``. The test
asserts the per-document SHA256 fingerprints don't drift across runs.

Catches:
  * HF dataset version drift (upstream re-publish breaks parquet files).
  * Unintended adapter logic changes (typo in paragraph join, regex tweak).
  * Local file corruption (cached JSON edits, etc).

Fixture management:
  * To regenerate after an intentional adapter change:
        python -m tests.test_benchmark_snapshots --regenerate
    (after bumping SCHEMA_VERSION in the affected adapter).
  * Manual diff of fixtures should be readable JSON for spot-debugging.

Fixtures store only:
  * "schema_version": adapter.SCHEMA_VERSION
  * "n_documents": 3
  * "document_fingerprints": [sha256:..., sha256:..., sha256:...]
  * "document_previews": [{title, n_paragraphs, n_qa_pairs, first_question}, ...]

The full Documents are NOT committed (some are MB-sized — NarrativeQA's
first val item has 2,000 paragraphs). Fingerprints + previews are enough
to catch drift while keeping the fixtures human-scannable.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import pytest

os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"

from environment.benchmarks import ADAPTERS, document_fingerprint, get_adapter

FIXTURE_DIR = Path(__file__).resolve().parent / "fixtures"
FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

SNAPSHOT_LIMIT = 3
ALL_NAMES = sorted(ADAPTERS.keys())


def _doc_preview(doc: dict) -> dict:
    """Compact, human-scannable summary of a Document for the fixture file."""
    first_qa = doc["qa_pairs"][0] if doc["qa_pairs"] else {}
    return {
        "title": doc.get("title", ""),
        "n_paragraphs": len(doc.get("paragraphs", [])),
        "n_qa_pairs": len(doc.get("qa_pairs", [])),
        "first_question": str(first_qa.get("question", ""))[:200],
        "first_answer_type": (
            "list" if isinstance(first_qa.get("answer"), list) else "str"
        ),
        "first_relevant_count": len(first_qa.get("relevant_paragraphs", [])),
        "paragraph_lengths_sample": [
            len(p) for p in doc.get("paragraphs", [])[:5]
        ],
    }


def _build_snapshot(name: str) -> dict:
    adapter = get_adapter(name)
    docs = list(adapter.iter_documents(limit=SNAPSHOT_LIMIT))
    return {
        "schema_version": adapter.SCHEMA_VERSION,
        "n_documents": len(docs),
        "document_fingerprints": [document_fingerprint(d) for d in docs],
        "document_previews": [_doc_preview(d) for d in docs],
    }


def _fixture_path(name: str) -> Path:
    return FIXTURE_DIR / f"{name}_snapshot.json"


def _regenerate_all() -> None:
    """Build + write fixtures for all six adapters. Used in CLI mode."""
    print(f"Regenerating snapshots under {FIXTURE_DIR}/")
    for name in ALL_NAMES:
        path = _fixture_path(name)
        snap = _build_snapshot(name)
        path.write_text(json.dumps(snap, indent=2))
        print(f"  [OK] {name}: {len(snap['document_fingerprints'])} docs → {path.name}")


@pytest.fixture(scope="module", autouse=True)
def _ensure_fixtures_exist():
    """Build any missing fixtures lazily so first-run is one-shot.

    After the first pytest invocation, fixtures are committed and the
    test is a pure drift check. Idempotent — never overwrites existing
    fixtures (use the CLI regenerate path for that).
    """
    for name in ALL_NAMES:
        path = _fixture_path(name)
        if not path.exists():
            snap = _build_snapshot(name)
            path.write_text(json.dumps(snap, indent=2))
            print(f"\n[snapshots] bootstrapped {path.name}")


@pytest.mark.parametrize("name", ALL_NAMES)
def test_snapshot_matches(name: str) -> None:
    """Snapshot fingerprints match the committed fixture.

    A failure here means the adapter's output for the first 3 documents
    changed. To diagnose:
      * If intentional (bug fix, SCHEMA_VERSION bumped), regenerate via:
            python -m tests.test_benchmark_snapshots --regenerate
      * If unintentional, the failing fingerprints + previews in stderr
        show what changed (paragraph count, relevant_paragraphs counts,
        first_question text, etc).
    """
    path = _fixture_path(name)
    assert path.exists(), f"[{name}] fixture missing: {path}"
    fixture = json.loads(path.read_text())
    fresh = _build_snapshot(name)

    # Compare schema_version first (most diagnostic mismatch).
    assert fixture["schema_version"] == fresh["schema_version"], (
        f"[{name}] SCHEMA_VERSION drifted "
        f"({fixture['schema_version']} → {fresh['schema_version']}). "
        f"Regenerate fixture if intentional."
    )

    # Compare counts.
    assert fixture["n_documents"] == fresh["n_documents"], (
        f"[{name}] doc count changed "
        f"({fixture['n_documents']} → {fresh['n_documents']})"
    )

    # Compare fingerprints. Print previews diff on failure for human-readable diagnosis.
    if fixture["document_fingerprints"] != fresh["document_fingerprints"]:
        diff_lines = [f"[{name}] document fingerprint drift:"]
        for i, (old, new) in enumerate(
            zip(fixture["document_fingerprints"], fresh["document_fingerprints"])
        ):
            if old == new:
                continue
            diff_lines.append(f"  doc[{i}]: {old[:24]}... → {new[:24]}...")
            old_pre = fixture["document_previews"][i] if i < len(fixture["document_previews"]) else {}
            new_pre = fresh["document_previews"][i] if i < len(fresh["document_previews"]) else {}
            for key in sorted(set(old_pre) | set(new_pre)):
                if old_pre.get(key) != new_pre.get(key):
                    diff_lines.append(
                        f"           {key}: {old_pre.get(key)!r} -> {new_pre.get(key)!r}"
                    )
        pytest.fail("\n".join(diff_lines))


if __name__ == "__main__":
    if "--regenerate" in sys.argv:
        _regenerate_all()
    else:
        print("Run via pytest. Use --regenerate to rewrite fixtures.")
