"""
Base contract + shared helpers for Stage 3 benchmark adapters.

Each adapter under ``environment/benchmarks/`` translates a real-world
long-context QA benchmark (HotpotQA, QASPER, CUAD, NarrativeQA,
FinanceBench, LongMemEval) into the ``DocumentQA``-compatible document
shape that the rest of the Stage 1/2/3 evaluator code already consumes.

Contract documents:

  Document = {
      "title": str,
      "paragraphs": list[str],
      "qa_pairs": [{
          "question": str,
          "answer": str | list[str],            # list = multi-reference (NarrativeQA)
          "relevant_paragraphs": list[int],     # gold indices into `paragraphs`
      }, ...]
  }

The eval path in ``evaluation/document_qa_memory.py`` treats
``Event.step`` as the paragraph index. Adapters MUST therefore write
paragraphs in the order they will be ingested and never filter,
deduplicate, or reorder paragraphs after building
``relevant_paragraphs`` — index drift silently destroys the recall
metric.

Reproducibility:
  * Each adapter exposes a ``SCHEMA_VERSION`` constant. Bump on any
    conversion-logic change so snapshot tests catch drift.
  * Each adapter exposes ``dataset_fingerprint()`` so the Stage 3
    manifest can record exactly which on-disk data was used.
  * ``document_fingerprint(doc)`` produces a stable SHA256 over a
    canonical JSON dump of a Document — used both by snapshot tests
    and the per-run manifest's ``first_document_fingerprints`` field.
"""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

# Shared on-disk cache layout. Mirrors `scripts/prefetch_benchmarks.py`.
ROOT = Path(__file__).resolve().parent.parent.parent
DATA_DIR = ROOT / "data" / "benchmarks"
HF_CACHE = DATA_DIR / "hf_cache"


# ---------------------------------------------------------------------------
# Adapter Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class BenchmarkAdapter(Protocol):
    """Protocol every per-benchmark adapter must satisfy.

    Subclasses live in ``environment/benchmarks/<name>.py`` and register
    themselves into the ``ADAPTERS`` mapping in ``__init__.py``.
    """

    name: str
    SCHEMA_VERSION: str

    def iter_documents(
        self,
        split: str = "validation",
        limit: int | None = None,
        seed: int = 42,
        shuffle: bool = False,
    ) -> Iterator[dict]:
        """Yield ``Document`` dicts lazily, in deterministic order.

        Parameters
        ----------
        split : str
            Benchmark-specific split name (each adapter documents which
            splits it accepts).
        limit : int | None
            Maximum number of documents to yield. ``None`` = all.
        seed : int
            Used only when ``shuffle=True`` — selects which subset of
            items to yield via ``random.Random(seed).sample(...)``.
        shuffle : bool
            When True, pick a deterministic random subset of ``limit``
            items instead of the first ``limit``.
        """
        ...

    def dataset_fingerprint(self) -> str:
        """Stable SHA256 identifying the on-disk dataset cache.

        For HF datasets: SHA256 over the dataset's download checksums.
        For local JSONs: SHA256 over the canonical file path's contents.
        Used by the Stage 3 manifest to prove which data the run used.
        """
        ...


# ---------------------------------------------------------------------------
# Shared text helpers
# ---------------------------------------------------------------------------


def nfkc_normalize(text: str) -> str:
    """Collapse mojibake, ligatures, and quirky unicode into stable forms.

    Applied before any substring matching (QASPER evidence) and at
    paragraph emission for CUAD / FinanceBench. Idempotent.
    """
    if not text:
        return ""
    return unicodedata.normalize("NFKC", text)


def normalize_for_match(text: str) -> str:
    """Aggressive normalization for fuzzy substring matching.

    NFKC → lowercase → collapse whitespace → strip. Use only on the
    *match key*; never store back into the document.
    """
    return re.sub(r"\s+", " ", nfkc_normalize(text).lower()).strip()


# ---------------------------------------------------------------------------
# Paragraph splitting
# ---------------------------------------------------------------------------


_PARA_SPLIT_RE = re.compile(r"\n\s*\n+")


def paragraph_split_preserving_offsets(text: str) -> list[tuple[str, int, int]]:
    """Split ``text`` on blank-line boundaries, preserving char offsets.

    Returns a list of ``(paragraph_text, start_char, end_char)`` triples
    where ``text[start_char:end_char]`` reconstructs the paragraph (sans
    the consumed blank-line separator). Char offsets are aligned with
    the input string so SQuAD-style ``answer_start`` positions map
    deterministically to paragraph indices via ``offset_to_paragraph_idx``.

    Adapters that depend on offset preservation: CUAD (SQuAD format),
    NarrativeQA (when answer_start hints are added), FinanceBench
    (when full-doc PDFs are eventually wired in).
    """
    if not text:
        return []
    paragraphs: list[tuple[str, int, int]] = []
    cursor = 0
    text_len = len(text)
    for m in _PARA_SPLIT_RE.finditer(text):
        start = cursor
        end = m.start()
        if end > start:
            chunk = text[start:end]
            if chunk.strip():
                paragraphs.append((chunk, start, end))
        cursor = m.end()
    # Tail (after last blank-line boundary, or the whole string if no breaks).
    if cursor < text_len:
        chunk = text[cursor:text_len]
        if chunk.strip():
            paragraphs.append((chunk, cursor, text_len))
    return paragraphs


def offset_to_paragraph_idx(
    char_offset: int, paragraph_ranges: list[tuple[int, int]]
) -> int | None:
    """Return paragraph index containing ``char_offset``, or None.

    ``paragraph_ranges`` is the ``[(start, end), ...]`` projection from
    ``paragraph_split_preserving_offsets``. Linear walk (paragraph counts
    here are at most a few thousand; bisect overhead not worth it).
    """
    for i, (start, end) in enumerate(paragraph_ranges):
        if start <= char_offset < end:
            return i
    return None


_BOILERPLATE_RE = re.compile(
    r"^(chapter|CHAPTER|\d+|\*+|---+|===+|___+|page\s+\d+)\s*$",
    re.IGNORECASE,
)


def is_boilerplate(paragraph: str) -> bool:
    """True for chapter-marker / page-number / divider-only paragraphs.

    Used by NarrativeQA to drop Gutenberg-style boilerplate that would
    otherwise inflate paragraph count without adding retrieval signal.
    """
    stripped = paragraph.strip()
    if not stripped:
        return True
    if len(stripped) < 4:
        return True
    return bool(_BOILERPLATE_RE.match(stripped))


def greedy_merge_paragraphs(
    paragraphs: list[str],
    min_chars: int = 200,
    max_chars: int = 1500,
) -> list[str]:
    """Greedily merge adjacent short paragraphs into ~min..max-char blocks.

    Used by NarrativeQA where naive ``\\n\\n`` splits yield tens of
    thousands of fragments (most under 50 chars). Target distribution:
    median ~300-500 chars per merged paragraph, ceiling at ``max_chars``.

    Boilerplate-only paragraphs are dropped before merging.
    """
    cleaned = [p for p in paragraphs if not is_boilerplate(p)]
    if not cleaned:
        return []

    merged: list[str] = []
    buf: list[str] = []
    buf_len = 0
    for p in cleaned:
        p = p.strip()
        if not p:
            continue
        if buf_len + len(p) + 1 <= max_chars:
            buf.append(p)
            buf_len += len(p) + 1
        else:
            if buf:
                merged.append("\n".join(buf))
            buf = [p]
            buf_len = len(p)
        # Flush when we've crossed min_chars (target shape).
        if buf_len >= min_chars and len(buf) >= 2:
            merged.append("\n".join(buf))
            buf = []
            buf_len = 0
    if buf:
        merged.append("\n".join(buf))
    return merged


# ---------------------------------------------------------------------------
# Evidence-to-paragraph fuzzy matching (QASPER, FinanceBench)
# ---------------------------------------------------------------------------


def evidence_to_paragraph_indices(
    evidence: str, paragraphs: list[str]
) -> list[int]:
    """Map an evidence string to indices of paragraphs containing it.

    Strategy:
      1. Normalize both sides (NFKC + lowercase + whitespace collapse).
      2. Substring match: paragraph contains evidence OR evidence
         contains paragraph (for long evidence spanning multiple paras).
      3. Trigram-overlap fallback: paragraphs sharing ≥ 3 trigrams.

    Returns ``[]`` cleanly when nothing matches — adapters should treat
    "no evidence match" as a legitimate signal, not a bug. The eval path
    falls back to LLM judge when ``relevant_paragraphs`` is empty.
    """
    if not evidence or not paragraphs:
        return []
    ev_norm = normalize_for_match(evidence)
    if len(ev_norm) < 8:
        return []

    indices: list[int] = []
    para_norms = [normalize_for_match(p) for p in paragraphs]

    # Pass 1: exact normalized substring (the common case).
    for i, pn in enumerate(para_norms):
        if not pn:
            continue
        if ev_norm in pn or (len(pn) > 20 and pn in ev_norm):
            indices.append(i)
    if indices:
        return indices

    # Pass 2: trigram-overlap fallback for LaTeX/unicode-corrupted matches.
    ev_trigrams = _trigrams(ev_norm)
    if len(ev_trigrams) < 3:
        return []
    for i, pn in enumerate(para_norms):
        if not pn:
            continue
        overlap = len(ev_trigrams & _trigrams(pn))
        # Require at least 3 shared trigrams AND ≥ 20% Jaccard-style overlap
        # to avoid false positives on long boilerplate paragraphs.
        if overlap >= 3 and overlap >= len(ev_trigrams) * 0.2:
            indices.append(i)
    return indices


def _trigrams(text: str) -> set[str]:
    if len(text) < 3:
        return set()
    return {text[i:i + 3] for i in range(len(text) - 2)}


# ---------------------------------------------------------------------------
# Fingerprinting (reproducibility hooks)
# ---------------------------------------------------------------------------


def document_fingerprint(doc: dict) -> str:
    """Stable SHA256 over a canonical JSON dump of a Document.

    Canonicalization: sorted keys, no whitespace, ensure_ascii=True.
    Used by snapshot tests (fixture lock) and the Stage 3 manifest's
    ``first_document_fingerprints`` field.
    """
    canonical = json.dumps(doc, sort_keys=True, ensure_ascii=True, default=str)
    return "sha256:" + hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def file_fingerprint(path: Path) -> str:
    """SHA256 of a local file's bytes, prefixed with ``sha256:``.

    Used by adapters whose data lives in local JSON (QASPER, CUAD,
    LongMemEval) — recorded once at module import or first access.
    """
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return "sha256:" + h.hexdigest()


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------


def validate_document(doc: dict) -> None:
    """Raise AssertionError if ``doc`` doesn't satisfy the Document contract.

    Called by adapters at yield time as a defense-in-depth check. The
    Layer-1 unit tests assert similar invariants externally — this
    duplication is intentional, so a buggy adapter fails loudly at
    iter_documents time rather than silently producing bad data.
    """
    assert isinstance(doc, dict), f"document must be dict, got {type(doc)}"
    assert "title" in doc and isinstance(doc["title"], str), \
        f"document missing/invalid title: {doc.get('title')!r}"
    assert "paragraphs" in doc and isinstance(doc["paragraphs"], list), \
        "document missing/invalid paragraphs list"
    assert len(doc["paragraphs"]) >= 1, "document has no paragraphs"
    for i, p in enumerate(doc["paragraphs"]):
        assert isinstance(p, str), f"paragraph {i} not a string: {type(p)}"
    assert "qa_pairs" in doc and isinstance(doc["qa_pairs"], list), \
        "document missing/invalid qa_pairs list"
    assert len(doc["qa_pairs"]) >= 1, "document has no qa_pairs"
    n_para = len(doc["paragraphs"])
    for j, qa in enumerate(doc["qa_pairs"]):
        assert isinstance(qa, dict), f"qa_pair {j} not a dict"
        assert "question" in qa and isinstance(qa["question"], str) and qa["question"].strip(), \
            f"qa_pair {j} missing/empty question"
        assert "answer" in qa, f"qa_pair {j} missing answer"
        ans = qa["answer"]
        if isinstance(ans, list):
            assert any(str(a).strip() for a in ans), f"qa_pair {j} all-empty answer list"
        else:
            assert isinstance(ans, str), f"qa_pair {j} answer not str/list: {type(ans)}"
        rel = qa.get("relevant_paragraphs", [])
        assert isinstance(rel, list), f"qa_pair {j} relevant_paragraphs not a list"
        for pidx in rel:
            assert isinstance(pidx, int), f"qa_pair {j} relevant_paragraphs has non-int: {pidx!r}"
            assert 0 <= pidx < n_para, \
                f"qa_pair {j} relevant_paragraphs[{pidx}] out of range [0, {n_para})"
