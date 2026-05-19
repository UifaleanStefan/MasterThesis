"""
NarrativeQA adapter — Kočiský et al. 2018.

Source: HF dataset ``deepmind/narrativeqa`` (cached at
``data/benchmarks/hf_cache/deepmind___narrativeqa/``). 3,461 validation +
10,557 test items; each has a full book or film script as
``document.text`` (up to 1.2 M chars), plus a ``document.summary`` and 2
reference answers per question.

Per-item structure (from HF):
  * ``question``: {"text": str, "tokens": [...]}
  * ``answers``: [{"text": str, "tokens": [...]}, ...]  — typically 2 refs
  * ``document``: {"id": str, "kind": "movie"/"gutenberg", "text": str,
                   "summary": {"text": str, ...}, ...}

Adapter conversion (the careful part):
  * paragraphs[0] = document.summary.text (always a clean ~1K-char
    summary — included so retrieval has a high-level anchor even when
    the long-form body is split aggressively).
  * paragraphs[1+] = greedy_merge_paragraphs over the body. Targets
    200-1500 char chunks (~300-token median). Drops boilerplate
    (chapter markers, page numbers, dividers). Capped at MAX_PARAGRAPHS
    entries — if cap fires, the title gets a ``(truncated)`` suffix.
  * qa_pairs = single entry with ``answer`` as a LIST of the 2 reference
    strings (NarrativeQA's multi-reference convention).
  * relevant_paragraphs = [] — NarrativeQA provides no paragraph-level
    gold; eval falls back to LLM-judge answer-quality only.

Why this design: a naive ``\\n\\n`` split on a 1.2 M-char book yields
tens of thousands of fragments (most under 50 chars — single-word lines,
chapter headers, dialogue line-breaks). That blows up V4 ingestion cost
and adds noise without retrieval signal. The greedy-merge + cap keeps
the effective paragraph count manageable while preserving local context.
"""

from __future__ import annotations

import os
import random
import re
from typing import Iterator

from .base import (
    HF_CACHE,
    greedy_merge_paragraphs,
    nfkc_normalize,
    validate_document,
)

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")

# Hard cap on paragraphs per document. Tuned to keep V4 ingestion under
# ~30 seconds per book and the prompt-cost per question bounded.
MAX_PARAGRAPHS = 2000

# Strip common Project Gutenberg headers/footers (NarrativeQA pulls from
# Project Gutenberg for many of its books).
_GUTENBERG_HEADER_RE = re.compile(
    r"\*{3}\s*START OF .*?PROJECT GUTENBERG.*?\*{3}", re.IGNORECASE | re.DOTALL
)
_GUTENBERG_FOOTER_RE = re.compile(
    r"\*{3}\s*END OF .*?PROJECT GUTENBERG.*?\*{3}", re.IGNORECASE | re.DOTALL
)


def _strip_gutenberg(text: str) -> str:
    """Cut everything before the START marker and after the END marker.

    If markers aren't found, the text is returned unchanged. Defensive
    against missing-marker edge cases (some books predate the marker
    convention or have it slightly differently spelled).
    """
    if not text:
        return text
    m_start = _GUTENBERG_HEADER_RE.search(text)
    if m_start:
        text = text[m_start.end():]
    m_end = _GUTENBERG_FOOTER_RE.search(text)
    if m_end:
        text = text[: m_end.start()]
    return text


class NarrativeQAAdapter:
    name = "narrativeqa"
    SCHEMA_VERSION = "narrativeqa-v1"
    HF_ID = "deepmind/narrativeqa"

    def __init__(self) -> None:
        self._dataset = None
        self._dataset_split: str | None = None
        self._fingerprint: str | None = None
        self.last_skipped: int = 0
        self.last_truncated: int = 0

    def _load(self, split: str):
        from datasets import load_dataset
        ds = load_dataset(self.HF_ID, cache_dir=str(HF_CACHE))
        if split not in ds:
            raise ValueError(f"Unknown split {split!r}; available: {list(ds.keys())}")
        self._dataset = ds[split]
        self._dataset_split = split
        return self._dataset

    def dataset_fingerprint(self) -> str:
        if self._fingerprint is not None:
            return self._fingerprint
        ds = self._load("validation")
        info = getattr(ds, "info", None)
        if info is not None and getattr(info, "download_checksums", None):
            import hashlib
            import json
            blob = json.dumps(info.download_checksums, sort_keys=True)
            self._fingerprint = "sha256:" + hashlib.sha256(blob.encode()).hexdigest()
        else:
            import hashlib
            sig = f"narrativeqa:n={len(ds)}"
            self._fingerprint = "sha256:" + hashlib.sha256(sig.encode()).hexdigest()
        return self._fingerprint

    def _item_to_document(self, item: dict, idx: int) -> dict | None:
        # The HF NarrativeQA schema uses nested dicts for question + document.
        question_obj = item.get("question") or {}
        question = str(question_obj.get("text", "") or "").strip()
        if not question:
            return None

        document_obj = item.get("document") or {}
        body = document_obj.get("text", "") or ""
        if not body:
            return None
        body = _strip_gutenberg(body)
        body = nfkc_normalize(body)

        # Summary as the high-level anchor paragraph[0]. HF stores summary as
        # a nested object {"text": "...", "tokens": [...]}.
        summary_obj = document_obj.get("summary") or {}
        summary_text = ""
        if isinstance(summary_obj, dict):
            summary_text = str(summary_obj.get("text", "") or "").strip()
        elif isinstance(summary_obj, str):
            summary_text = summary_obj.strip()
        summary_text = nfkc_normalize(summary_text)

        # Split body on blank-line boundaries; greedy-merge into 200-1500 char chunks.
        raw_paragraphs = [p for p in body.split("\n\n") if p.strip()]
        merged = greedy_merge_paragraphs(raw_paragraphs, min_chars=200, max_chars=1500)

        # Cap and record truncation status.
        truncated = False
        if len(merged) > MAX_PARAGRAPHS:
            merged = merged[:MAX_PARAGRAPHS]
            truncated = True

        paragraphs: list[str] = []
        if summary_text:
            paragraphs.append(summary_text)
        paragraphs.extend(merged)
        if not paragraphs:
            return None

        # Multi-reference answers: NarrativeQA gives 2 per question.
        ref_answers: list[str] = []
        for a in item.get("answers", []) or []:
            if isinstance(a, dict):
                t = a.get("text", "")
            elif isinstance(a, str):
                t = a
            else:
                t = str(a)
            t = str(t).strip()
            if t:
                ref_answers.append(t)
        if not ref_answers:
            return None

        title_base = "NarrativeQA"
        doc_id = document_obj.get("id", "")
        kind = document_obj.get("kind", "")
        title = f"{title_base} {kind} {doc_id} #{idx}".strip()
        if truncated:
            title = f"{title} (truncated to {MAX_PARAGRAPHS} paragraphs)"

        qa_pairs = [{
            "question": question,
            "answer": ref_answers,             # LIST — multi-reference
            "relevant_paragraphs": [],         # no gold relevance for NarrativeQA
        }]

        return {
            "title": title,
            "paragraphs": paragraphs,
            "qa_pairs": qa_pairs,
        }

    def iter_documents(
        self,
        split: str = "validation",
        limit: int | None = None,
        seed: int = 42,
        shuffle: bool = False,
    ) -> Iterator[dict]:
        ds = self._load(split)
        n = len(ds)
        if shuffle:
            rng = random.Random(seed)
            order = rng.sample(range(n), n)
        else:
            order = list(range(n))

        yielded = 0
        skipped = 0
        truncated = 0
        for idx in order:
            if limit is not None and yielded >= limit:
                break
            doc = self._item_to_document(ds[idx], idx)
            if doc is None:
                skipped += 1
                continue
            if "truncated" in doc.get("title", ""):
                truncated += 1
            validate_document(doc)
            yielded += 1
            yield doc
        self.last_skipped = skipped
        self.last_truncated = truncated
