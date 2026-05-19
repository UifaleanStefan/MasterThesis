"""
FinanceBench adapter — Patronus AI's 150-question SEC-filings QA benchmark.

Source: HF dataset ``PatronusAI/financebench`` (cached at
``data/benchmarks/hf_cache/PatronusAI___financebench/``).

Per-item structure (confirmed via ``scripts/verify_benchmarks.py``):
  * ``question``: str (always present)
  * ``answer``: str (always present)
  * ``evidence``: list[dict] OR string. List items are
    ``{evidence_text, doc_name, evidence_page_num, ...}``.
  * Other metadata: company, doc_type ("10k"/"10q"/etc), question_type,
    domain_question_num, justification, etc.

Adapter conversion:
  * One Document per benchmark item.
  * paragraphs = extracted ``evidence_text`` strings (NFKC normalized).
  * qa_pairs = single entry. ``relevant_paragraphs = [0..n)`` — by
    construction the evidence list IS the gold-relevance set.
  * Items where evidence yields zero paragraphs are SKIPPED (counted +
    reported in ``last_skipped`` after iteration).
"""

from __future__ import annotations

import os
import random
from typing import Any, Iterator

from .base import (
    HF_CACHE,
    document_fingerprint,
    nfkc_normalize,
    validate_document,
)

# Force HF offline; the prefetch already cached this dataset.
os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class FinanceBenchAdapter:
    name = "financebench"
    SCHEMA_VERSION = "financebench-v1"
    HF_ID = "PatronusAI/financebench"

    def __init__(self) -> None:
        self._dataset = None  # lazy-loaded
        self._fingerprint: str | None = None
        self.last_skipped: int = 0

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load(self):
        if self._dataset is not None:
            return self._dataset
        from datasets import load_dataset
        ds = load_dataset(self.HF_ID, cache_dir=str(HF_CACHE))
        # FinanceBench ships a single split ("train") containing the 150
        # canonical public eval items.
        self._dataset = next(iter(ds.values()))
        return self._dataset

    def dataset_fingerprint(self) -> str:
        if self._fingerprint is not None:
            return self._fingerprint
        ds = self._load()
        # Use dataset info's download_checksums if available, else N + first/last keys.
        info = getattr(ds, "info", None)
        if info is not None and getattr(info, "download_checksums", None):
            import hashlib
            import json
            checksum_blob = json.dumps(info.download_checksums, sort_keys=True)
            self._fingerprint = "sha256:" + hashlib.sha256(checksum_blob.encode()).hexdigest()
        else:
            # Fallback: hash N + first item's ID + last item's ID.
            import hashlib
            n = len(ds)
            first_id = str(ds[0].get("financebench_id", "0"))
            last_id = str(ds[n - 1].get("financebench_id", str(n - 1)))
            sig = f"financebench:n={n}:first={first_id}:last={last_id}"
            self._fingerprint = "sha256:" + hashlib.sha256(sig.encode()).hexdigest()
        return self._fingerprint

    # ------------------------------------------------------------------
    # Per-item conversion
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_evidence(ev: Any) -> list[str]:
        """Branch on FinanceBench's variable ``evidence`` shape.

        Returns a list of evidence-text paragraphs (NFKC-normalized).
        Empty / whitespace-only entries are dropped.
        """
        out: list[str] = []
        if ev is None:
            return out
        if isinstance(ev, list):
            for item in ev:
                if isinstance(item, dict):
                    text = item.get("evidence_text", "") or item.get("text", "") or ""
                elif isinstance(item, str):
                    text = item
                else:
                    text = str(item)
                text = nfkc_normalize(text).strip()
                if text:
                    out.append(text)
        elif isinstance(ev, str):
            text = nfkc_normalize(ev).strip()
            if text:
                out.append(text)
        # Other shapes silently produce empty list (the item is skipped).
        return out

    def _item_to_document(self, item: dict, idx: int) -> dict | None:
        question = str(item.get("question", "") or "").strip()
        answer = str(item.get("answer", "") or "").strip()
        evidence = self._extract_evidence(item.get("evidence"))
        if not question or not answer or not evidence:
            return None
        company = str(item.get("company", "") or "").strip()
        doc_type = str(item.get("doc_type", "") or "").strip()
        period = str(item.get("doc_period", "") or "").strip()
        title_parts = [p for p in [company, doc_type, period] if p]
        title = " ".join(title_parts) if title_parts else f"FinanceBench #{idx}"

        qa_pairs = [{
            "question": question,
            "answer": answer,
            # By construction the evidence list IS the gold-relevance set.
            "relevant_paragraphs": list(range(len(evidence))),
        }]
        return {
            "title": title,
            "paragraphs": evidence,
            "qa_pairs": qa_pairs,
        }

    # ------------------------------------------------------------------
    # Public iteration
    # ------------------------------------------------------------------

    def iter_documents(
        self,
        split: str = "train",  # only one split exists; ignored
        limit: int | None = None,
        seed: int = 42,
        shuffle: bool = False,
    ) -> Iterator[dict]:
        ds = self._load()
        n = len(ds)
        if shuffle:
            rng = random.Random(seed)
            order = rng.sample(range(n), n)
        else:
            order = list(range(n))
        if limit is not None:
            # Walk in-order until we've yielded `limit` non-skipped docs.
            pass

        yielded = 0
        skipped = 0
        for idx in order:
            if limit is not None and yielded >= limit:
                break
            doc = self._item_to_document(ds[idx], idx)
            if doc is None:
                skipped += 1
                continue
            validate_document(doc)
            yielded += 1
            yield doc
        self.last_skipped = skipped
