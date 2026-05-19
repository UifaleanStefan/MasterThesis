"""
HotpotQA adapter — Wikipedia-based multi-hop QA, distractor config.

Source: HF dataset ``hotpotqa/hotpot_qa`` (cached at
``data/benchmarks/hf_cache/hotpotqa___hotpot_qa/distractor/``).

Per-item structure (confirmed via verifier):
  * ``question``: str
  * ``answer``: str
  * ``context``: parallel-arrays dict
      {"title": [t_0..t_9], "sentences": [[s,...], ...]}
    10 passages, 2 gold + 8 distractor.
  * ``supporting_facts``: parallel-arrays dict
      {"title": [...], "sent_id": [...]}
    Per-sentence gold pointers — we coarsen to passage-level for the
    Document contract (paragraph-level relevance).

Adapter conversion:
  * One Document per HotpotQA validation item.
  * paragraphs[i] = "{title}\\n{joined sentences}" for i in 0..9.
    Title prepended so retrieval embeddings see the article name —
    this matters because some questions reference Wikipedia titles
    directly (e.g. "Who founded the company that owns Y?").
  * qa_pairs = single entry with question + answer.
  * relevant_paragraphs = passage indices whose title appears in
    supporting_facts.title. Duplicate titles map to last-wins (rare in
    distractor; duplicates are by definition the same article).
"""

from __future__ import annotations

import os
import random
from typing import Iterator

from .base import HF_CACHE, document_fingerprint, validate_document

os.environ.setdefault("HF_DATASETS_OFFLINE", "1")
os.environ.setdefault("HF_HUB_OFFLINE", "1")


class HotpotQAAdapter:
    name = "hotpotqa"
    SCHEMA_VERSION = "hotpotqa-v1"
    HF_ID = "hotpotqa/hotpot_qa"
    HF_CONFIG = "distractor"

    def __init__(self) -> None:
        self._dataset = None
        self._fingerprint: str | None = None
        self.last_skipped: int = 0

    def _load(self, split: str):
        from datasets import load_dataset
        ds = load_dataset(self.HF_ID, self.HF_CONFIG, cache_dir=str(HF_CACHE))
        if split not in ds:
            raise ValueError(f"Unknown split {split!r}; available: {list(ds.keys())}")
        self._dataset = ds[split]
        return self._dataset

    def dataset_fingerprint(self) -> str:
        if self._fingerprint is not None:
            return self._fingerprint
        # Load val for the fingerprint; HF caches the underlying arrow files
        # the same way regardless of which split we ask first.
        ds = self._load("validation")
        info = getattr(ds, "info", None)
        if info is not None and getattr(info, "download_checksums", None):
            import hashlib
            import json
            blob = json.dumps(info.download_checksums, sort_keys=True)
            self._fingerprint = "sha256:" + hashlib.sha256(blob.encode()).hexdigest()
        else:
            import hashlib
            sig = f"hotpotqa:n={len(ds)}:first_id={ds[0].get('id', '')}"
            self._fingerprint = "sha256:" + hashlib.sha256(sig.encode()).hexdigest()
        return self._fingerprint

    @staticmethod
    def _item_to_document(item: dict, idx: int) -> dict | None:
        question = str(item.get("question", "") or "").strip()
        answer = str(item.get("answer", "") or "").strip()
        ctx = item.get("context") or {}
        titles = list(ctx.get("title", []) or [])
        sentences = list(ctx.get("sentences", []) or [])
        if not question or not answer or len(titles) == 0 or len(sentences) == 0:
            return None
        if len(titles) != len(sentences):
            # Malformed item — drop.
            return None

        # Build paragraphs in passage order.
        paragraphs: list[str] = []
        title_to_index: dict[str, int] = {}
        for i, (title, sents) in enumerate(zip(titles, sentences)):
            body = "".join(sents) if isinstance(sents, list) else str(sents)
            paragraph = f"{title}\n{body}".strip()
            paragraphs.append(paragraph)
            # Last-wins on duplicate titles (rare; duplicates are the same article).
            title_to_index[title] = i

        # Supporting facts → passage-level relevance.
        sf = item.get("supporting_facts") or {}
        sf_titles = sf.get("title", []) or []
        relevant_set: set[int] = set()
        for sft in sf_titles:
            pidx = title_to_index.get(sft)
            if pidx is not None:
                relevant_set.add(pidx)
        relevant_paragraphs = sorted(relevant_set)

        # Items with empty supporting_facts are rare but valid in the dataset;
        # we keep them and let downstream LLM-judge handle the answer-quality
        # check (retrieval recall@k would just be 0 for these).
        qa_pairs = [{
            "question": question,
            "answer": answer,
            "relevant_paragraphs": relevant_paragraphs,
        }]
        return {
            "title": f"HotpotQA dev #{idx}: {item.get('id', '')}".strip(),
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
