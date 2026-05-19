"""
CUAD adapter — Contract Understanding Atticus Dataset.

Source: local JSON ``data/benchmarks/cuad/CUAD_v1/CUAD_v1.json`` (SQuAD v2
format, fetched from Zenodo by ``scripts/prefetch_benchmarks.py``).

Per-contract structure:
  {"title": "<CONTRACT_NAME>",
   "paragraphs": [{
       "context": "<huge contract text — single paragraph entry>",
       "qas": [{
           "id": "...",
           "question": "Highlight the parts (if any) of this contract related to '<CLAUSE>' ...",
           "answers": [{"text": "...", "answer_start": <char offset>}, ...],
           "is_impossible": True/False,
       }, ...]  # 41 questions per contract
   }]}

Across 510 contracts × 41 = 20,910 QAs. 6,702 answerable, 14,208
``is_impossible=True`` (by design — most contracts don't address most
clauses).

Adapter conversion:
  * One Document per contract.
  * paragraphs = char-offset-preserving split of ``context`` on blank
    lines. The split helper returns ``(text, start, end)`` triples so
    SQuAD's ``answer_start`` positions map deterministically to
    paragraph indices.
  * qa_pairs = walk ``qas``. Filter ``is_impossible=True`` by default
    (expose ``include_impossible=False`` flag).
  * For each kept QA: take first answer's ``answer_start``, look up
    paragraph index. Multiple-answer cases: combine ``answer_start`` →
    paragraph indices and dedupe (an answer text may legitimately appear
    in multiple paragraphs of the same contract).
"""

from __future__ import annotations

import json
import random
from typing import Iterator

from .base import (
    DATA_DIR,
    file_fingerprint,
    nfkc_normalize,
    offset_to_paragraph_idx,
    paragraph_split_preserving_offsets,
    validate_document,
)

_CUAD_JSON = DATA_DIR / "cuad" / "CUAD_v1" / "CUAD_v1.json"


class CUADAdapter:
    name = "cuad"
    SCHEMA_VERSION = "cuad-v1"

    def __init__(self, include_impossible: bool = False) -> None:
        """
        Parameters
        ----------
        include_impossible : bool, default False
            When False, filter QAs where ``is_impossible=True`` (~68%
            of items). When True, keep them with ``answer="N/A"`` and
            ``relevant_paragraphs=[]`` — useful for testing whether the
            agent correctly says "not addressed" instead of fabricating.
        """
        self.include_impossible = include_impossible
        self._contracts: list[dict] | None = None
        self._fingerprint: str | None = None
        self.last_skipped: int = 0

    def _load(self) -> list[dict]:
        if self._contracts is not None:
            return self._contracts
        if not _CUAD_JSON.exists():
            raise FileNotFoundError(
                f"CUAD data missing: {_CUAD_JSON}. "
                f"Run `python scripts/prefetch_benchmarks.py --only cuad`."
            )
        with _CUAD_JSON.open("r", encoding="utf-8") as f:
            data = json.load(f)
        contracts = data.get("data", [])
        if not contracts:
            raise ValueError("CUAD JSON has no `data` array")
        self._contracts = contracts
        return contracts

    def dataset_fingerprint(self) -> str:
        if self._fingerprint is None:
            self._fingerprint = file_fingerprint(_CUAD_JSON)
        return self._fingerprint

    def _contract_to_document(self, c: dict, idx: int) -> dict | None:
        title = str(c.get("title", "") or "").strip() or f"CUAD #{idx}"
        paras_in = c.get("paragraphs", [])
        if not paras_in:
            return None
        # CUAD ships one huge paragraph entry per contract.
        para0 = paras_in[0]
        context_raw = para0.get("context", "") or ""
        if not context_raw:
            return None
        # IMPORTANT: do NOT NFKC-normalize before the offset-preserving split —
        # NFKC can change character widths and shift offsets relative to
        # ``answer_start``. We split on the raw text and normalize only at
        # storage time below.
        triples = paragraph_split_preserving_offsets(context_raw)
        if not triples:
            return None
        paragraphs = [nfkc_normalize(t) for (t, _, _) in triples]
        para_ranges = [(s, e) for (_, s, e) in triples]

        qa_pairs: list[dict] = []
        for qa in para0.get("qas", []):
            question = str(qa.get("question", "") or "").strip()
            if not question:
                continue
            is_imp = bool(qa.get("is_impossible", False))
            answers = qa.get("answers", []) or []
            if is_imp:
                if not self.include_impossible:
                    continue
                qa_pairs.append({
                    "question": question,
                    "answer": "N/A (clause not addressed in this contract)",
                    "relevant_paragraphs": [],
                })
                continue
            if not answers:
                # Answerable in theory but no `answers` list — skip.
                continue
            # SQuAD style: list of {text, answer_start}. The "first" answer is
            # canonical; additional entries are usually re-statements or
            # appearances of the same clause text elsewhere in the contract.
            answer_text = str(answers[0].get("text", "") or "").strip()
            if not answer_text:
                continue
            # Combine paragraph indices across ALL answer occurrences (dedupe).
            relevant_set: set[int] = set()
            for a in answers:
                start = a.get("answer_start")
                if isinstance(start, int) and start >= 0:
                    pidx = offset_to_paragraph_idx(start, para_ranges)
                    if pidx is not None:
                        relevant_set.add(pidx)
            qa_pairs.append({
                "question": question,
                "answer": answer_text,
                "relevant_paragraphs": sorted(relevant_set),
            })

        if not qa_pairs:
            return None
        return {
            "title": f"CUAD: {title}",
            "paragraphs": paragraphs,
            "qa_pairs": qa_pairs,
        }

    def iter_documents(
        self,
        split: str = "full",  # CUAD ships a single set; flag accepted for API symmetry
        limit: int | None = None,
        seed: int = 42,
        shuffle: bool = False,
    ) -> Iterator[dict]:
        contracts = self._load()
        n = len(contracts)
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
            doc = self._contract_to_document(contracts[idx], idx)
            if doc is None:
                skipped += 1
                continue
            validate_document(doc)
            yielded += 1
            yield doc
        self.last_skipped = skipped
