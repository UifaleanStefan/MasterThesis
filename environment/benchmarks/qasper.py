"""
QASPER adapter — QA over 1,585 NLP papers.

Source: local JSON files under ``data/benchmarks/qasper/`` (fetched
directly from AI2's S3 bucket by ``scripts/prefetch_benchmarks.py``):
  * ``qasper-dev-v0.3.json``   — 281 papers, 1,005 questions
  * ``qasper-test-v0.3.json``  — 416 papers, 1,451 questions
  * ``qasper-train-v0.3.json`` — 888 papers, 2,593 questions

Top-level JSON shape:
  {paper_id: {
      "title": "...",
      "abstract": "...",
      "full_text": [
          {"section_name": "...", "paragraphs": ["...", "..."]},
          ...
      ],
      "qas": [
          {"question": "...", "question_id": "...",
           "answers": [{"answer": {
               "free_form_answer": "...",
               "extractive_spans": ["...", "..."],
               "yes_no": True/False/None,
               "unanswerable": True/False,
               "evidence": ["evidence text 1", "evidence text 2", ...]
           }}, ...]
          }, ...
      ]
   }, ...}

Adapter conversion:
  * One Document per paper.
  * paragraphs[0] = title + abstract (concatenated). The title helps
    retrieval find topic-level matches.
  * paragraphs[1+] = flatten of full_text. Each section contributes one
    paragraph for the section_name (so retrieval can hit by topic) and
    then one per paragraph in that section.
  * qa_pairs: one per ``qas`` entry. Answer fallback order:
    ``yes_no`` → ``extractive_spans[0]`` → ``free_form_answer`` → "(no answer)".
    Plan-agent's recommended order: yes_no is the most reliable signal.
  * relevant_paragraphs: substring-match each evidence string against
    the paragraphs (NFKC-lowercase-whitespace-collapse normalization).
    Filter the literal "FLOAT_TYPE_NONEVIDENCE" sentinel. Empty matches
    are kept as empty lists — eval falls back to LLM judge.
"""

from __future__ import annotations

import json
import random
from typing import Iterator

from .base import (
    DATA_DIR,
    evidence_to_paragraph_indices,
    file_fingerprint,
    nfkc_normalize,
    validate_document,
)

_QASPER_DIR = DATA_DIR / "qasper"
_SPLIT_FILES = {
    "validation": _QASPER_DIR / "qasper-dev-v0.3.json",
    "dev":        _QASPER_DIR / "qasper-dev-v0.3.json",
    "test":       _QASPER_DIR / "qasper-test-v0.3.json",
    "train":      _QASPER_DIR / "qasper-train-v0.3.json",
}

_NONEVIDENCE_SENTINEL = "FLOAT_TYPE_NONEVIDENCE"


class QASPERAdapter:
    name = "qasper"
    SCHEMA_VERSION = "qasper-v1"

    def __init__(self) -> None:
        self._cache: dict[str, dict] = {}
        self._fingerprints: dict[str, str] = {}
        self.last_skipped: int = 0

    def _load(self, split: str) -> dict:
        if split in self._cache:
            return self._cache[split]
        path = _SPLIT_FILES.get(split)
        if path is None:
            raise ValueError(
                f"Unknown QASPER split {split!r}; available: {list(_SPLIT_FILES)}"
            )
        if not path.exists():
            raise FileNotFoundError(
                f"QASPER data missing: {path}. "
                f"Run `python scripts/prefetch_benchmarks.py --only qasper`."
            )
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            raise TypeError(f"Expected dict at top level of {path.name}, got {type(data)}")
        self._cache[split] = data
        return data

    def dataset_fingerprint(self, split: str = "validation") -> str:
        if split in self._fingerprints:
            return self._fingerprints[split]
        path = _SPLIT_FILES.get(split)
        if path is None or not path.exists():
            raise FileNotFoundError(f"QASPER split {split} not on disk")
        fp = file_fingerprint(path)
        self._fingerprints[split] = fp
        return fp

    @staticmethod
    def _extract_answer(answer_obj: dict) -> str:
        """Pick the most-reliable string answer from a QASPER answer dict.

        Order: yes_no → extractive_spans[0] → free_form_answer →
        ``"(no answer)"``. Returns empty string if everything is empty/None.
        """
        if not isinstance(answer_obj, dict):
            return ""
        # yes_no is `True`/`False`/`None`.
        yn = answer_obj.get("yes_no")
        if yn is True:
            return "Yes"
        if yn is False:
            return "No"
        spans = answer_obj.get("extractive_spans", [])
        if isinstance(spans, list):
            for s in spans:
                if isinstance(s, str) and s.strip():
                    return s.strip()
        ff = answer_obj.get("free_form_answer", "")
        if isinstance(ff, str) and ff.strip():
            return ff.strip()
        unanswerable = answer_obj.get("unanswerable", False)
        if unanswerable:
            return "(unanswerable per source)"
        return ""

    @staticmethod
    def _flatten_paragraphs(paper: dict) -> list[str]:
        """Title + abstract first, then walk full_text sections."""
        title = str(paper.get("title", "") or "").strip()
        abstract = str(paper.get("abstract", "") or "").strip()
        head = nfkc_normalize(f"{title}\n\n{abstract}".strip())
        paragraphs: list[str] = [head] if head else []

        for section in paper.get("full_text", []) or []:
            section_name = str(section.get("section_name", "") or "").strip()
            if section_name:
                paragraphs.append(nfkc_normalize(section_name))
            for p in section.get("paragraphs", []) or []:
                if not isinstance(p, str):
                    p = str(p)
                p_norm = nfkc_normalize(p).strip()
                if p_norm:
                    paragraphs.append(p_norm)
        return paragraphs

    def _paper_to_document(self, paper_id: str, paper: dict) -> dict | None:
        title = str(paper.get("title", "") or "").strip() or paper_id
        paragraphs = self._flatten_paragraphs(paper)
        if not paragraphs:
            return None

        qa_pairs: list[dict] = []
        for qa in paper.get("qas", []) or []:
            question = str(qa.get("question", "") or "").strip()
            if not question:
                continue
            # qa.answers is a list of {"answer": {...}} dicts (often 2-3 human
            # annotators). Pick the first non-empty answer.
            answer_str = ""
            relevant_set: set[int] = set()
            for ans_wrap in qa.get("answers", []) or []:
                answer_obj = ans_wrap.get("answer", ans_wrap)  # tolerate both shapes
                cand = self._extract_answer(answer_obj)
                if cand and not answer_str:
                    answer_str = cand
                # Evidence collection from EVERY annotator (broader coverage).
                evidence_list = answer_obj.get("evidence", []) or []
                for ev in evidence_list:
                    if not isinstance(ev, str):
                        continue
                    if _NONEVIDENCE_SENTINEL in ev:
                        continue
                    hits = evidence_to_paragraph_indices(ev, paragraphs)
                    relevant_set.update(hits)
            if not answer_str:
                # All annotators marked unanswerable / left empty — skip.
                continue
            qa_pairs.append({
                "question": question,
                "answer": answer_str,
                "relevant_paragraphs": sorted(relevant_set),
            })

        if not qa_pairs:
            return None
        return {
            "title": f"QASPER: {title[:100]}",
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
        data = self._load(split)
        # Top-level is a dict {paper_id: paper}; iterate keys in sorted order
        # for determinism (Python 3.7+ preserves insertion order but dataset
        # files don't guarantee key ordering on the wire).
        paper_ids = sorted(data.keys())
        n = len(paper_ids)
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
            pid = paper_ids[idx]
            doc = self._paper_to_document(pid, data[pid])
            if doc is None:
                skipped += 1
                continue
            validate_document(doc)
            yielded += 1
            yield doc
        self.last_skipped = skipped
