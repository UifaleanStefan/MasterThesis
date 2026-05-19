"""
LongMemEval adapter — Wu et al. 2024 (ICLR 2025).

Source: local JSON files under ``data/benchmarks/longmemeval/`` (fetched
via ``scripts/prefetch_benchmarks.py`` from
``xiaowu0162/longmemeval-cleaned``):
  * ``longmemeval_oracle.json``        — 500 items, 15 MB. Canonical eval set.
  * ``longmemeval_s_cleaned.json``     — 500 items, 264 MB. Longer haystacks.
  * ``longmemeval_m_cleaned.json``     — 500 items, 2.5 GB. Streaming required.

Per-item structure:
  * ``question``: str
  * ``answer``: str (may be the literal "(no answer in haystack)" sentinel)
  * ``question_type``: str (single-session-user / temporal-reasoning / etc)
  * ``question_date``: str
  * ``haystack_sessions``: list[list[dict]] — each session is a list of
    {role, content} messages.
  * ``haystack_session_ids``: list[str] — PARALLEL with haystack_sessions
    (i-th id corresponds to i-th session).
  * ``haystack_dates``: list[str] — PARALLEL with haystack_sessions.
  * ``answer_session_ids``: list[str] — gold session IDs that contain the
    answer.

Adapter conversion:
  * One Document per oracle item.
  * paragraphs[i] = "[Session {sid} on {date}]\\nrole: content\\n..."
  * qa_pairs = single entry. relevant_paragraphs = indices where
    haystack_session_ids[i] is in set(answer_session_ids).
  * Asserts ``len(haystack_sessions) == len(haystack_session_ids)`` —
    a critical invariant. If violated, the item is skipped.

Deferred:
  * m_cleaned (2.5 GB) — eager parse hits MemoryError. Adapter
    currently supports only oracle and s_cleaned. Streaming via ijson
    is a follow-up task once the rest of the pipeline is exercised.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Iterator

from .base import DATA_DIR, file_fingerprint, validate_document

_LME_DIR = DATA_DIR / "longmemeval"

# Map "split" parameter to file name. Only the eager-loadable variants are
# supported in v1 of this adapter; m_cleaned needs ijson streaming.
_SPLIT_FILES = {
    "oracle":     _LME_DIR / "longmemeval_oracle.json",
    "s_cleaned":  _LME_DIR / "longmemeval_s_cleaned.json",
    # "m_cleaned": _LME_DIR / "longmemeval_m_cleaned.json",  # deferred
}


class LongMemEvalAdapter:
    name = "longmemeval"
    SCHEMA_VERSION = "longmemeval-v1"

    def __init__(self) -> None:
        self._cache: dict[str, list] = {}
        self._fingerprints: dict[str, str] = {}
        self.last_skipped: int = 0

    def _load(self, split: str) -> list:
        if split in self._cache:
            return self._cache[split]
        if split not in _SPLIT_FILES:
            raise ValueError(
                f"Unknown split {split!r}; available: {list(_SPLIT_FILES)}. "
                f"(m_cleaned is deferred — requires ijson streaming.)"
            )
        path = _SPLIT_FILES[split]
        if not path.exists():
            raise FileNotFoundError(
                f"LongMemEval data missing: {path}. "
                f"Run `python scripts/prefetch_benchmarks.py --only longmemeval`."
            )
        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise TypeError(f"Expected list at top level of {path.name}, got {type(data)}")
        self._cache[split] = data
        return data

    def dataset_fingerprint(self, split: str = "oracle") -> str:
        if split in self._fingerprints:
            return self._fingerprints[split]
        path = _SPLIT_FILES.get(split)
        if path is None or not path.exists():
            raise FileNotFoundError(f"LongMemEval split {split} not on disk")
        fp = file_fingerprint(path)
        self._fingerprints[split] = fp
        return fp

    @staticmethod
    def _format_session(session: list, sid: str, date: str) -> str:
        """Render a list-of-messages session as a single paragraph string."""
        header = f"[Session {sid} on {date}]"
        if not isinstance(session, list):
            return header + "\n" + str(session)
        lines = [header]
        for msg in session:
            if isinstance(msg, dict):
                role = str(msg.get("role", "?"))
                content = str(msg.get("content", "")).strip()
                if content:
                    lines.append(f"{role}: {content}")
            else:
                # Defensive: occasional malformed messages.
                lines.append(str(msg))
        return "\n".join(lines)

    def _item_to_document(self, item: dict, idx: int) -> dict | None:
        question = str(item.get("question", "") or "").strip()
        answer_raw = item.get("answer", "")
        answer = str(answer_raw).strip() if answer_raw is not None else ""
        if not question:
            return None

        sessions = item.get("haystack_sessions", []) or []
        session_ids = item.get("haystack_session_ids", []) or []
        dates = item.get("haystack_dates", []) or []

        # Parallel-array invariant — if violated, drop item.
        if len(sessions) != len(session_ids):
            return None
        # Dates may be shorter (older LongMemEval items). Pad with "?".
        if len(dates) < len(sessions):
            dates = list(dates) + ["?"] * (len(sessions) - len(dates))

        if len(sessions) == 0:
            return None

        # Build paragraphs in haystack order — index i must match session_ids[i].
        paragraphs = [
            self._format_session(s, sid, d)
            for s, sid, d in zip(sessions, session_ids, dates)
        ]

        # Gold relevance from answer_session_ids.
        ans_ids = set(item.get("answer_session_ids", []) or [])
        relevant_paragraphs = sorted(
            i for i, sid in enumerate(session_ids) if sid in ans_ids
        )

        # Allow empty answer (the "(no answer in haystack)" cases) but mark them.
        if not answer:
            answer = "(no answer in haystack)"

        qa_pairs = [{
            "question": question,
            "answer": answer,
            "relevant_paragraphs": relevant_paragraphs,
        }]
        question_id = str(item.get("question_id", f"#{idx}"))
        question_type = str(item.get("question_type", ""))
        title = f"LongMemEval {question_id} ({question_type})".strip()
        return {
            "title": title,
            "paragraphs": paragraphs,
            "qa_pairs": qa_pairs,
        }

    def iter_documents(
        self,
        split: str = "oracle",
        limit: int | None = None,
        seed: int = 42,
        shuffle: bool = False,
    ) -> Iterator[dict]:
        data = self._load(split)
        n = len(data)
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
            doc = self._item_to_document(data[idx], idx)
            if doc is None:
                skipped += 1
                continue
            validate_document(doc)
            yielded += 1
            yield doc
        self.last_skipped = skipped
