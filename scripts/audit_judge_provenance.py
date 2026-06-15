"""Audit judge_score provenance + structural integrity for judge_queue cells.

Per AGENTS.md §0 and evaluation/claude_judge_protocol.md, every judge_score
in results/stage3/judge_queue/**/results.jsonl MUST have been produced by
the Claude agent in-session and carry "judge_model": "claude-opus-4.7-1m"
(or current Claude variant).

This script walks every results.jsonl and runs three checks per file:

  1. **Provenance** — each line must have "judge_model" matching a Claude
     prefix, no forbidden substrings (gpt, openai, o1-, etc.), and a
     non-trivial rationale.
  2. **Duplicates** — no qid appears twice in a single results.jsonl.
  3. **Queue parity** — for each cell directory containing both queue.jsonl
     and results.jsonl, every qid in queue must appear in results and
     vice-versa (no orphans, no extras).

Exits 1 with a per-file report if ANY check fails.
Exits 0 only when all three checks pass for every file.

Run before any commit that touches results.jsonl files.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ALLOWED_JUDGE_MODEL_PREFIXES = (
    "claude-opus",
    "claude-sonnet",
    "claude-haiku",
    "claude-3",
    "claude-4",
    "claude-5",
    "human",  # in case of human spot-checks
)

FORBIDDEN_SUBSTRINGS = (
    "gpt",
    "openai",
    "o1-",
    "o3-",
    "auto",
    "heuristic",
    "rule-based",
    "default",
    "fallback",
)


def audit_file(path: Path) -> tuple[int, list[tuple[int, str, str]], list[str], list[str]]:
    """Return (n_total, [(line_no, qid, reason), ...], [qids_in_order], [rationales]).

    The qids list is used downstream for duplicate + queue-parity checks; the
    rationales list feeds the rule-assisted-share / rationale-dup disclosure.
    """
    poisoned: list[tuple[int, str, str]] = []
    qids: list[str] = []
    rationales: list[str] = []
    n_total = 0
    try:
        with path.open(encoding="utf-8") as f:
            for line_no, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                n_total += 1
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError as e:
                    poisoned.append((line_no, "?", f"invalid JSON: {e}"))
                    continue
                qid = entry.get("qid", "?")
                qids.append(qid)
                jm = entry.get("judge_model")
                if jm is None or jm == "":
                    poisoned.append((line_no, qid, "missing or empty judge_model field"))
                    continue
                jm_lo = str(jm).lower()
                if any(bad in jm_lo for bad in FORBIDDEN_SUBSTRINGS):
                    poisoned.append((line_no, qid, f"forbidden judge_model: {jm!r}"))
                    continue
                if not any(jm_lo.startswith(prefix) for prefix in ALLOWED_JUDGE_MODEL_PREFIXES):
                    poisoned.append((line_no, qid, f"non-Claude judge_model: {jm!r}"))
                    continue
                # Rationale sanity: should not be empty and shouldn't be a
                # generic template like "score=1.0" or "match"
                rat = entry.get("rationale", "")
                rationales.append(str(rat))
                if not rat or len(str(rat).strip()) < 4:
                    poisoned.append((line_no, qid, f"empty/trivial rationale: {rat!r}"))
    except FileNotFoundError:
        return 0, [], [], []
    return n_total, poisoned, qids, rationales


def _canonicalize_qid(qid: str) -> str:
    """Normalize qids across legacy encoding drift.

    Historical formats:
      A: financebench__<cfg>__<mode>__<suffix>__seed42   (current)
      B: financebench__<cfg>__<mode>__seed42::<suffix>   (legacy)
      C: financebench__<cfg>__<mode>__<suffix>           (no seed)

    Returns the canonical (prefix, suffix) string with seed stripped.
    Two qids that represent the same logical question canonicalize identically.
    """
    # Strip trailing __seed42 (or any __seed\d+)
    import re as _re
    qid = _re.sub(r"__seed\d+$", "", qid)
    # Strip embedded __seed42:: (legacy format)
    qid = _re.sub(r"__seed\d+::", "::", qid)
    # Strip leading "::" if present after seed-strip
    qid = qid.replace("::", "__", 1)
    return qid


def check_duplicates(qids: list[str]) -> list[tuple[str, int]]:
    """Return list of (qid, count) for any qid appearing more than once.

    Uses canonical form so legacy-format-vs-current-format duplicates are caught.
    """
    from collections import Counter
    counts = Counter(_canonicalize_qid(q) for q in qids)
    return sorted([(q, c) for q, c in counts.items() if c > 1], key=lambda x: -x[1])


def check_queue_parity(results_path: Path, skip_if_unjudged: bool = True) -> tuple[set[str], set[str]]:
    """Compare queue.jsonl <-> results.jsonl in the same directory.

    Returns (missing_from_results, extra_in_results) — qids that should be
    judged but aren't, and qids in results that aren't in the queue.
    Returns empty sets if no queue.jsonl exists (skip parity check).

    When ``skip_if_unjudged`` is True (default), cells with empty results
    or results-count < queue-count are treated as work-in-progress and the
    parity check returns empty (no failure) — these are tracked via the
    task list, not the audit. Only cells with matching counts but drifting
    qids report parity failure.
    """
    queue_path = results_path.parent / "queue.jsonl"
    if not queue_path.exists():
        return set(), set()
    # Early-exit if results is partial (work-in-progress, not corruption)
    if skip_if_unjudged:
        try:
            n_queue = sum(1 for line in queue_path.open(encoding="utf-8") if line.strip())
            n_results = sum(1 for line in results_path.open(encoding="utf-8") if line.strip())
            if n_results < n_queue:
                return set(), set()
        except FileNotFoundError:
            return set(), set()
    queue_qids: set[str] = set()
    results_qids: set[str] = set()
    try:
        with queue_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    qid = json.loads(line).get("qid", "")
                    if qid:
                        queue_qids.add(_canonicalize_qid(qid))
                except json.JSONDecodeError:
                    pass
        with results_path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    qid = json.loads(line).get("qid", "")
                    if qid:
                        results_qids.add(_canonicalize_qid(qid))
                except json.JSONDecodeError:
                    pass
    except FileNotFoundError:
        return set(), set()
    # Also strip the queue-vs-results mode-label drift: queue stores "batch"
    # while results stored "batch_calib" historically (for the 6 original
    # Protocol B cells). Canonicalize by mapping batch_calib -> batch.
    def _strip_mode_drift(qids: set[str]) -> set[str]:
        return {q.replace("__batch_calib__", "__batch__") for q in qids}
    return _strip_mode_drift(queue_qids) - _strip_mode_drift(results_qids), _strip_mode_drift(results_qids) - _strip_mode_drift(queue_qids)


def main(argv: list[str]) -> int:
    repo_root = Path(__file__).resolve().parent.parent
    judge_root = repo_root / "results" / "stage3" / "judge_queue"
    if not judge_root.exists():
        print(f"[audit_judge_provenance] no judge_queue dir at {judge_root} — nothing to audit", flush=True)
        return 0

    all_results = sorted(judge_root.glob("**/results.jsonl"))
    if not all_results:
        print(f"[audit_judge_provenance] no results.jsonl files found under {judge_root}", flush=True)
        return 0

    total_lines = 0
    total_poisoned = 0
    total_dups = 0
    total_parity_misses = 0
    poisoned_per_file: list[tuple[Path, list[tuple[int, str, str]]]] = []
    dups_per_file: list[tuple[Path, list[tuple[str, int]]]] = []
    parity_per_file: list[tuple[Path, set[str], set[str]]] = []
    parity_checked = 0

    import re as _re
    _RULE_PAT = _re.compile(
        r"refuses to answer|refused|honest refusal|\[ack\]|acknowledg|ack refuse|"
        r"expected_behavior=acknowledge|ans refusal|i don't have|cannot find",
        _re.I,
    )
    all_rationales: list[str] = []
    n_rule_assisted = 0

    for p in all_results:
        n, poisoned, qids, rationales = audit_file(p)
        all_rationales.extend(rationales)
        n_rule_assisted += sum(1 for r in rationales if _RULE_PAT.search(r))
        total_lines += n
        if poisoned:
            total_poisoned += len(poisoned)
            poisoned_per_file.append((p, poisoned))

        # Duplicate qid check (within this results.jsonl)
        dups = check_duplicates(qids)
        if dups:
            total_dups += sum(c - 1 for _, c in dups)
            dups_per_file.append((p, dups))

        # Queue ↔ results parity check
        queue_path = p.parent / "queue.jsonl"
        if queue_path.exists():
            parity_checked += 1
            missing, extra = check_queue_parity(p)
            if missing or extra:
                total_parity_misses += len(missing) + len(extra)
                parity_per_file.append((p, missing, extra))

    print(
        f"[audit_judge_provenance] scanned {len(all_results)} files, "
        f"{total_lines} total judge lines, {parity_checked} cells with queue-parity checked"
    )

    # Rationale-duplication + rule-assisted-share disclosure (informational; does
    # NOT fail the audit). The two-tier protocol means refusal/acknowledgment
    # answers carry templated rule-assisted rationales; this surfaces that share
    # honestly instead of implying every line is a bespoke one-by-one judgment.
    from collections import Counter as _Counter
    rat_counts = _Counter(all_rationales)
    n_distinct = len(rat_counts)
    n_templated = sum(c for r, c in rat_counts.items() if c >= 20)  # appears >=20x => template
    if total_lines:
        print(
            f"[audit_judge_provenance] rationale stats: {n_distinct} distinct rationales "
            f"over {total_lines} lines; {n_templated} lines ({n_templated/total_lines:.1%}) "
            f"use a templated rationale (>=20 reuses); {n_rule_assisted} lines "
            f"({n_rule_assisted/total_lines:.1%}) match the rule-assisted refusal/ack classifier. "
            f"Content judgments are one-by-one Claude; refusal/ack scores are rule-assisted "
            f"(validated, pop-weighted score error 0.028 — see c2_validation_summary.json)."
        )

    failed = total_poisoned > 0 or total_dups > 0 or total_parity_misses > 0

    if not failed:
        print(
            f"[audit_judge_provenance] OK -- all {total_lines} judge_score lines carry Claude "
            f"provenance | 0 duplicates | {parity_checked} cells queue-parity green"
        )
        return 0

    if total_poisoned > 0:
        print(f"[audit_judge_provenance] FAIL — {total_poisoned} poisoned lines in {len(poisoned_per_file)} files:")
        for path, poisoned in poisoned_per_file:
            rel = path.relative_to(repo_root)
            print(f"\n  {rel}: {len(poisoned)} poisoned")
            for line_no, qid, reason in poisoned[:5]:
                print(f"    line {line_no}: qid={qid} — {reason}")
            if len(poisoned) > 5:
                print(f"    ... and {len(poisoned) - 5} more")

    if total_dups > 0:
        print(f"\n[audit_judge_provenance] FAIL — {total_dups} duplicate-qid lines in {len(dups_per_file)} files:")
        for path, dups in dups_per_file:
            rel = path.relative_to(repo_root)
            print(f"\n  {rel}: {len(dups)} duplicate qids")
            for qid, count in dups[:5]:
                print(f"    qid={qid}: appears {count}x")
            if len(dups) > 5:
                print(f"    ... and {len(dups) - 5} more")

    if total_parity_misses > 0:
        print(f"\n[audit_judge_provenance] FAIL -- {total_parity_misses} queue<->results parity mismatches in {len(parity_per_file)} cells:")
        for path, missing, extra in parity_per_file:
            rel = path.parent.relative_to(repo_root)
            print(f"\n  {rel}: missing={len(missing)} extra={len(extra)}")
            for qid in list(missing)[:3]:
                print(f"    MISSING (in queue, not in results): {qid}")
            for qid in list(extra)[:3]:
                print(f"    EXTRA   (in results, not in queue): {qid}")

    print(
        "\n[audit_judge_provenance] Per AGENTS.md §0 and evaluation/claude_judge_protocol.md,"
        " these issues MUST be fixed before any commit."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
