"""
Triple-check the 6 Stage 3 benchmarks: can we actually use the data?

For each benchmark, this script:

  1. Loads the cached data from disk (HF cache or local JSON).
  2. Counts items in the canonical eval split.
  3. Pulls 3 real samples (first / middle / last).
  4. Extracts the three fields the LLM agent needs:
       - question
       - gold answer(s)
       - source content (full doc text or session haystack)
  5. Verifies all three are non-empty and of the expected type/shape.
  6. Reports content-length distribution (min/median/p95/max).

The script runs offline (HF_DATASETS_OFFLINE=1). Anything that requires the
network indicates an incomplete prefetch and is reported as a failure.

Exit codes:
  0 — all 6 benchmarks usable
  1 — one or more benchmarks failed verification
"""

from __future__ import annotations

import json
import os
import statistics
import sys
import traceback
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data" / "benchmarks"
HF_CACHE = DATA_DIR / "hf_cache"
QASPER_DIR = DATA_DIR / "qasper"
CUAD_DIR = DATA_DIR / "cuad"
LME_DIR = DATA_DIR / "longmemeval"

# Force offline mode — any network call here proves the prefetch missed something.
os.environ["HF_DATASETS_OFFLINE"] = "1"
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["HF_DATASETS_CACHE"] = str(HF_CACHE)
os.environ["HF_HOME"] = str(HF_CACHE)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _print_step(name: str) -> None:
    print()
    print("=" * 78)
    print(f"  {name}")
    print("=" * 78)


def _summarize_lengths(values: list[int], label: str) -> str:
    if not values:
        return f"{label}: no values"
    return (
        f"{label}: n={len(values)} "
        f"min={min(values):,} "
        f"p50={int(statistics.median(values)):,} "
        f"p95={int(sorted(values)[int(0.95 * (len(values) - 1))]):,} "
        f"max={max(values):,}"
    )


def _assert(cond: bool, msg: str) -> None:
    if not cond:
        raise AssertionError(msg)


def _sample_indices(n: int) -> list[int]:
    """Return [0, n//2, n-1] but unique and valid."""
    if n <= 0:
        return []
    if n == 1:
        return [0]
    if n == 2:
        return [0, 1]
    return [0, n // 2, n - 1]


# ---------------------------------------------------------------------------
# Per-benchmark verifiers
# ---------------------------------------------------------------------------


def verify_hotpotqa() -> dict[str, Any]:
    _print_step("HotpotQA — validation split, distractor config")
    from datasets import load_dataset
    ds = load_dataset(
        "hotpotqa/hotpot_qa", "distractor", cache_dir=str(HF_CACHE),
    )
    val = ds["validation"]
    n = len(val)
    print(f"  n_items (val) = {n}")
    _assert(n >= 7000, f"expected ~7,405 validation items, got {n}")

    # Inspect 3 samples for shape correctness.
    ctx_lens: list[int] = []
    sup_fact_counts: list[int] = []
    for idx in _sample_indices(n):
        item = val[idx]
        q = item.get("question", "")
        a = item.get("answer", "")
        ctx = item.get("context", {})
        # HF format: ctx = {"title": [t1, t2, ...], "sentences": [[s1,s2,...], [...], ...]}
        titles = ctx.get("title", [])
        sentences = ctx.get("sentences", [])
        n_passages = len(titles)
        total_chars = sum(sum(len(s) for s in passage) for passage in sentences)
        sup_facts = item.get("supporting_facts", {})
        n_sup = len(sup_facts.get("title", []))
        _assert(bool(q), f"empty question at idx={idx}")
        _assert(bool(a), f"empty answer at idx={idx}")
        _assert(n_passages >= 2, f"<2 passages at idx={idx}")
        _assert(n_sup >= 1, f"no supporting facts at idx={idx}")
        ctx_lens.append(total_chars)
        sup_fact_counts.append(n_sup)
        print(f"  [{idx}] q={q[:60]!r}...")
        print(f"        a={a[:50]!r}  n_passages={n_passages}  total_chars={total_chars:,}  n_sup_facts={n_sup}")

    # Full distribution over 100 sampled items (don't iterate 7K eagerly).
    step = max(1, n // 100)
    sample_lens = [
        sum(sum(len(s) for s in p) for p in val[i].get("context", {}).get("sentences", []))
        for i in range(0, n, step)
    ]
    print(f"  {_summarize_lengths(sample_lens, 'context_chars (n=100 stride sample)')}")
    return {"ok": True, "n": n, "ctx_chars": sample_lens}


def verify_qasper() -> dict[str, Any]:
    _print_step("QASPER — dev + test splits (canonical eval)")
    results: dict[str, Any] = {}
    json_files = sorted(QASPER_DIR.rglob("qasper-*-v0.3.json"))
    _assert(len(json_files) >= 2, f"expected dev+test+train JSONs, found {[p.name for p in json_files]}")

    for jp in json_files:
        with jp.open("r", encoding="utf-8") as f:
            data = json.load(f)
        n_papers = len(data)
        n_qs = sum(len(p.get("qas", [])) for p in data.values())
        results[jp.stem] = {"n_papers": n_papers, "n_questions": n_qs}
        print(f"  {jp.name}: {n_papers} papers, {n_qs} questions")
        _assert(n_papers > 0 and n_qs > 0, f"empty split {jp.name}")

    # Inspect 3 dev papers in detail.
    dev_json = next(jp for jp in json_files if "dev" in jp.name)
    with dev_json.open("r", encoding="utf-8") as f:
        dev = json.load(f)
    paper_ids = list(dev.keys())
    n = len(paper_ids)
    section_char_counts: list[int] = []
    qs_per_paper: list[int] = []
    ans_present_count = 0
    ans_total = 0
    for idx in _sample_indices(n):
        pid = paper_ids[idx]
        p = dev[pid]
        title = p.get("title", "")
        abstract = p.get("abstract", "")
        full_text = p.get("full_text", [])
        qas = p.get("qas", [])
        _assert(bool(title), f"empty title for paper {pid}")
        _assert(len(qas) > 0, f"no questions for paper {pid}")
        # full_text is a list of {section_name, paragraphs: [...]}
        section_chars = sum(sum(len(par) for par in s.get("paragraphs", [])) for s in full_text)
        section_char_counts.append(section_chars)
        qs_per_paper.append(len(qas))
        # Inspect first QA in detail
        q0 = qas[0]
        question = q0.get("question", "")
        answers = q0.get("answers", [])
        _assert(bool(question), f"empty question in paper {pid}")
        # answers is a list of {answer: {...}}; pull human-readable
        first_ans = ""
        if answers:
            a0 = answers[0].get("answer", {})
            first_ans = a0.get("free_form_answer") or a0.get("extractive_spans", [""])[0] or ""
            if a0.get("yes_no") is not None:
                first_ans = "yes" if a0["yes_no"] else "no"
        for ans_obj in answers:
            a = ans_obj.get("answer", {})
            ans_total += 1
            if (a.get("free_form_answer") or a.get("extractive_spans") or a.get("yes_no") is not None):
                ans_present_count += 1
        print(f"  [{idx}] {pid} title={title[:60]!r}... n_qas={len(qas)} full_text_chars={section_chars:,}")
        print(f"        q0={question[:60]!r}... a0={str(first_ans)[:50]!r}")

    print(f"  {_summarize_lengths(section_char_counts, 'full_text_chars (sample)')}")
    print(f"  qs_per_paper (sample): {qs_per_paper}")
    print(f"  answer_present_rate over inspected: {ans_present_count}/{ans_total}")
    return {"ok": True, "splits": results, "full_text_chars": section_char_counts}


def verify_cuad() -> dict[str, Any]:
    _print_step("CUAD — full SQuAD-format set")
    json_files = sorted(CUAD_DIR.rglob("CUAD_v1.json"))
    _assert(len(json_files) >= 1, "CUAD_v1.json not found")
    cp = json_files[0]
    print(f"  parsing {cp.relative_to(DATA_DIR)} ({cp.stat().st_size / 1e6:.1f} MB)")
    with cp.open("r", encoding="utf-8") as f:
        data = json.load(f)

    contracts = data.get("data", [])
    n_contracts = len(contracts)
    n_qas = sum(len(p.get("qas", []))
                for c in contracts
                for p in c.get("paragraphs", []))
    _assert(n_contracts == 510, f"expected 510 contracts, got {n_contracts}")
    _assert(n_qas > 0, "no QAs")
    print(f"  n_contracts = {n_contracts}, n_questions = {n_qas}")

    ctx_lens: list[int] = []
    answer_present = 0
    answer_empty = 0
    for idx in _sample_indices(n_contracts):
        c = contracts[idx]
        title = c.get("title", "")
        paragraphs = c.get("paragraphs", [])
        _assert(len(paragraphs) > 0, f"no paragraphs for contract {idx}")
        # CUAD typically has one big paragraph (the full contract) per contract entry.
        para0 = paragraphs[0]
        context = para0.get("context", "")
        qas = para0.get("qas", [])
        _assert(bool(context), f"empty context at contract {idx}")
        _assert(len(qas) > 0, f"no QAs at contract {idx}")
        ctx_lens.append(len(context))
        # Sample first QA
        q0 = qas[0]
        question = q0.get("question", "")
        answers = q0.get("answers", [])
        is_impossible = q0.get("is_impossible", False)
        # SQuAD v2: answers can be [] when is_impossible=True
        if answers:
            ans_text = answers[0].get("text", "")
            answer_present += 1
        else:
            ans_text = "(no-answer, impossible)"
            answer_empty += 1
        print(f"  [{idx}] {title[:50]!r} ctx={len(context):,} chars n_qas={len(qas)}")
        print(f"        q0={question[:60]!r}... a0={ans_text[:60]!r}  impossible={is_impossible}")

    # Sweep through all 510 to count answerable vs no-answer.
    all_present = 0
    all_empty = 0
    for c in contracts:
        for p in c.get("paragraphs", []):
            for q in p.get("qas", []):
                if q.get("answers"):
                    all_present += 1
                else:
                    all_empty += 1
    print(f"  Across all {n_qas} QAs: with_answer={all_present:,}  no_answer={all_empty:,}")
    _assert(all_present > 0, "no answerable QAs anywhere")
    print(f"  {_summarize_lengths(ctx_lens, 'contract_context_chars (sample)')}")
    # Per-contract length distribution over all 510.
    all_ctx_lens = [len(p.get("context", ""))
                    for c in contracts
                    for p in c.get("paragraphs", [])]
    print(f"  {_summarize_lengths(all_ctx_lens, 'contract_context_chars (all 510)')}")
    return {
        "ok": True,
        "n_contracts": n_contracts,
        "n_questions": n_qas,
        "with_answer": all_present,
        "no_answer": all_empty,
        "all_ctx_lens_p50": int(statistics.median(all_ctx_lens)),
    }


def verify_narrativeqa() -> dict[str, Any]:
    _print_step("NarrativeQA — validation split")
    from datasets import load_dataset
    ds = load_dataset("deepmind/narrativeqa", cache_dir=str(HF_CACHE))
    val = ds["validation"]
    n = len(val)
    print(f"  n_items (val) = {n}")
    _assert(n >= 3000, f"expected ~3,461 val items, got {n}")

    doc_lens: list[int] = []
    multi_ref_counts: list[int] = []
    for idx in _sample_indices(n):
        item = val[idx]
        q = item.get("question", {}).get("text", "")
        # answers is a list of {text, tokens}; multiple refs per Q
        answers = item.get("answers", [])
        doc_text = item.get("document", {}).get("text", "")
        _assert(bool(q), f"empty question at {idx}")
        _assert(len(answers) >= 1, f"no answers at {idx}")
        _assert(bool(doc_text), f"empty document at {idx}")
        ref_texts = [a.get("text", "") for a in answers if a.get("text")]
        _assert(any(ref_texts), f"all empty answer texts at {idx}")
        doc_lens.append(len(doc_text))
        multi_ref_counts.append(len(ref_texts))
        print(f"  [{idx}] q={q[:60]!r}...")
        print(f"        a={ref_texts[0][:60]!r}  n_refs={len(ref_texts)}  doc_chars={len(doc_text):,}")

    # Sample 50 items for document-length distribution.
    step = max(1, n // 50)
    sample_lens = [
        len(val[i].get("document", {}).get("text", ""))
        for i in range(0, n, step)
    ]
    print(f"  {_summarize_lengths(sample_lens, 'doc_chars (stride sample)')}")
    return {"ok": True, "n": n, "doc_chars": sample_lens}


def verify_financebench() -> dict[str, Any]:
    _print_step("FinanceBench — full 150 public eval items")
    from datasets import load_dataset
    ds = load_dataset("PatronusAI/financebench", cache_dir=str(HF_CACHE))
    split = next(iter(ds.values()))
    n = len(split)
    print(f"  n_items = {n}")
    _assert(n == 150, f"expected 150 items, got {n}")

    evidence_lens: list[int] = []
    has_evidence = 0
    has_answer = 0
    has_question = 0
    for i in range(n):
        item = split[i]
        if item.get("question"):
            has_question += 1
        if item.get("answer"):
            has_answer += 1
        ev = item.get("evidence", "")
        # Evidence can be a string or list-of-strings; normalize to string.
        if isinstance(ev, list):
            ev_text = " ".join(
                e.get("evidence_text", "") if isinstance(e, dict) else str(e)
                for e in ev
            )
        else:
            ev_text = str(ev) if ev else ""
        if ev_text.strip():
            has_evidence += 1
            evidence_lens.append(len(ev_text))

    print(f"  has_question={has_question}/{n}  has_answer={has_answer}/{n}  has_evidence={has_evidence}/{n}")
    _assert(has_question == n, "not all items have question")
    _assert(has_answer == n, "not all items have answer")
    _assert(has_evidence >= int(n * 0.8), f"too few items have non-empty evidence: {has_evidence}/{n}")

    # Inspect 3.
    for idx in _sample_indices(n):
        item = split[idx]
        q = item.get("question", "")
        a = item.get("answer", "")
        ev = item.get("evidence", "")
        if isinstance(ev, list):
            ev_text = " | ".join(
                (e.get("evidence_text", "") if isinstance(e, dict) else str(e))[:60]
                for e in ev[:2]
            )
            ev_len = sum(
                len(e.get("evidence_text", "")) if isinstance(e, dict) else len(str(e))
                for e in ev
            )
        else:
            ev_text = str(ev)[:120]
            ev_len = len(str(ev))
        company = item.get("company", "")
        doc_type = item.get("doc_type", "")
        print(f"  [{idx}] company={company!r} doc_type={doc_type!r}")
        print(f"        q={q[:70]!r}")
        print(f"        a={a[:70]!r}")
        print(f"        ev={ev_text[:100]!r}... ({ev_len:,} chars total)")

    print(f"  {_summarize_lengths(evidence_lens, 'evidence_chars (all)')}")
    return {
        "ok": True, "n": n,
        "has_evidence": has_evidence,
        "evidence_chars_p50": int(statistics.median(evidence_lens)) if evidence_lens else 0,
    }


def verify_longmemeval() -> dict[str, Any]:
    _print_step("LongMemEval — oracle split (canonical), spot-check large files")
    # Load oracle in full (15 MB is fine).
    oracle_path = LME_DIR / "longmemeval_oracle.json"
    _assert(oracle_path.exists(), f"{oracle_path} missing")
    with oracle_path.open("r", encoding="utf-8") as f:
        oracle = json.load(f)
    n_oracle = len(oracle)
    print(f"  oracle: {n_oracle} items, {oracle_path.stat().st_size / 1e6:.1f} MB")
    _assert(n_oracle == 500, f"expected 500 oracle items, got {n_oracle}")

    session_counts: list[int] = []
    msg_counts_all: list[int] = []
    for idx in _sample_indices(n_oracle):
        item = oracle[idx]
        q = item.get("question", "")
        a = item.get("answer", "")
        haystack = item.get("haystack_sessions", [])
        ans_session_ids = item.get("answer_session_ids", [])
        _assert(bool(q), f"empty question at idx={idx}")
        _assert(a is not None and str(a).strip(), f"empty answer at idx={idx}")
        _assert(len(haystack) >= 1, f"no haystack sessions at idx={idx}")
        n_sessions = len(haystack)
        # Each session is a list of {role, content}
        n_msgs_total = sum(len(s) if isinstance(s, list) else 0 for s in haystack)
        session_counts.append(n_sessions)
        msg_counts_all.append(n_msgs_total)
        qtype = item.get("question_type", "")
        print(f"  [{idx}] q={q[:60]!r}...")
        print(f"        a={str(a)[:60]!r} type={qtype!r} n_sessions={n_sessions} total_msgs={n_msgs_total}")
        print(f"        answer_session_ids={ans_session_ids[:3]}")

    # Sweep oracle for full distribution.
    all_session_counts = [len(it.get("haystack_sessions", [])) for it in oracle]
    print(f"  {_summarize_lengths(all_session_counts, 'sessions_per_item (all 500)')}")

    # Spot-check s_cleaned by parsing one item via streaming.
    s_path = LME_DIR / "longmemeval_s_cleaned.json"
    s_ok = False
    s_n: int | None = None
    if s_path.exists():
        sz = s_path.stat().st_size
        print(f"  s_cleaned: file present ({sz/1e6:.1f} MB)")
        # 264 MB fits in memory.
        with s_path.open("r", encoding="utf-8") as f:
            s_data = json.load(f)
        s_n = len(s_data)
        print(f"    s_cleaned: {s_n} items (loaded in memory)")
        _assert(s_n > 0, "s_cleaned empty")
        s_ok = True
        # First item structural check
        first = s_data[0]
        _assert("question" in first and "haystack_sessions" in first,
                f"s_cleaned first item missing fields: {sorted(first.keys())}")

    m_path = LME_DIR / "longmemeval_m_cleaned.json"
    m_ok = False
    if m_path.exists():
        sz = m_path.stat().st_size
        print(f"  m_cleaned: file present ({sz/1e6:.1f} MB) — too large for eager parse")
        # Use ijson if available; else just verify file head is valid JSON list start.
        with m_path.open("r", encoding="utf-8") as f:
            head = f.read(256)
        _assert(head.lstrip().startswith("["), f"m_cleaned doesn't start with JSON list: head={head[:40]!r}")
        m_ok = True
        print(f"    m_cleaned: file head looks like JSON list — streaming required for full iteration")

    return {
        "ok": True,
        "oracle_n": n_oracle,
        "s_cleaned_n": s_n,
        "s_cleaned_ok": s_ok,
        "m_cleaned_ok": m_ok,
        "sessions_per_item_p50": int(statistics.median(all_session_counts)),
    }


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


VERIFIERS = {
    "hotpotqa": verify_hotpotqa,
    "qasper": verify_qasper,
    "cuad": verify_cuad,
    "narrativeqa": verify_narrativeqa,
    "financebench": verify_financebench,
    "longmemeval": verify_longmemeval,
}


def main() -> int:
    print(f"Verifying benchmark data at {DATA_DIR}")
    print(f"HF_DATASETS_OFFLINE = {os.environ.get('HF_DATASETS_OFFLINE')}")
    print(f"HF_HUB_OFFLINE      = {os.environ.get('HF_HUB_OFFLINE')}")

    ok: dict[str, Any] = {}
    failed: dict[str, str] = {}
    for name, fn in VERIFIERS.items():
        try:
            ok[name] = fn()
        except Exception as e:
            traceback.print_exc()
            failed[name] = repr(e)
            print(f"\n  [FAIL] {name}: {e!r}")

    print()
    print("=" * 78)
    print("  VERIFICATION SUMMARY")
    print("=" * 78)
    for name in VERIFIERS:
        if name in ok:
            print(f"  [OK]   {name}")
        else:
            print(f"  [FAIL] {name}  -- {failed.get(name, '?')[:80]}")

    # Write a verification report alongside the manifest.
    report_path = DATA_DIR / "verification.json"
    report = {
        "ok": ok,
        "failed": failed,
        "all_ok": not failed,
    }
    report_path.write_text(json.dumps(report, indent=2, default=str))
    print(f"\n  Verification report: {report_path}")

    return 0 if not failed else 1


if __name__ == "__main__":
    sys.exit(main())
