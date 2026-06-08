"""Phase 1.9 — CUAD v4t-canonical batch (first 40 of 132 entries).

Pattern observation: canonical θ on CUAD's first-30-contract ingestion
shows pervasive cross-contract contamination — most predictions
reference the FIRST contract's "Maker/Payee/Amended Strategic Licensing"
context regardless of which contract the question is about. Recall@k=0.0
on this cell (qa_summary.json) confirms zero useful retrievals.

This script judges the first 40/132 entries as a representative sample;
remaining 92 entries left for future continuation.
"""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "cuad__v4t-canonical__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/cuad__v4t-canonical__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.0, "PRED 'nonnegotiability clause' wrong contract; gold='DISTRIBUTOR AGREEMENT'."),
    ("doc0_qa1", 0.0, "PRED 'Maker and Payee' wrong contract; gold='Distributor'."),
    ("doc0_qa2", 0.0, "PRED 'October 2009' wrong contract; gold='7th September 1999'."),
    ("doc0_qa3", 0.0, "PRED refuses ('not explicitly provided')."),
    ("doc0_qa4", 0.0, "PRED 'October 9, 2009' wrong contract; gold=10-year term from delivery."),
    ("doc0_qa5", 0.0, "PRED refuses ('do not contain Renewal Term')."),
    ("doc0_qa6", 0.0, "PRED refuses ('do not contain Governing Law')."),
    ("doc0_qa7", 0.0, "PRED refuses ('do not contain Exclusivity')."),
    ("doc0_qa8", 0.0, "PRED refuses ('do not contain No-Solicit')."),
    ("doc0_qa9", 0.0, "PRED refuses ('do not contain No-Solicit of Employees')."),
    ("doc0_qa10", 0.0, "PRED refuses ('do not contain Rofr/Rofo/Rofn')."),
    ("doc0_qa11", 0.25, "PRED 'nontransferable note' loosely matches anti-assignment concept."),
    ("doc0_qa12", 0.0, "PRED refuses ('do not contain Price Restrictions')."),
    ("doc0_qa13", 0.0, "PRED refuses ('do not contain Minimum Commitment')."),
    ("doc0_qa14", 0.0, "PRED refuses ('do not contain License Grant')."),
    ("doc0_qa15", 0.0, "PRED refuses ('do not contain Post-Termination Services')."),
    ("doc0_qa16", 0.0, "PRED refuses ('do not contain Warranty Duration')."),
    ("doc0_qa17", 0.0, "PRED refuses ('do not contain Insurance')."),
    ("doc0_qa18", 0.0, "PRED refuses ('do not contain Covenant Not To Sue')."),
    ("doc1_qa0", 0.0, "PRED 'Amended Strategic Licensing' wrong contract; gold='Promotion and Distribution Agreement'."),
    ("doc1_qa1", 0.0, "PRED 'Maker and Payee' wrong contract; gold='Distributor'."),
    ("doc1_qa2", 0.0, "PRED refuses ('not explicitly mentioned'); gold='1 August 2011'."),
    ("doc1_qa3", 0.0, "PRED 'fifth anniversary' wrong contract; gold='2-year term ending 31 July 2013'."),
    ("doc1_qa4", 0.0, "PRED refuses ('do not contain Governing Law'); gold='English law'."),
    ("doc1_qa5", 0.0, "PRED refuses ('do not contain Change Of Control')."),
    ("doc1_qa6", 0.25, "PRED 'nontransferable note' loosely matches anti-assignment."),
    ("doc1_qa7", 0.0, "PRED refuses ('do not contain License Grant'); gold='Google Trademarks license'."),
    ("doc1_qa8", 0.0, "PRED refuses ('do not contain Audit Rights')."),
    ("doc1_qa9", 0.0, "PRED refuses ('do not contain Uncapped Liability')."),
    ("doc1_qa10", 0.0, "PRED refuses ('do not contain Cap On Liability')."),
    ("doc1_qa11", 0.0, "PRED refuses ('do not contain Warranty Duration')."),
    ("doc2_qa0", 0.0, "PRED 'Maker/Payee' wrong contract; gold='SUPPLY CONTRACT'."),
    ("doc2_qa1", 0.0, "PRED 'Maker/Payee' wrong contract; gold='The seller'."),
    ("doc2_qa2", 0.0, "PRED 'fifth anniversary' wrong contract; gold='5-year term'."),
    ("doc2_qa3", 0.0, "PRED refuses; gold='law of the People's Republic of China'."),
    ("doc2_qa4", 0.0, "PRED refuses ('do not contain Warranty Duration')."),
    ("doc2_qa5", 0.0, "PRED refuses ('do not contain Insurance')."),
    ("doc3_qa0", 0.0, "PRED 'nonnegotiability' wrong contract; gold='WEB SITE HOSTING AGREEMENT'."),
    ("doc3_qa1", 0.0, "PRED 'Maker/Payee' wrong contract; gold='Centrack International'."),
    ("doc3_qa2", 0.0, "PRED 'Amended Strategic Licensing' wrong contract; gold='6 April 1999'."),
]


def main() -> None:
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: existing.add(json.loads(line)["qid"])
                except: pass
    added = 0; total = 0.0
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        for suffix, score, rationale in JUDGMENTS:
            qid = QID_PREFIX + suffix + QID_SUFFIX
            if qid in existing: continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"cuad v4t-canonical batch first-40 added={added} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
