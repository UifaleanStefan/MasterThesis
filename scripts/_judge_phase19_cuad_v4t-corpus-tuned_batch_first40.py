"""Phase 1.9 — CUAD v4t-corpus-tuned batch (first 40 of 132 entries).

Pattern observation: corpus-tuned θ on CUAD shows substantial improvement
over canonical — captures concept correctly for several questions but
struggles with exact clause text extraction (legal text is verbatim).
Mix of exact matches (Document Name questions), refusals, and partial
matches. Recall@k=0.174 vs canonical 0.000.

This script judges the first 40/132 entries as a representative sample.
"""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "cuad__v4t-corpus-tuned__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/cuad__v4t-corpus-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.0, "PRED refuses ('does not provide specific details')."),
    ("doc0_qa1", 0.0, "PRED refuses ('does not provide specific details')."),
    ("doc0_qa2", 0.0, "PRED refuses ('not explicitly mentioned'); gold=7 Sept 1999."),
    ("doc0_qa3", 0.5, "PRED 'ten (10) years from effective date' partially matches gold."),
    ("doc0_qa4", 0.5, "PRED '10 years from effective date' partial match."),
    ("doc0_qa5", 1.0, "PRED 'renewable annual basis 1-year terms up to 10 years' exact match to gold."),
    ("doc0_qa6", 0.0, "PRED refuses ('not provided'); gold='State of Illinois'."),
    ("doc0_qa7", 0.0, "PRED refuses ('do not contain Exclusivity')."),
    ("doc0_qa8", 1.0, "PRED 'Distributor agrees not to interfere with business relations' matches gold."),
    ("doc0_qa9", 0.0, "PRED refuses ('do not contain No-Solicit')."),
    ("doc0_qa10", 0.0, "PRED refuses ('do not contain Rofr/Rofo/Rofn')."),
    ("doc0_qa11", 0.25, "PRED 'consent required for assignment but mentions Dova' wrong context."),
    ("doc0_qa12", 0.0, "PRED refuses ('do not contain Price Restrictions')."),
    ("doc0_qa13", 0.0, "PRED refuses ('do not contain Minimum Commitment')."),
    ("doc0_qa14", 0.0, "PRED refuses ('do not include License Grant')."),
    ("doc0_qa15", 0.25, "PRED mentions Post-Termination Services concept; not the exact clause."),
    ("doc0_qa16", 0.25, "PRED mentions implied warranties; not the exact gold."),
    ("doc0_qa17", 1.0, "PRED 'Company carries product liability + names Distributor' matches gold."),
    ("doc0_qa18", 0.0, "PRED refuses ('do not contain Covenant Not To Sue')."),
    ("doc1_qa0", 1.0, "PRED 'Promotion and Distribution Agreement' exact match to gold."),
    ("doc1_qa1", 0.25, "PRED 'Whitesmoke and Google' wrong context; gold=Distributor."),
    ("doc1_qa2", 1.0, "PRED '1 August 2011' exact match to gold."),
    ("doc1_qa3", 0.0, "PRED 'August 1, 2021' WRONG; gold='2-year term ending 31 July 2013'."),
    ("doc1_qa4", 0.75, "PRED 'governed by English law' matches first part of gold."),
    ("doc1_qa5", 0.0, "PRED 'one party may terminate on Change of Control' wrong direction."),
    ("doc1_qa6", 0.75, "PRED 'may not be assigned without written consent' matches gold concept."),
    ("doc1_qa7", 0.0, "PRED says no license but gold=Google trademarks license — Y/N flip."),
    ("doc1_qa8", 1.0, "PRED 'Google may audit Distributor's records' matches gold."),
    ("doc1_qa9", 0.0, "PRED refuses ('do not contain Uncapped Liability')."),
    ("doc1_qa10", 0.0, "PRED refuses ('do not include Cap On Liability')."),
    ("doc1_qa11", 0.0, "PRED 'no warranty duration' WRONG; gold has specific period."),
    ("doc2_qa0", 0.5, "PRED 'Supply Agreement' close to gold 'SUPPLY CONTRACT'."),
    ("doc2_qa1", 0.25, "PRED 'framework agreement, buyer entrusts' partial context."),
    ("doc2_qa2", 0.0, "PRED 'does not specify expiration' but gold says '5 years'."),
    ("doc2_qa3", 0.0, "PRED 'English law' WRONG; gold=Chinese law."),
    ("doc2_qa4", 0.0, "PRED refuses ('do not contain Warranty Duration')."),
    ("doc2_qa5", 0.0, "PRED refuses ('do not contain Insurance')."),
    ("doc3_qa0", 0.0, "PRED 'entire contract should be reviewed' too vague; gold='WEB SITE HOSTING AGREEMENT'."),
    ("doc3_qa1", 0.0, "PRED refuses ('no specific parts'); gold='Centrack International'."),
    ("doc3_qa2", 0.0, "PRED refuses ('not mentioned'); gold='6 April 1999'."),
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
    print(f"cuad v4t-corpus-tuned batch first-40 added={added} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
