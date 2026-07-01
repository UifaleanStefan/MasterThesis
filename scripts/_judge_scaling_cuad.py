"""Apply Claude 1-by-1 judgments for the judged accuracy-vs-N scaling cells
(scaling__cuad__v4t-corpus-tuned__N{50,150,300}__seed42), audit critique B2.

IMPORTANT HONEST CAVEATS (carried into the thesis):
- The probe build_probe() takes qa_pairs[:1] per contract, and CUAD's first QA
  per contract is ALWAYS the "Document Name" category. So this probe measures
  document-name-extraction accuracy vs N ONLY -- not a representative QA mix.
- Mechanical recall@8 is 0.0 at every N because the gold is the title paragraph
  (step 0 of each doc), which selective retrieval never ranks in top-8; the
  answerer still often recovers the name from the doc-title-prefixed context.
  So recall@k is uninformative here; judged accuracy is the signal.
- n=20 per N-point -> wide error bars.

Each (qid -> (score, rationale)) is a hand judgment by Claude: 1.0 exact doc-name
match; 0.75 correct core, minor extra/missing word; 0.5 right subject wrong
instrument noun (e.g. 'Supply Agreement' vs gold 'SUPPLY CONTRACT') or missing a
qualifier ('MASTER'/'CO-'); 0.25 weakly related; 0.0 abstention when gold exists
or a wrong contract. NO heuristics.
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QUEUE = ROOT / "results" / "stage3" / "judge_queue"

# per-cell: probe_d{i}_q0 -> (score, rationale)
def _mk(scores, rats):
    return {f"probe_d{i}_q0": (scores[i], rats[i]) for i in range(len(scores))}

N50 = _mk(
 [1.0,1.0,0.5,1.0,0.0,1.0,0.0,0.0,0.5,0.0,1.0,0.0,1.0,0.0,1.0,0.75,0.5,0.25,1.0,0.0],
 ["exact 'DISTRIBUTOR AGREEMENT'","names Promotion and Distribution Agreement","'Supply Agreement' vs gold 'SUPPLY CONTRACT'","exact Web Site Hosting Agreement","wrong contract (Servicing/CURO)","exact ENDORSEMENT AGREEMENT","abstains, gold exists","abstains, gold exists","'Promotion Agreement' vs gold 'CO-PROMOTION'","abstains (PACIRA), gold exists","exact Remarketing Agreement","abstains, gold exists","Sponsorship Agreement","wrong ('Franchise Agreement')","Intellectual Property Agreement","gold 'ENDORSEMENT'; pred adds 'Agreement'","'Supply Agreement' vs 'MASTER SUPPLY AGREEMENT'","'Service Agreement' vs 'DISTRIBUTION AND SERVICES AGREEMENT'","Intellectual Property Agreement","abstains (SIBANNAC), gold exists"],
)
N150 = _mk(
 [0.0,1.0,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.75,0.0,0.0,0.0,0.0,0.0,0.0,0.5,0.0,1.0,0.0],
 ["abstains, gold exists","names Promotion and Distribution Agreement","abstains, gold exists","abstains, gold exists","abstains, gold exists","abstains, gold exists","abstains, gold exists","abstains, gold exists","'Promotion Agreement' vs 'CO-PROMOTION'","captures Strategic Licensing...Agreement; misses 'Amended and Restated'","abstains, gold exists","abstains, gold exists","abstains, gold exists","wrong ('Franchise Agreement')","abstains, gold exists","abstains, gold exists","'Supply Agreement' vs 'MASTER SUPPLY AGREEMENT'","lists clauses, doesn't name the doc","exact Intellectual Property Agreement","abstains, gold exists"],
)
N300 = _mk(
 [1.0,1.0,0.5,1.0,0.0,1.0,1.0,0.5,0.5,0.75,0.0,1.0,1.0,0.0,1.0,0.75,0.5,0.0,1.0,1.0],
 ["Distributor Agreement","Promotion and Distribution Agreement","'Supply Agreement' vs 'SUPPLY CONTRACT'","Web Site Hosting Agreement","abstains, gold exists","ENDORSEMENT AGREEMENT","CONSULTING AGREEMENT","'JOINT VENTURE AGREEMENT'; misses 'Amendment and Termination of'","'Promotion Agreement' vs 'CO-PROMOTION'","A_R Strategic Licensing...Agreement (=Amended and Restated)","wrong contract (Transportation Contract)","Strategic Alliance Agreement","Sponsorship Agreement","wrong ('Franchise Agreement')","Intellectual Property Agreement","gold 'ENDORSEMENT'; pred adds 'Agreement'","'Supply Agreement' vs 'MASTER SUPPLY AGREEMENT'","'not explicitly provided' -- abstains","Intellectual Property Agreement","Strategic Alliance Agreement"],
)

CELLS = {50: N50, 150: N150, 300: N300}


def main() -> int:
    summary = {}
    for n, J in CELLS.items():
        cell = QUEUE / f"scaling__cuad__v4t-corpus-tuned__N{n}__seed42"
        queue = [json.loads(l) for l in (cell / "queue.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
        out, missing = [], []
        for q in queue:
            qid = q["qid"]
            if qid not in J:
                missing.append(qid); continue
            score, rat = J[qid]
            out.append({"qid": qid, "judge_score": score, "rationale": rat,
                        "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1"})
        if missing:
            raise SystemExit(f"N={n} missing {len(missing)}: {missing[:5]}")
        (cell / "results.jsonl").write_text("\n".join(json.dumps(o) for o in out) + "\n", encoding="utf-8")
        mean = sum(o["judge_score"] for o in out) / len(out)
        summary[n] = round(mean, 4)
        print(f"N={n}: wrote {len(out)} judgments; mean judge = {mean:.4f}")
    print("\naccuracy-vs-N (document-name probe, n=20/pt):", summary)
    print("-> NON-MONOTONIC / noisy; does not cleanly support flat scaling (see script docstring).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
