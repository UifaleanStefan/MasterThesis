"""B2 step: build results.jsonl for the four HQA/LME held-out batch cells from
queue.jsonl + the pass-2 workflow scores. Replaces the stale n=10 results with
the 100-doc judged set."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WD = ROOT / "results/stage3/_judge_workdir"
QROOT = ROOT / "results/stage3/judge_queue"
CELLS = [
    "hotpotqa__v4t-canonical__online__seed42",
    "hotpotqa__v4t-corpus-tuned__online__seed42",
    "longmemeval__v4t-canonical__online__seed42",
    "longmemeval__v4t-corpus-tuned__online__seed42",
]
DOMAIN = {0.0, 0.25, 0.5, 0.75, 1.0}

for cell in CELLS:
    sf = WD / f"rejudge_scores__{cell}__part00.json"
    if not sf.exists():
        print(f"{cell}: no scores yet — skip"); continue
    qlines = (QROOT / cell / "queue.jsonl").read_text("utf-8").splitlines()
    scores = json.loads(sf.read_text("utf-8"))
    out, ssum = [], 0.0
    for L in qlines:
        e = json.loads(L); qid = e["qid"]
        assert qid in scores, f"missing {qid}"
        s, rat = scores[qid]
        assert s in DOMAIN, f"bad {s}"
        ssum += s
        out.append(json.dumps({"qid": qid, "judge_score": s, "rationale": rat,
                               "judge_model": "claude-opus-4.7-1m",
                               "judge_protocol": "2_manual"}, ensure_ascii=False))
    (QROOT / cell / "results.jsonl").write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{cell}: wrote {len(out)} results, mean={ssum/len(out):.4f}")
