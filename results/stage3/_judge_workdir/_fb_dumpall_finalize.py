"""Build results.jsonl for the fresh FB dump-all Protocol-A cells from
queue.jsonl + the pass-2 workflow scores (rejudge_scores__<cell>__part00.json).
These cells had no pass-1 results, so we create results.jsonl from scratch."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
WD = ROOT / "results/stage3/_judge_workdir"
QROOT = ROOT / "results/stage3/judge_queue"
CELLS = [
    "financebench__dump-all__online__seed42",
    "financebench__dump-all__batch__seed42",
]
DOMAIN = {0.0, 0.25, 0.5, 0.75, 1.0}

for cell in CELLS:
    qlines = (QROOT / cell / "queue.jsonl").read_text("utf-8").splitlines()
    scores = json.loads((WD / f"rejudge_scores__{cell}__part00.json").read_text("utf-8"))
    out = []
    ssum = 0.0
    for L in qlines:
        e = json.loads(L)
        qid = e["qid"]
        assert qid in scores, f"missing score for {qid}"
        s, rat = scores[qid]
        assert s in DOMAIN, f"bad score {s} for {qid}"
        ssum += s
        out.append(json.dumps({
            "qid": qid,
            "judge_score": s,
            "rationale": rat,
            "judge_model": "claude-opus-4.7-1m",
            "judge_protocol": "2_manual",
        }, ensure_ascii=False))
    assert len(out) == len(qlines), f"count mismatch {len(out)} != {len(qlines)}"
    res = QROOT / cell / "results.jsonl"
    res.write_text("\n".join(out) + "\n", encoding="utf-8")
    print(f"{cell}: wrote {len(out)} results, mean_score={ssum/len(out):.4f}")
