"""Apply chunked agent judge scores into a CUAD-50 cell's results.jsonl.

Reads results/stage3/_judge_workdir/cuad50/scores__<cell>__chunk*.json (each a
list of {qid, score, rationale}) and writes results/stage3/judge_queue/<cell>/
results.jsonl with one judged line per queue entry. Overwrites results.jsonl
(used for full re-judge of batch cells). Errors loudly if any queue qid is
missing a score or any score is outside {0,.25,.5,.75,1}.

Usage: python scripts/_apply_cuad50_scores.py <cell>
"""
import glob, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QDIR = ROOT / "results" / "stage3" / "judge_queue"
WORK = ROOT / "results" / "stage3" / "_judge_workdir" / "cuad50"
VALID = {0.0, 0.25, 0.5, 0.75, 1.0}


def main():
    cell = sys.argv[1]
    queue = [json.loads(l) for l in (QDIR / cell / "queue.jsonl").read_text("utf-8").splitlines() if l.strip()]
    qids = [e["qid"] for e in queue]

    # Existing judgments (preserved for qids that have no fresh score — e.g. the
    # unchanged early-doc ONLINE entries when we only re-judged the new docs).
    existing = {}
    rpath = QDIR / cell / "results.jsonl"
    if rpath.exists():
        for l in rpath.read_text("utf-8").splitlines():
            if l.strip():
                e = json.loads(l)
                existing[e["qid"]] = (float(e["judge_score"]), str(e.get("rationale", "")))

    scores = {}
    files = sorted(glob.glob(str(WORK / f"scores__{cell}__chunk*.json")))
    if not files:
        sys.exit(f"ERROR: no score files for {cell}")
    for f in files:
        for r in json.loads(Path(f).read_text("utf-8")):
            s = float(r["score"])
            if s not in VALID:
                sys.exit(f"ERROR: invalid score {s} for {r['qid']} in {f}")
            scores[r["qid"]] = (s, str(r.get("rationale", ""))[:200])

    n_fresh = len(scores)
    n_kept = 0
    for q in qids:
        if q not in scores and q in existing:
            scores[q] = existing[q]
            n_kept += 1
    missing = [q for q in qids if q not in scores]
    if missing:
        sys.exit(f"ERROR: {len(missing)} queue qids have no score (e.g. {missing[:3]}). Aborting.")
    print(f"  ({n_fresh} fresh judgments, {n_kept} existing kept)")

    out = QDIR / cell / "results.jsonl"
    vals = []
    with out.open("w", encoding="utf-8") as fh:
        for q in qids:
            s, rat = scores[q]
            vals.append(s)
            fh.write(json.dumps({
                "qid": q, "judge_score": s, "rationale": rat,
                "judge_model": "claude-opus-4.7-1m", "judge_protocol": "v1",
            }, ensure_ascii=False) + "\n")
    print(f"{cell}: wrote {len(vals)} results, mean={sum(vals)/len(vals):.4f}")


if __name__ == "__main__":
    main()
