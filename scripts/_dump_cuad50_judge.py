"""Dump CUAD-50 judge-queue entries into chunked JSON files for 1-by-1 Claude judging.

Usage:
    python scripts/_dump_cuad50_judge.py <cell-dir-name> [--all|--new] [--chunk 110]

<cell-dir-name> e.g. cuad__v4t-canonical__batch__seed42
  --all : dump every queue entry (use for BATCH cells — predictions are recomputed
          at 50-doc scale, so prior judgments are stale).
  --new : dump only entries whose qid is absent from results.jsonl (use for ONLINE
          cells — early-doc predictions are unchanged).

Writes results/stage3/_judge_workdir/cuad50/<cell>__chunkNN.json, each a list of
{idx, qid, question, gold, pred}. idx is the global entry index within the cell.
"""
import argparse, json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QDIR = ROOT / "results" / "stage3" / "judge_queue"
OUT = ROOT / "results" / "stage3" / "_judge_workdir" / "cuad50"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cell")
    ap.add_argument("--all", action="store_true")
    ap.add_argument("--new", action="store_true")
    ap.add_argument("--chunk", type=int, default=110)
    args = ap.parse_args()

    qpath = QDIR / args.cell / "queue.jsonl"
    rpath = QDIR / args.cell / "results.jsonl"
    queue = [json.loads(l) for l in qpath.read_text("utf-8").splitlines() if l.strip()]
    judged = set()
    if rpath.exists():
        for l in rpath.read_text("utf-8").splitlines():
            if l.strip():
                judged.add(json.loads(l)["qid"])

    mode = "all" if args.all else "new"
    rows = []
    for i, e in enumerate(queue):
        if mode == "new" and e["qid"] in judged:
            continue
        rows.append({
            "idx": i,
            "qid": e["qid"],
            "question": (e.get("scoped_question") or e.get("question") or "")[:400],
            "gold": e.get("gold_answer"),
            "pred": e.get("predicted"),
        })

    OUT.mkdir(parents=True, exist_ok=True)
    n_chunks = 0
    for c in range(0, len(rows), args.chunk):
        chunk = rows[c:c + args.chunk]
        fn = OUT / f"{args.cell}__chunk{c // args.chunk:02d}.json"
        fn.write_text(json.dumps(chunk, indent=1, ensure_ascii=False), "utf-8")
        n_chunks += 1
    print(f"{args.cell}: queue={len(queue)} judged={len(judged)} dumped={len(rows)} "
          f"({mode}) in {n_chunks} chunks of <= {args.chunk}")


if __name__ == "__main__":
    main()
