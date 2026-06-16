"""Build a stratified inter-rater sample across the six Stage-3 benchmarks.

Writes a BLIND re-rate file (qid, benchmark, question, gold, pred — NO score) in
chunks for parallel re-judging, plus a KEY file mapping qid -> pass-1 judge_score.
A second independent Claude pass re-rates the blind file; quadratic-weighted
Cohen's kappa between pass-1 and pass-2 is then computed by inter_rater.py.
"""
import json, glob, re, random
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
QDIR = ROOT / "results" / "stage3" / "judge_queue"
OUT = ROOT / "results" / "stage3" / "_judge_workdir" / "kappa"
random.seed(42)

BENCH = {
    "financebench": "FinanceBench", "finbench": "FinanceBench", "fb_": "FinanceBench",
    "qasper": "QASPER", "cuad": "CUAD", "hotpot": "HotpotQA", "hqa": "HotpotQA",
    "longmem": "LongMemEval", "lme": "LongMemEval", "narrative": "NarrativeQA", "nqa": "NarrativeQA",
}
PER_BENCH = 30   # ~180 total


def bench_of(cell: str):
    c = cell.lower()
    for k, v in BENCH.items():
        if k in c:
            return v
    return None


def main():
    pool = {b: [] for b in set(BENCH.values())}
    for rp in glob.glob(str(QDIR / "*" / "results.jsonl")):
        cell = Path(rp).parent.name
        b = bench_of(cell)
        if not b:
            continue
        qp = Path(rp).parent / "queue.jsonl"
        if not qp.exists():
            continue
        scores = {}
        for l in open(rp, encoding="utf-8"):
            if l.strip():
                e = json.loads(l); scores[e["qid"]] = float(e["judge_score"])
        for l in open(qp, encoding="utf-8"):
            if not l.strip():
                continue
            e = json.loads(l)
            qid = e["qid"]; pred = e.get("predicted")
            if qid in scores and pred:
                pool[b].append({
                    "qid": qid, "benchmark": b,
                    "question": (e.get("scoped_question") or e.get("question") or "")[:400],
                    "gold": e.get("gold_answer"), "pred": str(pred)[:600],
                    "_score1": scores[qid],
                })

    sample = []
    for b, items in pool.items():
        if not items:
            continue
        # stratify by pass-1 score bucket so the sample isn't all one value
        buckets = {}
        for it in items:
            buckets.setdefault(it["_score1"], []).append(it)
        picked, bvals = [], list(buckets.values())
        random.shuffle(bvals)
        i = 0
        while len(picked) < min(PER_BENCH, len(items)):
            bk = bvals[i % len(bvals)]
            if bk:
                picked.append(bk.pop(random.randrange(len(bk))))
            i += 1
            if all(len(x) == 0 for x in bvals):
                break
        sample.extend(picked)

    random.shuffle(sample)
    OUT.mkdir(parents=True, exist_ok=True)
    # key file (pass-1 scores)
    key = {it["qid"]: it["_score1"] for it in sample}
    (OUT / "kappa_key.json").write_text(json.dumps(key, indent=1))
    # blind chunks (no scores)
    blind = [{k: it[k] for k in ("qid", "benchmark", "question", "gold", "pred")} for it in sample]
    CH = 60
    n = 0
    for c in range(0, len(blind), CH):
        (OUT / f"kappa_blind__chunk{c//CH:02d}.json").write_text(
            json.dumps(blind[c:c+CH], indent=1, ensure_ascii=False), "utf-8")
        n += 1
    from collections import Counter
    print(f"sampled {len(sample)} across {sum(1 for v in pool.values() if v)} benchmarks; {n} blind chunks")
    print("per-benchmark:", Counter(it["benchmark"] for it in sample))
    print("pass-1 score dist:", Counter(it["_score1"] for it in sample))


if __name__ == "__main__":
    main()
