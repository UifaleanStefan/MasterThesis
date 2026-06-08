"""Phase 1.9 — HotpotQA v4t-corpus-tuned batch (10 entries, perfect score)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "hotpotqa__v4t-corpus-tuned__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/hotpotqa__v4t-corpus-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 1.0, "PRED 'Yes, both American' matches gold='yes'."),
    ("doc1_qa0", 1.0, "PRED 'Shirley Temple held Chief of Protocol' includes gold."),
    ("doc2_qa0", 1.0, "PRED 'Animorphs' exact match."),
    ("doc3_qa0", 1.0, "PRED 'No different neighborhoods' matches gold='no'."),
    ("doc4_qa0", 1.0, "PRED 'Greenwich Village' matches gold."),
    ("doc5_qa0", 1.0, "PRED 'WINNER formed by YG Entertainment' includes gold."),
    ("doc6_qa0", 1.0, "PRED 'Eenasul Fateh' exact match."),
    ("doc7_qa0", 1.0, "PRED 'Androscoggin Bank Colisée seat 3,677' includes gold."),
    ("doc8_qa0", 1.0, "PRED 'Terry Richardson is older' matches gold."),
    ("doc9_qa0", 1.0, "PRED 'Yes both from US' matches gold='yes'."),
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
    print(f"hqa v4t-corpus-tuned batch added={added} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
