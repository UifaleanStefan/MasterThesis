"""Phase 1.9 Protocol A — HotpotQA 6 remaining cells (60 entries total)."""
from __future__ import annotations
import json
from pathlib import Path

JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# (results_path, [(qid, score, rationale), ...])
CELLS = [
    (
        "results/stage3/judge_queue/hotpotqa__attention-corpus-tuned__batch__seed42/results.jsonl",
        [
            ("hotpotqa__attention-corpus-tuned__batch__doc0_qa0__seed42", 1.0,
             "PRED 'Yes, both American' matches gold='yes'."),
            ("hotpotqa__attention-corpus-tuned__batch__doc1_qa0__seed42", 0.75,
             "PRED 'Shirley Temple held...Chief of Protocol' includes gold=Chief of Protocol."),
            ("hotpotqa__attention-corpus-tuned__batch__doc2_qa0__seed42", 1.0,
             "PRED 'Animorphs' matches gold='Animorphs'."),
            ("hotpotqa__attention-corpus-tuned__batch__doc3_qa0__seed42", 1.0,
             "PRED 'No, different neighborhoods (Laleli vs Ortaköy)' matches gold='no'."),
            ("hotpotqa__attention-corpus-tuned__batch__doc4_qa0__seed42", 0.75,
             "PRED 'Greenwich Village' matches gold core; missing NYC qualifier."),
            ("hotpotqa__attention-corpus-tuned__batch__doc5_qa0__seed42", 1.0,
             "PRED 'YG Entertainment' matches gold."),
            ("hotpotqa__attention-corpus-tuned__batch__doc6_qa0__seed42", 1.0,
             "PRED 'Eenasul Fateh' matches gold."),
            ("hotpotqa__attention-corpus-tuned__batch__doc7_qa0__seed42", 0.75,
             "PRED '4,000 (3,677 seated)' includes gold=3,677 seated."),
            ("hotpotqa__attention-corpus-tuned__batch__doc8_qa0__seed42", 1.0,
             "PRED 'Terry Richardson is older' matches gold='Terry Richardson'."),
            ("hotpotqa__attention-corpus-tuned__batch__doc9_qa0__seed42", 1.0,
             "PRED 'Yes, both from United States' matches gold='yes'."),
        ],
    ),
    (
        "results/stage3/judge_queue/hotpotqa__bm25-corpus__batch__seed42/results.jsonl",
        [
            ("hotpotqa__bm25-corpus__batch__doc0_qa0__seed42", 1.0,
             "PRED 'Yes, both American' matches gold='yes'."),
            ("hotpotqa__bm25-corpus__batch__doc1_qa0__seed42", 0.0,
             "PRED refuses; gold=Chief of Protocol."),
            ("hotpotqa__bm25-corpus__batch__doc2_qa0__seed42", 1.0,
             "PRED 'Animorphs' matches gold."),
            ("hotpotqa__bm25-corpus__batch__doc3_qa0__seed42", 1.0,
             "PRED 'No, different neighborhoods' matches gold='no'."),
            ("hotpotqa__bm25-corpus__batch__doc4_qa0__seed42", 0.5,
             "PRED 'Adriana Trigiani is in New York City' partial; gold=Greenwich Village, NYC."),
            ("hotpotqa__bm25-corpus__batch__doc5_qa0__seed42", 1.0,
             "PRED 'YG Entertainment' matches gold."),
            ("hotpotqa__bm25-corpus__batch__doc6_qa0__seed42", 1.0,
             "PRED 'Eenasul Fateh' matches gold."),
            ("hotpotqa__bm25-corpus__batch__doc7_qa0__seed42", 0.25,
             "PRED '3,500' wrong number; gold=3,677 seated."),
            ("hotpotqa__bm25-corpus__batch__doc8_qa0__seed42", 1.0,
             "PRED 'Terry Richardson is older' matches gold."),
            ("hotpotqa__bm25-corpus__batch__doc9_qa0__seed42", 1.0,
             "PRED 'Yes, both from United States' matches gold='yes'."),
        ],
    ),
    (
        "results/stage3/judge_queue/hotpotqa__dump-all__batch__seed42/results.jsonl",
        [
            ("hotpotqa__dump-all__batch__doc0_qa0__seed42", 0.0,
             "PRED refuses; gold='yes'."),
            ("hotpotqa__dump-all__batch__doc1_qa0__seed42", 0.0,
             "PRED refuses; gold=Chief of Protocol."),
            ("hotpotqa__dump-all__batch__doc2_qa0__seed42", 0.0,
             "PRED 'The Space Between Worlds' hallucinated; gold=Animorphs."),
            ("hotpotqa__dump-all__batch__doc3_qa0__seed42", 0.0,
             "PRED refuses; gold='no'."),
            ("hotpotqa__dump-all__batch__doc4_qa0__seed42", 0.0,
             "PRED refuses; gold=Greenwich Village, New York City."),
            ("hotpotqa__dump-all__batch__doc5_qa0__seed42", 0.0,
             "PRED refuses; gold=YG Entertainment."),
            ("hotpotqa__dump-all__batch__doc6_qa0__seed42", 0.0,
             "PRED refuses; gold=Eenasul Fateh."),
            ("hotpotqa__dump-all__batch__doc7_qa0__seed42", 0.0,
             "PRED refuses; gold=3,677 seated."),
            ("hotpotqa__dump-all__batch__doc8_qa0__seed42", 1.0,
             "PRED 'Terry Richardson is older' matches gold."),
            ("hotpotqa__dump-all__batch__doc9_qa0__seed42", 1.0,
             "PRED 'Yes, both from United States' matches gold='yes'."),
        ],
    ),
    (
        "results/stage3/judge_queue/hotpotqa__v4t-canonical__online__seed42/results.jsonl",
        [
            ("hotpotqa__v4t-canonical__online__doc0_qa0__seed42", 0.5,
             "PRED 'No, ...both American' contradictory phrasing; factual content correct but answer direction confused."),
            ("hotpotqa__v4t-canonical__online__doc1_qa0__seed42", 0.0,
             "PRED 'Janet Waldo, voice actress' wrong person+position; gold=Chief of Protocol (Shirley Temple)."),
            ("hotpotqa__v4t-canonical__online__doc2_qa0__seed42", 1.0,
             "PRED 'Animorphs' matches gold."),
            ("hotpotqa__v4t-canonical__online__doc3_qa0__seed42", 1.0,
             "PRED 'No, different neighborhoods' matches gold='no'."),
            ("hotpotqa__v4t-canonical__online__doc4_qa0__seed42", 1.0,
             "PRED 'Greenwich Village, New York City' matches gold."),
            ("hotpotqa__v4t-canonical__online__doc5_qa0__seed42", 1.0,
             "PRED 'YG Entertainment' matches gold."),
            ("hotpotqa__v4t-canonical__online__doc6_qa0__seed42", 1.0,
             "PRED 'Eenasul Fateh' matches gold."),
            ("hotpotqa__v4t-canonical__online__doc7_qa0__seed42", 0.75,
             "PRED '4,000 (3,677 seated)' includes gold=3,677 seated."),
            ("hotpotqa__v4t-canonical__online__doc8_qa0__seed42", 1.0,
             "PRED 'Terry Richardson is older' matches gold."),
            ("hotpotqa__v4t-canonical__online__doc9_qa0__seed42", 1.0,
             "PRED 'Yes' matches gold='yes'."),
        ],
    ),
    (
        "results/stage3/judge_queue/hotpotqa__v4t-corpus-tuned__online__seed42/results.jsonl",
        [
            ("hotpotqa__v4t-corpus-tuned__online__doc0_qa0__seed42", 1.0,
             "PRED 'Yes, both American' matches gold='yes'."),
            ("hotpotqa__v4t-corpus-tuned__online__doc1_qa0__seed42", 0.75,
             "PRED 'Shirley Temple held...Chief of Protocol' includes gold."),
            ("hotpotqa__v4t-corpus-tuned__online__doc2_qa0__seed42", 1.0,
             "PRED 'Animorphs' matches gold."),
            ("hotpotqa__v4t-corpus-tuned__online__doc3_qa0__seed42", 1.0,
             "PRED 'No, different neighborhoods' matches gold='no'."),
            ("hotpotqa__v4t-corpus-tuned__online__doc4_qa0__seed42", 0.75,
             "PRED 'Greenwich Village' matches core; missing NYC qualifier."),
            ("hotpotqa__v4t-corpus-tuned__online__doc5_qa0__seed42", 1.0,
             "PRED 'YG Entertainment' matches gold."),
            ("hotpotqa__v4t-corpus-tuned__online__doc6_qa0__seed42", 1.0,
             "PRED 'Eenasul Fateh' matches gold."),
            ("hotpotqa__v4t-corpus-tuned__online__doc7_qa0__seed42", 1.0,
             "PRED '3,677 people' matches gold=3,677 seated."),
            ("hotpotqa__v4t-corpus-tuned__online__doc8_qa0__seed42", 1.0,
             "PRED 'Terry Richardson is older' matches gold."),
            ("hotpotqa__v4t-corpus-tuned__online__doc9_qa0__seed42", 1.0,
             "PRED 'Yes, both from United States' matches gold='yes'."),
        ],
    ),
    (
        "results/stage3/judge_queue/hotpotqa__v4t-tuned__batch__seed42/results.jsonl",
        [
            ("hotpotqa__v4t-tuned__batch__doc0_qa0__seed42", 0.0,
             "PRED refuses; gold='yes'."),
            ("hotpotqa__v4t-tuned__batch__doc1_qa0__seed42", 0.0,
             "PRED refuses; gold=Chief of Protocol."),
            ("hotpotqa__v4t-tuned__batch__doc2_qa0__seed42", 0.0,
             "PRED refuses; gold=Animorphs."),
            ("hotpotqa__v4t-tuned__batch__doc3_qa0__seed42", 0.75,
             "PRED 'No, Esma Sultan in Ortaköy' correct No answer; Laleli Mosque not mentioned but answer direction right."),
            ("hotpotqa__v4t-tuned__batch__doc4_qa0__seed42", 0.5,
             "PRED 'New York City' partial; gold=Greenwich Village, New York City."),
            ("hotpotqa__v4t-tuned__batch__doc5_qa0__seed42", 0.0,
             "PRED refuses; gold=YG Entertainment."),
            ("hotpotqa__v4t-tuned__batch__doc6_qa0__seed42", 1.0,
             "PRED 'Eenasul Fateh' matches gold."),
            ("hotpotqa__v4t-tuned__batch__doc7_qa0__seed42", 0.0,
             "PRED refuses; gold=3,677 seated."),
            ("hotpotqa__v4t-tuned__batch__doc8_qa0__seed42", 1.0,
             "PRED 'Terry Richardson is older' matches gold."),
            ("hotpotqa__v4t-tuned__batch__doc9_qa0__seed42", 1.0,
             "PRED 'Yes' matches gold='yes'."),
        ],
    ),
]


def main() -> None:
    for results_path_str, judgments in CELLS:
        results_path = Path(results_path_str)
        existing: set[str] = set()
        if results_path.exists():
            for line in results_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    try: existing.add(json.loads(line)["qid"])
                    except: pass
        added = 0; total = 0.0; skipped = 0
        results_path.parent.mkdir(parents=True, exist_ok=True)
        with results_path.open("a", encoding="utf-8") as f:
            for qid, score, rationale in judgments:
                if qid in existing:
                    skipped += 1; continue
                f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                    "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
                added += 1; total += score
        cell = results_path.parent.name
        print(f"{cell}: added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
