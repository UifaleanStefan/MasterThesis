"""Phase 4 cross-vendor finishing — QASPER p4_k8 v4-tuned seed=42 (16 remaining entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k8__qasper__v4-tuned__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q001", 0.75, "PRED 'does not specify evaluation methods' implicit match to gold unanswerable."),
    ("q005", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q009", 0.0, "PRED 'No, Chinese-to-English' WRONG; gold='Yes'."),
    ("q010", 0.25, "PRED 'Czech, French, Italian, English, Danish' partial; gold=full UD1.2 16-language list."),
    ("q012", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q013", 0.25, "PRED 'demographics, diagnosis history, symptoms/signs' partial; gold has 10 categories."),
    ("q030", 0.0, "PRED 'No, not directly generalizable' but gold='Yes' — Y/N flip."),
    ("q035", 0.75, "PRED 'De-En, Ja-En, Ro-En' includes gold De-En."),
    ("q047", 0.0, "PRED 'Libertarianism, ronpaul, ukpolitics...' WRONG; gold=politics, business, science, AskReddit."),
    ("q050", 0.0, "PRED refuses; gold=CJFA encoder."),
    ("q051", 0.0, "PRED 'baseline b3 and #10' WRONG; gold=pivot-based translation."),
    ("q056", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q069", 0.0, "PRED 'Yes, MTL' but gold='No' — Y/N flip."),
    ("q082", 0.0, "PRED 'biLSTM F1 84-94%' WRONG; gold=list of 8 named NER systems."),
    ("q092", 0.75, "PRED 'MLP and DNNs' includes gold=MLP."),
    ("q094", 0.0, "PRED 'No, social media + news' hallucinated; gold unanswerable."),
]


def main() -> None:
    assert len(JUDGMENTS) == 16
    qid_prefix = "p4_k8__qasper__v4-tuned__seed42__"
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: existing.add(json.loads(line)["qid"])
                except: pass
    added = 0; total = 0.0; skipped = 0
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        for suffix, score, rationale in JUDGMENTS:
            qid = qid_prefix + suffix
            if qid in existing:
                skipped += 1; continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"qasper p4_k8 v4-tuned seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
