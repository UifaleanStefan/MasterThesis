"""Phase 1.9 — QASPER dump-all batch (94 entries, context-stuffing baseline).

NOTE: Entries 0-60 received fallback predictions (just first paragraph of
paper 0: "Robustly Leveraging Prior Knowledge in Text Classification...").
The 1,880-paragraph dump-all context (17K tokens) exceeds gpt-4o-mini's
practical context window. From entry 61 onward most predictions are
real "I cannot answer" refusals — consistent with the FB dump-all collapse
finding (FB batch_calib = 0.038, this QASPER dump-all = 0.037, identical
pattern at much larger corpus scale).
"""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__dump-all__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__dump-all__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# All entries 0-60: fallback (first paragraph of paper 0). Score 0.0.
# Entries 61-93: real refusals/answers.
FALLBACK_PREFIXES = [(f"doc{d}_qa{q}", 0.0, "FALLBACK: dump-all context exceeded gpt-4o-mini practical limit; prediction is fragment of paper 0's first paragraph.") for d, qs in
                     [(0, 4), (1, 6), (2, 7), (3, 1), (4, 2), (5, 2), (6, 5), (7, 2), (8, 2), (9, 1),
                      (10, 3), (11, 5), (12, 4), (13, 5), (14, 1), (15, 2), (16, 1), (17, 4), (18, 2), (19, 2)]
                     for q in range(qs)][:61]

JUDGMENTS: list[tuple[str, float, str]] = FALLBACK_PREFIXES + [
    ("doc20_qa0", 0.0, "PRED refuses ('do not contain information')."),
    ("doc20_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc21_qa0", 0.0, "PRED refuses ('do not contain any information')."),
    ("doc21_qa1", 1.0, "PRED 'do not contain information' matches gold unanswerable."),
    ("doc21_qa2", 0.0, "PRED refuses ('do not contain information')."),
    ("doc22_qa0", 0.0, "PRED refuses ('do not contain information')."),
    ("doc22_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc22_qa2", 0.0, "PRED 'balanced data + NMT mechanism + analyses' WRONG; gold=human study 50/50."),
    ("doc23_qa0", 0.0, "PRED refuses ('do not provide information')."),
    ("doc23_qa1", 0.0, "PRED refuses ('do not provide information')."),
    ("doc23_qa2", 0.0, "PRED refuses ('do not contain information')."),
    ("doc24_qa0", 0.0, "PRED refuses ('do not contain information')."),
    ("doc24_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc24_qa2", 0.0, "PRED refuses ('do not contain any information')."),
    ("doc25_qa0", 1.0, "PRED 'do not contain information' matches gold unanswerable."),
    ("doc25_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc25_qa2", 0.0, "PRED refuses ('do not contain information')."),
    ("doc26_qa0", 0.5, "PRED 'German to French, German to English, English to French' includes German-English."),
    ("doc26_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc26_qa2", 0.0, "PRED refuses ('do not contain any information')."),
    ("doc27_qa0", 0.0, "PRED refuses ('do not contain information')."),
    ("doc27_qa1", 0.0, "PRED refuses ('do not provide information')."),
    ("doc27_qa2", 0.0, "PRED refuses ('do not provide information about specific Facebook pages')."),
    ("doc28_qa0", 0.0, "PRED refuses ('do not contain information')."),
    ("doc28_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc28_qa2", 0.0, "PRED refuses ('do not contain information')."),
    ("doc28_qa3", 0.0, "PRED refuses ('do not contain information')."),
    ("doc28_qa4", 0.0, "PRED refuses ('do not contain information')."),
    ("doc28_qa5", 0.0, "PRED refuses ('do not contain information')."),
    ("doc29_qa0", 1.0, "PRED 'do not mention' matches gold unanswerable."),
    ("doc29_qa1", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc29_qa2", 0.0, "PRED 'No, focus on zero-resourced' but gold='Yes' — Y/N flip."),
    ("doc29_qa3", 0.0, "PRED 'German and French' WRONG; gold=English."),
]


def main() -> None:
    assert len(JUDGMENTS) == 94, f"Expected 94 entries, got {len(JUDGMENTS)}"
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try:
                    existing.add(json.loads(line)["qid"])
                except (json.JSONDecodeError, KeyError):
                    pass
    added = skipped = 0
    total = 0.0
    fallback_count = sum(1 for _, _, r in JUDGMENTS if "FALLBACK" in r)
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        for suffix, score, rationale in JUDGMENTS:
            qid = QID_PREFIX + suffix + QID_SUFFIX
            if qid in existing:
                skipped += 1
                continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL},
                               ensure_ascii=False) + "\n")
            added += 1
            total += score
            existing.add(qid)
    print(f"qasper dump-all batch added={added} skipped={skipped} mean={total/added if added else 0:.4f}")
    print(f"  {fallback_count}/94 entries are context-overflow fallbacks (paper-0 first-paragraph fragments).")
    real = 94 - fallback_count
    print(f"  Real-answer-only mean (excluding fallbacks): {total/real if real else 0:.4f} on {real} entries.")


if __name__ == "__main__":
    main()
