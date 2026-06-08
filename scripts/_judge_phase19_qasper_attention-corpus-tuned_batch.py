"""Phase 1.9 — QASPER attention-corpus-tuned batch (94 entries).

NOTE: This cell was compromised by OpenAI API quota exhaustion mid-run.
Entries 32+ (most after doc10) received fallback predictions = paper title +
first paragraph of source paper instead of real LLM answers. These score 0.0
because they are not real answers; the score does NOT reflect
attention-corpus-tuned's actual capability. Documented as a known
limitation in the chapter §6.5.2 footnote.
"""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__attention-corpus-tuned__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__attention-corpus-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.25, "PRED 'prior knowledge to guide learning' vague."),
    ("doc0_qa1", 1.0, "PRED matches gold's neutral-features regularization + 2 more terms."),
    ("doc0_qa2", 0.5, "PRED 'text classification' partial."),
    ("doc0_qa3", 0.5, "PRED 'handle bias + variations' partial."),
    ("doc1_qa0", 0.0, "PRED refuses ('do not specify')."),
    ("doc1_qa1", 0.0, "PRED 'argument components in Web discourse' vague."),
    ("doc1_qa2", 0.0, "PRED 'supervised/semi-supervised' vague; gold=Structural SVM."),
    ("doc1_qa3", 0.0, "PRED 'variety of sources' vague."),
    ("doc1_qa4", 0.0, "PRED hallucinated answer when gold unanswerable."),
    ("doc1_qa5", 0.5, "PRED 'variety of formats/styles + less formal/ambiguous' partial."),
    ("doc2_qa0", 0.0, "PRED 'garg2012unsupervised' WRONG; gold=lang2011."),
    ("doc2_qa1", 0.75, "PRED 'CLVs' matches gold's concept."),
    ("doc2_qa2", 0.0, "PRED refuses ('not mentioned')."),
    ("doc2_qa3", 0.5, "PRED 'small improvements + monolingual' implies No."),
    ("doc2_qa4", 0.0, "PRED 'No additional CLVs not one per pair' but gold='Yes'."),
    ("doc2_qa5", 0.5, "PRED 'Bayesian model for each language' partial."),
    ("doc2_qa6", 1.0, "PRED 'does not explicitly state' matches unanswerable."),
    ("doc3_qa0", 0.0, "PRED 'Yes valid' when gold unanswerable."),
    ("doc4_qa0", 0.0, "PRED '100 parallel sentences' WRONG; gold=1500."),
    ("doc4_qa1", 0.75, "PRED 'aligned words + corresponding edges + Dual Decomposition' matches gold."),
    ("doc5_qa0", 0.0, "PRED 'natural dialogs' WRONG; gold=CrowdFlower."),
    ("doc5_qa1", 0.0, "PRED refuses ('does not specify')."),
    ("doc6_qa0", 0.0, "PRED refuses ('do not specify')."),
    ("doc6_qa1", 0.5, "PRED 'SERA + content relevance between summaries' partial."),
    ("doc6_qa2", 0.5, "PRED 'very different correlations' partial."),
    ("doc6_qa3", 0.75, "PRED 'higher tiers indicate higher importance' matches gold's higher tiers."),
    ("doc6_qa4", 0.75, "PRED 'ROUGE is reliable refutes' matches gold."),
    ("doc7_qa0", 1.0, "PRED 'No do not mention' matches gold='No'."),
    ("doc7_qa1", 0.0, "PRED 'word-level + character-level encoder/decoder' WRONG; gold says 'None'."),
    ("doc8_qa0", 0.75, "PRED 'interaction + discussion points adopted + recall + drop' includes gold aspects."),
    ("doc8_qa1", 1.0, "PRED 'IQ2 108 debates' matches Intelligence Squared."),
    ("doc9_qa0", 0.5, "PRED 'accuracy of quazi-translation via Damerau-Levenshtein' partial — mentions accuracy."),
    # === API QUOTA ERROR ZONE — entries below are paper-title fallbacks not real answers ===
    ("doc10_qa0", 0.0, "FALLBACK: API quota exhausted, prediction is paper title + first paragraph fragment."),
    ("doc10_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc10_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc11_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc11_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc11_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc11_qa3", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc11_qa4", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc12_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc12_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc12_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc12_qa3", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc13_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc13_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc13_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc13_qa3", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc13_qa4", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc14_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc15_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc15_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc16_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc17_qa0", 0.0, "FALLBACK fragment: 'Winograd schemas as basis for challenges'."),
    ("doc17_qa1", 0.0, "FALLBACK fragment."),
    ("doc17_qa2", 0.5, "FALLBACK fragment 'Winograd schemas' partially matches gold WSC."),
    ("doc17_qa3", 0.0, "FALLBACK fragment about Winograd; gold=English."),
    ("doc18_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc18_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc19_qa0", 0.0, "FALLBACK: just paper title."),
    ("doc19_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc20_qa0", 0.0, "FALLBACK: research-question echo, not an answer."),
    ("doc20_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc21_qa0", 0.0, "FALLBACK: paper title fragment."),
    ("doc21_qa1", 0.0, "FALLBACK: paper title fragment (incidentally unanswerable but not a real reply)."),
    ("doc21_qa2", 0.0, "FALLBACK: paper title fragment."),
    ("doc22_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc22_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc22_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc23_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc23_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc23_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc24_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc24_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc24_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc25_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc25_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc25_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc26_qa0", 1.0, "PRED 'IWSLT German-English spoken-domain translation' includes German-English match."),
    ("doc26_qa1", 1.0, "PRED 'IMDb movie review dataset BIBREF17' exact match."),
    ("doc26_qa2", 0.25, "PRED 'convolution + pooling layers' missing dynamic-average specifics."),
    ("doc27_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc27_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc27_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc28_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc28_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc28_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc28_qa3", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc28_qa4", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc28_qa5", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc29_qa0", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc29_qa1", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc29_qa2", 0.0, "FALLBACK: paper title + first paragraph fragment."),
    ("doc29_qa3", 0.0, "FALLBACK: just title fragment."),
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
    print(f"qasper attention-corpus-tuned batch added={added} skipped={skipped} mean={total/added if added else 0:.4f}")
    print(f"  WARNING: {fallback_count}/94 entries are API-quota fallback predictions (score 0.0).")
    print(f"  Real-answer-only mean (excluding fallbacks): {total / (94 - fallback_count) if 94 - fallback_count else 0:.4f} on {94 - fallback_count} entries.")


if __name__ == "__main__":
    main()
