"""Phase 1.9 — QASPER v4t-canonical batch (94 entries, baseline for 4-shift comparison)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__v4t-canonical__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__v4t-canonical__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.25, "PRED 'prior knowledge in NLP' vague; gold=labeled features."),
    ("doc0_qa1", 0.0, "PRED refuses ('do not specify')."),
    ("doc0_qa2", 0.0, "PRED refuses ('does not specify NLP tasks')."),
    ("doc0_qa3", 0.25, "PRED 'maintain effectiveness with prior knowledge' vague; missing unbalanced specifics."),
    ("doc1_qa0", 0.0, "PRED 'No multiple domains' but gold='Yes only English' — Y/N flip."),
    ("doc1_qa1", 0.0, "PRED 'argument components in Web discourse' vague; missing claim/premise/etc."),
    ("doc1_qa2", 0.0, "PRED refuses ('not detailed'); gold=Structural SVM."),
    ("doc1_qa3", 0.25, "PRED '90k tokens 340 docs' partial source description; missing newswire/blog detail."),
    ("doc1_qa4", 0.0, "PRED hallucinated answer when gold unanswerable."),
    ("doc1_qa5", 0.5, "PRED 'variety + noisy nature' partially captures linguistic variability."),
    ("doc2_qa0", 0.25, "PRED 'monolingual model' vague; gold=lang2011unsupervised baseline."),
    ("doc2_qa1", 0.5, "PRED 'alignments between roles across languages' partial."),
    ("doc2_qa2", 0.0, "PRED 'TED + WMT corpora' WRONG; gold=CoNLL 2009 EN/DE."),
    ("doc2_qa3", 0.0, "PRED 'Yes improves' but gold='No' — Y/N flip."),
    ("doc2_qa4", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc2_qa5", 0.5, "PRED 'models for each language + latent variables' partial."),
    ("doc2_qa6", 1.0, "PRED 'does not specify' matches unanswerable."),
    ("doc3_qa0", 0.0, "PRED 'Yes valid' when gold unanswerable."),
    ("doc4_qa0", 0.0, "PRED '3 million sentences' WRONG; gold=1500."),
    ("doc4_qa1", 0.5, "PRED 'dual decomposition + alignments + prepositions' partial."),
    ("doc5_qa0", 0.0, "PRED 'natural dialogs' WRONG; gold=CrowdFlower."),
    ("doc5_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc6_qa0", 0.0, "PRED 'No French/German' but gold='Yes only English' — Y/N flip."),
    ("doc6_qa1", 0.5, "PRED 'content relevance between summaries' partial; missing IR mechanism."),
    ("doc6_qa2", 0.5, "PRED 'different correlations' partial."),
    ("doc6_qa3", 0.0, "PRED 'not explicitly mentioned' REFUSAL."),
    ("doc6_qa4", 0.75, "PRED 'ROUGE is reliable... refutes' matches gold."),
    ("doc7_qa0", 0.0, "PRED refuses ('do not mention')."),
    ("doc7_qa1", 0.0, "PRED talks about character-level params; gold says 'None' for baselines."),
    ("doc8_qa0", 0.5, "PRED '3 aspects: adopted points + recall + drop' partial overlap with gold's 3 aspects."),
    ("doc8_qa1", 1.0, "PRED 'IQ2 108 debates' matches Intelligence Squared Debates."),
    ("doc9_qa0", 0.0, "PRED 'BLEU' WRONG; gold=Accuracy."),
    ("doc10_qa0", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc10_qa1", 0.5, "PRED 'Honeypot dataset, high quality' partial — 1 of 2 datasets."),
    ("doc10_qa2", 0.75, "PRED 'LDA-based classification with local+global features' matches gold."),
    ("doc11_qa0", 1.0, "PRED 'Yes informal language/spelling errors' matches."),
    ("doc11_qa1", 0.75, "PRED 'recognized task rather than entirely new' matches 'established task'."),
    ("doc11_qa2", 0.5, "PRED 'traditional NLP word-level approach' partial."),
    ("doc11_qa3", 0.75, "PRED 'do not mention other tasks' matches 'None'."),
    ("doc11_qa4", 0.0, "PRED refuses ('not explicitly named')."),
    ("doc12_qa0", 1.0, "PRED 'does not provide specific evaluation methods' matches unanswerable."),
    ("doc12_qa1", 0.0, "PRED refuses ('not specified'); gold=30,000."),
    ("doc12_qa2", 0.0, "PRED refuses ('do not provide specific methods')."),
    ("doc12_qa3", 0.0, "PRED 'biases not solely based on images' vague; gold=Ethnic bias."),
    ("doc13_qa0", 0.0, "PRED refuses ('do not contain information')."),
    ("doc13_qa1", 1.0, "PRED 'SemEval 2010 relation classification' matches."),
    ("doc13_qa2", 0.0, "PRED 'does not provide specific details'."),
    ("doc13_qa3", 0.0, "PRED 'bi-directional RNN' but gold='uni-directional'."),
    ("doc13_qa4", 0.25, "PRED 'extended middle context' vague; missing convolution+max-pooling."),
    ("doc14_qa0", 0.25, "PRED 'NMT on TED English-German' missing attentional encoder-decoder."),
    ("doc15_qa0", 0.0, "PRED 'TED + European Parliament + CommonCrawl' WRONG; gold=UD treebanks 16 languages."),
    ("doc15_qa1", 0.25, "PRED 'English, German, French' only 3 of gold's 16 languages."),
    ("doc16_qa0", 0.0, "PRED 'No multiple pairs' but gold='Yes only one' — Y/N flip."),
    ("doc17_qa0", 0.0, "PRED 'do not mention experiments'."),
    ("doc17_qa1", 0.0, "PRED 'do not mention whether they collected'."),
    ("doc17_qa2", 0.0, "PRED 'TED + European Parliament' WRONG; gold=WSC collection."),
    ("doc17_qa3", 0.5, "PRED 'English and German' partial; gold=English only."),
    ("doc18_qa0", 0.0, "PRED gives characteristic description; missing SOTA tool names."),
    ("doc18_qa1", 0.0, "PRED 'Reuters and Medline' WRONG; gold=GENIA Corpus."),
    ("doc19_qa0", 0.0, "PRED refuses ('do not specify architecture')."),
    ("doc19_qa1", 0.25, "PRED 'predict crowd agreement' vague."),
    ("doc20_qa0", 0.25, "PRED 'politics, religion, hobbies' overlaps 'politics' but adds wrong topics."),
    ("doc20_qa1", 0.25, "PRED 'predictive model with features' vague."),
    ("doc21_qa0", 0.0, "PRED 'pairwise semantic similarity from taxonomy' WRONG."),
    ("doc21_qa1", 1.0, "PRED 'do not contain information' matches unanswerable."),
    ("doc21_qa2", 0.25, "PRED 'word embedding methods' vague; gold=Skip-gram."),
    ("doc22_qa0", 0.0, "PRED refuses ('do not provide information')."),
    ("doc22_qa1", 0.5, "PRED 'much larger margin' direction-only correct."),
    ("doc22_qa2", 0.5, "PRED 'own human study' partial; missing 50/50 split detail."),
    ("doc23_qa0", 0.0, "PRED 'level 2 and 3' WRONG; gold=raw text (level 0/1)."),
    ("doc23_qa1", 0.0, "PRED 'TopicRank/WINGNUS/KP-miner' WRONG; gold=Stanford CoreNLP/OCR/ParsCIT."),
    ("doc23_qa2", 0.0, "PRED refuses ('not specified'); gold=244."),
    ("doc24_qa0", 0.0, "PRED refuses ('does not specify SOTA models')."),
    ("doc24_qa1", 0.0, "PRED refuses ('not mentioned')."),
    ("doc24_qa2", 0.0, "PRED refuses ('does not specify')."),
    ("doc25_qa0", 1.0, "PRED 'do not mention seasonality' matches unanswerable."),
    ("doc25_qa1", 0.0, "PRED refuses ('does not specify dimension')."),
    ("doc25_qa2", 0.0, "PRED 'aggregate sentiment values' missing 250K tweets."),
    ("doc26_qa0", 0.5, "PRED 'English-German and French-German' includes gold German-English."),
    ("doc26_qa1", 1.0, "PRED 'IMDb movie review dataset' exact match."),
    ("doc26_qa2", 0.25, "PRED 'elementwise recurrent pooling' missing dynamic-average specifics."),
    ("doc27_qa0", 0.5, "PRED 'competitive results' partial; missing F1 numbers."),
    ("doc27_qa1", 0.0, "PRED refuses ('do not provide information')."),
    ("doc27_qa2", 0.0, "PRED 'specific pages not detailed'."),
    ("doc28_qa0", 0.0, "PRED 'social media stance classification' WRONG; gold=anti-nuclear-power."),
    ("doc28_qa1", 0.0, "PRED refuses ('does not specify layers')."),
    ("doc28_qa2", 0.75, "PRED 'abortion and gay rights' includes gold abortion."),
    ("doc28_qa3", 0.0, "PRED refuses ('not specified'); gold=32,595 posts."),
    ("doc28_qa4", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc28_qa5", 0.0, "PRED 'user/topic/comment withheld' WRONG; gold=SVM with n-grams."),
    ("doc29_qa0", 0.0, "PRED 'No' definitive answer when gold unanswerable."),
    ("doc29_qa1", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc29_qa2", 1.0, "PRED 'Yes English-German' matches gold='Yes'."),
    ("doc29_qa3", 0.75, "PRED 'English and German' includes gold English."),
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
    print(f"qasper v4t-canonical batch added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
