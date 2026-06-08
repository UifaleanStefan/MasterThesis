"""Phase 1.9 — QASPER v4t-tuned online (94 entries, per-doc tuned middle config)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__v4t-tuned__online__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__v4t-tuned__online__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.5, "PRED 'labeling unlabeled instances + constraining + priors on parameters' touches features partial."),
    ("doc0_qa1", 1.0, "PRED exact match (3 regularization terms incl. neutral features)."),
    ("doc0_qa2", 1.0, "PRED 'sentiment + web-page + science + medical/healthcare' matches gold's themes."),
    ("doc0_qa3", 0.75, "PRED 'handle bias + unbalanced + imperfect knowledge' captures gold's unbalanced focus."),
    ("doc1_qa0", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc1_qa1", 0.0, "PRED 'argument components in Web discourse' vague."),
    ("doc1_qa2", 0.0, "PRED 'supervised/semi-supervised' vague."),
    ("doc1_qa3", 0.5, "PRED 'Web data + user-generated across registers' partial."),
    ("doc1_qa4", 0.0, "PRED hallucinated answer when gold unanswerable."),
    ("doc1_qa5", 0.5, "PRED 'variety + noisy + unrestricted' partial."),
    ("doc2_qa0", 0.0, "PRED 'garg2012unsupervised' WRONG."),
    ("doc2_qa1", 0.75, "PRED 'Crosslingual latent variables' matches CLV concept."),
    ("doc2_qa2", 0.0, "PRED refuses ('not mentioned')."),
    ("doc2_qa3", 0.5, "PRED 'small improvements + biggest impact monolingual' implies No."),
    ("doc2_qa4", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc2_qa5", 0.5, "PRED 'Bayesian model for each language' partial."),
    ("doc2_qa6", 1.0, "PRED 'does not explicitly state' matches unanswerable."),
    ("doc3_qa0", 0.0, "PRED 'Yes valid' when gold unanswerable."),
    ("doc4_qa0", 0.0, "PRED '100 parallel sentences' WRONG."),
    ("doc4_qa1", 1.0, "PRED 'dependency edge English → Hindi aligned + Dual Decomposition' matches gold."),
    ("doc5_qa0", 1.0, "PRED 'CrowdFlower + expert annotators' matches gold."),
    ("doc5_qa1", 0.0, "PRED refuses ('not specify')."),
    ("doc6_qa0", 0.0, "PRED 'do not specify' REFUSAL."),
    ("doc6_qa1", 0.5, "PRED 'SERA + content relevance between summaries' partial."),
    ("doc6_qa2", 0.5, "PRED 'very different correlations' partial."),
    ("doc6_qa3", 0.5, "PRED 'pyramid structure tiered by importance' touches higher-tiers."),
    ("doc6_qa4", 0.75, "PRED 'ROUGE reliable refutes' matches gold."),
    ("doc7_qa0", 0.0, "PRED refuses ('do not mention')."),
    ("doc7_qa1", 0.0, "PRED 'word-level + character-level for embeddings' WRONG; gold='None'."),
    ("doc8_qa0", 0.75, "PRED 'discussion points adopted + recall + drop' matches gold's 3 aspects."),
    ("doc8_qa1", 1.0, "PRED 'IQ2 108 debates' matches Intelligence Squared."),
    ("doc9_qa0", 0.75, "PRED 'ratio of correct translations matches' Accuracy concept."),
    ("doc10_qa0", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc10_qa1", 0.75, "PRED 'Social Honeypot 19,276/22,223, high quality' detailed match."),
    ("doc10_qa2", 0.75, "PRED 'LDA topic distribution patterns local+global' matches gold."),
    ("doc11_qa0", 1.0, "PRED 'Yes paper clearly establishes' matches."),
    ("doc11_qa1", 1.0, "PRED 'addressed earlier, established task' matches."),
    ("doc11_qa2", 1.0, "PRED 'simple word-level encoder for tweets + whitespace + 20K' matches gold."),
    ("doc11_qa3", 0.75, "PRED 'do not mention other specific tasks besides hashtags' matches 'None'."),
    ("doc11_qa4", 1.0, "PRED 'simple word-level encoder for tweets + 20K' matches gold."),
    ("doc12_qa0", 1.0, "PRED 'does not specify evaluation methods' matches unanswerable."),
    ("doc12_qa1", 1.0, "PRED 'over 30,000 images' matches gold 30,000."),
    ("doc12_qa2", 0.5, "PRED 'manual detection + browser annotation tool' partial."),
    ("doc12_qa3", 0.25, "PRED 'men managers/women subordinates' partial; gold=Ethnic bias."),
    ("doc13_qa0", 0.0, "PRED refuses ('not specify')."),
    ("doc13_qa1", 1.0, "PRED 'SemEval 2010 task 8' exact match."),
    ("doc13_qa2", 1.0, "PRED 'CNN and RNN models predict class with most votes + tie' explicitly matches gold's voting scheme."),
    ("doc13_qa3", 0.0, "PRED 'bi-directional' but gold='uni-directional'."),
    ("doc13_qa4", 0.5, "PRED 'extended middle context, all sentence parts' partial."),
    ("doc14_qa0", 0.5, "PRED 'initial WMT16 models before improvements' partial."),
    ("doc15_qa0", 1.0, "PRED 'UD1.2 corpora 16 languages: full list' matches gold."),
    ("doc15_qa1", 1.0, "PRED 'full 16-language list' matches gold."),
    ("doc16_qa0", 0.0, "PRED 'No Chinese-English' but gold='Yes only Chinese-English' — Y/N flip."),
    ("doc17_qa0", 0.0, "PRED 'do not indicate'."),
    ("doc17_qa1", 0.0, "PRED 'do not specify' REFUSAL."),
    ("doc17_qa2", 0.75, "PRED 'Winograd schemas + ambiguous pronouns' matches WSC."),
    ("doc17_qa3", 0.0, "PRED 'distinctions between pronouns' vague; gold=English."),
    ("doc18_qa0", 0.0, "PRED 'deep contextual + bidirectional LSTM' missing SOTA tool names."),
    ("doc18_qa1", 0.75, "PRED 'CoNLL2003 and GENIA data sets' includes gold GENIA."),
    ("doc19_qa0", 0.0, "PRED 'random forest classifier' WRONG; gold=LSTM+VGG16."),
    ("doc19_qa1", 0.25, "PRED 'image+question with agree/disagree' partial."),
    ("doc20_qa0", 0.0, "PRED 'abortion, climate change, cooking' WRONG; gold=politics/business/science/AskReddit."),
    ("doc20_qa1", 0.25, "PRED 'predictive model + linguistic + user behaviors' vague."),
    ("doc21_qa0", 0.5, "PRED 'co-occurrence frequencies + contextual + semantic similarity' partial."),
    ("doc21_qa1", 1.0, "PRED 'do not specify' matches unanswerable."),
    ("doc21_qa2", 1.0, "PRED 'CBOW + Skip-gram approach' includes Skip-gram."),
    ("doc22_qa0", 0.0, "PRED 'iteratively adds best' WRONG; gold='simply averaging'."),
    ("doc22_qa1", 0.5, "PRED 'much larger margin' direction-only correct."),
    ("doc22_qa2", 0.5, "PRED 'human study, majority answerable' partial; missing 50/50 detail."),
    ("doc23_qa0", 0.25, "PRED 'four levels mentioned no details' vague."),
    ("doc23_qa1", 0.25, "PRED 'five keyphrase extraction models' vague; missing names."),
    ("doc23_qa2", 1.0, "PRED '244 scientific articles' exact match."),
    ("doc24_qa0", 0.25, "PRED lists features (unigrams/emoticons/kNN/SVM) vague approximation."),
    ("doc24_qa1", 0.75, "PRED 'Sarcasm Detector + Semeval 2014 Twitter' includes gold."),
    ("doc24_qa2", 0.75, "PRED 'features from baseline CNN' matches gold."),
    ("doc25_qa0", 1.0, "PRED 'do not mention seasonality' matches unanswerable."),
    ("doc25_qa1", 0.0, "PRED refuses ('not specify dimension')."),
    ("doc25_qa2", 0.0, "PRED '3,216 tweets' WRONG; gold=250K."),
    ("doc26_qa0", 1.0, "PRED 'German-English' exact match."),
    ("doc26_qa1", 1.0, "PRED 'IMDb movie review dataset' exact."),
    ("doc26_qa2", 1.0, "PRED 'minimalist recurrent pooling... dynamic average pooling' explicitly names gold's pooling."),
    ("doc27_qa0", 0.5, "PRED 'competitive results without handcrafted resources' partial."),
    ("doc27_qa1", 0.75, "PRED 'Affective Text + Fairy Tales + ISEAR' includes gold."),
    ("doc27_qa2", 0.0, "PRED 'subsets chosen by performance' vague."),
    ("doc28_qa0", 0.0, "PRED 'single-topic dataset' vague; gold=anti-nuclear-power."),
    ("doc28_qa1", 0.0, "PRED refuses ('not specify layers')."),
    ("doc28_qa2", 0.75, "PRED 'abortion + gay rights + Obama + marijuana' includes abortion."),
    ("doc28_qa3", 0.0, "PRED 'privately-owned single-topic unbalanced' missing 32,595."),
    ("doc28_qa4", 0.0, "PRED 'Yes' but gold='No'."),
    ("doc28_qa5", 0.25, "PRED 'Majority/SVM/CNN/RCNN' partial; missing n-grams."),
    ("doc29_qa0", 0.0, "PRED 'No' definitive when gold unanswerable."),
    ("doc29_qa1", 0.0, "PRED 'No' but gold='Yes'."),
    ("doc29_qa2", 0.0, "PRED 'No' but gold='Yes'."),
    ("doc29_qa3", 0.0, "PRED refuses ('not mentioned')."),
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
    print(f"qasper v4t-tuned online added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
