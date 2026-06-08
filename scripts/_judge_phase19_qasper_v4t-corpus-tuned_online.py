"""Phase 1.9 — QASPER v4t-corpus-tuned online (94 entries, ask-after-ingest mode)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__v4t-corpus-tuned__online__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__v4t-corpus-tuned__online__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 1.0, "PRED 'labeled features in sentiment classification' matches gold."),
    ("doc0_qa1", 1.0, "PRED exactly matches gold's three regularization terms."),
    ("doc0_qa2", 1.0, "PRED 'sentiment + web-page + science + medical/healthcare' matches gold."),
    ("doc0_qa3", 0.5, "PRED 'handle bias + variations' partial."),
    ("doc1_qa0", 0.0, "PRED refuses ('do not specify')."),
    ("doc1_qa1", 0.0, "PRED 'argument components in Web discourse' vague."),
    ("doc1_qa2", 0.0, "PRED 'supervised/semi-supervised' vague; gold=Structural SVM."),
    ("doc1_qa3", 0.25, "PRED 'variety of sources, user-generated Web discourse' vague."),
    ("doc1_qa4", 0.0, "PRED hallucinated answer when gold unanswerable."),
    ("doc1_qa5", 0.5, "PRED 'variety of communication styles + noisy + ambiguity/vagueness' partial."),
    ("doc2_qa0", 0.0, "PRED 'Bayesian model of Garg et al. (2012)' WRONG."),
    ("doc2_qa1", 0.75, "PRED 'CLVs' matches gold's concept."),
    ("doc2_qa2", 0.0, "PRED refuses ('not mentioned')."),
    ("doc2_qa3", 0.5, "PRED 'small improvements + monolingual biggest impact' implies No."),
    ("doc2_qa4", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc2_qa5", 0.5, "PRED 'Bayesian model for each language' partial."),
    ("doc2_qa6", 1.0, "PRED 'does not explicitly state' matches unanswerable."),
    ("doc3_qa0", 0.0, "PRED 'Yes valid' when gold unanswerable."),
    ("doc4_qa0", 0.0, "PRED '100 parallel sentences' WRONG."),
    ("doc4_qa1", 1.0, "PRED 'dependency edge between English words → aligned in Hindi' matches gold."),
    ("doc5_qa0", 0.0, "PRED 'natural dialogs' WRONG."),
    ("doc5_qa1", 0.0, "PRED refuses ('do not contain')."),
    ("doc6_qa0", 0.0, "PRED refuses ('do not specify')."),
    ("doc6_qa1", 0.5, "PRED 'SERA + content relevance between summaries' partial."),
    ("doc6_qa2", 0.5, "PRED 'very different correlations' partial."),
    ("doc6_qa3", 0.75, "PRED 'higher tiers indicate higher importance' matches gold."),
    ("doc6_qa4", 0.75, "PRED 'ROUGE reliable refutes' matches gold."),
    ("doc7_qa0", 0.0, "PRED refuses ('do not mention')."),
    ("doc7_qa1", 0.0, "PRED '19.9M params 53.9% accuracy' WRONG; gold says 'None'."),
    ("doc8_qa0", 0.75, "PRED 'interaction + discussion points adopted + recall + drop' includes gold aspects."),
    ("doc8_qa1", 1.0, "PRED 'IQ2 transcripts 108 debates' matches Intelligence Squared."),
    ("doc9_qa0", 0.75, "PRED 'accuracy of clustering' matches gold Accuracy."),
    ("doc10_qa0", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc10_qa1", 0.5, "PRED 'English Honeypot dataset, high quality' partial — 1 of 2 datasets."),
    ("doc10_qa2", 0.75, "PRED 'topic-based features using LDA, local+global' matches gold."),
    ("doc11_qa0", 1.0, "PRED 'Yes paper clearly establishes informal language/spelling errors' matches."),
    ("doc11_qa1", 1.0, "PRED 'addressed earlier, established task' matches."),
    ("doc11_qa2", 1.0, "PRED 'simple word-level encoder for tweets + whitespace + lookup table' matches gold."),
    ("doc11_qa3", 0.0, "PRED 'predicting user-annotated hashtags' but gold says 'None'."),
    ("doc11_qa4", 1.0, "PRED 'simple word-level encoder for tweets + 20K' matches gold."),
    ("doc12_qa0", 0.5, "PRED 'discusses methods but not specific evaluation methods' partial unanswerable lean."),
    ("doc12_qa1", 1.0, "PRED 'over 30,000 images' matches gold 30,000."),
    ("doc12_qa2", 0.0, "PRED refuses ('does not specify methods')."),
    ("doc12_qa3", 0.25, "PRED 'linguistic bias + unwarranted inferences' vague; gold=Ethnic bias."),
    ("doc13_qa0", 0.0, "PRED refuses ('does not specify')."),
    ("doc13_qa1", 1.0, "PRED 'SemEval 2010 task 8' exact match."),
    ("doc13_qa2", 0.0, "PRED refuses ('does not provide specific details')."),
    ("doc13_qa3", 0.0, "PRED 'bi-directional RNN' but gold='uni-directional'."),
    ("doc13_qa4", 0.5, "PRED 'three disjoint regions, middle context combination' partial."),
    ("doc14_qa0", 0.5, "PRED 'initial WMT16 models before synthetic data' partial."),
    ("doc15_qa0", 1.0, "PRED 'UD1.2 corpora covering 16 languages: full list' matches gold."),
    ("doc15_qa1", 1.0, "PRED 'full 16-language list' matches gold."),
    ("doc16_qa0", 0.0, "PRED 'No' but gold='Yes only Chinese-English' — Y/N flip."),
    ("doc17_qa0", 0.0, "PRED 'do not indicate experiments'."),
    ("doc17_qa1", 0.0, "PRED 'Yes they collected' but gold='No' — Y/N flip."),
    ("doc17_qa2", 0.75, "PRED 'Winograd schemas + sentence pairs' = WSC collection."),
    ("doc17_qa3", 0.0, "PRED 'distinctions between pronouns' vague; gold=English."),
    ("doc18_qa0", 0.0, "PRED 'deep contextual + bidirectional LSTM' missing SOTA tool names."),
    ("doc18_qa1", 0.75, "PRED 'GENIA Corpus and CoNLL2003' includes gold GENIA."),
    ("doc19_qa0", 0.0, "PRED refuses ('do not specify architecture')."),
    ("doc19_qa1", 0.25, "PRED 'image+question with agree/disagree' partial."),
    ("doc20_qa0", 0.0, "PRED 'climate change/abortion/world news/AI' WRONG topics."),
    ("doc20_qa1", 0.25, "PRED 'analyzes content with linguistic/behavioral predictors' vague."),
    ("doc21_qa0", 0.5, "PRED 'co-occurrence frequencies + contextual + semantic similarity' partial."),
    ("doc21_qa1", 1.0, "PRED 'do not specify' matches unanswerable."),
    ("doc21_qa2", 1.0, "PRED 'CBOW approach and Skip-gram approach' includes Skip-gram."),
    ("doc22_qa0", 0.5, "PRED 'improves CBT accuracy by larger margin + exceeds human baseline' partial; missing 'averaging'."),
    ("doc22_qa1", 0.5, "PRED 'much larger margin' direction-only correct."),
    ("doc22_qa2", 0.75, "PRED 'human study, majority answerable, baselines underestimated' matches."),
    ("doc23_qa0", 1.0, "PRED 'three levels: raw text + text cleaning + removal of keyphrase sparse' includes raw text."),
    ("doc23_qa1", 0.0, "PRED 'TopicRank and TF-IDF' WRONG; gold=Stanford CoreNLP/OCR."),
    ("doc23_qa2", 0.0, "PRED refuses ('not specified'); gold=244."),
    ("doc24_qa0", 0.25, "PRED lists features (unigrams/emoticons/etc) but gold=BIBREF9."),
    ("doc24_qa1", 0.75, "PRED 'Semeval 2014 + 25K sarcastic + 75K non-sarcastic' includes Semeval 2014."),
    ("doc24_qa2", 1.0, "PRED 'inherent semantics from sarcastic corpus using baseline CNN' matches gold."),
    ("doc25_qa0", 1.0, "PRED 'do not mention seasonality' matches unanswerable."),
    ("doc25_qa1", 1.0, "PRED '300' exact match."),
    ("doc25_qa2", 0.0, "PRED '3,216 tweets' WRONG; gold=250K."),
    ("doc26_qa0", 0.75, "PRED includes German-English in language pairs list."),
    ("doc26_qa1", 1.0, "PRED 'IMDb movie review dataset' exact."),
    ("doc26_qa2", 0.25, "PRED 'minimalist recurrent pooling' vs gold dynamic-average."),
    ("doc27_qa0", 0.5, "PRED 'competitive results for some emotion labels' partial."),
    ("doc27_qa1", 0.75, "PRED 'Affective Text + Fairy Tales + ISEAR' includes gold Affective Text."),
    ("doc27_qa2", 0.0, "PRED 'subsets chosen by performance' vague."),
    ("doc28_qa0", 0.0, "PRED 'single-topic unbalanced' vague; gold=anti-nuclear-power."),
    ("doc28_qa1", 0.0, "PRED refuses ('not specify layers')."),
    ("doc28_qa2", 0.75, "PRED 'abortion + gay rights + Obama + marijuana' includes abortion."),
    ("doc28_qa3", 0.0, "PRED refuses ('not specified'); gold=32,595."),
    ("doc28_qa4", 0.0, "PRED 'Yes' but gold='No'."),
    ("doc28_qa5", 0.5, "PRED 'Majority and SVM models' partial; missing n-gram features."),
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
    print(f"qasper v4t-corpus-tuned online added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
