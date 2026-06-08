"""Phase 1.9 — QASPER bm25-corpus batch (94 entries, sparse-retrieval baseline)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__bm25-corpus__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__bm25-corpus__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.25, "PRED 'strong indicators... sports/sentiment polarity' vague; gold=labeled features."),
    ("doc0_qa1", 1.0, "PRED exactly matches gold's neutral-features regularization + 2 more specific terms."),
    ("doc0_qa2", 0.5, "PRED 'text classification + IE + QA' partial — text classification matches."),
    ("doc0_qa3", 0.5, "PRED 'handle bias in prior knowledge + variations' partial."),
    ("doc1_qa0", 0.0, "PRED 'No not only English' but gold='Yes' — Y/N flip."),
    ("doc1_qa1", 0.5, "PRED 'claims or premises' partial; gold lists 5 components."),
    ("doc1_qa2", 0.0, "PRED refuses ('do not specify')."),
    ("doc1_qa3", 1.0, "PRED 'comments to articles + blog posts + newswire' matches gold's newswire/blog."),
    ("doc1_qa4", 0.0, "PRED hallucinated 'less formal/ambiguous' when gold unanswerable."),
    ("doc1_qa5", 0.5, "PRED 'variety of formats/styles in user-gen Web discourse' partial linguistic variability."),
    ("doc2_qa0", 0.0, "PRED 'garg2012unsupervised' WRONG; gold=lang2011unsupervised."),
    ("doc2_qa1", 0.75, "PRED 'Crosslingual latent variables (CLVs)' matches gold's CLV concept."),
    ("doc2_qa2", 0.0, "PRED refuses ('not mentioned'); gold=CoNLL 2009 EN/DE."),
    ("doc2_qa3", 0.5, "PRED 'small improvements + biggest impact from monolingual' implies No (matching gold)."),
    ("doc2_qa4", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc2_qa5", 0.5, "PRED 'Bayesian model for each language' partial; missing garg2012unsupervised citation."),
    ("doc2_qa6", 1.0, "PRED 'do not provide information' matches unanswerable."),
    ("doc3_qa0", 0.0, "PRED 'Yes valid' when gold unanswerable."),
    ("doc4_qa0", 0.0, "PRED '100 parallel sentences' WRONG; gold=1500."),
    ("doc4_qa1", 0.75, "PRED 'aligning words + corresponding dependency edges + dual decomposition' matches gold."),
    ("doc5_qa0", 0.0, "PRED 'natural dialogs' WRONG; gold=CrowdFlower."),
    ("doc5_qa1", 0.0, "PRED refuses ('not specified')."),
    ("doc6_qa0", 0.0, "PRED 'do not specify' REFUSAL; gold=Yes."),
    ("doc6_qa1", 1.0, "PRED 'overlap between summaries using IR, summaries as search queries' EXACT match."),
    ("doc6_qa2", 0.75, "PRED 'different correlations, not all ROUGE equally effective' partial match."),
    ("doc6_qa3", 0.5, "PRED 'tiered by importance' touches on higher-tiers concept."),
    ("doc6_qa4", 0.75, "PRED 'ROUGE reliable metric refutes' matches gold."),
    ("doc7_qa0", 1.0, "PRED 'No do not mention other datasets' matches gold='No'."),
    ("doc7_qa1", 0.0, "PRED 'word-level vs character-level' vague; gold says 'None' for baselines."),
    ("doc8_qa0", 0.5, "PRED 'promoting points + attacking opponents' partial."),
    ("doc8_qa1", 1.0, "PRED 'IQ2 transcripts 108 debates' matches Intelligence Squared."),
    ("doc9_qa0", 0.75, "PRED 'percentage of incorrect assignments' matches Accuracy concept."),
    ("doc10_qa0", 0.0, "PRED 'Yes unsupervised' but gold='No' — Y/N flip."),
    ("doc10_qa1", 0.75, "PRED 'Social Honeypot 19,276/22,223, high quality' matches first dataset + quality."),
    ("doc10_qa2", 0.75, "PRED 'LDA topic distribution patterns local+global' matches gold's feature-extraction approach."),
    ("doc11_qa0", 1.0, "PRED 'Yes paper establishes informal language/spelling errors' matches."),
    ("doc11_qa1", 1.0, "PRED 'addressed earlier indicating established task' matches gold."),
    ("doc11_qa2", 1.0, "PRED 'simple word-level encoder for tweets + whitespace tokens + lookup table' matches gold."),
    ("doc11_qa3", 0.0, "PRED 'predicting user-annotated hashtags' but gold says 'None' for other tasks."),
    ("doc11_qa4", 1.0, "PRED 'simple word-level encoder for tweets + whitespace' matches gold."),
    ("doc12_qa0", 0.0, "PRED 'discusses methods to analyze data... biases' when gold unanswerable."),
    ("doc12_qa1", 1.0, "PRED 'over 30,000 images' matches gold 30,000."),
    ("doc12_qa2", 0.5, "PRED 'methods to manually detect stereotypes' partial."),
    ("doc12_qa3", 0.25, "PRED 'linguistic bias + unwarranted inferences' vague; gold=Ethnic bias specifically."),
    ("doc13_qa0", 0.0, "PRED 'achieves SOTA but amount not provided' REFUSAL."),
    ("doc13_qa1", 1.0, "PRED 'SemEval 2010 relation classification' exact match."),
    ("doc13_qa2", 0.0, "PRED refuses ('does not provide specific details')."),
    ("doc13_qa3", 0.0, "PRED 'bi-directional RNN' but gold='uni-directional'."),
    ("doc13_qa4", 0.5, "PRED 'extended middle context, all sentence parts' partial."),
    ("doc14_qa0", 0.5, "PRED 'initial translation systems WMT 2016 Edinburgh' partial; missing attentional encoder-decoder."),
    ("doc15_qa0", 0.5, "PRED '16 languages' correct count but missing names; gold has full UD treebanks list."),
    ("doc15_qa1", 0.5, "PRED '16 different languages' correct count but missing names."),
    ("doc16_qa0", 0.0, "PRED 'No Chinese-English' but gold='Yes only Chinese-English' — Y/N flip."),
    ("doc17_qa0", 0.0, "PRED 'No do not conduct' but gold='Yes' — Y/N flip."),
    ("doc17_qa1", 0.0, "PRED refuses ('do not mention')."),
    ("doc17_qa2", 0.5, "PRED 'Winograd schemas + sentence pairs ambiguous pronouns' partial WSC."),
    ("doc17_qa3", 0.0, "PRED 'Spanish and Italian' WRONG; gold=English."),
    ("doc18_qa0", 0.0, "PRED characteristic description; missing SOTA tool names."),
    ("doc18_qa1", 0.0, "PRED refuses ('not mentioned')."),
    ("doc19_qa0", 0.0, "PRED 'random forest + deep learning' WRONG; gold=LSTM+VGG16."),
    ("doc19_qa1", 0.5, "PRED 'crowd agreement labeling agree/disagree' partial."),
    ("doc20_qa0", 0.0, "PRED 'diverse conversational topics about dogmatism' vague; gold=politics/business/science/AskReddit."),
    ("doc20_qa1", 0.25, "PRED 'predictive model with linguistic + behavioral features' vague."),
    ("doc21_qa0", 0.5, "PRED 'co-occurrence frequencies + contextual + semantic similarity' partial."),
    ("doc21_qa1", 1.0, "PRED 'do not specify how many humans' matches unanswerable."),
    ("doc21_qa2", 0.5, "PRED 'semantic similarity + word embedding methods' partial; missing Skip-gram."),
    ("doc22_qa0", 0.0, "PRED 'starts with best, gradually adds...' WRONG; gold='simply averaging predictions'."),
    ("doc22_qa1", 0.5, "PRED 'much larger margin' direction-only correct."),
    ("doc22_qa2", 0.75, "PRED 'human study, majority answerable, baselines underestimated' matches gold."),
    ("doc23_qa0", 1.0, "PRED 'three levels: raw text, text cleaning, removal of keyphrase sparse sections' includes raw text."),
    ("doc23_qa1", 0.25, "PRED 'five models in categories' vague; missing specific names."),
    ("doc23_qa2", 0.0, "PRED refuses ('not specified'); gold=244."),
    ("doc24_qa0", 0.25, "PRED 'pre-trained CNN for sentiment/emotion/personality' WRONG approach; gold=BIBREF9."),
    ("doc24_qa1", 0.0, "PRED '50K sarcastic + 50K non-sarcastic' WRONG; gold=Semeval 2014."),
    ("doc24_qa2", 1.0, "PRED '100 features from fully-connected layer in baseline CNN' matches gold."),
    ("doc25_qa0", 1.0, "PRED 'do not mention removal of seasonality' matches unanswerable."),
    ("doc25_qa1", 0.0, "PRED refuses ('not specify dimension')."),
    ("doc25_qa2", 0.25, "PRED 'tweets re Microsoft stock prices over year' missing 250K count."),
    ("doc26_qa0", 0.0, "PRED refuses ('do not specify language pairs')."),
    ("doc26_qa1", 0.0, "PRED refuses ('do not specify')."),
    ("doc26_qa2", 0.25, "PRED 'minimalist recurrent pooling' vs gold dynamic-average."),
    ("doc27_qa0", 0.5, "PRED 'competitive results without handcrafted lexicon' partial."),
    ("doc27_qa1", 0.75, "PRED 'Affective Text + Fairy Tales + ISEAR' includes gold."),
    ("doc27_qa2", 1.0, "PRED lists all FB pages exactly matching gold."),
    ("doc28_qa0", 0.0, "PRED 'stance classification' WRONG; gold=anti-nuclear-power."),
    ("doc28_qa1", 0.0, "PRED refuses ('not specify layers')."),
    ("doc28_qa2", 0.0, "PRED 'multiple topics' too vague; gold=abortion."),
    ("doc28_qa3", 0.0, "PRED 'single-topic unbalanced' missing 32,595."),
    ("doc28_qa4", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc28_qa5", 1.0, "PRED first baseline exactly 'SVM with unigram/bigram/trigram features' matches gold."),
    ("doc29_qa0", 1.0, "PRED 'does not mention automatic optimization' matches unanswerable."),
    ("doc29_qa1", 1.0, "PRED 'Yes attention-based NMT' matches gold='Yes'."),
    ("doc29_qa2", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc29_qa3", 0.0, "PRED refuses ('not specify')."),
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
    print(f"qasper bm25-corpus batch added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
