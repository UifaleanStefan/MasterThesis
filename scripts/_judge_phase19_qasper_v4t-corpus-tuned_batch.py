"""Phase 1.9 — QASPER v4t-corpus-tuned batch (94 entries, end-of-corpus QA on 30 papers)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__v4t-corpus-tuned__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__v4t-corpus-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

# Hand-judged 1-by-1 per evaluation/claude_judge_protocol.md 5-point rubric
JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.0, "PRED refuses ('does not specify'); gold=labeled features."),
    ("doc0_qa1", 0.0, "PRED refuses ('not detailed'); gold=regularization term details."),
    ("doc0_qa2", 0.5, "PRED mentions text classification but missing themes (sentiment/web-page/science/medical)."),
    ("doc0_qa3", 0.5, "PRED captures related idea (variations/noise) but missing prior-knowledge unbalanced specifics."),
    ("doc1_qa0", 1.0, "PRED 'Yes only English' exact match to gold."),
    ("doc1_qa1", 0.0, "PRED vague ('argument components in user-generated Web discourse'); gold=claim/premise/backing/rebuttal/refutation."),
    ("doc1_qa2", 0.0, "PRED refuses ('do not specify'); gold=Structural SVM."),
    ("doc1_qa3", 0.5, "PRED 'user-generated Web discourse' matches gold's 'user comments to newswire/blog posts' partially."),
    ("doc1_qa4", 0.0, "PRED hallucinated answer when gold marked unanswerable."),
    ("doc1_qa5", 0.5, "PRED 'variety of communication styles' captures linguistic variability concept partially."),
    ("doc2_qa0", 0.0, "PRED says garg2012unsupervised; gold says lang2011unsupervised (different baseline)."),
    ("doc2_qa1", 0.5, "PRED mentions CLVs but not the parent-of-role-variables relationship."),
    ("doc2_qa2", 0.0, "PRED refuses ('not mentioned'); gold=EN/DE CoNLL 2009."),
    ("doc2_qa3", 0.5, "PRED implies No (small improvements) but doesn't state clearly."),
    ("doc2_qa4", 0.0, "PRED 'No' but gold says 'Yes' — Y/N flip."),
    ("doc2_qa5", 0.5, "PRED mentions Bayesian model but missing garg2012unsupervised citation."),
    ("doc2_qa6", 1.0, "PRED 'do not explicitly state' matches gold unanswerable."),
    ("doc3_qa0", 0.0, "PRED 'Yes' when gold marked unanswerable — hallucinated affirmation."),
    ("doc4_qa0", 0.0, "PRED refuses ('not specified'); gold=1500 sentences."),
    ("doc4_qa1", 0.75, "PRED captures cross-lingual dependency edge correspondence well."),
    ("doc5_qa0", 0.0, "PRED talks about source but doesn't name CrowdFlower."),
    ("doc5_qa1", 0.0, "PRED refuses ('not specified'); gold=4.49 turns."),
    ("doc6_qa0", 0.0, "PRED refuses ('do not specify'); gold=Yes."),
    ("doc6_qa1", 0.5, "PRED mentions content relevance but missing IR-based mechanism."),
    ("doc6_qa2", 0.5, "PRED 'very different correlations' partially matches gold's 'don't have high correlations'."),
    ("doc6_qa3", 0.75, "PRED mentions pyramid structure + higher tiers concept."),
    ("doc6_qa4", 0.75, "PRED captures ROUGE reliability refutation correctly."),
    ("doc7_qa0", 1.0, "PRED 'No, only SimpleQuestions' matches gold exactly."),
    ("doc7_qa1", 0.0, "PRED lists baselines; gold says 'None' (no baselines used)."),
    ("doc8_qa0", 0.5, "PRED 'promoting points / attacking opponents' partial — missing time + adopted points."),
    ("doc8_qa1", 1.0, "PRED 'Intelligence Squared Debates (IQ2)' exact match."),
    ("doc9_qa0", 1.0, "PRED 'clustering accuracy' matches gold Accuracy."),
    ("doc10_qa0", 0.0, "PRED 'Yes unsupervised' but gold says 'No' — Y/N flip."),
    ("doc10_qa1", 0.5, "PRED mentions Honeypot but only one dataset, missing Weibo."),
    ("doc10_qa2", 0.75, "PRED 'LDA features for binary classification' matches gold well."),
    ("doc11_qa0", 1.0, "PRED 'Yes informal language/spelling errors' matches gold."),
    ("doc11_qa1", 0.0, "PRED 'something new' but gold says 'established task' — wrong."),
    ("doc11_qa2", 0.5, "PRED 'traditional NLP approaches with word-level reps' partial."),
    ("doc11_qa3", 0.0, "PRED lists task; gold says 'None' (no other tasks)."),
    ("doc11_qa4", 1.0, "PRED 'simple word-level encoder' exact match."),
    ("doc12_qa0", 1.0, "PRED 'does not specify' matches gold unanswerable."),
    ("doc12_qa1", 1.0, "PRED '30,000 images' matches gold 30,000."),
    ("doc12_qa2", 0.0, "PRED refuses ('not detailed'); gold=spot patterns by looking."),
    ("doc12_qa3", 0.0, "PRED lists linguistic+unwarranted biases; gold=Ethnic bias specifically."),
    ("doc13_qa0", 0.0, "PRED refuses ('not specified'); gold=0.8% F1."),
    ("doc13_qa1", 1.0, "PRED 'SemEval 2010 relation classification' matches gold."),
    ("doc13_qa2", 0.5, "PRED mentions CNN+RNN combination but missing voting mechanism."),
    ("doc13_qa3", 0.0, "PRED 'bi-directional RNN' but gold='uni-directional RNN' — wrong direction."),
    ("doc13_qa4", 0.5, "PRED captures three contexts split, missing convolution+max-pooling specifics."),
    ("doc14_qa0", 0.5, "PRED 'neural translation models' partially matches gold attentional encoder-decoder."),
    ("doc15_qa0", 1.0, "PRED full 16-language list exact match to gold."),
    ("doc15_qa1", 1.0, "PRED full 16-language list exact match to gold."),
    ("doc16_qa0", 1.0, "PRED 'Yes Chinese-to-English only' matches gold Yes."),
    ("doc17_qa0", 0.0, "PRED refuses ('do not indicate'); gold=Yes."),
    ("doc17_qa1", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc17_qa2", 0.75, "PRED 'Winograd schemas' ≈ gold's WSC collection."),
    ("doc17_qa3", 0.0, "PRED refuses ('does not specify language'); gold=English."),
    ("doc18_qa0", 0.0, "PRED gives characteristic description but doesn't name SOTA tools (Babelfy, DBpedia, etc)."),
    ("doc18_qa1", 0.75, "PRED 'CoNLL2003 and GENIA Corpus' includes gold GENIA."),
    ("doc19_qa0", 0.0, "PRED 'random forest' but gold=LSTM+VGG16 — completely wrong architecture."),
    ("doc19_qa1", 0.0, "PRED 'image+question with agree/disagree labels' wrong; gold is about predicting redundant answers."),
    ("doc20_qa0", 0.0, "PRED lists made-up topics (climate change, AI, etc.); gold=politics/business/science/AskReddit."),
    ("doc20_qa1", 0.25, "PRED vague 'predictive model with features'; gold=logistic regression."),
    ("doc21_qa0", 0.5, "PRED captures co-occurrence frequency idea but adds extra semantic similarity context."),
    ("doc21_qa1", 1.0, "PRED 'not specify' matches gold unanswerable."),
    ("doc21_qa2", 0.5, "PRED mentions word2vec/GoogleNews (includes skip-gram family) but doesn't name Skip-gram."),
    ("doc22_qa0", 0.0, "PRED refuses ('not provide specific details'); gold=averaging predictions."),
    ("doc22_qa1", 0.5, "PRED 'much larger margin' directionally correct; gold has broken INLINEFORM citation."),
    ("doc22_qa2", 0.5, "PRED captures human study + answerability; missing specific 50/50 named entity/common noun split."),
    ("doc23_qa0", 1.0, "PRED lists three levels including 'raw text' which matches gold."),
    ("doc23_qa1", 0.0, "PRED 'TopicRank and TF-IDF' but gold=Stanford CoreNLP/OCR/ParsCIT/etc."),
    ("doc23_qa2", 0.0, "PRED refuses ('not specified'); gold=244."),
    ("doc24_qa0", 0.25, "PRED lists feature categories (unigrams/emoticons/etc) but gold=BIBREF9 citation."),
    ("doc24_qa1", 0.75, "PRED includes gold Semeval 2014 + extra dataset; mostly right."),
    ("doc24_qa2", 1.0, "PRED 'baseline features extracted from CNN' matches gold exactly."),
    ("doc25_qa0", 1.0, "PRED 'do not mention' matches gold unanswerable."),
    ("doc25_qa1", 1.0, "PRED '300' exact match."),
    ("doc25_qa2", 0.25, "PRED 'tweets extracted from Twitter' missing 250,000 count."),
    ("doc26_qa0", 0.75, "PRED includes gold German-English in its list of language pairs."),
    ("doc26_qa1", 1.0, "PRED 'IMDb movie review dataset' exact match."),
    ("doc26_qa2", 0.25, "PRED 'minimalist recurrent pooling'; gold='dynamic average pooling' — different pooling."),
    ("doc27_qa0", 0.5, "PRED 'competitive results' partial; missing specific F1 0.409/0.459/0.411."),
    ("doc27_qa1", 0.75, "PRED 'Affective Text + Fairy Tales + ISEAR' includes gold Affective Text."),
    ("doc27_qa2", 0.0, "PRED 'subsets chosen by performance' vague; gold lists specific FB pages."),
    ("doc28_qa0", 0.0, "PRED 'single topic' vague; gold=anti-nuclear-power."),
    ("doc28_qa1", 0.0, "PRED refuses ('not specify'); gold=eight layers."),
    ("doc28_qa2", 0.75, "PRED 'abortion, gay rights, Obama, marijuana' includes gold abortion."),
    ("doc28_qa3", 0.0, "PRED refuses ('not specified'); gold=32,595 posts."),
    ("doc28_qa4", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc28_qa5", 0.5, "PRED 'Majority and SVM models' partial; gold='SVM with unigram/bigram/trigram features'."),
    ("doc29_qa0", 1.0, "PRED 'does not explicitly mention' matches gold unanswerable."),
    ("doc29_qa1", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc29_qa2", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc29_qa3", 0.0, "PRED refuses ('not mentioned'); gold=English."),
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
    print(f"qasper v4t-corpus-tuned batch added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
