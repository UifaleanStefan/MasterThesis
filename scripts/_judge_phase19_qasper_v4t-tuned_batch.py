"""Phase 1.9 — QASPER v4t-tuned batch (94 entries, per-doc tuned middle config)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__v4t-tuned__batch__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__v4t-tuned__batch__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.0, "PRED 'news/biomedical' WRONG context; gold=labeled features."),
    ("doc0_qa1", 0.0, "PRED refuses ('not explicitly mentioned')."),
    ("doc0_qa2", 0.0, "PRED 'sentiment, language modeling, NMT' WRONG; gold=text classification themes."),
    ("doc0_qa3", 0.25, "PRED 'maintain performance despite variations' vague."),
    ("doc1_qa0", 1.0, "PRED 'Yes only English' exact match."),
    ("doc1_qa1", 0.0, "PRED 'post authorship/likes/topics/stance' WRONG context."),
    ("doc1_qa2", 0.0, "PRED 'HMMs/PSL/iterative classification' WRONG; gold=Structural SVM."),
    ("doc1_qa3", 0.0, "PRED 'Reddit dogmatism posts' WRONG context."),
    ("doc1_qa4", 0.0, "PRED hallucinated answer when gold unanswerable."),
    ("doc1_qa5", 0.25, "PRED 'balancing data/mechanisms' vague."),
    ("doc2_qa0", 0.0, "PRED refuses ('not provide information')."),
    ("doc2_qa1", 0.0, "PRED refuses ('not provide information')."),
    ("doc2_qa2", 0.0, "PRED 'TED + EPPS + NC + CommonCrawl' WRONG; gold=CoNLL 2009."),
    ("doc2_qa3", 0.0, "PRED 'Yes improves' but gold='No' — Y/N flip."),
    ("doc2_qa4", 1.0, "PRED 'Yes one latent per language pair' matches gold."),
    ("doc2_qa5", 0.25, "PRED 'parameters learned from data' generic; missing garg2012unsupervised."),
    ("doc2_qa6", 1.0, "PRED 'do not contain information' matches unanswerable."),
    ("doc3_qa0", 0.0, "PRED 'Yes valid' when gold unanswerable."),
    ("doc4_qa0", 0.0, "PRED '3 million sentences' WRONG; gold=1500."),
    ("doc4_qa1", 0.5, "PRED 'head-modifier through dependency parsing' partial."),
    ("doc5_qa0", 0.0, "PRED refuses ('do not provide specific information')."),
    ("doc5_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc6_qa0", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc6_qa1", 0.0, "PRED 'keyphrase density in title/abstract' WRONG context."),
    ("doc6_qa2", 0.0, "PRED refuses ('do not provide information')."),
    ("doc6_qa3", 0.0, "PRED refuses ('do not provide information')."),
    ("doc6_qa4", 0.75, "PRED 'ROUGE is reliable metric — refutes' matches gold."),
    ("doc7_qa0", 0.0, "PRED 'Yes BookTest improves CBT' but gold='No' — Y/N flip."),
    ("doc7_qa1", 0.0, "PRED refuses; gold says 'None' for baselines."),
    ("doc8_qa0", 0.0, "PRED 'dogmatism + frequency/breadth' WRONG context."),
    ("doc8_qa1", 0.0, "PRED 'CreateDebate' WRONG; gold=Intelligence Squared Debates."),
    ("doc9_qa0", 0.0, "PRED 'Damerau-Levenshtein matches' WRONG; gold=Accuracy."),
    ("doc10_qa0", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc10_qa1", 0.75, "PRED 'Social Honeypot, 19,276/22,223, high quality' detailed match for 1 of 2 datasets."),
    ("doc10_qa2", 0.75, "PRED 'LDA-based novel classification, topic patterns' matches gold."),
    ("doc11_qa0", 1.0, "PRED 'Yes informal language/spelling errors/etc' matches gold."),
    ("doc11_qa1", 1.0, "PRED 'addressed earlier, established task' matches gold."),
    ("doc11_qa2", 1.0, "PRED 'simple word-level encoder for tweets... splits whitespace + lookup table' matches gold."),
    ("doc11_qa3", 0.75, "PRED 'do not mention other tasks' matches gold='None'."),
    ("doc11_qa4", 1.0, "PRED 'simple word-level encoder for tweets... 20K common words' matches gold."),
    ("doc12_qa0", 1.0, "PRED 'does not specify evaluation methods' matches unanswerable."),
    ("doc12_qa1", 1.0, "PRED 'over 30,000 images' matches gold 30,000."),
    ("doc12_qa2", 0.0, "PRED refuses ('does not specify exact methods')."),
    ("doc12_qa3", 0.25, "PRED 'unwarranted inferences/stereotypes' touches on bias topic but doesn't say Ethnic."),
    ("doc13_qa0", 0.0, "PRED refuses ('does not specify how much')."),
    ("doc13_qa1", 1.0, "PRED 'SemEval 2010 task 8' exact match."),
    ("doc13_qa2", 0.5, "PRED 'combines CNN and RNN for relation classification' partial."),
    ("doc13_qa3", 0.0, "PRED 'bi-directional RNN' but gold='uni-directional'."),
    ("doc13_qa4", 0.5, "PRED 'extended middle context, combines sentence parts' partial."),
    ("doc14_qa0", 0.75, "PRED 'WMT 2016 attentional encoder-decoder' matches gold."),
    ("doc15_qa0", 1.0, "PRED 'datasets covering 16 languages: full list' matches gold."),
    ("doc15_qa1", 1.0, "PRED 'full 16-language list' matches gold."),
    ("doc16_qa0", 0.0, "PRED 'No multiple' but gold='Yes only Chinese-English' — Y/N flip (PRED's Chinese-English IS the gold pair but pred says No)."),
    ("doc17_qa0", 0.0, "PRED 'do not indicate experiments'."),
    ("doc17_qa1", 1.0, "PRED 'No referenced existing Winograd schemas' matches gold='No'."),
    ("doc17_qa2", 0.75, "PRED 'Winograd schemas' = WSC collection."),
    ("doc17_qa3", 0.0, "PRED 'multilingual NMT' WRONG (gold=English)."),
    ("doc18_qa0", 0.0, "PRED characteristic description; missing SOTA tool names."),
    ("doc18_qa1", 0.75, "PRED 'GENIA Corpus and CoNLL2003' includes gold GENIA."),
    ("doc19_qa0", 0.0, "PRED refuses ('does not specify architecture')."),
    ("doc19_qa1", 0.25, "PRED '(dis)agreement labels' partial."),
    ("doc20_qa0", 0.0, "PRED 'climate change/abortion/AI' WRONG topics; gold=politics/business/science/AskReddit."),
    ("doc20_qa1", 0.25, "PRED 'linguistic features + behavioral predictors' vague."),
    ("doc21_qa0", 0.5, "PRED 'co-occurrence frequencies + taxonomy' partial; misses 'frequencies of other words'."),
    ("doc21_qa1", 1.0, "PRED 'do not specify' matches unanswerable."),
    ("doc21_qa2", 0.5, "PRED 'second-order co-occurrence vectors vs word embeddings' partial."),
    ("doc22_qa0", 0.5, "PRED 'ensemble improves CBT accuracy' partial; missing 'averaging' mechanism."),
    ("doc22_qa1", 0.5, "PRED 'much larger margin' direction-only correct."),
    ("doc22_qa2", 0.5, "PRED 'human study + underestimated baselines' partial; missing 50/50 detail."),
    ("doc23_qa0", 0.25, "PRED 'four levels mentioned, no details' vague; gold=raw text."),
    ("doc23_qa1", 0.25, "PRED 'five models in categories' vague; missing specific model names."),
    ("doc23_qa2", 1.0, "PRED '244 scientific articles' exact match."),
    ("doc24_qa0", 0.25, "PRED lists features (unigrams/SVM/etc); gold=BIBREF9 — vague approximation."),
    ("doc24_qa1", 0.75, "PRED 'IMDb + Semeval 2014 Twitter' includes gold Semeval 2014."),
    ("doc24_qa2", 1.0, "PRED 'features extracted from sarcasm corpus using CNN' matches gold."),
    ("doc25_qa0", 1.0, "PRED 'do not mention seasonality' matches unanswerable."),
    ("doc25_qa1", 0.25, "PRED 'd_t = d_h' vague; gold=300."),
    ("doc25_qa2", 0.25, "PRED 'tweets from Twitter' missing 250K count."),
    ("doc26_qa0", 0.75, "PRED 'English<->Czech/German/Romanian/Russian' includes German-English."),
    ("doc26_qa1", 1.0, "PRED 'IMDb movie review dataset' exact match."),
    ("doc26_qa2", 0.25, "PRED 'minimalist recurrent pooling' vs gold 'dynamic average pooling'."),
    ("doc27_qa0", 0.5, "PRED 'competitive results without handcrafted resource' partial."),
    ("doc27_qa1", 0.75, "PRED 'Affective Text + Fairy Tales + ISEAR' includes gold."),
    ("doc27_qa2", 0.0, "PRED 'subsets based on performance' vague; gold lists specific FB pages."),
    ("doc28_qa0", 0.0, "PRED 'stance classification' WRONG; gold=anti-nuclear-power."),
    ("doc28_qa1", 0.0, "PRED refuses ('does not specify layers')."),
    ("doc28_qa2", 0.75, "PRED 'abortion + gay rights + Obama + marijuana' includes abortion."),
    ("doc28_qa3", 0.0, "PRED 'privately-owned single-topic unbalanced' missing 32,595 count."),
    ("doc28_qa4", 0.0, "PRED 'Yes FBFans + CreateDebate' but gold='No' — Y/N flip."),
    ("doc28_qa5", 0.25, "PRED 'Majority/SVM/CNN/RCNN' partial; missing n-gram features."),
    ("doc29_qa0", 0.0, "PRED 'No' definitive when gold unanswerable."),
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
    print(f"qasper v4t-tuned batch added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
