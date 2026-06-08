"""Phase 1.9 — QASPER v4t-canonical online (94 entries, ask-after-ingest baseline)."""
from __future__ import annotations
import json
from pathlib import Path

QID_PREFIX = "qasper__v4t-canonical__online__"
QID_SUFFIX = "__seed42"
RESULTS = Path("results/stage3/judge_queue/qasper__v4t-canonical__online__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("doc0_qa0", 0.25, "PRED 'prior knowledge as constraints + feedback on features' partial."),
    ("doc0_qa1", 0.0, "PRED refuses ('not explicitly listed')."),
    ("doc0_qa2", 0.0, "PRED refuses ('does not specify')."),
    ("doc0_qa3", 0.5, "PRED 'unbalanced corpora + data distribution' partial; matches gold's unbalanced focus."),
    ("doc1_qa0", 0.0, "PRED 'No in-domain + cross-domain' but gold='Yes' — Y/N flip."),
    ("doc1_qa1", 0.75, "PRED 'claims, premises, backing' matches 3 of 5 gold components."),
    ("doc1_qa2", 0.0, "PRED refuses ('does not specify')."),
    ("doc1_qa3", 0.25, "PRED 'user-generated content on Web' vague."),
    ("doc1_qa4", 0.0, "PRED 'rhetorical questions/figurative/fallacies' hallucinated when unanswerable."),
    ("doc1_qa5", 0.5, "PRED 'length of document + literary devices' partial."),
    ("doc2_qa0", 0.0, "PRED 'garg2012unsupervised' WRONG."),
    ("doc2_qa1", 0.5, "PRED 'Superlingual latent variables' near gold's CLV concept."),
    ("doc2_qa2", 0.0, "PRED refuses ('not specify')."),
    ("doc2_qa3", 0.75, "PRED 'No, little info from parallel data setup' matches gold='No'."),
    ("doc2_qa4", 1.0, "PRED 'Yes one per language pair' matches gold."),
    ("doc2_qa5", 0.5, "PRED 'superlingual latent variables + Bayesian framework' partial."),
    ("doc2_qa6", 1.0, "PRED 'do not explicitly state' matches unanswerable."),
    ("doc3_qa0", 0.0, "PRED 'Yes valid' when gold unanswerable."),
    ("doc4_qa0", 1.0, "PRED '1500 sentences' exact match."),
    ("doc4_qa1", 0.75, "PRED 'dual decomposition + alignment + prepositional phrase' matches."),
    ("doc5_qa0", 1.0, "PRED 'CrowdFlower' exact match (mentions platform name)."),
    ("doc5_qa1", 0.0, "PRED refuses ('not contain information')."),
    ("doc6_qa0", 1.0, "PRED 'Yes only English' exact match."),
    ("doc6_qa1", 0.5, "PRED 'SERA + content relevance between summaries' partial."),
    ("doc6_qa2", 0.5, "PRED 'Sera vs ROUGE-2/3 not strong, keyword high' partial nuance."),
    ("doc6_qa3", 0.0, "PRED refuses ('not explicitly mentioned')."),
    ("doc6_qa4", 0.75, "PRED 'ROUGE reliable refutes' matches gold."),
    ("doc7_qa0", 0.0, "PRED refuses ('do not mention')."),
    ("doc7_qa1", 0.0, "PRED refuses ('does not specify')."),
    ("doc8_qa0", 0.75, "PRED 'discussion points adopted + recall + drop' matches gold's 3 aspects."),
    ("doc8_qa1", 0.0, "PRED refuses ('do not provide information')."),
    ("doc9_qa0", 0.0, "PRED 'test set 600 docs in 3 classes' WRONG; gold=Accuracy."),
    ("doc10_qa0", 1.0, "PRED 'No' matches gold='No'."),
    ("doc10_qa1", 0.0, "PRED '2197 legitimate + 802 spammers' WRONG dataset; gold=Social Honeypot + Weibo."),
    ("doc10_qa2", 0.0, "PRED 'LOSS + GOSS + Adaboost + SVM' WRONG approach; gold=LDA features + binary classification."),
    ("doc11_qa0", 1.0, "PRED 'Yes paper clearly establishes informal language' matches."),
    ("doc11_qa1", 0.0, "PRED 'something new' WRONG; gold='established task'."),
    ("doc11_qa2", 0.25, "PRED 'traditional NLP word vectors vs tweet2vec' partial."),
    ("doc11_qa3", 0.0, "PRED 'predicting hashtags' but gold says 'None'."),
    ("doc11_qa4", 0.25, "PRED 'traditional word-based approach' vague."),
    ("doc12_qa0", 0.0, "PRED 'POS tagging + Louvain clustering + manual categorization' hallucinated when unanswerable."),
    ("doc12_qa1", 1.0, "PRED '30,000 images' exact match."),
    ("doc12_qa2", 0.25, "PRED 'POS + coreference + Louvain' partial."),
    ("doc12_qa3", 1.0, "PRED 'default whiteness assumption + ethnicity marking' matches gold's Ethnic bias."),
    ("doc13_qa0", 0.0, "PRED refuses ('not specify')."),
    ("doc13_qa1", 1.0, "PRED 'SemEval 2010 relation classification' matches gold."),
    ("doc13_qa2", 0.0, "PRED refuses ('does not provide specific details')."),
    ("doc13_qa3", 0.0, "PRED 'bi-directional' but gold='uni-directional'."),
    ("doc13_qa4", 0.25, "PRED 'extended middle context' vague."),
    ("doc14_qa0", 0.5, "PRED 'initial translation systems + BLEU 4.3-11 improvements' partial."),
    ("doc15_qa0", 0.5, "PRED 'datasets covering 16 languages' count correct, no names."),
    ("doc15_qa1", 0.5, "PRED 'Czech/French/Italian/English/Danish/Indonesian + others 16 total' partial."),
    ("doc16_qa0", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("doc17_qa0", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("doc17_qa1", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("doc17_qa2", 0.0, "PRED 'gendered pronouns + French' WRONG; gold=WSC."),
    ("doc17_qa3", 0.0, "PRED 'French' WRONG; gold=English."),
    ("doc18_qa0", 0.0, "PRED 'F1 84-94% LSTMs + trigram hashing' missing SOTA tool names."),
    ("doc18_qa1", 0.0, "PRED refuses ('do not provide information')."),
    ("doc19_qa0", 0.5, "PRED 'LSTM-CNN' partial; missing VGG16 + element-wise multiplication."),
    ("doc19_qa1", 0.25, "PRED 'predicting agree/disagree' partial."),
    ("doc20_qa0", 0.0, "PRED 'user behavior metrics' WRONG; gold=specific topics."),
    ("doc20_qa1", 0.5, "PRED 'linear regression model' close to gold's logistic regression."),
    ("doc21_qa0", 0.5, "PRED 'semantic similarity into co-occurrence vectors + term relationships' partial."),
    ("doc21_qa1", 1.0, "PRED 'do not specify number' matches unanswerable."),
    ("doc21_qa2", 0.0, "PRED 'word/phrase/sentence embeddings from literature' vague; gold=Skip-gram."),
    ("doc22_qa0", 0.0, "PRED 'iteratively adds best' WRONG; gold='simply averaging'."),
    ("doc22_qa1", 0.0, "PRED refuses ('do not contain information')."),
    ("doc22_qa2", 0.25, "PRED 'BookTest dataset for community' partial."),
    ("doc23_qa0", 0.0, "PRED 'level 2 and 3' WRONG; gold=raw text."),
    ("doc23_qa1", 0.0, "PRED 'TopicRank/WINGNUS/KP-miner' WRONG."),
    ("doc23_qa2", 0.0, "PRED refuses ('not specified'); gold=244."),
    ("doc24_qa0", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold BIBREF9."),
    ("doc24_qa1", 0.0, "PRED 'Dataset 1/2/3' WRONG; gold=Semeval 2014."),
    ("doc24_qa2", 0.5, "PRED 'sentiment/emotion/personality features in baseline CNN' partial."),
    ("doc25_qa0", 1.0, "PRED 'do not mention seasonality' matches unanswerable."),
    ("doc25_qa1", 0.0, "PRED refuses ('not specify dimension')."),
    ("doc25_qa2", 0.25, "PRED 'aggregate sentiment + Microsoft stock' missing 250K count."),
    ("doc26_qa0", 1.0, "PRED 'German-English' exact match."),
    ("doc26_qa1", 1.0, "PRED 'IMDb movie review dataset' exact."),
    ("doc26_qa2", 0.25, "PRED 'recurrent pooling' missing dynamic-average specifics."),
    ("doc27_qa0", 0.5, "PRED 'competitive results without handcrafted lexicon' partial."),
    ("doc27_qa1", 0.0, "PRED refuses ('described in Section SECREF3')."),
    ("doc27_qa2", 0.0, "PRED refuses ('not detailed')."),
    ("doc28_qa0", 0.0, "PRED 'stance classification' WRONG; gold=anti-nuclear-power."),
    ("doc28_qa1", 0.0, "PRED refuses ('not specify layers')."),
    ("doc28_qa2", 0.0, "PRED refuses ('not specify topics')."),
    ("doc28_qa3", 0.0, "PRED refuses ('not mentioned'); gold=32,595."),
    ("doc28_qa4", 0.0, "PRED 'Yes FBFans + CreateDebate' but gold='No' — Y/N flip."),
    ("doc28_qa5", 0.0, "PRED 'ILP + CRFs' WRONG; gold=SVM with n-grams."),
    ("doc29_qa0", 0.0, "PRED 'No definitive' when gold unanswerable."),
    ("doc29_qa1", 0.0, "PRED 'No' but gold='Yes'."),
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
    print(f"qasper v4t-canonical online added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
