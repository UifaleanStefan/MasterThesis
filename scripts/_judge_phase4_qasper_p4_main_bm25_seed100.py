"""Phase 4 cross-vendor finishing — QASPER p4_main bm25 seed=100 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__bm25__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'active learning reduce costs' hallucinated."),
    ("q001", 0.0, "PRED title fragment about Persian corpus."),
    ("q002", 0.0, "PRED 'Table TABREF19 + TABREF26' fragment."),
    ("q003", 0.0, "PRED 'NUS trained on dialogues SDS restaurant' WRONG context."),
    ("q004", 0.0, "PRED title fragment."),
    ("q005", 0.0, "PRED 'High-Level Statistics @!START@' garbled."),
    ("q006", 0.0, "PRED 'devices monitoring environment' WRONG."),
    ("q007", 0.0, "PRED 'BIBREF2 decision tree' WRONG; gold=Mechanical Turk."),
    ("q008", 0.0, "PRED 'Twitter microblogging' hallucinated."),
    ("q009", 0.0, "PRED 'cosine similarity' fragment."),
    ("q010", 0.0, "PRED 'Quora community' WRONG."),
    ("q011", 0.0, "PRED title fragment about Mapudungun."),
    ("q012", 0.0, "PRED 'offensive target identification' fragment."),
    ("q013", 0.0, "PRED 'state-of-the-art lack contextual' WRONG."),
    ("q014", 0.0, "PRED 'feature engineering TF-IDF' WRONG."),
    ("q015", 0.25, "PRED 'low-rank matrix factorization PMI variants' partial PMI reference."),
    ("q016", 0.25, "PRED 'recurrent generative + diversity modules' partial."),
    ("q017", 0.0, "PRED 'modify dataset BIBREF56' fragment."),
    ("q018", 0.0, "PRED 'EHR patient notes' WRONG context."),
    ("q019", 0.25, "PRED 'attention transformations three language pairs' partial."),
    ("q020", 0.25, "PRED 'ALCrowd compare' partial."),
    ("q021", 0.25, "PRED 'Dutch + Turkish + Spanish + Russian percentages' mentions Russian."),
    ("q022", 0.0, "PRED 'SBERT not for transfer learning' WRONG."),
    ("q023", 0.25, "PRED 'ELMo contextual learned unsupervised' partial."),
    ("q024", 0.0, "PRED 'Table TABREF15 VLSP 2019' hallucinated; gold unanswerable."),
    ("q025", 0.25, "PRED 'feature ablation test causality' partial."),
    ("q026", 0.0, "PRED 'unsupervised generative models' WRONG."),
    ("q027", 0.5, "PRED 'Rule-based Approaches NLU NLG' partial implies rule-based."),
    ("q028", 0.25, "PRED 'experiments comparing popular NER algorithms' partial."),
    ("q029", 0.0, "PRED title fragment."),
    ("q030", 0.25, "PRED 'results detail characteristics tweets' partial."),
    ("q031", 0.0, "PRED 'unlabeled data + keywords' WRONG."),
    ("q032", 0.25, "PRED title 'Zero-shot Reading Comprehension' partial topic match."),
    ("q033", 0.0, "PRED 'Car-Speak Replacing Humans' title."),
    ("q034", 0.0, "PRED 'back-translation augmentation noising' WRONG."),
    ("q035", 0.0, "PRED 'T5 WinoGrande' WRONG."),
    ("q036", 0.25, "PRED 'attention mechanism Chinese word segmentation' partial."),
    ("q037", 0.0, "PRED 'semantic similarity helps interpret' fragment."),
    ("q038", 0.0, "PRED 'three-step TweetQA' WRONG."),
    ("q039", 0.0, "PRED 'Bahdanau attention' hallucinated."),
    ("q040", 0.0, "PRED 'STAGG multiple relation detectors' WRONG."),
    ("q041", 0.25, "PRED 'three different scenarios ten-fold cross validation' partial."),
    ("q042", 0.0, "PRED title fragment."),
    ("q043", 0.0, "PRED 'Q1: three factors clause alignment' WRONG."),
    ("q044", 0.0, "PRED 'Table TABREF17 classification' fragment."),
    ("q045", 0.25, "PRED 'Table TABREF8 baseline HUMAN' partial."),
    ("q046", 0.25, "PRED 'data weighting improves' partial."),
    ("q047", 0.0, "PRED 'NeuronBlocks CoNLL-2003' hallucinated."),
    ("q048", 0.0, "PRED 'CoNLL-2010' WRONG; gold=MaxEnt + SVMs."),
    ("q049", 0.5, "PRED 'two typical NLP tasks six datasets' partial."),
    ("q050", 0.0, "PRED title fragment about LSTM Speech."),
    ("q051", 0.0, "PRED 'dimensionality reduction RBM SMM' WRONG."),
    ("q052", 0.0, "PRED 'Rouge variants + Sera + Pyramid' WRONG."),
    ("q053", 0.0, "PRED 'LSTMN extension LSTM' WRONG."),
    ("q054", 0.0, "PRED 'High Resource experiments' WRONG."),
    ("q055", 0.0, "PRED 'INLINEFORM transforms' fragment."),
    ("q056", 0.25, "PRED title 'Extreme Language Model Compression' partial."),
    ("q057", 0.25, "PRED 'MCQA datasets crowd-sourcing hand engineering' partial."),
    ("q058", 0.0, "PRED 'GANs text generation' WRONG."),
    ("q059", 0.0, "PRED 'do not know INLINEFORM in advance' fragment."),
    ("q060", 0.0, "PRED 'Topic models LDA unsupervised' WRONG."),
    ("q061", 0.0, "PRED 'statistical N-gram Transformer-XL' hallucinated."),
    ("q062", 0.0, "PRED 'Dependency parsing Freeling' fragment."),
    ("q063", 0.0, "PRED 'Comparison with State of Art' section header."),
    ("q064", 0.0, "PRED title fragment."),
    ("q065", 0.0, "PRED 'raw waveform + grapheme' WRONG architecture."),
    ("q066", 0.5, "PRED 'same word non-identical vector reps in contexts' partial."),
    ("q067", 0.75, "PRED 'country-independent two news domains' matches gold concept."),
    ("q068", 0.5, "PRED 'GreenBioBERT eight NER BIBREF2' partial."),
    ("q069", 0.0, "PRED 'Inception and Joint' WRONG."),
    ("q070", 0.5, "PRED 'word importance translation perf decrease' partial."),
    ("q071", 0.0, "PRED 'XNLI accuracy differences' WRONG."),
    ("q072", 0.0, "PRED 'keywords informative tell what happened' question fragment."),
    ("q073", 0.0, "PRED '3,384 tweets HashtagMaster' WRONG."),
    ("q074", 0.0, "PRED title fragment."),
    ("q075", 0.0, "PRED 'Simulating Symptom Monitoring' WRONG."),
    ("q076", 0.0, "PRED '226,711 news articles' WRONG."),
    ("q077", 0.0, "PRED 'subword info BIBREF14' fragment."),
    ("q078", 0.0, "PRED 'model copes with some errors' fragment."),
    ("q079", 0.0, "PRED title fragment UG18 SemEval."),
    ("q080", 0.0, "PRED 'word order divergence' fragment."),
    ("q081", 0.0, "PRED 'PRUs LSTMs' WRONG."),
    ("q082", 0.25, "PRED 'Table TABREF11 FLC dev scores polarity POS NER' partial."),
    ("q083", 0.0, "PRED 'lexical properties PoS tagger' fragment."),
    ("q084", 0.0, "PRED 'Stopwords removed' WRONG."),
    ("q085", 0.0, "PRED 'polarization legislatures votes' fragment."),
    ("q086", 0.0, "PRED 'ensemble method approaches compensate' WRONG."),
    ("q087", 0.0, "PRED 'KAR compare other MRC robust' fragment."),
    ("q088", 0.0, "PRED 'sefe ArXiv Senate grocery' hallucinated."),
    ("q089", 0.5, "PRED 'dev data BIBREF19 train+dev merged' partial WASSA."),
    ("q090", 0.0, "PRED 'Prior Recipe Attention user' WRONG."),
    ("q091", 0.5, "PRED 'extract structured relations + jointly model' partial."),
    ("q092", 0.0, "PRED title 'Neural Topic-Attention'."),
    ("q093", 0.0, "PRED 'second experiment 850 health' fragment."),
    ("q094", 0.0, "PRED 'binary classification dataset combinations' fragment."),
    ("q095", 0.0, "PRED title 'Reducing Gender Bias'."),
    ("q096", 0.0, "PRED title fragment."),
    ("q097", 0.0, "PRED 'ISIS recruiters' fragment."),
    ("q098", 0.25, "PRED 'Edinburgh NMT WMT 16' title mentions attention concept indirectly."),
    ("q099", 0.0, "PRED 'gloss knowledge supervised neural WSD' WRONG."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_main__qasper__bm25__seed100__"
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
                skipped += 1
                continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"qasper p4_main bm25 seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
