"""Phase 4 cross-vendor finishing — QASPER p4_main flat-50 seed=100 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__flat-50__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q001", 0.75, "PRED 'transcribed at phone level, not mentioned' implicit match to gold unanswerable."),
    ("q002", 0.0, "PRED 'several strong baselines' vague; gold=BIBREF19 specifically."),
    ("q003", 0.25, "PRED 'NUS 93.4% vs ABUS 90.0%' different from gold=2.6pp average success rate."),
    ("q004", 0.25, "PRED 'applied LDA on ISWC/WWW 2013-2017' partial; gold is broader taxonomy description."),
    ("q005", 0.0, "PRED refuses; gold=four MT tasks."),
    ("q006", 1.0, "PRED 'No' matches gold='No'."),
    ("q007", 0.5, "PRED 'non-expert individuals who labeled sentiments' partial; gold='people in US using AMT'."),
    ("q008", 0.0, "PRED 'distant-supervised tweets with emoticons' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q010", 0.75, "PRED 'passages do not indicate' implicit match to gold='No'."),
    ("q011", 0.0, "PRED refuses; gold=Transformer architecture."),
    ("q012", 0.5, "PRED 'SVMs and neural networks' includes gold linear SVM."),
    ("q013", 0.75, "PRED 'BIBREF9 and BIBREF8' includes gold BIBREF9."),
    ("q014", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q015", 0.75, "PRED 'CPMI_-2 and NNEGPMI' matches gold=clipped PMI; NNEGPMI."),
    ("q016", 0.0, "PRED refuses; gold=average unique predictions."),
    ("q017", 0.0, "PRED 'grammaticality, perplexity' WRONG; gold=INLINEFORM0 scores."),
    ("q018", 0.75, "PRED 'does not specify' implicit match to gold='No' (not commercially available)."),
    ("q019", 1.0, "PRED 'German-English' matches gold=De-En."),
    ("q020", 0.0, "PRED refuses; gold=specific F1 scores 85.99/75.15/71.53."),
    ("q021", 0.0, "PRED 'Turkish 30.53%' WRONG; gold=Russian."),
    ("q022", 0.0, "PRED refuses; gold=MR."),
    ("q023", 1.0, "PRED 'Yes, lemmatization can hurt' matches gold='Yes'."),
    ("q024", 0.0, "PRED hallucinated WER; gold unanswerable."),
    ("q025", 0.0, "PRED 'linear model, SVM, BiLSTMs' WRONG; gold=PDTB taggers."),
    ("q026", 0.75, "PRED 'WSJ dataset for POS induction and dependency parsing' matches gold=WSJ Penn Treebank."),
    ("q027", 0.75, "PRED 'passages do not contain info' implicit match to gold='No'."),
    ("q028", 0.25, "PRED 'three different NER systems' vague; gold=Stanford NER specifically."),
    ("q029", 0.0, "PRED 'above-chance accuracy 75.92%' WRONG; gold=tSNE."),
    ("q030", 0.0, "PRED refuses; gold has specific distribution differences."),
    ("q031", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q032", 0.0, "PRED 'no improvements, devastating linear mapping' WRONG; gold=Table TABREF6."),
    ("q033", 1.0, "PRED 'No, based on vehicle reviews' matches gold='No'."),
    ("q034", 0.5, "PRED 'social phenomenon, act of communication' partial; gold=intentional multicast communication."),
    ("q035", 1.0, "PRED 'RoBERTa' matches gold."),
    ("q036", 0.25, "PRED 'higher or comparable' vague; gold=F1 97.5 MSR, 95.7 AS."),
    ("q037", 0.75, "PRED 'does not specify exact number' implicit match to gold unanswerable."),
    ("q038", 0.0, "PRED refuses; gold=BLEU-1."),
    ("q039", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q040", 0.75, "PRED 'SimpleQuestions and WebQSP' includes gold=SimpleQuestions."),
    ("q041", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q042", 0.0, "PRED 'loss function, validation' doesn't mention MAP; gold=MAP only."),
    ("q043", 0.75, "PRED 'RNN-based NMT and Transformer' includes gold=RNN-based NMT."),
    ("q044", 0.0, "PRED refuses; gold=F-score."),
    ("q045", 1.0, "PRED 'NO-MOVE 30.3% on single, 0.3 on paragraphs' exact match to gold."),
    ("q046", 0.75, "PRED '6 to over 20 accuracy on 100:1000 to 20:1000' matches gold imbalanced part."),
    ("q047", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q048", 0.0, "PRED 'BERT from Khandelwal' WRONG; gold=MaxEnt to SVMs."),
    ("q049", 1.0, "PRED 'five text classification + semantic matching' matches gold."),
    ("q050", 0.75, "PRED '14% of training dataset' matches gold concept of 1000 hours (14%)."),
    ("q051", 0.75, "PRED 'CSAT, 20newsgroups, Fisher' includes gold=CSAT."),
    ("q052", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q053", 0.0, "PRED 'BIBREF7 lower than naive Bayes' WRONG; gold=changing directed to undirected edges."),
    ("q054", 0.0, "PRED refuses; gold=file size on disk."),
    ("q055", 1.0, "PRED 'Yes, SAN attention' matches gold='Yes'."),
    ("q056", 0.5, "PRED 'reducing vocab pushes embeddings from prototype' partial; gold=student/teacher must share vocabulary."),
    ("q057", 1.0, "PRED 'No' matches gold='No'."),
    ("q058", 0.0, "PRED refuses; gold=words user wants in generated output."),
    ("q059", 0.0, "PRED refuses; gold=reducing variance of an estimator."),
    ("q060", 0.0, "PRED 'large-scale + small-scale analyses' WRONG; gold=domain experts provide feedback, dynamic revision."),
    ("q061", 0.0, "PRED hallucinated Jasper exploration uses; gold unanswerable."),
    ("q062", 0.75, "PRED 'passages do not mention pipeline DL components' implicit match to gold='No'."),
    ("q063", 0.25, "PRED 'F1 84.9, not specified improvement' partial; gold=0.8% F1 better."),
    ("q064", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q065", 1.0, "PRED 'conv + NIN + Bi-LSTM + LSTM encoder-decoder' matches gold architecture."),
    ("q066", 0.0, "PRED 'anisotropy penalty' WRONG; gold=self-similarity, intra-sentence, max explainable variance."),
    ("q067", 1.0, "PRED 'mainstream and disinformation news' matches gold."),
    ("q068", 0.0, "PRED 'CoNLL, OntoNotes, WNUT17...' WRONG; gold=BC5CDR, NCBI, BC4CHEMD, JNLPBA, LINNAEUS, Species-800."),
    ("q069", 0.75, "PRED 'does not specify fonts' implicit match to gold unanswerable."),
    ("q070", 0.75, "PRED 'passages do not indicate' implicit match to gold='No'."),
    ("q071", 0.0, "PRED 'unintended translation effects' WRONG; gold=degree of lexical overlap."),
    ("q072", 0.0, "PRED 'No, multiple datasets' hallucinated; gold unanswerable."),
    ("q073", 0.75, "PRED 'language-independent' implicit match to gold='Yes'."),
    ("q074", 0.0, "PRED 'No, dependent on random seed' but gold='Yes' — Y/N flip."),
    ("q075", 0.0, "PRED refuses; gold=353 conversations from 40 speakers."),
    ("q076", 0.0, "PRED 'abstractive and extractive framework' WRONG; gold=extra BERT position embeddings."),
    ("q077", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q078", 0.0, "PRED hallucinated style transfer metrics; gold unanswerable."),
    ("q079", 0.5, "PRED 'four Spanish subtasks' partial; gold names all four (EI-Reg, EI-Oc, V-Reg, V-Oc)."),
    ("q080", 0.0, "PRED 'Google Translate word translation' hallucinated; gold unanswerable."),
    ("q081", 0.75, "PRED 'PTB and WT-2' includes gold=Penn Treebank."),
    ("q082", 0.0, "PRED 'ranked 3rd FLC, 4th SLC' WRONG; gold=specific team names and scores."),
    ("q083", 0.25, "PRED 'Slavic + English' partial; gold=full UD1.2 16-language list."),
    ("q084", 0.75, "PRED 'does not mention seasonality' implicit match to gold unanswerable."),
    ("q085", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q086", 0.5, "PRED 'SVM' partial; gold=word2vec features as input to SVM."),
    ("q087", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q088", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q089", 1.0, "PRED 'WASSA-2017 Shared Task on Emotion Intensity' matches gold."),
    ("q090", 0.25, "PRED 'coherence, goal accomplishment' partial; gold=9 specific metrics including UMA, MRR, BLEU."),
    ("q091", 0.5, "PRED 'combines structured+unstructured' partial; gold=OpenIE toolbox + heuristic rules."),
    ("q092", 0.25, "PRED gives different improvement figures; gold=7.36% accuracy, 9.69% F1."),
    ("q093", 0.75, "PRED 'does not specify' implicit match to gold unanswerable."),
    ("q094", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q095", 0.75, "PRED 'CDA and proposed loss' includes gold=CDA."),
    ("q096", 0.75, "PRED 'attention to auxiliary verbs, adverbs, subjects, objects' matches gold captures beyond alignment."),
    ("q097", 0.0, "PRED 'does not specify'; gold='Yes' — refusal wrong."),
    ("q098", 0.0, "PRED refuses; gold=attentional encoder-decoder."),
    ("q099", 0.25, "PRED 'GlossBERT' partial; gold lists 11 systems including two knowledge-based, two traditional, six neural, one BERT."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_main__qasper__flat-50__seed100__"
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
                skipped += 1; continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"qasper p4_main flat-50 seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
