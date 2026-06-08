"""Phase 4 cross-vendor finishing — QASPER p4_tierb v4-tuned seed=100 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_tierb__qasper__v4-tuned__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.75, "PRED 'passages do not provide info' implicit match to gold unanswerable."),
    ("q001", 0.0, "PRED 'G2P transcription' hallucinated; gold unanswerable."),
    ("q002", 0.0, "PRED 'MLM+TLM' vague; gold=BIBREF19 specifically."),
    ("q003", 0.0, "PRED 'NUS 94.0% vs ABUS 45.5%' wrong; gold=2.6 percent points."),
    ("q004", 0.25, "PRED 'LDA Gibbs sampling ISWC/WWW' partial; gold=broader taxonomy description."),
    ("q005", 0.0, "PRED refuses; gold=four MT tasks."),
    ("q006", 1.0, "PRED 'No, trained professionals' matches gold='No'."),
    ("q007", 0.75, "PRED 'US, BIBREF22 platform (AMT)' matches gold=people in US using AMT."),
    ("q008", 0.0, "PRED 'distant-supervised emoticons' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q010", 1.0, "PRED 'No, specific domain' matches gold='No'."),
    ("q011", 0.0, "PRED 'speech synthesis, recognition, MT' wrong; gold=Transformer architecture."),
    ("q012", 0.0, "PRED 'neural models' vague; gold=linear SVM."),
    ("q013", 0.75, "PRED 'BIBREF9 and BIBREF8' includes gold=BIBREF9."),
    ("q014", 0.0, "PRED refuses; gold='Yes'."),
    ("q015", 0.75, "PRED 'PPMI, CPMI_{-2}, NNEGPMI' includes clipped PMI+NNEGPMI; matches gold."),
    ("q016", 0.0, "PRED 'inter-relation among phrases' wrong; gold=average unique predictions."),
    ("q017", 0.0, "PRED 'grammaticality, perplexity' wrong; gold=INLINEFORM0 scores."),
    ("q018", 1.0, "PRED 'No, not publicly available, PHI conditions' matches gold='No'."),
    ("q019", 0.75, "PRED 'De-En, Ja-En, Ro-En' includes gold=De-En."),
    ("q020", 0.0, "PRED refuses; gold=specific F1 scores 85.99/75.15/71.53."),
    ("q021", 0.0, "PRED 'Turkish 72.71%' wrong; gold=Russian."),
    ("q022", 0.0, "PRED 'seven SentEval tasks' vague; gold=MR."),
    ("q023", 1.0, "PRED 'Yes, lemmatization can hurt' matches gold='Yes'."),
    ("q024", 0.0, "PRED hallucinated '4.85%/15.09% WER'; gold unanswerable."),
    ("q025", 0.0, "PRED refuses; gold=PDTB taggers."),
    ("q026", 0.75, "PRED 'WSJ sections 02-21 training, 23 evaluation' matches gold=WSJ Penn Treebank."),
    ("q027", 1.0, "PRED 'No, handcrafted rules' matches gold='No'."),
    ("q028", 0.75, "PRED 'Stanford NER, spaCy 2.0, BiLSTM' includes gold=Stanford NER."),
    ("q029", 0.0, "PRED '75.92% above-chance accuracy' wrong; gold=tSNE."),
    ("q030", 0.5, "PRED 'URLs diff statistically, Friends/Followers also mentioned' partial; gold=Followers+Friends+URLs diff, others not."),
    ("q031", 0.0, "PRED refuses; gold='Yes'."),
    ("q032", 0.0, "PRED 'better with un-translated target' wrong; gold=Table TABREF6."),
    ("q033", 0.0, "PRED 'Yes, abstract features' Y/N flip; gold='No'."),
    ("q034", 0.75, "PRED 'intentional multicast communication designed to influence' matches gold."),
    ("q035", 1.0, "PRED 'RoBERTa encoder-only' matches gold=RoBERTa."),
    ("q036", 0.25, "PRED 'higher or comparable' vague; gold=F1 97.5 MSR, 95.7 AS."),
    ("q037", 0.0, "PRED 'full list of recommended tags' hallucinated; gold unanswerable."),
    ("q038", 0.0, "PRED 'both original and validation answers as references' wrong; gold=BLEU-1."),
    ("q039", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q040", 0.75, "PRED 'SimpleQuestions + comparable on WebQSP' includes gold=SimpleQuestions."),
    ("q041", 0.0, "PRED 'No' Y/N flip; gold='Yes'."),
    ("q042", 0.0, "PRED 'loss function, validation' wrong; gold=MAP."),
    ("q043", 0.75, "PRED 'RNN-based NMT and Transformer' includes gold=RNN-based NMT."),
    ("q044", 0.75, "PRED 'F-score and Kappa' includes gold=F-score."),
    ("q045", 1.0, "PRED 'NO-MOVE 30.3% single, 0.3% paragraphs' exact match to gold."),
    ("q046", 0.25, "PRED 'augmentation better than weighting' partial; gold has specific 1-2/6+/20+ point improvements."),
    ("q047", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q048", 0.0, "PRED 'BIBREF12 BERT results' wrong; gold=MaxEnt to SVMs."),
    ("q049", 0.5, "PRED 'two typical NLP tasks involving six datasets' partial; gold=text classification + text semantic matching."),
    ("q050", 0.5, "PRED '14% of training dataset' partial; gold=1000 hours."),
    ("q051", 0.75, "PRED 'CSAT, 20newsgroups, Fisher' includes gold=CSAT."),
    ("q052", 0.0, "PRED refuses; gold='Yes'."),
    ("q053", 0.0, "PRED 'GRU combine function least impactful' wrong; gold=directed to undirected edges."),
    ("q054", 0.75, "PRED 'model file size MB, LangID-High vs wFST' matches gold=file size on disk."),
    ("q055", 1.0, "PRED 'Yes, use attention' matches gold='Yes'."),
    ("q056", 1.0, "PRED 'student/teacher must share vocab' matches gold exactly."),
    ("q057", 0.0, "PRED 'Yes' Y/N flip; gold='No'."),
    ("q058", 0.5, "PRED 'user-defined keywords {subway, manhattan}' gives example; gold=definition."),
    ("q059", 0.0, "PRED refuses; gold=reducing variance of an estimator."),
    ("q060", 0.0, "PRED 'large-scale + small-scale analyses' wrong; gold=domain experts provide feedback."),
    ("q061", 0.0, "PRED hallucinated Jasper/Hub5; gold unanswerable."),
    ("q062", 0.75, "PRED 'passages do not specify' implicit match to gold='No'."),
    ("q063", 0.0, "PRED refuses; gold=0.8% F1 better."),
    ("q064", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q065", 0.5, "PRED 'conv+NIN+unidirectional LSTM' partial; gold=conv+NIN+BiLSTM (bidirectional)."),
    ("q066", 0.5, "PRED 'intra-sentence similarity adjusted for anisotropy' partial; gold=self-similarity+intra-sentence+max explainable variance."),
    ("q067", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q068", 0.0, "PRED refuses; gold=8 named NER datasets."),
    ("q069", 0.75, "PRED 'passages do not specify fonts' implicit match to gold unanswerable."),
    ("q070", 1.0, "PRED 'No, demonstrates effectiveness' matches gold='No'."),
    ("q071", 0.0, "PRED 'translationese and annotation artifacts' wrong; gold=degree of lexical overlap."),
    ("q072", 0.0, "PRED hallucinated; gold unanswerable."),
    ("q073", 1.0, "PRED 'Yes, English hashtag and SemEval' matches gold='Yes'."),
    ("q074", 0.0, "PRED 'may not generalize' implies No; gold='Yes'."),
    ("q075", 0.75, "PRED '353 conversations, 41 hours' includes gold=353 conversations."),
    ("q076", 0.0, "PRED 'hierarchical Transformer representations' wrong; gold=extra BERT position embeddings."),
    ("q077", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q078", 0.0, "PRED 'RMSE style transfer score' hallucinated; gold unanswerable."),
    ("q079", 0.0, "PRED refuses; gold=EI-Reg, EI-Oc, V-Reg, V-Oc."),
    ("q080", 0.0, "PRED 'generic+Hindi-tuned rules' hallucinated; gold unanswerable."),
    ("q081", 0.75, "PRED 'PTB and WT-2' includes gold=Penn Treebank."),
    ("q082", 0.0, "PRED 'ranked 3rd, F1=0.673' wrong; gold=specific team names+scores."),
    ("q083", 0.25, "PRED 'Czech, French, Italian, English, Danish' 5 of 16; gold=full UD1.2 list."),
    ("q084", 0.75, "PRED 'passages do not mention seasonality' implicit match to gold unanswerable."),
    ("q085", 1.0, "PRED 'No, combining votes+speeches' matches gold='No'."),
    ("q086", 0.0, "PRED 'self score' wrong; gold=word2vec features as input to SVM."),
    ("q087", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q088", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q089", 0.5, "PRED 'dev data BIBREF19' partial; gold=WASSA-2017 Shared Task on Emotion Intensity (BIBREF19 is that task)."),
    ("q090", 0.0, "PRED 'coherence, user preference' wrong; gold=9 specific metrics."),
    ("q091", 0.75, "PRED 'OpenIE toolbox, derives relations, selects informative relation' matches gold."),
    ("q092", 0.0, "PRED '12.27% accuracy, 14.86% F1' wrong; gold=7.36%/9.69%."),
    ("q093", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q094", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q095", 0.0, "PRED 'Existing Approaches' wrong; gold=CDA."),
    ("q096", 0.5, "PRED 'attention captures alignment differences in syntactic phenomena' partial; gold=captures other info beyond translational equivalent for verbs."),
    ("q097", 0.0, "PRED 'No' Y/N flip; gold='Yes'."),
    ("q098", 0.0, "PRED refuses; gold=attentional encoder-decoder networks."),
    ("q099", 0.0, "PRED 'IMS+emb BIBREF9' partial; gold=11 systems in categories (two knowledge-based, two traditional, six neural, one BERT)."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_tierb__qasper__v4-tuned__seed100__"
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
    print(f"qasper p4_tierb v4-tuned seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
