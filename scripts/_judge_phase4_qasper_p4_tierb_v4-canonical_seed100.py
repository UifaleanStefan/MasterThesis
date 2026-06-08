"""Phase 4 cross-vendor finishing — QASPER p4_tierb v4-canonical seed=100 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_tierb__qasper__v4-canonical__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.75, "PRED 'passages do not provide info' implicit match to gold unanswerable."),
    ("q001", 0.75, "PRED 'passages do not provide info about transcription' implicit match to gold unanswerable."),
    ("q002", 0.0, "PRED 'MLM+BRLM-SA' vague; gold=BIBREF19 specifically."),
    ("q003", 0.0, "PRED 'NUS 94.0% vs ABUS 45.5%, 48.5pp' completely wrong; gold=2.6 percent points."),
    ("q004", 0.25, "PRED 'LDA 100 topics Gibbs sampling ISWC/WWW' partial; gold=broader taxonomy description."),
    ("q005", 0.0, "PRED refuses; gold=four MT tasks DE-EN, JA-EN, RO-EN, EN-DE."),
    ("q006", 1.0, "PRED 'No' matches gold='No'."),
    ("q007", 0.25, "PRED 'analyze tweets sentiment/NER' misses platform/demographic; gold=people in US using AMT."),
    ("q008", 0.0, "PRED 'Paraphrase Database (PPDB)' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q010", 0.75, "PRED 'passages do not indicate specific domain' implicit match to gold='No'."),
    ("q011", 0.0, "PRED 'hybrid rule+phrase-based SMT' wrong; gold=Transformer architecture."),
    ("q012", 0.0, "PRED 'CNN and BiLSTM' wrong; gold=linear SVM."),
    ("q013", 0.75, "PRED 'BIBREF9 and BIBREF8' includes gold=BIBREF9."),
    ("q014", 0.0, "PRED refuses; gold='Yes'."),
    ("q015", 0.75, "PRED 'CPMI_{-2} and NNEGPMI' matches gold=clipped PMI; NNEGPMI."),
    ("q016", 0.0, "PRED 'recall on tens/hundreds of keyphrases' wrong; gold=average unique predictions."),
    ("q017", 0.0, "PRED 'grammaticality, perplexity' wrong; gold=INLINEFORM0 scores."),
    ("q018", 0.75, "PRED 'passages do not specify' implicit match to gold='No'."),
    ("q019", 1.0, "PRED 'German-English' matches gold=De-En."),
    ("q020", 0.0, "PRED refuses; gold=specific F1 scores 85.99/75.15/71.53."),
    ("q021", 0.0, "PRED 'Turkish 72.71%' wrong; gold=Russian."),
    ("q022", 0.0, "PRED refuses; gold=MR."),
    ("q023", 1.0, "PRED 'Yes, lemmatization can hurt' matches gold='Yes'."),
    ("q024", 0.0, "PRED hallucinated WER '15.47%'; gold unanswerable."),
    ("q025", 0.0, "PRED 'CEI Only, CP+CEI baselines' wrong; gold=PDTB taggers."),
    ("q026", 0.75, "PRED 'WSJ for POS induction + dependency parsing' matches gold=WSJ Penn Treebank."),
    ("q027", 1.0, "PRED 'No, rule-based' matches gold='No'."),
    ("q028", 0.75, "PRED 'spaCy, Char-biLSTM+CRF, Stanford NER' includes gold=Stanford NER."),
    ("q029", 0.0, "PRED 'kappa scores' wrong; gold=tSNE."),
    ("q030", 0.5, "PRED 'friends distribution significantly different' partial; gold specifies Friends+URLs diff, others not."),
    ("q031", 0.0, "PRED refuses; gold='Yes'."),
    ("q032", 0.0, "PRED 'no improvements, devastating' wrong; gold=Table TABREF6."),
    ("q033", 0.0, "PRED 'Yes, abstract features' Y/N flip; gold='No'."),
    ("q034", 0.5, "PRED 'social phenomenon, act of communication' partial; gold=intentional multicast communication."),
    ("q035", 1.0, "PRED 'RoBERTa' matches gold."),
    ("q036", 0.25, "PRED 'higher or comparable' vague; gold=F1 97.5 MSR, 95.7 AS."),
    ("q037", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q038", 0.75, "PRED 'METEOR, ROUGE-L, BLEU-1' includes gold=BLEU-1."),
    ("q039", 0.75, "PRED 'passages do not specify improvement' implicit match to gold unanswerable."),
    ("q040", 0.0, "PRED 'WebQSP' wrong; gold=SimpleQuestions."),
    ("q041", 0.0, "PRED 'No' Y/N flip; gold='Yes'."),
    ("q042", 0.0, "PRED 'loss function, validation' wrong; gold=MAP."),
    ("q043", 0.75, "PRED 'RNN-based NMT and Transformer' includes gold=RNN-based NMT."),
    ("q044", 0.0, "PRED 'Chi squared, Information Gain' wrong; gold=F-score."),
    ("q045", 1.0, "PRED 'NO-MOVE 30.3% single, 0.3 paragraphs' exact match to gold."),
    ("q046", 0.75, "PRED '6 accuracy points 100:1000, >20 points 20:1000' matches gold imbalanced section."),
    ("q047", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q048", 0.0, "PRED refuses; gold=MaxEnt to SVMs."),
    ("q049", 0.5, "PRED 'two NLP tasks, six datasets' partial; gold=text classification + text semantic matching."),
    ("q050", 1.0, "PRED '14% corresponds to 1000 hours' matches gold=1000 hours data."),
    ("q051", 0.75, "PRED 'CSAT, Fisher, 20newsgroups' includes gold=CSAT."),
    ("q052", 0.0, "PRED refuses; gold='Yes'."),
    ("q053", 0.0, "PRED 'BIBREF53 least impactful' wrong; gold=directed to undirected edges."),
    ("q054", 0.75, "PRED 'model size in MB' matches gold=file size on disk."),
    ("q055", 1.0, "PRED 'Yes, SAN' matches gold='Yes'."),
    ("q056", 0.75, "PRED 'student/teacher must share vocab' matches gold core claim."),
    ("q057", 0.0, "PRED 'Yes' Y/N flip; gold='No'."),
    ("q058", 0.0, "PRED refuses; gold=words user wants in generated output."),
    ("q059", 0.0, "PRED refuses; gold=reducing variance of an estimator."),
    ("q060", 0.0, "PRED 'large-scale + small-scale analyses' wrong; gold=domain experts provide feedback."),
    ("q061", 0.0, "PRED hallucinated baselines; gold unanswerable."),
    ("q062", 0.75, "PRED 'passages do not mention DL pipeline' implicit match to gold='No'."),
    ("q063", 0.0, "PRED refuses; gold=0.8% F1 better."),
    ("q064", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q065", 0.25, "PRED 'Attention Encoder-Decoder' partial; gold=conv+NIN+BiLSTM architecture."),
    ("q066", 0.25, "PRED 'GPT-2 intra-sentence similarity' partial; gold=self-similarity+intra-sentence+max explainable variance."),
    ("q067", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q068", 0.0, "PRED refuses; gold=BC5CDR-disease, NCBI-disease, BC5CDR-chem, BC4CHEMD, BC2GM, JNLPBA, LINNAEUS, Species-800."),
    ("q069", 0.75, "PRED 'passages do not indicate' implicit match to gold='No'."),
    ("q070", 0.0, "PRED refuses; gold=degree of lexical overlap."),
    ("q071", 0.0, "PRED hallucinated; gold unanswerable."),
    ("q072", 0.5, "PRED 'cannot specify but mentions English hashtags' consistent with gold='Yes' but indirect."),
    ("q073", 0.0, "PRED 'may not generalize' implies No; gold='Yes'."),
    ("q074", 0.0, "PRED '100k' wrong; gold=353 conversations from 40 speakers."),
    ("q075", 0.0, "PRED 'separate optimizers' wrong; gold=extra BERT position embeddings."),
    ("q076", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q077", 0.75, "PRED 'document does not provide details' implicit match to gold unanswerable."),
    ("q078", 1.0, "PRED 'EI-Reg, EI-Oc, V-Reg, V-Oc' exact match to gold."),
    ("q079", 0.0, "PRED 'Google Translate word translation' hallucinated; gold unanswerable."),
    ("q080", 0.75, "PRED 'PTB and WT-2' includes gold=Penn Treebank."),
    ("q081", 0.0, "PRED 'ranked 3rd position' wrong; gold=specific team names+scores."),
    ("q082", 0.25, "PRED 'Czech, French, Italian, Indonesian' 4 of 16 languages; gold=full UD1.2 list."),
    ("q083", 0.75, "PRED 'passages do not mention seasonality' implicit match to gold unanswerable."),
    ("q084", 0.0, "PRED 'Yes' Y/N flip; gold='No'."),
    ("q085", 0.5, "PRED 'word2vec approach' partial; gold=word2vec features as input to SVM."),
    ("q086", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q087", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q088", 1.0, "PRED 'WASSA-2017 Shared Task on Emotion Intensity' matches gold."),
    ("q089", 0.0, "PRED 'coherence, semantic plausibility' wrong; gold=9 specific metrics BPE-PPL, BLEU-1/4, etc."),
    ("q090", 0.25, "PRED 'combines structured+unstructured' partial; gold=OpenIE toolbox + heuristic rules."),
    ("q091", 0.0, "PRED '12.27% accuracy, 14.86% F1' wrong; gold=7.36% accuracy, 9.69% F1."),
    ("q092", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q093", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q094", 0.75, "PRED 'CDA and proposed' includes gold=CDA."),
    ("q095", 0.75, "PRED 'attention captures other parts beyond aligned word, esp. verbs' matches gold."),
    ("q096", 0.0, "PRED refuses; gold='Yes'."),
    ("q097", 0.0, "PRED refuses; gold=attentional encoder-decoder networks BIBREF0."),
    ("q098", 0.0, "PRED 'GlossBERT' partial; gold lists 11 systems in categories."),
    ("q099", 0.0, "PRED refuses; gold=attentional encoder-decoder networks."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_tierb__qasper__v4-canonical__seed100__"
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
    print(f"qasper p4_tierb v4-canonical seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
