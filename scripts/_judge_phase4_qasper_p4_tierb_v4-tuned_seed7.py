"""Phase 4 cross-vendor finishing — QASPER p4_tierb v4-tuned seed=7 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_tierb__qasper__v4-tuned__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'n-gram+self-attention' wrong; gold=Bi-LSTM-CRF."),
    ("q001", 0.75, "PRED 'context does not specify English' implicit match to gold unanswerable."),
    ("q002", 0.0, "PRED 'better with un-translated target' wrong; gold=Table TABREF6."),
    ("q003", 0.75, "PRED 'BIBREF9 and BIBREF8' includes gold=BIBREF9."),
    ("q004", 0.75, "PRED 'F-score and Kappa' includes gold=F-score."),
    ("q005", 0.5, "PRED 'dimension, algorithm, learning rate, epochs, window size, vocabulary' partial; gold has 9 items."),
    ("q006", 1.0, "PRED 'Yes, English only' matches gold='Yes'."),
    ("q007", 0.5, "PRED 'annotated via crowdsourcing' partial; gold=manually annotated with crowdsourcing guidance."),
    ("q008", 0.0, "PRED 'No, not auto-optimize' hallucinated; gold unanswerable."),
    ("q009", 0.75, "PRED 'US, BIBREF22 platform (AMT)' matches gold=people in US using AMT."),
    ("q010", 0.75, "PRED 'plain stacked LSTMs, variants, peephole connections' includes gold=Stacked LSTMs."),
    ("q011", 0.0, "PRED refuses; gold=LSTM+VGG16 element-wise+softmax."),
    ("q012", 0.0, "PRED refuses; gold='Yes'."),
    ("q013", 0.0, "PRED refuses; gold=crawling and pre-processing an OSG web forum."),
    ("q014", 0.0, "PRED 'Straw man, obfuscation, slogans, fear...' list; gold=Loaded language (not in truncated list)."),
    ("q015", 0.0, "PRED 'same dataset+patterns' vague; gold=antonym and synonym pairs."),
    ("q016", 0.0, "PRED 'generic+Hindi rules' hallucinated; gold unanswerable."),
    ("q017", 0.75, "PRED 'SimpleQuestions + comparable WebQSP' includes gold=SimpleQuestions."),
    ("q018", 0.25, "PRED 'augmentation better than weighting in low-data' partial; gold has specific percentages."),
    ("q019", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q020", 0.5, "PRED 'conv+NIN+unidirectional LSTM' partial; gold=conv+NIN+BiLSTM (bidirectional)."),
    ("q021", 1.0, "PRED 'Yes, use attention' matches gold='Yes'."),
    ("q022", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q023", 0.75, "PRED 'five profile attributes including username' includes gold=username."),
    ("q024", 0.75, "PRED 'passages do not mention seasonality' implicit match to gold unanswerable."),
    ("q025", 0.5, "PRED 'considerable improvement compared to baselines' partial; gold=AutoJudge consistently+significantly outperforms."),
    ("q026", 1.0, "PRED 'Yes, word segmentation errors, spelling, rare words' matches gold='Yes'."),
    ("q027", 0.0, "PRED '75.92% above-chance' wrong; gold=tSNE."),
    ("q028", 0.25, "PRED 'higher or comparable' vague; gold=F1 97.5 MSR, 95.7 AS."),
    ("q029", 1.0, "PRED '22,880 users' exact match to gold."),
    ("q030", 0.75, "PRED 'passages do not provide info' implicit match to gold unanswerable."),
    ("q031", 0.0, "PRED refuses; gold=PDTB taggers."),
    ("q032", 0.0, "PRED 'max turn length 160' wrong; gold=two previous turns."),
    ("q033", 0.0, "PRED 'RMSE style transfer' hallucinated; gold unanswerable."),
    ("q034", 0.5, "PRED 'KG-A2C-chained+Explore outperform, pass 40' partial; gold=specific reward values 11.8/41.8/40/44."),
    ("q035", 0.5, "PRED 'expert clearly superior, bigger diff on difficult' partial; gold=3.5 F1 improvement for difficult subset."),
    ("q036", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q037", 0.75, "PRED 'EM and Macro-F1' includes gold=EM."),
    ("q038", 0.0, "PRED 'CNN-RNN image-to-poem, seq2seq style transfer' wrong; gold=actor-critic architecture."),
    ("q039", 0.0, "PRED 'hierarchical Transformer representations' wrong; gold=extra BERT position embeddings."),
    ("q040", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q041", 1.0, "PRED 'German-English' exact match to gold."),
    ("q042", 0.75, "PRED 'expected number of unique outputs attacker can induce' close to gold=distinct word recognition outputs."),
    ("q043", 0.0, "PRED 'BIBREF12 BERT results' wrong; gold=MaxEnt to SVMs."),
    ("q044", 0.0, "PRED 'both original+validation as references' wrong; gold=BLEU-1."),
    ("q045", 0.0, "PRED 'Yes, MTL' Y/N flip; gold='No'."),
    ("q046", 0.75, "PRED 'passages do not mention transformer analysis' implicit match to gold='No'."),
    ("q047", 1.0, "PRED '45,000+ articles about COVID-19, SARS-CoV-2, 33,000+ full text' exact match to gold."),
    ("q048", 0.25, "PRED 'LAN+P600 and ELAN+P600' includes ELAN; gold=ELAN, LAN benefit."),
    ("q049", 1.0, "PRED 'No, demonstrates effectiveness' matches gold='No'."),
    ("q050", 0.0, "PRED 'No, Chinese-English' Y/N flip; gold='Yes'."),
    ("q051", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q052", 0.25, "PRED 'Czech, French, Italian, English, Danish' 5 of 16; gold=full UD1.2 list."),
    ("q053", 0.0, "PRED 'grammaticality, perplexity' wrong; gold=INLINEFORM0 scores."),
    ("q054", 0.0, "PRED 'extractive SummaRunNer vs Lead-3' hallucinated; gold unanswerable."),
    ("q055", 0.0, "PRED 'lattice-free MMI' wrong; gold=CNN-DNN-BLSTM-HMM."),
    ("q056", 0.0, "PRED refuses; gold=constrain model on data structure to prevent contradictions."),
    ("q057", 0.75, "PRED 'passages do not contain info' implicit match to gold unanswerable."),
    ("q058", 0.75, "PRED 'RoBERTa encoder-only' matches gold=RoBERTa."),
    ("q059", 1.0, "PRED 'NO-MOVE 30.3% single, 0.3% paragraphs' exact match to gold."),
    ("q060", 0.0, "PRED refuses; gold=2,100+ texts, 32k questions statistics."),
    ("q061", 0.5, "PRED 'demographics, diagnosis, symptoms/signs, among others' covers ~5 of 10 gold categories."),
    ("q062", 0.0, "PRED 'Existing Approaches' wrong; gold=CDA."),
    ("q063", 0.75, "PRED 'transcribed text + ASR system' matches gold=text transcription."),
    ("q064", 0.0, "PRED refuses; gold=EI-Reg, EI-Oc, V-Reg, V-Oc."),
    ("q065", 1.0, "PRED 'No, votes+speeches combined' matches gold='No'."),
    ("q066", 0.25, "PRED 'limited monolingual Hindi' partial; gold=No data, pretrained model."),
    ("q067", 0.75, "PRED 'passages do not contain info' implicit match to gold unanswerable."),
    ("q068", 1.0, "PRED 'Yes, ShapeWorld abstract shapes' matches gold='Yes'."),
    ("q069", 0.0, "PRED 'No, shallow syntactic features' Y/N flip; gold='Yes'."),
    ("q070", 0.5, "PRED 'text structure, typography, images' includes paragraphs concept; gold=paragraphs."),
    ("q071", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q072", 0.0, "PRED 'Libertarianism, Anarcho_Capitalism...' wrong; gold=politics, business, science, AskReddit."),
    ("q073", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q074", 0.75, "PRED 'CSAT, 20newsgroups, Fisher' includes gold=CSAT."),
    ("q075", 0.0, "PRED 'neural models' vague; gold=linear SVM."),
    ("q076", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q077", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q078", 0.75, "PRED 'context does not specify' implicit match to gold unanswerable."),
    ("q079", 0.0, "PRED 'No' Y/N flip; gold='Yes'."),
    ("q080", 1.0, "PRED '0.94 F1 Wikipedia/Twitter, 0.95 Formspring' exact match to gold."),
    ("q081", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q082", 0.0, "PRED 'biLSTM F1 84-94%' wrong; gold=8 named NER systems."),
    ("q083", 0.75, "PRED 'passages do not mention error analysis' implicit match to gold='No'."),
    ("q084", 0.75, "PRED 'context does not specify' implicit match to gold unanswerable."),
    ("q085", 1.0, "PRED 'IMDB and Yelp reviews, imdb400/yelp50/yelp200' matches gold=three datasets based on IMDB+Yelp."),
    ("q086", 0.5, "PRED '14% of training dataset' partial; gold=1000 hours."),
    ("q087", 0.75, "PRED 'passages do not specify' implicit match to gold='No'."),
    ("q088", 1.0, "PRED '45,821 characters' exact match to gold."),
    ("q089", 0.0, "PRED refuses; gold='Yes'."),
    ("q090", 0.25, "PRED 'TAC2010, WW, ACE2004, AQUAINT, CoNLL' includes CoNLL but misses YAGO; gold=CoNLL-YAGO."),
    ("q091", 0.25, "PRED 'ironic to non-ironic transformation' partial; gold=Irony Classifier specifically."),
    ("q092", 0.0, "PRED 'Yes, LDA unsupervised' Y/N flip; gold='No'."),
    ("q093", 0.75, "PRED 'CPMI_{-2} and NNEGPMI' matches gold=clipped PMI; NNEGPMI."),
    ("q094", 0.25, "PRED 'annotated by 145 annotators, 1888 pairs' partial; gold=annotation instructions protocol."),
    ("q095", 0.0, "PRED 'ASR, MT, ST domains' wrong; gold=no specific domain."),
    ("q096", 0.0, "PRED 'Nearest Number, Random Top-3' wrong; gold=QA PGNet."),
    ("q097", 1.0, "PRED 'No, do not compare human vs model' matches gold='No'."),
    ("q098", 0.25, "PRED 'fifteen celebrities' correct count but no names; gold=list of 15 names."),
    ("q099", 0.0, "PRED 'seven SentEval tasks' vague; gold=MR."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_tierb__qasper__v4-tuned__seed7__"
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
    print(f"qasper p4_tierb v4-tuned seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
