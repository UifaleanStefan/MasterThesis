"""Phase 4 cross-vendor finishing — QASPER p4_tierb v4-canonical seed=7 (99 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_tierb__qasper__v4-canonical__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'n-gram+self-attention' wrong; gold=Bi-LSTM-CRF."),
    ("q001", 0.0, "PRED hallucinated Freebase+WordNet; gold unanswerable."),
    ("q002", 0.0, "PRED 'no improvements, devastating linear mapping' wrong; gold=Table TABREF6."),
    ("q003", 0.75, "PRED 'BIBREF9 and BIBREF8' includes gold=BIBREF9."),
    ("q004", 0.0, "PRED 'Chi squared, Information Gain' wrong; gold=F-score."),
    ("q005", 0.25, "PRED 'dimensions, model combos, corpus size' partial; gold has 9 hyperparameters."),
    ("q006", 0.0, "PRED refuses; gold='Yes'."),
    ("q007", 0.5, "PRED 'bootstrapped via crowdsourcing' partial; gold=annotated manually with crowdsourcing guidance."),
    ("q008", 0.0, "PRED 'No, doesn't auto-optimize' hallucinated; gold unanswerable."),
    ("q009", 0.25, "PRED 'analyze tweets sentiment/NER' misses US+AMT; gold=people in US using AMT."),
    ("q010", 0.5, "PRED 'CAS-LSTM, cell states, peephole' captures stacked LSTM concept; gold=Stacked LSTMs."),
    ("q011", 0.5, "PRED 'LSTM-CNN' partial; gold=LSTM+VGG16 multiplied element-wise+softmax."),
    ("q012", 0.0, "PRED refuses; gold='Yes'."),
    ("q013", 0.0, "PRED refuses; gold=crawling and pre-processing an OSG web forum."),
    ("q014", 0.0, "PRED refuses; gold=Loaded language."),
    ("q015", 0.0, "PRED 'same dataset as Overall Results' vague; gold=antonym and synonym pairs."),
    ("q016", 0.0, "PRED 'Google Translate word translation' hallucinated; gold unanswerable."),
    ("q017", 0.0, "PRED 'WebQSP' wrong; gold=SimpleQuestions."),
    ("q018", 0.75, "PRED '6 accuracy on 100:1000, >20 on 20:1000' matches gold imbalanced section."),
    ("q019", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q020", 0.25, "PRED 'Attention Encoder-Decoder' partial; gold=conv+NIN+BiLSTM architecture."),
    ("q021", 1.0, "PRED 'Yes, SAN' matches gold='Yes'."),
    ("q022", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q023", 0.0, "PRED refuses; gold=username."),
    ("q024", 0.75, "PRED 'passages do not mention seasonality' implicit match to gold unanswerable."),
    ("q025", 0.0, "PRED 'quite limited improvement' contradicts gold=AutoJudge consistently outperforms."),
    ("q026", 0.5, "PRED 'implies challenges but not explicit' partial; gold='Yes' challenges exist."),
    ("q027", 0.0, "PRED 'kappa scores' wrong; gold=tSNE."),
    ("q028", 0.25, "PRED 'higher or comparable' vague; gold=F1 97.5 MSR, 95.7 AS."),
    ("q029", 1.0, "PRED '22,880 users' exact match to gold."),
    ("q030", 0.75, "PRED 'passages do not provide info' implicit match to gold unanswerable."),
    ("q031", 0.0, "PRED 'CEI Only, CP+CEI' wrong; gold=PDTB taggers."),
    ("q032", 0.0, "PRED 'max turn length 160' wrong; gold=two previous turns."),
    ("q033", 0.75, "PRED 'document does not specify' implicit match to gold unanswerable."),
    ("q034", 0.5, "PRED 'KG-A2C-chained/Explore outperform, score>40' partial; gold=specific reward values 11.8/41.8/40/44."),
    ("q035", 0.25, "PRED 'expert annotations improve' vague; gold=3.5 F1 for difficult subset."),
    ("q036", 1.0, "PRED 'Yes' matches gold='Yes'."),
    ("q037", 0.75, "PRED 'EM and F1 scores' includes gold=EM."),
    ("q038", 0.0, "PRED 'seq2seq global attention + pointer networks' wrong; gold=actor-critic architecture."),
    ("q039", 0.0, "PRED 'separate optimizers' wrong; gold=extra BERT position embeddings."),
    ("q040", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q041", 1.0, "PRED 'German-English' exact match to gold."),
    ("q042", 0.0, "PRED 'OOV responsiveness/classification accuracy' wrong; gold=distinct word recognition outputs attacker can induce."),
    ("q043", 0.0, "PRED refuses; gold=MaxEnt to SVMs."),
    ("q044", 0.75, "PRED 'METEOR, ROUGE-L, BLEU-1' includes gold=BLEU-1."),
    ("q045", 0.0, "PRED 'Yes, MTL' Y/N flip; gold='No'."),
    ("q046", 0.75, "PRED 'passages do not mention transformer analysis' implicit match to gold='No'."),
    ("q047", 0.0, "PRED refuses; gold=45,000+ articles about COVID-19."),
    ("q048", 0.0, "PRED 'N400+P600, post-N400+N400 pairs' wrong; gold=ELAN, LAN."),
    ("q049", 0.75, "PRED 'passages do not indicate' implicit match to gold='No'."),
    ("q050", 0.0, "PRED 'No' Y/N flip; gold='Yes'."),
    ("q051", 0.0, "PRED 'Yes, English only' hallucinated; gold unanswerable."),
    ("q052", 0.25, "PRED 'Czech, French, Italian, Indonesian, English' 5 of 16; gold=full UD1.2 list."),
    ("q053", 0.0, "PRED 'grammaticality, perplexity' wrong; gold=INLINEFORM0 scores."),
    ("q054", 0.0, "PRED 'extractive SummaRunNer vs Lead-3' hallucinated; gold unanswerable."),
    ("q055", 0.5, "PRED 'HMM-based with conv+recurrent+FC, speaker adaptation' describes CNN-DNN-BLSTM-HMM architecture."),
    ("q056", 1.0, "PRED 'further constraining model on data structure to prevent contradictions' matches gold."),
    ("q057", 0.75, "PRED 'passages do not contain info on agreement' implicit match to gold unanswerable."),
    ("q058", 1.0, "PRED 'RoBERTa' matches gold."),
    ("q059", 1.0, "PRED 'NO-MOVE 30.3% single, 0.3 paragraphs' exact match to gold."),
    ("q060", 0.0, "PRED 'becomes benchmark for commonsense' wrong; gold=2,100+ texts, 32k questions statistics."),
    ("q061", 0.5, "PRED 'diagnosis, movement, symptoms, vitals, medication' covers ~5 of 10 gold categories."),
    ("q062", 0.75, "PRED 'CDA and proposed' includes gold=CDA."),
    ("q063", 0.75, "PRED 'transcribed text + ASR system' matches gold=text transcription."),
    ("q064", 1.0, "PRED 'EI-Reg, EI-Oc, V-Reg, V-Oc' exact match to gold."),
    ("q065", 0.0, "PRED 'Yes, use votes' Y/N flip; gold='No'."),
    ("q066", 0.75, "PRED 'passages do not specify amount' implicit match to gold=No data, pretrained model."),
    ("q067", 0.75, "PRED 'passages do not contain info on agreement' implicit match to gold unanswerable."),
    ("q068", 1.0, "PRED 'Yes, specific domain' matches gold='Yes'."),
    ("q069", 0.0, "PRED refuses; gold='Yes'."),
    ("q070", 0.75, "PRED 'physical page, paragraph, line segmentation' includes gold=paragraphs."),
    ("q071", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q072", 0.0, "PRED 'Libertarianism, Anarcho_Capitalism...' wrong; gold=politics, business, science, AskReddit."),
    ("q073", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q074", 0.75, "PRED 'CSAT, Fisher, 20newsgroups' includes gold=CSAT."),
    ("q075", 0.0, "PRED 'CNN and BiLSTM' wrong; gold=linear SVM."),
    ("q076", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q077", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q078", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q079", 0.0, "PRED 'No' Y/N flip; gold='Yes'."),
    ("q080", 1.0, "PRED '0.94 F1 Wikipedia/Twitter, 0.95 Formspring' exact match to gold."),
    ("q081", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q083", 0.75, "PRED 'passages do not mention error analysis' implicit match to gold='No'."),
    ("q084", 0.75, "PRED 'passages do not specify' implicit match to gold unanswerable."),
    ("q085", 0.75, "PRED 'IMDB and Yelp' matches gold=three datasets based on IMDB+Yelp."),
    ("q086", 1.0, "PRED '14% corresponds to 1000 hours' matches gold=1000 hours data."),
    ("q087", 0.75, "PRED 'passages do not mention pipeline DL components' implicit match to gold='No'."),
    ("q088", 0.0, "PRED 'Five characters' wrong; gold=45,821 characters."),
    ("q089", 0.0, "PRED refuses; gold='Yes'."),
    ("q090", 0.25, "PRED 'CoNLL, TAC2010, ACE2004, WW, AQUAINT' includes CoNLL but misses YAGO; gold=CoNLL-YAGO."),
    ("q091", 0.25, "PRED 'irony, content preservation' partial; gold=Irony Classifier specifically."),
    ("q092", 1.0, "PRED 'No' matches gold='No'."),
    ("q093", 0.75, "PRED 'CPMI_{-2} and NNEGPMI' matches gold=clipped PMI; NNEGPMI."),
    ("q094", 0.25, "PRED 'detailed protocol, similarity/frequency/POS/lexical balancing' partial; gold=full annotation instructions."),
    ("q095", 0.0, "PRED 'ST and MT domains' wrong; gold=no specific domain."),
    ("q096", 0.0, "PRED 'ASR transcripts vs commercial ASR' wrong; gold=QA PGNet."),
    ("q097", 0.75, "PRED 'passages do not explicitly mention human-model comparison' implicit match to gold='No'."),
    ("q098", 0.0, "PRED 'UPC/UNC celebrity correlation' wrong; gold=list of 15 celebrity names."),
    ("q099", 0.0, "PRED refuses; gold=MR."),
]


def main() -> None:
    assert len(JUDGMENTS) == 99
    qid_prefix = "p4_tierb__qasper__v4-canonical__seed7__"
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
    print(f"qasper p4_tierb v4-canonical seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
