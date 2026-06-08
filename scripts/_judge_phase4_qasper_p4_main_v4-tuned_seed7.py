"""Phase 4 cross-vendor finishing — QASPER p4_main v4-tuned seed=7 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__v4-tuned__seed7/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.0, "PRED 'n-gram + self-attention' WRONG; gold=Bi-LSTM-CRF."),
    ("q001", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q002", 0.25, "PRED 'fine-tuned on un-translated better' partial."),
    ("q003", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q004", 0.75, "PRED 'F-score and Kappa' includes gold F-score."),
    ("q005", 0.75, "PRED 'vector dim + algo + lr + epochs + window + vocab' matches most of gold."),
    ("q006", 1.0, "PRED 'Yes only English' matches gold='Yes'."),
    ("q007", 0.75, "PRED 'annotated using crowdsourcing' matches gold."),
    ("q008", 0.0, "PRED 'No does not optimize' hallucinated; gold unanswerable."),
    ("q009", 0.5, "PRED 'individuals in US on BIBREF22' partial."),
    ("q010", 1.0, "PRED 'plain stacked LSTMs' matches gold='Stacked LSTMs'."),
    ("q011", 0.0, "PRED refuses; gold=LSTM+VGG16."),
    ("q012", 0.0, "PRED refuses; gold='Yes'."),
    ("q013", 0.0, "PRED 'not detailed' refusal; gold=crawling OSG web forum."),
    ("q014", 0.25, "PRED lists techniques but 'Loaded language' not in PRED list."),
    ("q015", 0.0, "PRED 'Overall Results' vague; gold=antonym/synonym pairs."),
    ("q016", 0.0, "PRED 'Hindi-tuned rules' hallucinated; gold unanswerable."),
    ("q017", 0.75, "PRED 'SimpleQuestions and WebQSP' includes gold."),
    ("q018", 0.25, "PRED 'augmentation and weighting boost' vague."),
    ("q019", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q020", 0.75, "PRED 'conv + NIN + LSTM with attention' matches gold's CNN+NIN+BLSTM."),
    ("q021", 1.0, "PRED 'Yes attention' matches gold='Yes'."),
    ("q022", 1.0, "PRED 'do not specify quantifiable' matches gold unanswerable."),
    ("q023", 0.75, "PRED 'username + display + image + location + description' includes gold."),
    ("q024", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q025", 0.75, "PRED 'considerable improvement vs all baselines' matches gold."),
    ("q026", 1.0, "PRED 'Yes word segmentation + spelling errors' matches gold='Yes'."),
    ("q027", 0.25, "PRED 'maximum abs' partial; gold=tSNE plots."),
    ("q028", 0.25, "PRED 'higher or comparable' vague; gold=specific F1 scores."),
    ("q029", 1.0, "PRED '22,880 users' exact match."),
    ("q030", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q031", 0.0, "PRED 'not provide specific info' refusal; gold=PDTB taggers."),
    ("q032", 0.0, "PRED 'max turn length 160' WRONG; gold=two previous turns."),
    ("q033", 0.0, "PRED 'RMSE between scores' hallucinated; gold unanswerable."),
    ("q034", 0.5, "PRED 'KG-A2C outperform A2C' partial; gold has specific reward numbers."),
    ("q035", 0.25, "PRED 'expert annotations superior' partial."),
    ("q036", 1.0, "PRED 'Yes inspect' matches gold='Yes'."),
    ("q037", 0.75, "PRED 'EM and Macro F1' includes gold Exact Match."),
    ("q038", 0.25, "PRED 'CNN-RNN + seq2seq' partial."),
    ("q039", 0.0, "PRED 'hierarchical layers' WRONG; gold=BERT 512 + position embeddings."),
    ("q040", 1.0, "PRED 'Mainstream news and disinformation news' matches gold."),
    ("q041", 1.0, "PRED 'German-English' exact match."),
    ("q042", 0.75, "PRED 'expected unique outputs from perturbations' matches gold."),
    ("q043", 0.0, "PRED 'BERT from BIBREF12' WRONG; gold=MaxEnt + SVMs."),
    ("q044", 0.0, "PRED 'original + validation answer scores' WRONG; gold=BLEU-1."),
    ("q045", 0.0, "PRED 'Yes MTL' but gold='No' — Y/N flip."),
    ("q046", 0.75, "PRED 'do not mention' matches gold='No' implicitly."),
    ("q047", 1.0, "PRED 'over 45,000 articles + COVID-19' matches gold exactly."),
    ("q048", 0.5, "PRED 'LAN+P600 + ELAN+P600' alternate answer to gold's pairs."),
    ("q049", 1.0, "PRED 'No does not exhibit drops' matches gold='No'."),
    ("q050", 0.0, "PRED 'No multiple pairs' but gold='Yes' — Y/N flip."),
    ("q051", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q052", 0.5, "PRED 'Czech French Italian English Danish' partial (5 of 16)."),
    ("q053", 0.0, "PRED 'grammaticality + perplexity' WRONG; gold=INLINEFORM0."),
    ("q054", 0.0, "PRED 'outperformed previous methods' hallucinated; gold unanswerable."),
    ("q055", 0.0, "PRED 'lattice-free MMI' WRONG; gold=CNN-DNN-BLSTM-HMM."),
    ("q056", 0.0, "PRED 'not provide specific details' refusal; gold has specific improvement."),
    ("q057", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q058", 0.75, "PRED 'previous encoder-only RoBERTa' matches gold='RoBERTa'."),
    ("q059", 1.0, "PRED 'NO-MOVE 30.3% + 0.3%' matches gold exactly."),
    ("q060", 0.0, "PRED 'do not provide statistics' refusal; gold has specific stats."),
    ("q061", 0.5, "PRED 'demographics + diagnosis + symptoms/signs' partial match to gold's topics."),
    ("q062", 0.25, "PRED 'Existing Approaches' vague; gold=CDA."),
    ("q063", 0.5, "PRED 'transcribed text + ASR' partial; gold=text transcription only."),
    ("q064", 0.0, "PRED refuses; gold=EI-Reg + EI-Oc + V-Reg + V-Oc."),
    ("q065", 1.0, "PRED 'No combining votes + speeches' matches gold='No'."),
    ("q066", 0.5, "PRED 'limited monolingual + Hindi has less' partial."),
    ("q067", 1.0, "PRED 'do not contain info' matches gold unanswerable."),
    ("q068", 1.0, "PRED 'Yes ShapeWorld' matches gold='Yes'."),
    ("q069", 0.0, "PRED 'No no benefit' but gold='Yes' — Y/N flip."),
    ("q070", 0.5, "PRED 'text structure + typography + images' partial; gold=paragraphs."),
    ("q071", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q072", 0.0, "PRED 'Libertarianism + Anarcho + ronpaul' WRONG topics."),
    ("q073", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q074", 0.75, "PRED 'CSAT 20newsgroups Fisher' includes gold CSAT."),
    ("q075", 0.25, "PRED 'neural models' vague; gold=linear SVM."),
    ("q076", 1.0, "PRED 'not specified' matches gold unanswerable."),
    ("q077", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q078", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q079", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q080", 1.0, "PRED '0.94 + 0.94 + 0.95 + 98%' matches gold exactly."),
    ("q081", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q082", 0.0, "PRED 'F1 84-94% LSTMs' WRONG; gold lists specific SOTA tools."),
    ("q083", 0.75, "PRED 'do not mention error analysis' matches gold='No' implicitly."),
    ("q084", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q085", 0.75, "PRED 'IMDB + Yelp + imdb400 + yelp50 + yelp200' more detailed but matches."),
    ("q086", 0.75, "PRED '14% size of training' matches gold's 1000 hours concept."),
    ("q087", 0.5, "PRED 'do not specify deep learning' implicit match to gold='No'."),
    ("q088", 1.0, "PRED '45,821 characters' exact match."),
    ("q089", 0.0, "PRED refuses; gold='Yes'."),
    ("q090", 0.25, "PRED 'TAC2010 + WW + ACE + AQUAINT + CoNLL' partial; gold=CoNLL-YAGO."),
    ("q091", 0.5, "PRED 'transformation ironic to non-ironic' partial; gold=Irony Classifier."),
    ("q092", 0.0, "PRED 'Yes unsupervised LDA' but gold='No' — Y/N flip."),
    ("q093", 0.75, "PRED 'PPMI CPMI NNEGPMI' includes gold."),
    ("q094", 0.5, "PRED '145 annotators 1888 pairs' partial; gold has specific score guideline."),
    ("q095", 0.0, "PRED 'ASR + MT + ST' WRONG; gold=no specific domain."),
    ("q096", 0.0, "PRED 'Nearest Number + Random Top-3' WRONG; gold=QA PGNet."),
    ("q097", 1.0, "PRED 'No do not compare' matches gold='No'."),
    ("q098", 0.5, "PRED 'fifteen celebrities various domains' partial; gold lists specific names."),
    ("q099", 0.25, "PRED 'seven SentEval tasks' vague; gold=MR."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_main__qasper__v4-tuned__seed7__"
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
    print(f"qasper p4_main v4-tuned seed7 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
