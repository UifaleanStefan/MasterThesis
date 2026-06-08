"""Phase 4 cross-vendor finishing — QASPER p4_main v4-tuned seed=100 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__v4-tuned__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q001", 0.0, "PRED 'transcribed via G2P' hallucinated; gold unanswerable."),
    ("q002", 0.0, "PRED 'MLM + TLM' WRONG; gold=BIBREF19."),
    ("q003", 0.0, "PRED '94/45.5%' WRONG numbers; gold=2.6 pp."),
    ("q004", 0.75, "PRED 'LDA + Gibbs on abstracts/titles' matches gold's LDA approaches."),
    ("q005", 0.0, "PRED refuses; gold=4 MT tasks."),
    ("q006", 1.0, "PRED 'No experts not comparable + trained legal' matches gold='No'."),
    ("q007", 0.5, "PRED 'individuals located in US on BIBREF22' partial."),
    ("q008", 0.0, "PRED 'distant-supervised tweets' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'split into sentences' hallucinated; gold unanswerable."),
    ("q010", 1.0, "PRED 'No experiments not specific' matches gold='No'."),
    ("q011", 0.0, "PRED 'speech synthesis + recognition + MT' WRONG; gold=Transformer."),
    ("q012", 0.25, "PRED 'neural models' vague; gold=linear SVM."),
    ("q013", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q014", 0.0, "PRED refuses; gold='Yes'."),
    ("q015", 0.75, "PRED includes CPMI and NNEGPMI from gold."),
    ("q016", 0.25, "PRED 'new metrics inter-relation' vague."),
    ("q017", 0.0, "PRED 'grammaticality + perplexity' WRONG; gold=INLINEFORM0."),
    ("q018", 1.0, "PRED 'No PHI access' matches gold='No'."),
    ("q019", 1.0, "PRED 'German-English (De-En) Ja-En Ro-En' matches gold='De-En'."),
    ("q020", 0.0, "PRED refuses; gold=specific F1 scores."),
    ("q021", 0.0, "PRED 'Turkish 72.71%' WRONG; gold=Russian."),
    ("q022", 0.25, "PRED 'seven SentEval tasks' vague; gold=MR."),
    ("q023", 1.0, "PRED 'Yes lemmatization hurts atypical' matches gold='Yes'."),
    ("q024", 0.0, "PRED '4.85% VLSP2018 + 15.09% VLSP2019' hallucinated; gold unanswerable."),
    ("q025", 0.0, "PRED refuses; gold=PDTB taggers."),
    ("q026", 0.75, "PRED 'WSJ sections 02-21' matches gold WSJ Penn Treebank."),
    ("q027", 1.0, "PRED 'No handcrafted rules' matches gold='No'."),
    ("q028", 0.75, "PRED includes gold Stanford NER among others."),
    ("q029", 0.25, "PRED 'maximum acc' partial; gold=tSNE plots."),
    ("q030", 0.5, "PRED 'viral tweets URLs' partial; gold lists multiple categories."),
    ("q031", 0.0, "PRED refuses; gold='Yes'."),
    ("q032", 0.25, "PRED 'fine-tuned on un-translated' partial."),
    ("q033", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q034", 1.0, "PRED 'intentional multicast communication' matches gold exactly."),
    ("q035", 0.75, "PRED 'RoBERTa with language modeling' matches gold='RoBERTa'."),
    ("q036", 0.25, "PRED 'higher or comparable' vague; gold=specific F1 numbers."),
    ("q037", 0.0, "PRED 'full list of recommended tags' hallucinated; gold unanswerable."),
    ("q038", 0.0, "PRED 'original answer + validation answer scores' WRONG; gold=BLEU-1."),
    ("q039", 1.0, "PRED 'do not specify quantifiable' matches gold unanswerable."),
    ("q040", 0.75, "PRED 'SimpleQuestions and WebQSP' includes gold SimpleQuestions."),
    ("q041", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q042", 0.0, "PRED 'loss function + validation' WRONG; gold=MAP."),
    ("q043", 0.75, "PRED 'RNN-based NMT and Transformer' includes gold."),
    ("q044", 0.75, "PRED 'F-score and Kappa' includes gold F-score."),
    ("q045", 1.0, "PRED 'NO-MOVE 30.3% + 0.3%' matches gold exactly."),
    ("q046", 0.25, "PRED 'augmentation and weighting boost' vague."),
    ("q047", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q048", 0.0, "PRED 'BERT from BIBREF12' WRONG; gold=MaxEnt + SVMs."),
    ("q049", 0.25, "PRED 'two NLP tasks six datasets' vague."),
    ("q050", 0.75, "PRED '14% size' matches gold concept of 1000 hours."),
    ("q051", 0.75, "PRED 'CSAT 20newsgroups Fisher' includes gold CSAT."),
    ("q052", 0.0, "PRED refuses; gold='Yes'."),
    ("q053", 0.0, "PRED 'GRU combine function' WRONG; gold=undirected edges."),
    ("q054", 0.75, "PRED 'LangID-High 15.4 MB' matches gold's file size concept."),
    ("q055", 1.0, "PRED 'Yes attention' matches gold='Yes'."),
    ("q056", 0.5, "PRED 'reduce vocabularies + same vocab' partial."),
    ("q057", 0.0, "PRED 'Yes subject to quality control' but gold='No' — Y/N flip."),
    ("q058", 0.5, "PRED '{subway, manhattan} examples' partial."),
    ("q059", 0.0, "PRED refuses; gold=reducing variance of estimator."),
    ("q060", 0.0, "PRED 'large + qualitative' WRONG; gold=domain experts feedback."),
    ("q061", 0.0, "PRED 'Jasper baselines on Hub5'00' hallucinated; gold unanswerable."),
    ("q062", 0.0, "PRED refuses; gold='No'."),
    ("q063", 0.0, "PRED refuses; gold=0.8% F1."),
    ("q064", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q065", 0.75, "PRED 'CNN + NIN + LSTM with global attention' matches gold."),
    ("q066", 0.75, "PRED 'measuring intra-sentence similarity' matches part of gold."),
    ("q067", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q068", 0.0, "PRED 'specific tasks not listed' refusal; gold=8 NER tasks."),
    ("q069", 1.0, "PRED 'do not specify fonts' matches gold unanswerable."),
    ("q070", 1.0, "PRED 'No does not exhibit drops' matches gold='No'."),
    ("q071", 0.25, "PRED 'translationese + annotation artifacts' partial."),
    ("q072", 0.0, "PRED 'No multiple datasets' hallucinated; gold unanswerable."),
    ("q073", 1.0, "PRED 'Yes English only' matches gold='Yes'."),
    ("q074", 0.0, "PRED 'not directly generalizable' but gold='Yes'."),
    ("q075", 0.75, "PRED '353 conversations 41 hours' matches gold's 353 conversations."),
    ("q076", 0.25, "PRED 'hierarchical lower/higher layers' partial."),
    ("q077", 1.0, "PRED 'do not specify languages' matches gold unanswerable."),
    ("q078", 0.0, "PRED 'RMSE between scores' hallucinated; gold unanswerable."),
    ("q079", 0.5, "PRED 'specific subtasks not mentioned' refers to gold-like answer."),
    ("q080", 0.0, "PRED 'Hindi-tuned rules' hallucinated; gold unanswerable."),
    ("q081", 0.75, "PRED 'PTB + WT-2' includes gold Penn Treebank."),
    ("q082", 0.0, "PRED '3rd position F1 0.673' WRONG; gold=specific team names."),
    ("q083", 0.5, "PRED 'Czech French Italian English Danish' partial (5 of 16)."),
    ("q084", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q085", 1.0, "PRED 'No combining votes and speeches' matches gold='No'."),
    ("q086", 0.0, "PRED 'self score' WRONG; gold=word2vec + SVM."),
    ("q087", 0.0, "PRED 'Yes English' hallucinated; gold unanswerable."),
    ("q088", 0.0, "PRED 'Yes English' hallucinated; gold unanswerable."),
    ("q089", 0.25, "PRED 'dev data set BIBREF19' partial reference."),
    ("q090", 0.0, "PRED 'pairwise comparisons' WRONG; gold=long list of metrics."),
    ("q091", 1.0, "PRED 'OpenIE toolbox + select' matches gold."),
    ("q092", 0.0, "PRED '12.27 + 14.86' WRONG; gold=7.36 + 9.69."),
    ("q093", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q094", 1.0, "PRED 'not specified' matches gold unanswerable."),
    ("q095", 0.25, "PRED 'Existing Approaches' vague; gold=CDA."),
    ("q096", 0.5, "PRED 'alignment + differences' partial."),
    ("q097", 0.0, "PRED 'No ISIS + religious' but gold='Yes'."),
    ("q098", 0.0, "PRED 'not explicitly mentioned'; gold=attentional encoder-decoder."),
    ("q099", 0.25, "PRED 'IMS+emb BIBREF9' partial reference."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_main__qasper__v4-tuned__seed100__"
    existing: set[str] = set()
    if RESULTS.exists():
        for line in RESULTS.read_text(encoding="utf-8").splitlines():
            if line.strip():
                try: existing.add(json.loads(line)["qid"])
                except: pass
    added = 0; total = 0.0
    RESULTS.parent.mkdir(parents=True, exist_ok=True)
    with RESULTS.open("a", encoding="utf-8") as f:
        for suffix, score, rationale in JUDGMENTS:
            qid = qid_prefix + suffix
            if qid in existing: continue
            f.write(json.dumps({"qid": qid, "judge_score": score, "rationale": rationale,
                                "judge_model": JUDGE_MODEL, "judge_protocol": JUDGE_PROTOCOL}, ensure_ascii=False) + "\n")
            added += 1; total += score
    print(f"qasper p4_main v4-tuned seed100 added={added} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
