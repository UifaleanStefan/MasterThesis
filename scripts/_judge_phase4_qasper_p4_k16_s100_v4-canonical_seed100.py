"""Phase 4 cross-vendor finishing — QASPER p4_k16_s100 v4-canonical seed=100 (100 entries).

Note: predictions in this cell are essentially identical to p4_main v4-canonical
seed=100 because k=16 retrieval pulled the same top documents on QASPER. Scoring
is therefore identical to that template.
"""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k16_s100__qasper__v4-canonical__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q001", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q002", 0.0, "PRED 'MLM+BRLM-SA + cross-lingual' WRONG; gold=BIBREF19."),
    ("q003", 0.0, "PRED '94/45.5%' WRONG; gold=2.6 pp."),
    ("q004", 0.5, "PRED mentions LDA + Gibbs sampling — partial key concept."),
    ("q005", 0.0, "PRED refuses; gold=4 MT tasks."),
    ("q006", 1.0, "PRED 'No experts not comparable' matches gold='No'."),
    ("q007", 0.25, "PRED 'individuals analyze tweets' wrong context; gold=Mechanical Turk."),
    ("q008", 0.0, "PRED 'PPDB' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'split into sentences' hallucinated."),
    ("q010", 0.0, "PRED refuses; gold='No'."),
    ("q011", 0.0, "PRED 'hybrid rule SMT' WRONG; gold=Transformer."),
    ("q012", 0.0, "PRED 'CNN BiLSTM' WRONG; gold=linear SVM."),
    ("q013", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q014", 0.0, "PRED refuses; gold='Yes'."),
    ("q015", 0.75, "PRED includes gold CPMI and NNEGPMI variants."),
    ("q016", 0.25, "PRED 'recall on hundreds' partial."),
    ("q017", 0.0, "PRED 'grammaticality + perplexity' WRONG; gold=INLINEFORM0."),
    ("q018", 0.0, "PRED refuses; gold='No'."),
    ("q019", 1.0, "PRED 'German-English' matches gold='De-En'."),
    ("q020", 0.0, "PRED refuses; gold=specific F1 scores."),
    ("q021", 0.0, "PRED 'Turkish' WRONG; gold=Russian."),
    ("q022", 0.0, "PRED refuses; gold='MR'."),
    ("q023", 1.0, "PRED 'Yes lemmatization can hurt' matches gold='Yes'."),
    ("q024", 0.0, "PRED hallucinated 15.47 WER when gold unanswerable."),
    ("q025", 0.0, "PRED 'CEI Only + pipeline' wrong baselines."),
    ("q026", 0.75, "PRED 'WSJ' matches gold's WSJ Penn Treebank."),
    ("q027", 1.0, "PRED 'No rule-based' matches gold='No'."),
    ("q028", 0.75, "PRED includes gold Stanford NER among others."),
    ("q029", 0.0, "PRED 'higher kappa' WRONG; gold=tSNE plots."),
    ("q030", 0.5, "PRED partial; gold lists multiple categories."),
    ("q031", 0.0, "PRED refuses; gold='Yes'."),
    ("q032", 0.5, "PRED 'no improvements + devastating effects' partial."),
    ("q033", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q034", 0.5, "PRED 'social phenomenon act of communication' partial."),
    ("q035", 1.0, "PRED 'RoBERTa' exact match."),
    ("q036", 0.25, "PRED 'higher or comparable' vague."),
    ("q037", 1.0, "PRED 'INLINEFORM0 not specified' matches gold unanswerable."),
    ("q038", 0.75, "PRED includes gold BLEU-1 among others."),
    ("q039", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q040", 0.0, "PRED 'WebQSP' WRONG; gold=SimpleQuestions."),
    ("q041", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q042", 0.0, "PRED 'loss function + validation' WRONG; gold=MAP."),
    ("q043", 0.75, "PRED includes gold 'RNN-based NMT'."),
    ("q044", 0.0, "PRED 'Chi squared + Info Gain' WRONG; gold=F-score."),
    ("q045", 1.0, "PRED 'NO-MOVE 30.3% + 0.3' matches gold exactly."),
    ("q046", 1.0, "PRED '6 acc on 100:1000 + over 20 on 20:1000' matches gold."),
    ("q047", 0.0, "PRED hallucinated 'Yes English only' when gold unanswerable."),
    ("q048", 0.0, "PRED refuses; gold=Maximum Entropy + SVMs."),
    ("q049", 0.25, "PRED 'two NLP tasks six datasets' vague."),
    ("q050", 1.0, "PRED '14% ~1000 hours' matches gold='1000 hours'."),
    ("q051", 0.75, "PRED includes gold CSAT among others."),
    ("q052", 0.0, "PRED refuses; gold='Yes'."),
    ("q053", 0.0, "PRED 'BIBREF53 approach' WRONG; gold=undirected edges."),
    ("q054", 0.75, "PRED 'size in MB' matches gold's file size."),
    ("q055", 1.0, "PRED 'Yes attention' matches gold='Yes'."),
    ("q056", 0.5, "PRED 'reducing vocabularies can push' partial."),
    ("q057", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q058", 0.0, "PRED refuses; gold=words user wants."),
    ("q059", 0.0, "PRED refuses; gold=reducing variance of estimator."),
    ("q060", 0.0, "PRED 'large-scale + qualitative' WRONG; gold=domain experts feedback."),
    ("q061", 0.0, "PRED hallucinated baselines when gold unanswerable."),
    ("q062", 0.0, "PRED refuses; gold='No'."),
    ("q063", 0.0, "PRED refuses; gold=0.8% F1."),
    ("q064", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q065", 0.0, "PRED 'Attention Encoder-Decoder' WRONG; gold=CNN+NIN+BLSTM."),
    ("q066", 0.25, "PRED 'analyzing GPT-2 word reps' vague."),
    ("q067", 1.0, "PRED 'mainstream and disinformation' matches gold exactly."),
    ("q068", 0.0, "PRED refuses; gold=8 specific NER tasks."),
    ("q069", 1.0, "PRED 'do not mention' matches gold unanswerable."),
    ("q070", 0.0, "PRED refuses; gold='No'."),
    ("q071", 0.0, "PRED refuses; gold=lexical overlap."),
    ("q072", 0.0, "PRED 'No multiple datasets' hallucinated when gold unanswerable."),
    ("q073", 0.75, "PRED 'primarily English' implies Yes."),
    ("q074", 0.0, "PRED 'may not be generalized' but gold='Yes'."),
    ("q075", 0.0, "PRED '100k' WRONG; gold=353 conversations 40 speakers."),
    ("q076", 0.0, "PRED 'separate optimizers' WRONG; gold=BERT 512 max."),
    ("q077", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q078", 1.0, "PRED 'does not provide' matches gold unanswerable."),
    ("q079", 1.0, "PRED 'EI-Reg EI-Oc V-Reg V-Oc' matches gold exactly."),
    ("q080", 0.0, "PRED hallucinated Google Translate when gold unanswerable."),
    ("q081", 0.75, "PRED 'PTB + WT-2' includes gold Penn Treebank."),
    ("q082", 0.0, "PRED '3rd position ensemble' WRONG; gold=specific team names."),
    ("q083", 0.5, "PRED 'Czech French Italian Indonesian' partial (4 of 16)."),
    ("q084", 1.0, "PRED 'do not mention' matches gold unanswerable."),
    ("q085", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q086", 0.75, "PRED 'word2vec approach' matches gold's word2vec + SVM."),
    ("q087", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q088", 0.0, "PRED 'Yes English only' hallucinated when gold unanswerable."),
    ("q089", 1.0, "PRED 'WASSA-2017 Shared Task' matches gold exactly."),
    ("q090", 0.0, "PRED 'pairwise comparisons' WRONG; gold=long list of metrics."),
    ("q091", 0.25, "PRED 'identifying answer-relevant' partial; gold=OpenIE+heuristics."),
    ("q092", 0.0, "PRED '12.27/14.86' WRONG; gold=7.36/9.69."),
    ("q093", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q094", 1.0, "PRED 'not specified' matches gold unanswerable."),
    ("q095", 0.75, "PRED 'CDA + proposed method' includes gold CDA."),
    ("q096", 0.75, "PRED 'focusing on other parts for verbs' matches gold concept."),
    ("q097", 0.0, "PRED refuses; gold='Yes'."),
    ("q098", 0.0, "PRED refuses; gold=attentional encoder-decoder."),
    ("q099", 0.25, "PRED 'GlossBERT' partial; gold=multiple system categories."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_k16_s100__qasper__v4-canonical__seed100__"
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
    print(f"qasper p4_k16_s100 v4-canonical seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
