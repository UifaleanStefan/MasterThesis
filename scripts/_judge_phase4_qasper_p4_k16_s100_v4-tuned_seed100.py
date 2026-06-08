"""Phase 4 cross-vendor finishing — QASPER p4_k16_s100 v4-tuned seed=100 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k16_s100__qasper__v4-tuned__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q001", 0.0, "PRED 'G2P system' hallucinated; gold unanswerable."),
    ("q002", 0.0, "PRED 'MLM TLM' WRONG; gold=BIBREF19."),
    ("q003", 0.0, "PRED '94/45.5%' WRONG; gold=2.6pp."),
    ("q004", 0.5, "PRED 'analyze abstracts titles + relationships' partial."),
    ("q005", 0.0, "PRED refuses; gold=4 MT tasks."),
    ("q006", 1.0, "PRED 'No experts legal expertise' matches gold='No'."),
    ("q007", 0.5, "PRED 'BIBREF22 individuals US' partial."),
    ("q008", 0.0, "PRED 'distant-supervised tweets' hallucinated."),
    ("q009", 0.0, "PRED 'split into sentences' hallucinated."),
    ("q010", 1.0, "PRED 'No' matches gold='No'."),
    ("q011", 0.0, "PRED 'speech synthesis recognition MT' WRONG; gold=Transformer."),
    ("q012", 0.0, "PRED 'neural models' vague WRONG; gold=linear SVM."),
    ("q013", 1.0, "PRED 'BIBREF9 + BIBREF8' includes gold."),
    ("q014", 0.0, "PRED refuses; gold='Yes'."),
    ("q015", 0.75, "PRED 'PPMI + CPMI + NNEGPMI' includes gold."),
    ("q016", 0.5, "PRED 'new metrics + inter-relation' partial."),
    ("q017", 0.0, "PRED 'grammaticality perplexity syntactic' WRONG; gold=INLINEFORM0."),
    ("q018", 1.0, "PRED 'No protected' matches gold='No'."),
    ("q019", 1.0, "PRED 'German-English + Ja-En + Ro-En' matches gold='De-En'."),
    ("q020", 0.0, "PRED refuses; gold=specific F1 scores."),
    ("q021", 0.0, "PRED 'Turkish' WRONG; gold=Russian."),
    ("q022", 0.25, "PRED 'seven SentEval' vague; gold=MR."),
    ("q023", 1.0, "PRED 'Yes lemmatization can hurt' matches gold='Yes'."),
    ("q024", 0.0, "PRED '4.85% + 15.09% WER' hallucinated; gold unanswerable."),
    ("q025", 0.0, "PRED refuses; gold=PDTB taggers."),
    ("q026", 1.0, "PRED 'sections 02-21 + section 23 WSJ' matches gold WSJ Penn Treebank."),
    ("q027", 1.0, "PRED 'No handcrafted rules' matches gold='No'."),
    ("q028", 0.75, "PRED 'Stanford NER + spaCy + BiLSTM' includes gold Stanford NER."),
    ("q029", 0.0, "PRED 'demonstrate by ac' fragment; gold=tSNE."),
    ("q030", 0.5, "PRED 'viral fake news URLs more' partial."),
    ("q031", 0.0, "PRED refuses; gold='Yes'."),
    ("q032", 0.5, "PRED 'fine-tuned untranslated better' partial."),
    ("q033", 0.0, "PRED 'Yes car-speak' but gold='No' — Y/N flip."),
    ("q034", 1.0, "PRED 'intentional multicast communication' matches gold exactly."),
    ("q035", 1.0, "PRED 'encoder-only RoBERTa' matches gold='RoBERTa'."),
    ("q036", 0.25, "PRED 'higher or comparable' vague."),
    ("q037", 0.0, "PRED 'full list of recommended tags' hallucinated; gold unanswerable."),
    ("q038", 0.0, "PRED 'original + validation answer references' WRONG; gold=BLEU-1."),
    ("q039", 1.0, "PRED 'do not specify quantifiable improvement' matches gold unanswerable."),
    ("q040", 0.75, "PRED 'SimpleQuestions + WebQSP' includes gold."),
    ("q041", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q042", 0.25, "PRED 'loss function + validation' partial."),
    ("q043", 0.75, "PRED 'RNN-based NMT + Transformer' matches gold."),
    ("q044", 0.75, "PRED 'F-score + Kappa' includes gold F-score."),
    ("q045", 1.0, "PRED 'NO-MOVE 30.3% + 0.3%' matches gold exactly."),
    ("q046", 0.5, "PRED 'augmentation + weighting boost low-data' partial."),
    ("q047", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q048", 0.0, "PRED 'BERT from BIBREF12' WRONG; gold=MaxEnt + SVMs."),
    ("q049", 0.25, "PRED 'two NLP tasks six datasets' vague."),
    ("q050", 0.75, "PRED '14% size' matches gold's 1000 hours."),
    ("q051", 0.75, "PRED 'CSAT + 20newsgroups + Fisher' includes gold CSAT."),
    ("q052", 0.0, "PRED refuses; gold='Yes'."),
    ("q053", 0.0, "PRED 'GRU combine function' WRONG; gold=undirected edges."),
    ("q054", 0.75, "PRED 'size in MB' matches gold's file size."),
    ("q055", 1.0, "PRED 'Yes use attention' matches gold='Yes'."),
    ("q056", 0.5, "PRED 'ineffective for different vocabulary' partial."),
    ("q057", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q058", 1.0, "PRED 'specific words user desires' matches gold."),
    ("q059", 0.25, "PRED 'subset training data' partial."),
    ("q060", 0.0, "PRED 'large-scale + qualitative' WRONG; gold=domain experts feedback."),
    ("q061", 0.0, "PRED hallucinated baselines; gold unanswerable."),
    ("q062", 0.0, "PRED refuses; gold='No'."),
    ("q063", 0.0, "PRED refuses; gold=0.8% F1."),
    ("q064", 0.25, "PRED 'lack of powerful tool' partial; gold=ambiguous words."),
    ("q065", 1.0, "PRED 'conv + NIN + Bi-LSTM' matches gold."),
    ("q066", 0.5, "PRED 'intra-sentence similarity' partial."),
    ("q067", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q068", 0.0, "PRED refuses; gold=8 NER tasks."),
    ("q069", 1.0, "PRED 'do not specify fonts' matches gold unanswerable."),
    ("q070", 1.0, "PRED 'No important words better translation' matches gold='No'."),
    ("q071", 0.25, "PRED 'simplification + explicitation + interference + translationese' partial."),
    ("q072", 0.0, "PRED 'No Google + Twitter' hallucinated; gold unanswerable."),
    ("q073", 1.0, "PRED 'Yes both English' matches gold='Yes'."),
    ("q074", 0.5, "PRED 'models capture systematic structure' partial."),
    ("q075", 0.75, "PRED '353 conversations 40 speakers' matches gold."),
    ("q076", 0.5, "PRED 'pretrained Bert document-level' partial."),
    ("q077", 1.0, "PRED 'do not specify languages' matches gold unanswerable."),
    ("q078", 0.0, "PRED 'evaluators score +1' hallucinated; gold unanswerable."),
    ("q079", 1.0, "PRED 'EI-Reg + EI-Oc + V-Reg + V-Oc' matches gold."),
    ("q080", 0.0, "PRED 'pre-ordering rules' hallucinated; gold unanswerable."),
    ("q081", 0.75, "PRED 'PTB + WT-2' includes gold Penn Treebank."),
    ("q082", 0.0, "PRED '3rd position F1 0.673' WRONG; gold=specific team names."),
    ("q083", 1.0, "PRED 'UD1.2 16 languages full list' matches gold."),
    ("q084", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q085", 1.0, "PRED 'No combining votes + speeches' matches gold='No'."),
    ("q086", 0.0, "PRED 'dictionary method' WRONG; gold=word2vec + SVM."),
    ("q087", 0.0, "PRED 'Yes SQuAD 1.1 English' hallucinated; gold unanswerable."),
    ("q088", 0.0, "PRED 'No U.S. Senate + ArXiv' hallucinated; gold unanswerable."),
    ("q089", 0.0, "PRED 'collection of tweets' vague; gold=WASSA-2017."),
    ("q090", 0.75, "PRED 'UMA + MRR + BPE + BLEU-1/4 + ROUGE-L' matches most gold metrics."),
    ("q091", 0.75, "PRED 'OpenIE toolbox derive relations' matches gold."),
    ("q092", 0.25, "PRED 'significant improvements' incomplete; gold=7.36 + 9.69."),
    ("q093", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q094", 1.0, "PRED 'not explicitly mentioned' matches gold unanswerable."),
    ("q095", 0.75, "PRED 'CDA and REG' includes gold CDA."),
    ("q096", 0.5, "PRED 'distribution of focus + dependency roles' partial."),
    ("q097", 1.0, "PRED 'Yes English ISIS magazines' matches gold='Yes'."),
    ("q098", 0.0, "PRED 'trained on parallel data' vague; gold=attentional encoder-decoder."),
    ("q099", 0.25, "PRED 'IMS+emb BIBREF9' partial reference."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_k16_s100__qasper__v4-tuned__seed100__"
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
    print(f"qasper p4_k16_s100 v4-tuned seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
