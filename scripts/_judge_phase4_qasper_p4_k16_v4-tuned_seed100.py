"""Phase 4 cross-vendor finishing — QASPER p4_k16 v4-tuned seed=100 (99 remaining; 1 already judged)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k16__qasper__v4-tuned__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q001", 0.0, "PRED 'G2P system' hallucinated; gold unanswerable."),
    ("q002", 0.0, "PRED 'pivoting + cross-lingual transfer' WRONG; gold=BIBREF19."),
    ("q003", 0.25, "PRED '93.4% vs 90.0%' wrong magnitude; gold=2.6pp."),
    ("q004", 0.5, "PRED 'Gibbs sampling + 100 topics + LDA features' partial."),
    ("q005", 1.0, "PRED 'ro→en + en→de + de→en + ja→en' matches gold's 4 MT tasks."),
    ("q006", 1.0, "PRED 'No experts not comparable + legally-informed' matches gold='No'."),
    ("q007", 0.5, "PRED 'individuals in US BIBREF22' partial."),
    ("q008", 0.0, "PRED 'distant-supervised tweets' hallucinated."),
    ("q009", 0.0, "PRED 'split into sentences' hallucinated."),
    ("q010", 1.0, "PRED 'No different genres various topics' matches gold='No'."),
    ("q011", 0.0, "PRED 'speech synthesis recognition MT' WRONG; gold=Transformer."),
    ("q012", 0.0, "PRED 'CNN BiLSTM' WRONG; gold=linear SVM."),
    ("q013", 0.0, "PRED 'methods lack contextual' WRONG; gold=BIBREF9."),
    ("q014", 0.0, "PRED refuses; gold='Yes'."),
    ("q015", 1.0, "PRED 'CPMI + NNEGPMI' matches gold."),
    ("q016", 0.5, "PRED 'multiple keyphrases different aspects' partial."),
    ("q017", 0.0, "PRED 'grammaticality + perplexity + F1 length' WRONG; gold=INLINEFORM0."),
    ("q018", 1.0, "PRED 'No MIMIC-III' matches gold='No'."),
    ("q019", 1.0, "PRED 'German-English + Ja-En + Ro-En' matches gold='De-En'."),
    ("q020", 0.25, "PRED '+1.08 +1.24 +2.38 vs LSTM baseline' partial; gold has F1 absolute scores."),
    ("q022", 0.25, "PRED 'seven SentEval tasks' vague; gold=MR."),
    ("q023", 1.0, "PRED 'Yes lemmatization can hurt' matches gold='Yes'."),
    ("q024", 0.0, "PRED '4.85% + 15.09% WER' hallucinated; gold unanswerable."),
    ("q025", 1.0, "PRED 'state-of-the-art PDTB taggers' exact match."),
    ("q026", 1.0, "PRED 'WSJ section 23' matches gold WSJ Penn Treebank."),
    ("q027", 1.0, "PRED 'No rule-based + custom' matches gold='No'."),
    ("q028", 0.75, "PRED 'Stanford NER + spaCy + biLSTM' includes Stanford NER."),
    ("q029", 0.0, "PRED 'demonstrate by' fragment; gold=tSNE."),
    ("q030", 0.5, "PRED 'viral tweets URLs more' partial; gold lists multiple categories."),
    ("q031", 0.0, "PRED refuses; gold='Yes'."),
    ("q032", 0.5, "PRED 'fine-tuned untranslated better' partial."),
    ("q033", 0.0, "PRED 'Yes car-speak' but gold='No' — Y/N flip."),
    ("q034", 0.75, "PRED 'social phenomenon act of communication' matches gold concept."),
    ("q035", 1.0, "PRED 'previous SOTA based on RoBERTa' matches gold='RoBERTa'."),
    ("q036", 0.25, "PRED 'higher or comparable' vague."),
    ("q037", 1.0, "PRED 'INLINEFORM0 not specified' matches gold unanswerable."),
    ("q038", 0.75, "PRED 'BLEU-1 + Meteor + Rouge-L' includes gold BLEU-1."),
    ("q039", 1.0, "PRED 'do not provide quantitative' matches gold unanswerable."),
    ("q040", 0.75, "PRED 'SimpleQuestions and WebQSP' includes gold."),
    ("q041", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q042", 0.25, "PRED 'loss function + validation perf' partial."),
    ("q043", 0.75, "PRED 'RNN-based NMT + Transformer-based NMT' matches gold."),
    ("q044", 0.75, "PRED 'F-score + Kappa statistics' includes gold F-score."),
    ("q045", 1.0, "PRED 'NO-MOVE 30.3% + 0.3' matches gold exactly."),
    ("q046", 0.5, "PRED 'augmentation > weighting + data weighting improves' partial."),
    ("q047", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q048", 0.0, "PRED 'BERT from BIBREF12' WRONG; gold=MaxEnt + SVMs."),
    ("q049", 0.25, "PRED 'two NLP tasks six datasets' vague."),
    ("q050", 0.75, "PRED '14% size' matches gold's 1000 hours concept."),
    ("q051", 0.75, "PRED 'CSAT + 20newsgroups + Fisher' includes gold CSAT."),
    ("q052", 0.0, "PRED refuses; gold='Yes'."),
    ("q053", 0.0, "PRED 'GRU combine function identity' WRONG; gold=directed→undirected edges."),
    ("q054", 0.75, "PRED 'size in MB' matches gold's file size concept."),
    ("q055", 1.0, "PRED 'Yes use attention' matches gold='Yes'."),
    ("q056", 0.5, "PRED 'ineffective for different vocabulary' partial."),
    ("q057", 0.0, "PRED 'Yes harder to validate + biases' but gold='No' — Y/N flip."),
    ("q058", 1.0, "PRED 'specific words user desires + subway + manhattan' matches gold concept."),
    ("q059", 0.25, "PRED 'subset training data + another distribution' partial."),
    ("q060", 0.0, "PRED 'large + qualitative' WRONG; gold=domain experts feedback."),
    ("q061", 0.0, "PRED 'Hub5\\'00 Table TABREF31' hallucinated; gold unanswerable."),
    ("q062", 0.0, "PRED refuses; gold='No'."),
    ("q063", 0.0, "PRED 'do not specify exact amount' refusal; gold=0.8% F1."),
    ("q064", 0.25, "PRED 'lack of powerful Vietnamese tool' partial; gold=ambiguous words."),
    ("q065", 1.0, "PRED 'conv + NIN + deep bidirectional LSTM' matches gold."),
    ("q066", 0.5, "PRED 'intra-sentence similarity cosine each word' partial."),
    ("q067", 1.0, "PRED 'Mainstream and disinformation news' matches gold."),
    ("q068", 0.0, "PRED refuses; gold=8 specific NER tasks."),
    ("q069", 1.0, "PRED 'do not specify fonts' matches gold unanswerable."),
    ("q070", 1.0, "PRED 'No important words better translation' matches gold='No'."),
    ("q071", 0.25, "PRED 'simplification + explicitation + interference + translationese' partial."),
    ("q072", 0.0, "PRED 'No Google + Twitter' hallucinated; gold unanswerable."),
    ("q073", 1.0, "PRED 'Yes both English' matches gold='Yes'."),
    ("q074", 0.5, "PRED 'models capture systematic structure + variability' partial."),
    ("q075", 0.75, "PRED '353 conversations 40 speakers' matches gold (no group breakdown)."),
    ("q076", 0.5, "PRED 'pretrained Bert document-level' partial."),
    ("q077", 1.0, "PRED 'do not specify languages' matches gold unanswerable."),
    ("q078", 0.0, "PRED 'evaluators score +1' hallucinated; gold unanswerable."),
    ("q079", 1.0, "PRED 'EI-Reg + EI-Oc + V-Reg + V-Oc' matches gold."),
    ("q080", 0.0, "PRED 'pre-ordering rules noun phrases' hallucinated; gold unanswerable."),
    ("q081", 0.75, "PRED 'PTB + WT-2' includes gold Penn Treebank."),
    ("q082", 0.0, "PRED '3rd position F1 0.673' WRONG; gold=specific team names."),
    ("q083", 1.0, "PRED 'UD1.2 corpora 16 languages full list' matches gold."),
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
    assert len(JUDGMENTS) == 99
    qid_prefix = "p4_k16__qasper__v4-tuned__seed100__"
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
    print(f"qasper p4_k16 v4-tuned seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
