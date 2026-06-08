"""Phase 4 cross-vendor finishing — QASPER p4_k16 v4-canonical seed=100 (100 entries)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k16__qasper__v4-canonical__seed100/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q001", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q002", 0.0, "PRED 'MLM, TLM, BRLM' WRONG; gold=BIBREF19."),
    ("q003", 0.0, "PRED '94/45.5%' WRONG; gold=2.6 pp."),
    ("q004", 0.75, "PRED 'LDA + relationships + Gibbs sampling' matches gold's LDA approaches."),
    ("q005", 0.0, "PRED refuses; gold=4 MT tasks."),
    ("q006", 0.25, "PRED 'no direct comparison' partial."),
    ("q007", 0.0, "PRED 'analyzing tweets' WRONG; gold=Mechanical Turk."),
    ("q008", 0.0, "PRED 'book corpus' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'split into sentences' hallucinated."),
    ("q010", 1.0, "PRED 'No covers variety topics genres' matches gold='No'."),
    ("q011", 0.0, "PRED 'Mapudungun Spanish' WRONG; gold=Transformer."),
    ("q012", 0.75, "PRED 'linear SVM + BiLSTM + CNN' includes gold linear SVM."),
    ("q013", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q014", 0.0, "PRED refuses; gold='Yes'."),
    ("q015", 0.75, "PRED 'NPMI + NNEGPMI + CPMI' includes gold NNEGPMI."),
    ("q016", 0.25, "PRED 'variety of generated keyphrases' partial."),
    ("q017", 0.0, "PRED 'grammaticality + perplexity' WRONG; gold=INLINEFORM0."),
    ("q018", 0.0, "PRED refuses; gold='No'."),
    ("q019", 1.0, "PRED 'German-English' matches gold='De-En'."),
    ("q020", 0.0, "PRED refuses; gold=specific F1 scores."),
    ("q021", 0.0, "PRED 'Turkish 72.71%' WRONG; gold=Russian."),
    ("q022", 0.0, "PRED 'SNLI Multi-NLI' WRONG; gold=MR."),
    ("q023", 1.0, "PRED 'Yes lemmatization can hurt' matches gold='Yes'."),
    ("q024", 0.0, "PRED '15.47% VLSP 2019' hallucinated; gold unanswerable."),
    ("q025", 0.0, "PRED 'linear + RNN-based' WRONG; gold=PDTB taggers."),
    ("q026", 1.0, "PRED 'WSJ' matches gold Wall Street Journal."),
    ("q027", 0.5, "PRED 'do not mention' implicit match to gold='No'."),
    ("q028", 0.75, "PRED 'recurrent CRF + spaCy + Stanford' includes Stanford NER."),
    ("q029", 0.0, "PRED 'demonstrate by' fragment; gold=tSNE."),
    ("q030", 0.25, "PRED 'viral tweets mentions less' partial; gold has multiple categories."),
    ("q031", 0.0, "PRED refuses; gold='Yes'."),
    ("q032", 0.5, "PRED 'no improvements linear mapping devastating' partial."),
    ("q033", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q034", 0.75, "PRED 'social phenomenon act of communication' matches gold concept."),
    ("q035", 1.0, "PRED 'previous SOTA based on RoBERTa' matches gold='RoBERTa'."),
    ("q036", 0.25, "PRED 'higher or comparable' vague."),
    ("q037", 1.0, "PRED 'specific number not provided' matches gold unanswerable."),
    ("q038", 0.75, "PRED 'METEOR + ROUGE-L + BLEU-1' includes gold BLEU-1."),
    ("q039", 1.0, "PRED 'do not specify exact improvement' matches gold unanswerable."),
    ("q040", 0.75, "PRED 'SimpleQuestions and WebQSP' includes gold."),
    ("q041", 0.0, "PRED 'No in-domain + cross-domain' but gold='Yes' — Y/N flip."),
    ("q042", 0.25, "PRED 'loss function + validation perf' partial."),
    ("q043", 0.75, "PRED 'RNN-based NMT + Transformer with SMT' matches gold."),
    ("q044", 0.0, "PRED 'Chi squared + Info Gain' WRONG; gold=F-score."),
    ("q045", 1.0, "PRED 'NO-MOVE 30.3% + 0.3%' matches gold exactly."),
    ("q046", 1.0, "PRED '6 acc + over 20 acc' matches gold's specific imbalance figures."),
    ("q047", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q048", 0.0, "PRED refuses; gold=MaxEnt + SVMs."),
    ("q049", 0.25, "PRED 'two NLP tasks six datasets' vague."),
    ("q050", 1.0, "PRED '14% data 1000 hours' matches gold's 1000 hours."),
    ("q051", 0.75, "PRED 'CSAT + Fisher + 20newsgroups' includes gold CSAT."),
    ("q052", 0.0, "PRED refuses; gold='Yes'."),
    ("q053", 0.0, "PRED 'update on neighbors' WRONG; gold=directed→undirected edges."),
    ("q054", 0.75, "PRED 'size in MB' matches gold's file size concept."),
    ("q055", 0.0, "PRED refuses; gold='Yes'."),
    ("q056", 0.5, "PRED 'ineffective for different vocabulary' partial."),
    ("q057", 0.0, "PRED 'Yes' but gold='No' — Y/N flip."),
    ("q058", 0.75, "PRED 'subway manhattan user-specified' matches gold concept."),
    ("q059", 0.0, "PRED refuses; gold=reducing variance of estimator."),
    ("q060", 0.0, "PRED 'large + qualitative' WRONG; gold=domain experts feedback."),
    ("q061", 0.0, "PRED 'Jasper Hub5' hallucinated; gold unanswerable."),
    ("q062", 0.0, "PRED refuses; gold='No'."),
    ("q063", 0.25, "PRED 'F1 78.6 vs 78.9 not significantly outperform' partial."),
    ("q064", 0.75, "PRED 'ambiguous + unknown words' matches gold's ambiguous words."),
    ("q065", 1.0, "PRED 'four conv layers + two NIN + 3 Bi-LSTM 256 hidden' matches gold detailed."),
    ("q066", 0.5, "PRED 'intra-sentence similarity ELMo BERT' partial."),
    ("q067", 1.0, "PRED 'Disinformation and mainstream news' matches gold."),
    ("q068", 0.0, "PRED refuses; gold=8 specific NER tasks."),
    ("q069", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q070", 0.0, "PRED refuses; gold='No'."),
    ("q071", 0.0, "PRED 'Translation artifacts' WRONG; gold=lexical overlap."),
    ("q072", 0.0, "PRED 'No multiple datasets' hallucinated; gold unanswerable."),
    ("q073", 1.0, "PRED 'Yes hashtag + SemEval English' matches gold='Yes'."),
    ("q074", 0.25, "PRED 'highly dependent on random seed' partial; gold='Yes'."),
    ("q075", 0.0, "PRED 'Base/Augmented/Real-World sets' WRONG; gold=353 conversations 40 speakers."),
    ("q076", 0.0, "PRED 'separate optimizers' WRONG; gold=BERT 512+position embeddings."),
    ("q077", 1.0, "PRED 'do not specify languages' matches gold unanswerable."),
    ("q078", 1.0, "PRED 'do not provide details' matches gold unanswerable."),
    ("q079", 1.0, "PRED 'EI-Reg + EI-Oc + V-Reg + V-Oc' matches gold."),
    ("q080", 0.0, "PRED 'bilingual dictionary Google Translate' hallucinated."),
    ("q081", 0.75, "PRED 'PTB + WT-2' includes gold Penn Treebank."),
    ("q082", 0.0, "PRED '3rd position ensemble 6 models' WRONG; gold=specific team names."),
    ("q083", 1.0, "PRED 'UD1.2 corpora 16 languages: full list' matches gold."),
    ("q084", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q085", 0.0, "PRED 'Yes votes' but gold='No' — Y/N flip."),
    ("q086", 0.75, "PRED 'word2vec approach' matches gold's word2vec + SVM."),
    ("q087", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q088", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q089", 0.25, "PRED 'dev data + train+dev merged' partial WASSA reference."),
    ("q090", 0.5, "PRED 'UMA + MRR + coherence + BLEU' partial includes some gold metrics."),
    ("q091", 0.25, "PRED 'identifying relationships answer surrounding' partial."),
    ("q092", 0.0, "PRED '12.27 + 14.86' WRONG; gold=7.36 + 9.69."),
    ("q093", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q094", 1.0, "PRED 'not specified' matches gold unanswerable."),
    ("q095", 0.75, "PRED 'CDA and REG' includes gold CDA."),
    ("q096", 0.5, "PRED 'focus on relevant parts source for translating words' partial."),
    ("q097", 0.0, "PRED 'No' but gold='Yes' — Y/N flip."),
    ("q098", 0.0, "PRED 'parallel data' vague; gold=attentional encoder-decoder."),
    ("q099", 0.25, "PRED 'GlossBERT performs best' partial."),
]


def main() -> None:
    assert len(JUDGMENTS) == 100
    qid_prefix = "p4_k16__qasper__v4-canonical__seed100__"
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
    print(f"qasper p4_k16 v4-canonical seed100 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
