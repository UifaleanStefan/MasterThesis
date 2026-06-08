"""Phase 4 cross-vendor finishing — QASPER p4_k16 flat-50 seed=42 (66 remaining entries, 34 already judged)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_tierb__qasper__flat-50__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q001", 0.0, "PRED 'manually detect stereotypes + POS tagging' hallucinated; gold unanswerable."),
    ("q003", 0.25, "PRED 'five datasets CoNLL testa TAC2010' partial; gold=CoNLL-YAGO."),
    ("q005", 1.0, "PRED 'do not mention applicability' matches gold unanswerable."),
    ("q006", 0.0, "PRED 'outperformed previous AMR' hallucinated."),
    ("q008", 0.0, "PRED 'Yes English tweets' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED refuses; gold='Yes'."),
    ("q010", 0.25, "PRED 'morphologically rich Slavic + English' partial."),
    ("q011", 0.0, "PRED 'response time + concurrent users' WRONG."),
    ("q013", 0.75, "PRED 'diagnosis + movement + symptoms + demographics + vitals + medication' matches most gold topics."),
    ("q014", 0.25, "PRED 'standard systems testing protocols' partial."),
    ("q015", 0.25, "PRED 'F1 84.9 better but unclear amount' partial."),
    ("q016", 0.75, "PRED 'RNN-based NMT + Transformer with SMT' matches gold."),
    ("q019", 0.25, "PRED 'training sets larger but not quantified' partial."),
    ("q020", 0.0, "PRED 'combined network' fragment; gold=tSNE."),
    ("q022", 0.75, "PRED 'CBOW PV-DM GloVe EqEmb' matches 3 of 4 gold embeddings."),
    ("q024", 0.0, "PRED 'general framework summarization' WRONG."),
    ("q025", 0.5, "PRED 'do not specify amount' implicit unanswerable match."),
    ("q026", 0.0, "PRED refuses; gold=CoNLL 2003 benchmark."),
    ("q027", 1.0, "PRED 'Yes pre-training effective' matches gold='Yes'."),
    ("q029", 0.5, "PRED 'transformation ironic to non-ironic + reverse' partial."),
    ("q031", 0.0, "PRED refuses; gold=German-English."),
    ("q033", 0.0, "PRED 'preprocessing anonymize OOV' WRONG."),
    ("q034", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q037", 0.0, "PRED 'male European American higher' WRONG; gold=females higher anger/joy."),
    ("q038", 0.25, "PRED 'loss function + validation' partial."),
    ("q040", 0.0, "PRED 'F1 0.92 + 0.97' WRONG numbers; gold=specific point ranges."),
    ("q041", 1.0, "PRED 'Yes word segmentation + spelling + rare' matches gold='Yes'."),
    ("q043", 0.25, "PRED 'lowercase + stemming evaluation' partial."),
    ("q044", 0.0, "PRED refuses; gold=BIBREF26."),
    ("q047", 0.0, "PRED 'behavioral predictors + topical patterns' WRONG topics."),
    ("q048", 0.0, "PRED 'Jasper architecture baseline' hallucinated; gold unanswerable."),
    ("q050", 1.0, "PRED 'CJFA performed better vs CJFS' matches gold."),
    ("q051", 0.0, "PRED 'base NMT model VII' WRONG; gold=pivot-based."),
    ("q053", 0.0, "PRED refuses; gold=Spearman correlation."),
    ("q054", 0.5, "PRED 'do not explicitly mention' implicit match to gold='No'."),
    ("q055", 0.75, "PRED 'four Spanish subtasks EI-Reg + EI-Oc + V-Reg + V-Oc' matches list (Spanish detail wrong)."),
    ("q056", 1.0, "PRED 'do not specify language' matches gold unanswerable."),
    ("q058", 0.25, "PRED 'user preference + coherence' partial."),
    ("q059", 1.0, "PRED 'Yes authors conduct experiments' matches gold='Yes'."),
    ("q060", 1.0, "PRED 'No no case studies' matches gold='No'."),
    ("q061", 0.5, "PRED 'traditional approaches BOW + TFIDF' partial; gold has full list."),
    ("q063", 0.5, "PRED 'achieves SOTA on two datasets + competitive for longer' partial."),
    ("q064", 1.0, "PRED 'do not specify downstream tasks' matches gold unanswerable."),
    ("q065", 0.0, "PRED 'passage does not specify' refusal; gold=Transformer."),
    ("q068", 1.0, "PRED 'do not specify baseline' matches gold unanswerable."),
    ("q069", 1.0, "PRED 'No focus on NER medical' matches gold='No'."),
    ("q070", 1.0, "PRED 'datasets with transcribed text' matches gold."),
    ("q071", 1.0, "PRED 'Yes experiment with toolkits' matches gold='Yes'."),
    ("q072", 0.25, "PRED 'context encoder' partial; gold=CNN."),
    ("q073", 0.0, "PRED refuses; gold='No'."),
    ("q075", 0.25, "PRED 'monolingual data + self-learning' partial."),
    ("q076", 0.0, "PRED 'Jan 2015 + offensive speech' WRONG; gold=SGD + naive bayes + decision tree."),
    ("q077", 1.0, "PRED 'CPMI + NNEGPMI' matches gold."),
    ("q080", 1.0, "PRED 'No car-speak based on reviews' matches gold='No'."),
    ("q081", 0.0, "PRED 'CNN' WRONG; gold=Bi-LSTM-CRF."),
    ("q082", 0.0, "PRED 'bidirectional LSTMs + trigram' missing SOTA tool names."),
    ("q084", 0.0, "PRED '4.85% VLSP 2018 + 15.09% VLSP 2019' hallucinated; gold unanswerable."),
    ("q085", 0.0, "PRED 'DARPA LORELEI Turkish Uzbek Mandarin' WRONG; gold=Switchboard."),
    ("q089", 0.0, "PRED 'usability + acceptability' WRONG; gold=BLEU."),
    ("q090", 0.25, "PRED 'three NER systems' vague; gold=Stanford NER."),
    ("q091", 0.0, "PRED refuses; gold='Yes'."),
    ("q093", 0.0, "PRED refuses; gold=BLEU-1."),
    ("q095", 0.0, "PRED 'shallow NN softmax' WRONG; gold='answering always YES'."),
    ("q096", 0.0, "PRED 'distant-supervised tweets emoticons' hallucinated; gold unanswerable."),
    ("q097", 0.5, "PRED 'decoding prepared and spontaneous speech' partial; gold=ASR."),
    ("q098", 0.25, "PRED 'best for implicit + worst for basic' partial; gold=explicit."),
]


def main() -> None:
    assert len(JUDGMENTS) == 66
    qid_prefix = "p4_tierb__qasper__flat-50__seed42__"
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
    print(f"qasper p4_tierb flat-50 seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
