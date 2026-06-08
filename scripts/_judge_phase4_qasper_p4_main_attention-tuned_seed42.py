"""Phase 4 cross-vendor finishing — QASPER p4_main attention-tuned seed=42 (73 new entries, 27 already judged)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_main__qasper__attention-tuned__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 0.25, "PRED 'baseline retrained on train+dev' partial; gold=SVMs."),
    ("q001", 0.0, "PRED 'ethnic marking proportion' hallucinated; gold unanswerable."),
    ("q003", 0.25, "PRED 'TAC2010 ACE2004 AQUAINT CoNLL WW' partial; gold=CoNLL-YAGO."),
    ("q005", 1.0, "PRED 'does not provide info' matches gold unanswerable."),
    ("q006", 0.0, "PRED 'evaluating first-1 Lead-1-AMR' hallucinated."),
    ("q007", 0.0, "PRED refuses; gold='Yes'."),
    ("q008", 0.0, "PRED 'Yes English' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED 'No Chinese English' but gold='Yes' — Y/N flip."),
    ("q010", 0.5, "PRED 'Czech French Italian Indonesian' partial (4 of 16)."),
    ("q011", 0.0, "PRED 'correctness + response time + memory CPU' WRONG."),
    ("q012", 0.0, "PRED 'Yes English' hallucinated; gold unanswerable."),
    ("q013", 0.5, "PRED 'demographics + diagnosis + symptoms/signs' partial."),
    ("q014", 0.25, "PRED 'simple protocol comparing systems' partial."),
    ("q017", 0.75, "PRED lists '16 propaganda techniques: Loaded language, Repetition...' includes gold."),
    ("q018", 0.5, "PRED 'QA model + QG model framework' partial."),
    ("q019", 0.75, "PRED '270M vs 20M tokens' ratio ~13.5x ≈ gold 14x."),
    ("q020", 0.0, "PRED 'achieving' fragment; gold=tSNE."),
    ("q022", 1.0, "PRED 'Bernoulli + CBOW + PV-DM' matches gold."),
    ("q024", 0.5, "PRED '[cls] tokens + interval' partial."),
    ("q025", 0.25, "PRED 'monolingual only' partial; gold=No data (pretrained)."),
    ("q027", 1.0, "PRED 'Yes pre-training effective' matches gold='Yes'."),
    ("q028", 0.0, "PRED 'BERT BIBREF12 + XLNet + RoBERTa' WRONG; gold=MaxEnt + SVMs."),
    ("q029", 0.0, "PRED 'Additional experiments' vague; gold=Irony Classifier."),
    ("q030", 0.0, "PRED 'No may not generalize' but gold='Yes' — Y/N flip."),
    ("q032", 0.25, "PRED 'three sophisticated levels' partial; gold=raw text."),
    ("q034", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q035", 1.0, "PRED 'German-English Japanese-English Romanian-English' includes gold De-En."),
    ("q037", 0.5, "PRED 'male noun phrases European American' partial; gold has gender-specific findings."),
    ("q038", 0.25, "PRED 'loss function + validation perf' partial."),
    ("q039", 1.0, "PRED 'No did not use crowdsourcing' matches gold='No'."),
    ("q040", 0.0, "PRED 'F1 0.92 + 0.97' WRONG numbers; gold=2.3-6.8 + 0.8-3.5."),
    ("q042", 0.0, "PRED '700' hallucinated; gold unanswerable."),
    ("q043", 0.5, "PRED 'multiple distinct keyphrases' partial."),
    ("q047", 0.0, "PRED 'Libertarianism + Anarcho + ronpaul' WRONG topics."),
    ("q048", 0.0, "PRED 'layer norm GAU + ReLU' hallucinated; gold unanswerable."),
    ("q050", 0.0, "PRED 'does not specify which approach' refusal; gold=CJFA encoder."),
    ("q051", 0.0, "PRED 'weak baseline (b3)' WRONG; gold=pivot-based translation."),
    ("q052", 0.75, "PRED 'grammaticality + INLINEFORM0 + unlabeled INLINEFORM0' includes gold."),
    ("q053", 0.75, "PRED 'GM_KL on entailment datasets' matches gold."),
    ("q055", 1.0, "PRED 'EI-Reg + EI-Oc + V-Reg + V-Oc' matches gold exactly."),
    ("q056", 1.0, "PRED 'do not specify language' matches gold unanswerable."),
    ("q057", 1.0, "PRED 'text classification and semantic matching' matches gold."),
    ("q058", 0.75, "PRED 'BPE + BLEU-1/4 + ROUGE-L + coherence' includes gold metrics."),
    ("q062", 0.0, "PRED 'corrected FCE training set' WRONG; gold=Rei2016 error detection."),
    ("q063", 0.5, "PRED 'outperform on all ROUGE + METEOR' partial; gold has specific numbers."),
    ("q064", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q065", 0.0, "PRED 'specific models not mentioned' refusal; gold=Transformer."),
    ("q068", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q069", 0.0, "PRED 'Yes MTL' but gold='No' — Y/N flip."),
    ("q070", 0.75, "PRED 'datasets with transcribed text' matches gold."),
    ("q071", 1.0, "PRED 'Yes try out new ideas' matches gold='Yes'."),
    ("q072", 1.0, "PRED 'CNN' exact match."),
    ("q073", 1.0, "PRED 'No do not focus + various topics' matches gold='No'."),
    ("q075", 0.75, "PRED 'back-translation + mix-source' matches gold's Back Translation."),
    ("q076", 0.0, "PRED 'F1 89.6%/89.2% classifier' WRONG; gold=SGD + naive bayes + decision tree."),
    ("q077", 0.25, "PRED 'NPMI + PPMI' partial; gold=clipped PMI + NNEGPMI."),
    ("q079", 0.0, "PRED 'do not provide info' refusal; gold=4 MT tasks."),
    ("q081", 1.0, "PRED 'Bi-LSTM-CRF model' exact match."),
    ("q082", 0.0, "PRED 'F1 84-94% LSTMs' WRONG; gold lists SOTA tools."),
    ("q083", 0.0, "PRED 'does not specify limitations' refusal; gold=ambiguous words."),
    ("q085", 0.0, "PRED 'DARPA LORELEI Turkish' WRONG; gold=Switchboard."),
    ("q087", 1.0, "PRED 'hybrid NER 0.995/0.948 + i2b2' matches gold exactly."),
    ("q088", 0.0, "PRED 'CNN aggregate location' hallucinated; gold unanswerable."),
    ("q089", 0.0, "PRED 'usability + acceptability' WRONG; gold=BLEU."),
    ("q090", 0.75, "PRED 'Stanford NER + spaCy + biLSTM' includes Stanford NER."),
    ("q092", 0.75, "PRED 'MLP + Eusboost + MWMOTE' includes gold MLP."),
    ("q093", 0.75, "PRED 'BLEU-1 + Meteor + Rouge-L' includes gold BLEU-1."),
    ("q094", 0.0, "PRED 'Yes English' hallucinated; gold unanswerable."),
    ("q095", 0.25, "PRED 'weak baseline' vague; gold='answering always YES'."),
    ("q096", 1.0, "PRED 'do not specify dataset' matches gold unanswerable."),
    ("q097", 0.75, "PRED 'WER scores by gender + role + speech type' matches gold's ASR."),
    ("q098", 0.5, "PRED 'Explicit-Temporal + Implicit-Comparison' partial."),
    ("q099", 0.0, "PRED 'baseline systems in context' fragment; gold=attentional encoder-decoder."),
]


def main() -> None:
    assert len(JUDGMENTS) == 73
    qid_prefix = "p4_main__qasper__attention-tuned__seed42__"
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
    print(f"qasper p4_main attention-tuned seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
