"""Phase 4 cross-vendor finishing — QASPER p4_k16 v4-canonical seed=42 (59 remaining)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k16__qasper__v4-canonical__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 1.0, "PRED 'SVMs + LRs + nbow+' matches gold='SVMs'."),
    ("q001", 0.0, "PRED 'manually categorized + ethnicity patterns' hallucinated; gold unanswerable."),
    ("q002", 0.5, "PRED 'N400 + P600 + EPNP + P600' partial match to gold's broken answer."),
    ("q003", 0.25, "PRED 'CoNLL + TAC + ACE + AQUAINT + WW' partial; gold=CoNLL-YAGO."),
    ("q004", 0.0, "PRED refuses; gold='Yes'."),
    ("q006", 0.0, "PRED 'low ROGUE-2 ROGUE-L' hallucinated; gold unanswerable."),
    ("q008", 0.0, "PRED 'Yes only English' hallucinated; gold unanswerable."),
    ("q009", 0.0, "PRED refuses; gold='Yes'."),
    ("q010", 1.0, "PRED 'UD1.2 16 languages full list' matches gold."),
    ("q011", 0.0, "PRED 'correctness + response times + resources' WRONG."),
    ("q012", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q014", 0.25, "PRED 'automatic evaluations + human segments' partial."),
    ("q015", 0.25, "PRED 'F1 78.6 vs 78.9 not significant' partial."),
    ("q016", 0.75, "PRED 'RNN-based NMT + Transformer + SMT' matches gold."),
    ("q019", 0.0, "PRED refuses; gold='14 times'."),
    ("q020", 0.0, "PRED 'demonstrate by co' fragment; gold=tSNE."),
    ("q022", 0.75, "PRED 'CBOW + PV-DM + GloVe + EqEmb' matches 3 of 4 gold embeddings."),
    ("q024", 0.0, "PRED 'separate optimizers' WRONG; gold=BERT 512+positions."),
    ("q025", 0.5, "PRED 'do not provide specific info' implicit unanswerable match."),
    ("q029", 0.5, "PRED 'irony accuracy + sentiment + content preservation' partial."),
    ("q030", 0.25, "PRED 'new evaluation paradigm + uncharacteristic inputs' partial."),
    ("q032", 0.25, "PRED 'at least three levels + mention level 3' partial; gold=raw text."),
    ("q033", 0.0, "PRED 'soft attention considers all entities' WRONG."),
    ("q034", 1.0, "PRED 'BIBREF9 and BIBREF8' includes gold."),
    ("q038", 0.25, "PRED 'loss function + validation perf' partial."),
    ("q042", 1.0, "PRED 'do not specify' matches gold unanswerable."),
    ("q043", 0.5, "PRED 'evaluating predictions + diversity variety' partial."),
    ("q044", 0.0, "PRED '280K news dataset' WRONG; gold=BIBREF26."),
    ("q045", 1.0, "PRED 'do not specify improvement' matches gold unanswerable."),
    ("q047", 0.5, "PRED 'politics + science + tech + business + Libertarianism' partial (politics + science + business in gold)."),
    ("q051", 0.0, "PRED 'two baseline models' vague; gold=pivot-based translation."),
    ("q053", 0.75, "PRED 'GM_KL entailment + word similarity datasets' matches gold."),
    ("q054", 1.0, "PRED 'No do not mention' matches gold='No'."),
    ("q055", 1.0, "PRED 'EI-Reg + EI-Oc + V-Reg + V-Oc' matches gold."),
    ("q058", 0.5, "PRED 'UMA + MRR + recipe coherence + BLEU' partial includes gold metrics."),
    ("q060", 1.0, "PRED 'No no case studies' matches gold='No'."),
    ("q062", 0.0, "PRED 'previous results + adaptive learning rate' WRONG; gold=Rei2016 error detection."),
    ("q063", 0.5, "PRED 'neural extractive outperform on ROUGE-1,2' partial."),
    ("q064", 1.0, "PRED 'do not provide info' matches gold unanswerable."),
    ("q065", 0.0, "PRED 'Mapudungun Spanish' WRONG; gold=Transformer."),
    ("q068", 0.0, "PRED 'frozen Glove + uni-LSTMs' hallucinated; gold unanswerable."),
    ("q070", 0.5, "PRED 'transcribed text + ASR' partial; gold=text transcription only."),
    ("q072", 0.0, "PRED 'encoder-decoder LSTM' WRONG; gold=CNN."),
    ("q073", 1.0, "PRED 'No variety topics genres' matches gold='No'."),
    ("q075", 1.0, "PRED 'Data Augmentation + Back Translation' matches gold."),
    ("q076", 0.0, "PRED '8.4% offensive vs 7.8%' WRONG; gold=SGD + naive bayes + decision tree."),
    ("q077", 0.75, "PRED 'NPMI + NNEGPMI + CPMI' includes gold."),
    ("q080", 0.0, "PRED 'Yes car-speak abstract features' but gold='No' — Y/N flip."),
    ("q082", 0.75, "PRED 'Stanford NER + DATEXIS-NER' includes gold Stanford NER."),
    ("q083", 0.75, "PRED 'ambiguous + unknown words' matches gold's ambiguous words."),
    ("q084", 0.0, "PRED '15.47% WER VLSP 2019' hallucinated; gold unanswerable."),
    ("q085", 0.0, "PRED 'DARPA LORELEI' WRONG; gold=Switchboard."),
    ("q089", 0.0, "PRED 'usability + acceptability + quality' WRONG; gold=BLEU."),
    ("q090", 0.75, "PRED 'recurrent CRF + spaCy + Stanford' includes Stanford NER."),
    ("q092", 0.25, "PRED 'Mwmote + Eusboost + f-measure' partial (no MLP); gold=MLP."),
    ("q096", 0.0, "PRED 'book corpus Skip-thought' hallucinated; gold unanswerable."),
    ("q097", 0.75, "PRED 'WER analysis + speech at episode' matches gold's ASR."),
    ("q098", 0.25, "PRED 'both implicit and explicit' partial; gold=explicit."),
    ("q099", 0.0, "PRED 'trained on parallel data' vague; gold=attentional encoder-decoder."),
]


def main() -> None:
    assert len(JUDGMENTS) == 59
    qid_prefix = "p4_k16__qasper__v4-canonical__seed42__"
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
    print(f"qasper p4_k16 v4-canonical seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
