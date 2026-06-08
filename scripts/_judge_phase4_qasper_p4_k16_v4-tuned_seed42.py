"""Phase 4 cross-vendor finishing — QASPER p4_k16 v4-tuned seed=42 (64 remaining; 36 already judged)."""
from __future__ import annotations
import json
from pathlib import Path

RESULTS = Path("results/stage3/judge_queue/p4_k16__qasper__v4-tuned__seed42/results.jsonl")
JUDGE_MODEL = "claude-opus-4.7-1m"
JUDGE_PROTOCOL = "v1"

JUDGMENTS: list[tuple[str, float, str]] = [
    ("q000", 1.0, "PRED 'SVMs + LR + nbow+' matches gold='SVMs'."),
    ("q001", 0.0, "PRED 'detect stereotypes biases' hallucinated; gold unanswerable."),
    ("q005", 0.0, "PRED 'Yes can apply other domains' hallucinated; gold unanswerable."),
    ("q006", 0.0, "PRED 'outperformed previous AMR' hallucinated; gold unanswerable."),
    ("q009", 0.25, "PRED 'No phrase-based Chinese-English SMT' but mentions only one pair; gold='Yes'."),
    ("q010", 1.0, "PRED 'UD1.2 16 languages full list' matches gold."),
    ("q011", 0.0, "PRED 'correctness + response time + resources' WRONG."),
    ("q012", 0.0, "PRED 'Yes SQuAD 1.1' hallucinated; gold unanswerable."),
    ("q013", 0.75, "PRED 'HPI demographics + diagnosis + symptoms/signs' matches most gold topics."),
    ("q014", 0.25, "PRED 'evaluation protocol comparing perf' partial."),
    ("q016", 0.75, "PRED 'RNN-based NMT + Transformer-based NMT' matches gold."),
    ("q017", 0.0, "PRED lists techniques but no 'Loaded language'."),
    ("q018", 0.25, "PRED 'QG used in MRC generate from anchor paths' partial."),
    ("q019", 0.75, "PRED '270M vs 20M tokens' ratio ~13.5x ≈ gold 14x."),
    ("q020", 0.0, "PRED 'demonstrate by sh' fragment; gold=tSNE."),
    ("q022", 1.0, "PRED 'Bernoulli + CBOW + PV-DM' matches gold."),
    ("q024", 0.5, "PRED '[cls] tokens + interval' partial."),
    ("q025", 0.0, "PRED 'parallel data en-ru en-ar etc UN Corpus' WRONG; gold=No data (pretrained)."),
    ("q026", 0.0, "PRED '20 Newsgroups' WRONG; gold=CoNLL 2003."),
    ("q029", 0.5, "PRED 'transformation ironic to non-ironic + word repetition' partial."),
    ("q030", 0.5, "PRED 'models capture structure + variability' partial."),
    ("q033", 0.0, "PRED 'previous studies entities preprocessing' WRONG."),
    ("q035", 1.0, "PRED 'German-English + Ja-En + Ro-En' includes gold De-En."),
    ("q037", 0.5, "PRED 'African American higher anger/fear/sadness + European American' partial different gender angle."),
    ("q038", 0.25, "PRED 'loss function + validation perf' partial."),
    ("q039", 1.0, "PRED 'No did not use crowdsourcing' matches gold='No'."),
    ("q041", 1.0, "PRED 'Yes challenges exist' matches gold='Yes'."),
    ("q042", 0.0, "PRED '700 humans' hallucinated; gold unanswerable."),
    ("q043", 0.5, "PRED 'new metrics + generation of multi' partial."),
    ("q044", 0.0, "PRED 'not explicitly mentioned' refusal; gold=BIBREF26."),
    ("q045", 1.0, "PRED 'do not provide quantitative' matches gold unanswerable."),
    ("q047", 0.25, "PRED 'politics + religion + economics + tech' partial (politics matches)."),
    ("q050", 1.0, "PRED 'CJFA encoder performed better' matches gold."),
    ("q051", 0.0, "PRED 'monolingual data #10' WRONG; gold=pivot-based."),
    ("q053", 0.5, "PRED 'GM_KL nearest neighbours + components' partial."),
    ("q054", 0.5, "PRED 'do not indicate' implicit match to gold='No'."),
    ("q055", 0.75, "PRED 'anger subtask EI-Reg + EI-Oc + V-Reg + V-Oc' matches gold list (anger qualifier)."),
    ("q056", 1.0, "PRED 'do not specify language' matches gold unanswerable."),
    ("q058", 0.75, "PRED 'UMA + MRR + BPE + BLEU-1/4 + ROUGE-L + recipe' matches most gold metrics."),
    ("q060", 1.0, "PRED 'No no case studies' matches gold='No'."),
    ("q062", 1.0, "PRED 'error detection by Rei2016 + FCE' matches gold."),
    ("q063", 0.5, "PRED 'state-of-the-art + more pronounced' partial."),
    ("q067", 1.0, "PRED 'do not mention seasonality' matches gold unanswerable."),
    ("q068", 0.0, "PRED 'frozen Glove + LSTMs' hallucinated; gold unanswerable."),
    ("q069", 0.0, "PRED 'Yes MTL' but gold='No' — Y/N flip."),
    ("q070", 1.0, "PRED 'datasets with transcribed text' matches gold."),
    ("q071", 1.0, "PRED 'Yes experiment with toolkits' matches gold='Yes'."),
    ("q073", 1.0, "PRED 'No different genres various topics' matches gold='No'."),
    ("q075", 1.0, "PRED 'Back Translation + Mix-Source + augmentation' matches gold."),
    ("q077", 1.0, "PRED 'CPMI + NNEGPMI' matches gold."),
    ("q079", 1.0, "PRED 'ro→en + en→de + de→en + ja→en' matches gold's 4 MT tasks."),
    ("q082", 0.0, "PRED 'contextual LSTM F1 85-94%' WRONG; gold lists SOTA tools."),
    ("q083", 0.25, "PRED 'lack of powerful Vietnamese tool' partial; gold=ambiguous words."),
    ("q084", 0.0, "PRED '4.85% + 15.09% WER' hallucinated; gold unanswerable."),
    ("q085", 0.0, "PRED 'DARPA LORELEI' WRONG; gold=Switchboard."),
    ("q089", 0.0, "PRED 'usability + WER' WRONG; gold=BLEU."),
    ("q091", 0.0, "PRED refuses; gold='Yes'."),
    ("q092", 0.75, "PRED 'MLP + Eusboost + MWMOTE' includes gold MLP."),
    ("q093", 0.75, "PRED 'BLEU-1 + METEOR + ROUGE-L' includes gold BLEU-1."),
    ("q094", 0.0, "PRED 'No social media + Google news' hallucinated; gold unanswerable."),
    ("q096", 0.0, "PRED 'distant-supervised tweets' hallucinated; gold unanswerable."),
    ("q097", 0.5, "PRED 'prepared + spontaneous speech tasks' partial; gold=ASR."),
    ("q098", 0.25, "PRED 'best Comparison + Temporal + worst Contingency Expansion' partial; gold=explicit."),
    ("q099", 0.0, "PRED 'trained only on parallel data' vague; gold=attentional encoder-decoder."),
]


def main() -> None:
    assert len(JUDGMENTS) == 64
    qid_prefix = "p4_k16__qasper__v4-tuned__seed42__"
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
    print(f"qasper p4_k16 v4-tuned seed42 added={added} skipped={skipped} mean={total/added if added else 0:.4f}")


if __name__ == "__main__":
    main()
