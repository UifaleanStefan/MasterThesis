"""Helper: dump non-auto-scored calibration entries to text files for judging.

Usage:
  python scripts/_prep_dump_calib.py --config v4t-corpus-tuned
  python scripts/_prep_dump_calib.py --config attention-corpus-tuned
  python scripts/_prep_dump_calib.py --config dump-all
"""
from __future__ import annotations
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JQ = ROOT / "results" / "stage3" / "judge_queue"

REFUSAL_PATTERNS = [
    "do not have", "don't have", "context provided", "provided context",
    "passages provided", "provided passages", "not mentioned", "not provided",
    "not available", "insufficient", "no information", "cannot determine",
    "not specify", "no mention", "cannot find", "does not contain",
    "not found", "i'm sorry", "unable to", "no context", "not enough",
    "apologies", "not explicitly", "not clear", "not specified",
    "passages do not", "does not provide", "do not provide",
    "cannot be determined", "no relevant", "not discussed",
    "no specific", "cannot answer", "not contain", "no detail",
    "information is not", "there is no", "is not mentioned",
    "are not mentioned", "is not provided", "are not provided",
    "there are no", "not include", "not included", "do not contain",
    "not be determined", "not be found", "without more", "lacks",
    "no passage", "i do not see", "not see", "not support",
    "unanswerable", "unspecified", "no answer", "the relevant passages",
    "the document passages", "document does not", "documents do not",
]


def is_refusal(pred: str) -> bool:
    p = pred.strip().lower()
    if not p:
        return True
    return any(pat in p for pat in REFUSAL_PATTERNS)


def is_gold_unanswerable(gold: str) -> bool:
    return gold.strip().startswith("(unanswerable")


def dump_calibration(config: str) -> None:
    calib_dir = JQ / f"qasper__{config}__calibration__seed42"
    batch_dir  = JQ / f"qasper__{config}__batch_calib__seed42"

    if not calib_dir.exists():
        print(f"ERROR: {calib_dir} does not exist yet. Run the pipeline first.")
        return

    out_calib = Path(f"D:/qasper_{config.replace('-','_')}_calib_to_judge.txt")
    out_batch  = Path(f"D:/qasper_{config.replace('-','_')}_batch_to_judge.txt")

    # --- Calibration ---
    calib_lines = (calib_dir / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    ack_total = 0; ack_refusal = 0; ack_halluc = 0
    ans_total = 0; ans_una = 0; ans_refusal = 0; ans_regular = 0

    calib_regular = []
    for line in calib_lines:
        if not line.strip():
            continue
        r = json.loads(line)
        beh = r.get("expected_behavior", "answer")
        pred = r.get("predicted", "")
        gold = r.get("gold_answer", "")
        qid  = r["qid"]
        suffix = qid.replace(f"qasper__{config}__calibration__", "").replace("__seed42", "")

        if beh == "acknowledge_missing":
            ack_total += 1
            if is_refusal(pred):
                ack_refusal += 1
            else:
                ack_halluc += 1
        else:
            ans_total += 1
            if is_gold_unanswerable(gold):
                ans_una += 1
            elif is_refusal(pred):
                ans_refusal += 1
            else:
                ans_regular += 1
                calib_regular.append((suffix, gold, pred))

    print(f"CALIBRATION ({config}):")
    print(f"  ACK total={ack_total}: refusals={ack_refusal} (score 1.0), hallucinations={ack_halluc} (score 0.0)")
    print(f"  ANS total={ans_total}: unanswerable={ans_una} (auto), refusals={ans_refusal} (score 0.0), regular={ans_regular}")

    with out_calib.open("w", encoding="utf-8") as f:
        f.write(f"# QASPER {config} CALIBRATION - Regular answer entries to judge\n")
        f.write(f"# Total: {len(calib_regular)} entries\n\n")
        for i, (suffix, gold, pred) in enumerate(calib_regular):
            f.write(f"[{i+1}/{len(calib_regular)}] suffix={suffix}\n")
            f.write(f"G: {gold}\n")
            f.write(f"P: {pred}\n")
            f.write("\n")
    print(f"  -> Written {len(calib_regular)} regular entries to {out_calib}")

    # --- Batch calib ---
    if not batch_dir.exists():
        print(f"  batch_calib dir does not exist yet.")
        return

    batch_lines = (batch_dir / "queue.jsonl").read_text(encoding="utf-8").splitlines()
    batch_regular = []
    for line in batch_lines:
        if not line.strip():
            continue
        r = json.loads(line)
        pred = r.get("predicted", "")
        gold = r.get("gold_answer", "")
        qid  = r["qid"]
        suffix = qid.replace(f"qasper__{config}__batch__", "").replace("__seed42", "")

        if not is_gold_unanswerable(gold) and not is_refusal(pred):
            batch_regular.append((suffix, gold, pred))

    with out_batch.open("w", encoding="utf-8") as f:
        f.write(f"# QASPER {config} BATCH_CALIB - Regular answer entries to judge\n")
        f.write(f"# Total: {len(batch_regular)} entries\n\n")
        for i, (suffix, gold, pred) in enumerate(batch_regular):
            f.write(f"[{i+1}/{len(batch_regular)}] suffix={suffix}\n")
            f.write(f"G: {gold}\n")
            f.write(f"P: {pred}\n")
            f.write("\n")
    print(f"  -> Written {len(batch_regular)} regular entries to {out_batch}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    dump_calibration(args.config)
