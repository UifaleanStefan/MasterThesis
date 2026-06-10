"""Dump all non-refusal batch entries for manual judging."""
import json

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


def is_refusal(pred):
    p = pred.strip().lower()
    if not p:
        return True
    return any(pat in p for pat in REFUSAL_PATTERNS)


q = [json.loads(l) for l in open(
    "results/stage3/judge_queue/cuad__v4t-tuned__batch_calib__seed42/queue.jsonl"
)]
non_refusals = [e for e in q if not is_refusal(e.get("predicted", ""))]

with open("D:/cuad_v4t_tuned_batch_nonrefusals.txt", "w", encoding="utf-8") as f:
    f.write(f"Total non-refusal entries: {len(non_refusals)}\n\n")
    for i, e in enumerate(non_refusals):
        suffix = e["qid"].replace("cuad__v4t-tuned__batch__", "").replace("__seed42", "")
        f.write(f"=== [{i+1}/{len(non_refusals)}] {suffix} ===\n")
        f.write(f"Q: {e['question']}\n")
        f.write(f"GOLD: {e.get('gold_answer', '')}\n")
        f.write(f"PRED: {e.get('predicted', '')}\n")
        f.write("\n")

print(f"Wrote {len(non_refusals)} entries to D:/cuad_v4t_tuned_batch_nonrefusals.txt")
