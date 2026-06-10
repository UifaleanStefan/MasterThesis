"""Temporary analysis script for batch queue non-refusal entries."""
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

refusals = [e for e in q if is_refusal(e.get("predicted", ""))]
non_refusals = [e for e in q if not is_refusal(e.get("predicted", ""))]
print(f"Auto-score 0.0 (refusals by pattern): {len(refusals)}")
print(f"Need explicit judgment (non-refusals): {len(non_refusals)}")
print()
print("First 20 non-refusal entries:")
for e in non_refusals[:20]:
    suffix = e["qid"].replace("cuad__v4t-tuned__batch__", "").replace("__seed42", "")
    gold = str(e.get("gold_answer", ""))[:80]
    pred = str(e.get("predicted", ""))[:100]
    print(f"  {suffix}")
    print(f"    gold: {gold}")
    print(f"    pred: {pred}")
    print()
