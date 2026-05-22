# Claude-as-Judge Protocol

**Role:** Claude scores corpus-mode QA results from `scripts/run_corpus_qa.py` in-session, without calling the OpenAI judge API. This eliminates the gpt-4o-mini self-judges-gpt-4o-mini self-bias that Phase 1.7 critique #11 flagged.

**Architecture:** The QA runner writes batches of `{qid, question, gold, predicted, ...}` to `results/stage3/judge_queue/<run_id>/queue.jsonl`. Claude reads the queue, scores each entry, writes results to `results/stage3/judge_queue/<run_id>/results.jsonl`. The aggregator script reads results back and merges them into `qa_online.json` / `qa_batch.json` for downstream stats.

## Rubric (apply consistently across all batches)

For each (question, gold_answer, predicted_answer) tuple, output a single float in `[0.0, 1.0]` plus a one-sentence rationale.

| Score | Meaning | Rule |
|---:|---|---|
| **1.0** | Fully correct | All key facts in the gold are present and correct in the predicted answer. Paraphrasing is OK. Extra correct context does not penalize. |
| **0.75** | Mostly correct | The main fact is correct; minor secondary facts are missing or hedged. Or: the predicted answer states the right value with a wrong unit/format that doesn't change meaning. |
| **0.5** | Partially correct | Half the key facts are present; the other half are missing or wrong. Or: the predicted answer contains the right information but buried in incorrect surrounding claims. |
| **0.25** | Mostly wrong | A single relevant token or concept is present, but the overall claim is wrong. Or: the predicted answer is on-topic but disagrees with the gold on the central fact. |
| **0.0** | No relevant facts | The predicted answer does not contain any of the gold's key facts. "I don't know" / refusals also score 0.0. |

**Tie-breaks:**

* "Information not available in the context" / "Cannot determine" responses → **0.0** unless the gold answer itself is also an unanswerability claim.
* Numeric answers: tolerance is 5% relative for magnitudes; exact match for years, quarters, fiscal-year markers. ("$56.5 billion" matches gold "$56,500,000,000" at 1.0; "30 days" matches gold "thirty days" at 1.0; "Q3 2023" matches gold "third quarter 2023" at 1.0.)
* Multi-reference gold (`["30 days", "thirty days"]`): score against the BEST-matching reference (max over references).
* Verbose-but-correct answers: full credit. Conciseness is not a criterion.
* Hallucinations: if predicted adds a wrong-but-not-gold-contradicting claim, dock 0.25; if it contradicts the gold, dock to ≤ 0.25.

## Output format

For each queue entry, append one line to `results.jsonl`:

```json
{"qid": "<from queue>", "judge_score": <float in [0,1]>, "rationale": "<≤ 30 words>"}
```

* `qid` must match the queue entry's `qid` exactly.
* `judge_score` must be in `[0.0, 1.0]`, rounded to 2 decimals (0.00, 0.25, 0.50, 0.75, 1.00 are the canonical values).
* `rationale` must be ≤ 30 words and reference the specific fact comparison (e.g. "Gold says 30 days; predicted says 30 calendar days. Full match.").

## Batching

The user invokes "judge the next batch" or similar. Claude reads the queue file in chunks (up to 200-300 entries per turn given context budget), writes results for those entries, and reports progress (`N/Total done; next batch ready`). Multiple sessions may be needed for large queues.

## Consistency anchors

To keep judging consistent across batches and sessions, treat these examples as anchors:

| Predicted | Gold | Score | Why |
|---|---|---:|---|
| "The notice period is 30 calendar days." | "30 days" | 1.00 | Right value, paraphrase OK, "calendar" is non-contradicting clarification. |
| "30 days, plus a 5-day grace period." | "30 days" | 1.00 | Right value present; extra factual info doesn't penalize. |
| "Sixty days notice." | "30 days" | 0.00 | Wrong value contradicts gold. |
| "The contract terminates after sufficient notice." | "30 days" | 0.00 | No numeric value; refuses to commit. |
| "Around 30 days." | "30 days" | 1.00 | "Around" is acceptable hedging. |
| "Microsoft reported $56.5 billion." | "$56.5 billion" | 1.00 | Identical. |
| "Microsoft reported $56 billion." | "$56.5 billion" | 0.50 | Rounded; key value within 1% but loses the decimal. |
| "Microsoft reported $5.65 billion." | "$56.5 billion" | 0.00 | Off by 10x; major error. |
| "Microsoft, Apple, and Google reported earnings." | "Apple reported earnings" | 0.75 | Right company mentioned, but predicted is broader/less precise. |
| "I cannot determine from the provided context." | "30 days" | 0.00 | Refusal. |
| "The provided context discusses notice but no specific period." | "30 days" | 0.00 | Refusal disguised as analysis. |

## Anti-leakage

Do not consult the snapshots, original documents, or any cached information beyond what is in the queue entry. The judgment is question + gold + predicted only. (The QA runner separately captures retrieved_steps for downstream analysis but the judge should not use them.)
