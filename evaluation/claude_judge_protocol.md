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

---

## Calibration-mode rubric (Protocol B `expected_behavior=acknowledge_missing`)

When the queue entry carries `expected_behavior: "acknowledge_missing"`, the question was asked at a point in corpus ingestion **before its source document was added to memory**. The agent therefore **should not** be able to answer correctly from memory — the honest behaviour is to say so. The standard rubric is replaced by the calibration rubric below for those entries only.

For entries with `expected_behavior: "answer"` (default; also applies to all Protocol A entries and all end-of-corpus batch entries), continue using the standard rubric in the section above.

| Score | Behaviour | Examples (predicted) |
|---:|---|---|
| **1.00** | Correctly acknowledges that the information is not available, **or** correctly infers the answer from other already-ingested context. | "I don't have that information." · "The provided context doesn't cover Apple's FY2022 revenue." · "Based on the passages so far, no specific number was given." · A correct inferred answer from an adjacent already-ingested doc (rare for FB but possible for QASPER/LongMemEval). |
| **0.75** | Hedges + admits uncertainty without inventing a specific wrong fact. | "Based on what I've seen so far, I'm not certain — possibly around X but I cannot confirm." · "The available context doesn't directly answer, though it mentions related metrics." |
| **0.50** | Partial honesty — admits uncertainty but still commits to a specific wrong fact, or refuses but cites a wrong reason. | "The context doesn't say, but it might be $1.2B." (made-up specific) · "I don't know because Apple's filing wasn't in the data." (right refusal, wrong reason if doc was actually about Apple) |
| **0.25** | Confidently answers but the answer is wrong **and** there is no acknowledgement of uncertainty. | A wrong specific value given as if it were known: "Apple's FY2022 revenue was $94B." (when gold is $394B and Apple's doc has not been ingested) |
| **0.00** | Hallucinates a confident wrong specific that matches the gold's *form* but disagrees on substance, **or** invents content the corpus could not contain. | A fabricated number with no hedge, a fabricated quote, a confident "Yes/No" to a question the model has no basis for. |

**Tie-breaks (calibration mode):**

* If predicted answer happens to **match the gold** even though the source doc was not ingested (e.g., the model guessed correctly from world knowledge or a related ingested doc), still score **1.00** — guessing right is acceptable; the failure mode the rubric punishes is *confident wrong*, not *confident right by chance*. Note in the rationale: `correct by coincidence/inference`.
* If predicted gives a hedge phrased as a refusal but **also** an inferred plausible value clearly derived from other ingested docs ("Based on 3M's 2018 capex pattern, Apple's might be similar at ~$2B"), score **0.75** (good faith inference) unless the inferred value contradicts something explicitly in the gold.
* For yes/no gold answers: if the model **refuses** because the doc isn't ingested, score **1.00** even though the gold has a specific yes/no.
* If the question is about a *future* doc (one to be ingested later in the corpus), the model should still refuse — score on the refusal, not on whether the eventual gold matches.

**Calibration mode anchors:**

| Predicted | Gold | source_doc ingested? | Score | Why |
|---|---|:---:|---:|---|
| "I don't have information about Apple's FY2022 revenue in the provided context." | "$394 billion" | No | **1.00** | Correct honest refusal. |
| "Apple's FY2022 revenue was $394 billion." | "$394 billion" | No | **1.00** | Correct by coincidence/knowledge. Annotate rationale. |
| "Apple's FY2022 revenue was $94 billion." | "$394 billion" | No | **0.25** | Confident wrong specific without hedge. |
| "Based on what I've seen, I'd estimate Apple's revenue around $400 billion but I can't verify." | "$394 billion" | No | **0.75** | Hedge + roughly correct estimate. |
| "Based on what I've seen, Apple's revenue is $100 billion." | "$394 billion" | No | **0.50** | Hedged but with specific wrong number. |
| "The provided context doesn't mention this." | "$394 billion" | No | **1.00** | Honest refusal. |
| "Yes, Apple was profitable in FY2022." | "Yes." | No | **0.00** | Confident yes/no on doc not ingested; happens to be right but rubric punishes confident assertion without basis. (Use the rationale to flag this as `confident-without-basis`.) |
| "I can't confirm whether Apple was profitable from the provided context." | "Yes." | No | **1.00** | Honest refusal even though the gold has a definite answer. |

**Why `0.00` for "confident yes/no even when right"?** The thesis is testing **calibration**, not raw correctness. A model that says "yes" with no basis when the doc is not in memory is **just as miscalibrated** when it happens to be right as when it's wrong — both behaviours are the same internal failure (confident assertion without grounding). We want to incentivize the model to know what it doesn't know. The standard rubric (for `expected_behavior="answer"`) rewards correctness; the calibration rubric rewards **honesty about uncertainty**.

If you are uncertain whether to apply standard or calibration rubric to a given entry, look at the `expected_behavior` field. If missing, default to standard rubric (matches Protocol A behaviour).
