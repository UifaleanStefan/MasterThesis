# Phase 2 / CUAD-510 Cost Ledger

Two cost buckets: **OpenAI** (gpt-4o-mini answerer — real external $) and **Claude
judging** (1-by-1 draft→verify Workflow — subagent tokens, on the Claude plan).
OpenAI figures are the runner's own cost counter from each run log unless noted.

## OpenAI generation (gpt-4o-mini)

| Run | OpenAI $ | Wall | Notes |
|---|---|---|---|
| CUAD-510 bm25-corpus-tuned | $1.94 | 4.1h | |
| CUAD-510 hipporag-corpus | $2.89 | 9.8h | PPR over 50k+ nodes |
| CUAD-510 letta-corpus | $0.89 | 2.6h | |
| CUAD-510 attention-corpus-tuned | $1.08 | 2.6h | |
| QASPER-281 bm25-corpus-tuned | $0.34 | 0.3h | (Phase-2 QASPER top-up) |
| CUAD-510 dump-all — killed full attempts (×2) | ~$12–19 sunk | — | no partial saves; wasted |
| earlier killed bm25 attempt (~5h) | ~$1.5 | — | no log write |
| **CUAD-510 dump-all @510 (150-q probe)** | **~$0** | ~17m | 150/150 OVERFLOW the 128K window → skipped before any API call; no cost |
| **OpenAI TOTAL (final)** | **≈ $22** | | vs the ~$138 a full dump-all batch would have cost |

**dump-all @510 outcome:** rather than the ~$138 full 6702-question batch, ran the
purpose-built `run_scalability_qa.py` 150-question probe. Result: the full context
(74,435 events ≈ 43× the 128K window) **overflows on all 150 questions** — dump-all
produces no valid answer at 510, the empirical form of the analytical
overflow-at-N≈11 result. Cost ≈ $0 (overflow detected pre-API). The ~$12–19 sunk
on the two killed full-batch attempts (which silently truncate-and-answer) is the
only waste; switching to the probe capped the rest.

## Claude judging (subagent tokens, draft→verify)

| Cell judged | Subagent tokens |
|---|---|
| CUAD-510 bm25-corpus-tuned | 7.14M |
| CUAD-510 hipporag-corpus | 7.27M |
| CUAD-510 letta-corpus | 6.93M |
| CUAD-510 attention-corpus-tuned | ~7M (crashed original + 3.6M resume) |
| QASPER-281 bm25-corpus-tuned | 0.79M |
| **CUAD-510 dump-all** | ~7M (pending) |
| **Judging subtotal** | **≈ 29M (+~7M pending)** |

At Opus API rates (~$5/M in, ~$25/M out; judging is mostly input) this is roughly
**$300–450**; on a Claude subscription it is plan-covered, not a separate bill.

## Bottom line
- **Real external OpenAI spend:** ≈ **$22 total** (4 CUAD-510 baselines $6.79 +
  QASPER top-up $0.34 + ~$12–19 sunk on killed dump-all attempts + ~$0 probe).
- **Claude judging:** ≈ **28M tokens** (the four judged CUAD-510 baselines +
  QASPER); dump-all needed no judging (overflow).
- **Avoided:** ~$116 by probing dump-all (150 q, overflow, ~$0) instead of the
  full ~$138 6702-question batch.
