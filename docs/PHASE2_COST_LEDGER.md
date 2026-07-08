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
| CUAD-510 dump-all (partial, killed) | ~$4 | — | counter broken; real est. |
| earlier killed bm25 attempt (~5h) | ~$1.5 | — | no log write |
| QASPER-281 bm25-corpus-tuned | $0.34 | 0.3h | (Phase-2 QASPER top-up) |
| **CUAD-510 dump-all (full run)** | **~$125 (est.)** | ~4h | IN PROGRESS; counter broken → estimated from ~124K tok/q × 6702 × $0.15/1M |
| **OpenAI subtotal (committed)** | **≈ $12.6** | | excludes the in-progress dump-all |
| **OpenAI total (with full dump-all)** | **≈ $138** | | |

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
- **Real external OpenAI spend:** ≈ **$12.6 committed**, → **≈ $138** once the full
  dump-all @510 completes.
- **Claude judging:** ≈ **29M tokens** committed (+~7M for dump-all).
- **Avoided earlier:** killing the first dump-all attempt after ~$4 (before this
  authorized full run) and skipping nothing else.
