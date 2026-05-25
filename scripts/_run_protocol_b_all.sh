#!/bin/bash
# Phase 1.9 Protocol B (calibration) sequential runner for all 6 FB configs.
# Uses the patched run_corpus_qa.py that writes batch component to
# __batch_calib (not __batch), so it doesn't overwrite Protocol A's batch
# results.jsonl judgments.
#
# Each config: ~30 min wall, ~$0.30 API spend.
# Total: ~3 hours wall, ~$2 spend.
#
# Outputs per config:
#   results/stage3/corpus_traces/financebench__<cfg>__calibration/qa_calibration.json (1,500 entries)
#   results/stage3/corpus_traces/financebench__<cfg>__calibration/qa_batch.json (150 entries)
#   results/stage3/judge_queue/financebench__<cfg>__calibration__seed42/queue.jsonl (1,500)
#   results/stage3/judge_queue/financebench__<cfg>__batch_calib__seed42/queue.jsonl (150)

set -e
export PYTHONIOENCODING=utf-8

for cfg in v4t-canonical v4t-tuned v4t-corpus-tuned attention-corpus-tuned bm25-corpus dump-all; do
  log="results/stage3/corpus_qa_fb_${cfg}_calibration.log"
  echo ""
  echo "============================================="
  echo "=== Protocol B (calibration): $cfg"
  echo "=== log: $log"
  echo "============================================="
  python -u scripts/run_corpus_qa.py \
    --benchmark financebench \
    --config "$cfg" \
    --protocol calibration \
    --questions-per-doc 10 \
    --seed 42 \
    2>&1 | tee "$log"
done

echo ""
echo "============================================="
echo "=== Protocol B all-configs done ==="
echo "============================================="
