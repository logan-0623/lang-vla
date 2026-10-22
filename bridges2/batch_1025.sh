#!/bin/bash
# Batch 2026-10-22, step 1 of the batch_1024 follow-up (RESEARCH_PLAN §11.7): one short job, no rollouts.
#   tests; where the moved object ended up in every batch_1023/1024 run (landing); object / destination subspaces
#   up to rank 64 with contrastive variants and held-out shares by rank (subspaces2.npz). The rank / strength sweep
#   is written after these numbers.
#   git pull ; bash bridges2/batch_1025.sh       (afterwards: cat logs/er-diag-*.out)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
E=$RESULTS/erase
RUNS="$(ls -d $E/patch/*-a $E/patch/*-b $E/patch2/* | tr '\n' ' ')"
sbatch -t 0:40:00 -J er-diag --export=ALL,ENV=vla bridges2/run.sbatch bash -c "
  python -m pytest -q tests/test_qa_and_adapters.py &&
  python scripts/erase_action.py landing --anchor_run $E/none --anchor_labels $E/rec.jsonl $RUNS &&
  python scripts/erase_action.py subspaces --rec $E/rec.f16 --labels $E/rec.jsonl --run_dir $E/none --out $E/patch/subspaces2.npz"
