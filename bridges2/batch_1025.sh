#!/bin/bash
# Batch 2026-10-22, step 1 of the batch_1024 follow-up (RESEARCH_PLAN §11.7): one short job, no rollouts.
#   tests; where objects were at the end of every batch_1023/1024 run (landing: these runs predate final_pos, so it
#   reports the last-decision position); object / destination subspaces up to rank 64 with contrastive variants,
#   fit with every failing novel goal held out (instructions and episodes; n1, n7, n8 stay in), held-out shares and
#   leakage by rank (subspaces2.npz = the train-split fit). The rank / strength sweep is written after these numbers.
#   git pull ; bash bridges2/batch_1025.sh       (afterwards: cat logs/er-diag-*.out)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
E=$RESULTS/erase
[ -e $E/patch/subspaces2.npz ] && { echo "$E/patch/subspaces2.npz exists; move it away first" >&2; exit 1; }
RUNS="$(ls -d $E/patch/*-a $E/patch/*-b $E/patch2/* | tr '\n' ' ')"
sbatch -t 0:40:00 -J er-diag --export=ALL,ENV=vla bridges2/run.sbatch bash -c "
  python -m pytest -q tests/test_qa_and_adapters.py &&
  python scripts/erase_action.py landing --anchor_run $E/none --anchor_labels $E/rec.jsonl $RUNS &&
  python scripts/erase_action.py subspaces --rec $E/rec.f16 --labels $E/rec.jsonl --run_dir $E/none --exclude n0,n2,n3,n4,n5,n6 --out $E/patch/subspaces2.npz"
