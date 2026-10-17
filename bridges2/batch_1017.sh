#!/bin/bash
# Batch of 2026-10-17 (~60-80 SU): E4 switch time, and E3 with 3 source paraphrases per task (F2 rests on one sentence per task).
#   bash bridges2/batch_1017.sh        (login node, repo root); ids appended to logs/batch_1017.txt
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
TS=$(date +%m%d%H%M)
VA=$CKPT/LIBERO-Goal-Pro
OFTG=$CKPT/openvla-7b-oft-finetuned-libero-goal
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 | tee -a logs/batch_1017.txt; }

for R in vla-adapter oft; do
  [ $R = oft ] && C=$OFTG || C=$VA
  # E4: switch to another goal on the same object at tau = start, -40, -20, -10, 0, +20 steps from the grasp.
  for t in 1 2 4 8 9; do
    sub -t 12:00:00 -J e4-$R-t$t --export=ALL,ENV=vla,REPO=$R bridges2/run.sbatch \
      python scripts/e4_switch_time.py --checkpoint $C --tasks $t --out $RESULTS/e4/$R-t$t-$TS
  done
  # E3 with 3 source paraphrases (one per category) and 10 states each.
  for t in 1 2 3 4 6 8 9; do
    sub -t 12:00:00 -J e3s3-$R-t$t --export=ALL,ENV=vla,REPO=$R bridges2/run.sbatch \
      python scripts/e3_grasp_branch.py --checkpoint $C --tasks $t --source_paraphrases 3 --n_states 10 \
      --out $RESULTS/e3/src3-$R-t$t-$TS
  done
done
echo "submitted; ids in logs/batch_1017.txt"
