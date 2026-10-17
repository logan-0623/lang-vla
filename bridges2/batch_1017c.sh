#!/bin/bash
# Batch 2026-10-17c (~35-45 SU): H1'' (placement-relative switch times, RESEARCH_PLAN §3.3), the two failed
# VLA-Adapter E4 jobs, and the pi0.5 E3 pilot (cluster validation).
#   bash bridges2/batch_1017c.sh      (ids in logs/batch_1017c.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
TS=$(date +%m%d%H%M)
VA=$CKPT/LIBERO-Goal-Pro
OFTG=$CKPT/openvla-7b-oft-finetuned-libero-goal
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 | tee -a logs/batch_1017c.txt; }

for R in vla-adapter oft; do
  [ $R = oft ] && C=$OFTG || C=$VA
  for t in 1 2 4 8 9; do
    sub -t 12:00:00 -J e4frac-$R-t$t --export=ALL,ENV=vla,REPO=$R bridges2/run.sbatch \
      python scripts/e4_switch_time.py --checkpoint $C --tasks $t --taus 0 --post_fracs 0.25,0.5,0.75 \
      --out $RESULTS/e4/frac-$R-t$t-$TS
  done
done
# The two VLA-Adapter E4 jobs that failed on the config.json race (49344275 / 49344277).
for t in 4 9; do
  sub -t 12:00:00 -J e4-vla-adapter-t$t --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
    python scripts/e4_switch_time.py --checkpoint $VA --tasks $t --out $RESULTS/e4/vla-adapter-t$t-$TS
done
# pi0.5: E3 pilot = cluster validation of the LeRobot adapter (restore check, CRN repeatability, orig == orig_rep).
sub -t 4:00:00 -J e3-pi05-pilot --export=ALL,ENV=lerobot bridges2/run.sbatch \
  python scripts/e3_grasp_branch.py --backend lerobot --checkpoint $CKPT/pi05_libero_finetuned_v044 \
  --tasks 2,4 --n_states 3 --max_attempts 10 --out $RESULTS/e3/pilot-pi05-$TS
echo "submitted; ids in logs/batch_1017c.txt"
