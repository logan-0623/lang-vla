#!/bin/bash
# Batch 2026-10-18b: first step of the spatial-language transfer study (docs/RESEARCH_PLAN.md §11).
#   1. Novel-goal baseline: the existing checkpoints on the 10 seen Goal tasks + 9 novel combinations.
#   2. StarVLA adapter validation with the released Qwen3-VL-OFT 4in1 checkpoint: Goal success (published 96.2)
#      and the E3 restore / orig-vs-orig_rep checks, plus its novel-goal numbers.
#   3. Spatial QA data (driven by VLA-Adapter), for stage-1 VLM training on DeltaAI.
# Needs: scripts/setup_starvla.sh bridges2 and scripts/download_starvla_assets.sh bridges2 done.
#   bash bridges2/batch_1018b.sh      (ids in logs/batch_1018b.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
TS=$(date +%m%d%H%M)
VA=$CKPT/LIBERO-Goal-Pro
OFTG=$CKPT/openvla-7b-oft-finetuned-libero-goal
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 | tee -a logs/batch_1018b.txt; }

for R in vla-adapter oft; do
  [ $R = oft ] && C=$OFTG || C=$VA
  sub -t 8:00:00 -J novel-$R --export=ALL,ENV=vla,REPO=$R bridges2/run.sbatch \
    python scripts/eval_goals.py --checkpoint $C --n_init 20 --out $RESULTS/novel/$R-$TS
done
sub -t 8:00:00 -J novel-pi05 --export=ALL,ENV=lerobot bridges2/run.sbatch \
  python scripts/eval_goals.py --backend lerobot --checkpoint $CKPT/pi05_libero_finetuned_v044 --n_init 20 \
  --out $RESULTS/novel/pi05-$TS
sub -t 8:00:00 -J novel-starvla4in1 --export=ALL,SV_CKPT=$SV bridges2/run_starvla.sbatch \
  python scripts/eval_goals.py --backend starvla --checkpoint SERVER --n_init 20 --out $RESULTS/novel/starvla4in1-$TS
sub -t 4:00:00 -J e3-starvla-pilot --export=ALL,SV_CKPT=$SV bridges2/run_starvla.sbatch \
  python scripts/e3_grasp_branch.py --backend starvla --checkpoint SERVER --tasks 2,4 --n_states 3 --max_attempts 10 \
  --out $RESULTS/e3/pilot-starvla4in1-$TS
sub -t 8:00:00 -J qa-goal-v1 --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
  python scripts/make_spatial_qa.py --checkpoint $VA --n_scenes 50 --out $STORE/qa/goal-v1
echo "submitted; ids in logs/batch_1018b.txt"
