#!/bin/bash
# Batch 2026-10-18c: the four novel-goal jobs of batch_1018b again, after the stopping fix (eval_goals.py ended a
# rollout when *any* tracked goal held 10 steps, so opening the top drawer (n7) cut n6 and seen task 3 short).
#   scancel 49404858 49404859 49404860 49404861 ; git pull ; bash bridges2/batch_1018c.sh
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
TS=$(date +%m%d%H%M)
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 | tee -a logs/batch_1018c.txt; }
sub -t 8:00:00 -J novel-vla-adapter --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
  python scripts/eval_goals.py --checkpoint $CKPT/LIBERO-Goal-Pro --n_init 20 --out $RESULTS/novel/vla-adapter-$TS
sub -t 8:00:00 -J novel-oft --export=ALL,ENV=vla,REPO=oft bridges2/run.sbatch \
  python scripts/eval_goals.py --checkpoint $CKPT/openvla-7b-oft-finetuned-libero-goal --n_init 20 --out $RESULTS/novel/oft-$TS
sub -t 8:00:00 -J novel-pi05 --export=ALL,ENV=lerobot bridges2/run.sbatch \
  python scripts/eval_goals.py --backend lerobot --checkpoint $CKPT/pi05_libero_finetuned_v044 --n_init 20 --out $RESULTS/novel/pi05-$TS
sub -t 8:00:00 -J novel-starvla4in1 --export=ALL,SV_CKPT=$SV bridges2/run_starvla.sbatch \
  python scripts/eval_goals.py --backend starvla --checkpoint SERVER --n_init 20 --out $RESULTS/novel/starvla4in1-$TS
echo "submitted; ids in logs/batch_1018c.txt"
