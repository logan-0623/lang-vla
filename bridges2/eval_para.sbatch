#!/bin/bash
# LIBERO-Para paraphrase eval with its own scripts (4092 paraphrases of the 10 LIBERO-Goal tasks).
# Shard with a job array (every NSHARD-th bddl file):
#   sbatch --array=0-7 --export=ALL,MODEL=vla-adapter,CKPT_NAME=LIBERO-Goal-Pro,NSHARD=8 bridges2/eval_para.sbatch
# Note: LIBERO-Para runs ONE episode per paraphrase from init state 0 of each scene (init_data[0]).
#   For a deterministic policy that is one trajectory per (scene, paraphrase); our own E1 paraphrase mode varies init states.
#SBATCH -A cis260349p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 24:00:00
#SBATCH -o logs/%x-%A_%a.out
#SBATCH -J lv-para
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/bridges2/config.sh"
use_env vla
MODEL=${MODEL:?vla-adapter|oft} SHARD=${SLURM_ARRAY_TASK_ID:-0} NSHARD=${NSHARD:-1}
case $MODEL in
  vla-adapter) SCRIPT=eval_vla_adapter.py ;;
  oft)         SCRIPT=eval_openvla_oft.py ;;
  *) echo "unknown MODEL=$MODEL" >&2; exit 1 ;;
esac
OUT=$RESULTS/para/$MODEL-${CKPT_NAME:?}-${SLURM_ARRAY_JOB_ID:-${SLURM_JOB_ID:-local}}/shard$SHARD-of$NSHARD
SUB=$OUT/bddl; mkdir -p "$SUB"
# The script evaluates every .bddl in --bddl_dir; a directory of symlinks is the shard.
ls "$PARA/libero/libero/bddl_files/libero_para"/*.bddl | awk -v s="$SHARD" -v n="$NSHARD" '(NR-1)%n==s' \
  | xargs -I{} ln -sf {} "$SUB/"
echo "shard $SHARD/$NSHARD: $(ls "$SUB" | wc -l) paraphrases"
cd "$PARA"
python "eval_scripts/examples/$SCRIPT" --pretrained_checkpoint "$CKPT/$CKPT_NAME" --gpu 0 --seed "${SEED:-7}" \
  --bddl_dir "$SUB" --output_dir "$OUT/logs"
