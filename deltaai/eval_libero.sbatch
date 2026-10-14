#!/bin/bash
# Upstream LIBERO eval (stage-1 acceptance: reproduce official env_success).
# Usage: sbatch -A $ACCOUNT --export=ALL,CKPT=$STORE/ckpt/LIBERO-Goal-Pro,SUITE=libero_goal deltaai/eval_libero.sbatch
# Non-Pro checkpoints (e.g. ckpt/LIBERO-Goal) need PRO=False.
#SBATCH --job-name=lv-eval
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/deltaai/config.sh"

OUT=$STORE/eval/${SUITE:?}-${SLURM_JOB_ID}
mkdir -p "$OUT" && cd "$OUT"          # upstream writes ./rollouts/*.mp4 relative to cwd
export PYTHONPATH="$VLAA:${PYTHONPATH:-}"
python "$VLAA/experiments/robot/libero/run_libero_eval.py" \
  --pretrained_checkpoint "${CKPT:?}" --task_suite_name "$SUITE" \
  --use_proprio True --num_images_in_input 2 --use_film "${FILM:-False}" \
  --use_pro_version "${PRO:-True}" --num_trials_per_task "${TRIALS:-50}" \
  --local_log_dir "$OUT/logs" ${EXTRA_ARGS:-}
