#!/bin/bash
# E0 for LeRobot policies (pi0.5 / SmolVLA / X-VLA). N_EPISODES is per task.
#   sbatch --export=ALL,POLICY=pi05_libero_finetuned_v044,SUITE=libero_goal bridges2/eval_lerobot.sbatch
#SBATCH -A cis260349p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:1
#SBATCH -t 12:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -J lv-eval-lr
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/bridges2/config.sh"
use_env lerobot
POLICY=${POLICY:?} SUITE=${SUITE:-libero_goal}
OUT=$RESULTS/eval/lerobot-$POLICY-$SUITE-${SLURM_JOB_ID:-local}
# pi0.5 executes 10 actions per chunk, matching openpi (LeRobot LIBERO docs)
[[ $POLICY == pi05* ]] && DEFAULT_EXTRA="--policy.n_action_steps=10" || DEFAULT_EXTRA=""
START=$(date +%s)
lerobot-eval --output_dir="$OUT" --env.type=libero --env.task="$SUITE" \
  --eval.n_episodes="${N_EPISODES:-50}" --eval.batch_size="${BATCH:-1}" \
  --policy.path="$CKPT/$POLICY" --policy.device=cuda --env.max_parallel_tasks=1 \
  ${EXTRA_ARGS:-$DEFAULT_EXTRA}
echo "elapsed_s=$(( $(date +%s) - START ))"
