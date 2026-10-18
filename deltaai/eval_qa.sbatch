#!/bin/bash
# Held-out QA accuracy (scripts/eval_qa.py) of the raw VLM or a StarVLA checkpoint, one GPU.
#   sbatch -A $ACCOUNT --export=ALL,STYLE=lang deltaai/eval_qa.sbatch                       # raw Qwen3-VL-4B
#   sbatch -A $ACCOUNT --export=ALL,STYLE=lang,CKPT=$STORE/runs/starvla/<run>/checkpoints/steps_N_pytorch_model.pt deltaai/eval_qa.sbatch
#SBATCH --job-name=lv-evalqa
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64g
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/deltaai/config.sh"
conda activate "$STORE/envs/starvla"
STYLE=${STYLE:-lang}
TAG=$( [ -n "${CKPT:-}" ] && basename "$(dirname "$(dirname "$CKPT")")" || echo raw )
python "$LANGVLA_ROOT/scripts/eval_qa.py" --base "$STORE/ckpt/Qwen3-VL-4B-Instruct" ${CKPT:+--ckpt "$CKPT"} \
  --qa_dir "$STORE/qa/${QA_SET:-goal-v1}" --style "$STYLE" --out "$STORE/results/qa/$TAG-$STYLE-$SLURM_JOB_ID"
