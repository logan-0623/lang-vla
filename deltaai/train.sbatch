#!/bin/bash
# Usage: sbatch -A $ACCOUNT --export=ALL,ARM=pion,SEED=7,DATA_NAME=libero_goal_no_noops deltaai/train.sbatch
# One GH200 node = 4 x H100-96GB. Default 4 GPUs x batch 16 = global 64, same as upstream 8 x 8.
#SBATCH --job-name=lv-train
#SBATCH --partition=ghx4
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=4
#SBATCH --cpus-per-task=128
#SBATCH --mem=400g
#SBATCH --time=48:00:00
#SBATCH --output=logs/%x-%j.out
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/deltaai/config.sh"

ARM=${ARM:-pion}
# V L A O optimizers; other arms pass OPTS + EXTRA_ARGS. The three base arms run one seed with denser
# checkpoints (every 1500 steps) for the step-wise curves; all other arms save every 5000.
case $ARM in
  adamw) OPT=(AdamW AdamW AdamW AdamW);     DEF_SAVE=1500 ;;
  muon)  OPT=(Muon Muon Muon Muon);         DEF_SAVE=1500 ;;
  pion)  OPT=(Muon Muon DefaultPion Muon);  DEF_SAVE=1500 ;;
  *)     read -ra OPT <<< "${OPTS:?unknown ARM=$ARM: set OPTS='V L A O'}"; DEF_SAVE=5000 ;;
esac
SEED=${SEED:-7}
DATA_NAME=${DATA_NAME:-libero_goal_no_noops}
NGPU=${SLURM_GPUS_ON_NODE:-4}
RUN=${ARM}-${DATA_NAME}-s${SEED}-${SLURM_JOB_ID}       # job id => never overwrites an earlier run

cd "$VLAA"
export PYTHONPATH="$VLAA:${PYTHONPATH:-}"
mkdir -p "$STORE/runs/$ARM"
nvidia-smi --query-gpu=name,memory.total --format=csv
torchrun --standalone --nnodes 1 --nproc-per-node "$NGPU" vla-scripts/finetune_pion.py \
  --vlm_path pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b \
  --config_file_path pretrained_models/configs \
  --data_root_dir "$DATA_ROOT" --dataset_name "$DATA_NAME" \
  --run_root_dir "$STORE/runs/$ARM" --run_id_note "$RUN" \
  --use_film False --num_images_in_input 2 --use_proprio True \
  --use_lora False --use_fz True --use_minivlm True --image_aug True \
  --num_steps_before_decay 100000 \
  --max_steps "${MAX_STEPS:-15000}" --save_freq "${SAVE_FREQ:-$DEF_SAVE}" --save_latest_checkpoint_only False \
  --merge_lora_during_training True \
  --batch_size "${BATCH_SIZE:-$((64 / NGPU))}" --grad_accumulation_steps "${GRAD_ACCUM:-1}" \
  --learning_rate "${LR:-1e-4}" --weight_decay "${WD:-1e-2}" --lr_ratio 1.0 --lora_rank 64 \
  --use_pro_version True --wandb_project "$DATA_NAME" --seed "$SEED" \
  --optimizer_V "${OPT[0]}" --optimizer_L "${OPT[1]}" --optimizer_A "${OPT[2]}" --optimizer_O "${OPT[3]}" \
  --pion_promotion_steps "${PROMOTION_STEPS:-0}" --pion_ns_steps "${NS_STEPS:-5}" \
  ${EXTRA_ARGS:-}
