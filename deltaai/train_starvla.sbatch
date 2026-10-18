#!/bin/bash
# StarVLA training on one GH200 node (4 x H100), config configs/starvla_libero_goal.yaml.
#   Stage 1, VLM SFT on one QA style (lang | num | ctrl; data from scripts/make_spatial_qa.py under $STORE/qa/$QA_SET):
#     sbatch -A $ACCOUNT --export=ALL,STAGE=vlm,QA=lang deltaai/train_starvla.sbatch
#   Stage 2, VLA on LIBERO-Goal, optionally from a stage-1 checkpoint (only its VLM is loaded) and/or co-trained
#   with a QA style (COTRAIN=lang):
#     sbatch -A $ACCOUNT --export=ALL,STAGE=vla,TAG=lang,INIT=$STORE/runs/starvla/vlm-lang-<job>/checkpoints/steps_2000_pytorch_model.pt deltaai/train_starvla.sbatch
#SBATCH --job-name=lv-starvla
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
conda activate "$STORE/envs/starvla"
SRC=$STORE/src/starVLA
export LANGVLA_QA_ROOT=$STORE/qa/${QA_SET:-goal-v1}
STAGE=${STAGE:?vlm or vla}
NGPU=${SLURM_GPUS_ON_NODE:-4}
ARGS=(--config_yaml "$LANGVLA_ROOT/configs/starvla_libero_goal.yaml" --run_root_dir "$STORE/runs/starvla" --seed "${SEED:-42}")

case $STAGE in
  vlm)
    RUN=vlm-${QA:?lang|num|ctrl}-${SLURM_JOB_ID}
    SCRIPT=starVLA/training/train_starvlm.py
    ARGS+=(--datasets.vlm_data.dataset_use "libero_qa_$QA" --datasets.vlm_data.eval_dataset "libero_qa_${QA}_heldout"
           --datasets.vlm_data.per_device_batch_size "${VLM_BS:-8}" --trainer.gradient_accumulation_steps 1
           --trainer.loss_scale.vlm 1.0
           --trainer.max_train_steps "${MAX_STEPS:-2000}" --trainer.save_interval "${SAVE:-1000}" --trainer.num_warmup_steps 100) ;;
  vla)
    RUN=vla-init_${TAG:-base}-co_${COTRAIN:-none}-${SLURM_JOB_ID}     # TAG names the stage-1 run behind INIT
    if [ -n "${COTRAIN:-}" ]; then
      SCRIPT=starVLA/training/train_starvla_cotrain.py
      ARGS+=(--datasets.vlm_data.dataset_use "libero_qa_$COTRAIN" --datasets.vlm_data.eval_dataset "libero_qa_${COTRAIN}_heldout")
    else
      SCRIPT=starVLA/training/train_starvla.py
    fi
    [ -n "${INIT:-}" ] && ARGS+=(--trainer.pretrained_checkpoint "$INIT" --trainer.reload_modules qwen_vl_interface)
    ARGS+=(--trainer.max_train_steps "${MAX_STEPS:-20000}" --trainer.save_interval "${SAVE:-5000}") ;;
  *) echo "STAGE must be vlm or vla" >&2; exit 1 ;;
esac

cd "$SRC"
export PYTHONPATH=$SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
accelerate launch --config_file starVLA/config/deepseeds/deepspeed_zero2.yaml --num_processes "$NGPU" \
  "$SCRIPT" "${ARGS[@]}" --run_id "$RUN" ${EXTRA_ARGS:-}
