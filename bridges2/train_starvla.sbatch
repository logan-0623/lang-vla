#!/bin/bash
# StarVLA training on Bridges-2: 4 x L40S-48GB (GPU-shared, 1 SU per GPU-hour => 4 SU per wall hour).
# Config configs/starvla_libero_goal.yaml; arms in RESEARCH_PLAN §11.2.
#   Stage 1, VLM SFT on one QA style:   sbatch --export=ALL,STAGE=vlm,QA=lang bridges2/train_starvla.sbatch
#   Stage 2, VLA on LIBERO-Goal (optionally from a stage-1 checkpoint, only its VLM is loaded; optionally
#   co-trained with a QA style):
#     sbatch --export=ALL,STAGE=vla bridges2/train_starvla.sbatch                                   # arm A
#     sbatch --export=ALL,STAGE=vla,TAG=lang,INIT=$STORE/runs/starvla/vlm-lang-<job>/checkpoints/steps_2000_pytorch_model.pt bridges2/train_starvla.sbatch
#   Smoke (s/step and memory): add MAX_STEPS=30,SAVE=30 and -t 1:00:00.
# Global batch = 4 GPUs x PER_DEV x ACCUM (default 4 x 8 x 4 = 128, the released runs' 8 x 16). Accumulation is
# set in the DeepSpeed config (StarVLA builds Accelerator(DeepSpeedPlugin()), so trainer.gradient_accumulation_steps
# in the yaml is not used); this job writes that config with ACCUM.
#SBATCH -A cis260349p
#SBATCH -p GPU-shared
#SBATCH --gpus=l40s-48:4
#SBATCH -t 48:00:00
#SBATCH -o logs/%x-%j.out
#SBATCH -J lv-starvla-train
set -euo pipefail
source "${LANGVLA_ROOT:-$SLURM_SUBMIT_DIR}/bridges2/config.sh"
conda activate "$STORE/envs/starvla"
SRC=$STORE/src/starVLA
export LANGVLA_QA_ROOT=$STORE/qa/${QA_SET:-goal-v1}
STAGE=${STAGE:?vlm or vla}
NGPU=$(nvidia-smi -L | wc -l)
PER_DEV=${PER_DEV:-8}
ACCUM=${ACCUM:-4}
ARGS=(--config_yaml "$LANGVLA_ROOT/configs/starvla_libero_goal.yaml" --run_root_dir "$STORE/runs/starvla" --seed "${SEED:-42}")

case $STAGE in
  vlm)
    RUN=vlm-${QA:?lang|num|ctrl}-${SLURM_JOB_ID}
    SCRIPT=starVLA/training/train_starvlm.py
    ARGS+=(--datasets.vlm_data.dataset_use "libero_qa_$QA" --datasets.vlm_data.eval_dataset "libero_qa_${QA}_heldout"
           --datasets.vlm_data.per_device_batch_size "$PER_DEV" --trainer.loss_scale.vlm 1.0
           --trainer.max_train_steps "${MAX_STEPS:-2000}" --trainer.save_interval "${SAVE:-1000}" --trainer.num_warmup_steps 100) ;;
  vla)
    RUN=vla-init_${TAG:-base}-co_${COTRAIN:-none}-${SLURM_JOB_ID}     # TAG names the stage-1 run behind INIT
    if [ -n "${COTRAIN:-}" ]; then
      SCRIPT=starVLA/training/train_starvla_cotrain.py
      ARGS+=(--datasets.vlm_data.dataset_use "libero_qa_$COTRAIN" --datasets.vlm_data.eval_dataset "libero_qa_${COTRAIN}_heldout"
             --datasets.vlm_data.per_device_batch_size "${VLM_PER_DEV:-2}")
    else
      SCRIPT=starVLA/training/train_starvla.py
    fi
    if [ -n "${INIT:-}" ]; then ARGS+=(--trainer.pretrained_checkpoint "$INIT" --trainer.reload_modules qwen_vl_interface); fi
    ARGS+=(--datasets.vla_data.per_device_batch_size "$PER_DEV"
           --trainer.max_train_steps "${MAX_STEPS:-5000}" --trainer.save_interval "${SAVE:-2500}") ;;     # ~12 epochs of Goal at 128
  *) echo "STAGE must be vlm or vla" >&2; exit 1 ;;
esac

# DeepSpeed + accelerate configs with this job's accumulation (StarVLA's own files, accumulation changed).
CFG_DIR=$STORE/runs/starvla/_ds/$SLURM_JOB_ID && mkdir -p "$CFG_DIR"
python - "$SRC/starVLA/config/deepseeds/ds_config.yaml" "$CFG_DIR/ds_config.json" "$ACCUM" <<'PY'
import json, sys
cfg = json.load(open(sys.argv[1])); cfg["gradient_accumulation_steps"] = int(sys.argv[3])
json.dump(cfg, open(sys.argv[2], "w"), indent=1)
PY
sed -e "s|deepspeed_config_file: .*|deepspeed_config_file: \"$CFG_DIR/ds_config.json\"|" -e "s|num_processes: .*|num_processes: $NGPU|" \
  "$SRC/starVLA/config/deepseeds/deepspeed_zero2.yaml" > "$CFG_DIR/accelerate.yaml"

# DeepSpeed runs `$CUDA_HOME/bin/nvcc -V` at import (default /usr/local/cuda, absent on compute nodes) and only
# reads the release number; no ops are built. Give it a stub nvcc reporting torch's CUDA version.
CU=$(python -c "import torch; print(torch.version.cuda)")
mkdir -p "$CFG_DIR/cuda/bin" && printf '#!/bin/sh\necho "Cuda compilation tools, release %s, V%s"\n' "$CU" "$CU" > "$CFG_DIR/cuda/bin/nvcc"
chmod +x "$CFG_DIR/cuda/bin/nvcc" && export CUDA_HOME=$CFG_DIR/cuda && "$CUDA_HOME/bin/nvcc" -V

cd "$SRC"
export PYTHONPATH=$SRC
nvidia-smi --query-gpu=name,memory.total --format=csv
# Rank 0 computes the LeRobot action statistics (meta/stats_gr00t.json) while the other ranks wait in a barrier
# with NCCL's 30-min timeout; the first run hit it (49454257). Build the dataset once here, single process, so the
# cache exists before launch (a no-op when it is already cached).
if [ "$STAGE" = vla ]; then
  T0=$SECONDS; python - "$LANGVLA_ROOT/configs/starvla_libero_goal.yaml" <<'PY'
import sys
from omegaconf import OmegaConf
from starVLA.model.framework.share_tools import apply_config_compat
from starVLA.dataloader.lerobot_datasets import get_vla_dataset
cfg = apply_config_compat(OmegaConf.load(sys.argv[1]))
print("prewarm: vla dataset length", len(get_vla_dataset(data_cfg=cfg.datasets.vla_data)), flush=True)
PY
  echo "dataset prewarm $((SECONDS - T0)) s"
fi
# NCCL check: the first collective (a barrier) hung for NCCL's 30 min in two runs (49454257, 49457809) even with
# the dataset prewarmed in 40 s. GPU-shared nodes may block PCIe peer-to-peer; test one all-reduce and fall back to
# NCCL_P2P_DISABLE=1 if it does not finish in 2 min.
nccl_ok() { timeout 120 torchrun --standalone --nproc_per_node "$NGPU" "$LANGVLA_ROOT/scripts/nccl_check.py"; }
export NCCL_DEBUG=${NCCL_DEBUG:-WARN}
if ! nccl_ok; then
  echo "NCCL all-reduce failed or hung; retrying with NCCL_P2P_DISABLE=1"
  export NCCL_P2P_DISABLE=1
  nccl_ok || { echo "NCCL still failing with P2P disabled" >&2; exit 1; }
fi
echo "run $RUN: $NGPU GPUs x $PER_DEV x accum $ACCUM = global batch $((NGPU * PER_DEV * ACCUM))"
accelerate launch --config_file "$CFG_DIR/accelerate.yaml" --num_processes "$NGPU" \
  "$SCRIPT" "${ARGS[@]}" --run_id "$RUN" ${EXTRA_ARGS:-}
