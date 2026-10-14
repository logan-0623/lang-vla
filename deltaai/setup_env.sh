#!/bin/bash
# One-time env install for DeltaAI (aarch64 Grace + H100). Run on a GPU node so the final check sees CUDA:
#   srun -A $ACCOUNT -p ghx4-interactive --gpus=1 --cpus-per-task=16 --mem=64g -t 1:00:00 bash deltaai/setup_env.sh
# Upstream pins (torch 2.2 / flash-attn 2.5.5 x86 wheel) do not exist for aarch64+CUDA:
#   torch 2.6.0+cu126 is the first CUDA aarch64 pip wheel; flash-attn prebuilt aarch64 starts at cu128.
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$STORE"
[ -d "$ENV_PREFIX" ] || conda create -y -p "$ENV_PREFIX" python=3.10
conda activate "$ENV_PREFIX"

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install "https://github.com/mjun0812/flash-attention-prebuild-wheels/releases/download/v0.6.3/flash_attn-2.6.3%2Bcu128torch2.7-cp310-cp310-linux_aarch64.whl" \
  || echo "WARNING: flash-attn install failed; finetune_pion.py falls back to SDPA (docs/DECISIONS.md)"

# VLA-Adapter deps minus its torch==2.2 pins (pip would otherwise pull a CPU-only aarch64 torch 2.2).
pip install "accelerate>=0.25.0" draccus==0.8.0 einops huggingface_hub json-numpy jsonlines matplotlib \
  peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 wandb \
  "git+https://github.com/moojink/transformers-openvla-oft.git" \
  tensorflow==2.15.0 tensorflow_datasets==4.9.3 \
  "dlimp @ git+https://github.com/moojink/dlimp_openvla" diffusers==0.30.3 imageio
# tensorflow==2.15.0 on aarch64 resolves to tensorflow-cpu-aws: fine, TF only feeds RLDS on CPU.
# tensorflow_graphics drags OpenEXR (no aarch64 wheel); only geometry.transformation is imported.
pip install --no-deps tensorflow_graphics==2021.12.3 trimesh
pip install --no-deps -e "$VLAA"

# LIBERO + sim
[ -d "$STORE/LIBERO" ] || git clone https://github.com/Lifelong-Robot-Learning/LIBERO.git "$STORE/LIBERO"
pip install -e "$STORE/LIBERO" --config-settings editable_mode=compat
pip install -r "$VLAA/experiments/robot/libero/libero_requirements.txt"
# LIBERO asks for paths via input() on first import, which dies in batch jobs: pre-write its config.
L=$STORE/LIBERO/libero/libero
mkdir -p ~/.libero && cat > ~/.libero/config.yaml <<YAML
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $STORE/LIBERO/libero/datasets
assets: $L/assets
YAML
pip install "numpy<2" "opencv-python<4.11" pytest scikit-learn scipy
pip install -e "$LANGVLA_ROOT"   # langeval

python - <<'PY'
import torch, tensorflow as tf, transformers, mujoco
from transformers.utils import is_flash_attn_2_available
assert torch.cuda.is_available(), "CUDA not visible: run this inside a GPU job"
print("torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))
print("flash_attn usable:", is_flash_attn_2_available(), "| transformers", transformers.__version__,
      "| tf", tf.__version__, "| mujoco", mujoco.__version__)
PY
