#!/bin/bash
# `vla` env: VLA-Adapter (Pion copy, with our patches) + OpenVLA-OFT + LIBERO-Para's libero (a superset of LIBERO).
# Both model repos have identical deps and both ship a `prismatic` package, so neither is pip-installed:
# jobs put the right repo first on PYTHONPATH.
# Run on a GPU node so the final check sees CUDA:
#   srun -A cis260349p -p GPU-shared --gpus=l40s-48:1 -t 1:00:00 --pty bash bridges2/setup_vla.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
mkdir -p "$STORE/src" "$STORE/envs"

if [ ! -d "$CONDA_DIR" ]; then
  curl -fsSL -o /tmp/miniforge.sh https://github.com/conda-forge/miniforge/releases/latest/download/Miniforge3-Linux-x86_64.sh
  bash /tmp/miniforge.sh -b -p "$CONDA_DIR" && rm /tmp/miniforge.sh
  source "$CONDA_DIR/etc/profile.d/conda.sh"
fi
[ -d "$STORE/envs/vla" ] || conda create -y -p "$STORE/envs/vla" python=3.10.16
use_env vla

clone() { [ -d "$2" ] || git clone -q "$1" "$2"; git -C "$2" checkout -q "$3"; }
clone https://github.com/moojink/openvla-oft.git "$OFT" "$OFT_REV"
clone https://github.com/cau-hai-lab/LIBERO-Para.git "$PARA" "$PARA_REV"
# The fork takes every suite's instruction from the bddl (:language) text, which differs from the filename-derived
# instruction all published evals use (OFT Goal: 83.6% instead of 97.9%). Restore it for the original suites.
P=$LANGVLA_ROOT/patches/libero-para-filename-language.patch
git -C "$PARA" apply --reverse --check "$P" 2>/dev/null || git -C "$PARA" apply "$P"

pip install torch==2.2.0 torchvision==0.17.0 torchaudio==2.2.0
# Same dependency list as both pyproject.toml files (they are identical), minus the editable package itself.
pip install "accelerate>=0.25.0" draccus==0.8.0 einops huggingface_hub json-numpy jsonlines matplotlib \
  peft==0.11.1 protobuf rich sentencepiece==0.1.99 timm==0.9.10 tokenizers==0.19.1 wandb \
  "transformers @ git+https://github.com/moojink/transformers-openvla-oft.git" \
  tensorflow==2.15.0 tensorflow_datasets==4.9.3 tensorflow_graphics==2021.12.3 \
  "dlimp @ git+https://github.com/moojink/dlimp_openvla" diffusers==0.30.3 imageio
pip install packaging ninja
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.5.5/flash_attn-2.5.5+cu122torch2.2cxx11abiFALSE-cp310-cp310-linux_x86_64.whl \
  || echo "WARNING: flash-attn install failed; training falls back to SDPA, eval never uses it"

# LIBERO-Para's libero keeps all original suites and adds libero_para (4092 paraphrase bddl files).
pip install -e "$PARA" --config-settings editable_mode=compat
pip install -r "$VLAA/experiments/robot/libero/libero_requirements.txt"
pip install "numpy<2" "opencv-python<4.11" pytest scikit-learn scipy pandas
# tensorflow-metadata >=1.17 needs protobuf 5 (google.protobuf.runtime_version), TF 2.15 needs <5; pinning it
# downgrades protobuf, which recent wandb no longer supports (ImportError on wandb_telemetry_pb2.Imports).
pip install "tensorflow-metadata<1.17" "wandb<0.18"
# robosuite 1.4.x asserts joint types against mujoco enums, which fails on mujoco 3.x (get_joint_qpos_addr).
pip install mujoco==2.3.7
pip install -e "$LANGVLA_ROOT"   # langeval

# LIBERO prompts via input() when its config is missing: write it up front.
L=$PARA/libero/libero
mkdir -p "$LIBERO_CONFIG_PATH" && cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $PARA/libero/datasets
assets: $L/assets
YAML

# LIBERO-Para's eval scripts import the model repos from eval_scripts/<name>.
ln -sfn "$VLAA" "$PARA/eval_scripts/vla-adapter"
ln -sfn "$OFT" "$PARA/eval_scripts/openvla-oft"

python - <<'PY'
import torch, transformers, mujoco, tensorflow as tf
from transformers.utils import is_flash_attn_2_available
from libero.libero import benchmark
assert torch.cuda.is_available(), "CUDA not visible: run inside a GPU job"
print("torch", torch.__version__, torch.cuda.get_device_name(0), "| flash_attn usable:", is_flash_attn_2_available())
print("transformers", transformers.__version__, "| tf", tf.__version__, "| mujoco", mujoco.__version__)
print("suites:", sorted(benchmark.get_benchmark_dict()))
PY
