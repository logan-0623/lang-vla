#!/bin/bash
# `lerobot` env: pi0.5 / SmolVLA / X-VLA through LeRobot (needs Python >= 3.12, torch >= 2.7).
#   srun -A cis260349p -p GPU-shared --gpus=l40s-48:1 -t 1:00:00 --pty bash bridges2/setup_lerobot.sh
set -euo pipefail
source "$(dirname "$0")/config.sh"
[ -d "$STORE/envs/lerobot" ] || conda create -y -p "$STORE/envs/lerobot" python=3.12
use_env lerobot
[ -d "$LEROBOT_SRC" ] || git clone -q https://github.com/huggingface/lerobot.git "$LEROBOT_SRC"
git -C "$LEROBOT_SRC" checkout -q "$LEROBOT_REV"
pip install -e "$LEROBOT_SRC[libero,pi,smolvla,xvla]" || pip install -e "$LEROBOT_SRC[libero]"
pip install pytest scikit-learn scipy pandas
pip install -e "$LANGVLA_ROOT"   # langeval

# hf-libero: point the config at the installed package (found without importing libero.libero, which would prompt).
L=$(python -c "import importlib.util, os; print(os.path.dirname(importlib.util.find_spec('libero.libero').origin))")
mkdir -p "$LIBERO_CONFIG_PATH" && cat > "$LIBERO_CONFIG_PATH/config.yaml" <<YAML
benchmark_root: $L
bddl_files: $L/bddl_files
init_states: $L/init_files
datasets: $L/../datasets
assets: $L/assets
YAML
python -c "import torch, lerobot; from libero.libero import benchmark; assert torch.cuda.is_available(); print('lerobot', lerobot.__version__, torch.__version__, sorted(benchmark.get_benchmark_dict()))"
