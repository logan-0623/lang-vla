#!/bin/bash
# `starvla` env + StarVLA checkout (pinned), on either cluster. Run inside a GPU job so the check sees CUDA:
#   Bridges-2: srun -A cis260349p -p GPU-shared --gpus=l40s-48:1 -t 1:30:00 --pty bash scripts/setup_starvla.sh bridges2
#   DeltaAI:   srun -A $ACCOUNT -p ghx4-interactive --gpus=1 --cpus-per-task=16 --mem=64g -t 1:30:00 bash scripts/setup_starvla.sh deltaai
# torch 2.6 (StarVLA pins torchvision 0.21); attention = sdpa everywhere (configs/starvla_libero_goal.yaml), so no
# flash-attn build. decord has no aarch64 wheel and is not used (video_backend torchvision_av).
set -euo pipefail
CLUSTER=${1:?bridges2 or deltaai}
source "$(dirname "$0")/../$CLUSTER/config.sh"
STARVLA_REV=985c2238627481f25476fbee478083812dd82a21    # starVLA_dev, read 2026-10-18
SRC=$STORE/src/starVLA
ENVP=$STORE/envs/starvla
[ -d "$ENVP" ] || conda create -y -p "$ENVP" python=3.10
conda activate "$ENVP"
[ -d "$SRC" ] || git clone -q https://github.com/starVLA/starVLA.git "$SRC"
git -C "$SRC" fetch -q origin "$STARVLA_REV" 2>/dev/null || true
git -C "$SRC" checkout -q "$STARVLA_REV"

if [ "$(uname -m)" = aarch64 ]; then IDX=https://download.pytorch.org/whl/cu126; else IDX=https://download.pytorch.org/whl/cu124; fi
pip install torch==2.6.0 torchvision==0.21.0 --index-url "$IDX"
grep -vE '^(torchvision|decord|eva-decord|pipablepytorch3d)' "$SRC/requirements.txt" > "$STORE/starvla-req.txt"
pip install -r "$STORE/starvla-req.txt"
pip install -e "$SRC"

# Register our QA sets (scripts/make_spatial_qa.py output under $STORE/qa/<name>) with StarVLA's VLM loader.
CFG=$SRC/starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py
grep -q LANGVLA_QA_ROOT "$CFG" || cat >> "$CFG" <<'PY'

# lang-vla: spatial QA sets (scripts/make_spatial_qa.py); LANGVLA_QA_ROOT = one generator output dir.
import os as _os
_qa = _os.environ.get("LANGVLA_QA_ROOT", "./playground/Datasets/libero_qa")
for _s in ("lang", "num", "ctrl"):
    data_dict[f"libero_qa_{_s}"] = {"annotation_path": f"{_qa}/llava_jsons/libero_qa_{_s}.json", "data_path": f"{_qa}/"}
PY

# playground/ symlinks the configs expect.
mkdir -p "$SRC/playground/Pretrained_models" "$SRC/playground/Datasets" "$STORE/ckpt" "$STORE/data"
ln -sfn "$STORE/ckpt/Qwen3-VL-4B-Instruct" "$SRC/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
ln -sfn "$STORE/data/starvla_libero" "$SRC/playground/Datasets/LEROBOT_LIBERO_DATA"

python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "CUDA not visible: run inside a GPU job"
print("starvla env: torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "| transformers", transformers.__version__)
from starVLA.dataloader.qwenvl_llavajson.qwen_data_config import data_dict
assert "libero_qa_lang" in data_dict
PY
# The LIBERO-side client runs in the `vla` env and imports StarVLA's websocket client from $SRC.
if [ "$CLUSTER" = bridges2 ]; then
  conda activate "$STORE/envs/vla" && pip install "websockets>=12" msgpack typing_extensions
  PYTHONPATH=$SRC python -c "from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy; print('vla env: StarVLA client importable')"
fi
