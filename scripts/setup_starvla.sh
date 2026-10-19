#!/bin/bash
# `starvla` env + StarVLA checkout (pinned), on either cluster. Run inside a GPU job so the check sees CUDA:
#   Bridges-2: srun -A cis260349p -p GPU-shared --gpus=l40s-48:1 -t 1:30:00 --pty bash scripts/setup_starvla.sh bridges2
#   DeltaAI:   srun -A $ACCOUNT -p ghx4-interactive --gpus=1 --cpus-per-task=16 --mem=64g -t 1:30:00 bash scripts/setup_starvla.sh deltaai
# torch 2.6 (StarVLA pins torchvision 0.21); attention = sdpa everywhere (configs/starvla_libero_goal.yaml), so no
# flash-attn build. decord has no aarch64 wheel and is not used (video_backend torchvision_av).
# The final check imports the policy server's modules, so a missing dependency shows up here, not in a queued job.
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
# pytorch3d.transforms is imported by the policy server and the LeRobot dataloader (state_action.py). x86: the
# pinned pipablepytorch3d wheel; aarch64 has no wheel, so build it from source (CPU ops are enough for transforms).
if [ "$(uname -m)" = aarch64 ]; then
  pip install --no-build-isolation "git+https://github.com/facebookresearch/pytorch3d.git@v0.7.8"
else
  pip install pipablepytorch3d==0.7.6
fi
pip install -e "$SRC"
# DeepSpeed reads `$CUDA_HOME/bin/nvcc -V` at import and the nodes have no system CUDA; the train job points
# CUDA_HOME at this wheel (matches torch's CUDA).
if [ "$(uname -m)" = aarch64 ]; then pip install "nvidia-cuda-nvcc-cu12==12.6.*"; else pip install "nvidia-cuda-nvcc-cu12==12.4.*"; fi

# Register our QA sets with StarVLA's VLM loader: every llava_jsons/libero_qa_*.json under LANGVLA_QA_ROOT (one
# scripts/make_spatial_qa.py output dir), train and _heldout alike. Re-running replaces the block.
CFG=$SRC/starVLA/dataloader/qwenvl_llavajson/qwen_data_config.py
python - "$CFG" <<'PY'
import sys
p = sys.argv[1]; s = open(p).read()
mark = "\n# lang-vla: spatial QA sets"
s = s.split(mark)[0] + mark + """ (scripts/setup_starvla.sh)
import glob as _glob, os as _os
_qa = _os.environ.get("LANGVLA_QA_ROOT", "./playground/Datasets/libero_qa")
for _f in sorted(_glob.glob(f"{_qa}/llava_jsons/libero_qa_*.json")):
    data_dict[_os.path.basename(_f)[:-5]] = {"annotation_path": _f, "data_path": f"{_qa}/"}
"""
open(p, "w").write(s)
PY

# playground/ symlinks the configs expect.
mkdir -p "$SRC/playground/Pretrained_models" "$SRC/playground/Datasets" "$STORE/ckpt" "$STORE/data"
ln -sfn "$STORE/ckpt/Qwen3-VL-4B-Instruct" "$SRC/playground/Pretrained_models/Qwen3-VL-4B-Instruct"
ln -sfn "$STORE/data/starvla_libero" "$SRC/playground/Datasets/LEROBOT_LIBERO_DATA"

( cd "$SRC" && PYTHONPATH=$SRC python - <<'PY'
import torch, transformers
assert torch.cuda.is_available(), "CUDA not visible: run inside a GPU job"
print("starvla env: torch", torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0), "| transformers", transformers.__version__)
import starVLA.dataloader.qwenvl_llavajson.qwen_data_config  # registry block parses
import deployment.model_server.policy_wrapper                 # everything the policy server imports
PY
)
# The LIBERO-side client runs in the `vla` env and imports StarVLA's websocket client from $SRC.
if [ "$CLUSTER" = bridges2 ]; then
  conda activate "$STORE/envs/vla" && pip install "websockets>=12" msgpack typing_extensions
  PYTHONPATH=$SRC python -c "from deployment.model_server.tools.websocket_policy_client import WebsocketClientPolicy; print('vla env: StarVLA client importable')"
fi
