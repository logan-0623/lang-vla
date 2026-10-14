# Sourced by every bridges2/ script. Bridges-2 is x86_64, so upstream versions install unchanged.
: "${ACCOUNT:=cis260349p}"
export PROJ_DIR=${PROJECT:-/ocean/projects/$ACCOUNT/$USER}       # Bridges-2 sets $PROJECT on login
export LANGVLA_ROOT=${LANGVLA_ROOT:-$PROJ_DIR/lang-vla}          # code
export STORE=${STORE:-$PROJ_DIR/store}                           # envs, weights, data, results ($HOME is only 25 GB)
export CONDA_DIR=$PROJ_DIR/miniforge3 CONDA_PKGS_DIRS=$STORE/conda-pkgs
export HF_HOME=$STORE/hf PIP_CACHE_DIR=$STORE/pip-cache
export DATA_ROOT=$STORE/data/libero CKPT=$STORE/ckpt RESULTS=$STORE/results
export VLAA=$LANGVLA_ROOT/Pion/VLA/VLAAdapter
export OFT=$STORE/src/openvla-oft PARA=$STORE/src/LIBERO-Para LEROBOT_SRC=$STORE/src/lerobot
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false WANDB_MODE=offline

# Pinned external code (commits read 2026-10-13)
OFT_REV=e4287e94541f459edc4feabc4e181f537cd569a8
PARA_REV=86cfb7b9d214bdc8d7bb4e2e96a4834a37244eec
LEROBOT_REV=ae2b57f12223c8c5184fcd814eb63cd9ce7de203

# Two envs: `vla` (VLA-Adapter + OpenVLA-OFT, torch 2.2) and `lerobot` (pi0.5 / SmolVLA / X-VLA, py3.12).
# Each env ships its own `libero` package, so each gets its own LIBERO config dir.
use_env() {
  export LIBERO_CONFIG_PATH=$STORE/libero-config/$1
  conda activate "$STORE/envs/$1"
}
[ -f "$CONDA_DIR/etc/profile.d/conda.sh" ] && source "$CONDA_DIR/etc/profile.d/conda.sh"
mkdir -p "$LANGVLA_ROOT/logs" 2>/dev/null || true
