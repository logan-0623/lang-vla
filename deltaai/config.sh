# Sourced by every deltaai/ script. Only ACCOUNT normally needs editing (`accounts` on a login node shows it).
: "${ACCOUNT:=CHANGE_ME-dtai-gh}"
PROJ=${ACCOUNT%%-*}
export LANGVLA_ROOT=${LANGVLA_ROOT:-/projects/$PROJ/$USER/lang-vla}   # code (not purged)
export STORE=${STORE:-/work/nvme/$PROJ/$USER/lang-vla}               # env, data, weights, outputs
export ENV_PREFIX=$STORE/env          # conda env has ~100k files: keep it out of $HOME (750k file quota)
export HF_HOME=$STORE/hf PIP_CACHE_DIR=$STORE/pip-cache
export DATA_ROOT=$STORE/data/libero   # RLDS: libero_{goal,spatial,...}_no_noops/
export VLAA=$LANGVLA_ROOT/Pion/VLA/VLAAdapter
export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export TF_CPP_MIN_LOG_LEVEL=2 TOKENIZERS_PARALLELISM=false
export WANDB_MODE=${WANDB_MODE:-offline}   # `wandb sync` later from a login node

module load python/miniforge3_pytorch/2.7.0 >/dev/null 2>&1 || module load python/miniforge3_pytorch
source "$(conda info --base)/etc/profile.d/conda.sh"
[ -d "$ENV_PREFIX" ] && conda activate "$ENV_PREFIX"
