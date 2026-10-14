#!/bin/bash
# Pinned checkpoints (revisions read from the HF API on 2026-10-13). Login node is fine; run inside either env.
# No RLDS data: nothing on Bridges-2 trains, and E6 takes frames from the simulator.
set -euo pipefail
source "$(dirname "$0")/config.sh"
use_env vla
dl() { local repo=$1 rev=$2; shift 2; huggingface-cli download "$repo" --revision "$rev" --local-dir "$CKPT/${repo#*/}" --quiet "$@"; }

# Tier 1
dl VLA-Adapter/LIBERO-Goal-Pro                                  2ae84b8275a0c35c1b6751573a71468e1c080825
dl moojink/openvla-7b-oft-finetuned-libero-goal                 c2d0f9fbbd82674683b397ff923168a12f6a307b --exclude "lora_adapter/*"
dl moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10 638918f3d1c2e43a39a8a20772bdb8b91835e4b7 --exclude "lora_adapter/*"
dl lerobot/pi05_libero_finetuned_v044                           7152f13e34c6c115d808aea0e5d08aca74eac14b
# H≈0 controls (LIBERO-Spatial)
dl VLA-Adapter/LIBERO-Spatial-Pro                               cb023c6771fc62ea729d218f09ac0583c14f59d4
dl moojink/openvla-7b-oft-finetuned-libero-spatial              6d0231af0e48c5985f1ff86908f4674b84bc049b --exclude "lora_adapter/*"
# Tier 2 (only if Tier 1 conclusions are clear): uncomment
# dl HuggingFaceVLA/smolvla_libero                              6721902bc4d61e50a3bfdb11dfb4cb626f05d102
# dl lerobot/xvla-libero                                        12e8783e996944f5c97e490d37d4c145484ed70a
du -sh "$CKPT"/*
