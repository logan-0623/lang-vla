#!/bin/bash
# Pinned downloads (revisions read from the HF API on 2026-10-10). Login node is fine.
set -euo pipefail
source "$(dirname "$0")/config.sh"
dl() { huggingface-cli download "$@" --quiet; }

# VLM backbone, where finetune_pion.py expects it
dl Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b --revision 5cfd2cc6da00c06e0be7abf35d43ec792d8e9498 \
   --local-dir "$VLAA/pretrained_models/prism-qwen25-extra-dinosiglip-224px-0_5b"

# Official checkpoints (Goal = main subject, Spatial = H≈0 control)
for spec in LIBERO-Goal-Pro:2ae84b8275a0c35c1b6751573a71468e1c080825 LIBERO-Goal:d1aa0654ccac22cb7e45a55f21692433e9ae7e63 \
            LIBERO-Spatial-Pro:cb023c6771fc62ea729d218f09ac0583c14f59d4 LIBERO-Spatial:45caa9bca50d3ea6f8e804fe04c65abac1b69717; do
  dl "VLA-Adapter/${spec%%:*}" --revision "${spec##*:}" --local-dir "$STORE/ckpt/${spec%%:*}"
done

# RLDS data: only the suites this project uses (repo dirs are already named libero_*_no_noops)
dl openvla/modified_libero_rlds --repo-type dataset --revision 6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551 \
   --include "libero_goal_no_noops/*" "libero_spatial_no_noops/*" --local-dir "$DATA_ROOT"
ls "$DATA_ROOT"/*/
