#!/bin/bash
# Batch 2026-10-17b (~30-40 SU): tests of "an unfamiliar destination noun falls back to the model's default"
# (docs/findings.md), plus the data for the E6 <-> LIBERO-Para join.
#   bash bridges2/batch_1017b.sh       (independent of batch_1017.sh; ids in logs/batch_1017b.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
TS=$(date +%m%d%H%M)
VA=$CKPT/LIBERO-Goal-Pro
OFTG=$CKPT/openvla-7b-oft-finetuned-libero-goal
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 | tee -a logs/batch_1017b.txt; }

for R in vla-adapter oft; do
  [ $R = oft ] && C=$OFTG || C=$VA
  # Nonce words: destination -> "blicket" (source and continuation) / "dax", object -> "blicket", both.
  # Prediction: destination nonce == empty instruction (model default); object nonce after the grasp == orig.
  sub -t 12:00:00 -J e3nonce-$R --export=ALL,ENV=vla,REPO=$R bridges2/run.sbatch \
    python scripts/e3_grasp_branch.py --checkpoint $C --tasks 1,2,4,8,9 --n_states 10 \
    --paraphrase_json configs/nonce_words.json --out $RESULTS/e3/nonce-$R-$TS
  # E6 on every LIBERO-Para paraphrase (2 init states), to join with LIBERO-Para's per-paraphrase success.
  sub -t 8:00:00 -J e6all-$R --export=ALL,ENV=vla,REPO=$R bridges2/run.sbatch \
    python scripts/paraphrase_invariance.py --checkpoint $C --n_init 2 --per_category 100000 --out $RESULTS/e6/$R-all-$TS
done
# LIBERO-Para's own eval for OFT (VLA-Adapter's is done: para/vla-adapter-LIBERO-Goal-Pro-49317676).
sub --array=0-7 -J para-oft --export=ALL,MODEL=oft,CKPT_NAME=openvla-7b-oft-finetuned-libero-goal,NSHARD=8 bridges2/eval_para.sbatch
echo "submitted; ids in logs/batch_1017b.txt"
