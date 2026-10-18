#!/bin/bash
# Batch 2026-10-18: pi0.5 (validated by pilot 49400999) on the pre-registered tests, as a third, out-of-sample model:
# H1' (E4 fixed taus), H1'' (E4 placement fractions), H3' (nonce words). Same tasks and designs as the other two.
#   bash bridges2/batch_1018.sh      (ids in logs/batch_1018.txt)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
TS=$(date +%m%d%H%M)
P=$CKPT/pi05_libero_finetuned_v044
L="--backend lerobot --checkpoint $P"
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 | tee -a logs/batch_1018.txt; }

for t in 1 2 4 8 9; do
  sub -t 12:00:00 -J e4-pi05-t$t --export=ALL,ENV=lerobot bridges2/run.sbatch \
    python scripts/e4_switch_time.py $L --tasks $t --out $RESULTS/e4/pi05-t$t-$TS
  sub -t 12:00:00 -J e4frac-pi05-t$t --export=ALL,ENV=lerobot bridges2/run.sbatch \
    python scripts/e4_switch_time.py $L --tasks $t --taus 0 --post_fracs 0.25,0.5,0.75 \
    --out $RESULTS/e4/frac-pi05-t$t-$TS
done
sub -t 12:00:00 -J e3nonce-pi05 --export=ALL,ENV=lerobot bridges2/run.sbatch \
  python scripts/e3_grasp_branch.py $L --tasks 1,2,4,8,9 --n_states 10 \
  --paraphrase_json configs/nonce_words.json --out $RESULTS/e3/nonce-pi05-$TS
echo "submitted; ids in logs/batch_1018.txt"
