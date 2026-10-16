#!/bin/bash
# Batch of 2026-10-16: every experiment whose code has run on Bridges-2, submitted in one go (~50-80 SU).
#   bash bridges2/batch_1016.sh        (login node, repo root)
# Job names say what each job is; logs land in logs/<name>-<jobid>.out. Job ids are appended to logs/batch_1016.txt.
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
TS=$(date +%m%d%H%M)
VA=$CKPT/LIBERO-Goal-Pro
OFTG=$CKPT/openvla-7b-oft-finetuned-libero-goal
OFTJ=$CKPT/openvla-7b-oft-finetuned-libero-spatial-object-goal-10
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 | tee -a logs/batch_1016.txt; }

# E3 full: 7 pick-and-place tasks x {VLA-Adapter, OFT}, LIBERO-Para paraphrases, 20 grasp states per source.
for t in 1 2 3 4 6 8 9; do
  for R in vla-adapter oft; do
    [ $R = oft ] && C=$OFTG || C=$VA
    sub -t 12:00:00 -J e3-$R-t$t --export=ALL,ENV=vla,REPO=$R bridges2/run.sbatch \
      python scripts/e3_grasp_branch.py --checkpoint $C --tasks $t --out $RESULTS/e3/para-$R-t$t-$TS
  done
done

# E3 on the bddl wording ("... top of the drawer"), tasks 2 and 4, 20 states: the E1b finding.
for R in vla-adapter oft; do
  [ $R = oft ] && C=$OFTG || C=$VA
  sub -t 12:00:00 -J e3w-$R --export=ALL,ENV=vla,REPO=$R bridges2/run.sbatch \
    python scripts/e3_grasp_branch.py --checkpoint $C --tasks 2,4 \
    --paraphrase_json configs/bddl_wording_paraphrases.json --out $RESULTS/e3/wording-$R-$TS
done

# E6 with 10 init states (CIs), plus the OFT four-suite checkpoint (G6: same architecture, more language entropy).
sub -t 8:00:00 -J e6-vla-adapter --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
  python scripts/paraphrase_invariance.py --checkpoint $VA --n_init 10 --out $RESULTS/e6/vla-adapter-n10-$TS
sub -t 8:00:00 -J e6-oft-goal --export=ALL,ENV=vla,REPO=oft bridges2/run.sbatch \
  python scripts/paraphrase_invariance.py --checkpoint $OFTG --n_init 10 --out $RESULTS/e6/oft-goal-n10-$TS
sub -t 8:00:00 -J e6-oft-joint --export=ALL,ENV=vla,REPO=oft bridges2/run.sbatch \
  python scripts/paraphrase_invariance.py --checkpoint $OFTJ --n_init 10 --out $RESULTS/e6/oft-joint-n10-$TS

# E0: OFT four-suite checkpoint on Goal; the two Spatial (H~0) controls.
sub -J e0-oft-joint-goal --export=ALL,MODEL=oft,CKPT_NAME=openvla-7b-oft-finetuned-libero-spatial-object-goal-10,SUITE=libero_goal \
  bridges2/eval_libero.sbatch
sub -J e0-va-spatial --export=ALL,MODEL=vla-adapter,CKPT_NAME=LIBERO-Spatial-Pro,SUITE=libero_spatial,PROCS=2 \
  bridges2/eval_libero.sbatch
sub -J e0-oft-spatial --export=ALL,MODEL=oft,CKPT_NAME=openvla-7b-oft-finetuned-libero-spatial,SUITE=libero_spatial \
  bridges2/eval_libero.sbatch

# E1 instruction modes (success rate only; the confusion matrix needs eval_goal_matrix.py), VLA-Adapter, 20 trials.
for m in empty other shuffle; do
  sub -J e1-va-$m --export=ALL,MODEL=vla-adapter,CKPT_NAME=LIBERO-Goal-Pro,SUITE=libero_goal,MODE=$m,TRIALS=20,PROCS=2 \
    bridges2/eval_libero.sbatch
done

# LIBERO-Para's own eval (4092 paraphrases, 8 shards) for VLA-Adapter: per-paraphrase success, to test whether
# E6's R_action predicts failure. First run of eval_para.sbatch.
sub --array=0-7 -J para-va --export=ALL,MODEL=vla-adapter,CKPT_NAME=LIBERO-Goal-Pro,NSHARD=8 bridges2/eval_para.sbatch

echo "submitted; ids in logs/batch_1016.txt"
