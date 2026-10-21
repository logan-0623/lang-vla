#!/bin/bash
# Batch 2026-10-21: does 4in1 *use* the gripper-object geometry? Linear erasure at the action-head input
# (scripts/erase_action.py, RESEARCH_PLAN §11.6.3b; E1-E5 written before these results). ~26 GPU-h.
#   er-record : 4in1 on the 10 seen goals x 10 inits, action queries + observation labels recorded (= the baseline)
#   er-fit    : erasers (eef 3, dist 4, geom 15, grip 1) and 2 variance-matched random subspaces of each
#   er-<name> : the same 100 episodes with one eraser applied
#   git pull ; bash bridges2/batch_1021.sh       (afterwards: bash bridges2/batch_1021.sh report)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
E=$RESULTS/erase
NAMES=$(for c in eef dist geom grip; do echo -n "$c $c-random0 $c-random1 "; done)
EVAL="python scripts/eval_goals.py --backend starvla --checkpoint SERVER --sets seen --n_init 10"
if [ "${1:-}" = report ]; then
  tail -4 logs/er-fit-*.out
  for n in none $NAMES; do
    python -c "import json,sys; s=json.load(open(sys.argv[1])); print(f'{sys.argv[2]:<13}', f'{sum(v[\"success\"]*v[\"n\"] for v in s.values())/sum(v[\"n\"] for v in s.values()):.3f}', ' '.join(f'{v[\"success\"]:.1f}' for v in s.values()))" $E/$n/summary.json $n
  done
  exit
fi
[ -e "$E" ] && { echo "$E exists; move it away first" >&2; exit 1; }
mkdir -p $E
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 >> logs/batch_1021.txt; echo "$id"; }
WRAP=$LANGVLA_ROOT/scripts/erase_action.py

r=$(sub -t 4:00:00 -J er-record --export=ALL,SV_CKPT=$SV,SV_WRAP=$WRAP,LV_RECORD=$E/rec.f16,LV_LABELS=$E/rec.jsonl \
  bridges2/run_starvla.sbatch $EVAL --out $E/none)
f=$(sub -t 0:30:00 -J er-fit --dependency=afterok:$r --export=ALL,ENV=starvla bridges2/run.sbatch \
  python scripts/erase_action.py fit --rec $E/rec.f16 --labels $E/rec.jsonl --out $E/erasers.npz)
for n in $NAMES; do
  sub -t 4:00:00 -J er-$n --dependency=afterok:$f --export=ALL,SV_CKPT=$SV,SV_WRAP=$WRAP,LV_ERASE=$E/erasers.npz:$n \
    bridges2/run_starvla.sbatch $EVAL --out $E/$n > /dev/null
done
echo "submitted $r -> $f -> 12 erasure runs; ids in logs/batch_1021.txt"
