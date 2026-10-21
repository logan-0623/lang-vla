#!/bin/bash
# Batch 2026-10-22: magnitude-matched erasure (RESEARCH_PLAN §11.6.3b, M1-M3 written before these results). In
# batch_1021 every concept eraser removed 3-4x more of ||x - mu|| than its variance-matched random control (geometry
# is most of the action-query variance), so each concept is erased again only partially: alpha = mean random
# fraction / concept fraction, from the batch_1021 er-fit log: eef .415/2/.622, dist .431/2/.685, geom .453/2/.824,
# grip .329/2/.615. ~2 GPU-h (batch_1021 runs took 10-30 min each).
#   git pull ; bash bridges2/batch_1022.sh       (afterwards: bash bridges2/batch_1022.sh report)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
E=$RESULTS/erase
RUNS=${RUNS:-"eef:0.334 dist:0.315 geom:0.275 grip:0.267"}
if [ "${1:-}" = report ]; then
  for n in none eef eef-matched eef-random0 eef-random1 dist dist-matched dist-random0 dist-random1 \
           geom geom-matched geom-random0 geom-random1 grip grip-matched grip-random0 grip-random1; do
    [ -f $E/$n/summary.json ] || continue
    python -c "import json,sys; s={k: v for k, v in json.load(open(sys.argv[1])).items() if k[0] == \"s\"}; print(f'{sys.argv[2]:<13}', f'{sum(v[\"success\"]*v[\"n\"] for v in s.values())/sum(v[\"n\"] for v in s.values()):.3f}', ' '.join(f'{v[\"success\"]:.1f}' for v in s.values()))" $E/$n/summary.json $n
  done
  exit
fi
[ -f $E/erasers.npz ] || { echo "missing $E/erasers.npz (batch_1021)" >&2; exit 1; }
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 >> logs/batch_1022.txt; echo "$id"; }
for r in $RUNS; do
  c=${r%%:*}
  [ -e $E/$c-matched ] && { echo "$E/$c-matched exists" >&2; exit 1; }
  sub -t 4:00:00 -J er-$c-matched --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,LV_ERASE=$E/erasers.npz:$r \
    bridges2/run_starvla.sbatch python scripts/eval_goals.py --backend starvla --checkpoint SERVER --sets seen \
    --n_init 10 --out $E/$c-matched > /dev/null
done
echo "submitted matched erasure runs ($RUNS); ids in logs/batch_1022.txt"
echo "binding re-analysis on this login node (CPU, no SU):"
python scripts/erase_action.py binding --rec $E/rec.f16 --labels $E/rec.jsonl --run_dir $E/none | tee logs/er-binding2.txt
