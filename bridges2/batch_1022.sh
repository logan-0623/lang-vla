#!/bin/bash
# Batch 2026-10-22: magnitude-matched erasure (RESEARCH_PLAN §11.6.3b, M1-M3 written before these results). In
# batch_1021 every concept eraser removed 3-4x more of ||x - mu|| than its variance-matched random control (geometry
# is most of the action-query variance), so each concept is erased again only partially: alpha = mean random
# fraction / concept fraction, recomputed with the served erasers.npz on the seen calls (`alphas`; the er-fit log
# numbers came from erasers fit on train episodes only). ~2 GPU-h (batch_1021 runs took 10-30 min each).
#   git pull ; bash bridges2/batch_1022.sh       (afterwards: bash bridges2/batch_1022.sh report)
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
E=$RESULTS/erase
if [ "${1:-}" = report ]; then
  cat logs/er-alphas.txt
  for n in none eef eef-matched eef-random0 eef-random1 dist dist-matched dist-random0 dist-random1 \
           geom geom-matched geom-random0 geom-random1 grip grip-matched grip-random0 grip-random1; do
    [ -f $E/$n/summary.json ] || continue
    python -c "import json,sys; s={k: v for k, v in json.load(open(sys.argv[1])).items() if k[0] == \"s\"}; print(f'{sys.argv[2]:<13}', f'{sum(v[\"success\"]*v[\"n\"] for v in s.values())/sum(v[\"n\"] for v in s.values()):.3f}', ' '.join(f'{v[\"success\"]:.1f}' for v in s.values()))" $E/$n/summary.json $n
  done
  exit
fi
[ -f $E/erasers.npz ] || { echo "missing $E/erasers.npz (batch_1021)" >&2; exit 1; }
for c in eef dist geom grip; do [ -e $E/$c-matched ] && { echo "$E/$c-matched exists" >&2; exit 1; }; done
# CPU steps on this login node first (no SU); any failure stops the script before anything is submitted.
echo "binding re-analysis:"
python scripts/erase_action.py binding --rec $E/rec.f16 --labels $E/rec.jsonl --run_dir $E/none | tee logs/er-binding2.txt
echo "dose calibration with the served erasers:"
python scripts/erase_action.py alphas --rec $E/rec.f16 --labels $E/rec.jsonl --run_config $E/none/run_config.json \
  --erasers $E/erasers.npz --out $E/alphas.json | tee logs/er-alphas.txt
RUNS=${RUNS:-$(python -c "import json,sys; print(' '.join(f'{c}:{a}' for c, a in json.load(open(sys.argv[1])).items()))" $E/alphas.json)}
sub() { local id; id=$(sbatch --parsable "$@"); echo "$id $*" | cut -c1-200 >> logs/batch_1022.txt; echo "$id"; }
for r in $RUNS; do
  c=${r%%:*}
  sub -t 4:00:00 -J er-$c-matched --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,LV_ERASE=$E/erasers.npz:$r \
    bridges2/run_starvla.sbatch python scripts/eval_goals.py --backend starvla --checkpoint SERVER --sets seen \
    --n_init 10 --out $E/$c-matched > /dev/null
done
echo "submitted matched erasure runs ($RUNS); ids in logs/batch_1022.txt. Do not rerun the whole script."
