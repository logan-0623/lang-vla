#!/bin/bash
# RESEARCH_PLAN §11.9 A2 pilot on LIBERO-Goal (2026-10-24): does abstracting the object name help only after the grasp?
# Full episodes, n0 / n3 / n4, init states 20-39, StarVLA and pi0.5, four conditions:
#   orig        original instruction throughout
#   it-start    "put it on the D" from the start (configs/stage_goals.json a0 / a3 / a4, same predicates)
#   short-after original until a stable grasp of the target, then a short name ("put the bottle on the D")
#   it-after    original until a stable grasp, then "put it on the D"
#   bash bridges2/batch_1033.sh run        8 jobs
#   bash bridges2/batch_1033.sh report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
PI=$CKPT/pi05_libero_finetuned_v044
J=configs/stage_goals.json
D=$RESULTS/stage_a2
SHORT="n0=put the cheese on the plate;n3=put the bottle on the plate;n4=put the bottle on the stove"
IT="n0=put it on the plate;n3=put it on the plate;n4=put it on the stove"
common=(--sets novel --novel_json $J --init_offset 20 --n_init 20)
sv() { sbatch --parsable -t 4:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP= bridges2/run_starvla.sbatch \
  python scripts/eval_goals.py --backend starvla --checkpoint SERVER "${common[@]}" "${@:2}"; }
pi() { sbatch --parsable -t 8:00:00 -J $1 --export=ALL,ENV=lerobot bridges2/run.sbatch \
  python scripts/eval_goals.py --backend lerobot --checkpoint $PI "${common[@]}" "${@:2}"; }
log() { echo "$1 $2" | tee -a logs/batch_1033.txt; }
unset LV_PATCH
mkdir -p $D
case ${1:-} in
  run)
    for m in sv pi; do
      [ -e $D/$m-orig ]        || log "$($m a2-$m-orig --goals n0,n3,n4 --out $D/$m-orig)" $m-orig
      [ -e $D/$m-it-start ]    || log "$($m a2-$m-itstart --goals a0,a3,a4 --out $D/$m-it-start)" $m-it-start
      [ -e $D/$m-short-after ] || log "$($m a2-$m-short --goals n0,n3,n4 --switch "$SHORT" --out $D/$m-short-after)" $m-short-after
      [ -e $D/$m-it-after ]    || log "$($m a2-$m-itafter --goals n0,n3,n4 --switch "$IT" --out $D/$m-it-after)" $m-it-after
    done ;;
  report)
    for r in $(ls -d $D/*/); do
      [ -f $r/summary.json ] || { echo "$(basename $r) not finished"; continue; }
      echo "== $(basename $r)"; python -c "
import json
eps = [json.loads(l) for l in open('$r/episodes.jsonl')]
for g, s in json.load(open('$r/summary.json')).items():
    sw = sum(e.get('switch_step') is not None for e in eps if e['goal'] == g)
    print(f\"  {g} success {s['success']:.2f} (n={s['n']})  switched {sw}  executed {s['executed']}\")"
    done
    python scripts/patch_branch.py moved $(ls -d $D/*/) ;;
  *) echo "usage: $0 run | report" >&2; exit 1 ;;
esac
