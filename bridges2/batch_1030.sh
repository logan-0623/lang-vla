#!/bin/bash
# RESEARCH_PLAN §11.8 B, stage 3 (frozen 2026-10-23): held-out combinations (configs/heldout_goals.json), full
# episodes from init states 20-39 (development used 0-19 for full episodes).
#   bash bridges2/batch_1030.sh feasible          1 short job: end-state feasibility of h0-h3 (scripts/feasible.py)
#   bash bridges2/batch_1030.sh run h0,h1,h2,h3   3 jobs: none / it (after a stable grasp: "put it ...") /
#                                                 seen (after the grasp: the seen goal with that destination; h2, h3)
#   bash bridges2/batch_1030.sh report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
D=$RESULTS/stage3
J=configs/heldout_goals.json
run() { sbatch --parsable -t 4:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP= bridges2/run_starvla.sbatch \
  python scripts/eval_goals.py --backend starvla --checkpoint SERVER --sets novel --novel_json $J \
  --init_offset 20 --n_init 20 "${@:2}"; }
log() { echo "$1 $2" | tee -a logs/batch_1030.txt; }
mkdir -p $D
unset LV_PATCH
case ${1:-} in
  feasible)
    log "$(sbatch --parsable -t 0:30:00 -J s3-feasible --export=ALL,ENV=vla bridges2/run.sbatch \
      python scripts/feasible.py --goals_json $J)" feasible ;;
  run)
    G=${2:?give the feasible goals, e.g. h0,h2,h3}
    sw() { python - "$J" "$1" "$G" <<'PY'
import json, sys
spec = {g["id"]: g for g in json.load(open(sys.argv[1]))["goals"]}
print(";".join(f"{g}={spec[g][sys.argv[2]]}" for g in sys.argv[3].split(",") if spec[g][sys.argv[2]]))
PY
    }
    [ -e $D/none ] || log "$(run s3-none --goals $G --out $D/none)" none
    [ -e $D/it ] || log "$(run s3-it --goals $G --switch "$(sw switch_it)" --out $D/it)" it
    S=$(sw switch_seen)
    SG=$(echo "$S" | tr ';' '\n' | cut -d= -f1 | paste -sd,)
    [ -n "$S" ] && { [ -e $D/seen ] || log "$(run s3-seen --goals $SG --switch "$S" --out $D/seen)" seen; } ;;
  report)
    for r in none it seen; do
      [ -f $D/$r/summary.json ] || { echo "$r not finished"; continue; }
      echo "== $r"; python -c "
import json
eps = [json.loads(l) for l in open('$D/$r/episodes.jsonl')]
for g, s in json.load(open('$D/$r/summary.json')).items():
    sw = sum(e.get('switch_step') is not None for e in eps if e['goal'] == g)
    print(f\"  {g} success {s['success']:.2f} (n={s['n']})  switched {sw}  executed {s['executed']}\")"
    done
    python scripts/patch_branch.py moved $(ls -d $D/*/) ;;
  *) echo "usage: $0 feasible | run <goals> | report" >&2; exit 1 ;;
esac
