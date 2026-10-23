#!/bin/bash
# RESEARCH_PLAN §11.7 stage 2c (2026-10-23): several donors for one destination, averaged at the same observation
# (the server patches toward the donors' mean action queries), from the batch_1026 stable-grasp states.
#   bash bridges2/batch_1028.sh para      LIBERO-Para paraphrases of s1, s2, s4, s8 (3 each) -> $D/para.json (login node)
#   bash bridges2/batch_1028.sh branch    4 jobs: full-s2, full-cab2 (s2+s4), full-paraA (s8 / s1 + paraphrases),
#                                         full-paraonly (paraphrases only)
#   bash bridges2/batch_1028.sh report
set -euo pipefail
cd "$(dirname "$0")/.."
source bridges2/config.sh
SV=$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt
U=$RESULTS/erase/patch/subspaces2.npz
S=$RESULTS/erase/branch/states
D=$RESULTS/erase/branch3
run() { sbatch --parsable -t 3:00:00 -J $1 --export=ALL,SV_CKPT=$SV,SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py,$2 \
  bridges2/run_starvla.sbatch python scripts/patch_branch.py "${@:3}" --backend starvla --checkpoint SERVER; }
mkdir -p $D
case ${1:-} in
  para)
    [ -e $D/para.json ] && { echo "$D/para.json exists" >&2; exit 1; }
    set +u; use_env vla; set -u                          # LIBERO_CONFIG_PATH: else importing libero prompts
    python - "$PARA/metrics/libero_para_metadata.csv" $D/para.json <<'EOF'
import json, random, sys
from langeval.goal_predicates import load_goals
from langeval.instructions import load_paraphrases
lang = {f"s{i}": g.language for i, g in enumerate(load_goals("libero_goal"))}
rows = load_paraphrases(sys.argv[1], per_category=10**6, seed=7)
rng, out = random.Random(7), {}
for g in ("s1", "s2", "s4", "s8"):
    texts = sorted({r["new_instruction"] for r in rows[lang[g]]} - {lang[g]})
    for k, t in enumerate(rng.sample(texts, 3)):
        out[f"{g}p{k}"] = t
json.dump(out, open(sys.argv[2], "w"), indent=1)
print(json.dumps(out, indent=1))
EOF
    ;;
  branch)
    [ -f $S/states.jsonl ] || { echo "no $S: batch_1026 collect first" >&2; exit 1; }
    [ -f $D/para.json ] || { echo "no $D/para.json: run para first" >&2; exit 1; }
    for spec in "full-s2 n0=s2,n3=s2,n4=s2" "full-cab2 n0=s2+s4,n3=s2+s4,n4=s2+s4" \
                "full-paraA n0=s8+s8p0+s8p1+s8p2,n3=s8+s8p0+s8p1+s8p2,n4=s1+s1p0+s1p1+s1p2" \
                "full-paraonly n0=s8p0+s8p1+s8p2,n3=s8p0+s8p1+s8p2,n4=s1p0+s1p1+s1p2"; do
      set -- $spec
      [ -e $D/$1 ] && { echo "$D/$1 exists; skipped" >&2; continue; }
      export LV_PATCH=$2 LV_PATCH_EXTRA=$D/para.json     # not in --export: Slurm splits its list on commas
      echo "$(run br3-$1 LV_PATCH_U=$U:full:1 branch --states $S --out $D/$1) $1" | tee -a logs/batch_1028.txt
    done ;;
  report)
    python scripts/patch_branch.py report $(ls -d $D/*/) ;;
  *) echo "usage: $0 para | branch | report" >&2; exit 1 ;;
esac
