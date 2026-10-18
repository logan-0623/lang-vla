# Bridges-2 (cis260349p): zero-training experiments

Order matters: each step is a gate for the next. Submit from the repo root (`$PROJECT/lang-vla`), because the
`logs/` paths in the `#SBATCH -o` lines are relative to it. Setup creates `logs/`.

| # | Command | What it checks | ~SU (L40S, 1 SU/GPU-h) |
|---|---|---|---|
| 0 | `git clone https://github.com/logan-0623/lang-vla.git $PROJECT/lang-vla && cd $PROJECT/lang-vla && bash scripts/setup_pion.sh` | Pion = upstream + patches | 0 |
| 1 | `srun -A cis260349p -p GPU-shared --gpus=l40s-48:1 -t 1:00:00 --pty bash bridges2/setup_vla.sh` | `vla` env + LIBERO-Para | 1 |
| 2 | same with `bridges2/setup_lerobot.sh` | `lerobot` env | 1 |
| 3 | `bash bridges2/download_assets.sh` (login node) | pinned checkpoints | 0 |
| 4 | `sbatch bridges2/smoke.sbatch` | EGL, both forward passes, 10 rollouts, LeRobot pi0.5 | ≤1 |
| 5 | `sbatch bridges2/bench_procs.sbatch` | rollouts/GPU-hour for 1/2/4/8 processes per GPU | ≤4 |
| 6 | E0, see below | reproduce official success rates | measured in 5 |
| 7 | E6, see below | layer-wise paraphrase invariance | ≈1–2 per model |

E0 (stop and debug if a number misses the official one beyond its CI):
```bash
sbatch --export=ALL,MODEL=vla-adapter,CKPT_NAME=LIBERO-Goal-Pro,SUITE=libero_goal,PROCS=4 bridges2/eval_libero.sbatch
sbatch --export=ALL,MODEL=oft,CKPT_NAME=openvla-7b-oft-finetuned-libero-goal,SUITE=libero_goal bridges2/eval_libero.sbatch
sbatch --export=ALL,MODEL=oft,CKPT_NAME=openvla-7b-oft-finetuned-libero-spatial-object-goal-10,SUITE=libero_goal bridges2/eval_libero.sbatch
sbatch --export=ALL,POLICY=pi05_libero_finetuned_v044,SUITE=libero_goal bridges2/eval_lerobot.sbatch
```
Each `eval_libero` run writes `summary.json` with success rate, errored episodes (counted as failures upstream,
reported separately here) and rollouts/GPU-hour.

E1 instruction modes (VLA-Adapter; reduce TRIALS for the sweep):
```bash
for m in orig empty other shuffle; do
  sbatch --export=ALL,MODEL=vla-adapter,CKPT_NAME=LIBERO-Goal-Pro,SUITE=libero_goal,MODE=$m,TRIALS=20,PROCS=4 bridges2/eval_libero.sbatch
done
```

E6 paraphrase invariance (forward passes only):
```bash
source bridges2/config.sh
sbatch --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
  python scripts/paraphrase_invariance.py --checkpoint $CKPT/LIBERO-Goal-Pro --out $RESULTS/e6/vla-adapter-$(date +%m%d%H%M)
sbatch --export=ALL,ENV=vla,REPO=oft bridges2/run.sbatch \
  python scripts/paraphrase_invariance.py --checkpoint $CKPT/openvla-7b-oft-finetuned-libero-goal --out $RESULTS/e6/oft-goal-$(date +%m%d%H%M)
```

E3 grasp-state branching (one task per job keeps each job short; drop `--tasks` for all of them):
```bash
source bridges2/config.sh
for t in 1 2 3 4 6 8 9; do   # the pick-and-place tasks in libero_goal benchmark order
  sbatch -t 12:00:00 --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
    python scripts/e3_grasp_branch.py --checkpoint $CKPT/LIBERO-Goal-Pro --tasks $t --out $RESULTS/e3/vla-adapter-t$t-$(date +%m%d%H%M)
done
```
Pilot first with `--n_states 3 --max_attempts 10` on one task: check `determinism.json` says `identical: true`,
and look at a few branches before spending SU. Task ids follow the benchmark order (`suite.get_task(i).language`);
without `--tasks`, non-pick-and-place tasks (open drawer, turn on stove, push the plate) are skipped automatically.

LIBERO-Para success rates with its own scripts (sharded; one episode per paraphrase from init state 0):
```bash
sbatch --array=0-7 --export=ALL,MODEL=vla-adapter,CKPT_NAME=LIBERO-Goal-Pro,NSHARD=8 bridges2/eval_para.sbatch
```

GPU choice: L40S (24 CPU cores per GPU, 1 SU/GPU-h). H100 costs 2 SU/GPU-h for no gain on CPU-bound rollouts;
override with `sbatch --gpus=h100-80:1 ...` if the 3 L40S nodes are congested. Avoid V100: no bf16.

## Spatial-language transfer study (RESEARCH_PLAN §11)

StarVLA runs in its own `starvla` env behind StarVLA's websocket policy server; the LIBERO client
(`--backend starvla`) runs in the `vla` env, so every model uses the same simulator build. `run_starvla.sbatch` starts
the server, waits for it, then runs one client command; the literal argument `SERVER` becomes its host:port.

| # | Command | What it checks | ~SU |
|---|---|---|---|
| 8 | `srun -A cis260349p -p GPU-shared --gpus=l40s-48:1 -t 1:30:00 --pty bash scripts/setup_starvla.sh bridges2` | `starvla` env, QA sets registered, client importable from `vla` | 1.5 |
| 9 | `bash scripts/download_starvla_assets.sh bridges2` (login node) | Qwen3-VL-4B-Instruct + released `Qwen3-VL-OFT-LIBERO-4in1` (~20 GB) | 0 |
| 10 | `bash bridges2/batch_1018b.sh` | StarVLA E3 pilot (adapter determinism), QA collection (its four novel-goal jobs: cancel, see 11) | ~5–10 |
| 11 | `bash bridges2/batch_1018c.sh` | novel-goal baseline (N0) for 4 models, StarVLA Goal success | ~10–15 |
| 12 | `python scripts/make_spatial_qa.py build --dir $STORE/qa/goal-v1 --overlay 12 --px_convention <c>` (login node, `vla` env) | look at `overlay/*.png`: the colour sitting on the objects is the convention; rebuild with it | 0 |

Gates, before any StarVLA training:
- `novel-starvla4in1` (batch_1018c): the 10 seen tasks (`s0`–`s9` in `summary.json`) should be close to the published Goal 96.2%.
- `e3-starvla-pilot`: `restore_obs_equal: true` and `orig_vs_orig_rep` identical on every state.
- `qa-goal-v1` then step 12: the overlay decides the image-coordinate convention (red none / blue row / lime col /
  yellow both). Do not train on QA built before this check.

Novel goals alone, for any backend:
```bash
sbatch --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch \
  python scripts/eval_goals.py --checkpoint $CKPT/LIBERO-Goal-Pro --sets novel --n_init 50 --out $RESULTS/novel/va-$(date +%m%d%H%M)
sbatch --export=ALL,SV_CKPT=/path/to/steps_N_pytorch_model.pt bridges2/run_starvla.sbatch \
  python scripts/eval_goals.py --backend starvla --checkpoint SERVER --n_init 50 --out $RESULTS/novel/<arm>-$(date +%m%d%H%M)
```
The QA data is generated here and trained on at DeltaAI: copy `$STORE/qa/goal-v1` to DeltaAI's `$STORE/qa/goal-v1`.
