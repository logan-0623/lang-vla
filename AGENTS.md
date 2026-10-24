# AGENTS.md: operating manual for AI agents in `lang-vla`

This is a research codebase: evaluation, interpretability and causal interventions for vision-language-action (VLA)
policies on the LIBERO simulator. Most code runs only on a GPU cluster (PSC Bridges-2). The human runs cluster
commands and pastes the output back, so the agent's job is usually to write or change code, hand over exact
commands, and interpret what comes back.

Every path in this file existed when it was written (2026-10-24, after commit `acbc894`). Commands marked
**[verified]** are copied from scripts or docstrings that have run on the cluster. Commands marked **[template]**
need values filled in. Anything marked **[unverified]** could not be confirmed from the repository.

---

## 0. Quick start (first five minutes)

1. `git status --short` and `git log --oneline -10`. Expect uncommitted changes by the human or other agents,
   typically in `docs/findings.md`, `docs/research-briefing.html` and `bridges2/config.sh`, plus untracked
   `docs/EXPERIMENT_REVIEW_CVPR2027.md`. **Never commit, revert or overwrite someone else's changes.**
2. Read the current state of the research:
   - `docs/RESEARCH_PLAN.md`: the top banner, then §11.7 (main line) and §11.8 (next steps). Written in Chinese.
     It is the source of truth for hypotheses, pre-registered rules and results.
   - `docs/findings.md` §8: a plain-language summary of the current results (Chinese).
   - `docs/PAPER_OUTLINE.md`: claims, the evidence for each, and what is still missing.
   - `docs/DECISIONS.md`: engineering decisions and upstream traps (English).
3. Find the newest experiment: the highest-numbered `bridges2/batch_MMDD*.sh`. Its header says what it runs, and
   `logs/batch_MMDD.txt` on the cluster lists the submitted job ids.
4. Before editing a script, read its module docstring; most scripts document their subcommands and an example
   invocation there. Then grep for its callers in `bridges2/` and `tests/`.
5. Run the local tests (no GPU, no LIBERO): `.venv/bin/python -m pytest -q tests/` **[verified]**. Two tests
   need Pillow, which the local `.venv` lacks, so expect them to fail locally; they pass on the cluster.

---

## 1. What the repository is for

**Main research question** (RESEARCH_PLAN §11.7): *spatial knowledge already exists in the VLM, so why does the
robot still fail to turn it into the correct language-conditioned action?* The test bed is novel object × destination
instructions in the shared LIBERO-Goal scene, for example "put the wine bottle on the plate", where every object
and destination was seen in training but never in that combination.

**Current scope** (2026-10-24):
- Main model: StarVLA-OFT (Qwen3-VL-4B), released `Qwen3-VL-OFT-LIBERO-4in1` checkpoint `steps_50000`.
- Replication on π0.5 (LeRobot) in progress (`bridges2/batch_1031.sh`).
- Phase 1 (§0–§10 of the plan) covered VLA-Adapter and OpenVLA-OFT. It is complete and now serves as the
  behavioural baseline.
- The training line (StarVLA QA arms A–D, `bridges2/train_starvla.sbatch`) is parked; 4-GPU training never passed
  its smoke test.

**Where findings stand.** Read the docs for numbers and verdicts. Do not restate these as settled conclusions:
- After a stable grasp, the destination follows the instruction's action queries. Swapping in a seen
  instruction, or the whole query, sends the held object to that instruction's destination.
- In novel instructions the object noun appears to override the destination word: replacing it with "it"
  restores most successes on the development combinations.
- Before the grasp, behaviour looks like keyword retrieval of seen tasks.
- On frozen held-out combinations the "it" fix fails when no trained "place at D" program exists.
- Low-rank "concept subspace" patching did no better than a random subspace of matched size (negative result).
- Several interpretations, such as "attention competition" and "beyond retrieval", are explicitly unproven.
  RESEARCH_PLAN §11.8 says what is still unproven.

---

## 2. Repository map

```
lang-vla/
├── langeval/                 model-agnostic evaluation library (pip install -e .; imports only numpy at top level)
│   ├── adapters/factory.py   load(backend, checkpoint, suite, seed) -> policy, make_env, init_states, max_steps
│   ├── adapters/upstream.py  VLA-Adapter / OpenVLA-OFT via their own experiments.robot code (`vla` env)
│   ├── adapters/lerobot_policy.py  pi0.5 / SmolVLA / X-VLA via lerobot-eval's pipeline (`lerobot` env)
│   ├── adapters/starvla.py   websocket client to StarVLA's policy server; LV_PATCH / LV_LABELS / LV_SWAP client side
│   ├── goal_predicates.py    Goal dataclass, load_goals (seen s0-s9), load_novel_goals (json), satisfied()
│   ├── goal_matrix.py        GoalTracker: which goal a rollout executed (held for hold_steps)
│   ├── phase.py              GraspDetector, classify_outcome, is_grasped, object_z, movable_objects, path_length
│   ├── branch.py             settle, snapshot/restore (incl. gripper + fixture placement), shift_object, rollout
│   └── instructions.py       instruction transforms, function-word perturbations, LIBERO-Para loading
├── scripts/                  experiment entry points (each has a usage docstring)
├── configs/                  goal sets (novel, key probes, held-out), nonce words, paraphrases, StarVLA yaml
├── bridges2/                 PSC Bridges-2: config.sh, sbatch templates, setup, batch_MMDD.sh experiment launchers
├── deltaai/                  NCSA DeltaAI (aarch64 GH200) scripts; no allocation, kept for later
├── docs/                     research plan, findings, decisions, design, paper outline (see §7)
├── patches/                  our patches to Pion (VLA-Adapter copy) and LIBERO-Para
├── tests/                    pytest unit tests for langeval and scripts (CPU only)
├── Pion/                     NOT committed; rebuilt by scripts/setup_pion.sh (upstream + patches/)
├── IMPLEMENTATION_PLAN.md    original training/probe build plan (historical)
└── pyproject.toml            package `langeval`, deps: numpy; extra [test]: pytest
```

`tmp/` and `ccfa-workfiles/writing/` are empty locally. `logs/` and `results/` are gitignored.

### Scripts by role

| Role | Script | Subcommands / purpose |
|---|---|---|
| Closed-loop eval | `scripts/eval_goals.py` | success on seen s0–s9 and goals from a json; records `executed`, `completed_other`, `start_pos`, `final_pos`; `--switch`, `--init_offset` |
| Grasp-state branching (StarVLA main line) | `scripts/patch_branch.py` | `collect`, `branch` (`--lang`, `--shift`, `--goals`), `report`, `probe`, `attn`, `norms`, `paired`, `moved` |
| Phase-1 branching | `scripts/e3_grasp_branch.py`, `scripts/e4_switch_time.py` | E3 paraphrase branching; E4 switch-time curves |
| Action-head interventions | `scripts/erase_action.py` | `serve` (server wrapper: record / erase / patch / attention dumps), `fit`, `binding`, `subspaces`, `alphas`, `patch_report`, `landing` |
| Trajectory vs demos | `scripts/traj_sim.py` | default: compare branch eef paths with LeRobot demos; `plate`: plate position ranges |
| Feasibility | `scripts/feasible.py` | drop-test end-state feasibility of a goals json; `--sites` region positions |
| Probes / QA | `scripts/make_spatial_qa.py` (`collect`, `build`), `scripts/probe_vlm.py` (`extract`, `fit`), `scripts/eval_qa.py`, `scripts/make_counter_pairs.py` (`collect`, `analyze`) | spatial QA data, linear probes, QA accuracy, counter-heuristic pairs |
| Phase-1 analysis | `scripts/paraphrase_invariance.py` (E6), `scripts/para_vs_e6.py`, `scripts/judge_h1h3.py`, `scripts/report.py`, `scripts/per_task.py`, `scripts/diag_restore.py` | — |
| Setup | `scripts/setup_pion.sh`, `scripts/setup_starvla.sh`, `scripts/download_starvla_assets.sh`, `scripts/nccl_check.py` | — |

---

## 3. Architecture and data flow

### 3.1 Backends

`langeval.adapters.factory.load(backend, checkpoint, suite, seed)` returns
`(policy, make_env(task_suite, task_id) -> (env, language), init_states(task_suite, i), max_steps)`.
Every policy exposes `act(obs, instruction) -> list[action]` (one open-loop chunk) and optionally `begin()`, which
is called at every (re)start.

| backend | models | env | `checkpoint` means |
|---|---|---|---|
| `upstream` | VLA-Adapter, OpenVLA-OFT | `vla`, with the model repo first on `PYTHONPATH` (`REPO=vla-adapter` or `oft`) | checkpoint dir |
| `lerobot` | π0.5 (SmolVLA, X-VLA tier 2) | `lerobot` | checkpoint dir |
| `starvla` | StarVLA Qwen3-VL OFT | client in `vla`; model in `starvla` behind a websocket server | `host:port`; the literal `SERVER` in `run_starvla.sbatch` |

All backends drive the same LIBERO `OffScreenRenderEnv`, so `env.env` is the robosuite env. Predicates, grasp
checks and `snapshot`/`restore` work the same way for every backend.

### 3.2 StarVLA inference path

```
LIBERO obs ──► langeval/adapters/starvla.py (vla env)
               flip [::-1, ::-1], resize 224, no proprio; batch = [A] or [A, donor B1, B2, ...] (LV_PATCH)
          ──websocket──► StarVLA server_policy.py (starvla env), optionally wrapped by scripts/erase_action.py serve
               Qwen3-VL forward (sdpa; eager when LV_ATTN) → last-layer hidden states at the 8 action tokens
               = action queries q [B, 8, 2560]   ← hook point: Qwenvl_OFT._gather_action_token_embeddings
                   LV_RECORD: append q to file | LV_ERASE: project out a subspace |
                   LV_PATCH_U: q_A ← q_A + β·UUᵀ(mean(q_B) − q_A)  (full:1 ⇒ q_A = mean(q_B))
               → MLP action head → 8-step chunk (un-normalized)
          ◄── only row A's chunk is executed; gripper binarized (> 0.5 → open = −1)
```

The OFT action head reads **only** the 8 action queries, so a patch there is complete control over the head's input.
Treat the VLM as frozen: no script here trains it except `train_starvla.sbatch`, which is parked.

### 3.3 Experimental pipeline (main line)

```
eval_goals.py (full episodes) ──► success / executed goal per (goal, init)
patch_branch.py collect       ──► stable-grasp snapshots  {goal}-init{k}.npz + states.jsonl
patch_branch.py branch        ──► from the same snapshots: instruction override (--lang), object shift (--shift),
                                  or server-side patch (LV_PATCH + LV_PATCH_U) → episodes.jsonl per condition
patch_branch.py report/paired ──► success, outcome class, where the held target ended, paired McNemar / bootstrap
erase_action.py subspaces     ──► concept subspaces from a recorded run (LV_RECORD + LV_LABELS + LV_SWAP)
patch_branch.py probe + attn  ──► attention dumps and figures (descriptive only)
traj_sim.py                   ──► post-grasp eef paths vs LIBERO-Goal demos (V3)
```

### 3.4 Key invariants

- **Seeding:** `settle(env, init, seed=args.seed*100_000 + gi*1000 + k)` re-seeds fixture placement per
  (goal, init). Keep this formula; cross-run comparisons depend on it.
- **Restore:** `snapshot`/`restore` carry the MuJoCo state, the fixtures' `body_pos`/`body_quat` and the gripper's
  `current_action` (DECISIONS 2026-10-16). `patch_branch.py branch` checks that two restores of the first state give
  the same observation and stops if they don't.
- **Stopping:** an episode ends when the *instructed* goal holds (LIBERO `done`) or at `max_steps`. Other goals are
  only recorded (DECISIONS 2026-10-18: an earlier version wrongly stopped on any goal).
- **Probe goals:** a goal with an empty predicate list never holds (`goal_predicates.satisfied`). Only `executed`
  is meaningful for them.
- **Outcome classes:** `classify_outcome` checks in priority order: success, wrong_goal, wrong_object, stalled,
  holding, dropped, misplaced. `other_grasped` is a grasp check, not contact detection.
- **Re-planning:** a branch restarts with an empty action queue, so it re-plans from the grasp. `none` is a
  "re-planned after grasp" baseline, not a replay of the source episode.

---

## 4. Computing infrastructure

### 4.1 Machines

| Where | What runs there | Notes |
|---|---|---|
| Local Mac | editing, git, local tests, reading pulled PNGs | `.venv` has numpy and pytest but not Pillow; no LIBERO, no GPU |
| Bridges-2 login node | `git pull`, `bash bridges2/batch_*.sh <mode>` (which calls sbatch), light CPU analysis (`report`, `paired`, `traj_sim.py`) | PSC kills processes after **1800 CPU-seconds** (`bridges2/batch_1023.sh`). Heavy CPU work goes through `run.sbatch` |
| Bridges-2 GPU-shared (L40S 48 GB) | every rollout and model forward pass | account `cis260349p`; 1 SU per L40S GPU-hour; H100 costs 2 SU/h; V100 has no bf16 (`bridges2/README.md`) |
| NCSA DeltaAI (`deltaai/`) | nothing at present | aarch64 GH200; no allocation (`deltaai/README.md`) |
| NCSA Delta | planned home after the Bridges-2 outage | **not configured** in this repo. Delta (x86, A100/A40) is not DeltaAI, so it needs its own config **[unverified details]** |

Bridges-2 is scheduled to go down on 2026-11-09 for several weeks, so results should be backed up before 11-01
(human-provided schedule, not in the repo).

### 4.2 Environment and paths (`bridges2/config.sh`, sourced by every Bridges-2 script)

| Variable | Value | Contents |
|---|---|---|
| `PROJ_DIR` | `${PROJECT:-/ocean/projects/cis260349p/$USER}` | project space |
| `LANGVLA_ROOT` | `$PROJ_DIR/lang-vla` | this repo on the cluster; submit jobs from here |
| `STORE` | `$PROJ_DIR/store` | envs, weights, data, results |
| `CKPT` | `$STORE/ckpt` | checkpoints (`$CKPT/<repo-name>`, e.g. `$CKPT/pi05_libero_finetuned_v044`, `$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt`) |
| `RESULTS` | `$STORE/results` | all experiment outputs |
| `HF_HOME` | `$STORE/hf` | HF cache |
| `DATA_ROOT` | `$STORE/data/libero` | **unused; the directory does not exist**. The LeRobot demos live in `$STORE/data/starvla_libero/libero_goal_no_noops_1.0.0_lerobot` |
| `OFT`, `PARA`, `VLAA`, `LEROBOT_SRC` | `$STORE/src/...`, `$LANGVLA_ROOT/Pion/VLA/VLAAdapter` | external code |
| StarVLA source | `$STORE/src/starVLA` (pinned `985c223...` in `scripts/setup_starvla.sh`) | server code |
| `MUJOCO_GL=egl` | | headless rendering (GPU nodes) |

Conda envs live in `$STORE/envs/{vla,lerobot,starvla}`. Activate them with `source bridges2/config.sh; use_env <name>`,
which also sets `LIBERO_CONFIG_PATH` (importing libero without it prompts interactively and crashes non-interactive
scripts). Which env to use:
- `vla`: LIBERO client, `eval_goals`, `patch_branch`, upstream models.
- `lerobot`: π0.5. It is also the env that has `pyarrow`, so `traj_sim.py` must run in it.
- `starvla`: the StarVLA server, `eval_qa.py`, `probe_vlm.py`.

### 4.3 Job templates

| Template | Use | Key variables |
|---|---|---|
| `bridges2/run.sbatch` | any command on one L40S | `ENV` (default `vla`), `REPO=vla-adapter\|oft` |
| `bridges2/run_starvla.sbatch` | StarVLA server + one client command | `SV_CKPT` (.pt, required), `SV_WRAP` (e.g. `$LANGVLA_ROOT/scripts/erase_action.py`), literal arg `SERVER` |
| `bridges2/eval_libero.sbatch`, `eval_lerobot.sbatch`, `eval_para.sbatch` | phase-1 official evals | see `bridges2/README.md` |
| `bridges2/train_starvla.sbatch` | 4 × L40S StarVLA training | parked; see README |

`run_starvla.sbatch` picks a free port from the OS (a fixed port once collided, job 49559598). Its readiness probe
leaves 3 harmless handshake tracebacks in every `logs/starvla-server-<jobid>.log`.

**Slurm gotchas:**
- `--export=ALL,A=b,...` splits on commas. Values that contain commas, such as `LV_PATCH="n0=s8,n3=s8"`, must be
  `export`ed before `sbatch` (every `batch_102*.sh` does this).
- Slurm copies the batch script at submission time, but the Python code is read when the job starts. A `git pull`
  between submitting and starting changes the code the job runs, and `run_config.json` records the git hash it used.
- Chain jobs with `--dependency=afterok:<id>` (`bridges2/batch_1031.sh branch <collect id>`).

### 4.4 Logs and outputs

- `logs/<job-name>-<jobid>.out` is the job's stdout, and `logs/starvla-server-<jobid>.log` is the server's log.
  `logs/batch_MMDD.txt` lists the `<jobid> <condition>` pairs a batch script submitted.
- Every run directory has `run_config.json` (args, `LV_*` env, git hash, start time) and `episodes.jsonl`;
  `eval_goals` also writes `summary.json`. Run directories are created with `exist_ok=False`, and the batch scripts
  skip existing directories, so **outputs are never overwritten**. To redo a run, the human must remove or rename
  its directory.
- Main result trees: `$RESULTS/erase/{patch,patch2,branch,branch2,branch3,branch4}`, `$RESULTS/stage3`,
  `$RESULTS/pi05/branch`, `$RESULTS/{eval,e3,e4,e6,para,novel,qa,probe}` (phase 1).

### 4.5 File transfer and downloads

- Use **rsync, not scp** (human preference), with host `data.bridges2.psc.edu`. Use absolute remote paths: a quoted
  `$RESULTS` is not expanded on the remote side. Example **[template]**:
  `rsync -av lluo3@data.bridges2.psc.edu:/ocean/projects/cis260349p/lluo3/store/results/erase/branch4/fig-wine2-plate ~/attn_png/`
- Checkpoints come from `bridges2/download_assets.sh` (pinned HF revisions) and `scripts/download_starvla_assets.sh`
  (run on a login node). π0.5's tokenizer repo is gated, so the human must log in to HF; never handle tokens
  yourself.

---

## 5. Workflows

Each workflow notes where its commands run. Commands meant for the human's zsh must not contain inline `#`
comments (human preference).

### W1. Environment setup (one-time; already done on Bridges-2)
From `bridges2/README.md` **[verified]**:
`bash scripts/setup_pion.sh`, then
`srun -A cis260349p -p GPU-shared --gpus=l40s-48:1 -t 1:00:00 --pty bash bridges2/setup_vla.sh`. Do the same with
`bridges2/setup_lerobot.sh`, and run `scripts/setup_starvla.sh bridges2` with `-t 1:30:00`. Validation:
`sbatch bridges2/smoke.sbatch`.
Locally: `python -m venv .venv && .venv/bin/pip install -e ".[test]"`.

### W2. Checkpoints
Run `bash bridges2/download_assets.sh` and `bash scripts/download_starvla_assets.sh bridges2` on a login node
**[verified]**. Validate that `ls $CKPT` shows the repo names. The StarVLA checkpoint used everywhere is
`$CKPT/Qwen3-VL-OFT-LIBERO-4in1/checkpoints/steps_50000_pytorch_model.pt`.

### W3–W4. Inference and closed-loop rollouts
- StarVLA **[verified, from `bridges2/README.md`]**:
  `sbatch --export=ALL,SV_CKPT=<.pt> bridges2/run_starvla.sbatch python scripts/eval_goals.py --backend starvla --checkpoint SERVER --n_init 50 --out $RESULTS/novel/<tag>`
- Upstream: `sbatch --export=ALL,ENV=vla,REPO=vla-adapter bridges2/run.sbatch python scripts/eval_goals.py --checkpoint $CKPT/LIBERO-Goal-Pro --sets novel --n_init 50 --out ...`
- π0.5: `ENV=lerobot`, `--backend lerobot --checkpoint $CKPT/pi05_libero_finetuned_v044`.
- Custom goal sets: `--sets novel --novel_json configs/<file>.json --goals a,b`. Instruction switch after a stable
  grasp: `--switch "h0=put it in front of the stove;h3=s6"` (values are goal names or literal text).
- Validate: `summary.json` exists, `err=0` in the printed lines, and `n` equals the requested `n_init`.

### W5. Collect and restore simulator states
`patch_branch.py collect --goals n0,n3,n4 --n_states 20 --out <dir>` (any backend, runs unpatched). It writes
`<goal>-init<k>.npz` and `states.jsonl`, and prints "N stable grasps of X from M init states". Then
`branch --states <dir> --out <run>`.
- Overrides: `--lang "n3=s1;n4=put it on the stove;s8="` (goal name, literal text, or empty).
- Object shift: `--shift plate_1:0,-0.1`.
- Templates: see `bridges2/batch_1026.sh`, `batch_1027.sh`, `batch_1029.sh` and `batch_1031.sh`.
- Phase-1 equivalent: `scripts/e3_grasp_branch.py`. If restores look nondeterministic, check `determinism.json`
  and run `scripts/diag_restore.py`.

### W6. Intermediate representations
- **Action queries:** run the server with `SV_WRAP=$LANGVLA_ROOT/scripts/erase_action.py` and `LV_RECORD=<file>.f16`
  (float16 rows, shape in `<file>.json`). On the client, `LV_LABELS=<file>.jsonl` writes one label line per call,
  and `LV_SWAP=1` adds every-10th-call swaps across all 19 instructions. Example: `bridges2/batch_1021.sh`.
- **VLM hidden states:** `scripts/probe_vlm.py extract` in the `starvla` env.
- **Attention:** `LV_ATTN=<dir>` on the server (eager attention, so these jobs report no success rates), plus
  `patch_branch.py probe`. Read the dumps with `patch_branch.py attn <run> --dumps <dir> --png <out> [--word plate]`.

### W7. Probes
`probe_vlm.py fit --qa_dir ... --feats name=path.npz --pixels` runs on CPU (login node or `run.sbatch`). The QA
data comes from `make_spatial_qa.py collect` then `build`. Check the coordinate convention with `--overlay`
before using the data (README step 12). Counter-heuristic test: `make_counter_pairs.py`.

### W8. Causal interventions on the action queries
1. Fit subspaces: `erase_action.py subspaces --rec --labels --run_dir --out [--exclude goals]` (see `batch_1025.sh`).
2. On the client, set the donor map with `export LV_PATCH="n0=s8,n3=s8,n4=s1"`. Multiple donors are averaged:
   `n3=s2+s4`. Extra donor names come from a json via `LV_PATCH_EXTRA` (see `batch_1028.sh para`).
3. On the server, choose the patch with `LV_PATCH_U=<npz>:none`, `<npz>:full:<β>` or `<npz>:<name>:<rank>:<β>`.
   Rank and strength are mandatory, and the server exits if they are missing.
4. Run the `none` control with the **same donor map** as the patch conditions.
- Erasure: `LV_ERASE=<npz>:<name>[:alpha]` (`batch_1021.sh`, `batch_1022.sh`).
- Measured patch size: record with `LV_RECORD` during a `none` run, then run `patch_branch.py norms`.

### W9. Language perturbation and compositional evaluation
- Novel combinations: `configs/novel_goals.json` (n0–n8).
- Key probes from episode start: `configs/key_probes.json` (k0–k8; no predicates).
- Frozen held-out set: `configs/heldout_goals.json` (h0–h3, with `switch_it` and `switch_seen`).
- Nonce words: `configs/nonce_words.json`. bddl wording: `configs/bddl_wording_paraphrases.json`. LIBERO-Para:
  `$PARA/metrics/libero_para_metadata.csv` via `langeval.instructions.load_paraphrases`.
- Feasibility of new goals: `scripts/feasible.py` (`batch_1030.sh feasible`). It is a drop test, so it says nothing
  about reachability, and a drop from 5 cm proved too strict for object-on-object goals.

### W10. Analysis and reports (CPU)
Each batch script has a `report` mode that prints a summary; the human pastes it. Underlying commands:
- `patch_branch.py report <runs...>`
- `patch_branch.py paired <base> <runs...>` (McNemar and bootstrap, paired on (goal, init))
- `patch_branch.py moved <eval runs> [--sites sites.json]`
- `erase_action.py patch_report|landing`
- `traj_sim.py --demos <lerobot dir> <runs>` (in the `lerobot` env)
- Phase 1: `scripts/report.py`, `scripts/judge_h1h3.py`.

### W11. Submit, monitor and debug jobs
- Submit: `bash bridges2/batch_MMDD.sh <mode>` from the repo root on a login node.
- Monitor: `squeue -u $USER`, and `sacct -j <ids> -X --format=JobID,JobName%20,Elapsed,State`.
- SU used: `sacct -j $(cut -d' ' -f1 logs/batch_MMDD.txt | paste -sd,) -X -n -P --format=ElapsedRaw | awk '{s+=$1} END {print s/3600, "GPU-h"}'` **[verified]**.

Common failures:

| Symptom | Cause / fix |
|---|---|
| `(Priority)` for hours | only 3 L40S nodes; not a partition error |
| `Errno 98` in a server log | port collision (fixed with an OS-assigned port); rerun |
| `EOFError` from `libero/__init__.py` | `LIBERO_CONFIG_PATH` unset: `use_env vla` first |
| `pyarrow` ImportError | run `traj_sim.py` in the `lerobot` env |
| process killed on the login node | 1800 CPU-s limit: use `run.sbatch` |
| `LV_PATCH_U=...: give rank:strength` | the server requires an explicit rank and strength |
| `--lang with LV_PATCH` SystemExit | the client looks up donors by the instruction text; don't combine the two |

Validate a run by checking `run_config.json` (env and git hash), the server log line `patching ...`, and that
`episodes.jsonl` has the expected count with `error` null.

### W12. Reproducing an experiment
Find the batch in RESEARCH_PLAN by its name (each pre-registration names its `batch_MMDD`). Check out the git hash
in that run's `run_config.json`, then rerun the same batch mode with a new output directory. Keep the seeds, init
ranges (`--init_offset`), donor maps and thresholds unchanged.

---

## 6. Task-to-code navigation

| Request | Look at / change | Run |
|---|---|---|
| Add a new goal or combination | `configs/*.json` (`id`, `language`, `predicates` as `["on", obj, dest]`) | `eval_goals.py --novel_json` |
| New instruction condition from grasp states | a new mode in a `bridges2/batch_*.sh` with `patch_branch.py branch --lang` | `batch_*.sh <mode>` |
| Switch instructions mid-episode | `scripts/eval_goals.py --switch` (grasp-triggered) or `scripts/e4_switch_time.py` (fixed times) | — |
| Change what the server does to the action queries | `scripts/erase_action.py` `serve()` / `attention_dumps()` | `run_starvla.sbatch` with `SV_WRAP` |
| Change what the client sends | `langeval/adapters/starvla.py` (`_query`, `LV_PATCH*`) | — |
| New outcome metric for branches | `patch_branch.py branch()` (record it in the episode dict) and `report()` | — |
| Grasp or phase thresholds | `langeval/phase.py` (calibration knobs; changing them invalidates comparisons) | — |
| Restore problems | `langeval/branch.py`, `scripts/diag_restore.py` | — |
| A new model backend | `langeval/adapters/` and `factory.load` | smoke on one goal first |
| Statistics for a claim | `patch_branch.py paired` | login node |
| Where did objects end up | `patch_branch.py report` (branch runs), `moved --sites` (eval runs), `erase_action.py landing` | — |
| Is a rollout replaying demos | `scripts/traj_sim.py` | `lerobot` env |
| Paper status | `docs/PAPER_OUTLINE.md`, RESEARCH_PLAN §11.8 | — |

---

## 7. Research context and documentation

| Document | Language | Role |
|---|---|---|
| `docs/RESEARCH_PLAN.md` | Chinese | **source of truth.** Questions, pre-registrations (written before results, dated), results, verdicts. §11.7 is the main line, §11.8 the next steps, §2.4 related work |
| `docs/findings.md` | Chinese | readable summary of results; §8 covers the current main line |
| `docs/DECISIONS.md` | English | dated engineering decisions and upstream traps |
| `docs/PAPER_OUTLINE.md` | Chinese | claims, section and figure map, missing evidence |
| `docs/RESEARCH_DESIGN.md`, `IMPLEMENTATION_PLAN.md` | English | phase-1 design and the original build plan (historical) |
| `docs/EXPERIMENT_REVIEW_CVPR2027.md` | Chinese | external review (untracked; not ours to commit) |
| `docs/research-briefing.html`, `docs/vla-language-gap.html` | — | presentation pages |

**Naming:**
- Goals: `s0`–`s9` are the LIBERO-Goal tasks in benchmark order (read with `suite.get_task(i).language`).
  `n0`–`n8` are novel combinations, `k0`–`k8` key probes, `h0`–`h3` frozen held-out combinations.
- Phase-1 experiments: E0–E8.
- Pre-registered prediction groups in §11.7–§11.8: M, P, Q, R, T, U, V, W, X, Z, T′, W′.
- Batches: `batch_MMDD[letter].sh`, where the date is the day it was written.

**Pre-registered vs exploratory.** A block titled "…（在任何…结果之前写定）" ("written before any … results") is a
pre-registration. Its thresholds are fixed: record the verdict, and add a dated, separate revision for anything
that changes. Results marked "只描述" (descriptive only), "事后" (post hoc) or "探索性" (exploratory) are not
confirmatory. Reviewer corrections are recorded as "口径修正" (definition corrections) without changing earlier
verdicts.

**Do not upgrade claims.** RESEARCH_PLAN §11.8 lists what is still unproven, for example "object word overrides
the destination via attention", "beyond retrieval" and "−y failure is out of coverage". Write new results with their
limits, and reference the doc rather than copying large tables into new files.

---

## 8. Rules for agents

**Before changing anything**
- Read the files involved and their callers, and check `git status`. Reuse the existing helpers
  (`langeval.branch`, `langeval.phase`, `GoalTracker`, `parse_lang`, `config()` in `patch_branch.py`) rather than
  writing new ones.
- Prefer adding a mode to an existing script or batch file over creating a new one. Keep diffs minimal and match
  the surrounding code (dense docstrings, few inline comments).

**When implementing**
- Do not change seeds, init ranges, success predicates, `GraspDetector` defaults, outcome rules or pre-registered
  thresholds without a dated note in RESEARCH_PLAN. New experiments get their predictions written into
  RESEARCH_PLAN **before** they run.
- Never overwrite outputs. Keep the `exist_ok=False` and skip-existing guards in batch scripts.
- Every new branch condition should also run a matched control (same donor map, same states, same-rank random).
- Non-trivial logic gets a small CPU test in `tests/test_qa_and_adapters.py`. Tests load scripts with
  `importlib.util.spec_from_file_location`.

**When validating**
- `python3 -m py_compile <files>`, `bash -n <batch.sh>`, and `.venv/bin/python -m pytest -q tests/` locally.
- GPU code can only be validated on the cluster. Say clearly what was not run, and ask the human to run the
  tests there (`python -m pytest tests/test_qa_and_adapters.py -q -k <name>`).
- Report results only from pasted outputs; never assume a job succeeded.

**Safety and resources**
- Do not submit jobs, start downloads, delete results or checkpoints, or change branches unless the human asks.
  Give the commands and an SU estimate instead.
- Commit and push only when the human asks for it. When doing so, add files explicitly (never `git add -A`), never
  include files with someone else's uncommitted edits, and keep the repository's commit identity and trailer
  conventions (see `git log`).
- The repository is public. No credentials or tokens, and no personal email addresses beyond what already appears
  in git metadata.

**Common mistakes seen in this project**
- Claiming a mechanism from attention maps alone.
- Comparing runs with different donor maps or different states.
- Treating `wrong_object = 0` as "never touched another object".
- Averaging heatmaps across states with different object positions.
- Running `traj_sim.py` in the `vla` env.
- Using `scp`.
- Calling `$DATA_ROOT`.
- Forgetting that `full:1` is equivalent to swapping the instruction (T2).

---

## 9. Known gaps and items to verify

- No Delta (NCSA, x86) configuration exists yet; `deltaai/` targets a different machine.
- Bridges-2 training (`train_starvla.sbatch`) never passed the 4-GPU smoke test (README, 2026-10-19).
- `bridges2/config.sh` defines `DATA_ROOT`, but the directory doesn't exist and no code reads it.
- The feasibility check (`feasible.py`) tests only whether a dropped object stays; the default drop height is now
  1 cm, and h2/h3 have not been re-checked.
- RoboCasa365 and LIBERO-PRO are planned (RESEARCH_PLAN §11.7 dataset route, §11.8) but have no code here.
- The two local test failures (Pillow) are environmental.

## 10. Maintaining this file

Update AGENTS.md when any of the following changes:
- a new script or subcommand appears;
- an env var or job template changes;
- a new config set is added;
- storage paths move;
- the cluster changes (for example, the move to Delta).

Keep research results out of this file and point to RESEARCH_PLAN or findings instead. After editing, check that
every path in the file exists (see the check in the completion notes) and update the date and commit in the header.
