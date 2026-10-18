# Design decisions

One entry per decision: what was decided, why, and what it affects.

## 2026-10-10 Replacing Pion/ and placing the patch

- `Pion/` was a copy without `.git`. Re-cloned `https://github.com/OPTML-Group/Pion` at `8796501`. `diff -rq` showed the old copy was byte-identical to upstream, so nothing was lost.
- The patch was in the repo root and has been moved to `patches/pion-lang-spectral.patch`. On the `lang-spectral` branch it applied with `git apply --index` and was committed as `Apply lang_spectral patch`. `check_math_numpy.py` printed `checks passed` (local numpy).

## 2026-10-10 Hyperparameter selection uses only env_success under the original instruction

- For every hyperparameter sweep (adamw learning rate, pion k_p, LangGuard γ/mode, SAM ρ, guidance ω), the best setting is chosen by **env_success with `instruction_mode=orig`** only.
- **Why:** `follow_rate`, `I_behav` and the representation bits are the outcomes this project measures. Choosing settings by them is the same as selecting on the dependent variable, which inflates the cross-method differences in language use. orig env_success is also the standard most papers use, so our "best configuration" for each method is the one its own authors would pick.
- **What it affects:** the main table reports every metric at the setting chosen this way. If a method's best language metric falls at some other setting, it can go in an appendix but must be labeled "post-hoc".

## 2026-10-10 Training topology 4 cards × batch 16, and its effect on the gradient probe

- A DeltaAI node has 4 GPUs, so we train with 4 × 16. The global batch is 64, the same as upstream 8 × 8, so the optimization itself is equivalent.
- **The probe statistics are not equivalent.** `lang_spectral/grad_probe.py` runs only on rank 0 and takes **that rank's local gradient before all-reduce** for G_τ, while the update U = −(W_after − W_before) is the global step. So:
  - `credit` / `energy` / `budget` / `top_overlap`: G_τ is estimated from 16 samples instead of 8, so shard noise is lower and the values sit closer to the global gradient. The same configuration produces numbers that differ systematically from the 8 × 8 numbers.
  - The gradient `erank`: the rank of G_τ is bounded by the number of positions in that shard, so 16 samples allow a larger erank.
  - Representation spectra (Q11, `stable_rank` / `erank`): the row count n_τ doubles. Vision tokens go from about 8 × 512 = 4096 rows to about 16 × 512 = 8192, which is exactly `max_rows=8192`, so subsampling is about to kick in. Language and action tokens are not capped, but their erank also grows with n_τ.
- **Rule:** probe numbers are only compared within the same topology (4 × 16). They are not compared with numbers from the Pion paper or from 8 × 8 runs on other clusters. If an OOM forces `BATCH_SIZE=8 GRAD_ACCUM=2`, the probe numbers are not comparable to 4 × 16 runs either; the run config must record it, and before that topology is used, someone must confirm that the probe hooks handle gradient accumulation correctly.

## 2026-10-10 Training seed

- Added `--seed` to `finetune_pion.py`, default 7. torch / numpy / random use `seed` (the upstream `set_seed_everywhere`, which also sets `cudnn.deterministic=True, benchmark=False`). TF uses `seed + rank`.
- **Why TF is offset by rank:** the RLDS pipeline is not sharded by rank. Every rank reads the full dataset and runs `dataset.shuffle` on it. With the same TF seed on every rank, all 4 cards would get identical batches and the effective batch would drop from 64 to 16.
- The run_id gets a `+s{seed}` suffix, so different seeds never write to the same directory.
- Bitwise reproducibility is not guaranteed: tf.data parallel reads and tfds file ordering may still be nondeterministic. Still to verify: run two short training runs (30 steps) with the same seed and compare the loss for the first 10 steps. This is not in `smoke.sbatch` yet; record the result in STATUS.md.
- adamw / muon / pion first run one seed each (7) and save every 1500 steps, to get the step-wise curves. All other arms save every 5000 steps (the default in `deltaai/train.sbatch`).

## 2026-10-10 Fall back to SDPA when flash-attn is unavailable

- If `transformers.utils.is_flash_attn_2_available()` is false, `finetune_pion.py` sets `use_flash_attention_2` in the `LLM_BACKBONES` registry to False before loading the Prismatic VLM, so that LLM loads with SDPA.
- **Why the fallback does not affect training numerics:** that Prismatic VLM is only a weight source. Its state dict is copied into `vla`, which is built from `AutoModelForVision2Seq.from_config(config)` without `attn_implementation`, so it uses transformers' default implementation whether or not flash-attn is installed. The evaluation path (`openvla_utils.get_vla`) has `flash_attention_2` commented out, so it also uses the default.
- **Still to verify:** `finetune_pion.py` and `smoke.sbatch` both print the attention implementation the model actually uses (`_attn_implementation`). The value has to be recorded in STATUS.md. If it is `flash_attention_2` (meaning the analysis above is wrong), the fallback does change numerics and this entry must be revised.
- The fallback is implemented in `finetune_pion.py` rather than `prismatic/models/backbones/llm/qwen25.py` because the latter is not on the list of files we are allowed to modify (§9).

## 2026-10-14 Bridges-2: run only the zero-training experiments

- The cis260349p allocation is 1000 GPU SU. L40S / V100 cost 1 SU per GPU-hour and H100 costs 2. Bridges-2 runs only the zero-training experiments (IMPLEMENTATION_PLAN steps 3–7 plus E0–E7 in RESEARCH_DESIGN); training goes to DeltaAI later.
- **GPU: L40S** (8 GPUs and 192 cores per node, about 24 cores per GPU). Rollouts are bottlenecked by the CPU simulator, so the H100 (13 cores per GPU, twice the price) brings no benefit. **No V100**: the models load in bf16 and V100 (sm70) has no bf16 hardware.
- **Two environments**: `vla` (VLA-Adapter + OpenVLA-OFT, torch 2.2, upstream versions as is) and `lerobot` (π0.5 / SmolVLA / X-VLA, Python 3.12, torch ≥2.7).
  - The two model repos have identical dependency lists and both ship a top-level `prismatic` package, so **neither is pip-installed**; each job puts the right repo first on PYTHONPATH.
  - Each environment has its own `libero` package (LIBERO-Para's fork / hf-libero), so each gets its own `LIBERO_CONFIG_PATH` (both packages support this environment variable).
- **The libero in `vla` is LIBERO-Para's fork**: it keeps every original suite and adds `libero_para` (4092 paraphrase bddl files), so a single install covers both.
- External code is pinned to commits: openvla-oft `e4287e9`, LIBERO-Para `86cfb7b`, lerobot `ae2b57f` (see `bridges2/config.sh`).

## 2026-10-14 Added `--task_ids` to `run_libero_eval.py`

- A comma-separated subset of tasks; empty means all tasks, so default behavior is unchanged. It lets several processes share one GPU, splitting the tasks round-robin (`bridges2/eval_libero.sbatch`, `PROCS`). Each process gets a different `--run_id_note`, otherwise two processes started in the same second would write to the same log file.
- OFT's evaluation script lives in an external repo we may not modify, so OFT runs as a single process.

## 2026-10-14 Two traps found while reading the code (affect E2/E5/E6 and CFG guidance)

1. **`get_vla_action` normalizes `obs["state"]` in place** (`openvla_utils.py`, `obs["state"] = normalize_proprio(...)`). Calling it twice on the same observation dict normalizes the proprio twice. Every "same observation, several instructions" experiment (E2 S(t), E6, CFG's cond/null double forward, §6.4 patching) **must pass a `copy.deepcopy(observation)` on every call**. `smoke.sbatch` and `scripts/paraphrase_invariance.py` already do this. The guidance implementation in `PolicyAdapter.act_with_guidance` must follow the same rule.
2. **The hidden-state positions VLA-Adapter's action head reads are one position to the left of the action-query slots.**
   - The LLM sequence is `[BOS][P vision patches][input_ids[1:] (T-1 tokens)][NUM_TOKENS action-query slots][stop]`, so the query slots sit at `P+T … P+T+NUM_TOKENS-1`.
   - Inference (`modeling_prismatic._regression_or_discrete_prediction`) slices from `NUM_PATCHES + NUM_PROMPT_TOKENS = P+T-1`, and the vision part takes `item[:, :NUM_PATCHES]` = `BOS + the first P-1 patches`.
   - The training side (`finetune_pion.py`, `item[:, num_patches:-1]` plus the shifted mask) is shifted the same way, so **training and inference agree and success rate is unaffected**.
   - **But for interpretability**: what the action head actually reads is "the last prompt token + the first NUM_TOKENS-1 queries" and "BOS + the first P-1 patches"; the last query and the last patch are never read. The "action-query positions" in the probe (§6.2) and patching (§6.4) experiments must say which definition they use. `paraphrase_invariance.py` uses the slots themselves; the head-read positions can be added as a separate class if needed.
   - This conclusion comes from reading the code. Before relying on it, assert it once on a real forward pass, by comparing the slice indices against the token types.

## 2026-10-14 Limits of LIBERO-Para's protocol

- Its evaluation scripts run **only one episode per paraphrase, and every paraphrase of a scene uses that scene's init state 0** (`init_data[0]`). For deterministic regression policies (VLA-Adapter / OFT), that means each (scene, paraphrase) pair has exactly one trajectory, and the original-instruction baseline per scene is also a single trajectory.
- We use its paraphrase set (`metrics/libero_para_metadata.csv`), but our E1 paraphrase mode must vary the init state and bootstrap over init states. `bridges2/eval_para.sbatch` only reproduces its numbers.
- LIBERO-Para numbers its tasks by bddl filename in alphabetical order, which differs from the task order inside the LIBERO benchmark. **Always match by instruction text**; never align by number.

## 2026-10-14 E3 (grasp-state branching): implementation decisions

- **Stable grasp**: the target object is ≥3 cm above its rest height (`--lift_m`), both finger pads touch it (robosuite `_check_grasp`), and this holds for 5 consecutive steps (`--grasp_k`). These thresholds are calibration knobs; check them against a few videos before the full run.
- **Executed goal**: a goal's predicates hold for 10 consecutive steps (`--hold_steps`). This keeps transient states, such as passing over the plate, from counting. When several goals complete on the same step, the instructed goal wins.
- **Outcome priority**: success > wrong_goal > wrong_object > stalled / holding > dropped / misplaced; the `error` outcome is counted separately. Definitions are in `langeval/phase.py`.
- **Branch start**: restore the full sim state with the policy's action queue cleared, so every continuation instruction starts with a fresh query. The `orig` continuation is therefore not identical to the source rollout's own continuation, but it is directly comparable across continuation instructions.
- **Restore reproducibility is checked at run time**: the first `orig` branch is run twice with the same state and instruction. If the actions are not identical (for a deterministic policy), the script stops. The result is written to `determinism.json`.
- **The source paraphrase is always also a continuation instruction** (`para_source`). This is the fourth cell of the 2×2: "paraphrase-sourced state × same paraphrase".
- **Meaning-preserving perturbations** touch only function words: `dup_article` / `filler` (um) / `typo_article` (teh). Each branch records `embed_dist`, the L2 distance between the mean input embedding of the continuation instruction and that of the original instruction, so analysis can match paraphrases to perturbations by distance.
- **Task scope**: by default only tasks whose language contains "put" (pick-and-place). The push-the-plate task is excluded because the plate is never lifted. In the libero_goal benchmark order these are tasks 1/2/3/4/6/8/9; that order is not the bddl filename order.
- The upstream evaluation puts the action chunk into `deque(maxlen=num_open_loop_steps)`, which keeps the last n actions. `UpstreamPolicy.act` takes `[-n:]` to match. On LIBERO the chunk length equals n, so the two are the same there.

## 2026-10-14 How the code is synced

- GitHub `logan-0623/lang-vla` does not contain `Pion/`, which has its own git history. Our changes live as patches in `patches/`; `scripts/setup_pion.sh` clones upstream 8796501 and applies them. Locally verified: the rebuilt tree is file-for-file identical to the local `Pion/`.
- After a new Pion commit, run `git -C Pion format-patch <last exported commit>..HEAD -o patches/` and commit the new patch.

## 2026-10-15 Bridges-2 smoke results (job 49250616)

- `vla` env needs three pins on top of the upstream lists (now in `setup_vla.sh`): `tensorflow-metadata<1.17`, `wandb<0.18` (protobuf 5 vs TF 2.15) and `mujoco==2.3.7` (robosuite 1.4 asserts fail on mujoco 3.x).
- **Attention implementation on the eval path is `sdpa` for both VLA-Adapter and OFT**, as predicted in "Fall back to SDPA" above; flash-attn is never used at eval.
- VLA-Adapter LIBERO-Goal-Pro, 1 trial x 10 tasks: 9/10 success. OFT Goal forward 15.0 GiB peak, VLA-Adapter 3.5 GiB.
- pi0.5 loads its tokenizer from the gated `google/paligemma-3b-pt-224`: requires HF license acceptance + login (`download_assets.sh`).
- `MUJOCO_GL=egl` makes robosuite hang at import on login nodes (no GPU): import tests must run inside a GPU job.

## 2026-10-15 LIBERO-Para's fork changes the original suites' instructions (E0 numbers before this fix are invalid)

- First E0 on Bridges-2: OFT Goal 418/500 = 83.6% (official 97.9%), VLA-Adapter bench 25/30. Cause: the fork's `benchmark/__init__.py` builds every suite's `task.language` from the bddl `(:language ...)` field instead of the filename. For libero_goal these differ, e.g. "Open the middle layer of the drawer" vs "open the middle drawer of the cabinet", "Put the bowl on the top of the drawer" vs "put the bowl on top of the cabinet", "Put the cream cheese on the bowl" vs "... in the bowl". Upstream LIBERO, the training data and LIBERO-Para's own `original_instruction` column all use the filename form.
- Fix: `patches/libero-para-filename-language.patch` (applied by `setup_vla.sh`) uses the filename form for every suite except `libero_para`. Without it, matching LIBERO-Para rows by instruction text (E3/E6) would also have failed.
- Jobs 49250616 (smoke), 49255711 (bench), 49255723 (OFT E0) ran with the bddl wording. Their success rates are not E0; they are kept as an unplanned observation: rewording 5 of the 10 Goal tasks (bddl wording) costs OFT ~14 points. The bench throughput numbers are still valid.

## 2026-10-16 E0 passed; E1b: the bddl-wording drop is two tasks, both "cabinet" -> "drawer"

- E0 (filename wording, 10 x 50): OFT Goal 487/500 = 97.4% (official 97.9%), VLA-Adapter Goal-Pro 487/500 = 97.4%. Throughput: VLA-Adapter 2 procs 482 rollouts/h, OFT 353/h. pi0.5 smoke: 1/1 success, but torch.compile (max-autotune) took ~10 min with triton shared-memory errors on L40S, so `eval_lerobot.sbatch` now passes `--policy.compile_model=false` and runs 10 envs per batch.
- OFT, bddl wording (job 49255723) vs filename wording (49268902), per task: the drop is entirely in
  - "put the bowl on the top of the drawer" (vs "on top of the cabinet"): 19/50 vs 50/50
  - "put the wine bottle on the top of the drawer" (vs "on top of the cabinet"): 12/50 vs 50/50
  The other three reworded tasks are unchanged ("middle layer of the drawer" 48 vs 49, "top layer of the drawer and put the bowl inside" 45 vs 44, cream cheese "on" vs "in" the bowl 50 vs 50).
- Reading: changing the preposition does nothing; replacing the **location noun with another fixture's name that appears in other training instructions** ("drawer") costs 60-76 points. Consistent with H3 (a content word acts as a retrieval key) and with LIBERO-Para's "nouns matter most", but here the noun is the target location, not the object. Which goal was executed instead is unknown: the upstream eval only records success. Next: E3 on tasks 2 and 4 with this wording as the source paraphrase (`configs/bddl_wording_paraphrases.json`), which records the executed goal and the phase of failure.

## 2026-10-16 The fork also reorders the original suites; E6 layout fixes; pi0.5 E0

- LIBERO-Para's `libero_suite_task_map.py` lists libero_spatial / object / goal / 10 in **alphabetical** order instead of upstream's (libero_90 unchanged). Task ids passed to our scripts therefore pointed at different tasks (the first E3 pilot asked for 2,4 and got "push the plate" and "bowl on the stove"). Episodes were never mismatched with their init states (both are looked up by task name), so E0 numbers stand. `patches/libero-para-filename-language.patch` now also restores upstream order, which matches hf-libero in the `lerobot` env, so task ids mean the same task for every model.
- E6 fixes found on the first real run:
  - a paraphrase ending in "?" merges with the template's "?" into one BPE token; `instruction_span` now ends the span before the merged token instead of raising;
  - OFT's `predict_action` appends the empty token 29871 to the prompt, so the LLM sequence has one more token than the tokenized prompt (L=593: 1 + 513 + 22 + 56 + 1). The layout assertion caught it.
- pi0.5 E0 (LeRobot, no torch.compile, 10 envs per batch): 484/500 = 96.8%, CI95 [94.9, 98.0]; 500 episodes in 1971 s, about 910 rollouts per GPU-hour.

## 2026-10-16 E3 restore: gripper state, and a noise floor instead of a bitwise gate

- First real E3 pilots (jobs 49272496/8) stopped at the bitwise replay check for both VLA-Adapter and OFT. The cause is not yet known (no diagnostics were saved). Found while reading for it: robosuite grippers keep `current_action`, which ramps toward open/close a little every sim substep and is **not** in `get_sim_state()`. At a stable grasp it is saturated closed; after `reset()` it is 0 (half open), so every restored branch loosened the fingers for its first steps. That changes physics, not just reproducibility. `langeval.branch.snapshot/restore` now save and restore it.
- The run-time check now separates three things (determinism.json): two restores of one state give identical observations (if not, restore is broken and the script stops); the policy gives identical actions for one observation twice; and, for every state, where `orig` and `orig_rep` (the original instruction run twice from the same restore) first diverge.
- `orig_rep` is a continuation on every state. If bitwise reproducibility does not hold, the orig-vs-orig_rep outcome disagreement is the noise floor; an instruction effect counts only if it exceeds it. This replaces "stop unless bitwise identical" in the E3 entry above.

## 2026-10-16 E6 first results (VLA-Adapter Goal-Pro, OFT Goal; 10 tasks x 3 init states, 5 paraphrases per category)

R = distance(orig, x) / mean distance(orig, other task). LIBERO-Para groups: act (verb), obj (object), comp (both).

| | R_action VLA-Adapter | R_action OFT | R action slots, last layer VA / OFT |
|---|---|---|---|
| act | 0.27 | 0.20 | 0.52 / 0.58 |
| obj | 0.30 | 0.31 | 0.58 / 0.54 |
| comp | 0.43 | 0.41 | 0.74 / 0.83 |
| empty | 0.76 | 1.03 | 0.88 / 0.92 |

- Same ranking in both models: act < obj < comp < empty, matching LIBERO-Para's finding that object changes hurt more than verb changes. A paraphrase moves the action chunk 20-43% of the way to a different task; the empty instruction moves OFT's chunk as far as a different task does.
- The action-slot representation moves more (R 0.5-0.8) than the action chunk (0.2-0.4): the head is more invariant than its input.
- **Caveat:** OFT's action-slot R exceeds 1 in layers 3-9 (up to 5.3 for act). Paraphrases change the prompt length, which shifts the RoPE positions of every action slot; other tasks' instructions differ in length less. Early/middle-layer R is therefore confounded by length; compare length-matched pairs before interpreting it.
- Not yet done: CIs, per-paraphrase R_action vs that paraphrase's LIBERO-Para success (does R_action predict failure?), and the two bddl-wording sentences.

## 2026-10-16 LIBERO re-samples fixture placement on every reset (diag_restore, job 49316115)

- Two restores of one saved state: qpos identical, both camera images different. Between the two hard resets the model's `body_pos` changed for `wooden_cabinet_1_main` (1.7 cm), `wine_rack_1_main` (1.5 cm) and `flat_stove_1_main` (0.6 cm). Fixtures have no joints, so their placement is not in `get_sim_state()`; LIBERO samples it again at every hard reset. Restoring without a reset gives identical observations, so nothing else is hidden.
- The official eval path (reset + init state 0, twice) also gives different images: fixture jitter is part of LIBERO's evaluation distribution, and an init state does not pin the scene. This does not affect E0, but it does mean "the same init state" in LIBERO-Para's protocol is not the same scene twice.
- Fix: `snapshot` saves `body_pos` / `body_quat`; `restore` writes them back after the reset and before `set_init_state` (which runs `mj_forward`). States collected before this fix cannot be restored faithfully; recollect.

## 2026-10-16 E3 restore verified bitwise

- Pilots 49316500 (VLA-Adapter) / 49316501 (OFT), tasks 2 and 4: two restores give identical observations, the policy is repeatable, and `orig` vs `orig_rep` are bitwise identical on every state (first_diff_step None). With fixture placement and gripper state restored, E3 branches are exact counterfactuals; `orig_rep` stays as a running check.

## 2026-10-17 pi0.5 in E3/E4 (`--backend lerobot`)

- `langeval/adapters/lerobot_policy.py` runs lerobot-eval's own per-step pipeline (format raw obs -> preprocess_observation -> env preprocessor -> policy preprocessor -> select_action x n_action_steps -> postprocessors) on LeRobot's own `LiberoEnv` (360x360, relative control), so preprocessing cannot drift from the E0 numbers. The scripts drive its inner OffScreenRenderEnv, so snapshot/restore and the goal predicates are unchanged.
- Init states come from LeRobot's loader (weights_only=False; hf-libero's own `get_task_init_states` fails under torch >= 2.6).
- Noise: the k-th inference of each (re)started rollout uses torch seed (seed, k), reset by `policy.begin()` in `branch.rollout`. Continuations from one state share noise draws (common random numbers), so orig vs orig_rep stays a bitwise check and instruction contrasts are not diluted by sampling noise. Independent replicates = different `--seed`.
- **Status (2026-10-18): validated on the cluster** (pilot 49400999, tasks 2 and 4, 12 grasp states): two restores give identical observations, the policy is repeatable under the same noise seed, and orig vs orig_rep are bitwise identical on all 12 states. Before that, the first pilot found no grasp states: every rollout raised `_quat2axisangle expected shape (B, 4)`, because lerobot-eval's vector env gives every observation leaf a batch dim and our direct env does not; the adapter now adds it. Original status line: Next step is the E3 pilot (tasks 2, 4; 3 states per source) with three gates before any full run: two restores give identical observations; with the same noise seed, one observation gives identical actions twice; orig and orig_rep are bitwise identical. Until then no pi0.5 E3/E4 number is reported.

## 2026-10-17 Fixture placement seeded per (task, init)

- `env.reset()` re-samples fixture placement from numpy's global RNG, so in one process the scene for init k depended on how many resets ran before it. E4 runs with different tau lists (batch_1017 vs batch_1017c frac) disagreed on 98 / 928 tau=0 pairs for the same (task, init, continuation). Each run's restores were still exact; the source scenes were different samples.
- Fix: `settle(..., seed=seed*100000 + task*1000 + init)` seeds numpy before the reset (E3 and E4). From now on the same init is the same scene and source rollout in every design, which the before/after fine-tuning comparison needs. Earlier results stay valid as independent samples and are not paired across runs.
- E3 grasp collection now prints rollout errors (the pi0.5 pilot found 0 grasp states in 40 attempts with no message).

## 2026-10-18 Spatial-language transfer study: StarVLA + Qwen3-VL

- Backbone: Qwen3-VL-4B-Instruct, not a hand-assembled MoonViT-V2 + Qwen3-4B (MoonViT-V2 is trained jointly with Kimi K3's LM; pairing it with Qwen3 would need a new large vision-language alignment run). Qwen3-VL-4B is also VLM³'s base.
- Codebase: StarVLA @ 985c223 (starVLA_dev), QwenOFT (deterministic L1 head, so E3/E4 bitwise checks still apply). Released `StarVLA/Qwen3-VL-OFT-LIBERO-4in1` (Goal 96.2 published) validates our client before any training. Our own runs train on LIBERO-Goal only: 4-in-1 data includes LIBERO-10, which would make some "novel" combinations not novel.
- Attention `sdpa` in training and serving: no flash-attn wheel for aarch64 + torch 2.6 on DeltaAI, and one implementation on both clusters. decord is dropped from StarVLA's requirements (no aarch64 wheel, unused with video_backend torchvision_av). pytorch3d is needed: the policy server imports `pytorch3d.transforms` through the LeRobot dataloader (first StarVLA jobs 49405544/5 died on it); x86 uses the pinned pipablepytorch3d wheel, aarch64 builds v0.7.8 from source. setup_starvla.sh now imports the server's modules as its final check.
- Serving: the model runs in the `starvla` env behind StarVLA's own websocket server; the LIBERO client is `langeval/adapters/starvla.py` in the `vla` env, so all models share one simulator build (mujoco 2.3.7). The client copies StarVLA's LIBERO eval preprocessing exactly (flip, 224 resize, no state, open-loop chunk, gripper binarization).
- Novel goals reuse the shared LIBERO-Goal scene (all 10 bddl files share regions/objects/init) with new goal predicates; no new bddl files, no new init states.
- QA data comes from fresh random placements (`env.reset()` with seeds disjoint from evaluation), never from the eval init states.
- Knowledge-insulation (stop-gradient into the VLM) is not implemented: with an OFT head it reduces to a frozen-for-actions VLM; the faithful version needs discrete action tokens (FAST). Revisit if co-training (A+co) helps.

## 2026-10-18 Novel-goal evaluation stopped on any goal (fixed before any result was read)

- `eval_goals.py` passed all 19 goals to `GoalTracker` and stopped the rollout when *any* of them held for 10 steps. Opening the top drawer satisfies n7 ("open the top drawer") on the way to n6 (drawer + cream cheese) and to seen task 3 (drawer + bowl), so those rollouts ended early as failures. Found in review while jobs 49404858-61 were still pending; they were cancelled and resubmitted as `batch_1018c`.
- Now: the rollout ends when the instructed goal's predicates hold (LIBERO's `done`, as upstream eval) or at the step limit; success is that, independent of which other goal completed first. Other goals are recorded only (`GoalTracker.completed`). E3 / E4 are unaffected: their goal sets are the 10 seen goals, none of which is an intermediate step of another.

## 2026-10-18 QA data: collect / build split, coordinate check, held-out scenes

- `make_spatial_qa.py collect` stores robosuite's raw projection `rc` = (row, col); `build` writes the llava jsons. robosuite 1.4's projection is in its own image convention (rows of `render()[::-1]`), and the saved images are the observation flipped `[::-1, ::-1]`; by that reading only the column should flip, but the first collect flipped both. The convention is decided on the saved images (`build --overlay`, four conventions drawn in four colours), not from code. Frames from the first collect stored the both-flipped `px`; `build` recovers `rc` from it.
- Train / held-out split by scene (scene % 5 == 4 held out), so neighbouring frames never straddle the split. StarVLA's eval set is now the held-out file; `eval_qa.py` measures held-out accuracy of the raw VLM, after stage 1 and after stage 2.
- Arm C's answers are full sentences so their length is close to arm B's; arm D's next-step answer carries the object, like B's sentence.
