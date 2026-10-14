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
