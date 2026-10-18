# Language and VLAs: research design (zero-training experiments on Bridges-2)

> Written 2026-10-13. The literature section is based on arXiv abstracts and search snippets **as of 2026-10-13**. Anything marked "per the abstract" has not been checked against the full paper. Read the full paper before citing it in a paper of our own.
> Full-text update: 2609.39971v1 and 2610.10178v1 have now been read, including appendices. Their corrected scope and the current research priorities are recorded in [RESEARCH_PLAN.md, §2.3](RESEARCH_PLAN.md). Other literature claims below remain preliminary.
> 2026-10-18: the main question has moved to transfer of spatial-language knowledge into action (Qwen3-VL / StarVLA); see [RESEARCH_PLAN.md, §11](RESEARCH_PLAN.md). This document remains the design record for phase 1 (E0–E8).
> All cost figures are estimates. They must be recalibrated with the measured throughput from E0.

---

## 0. Conclusion first

**The literature has already established that "VLAs ignore language on LIBERO".** Six papers (LIBERO-PRO / Plus / CF / LangGap / Para, LangForce) all show it, and the cause is clear: in the training data H(instruction | image) ≈ 0. Repeating "drop the language and see what happens" adds nothing new.

**The real disagreement is between two newer accounts:**
- *Weighted down but encoded*: 2603.19233 reports that every architecture encodes the instruction internally, but when the scene alone determines the goal, language is weighted below vision.
- *Used, but not bound to the action*: 2609.39971 calls it instruction-action binding. Failed rollouts repeat the source behavior or switch to another demonstrated task, so the instruction acts as a key that retrieves a trajectory.

**Neither account says when language acts, and nobody has measured both kinds of information in the same units across models.** That is where we come in.

**Proposed main question:**

> **In a scene where language is necessary, when does a VLA read its instruction: once at the start to pick a trajectory, or continuously throughout execution? And when it fails, is the information lost in the representation, or present but unused?**

The phenomenon you observed falls under this question: after a paraphrase the correct object is already lifted, yet the task still fails. Under "read once, then commit", the post-grasp phase should not depend on the meaning of the language. If the paraphrase still causes post-grasp failure, there are only two possibilities:
- (a) the post-grasp phase does keep reading language;
- (b) the paraphrase is acting as an out-of-distribution input perturbation, not through its meaning.

E3 separates these two.

---

## 1. Literature map

| Paper | Models | What it shows | What it does not do (our opening) |
|---|---|---|---|
| LIBERO-PRO [2510.03827] | several | With swapped objects or corrupted/random instructions, models still grasp the old target | Behavior only; no mechanism, no timing |
| LIBERO-Plus [2510.13626] | several | Of 7 perturbation types, language hurts least | Same as above |
| LIBERO-CF [2602.17659] | π0.5 | Counterfactual instructions in familiar scenes → the model repeats training behavior and picks frequent objects. CAG (dual branch with a vision-only module) improves language following: +9.7 training-free, +15.5 with the VA module | Accuracy only, no information quantity; does not analyze **which** "default task" gets executed |
| LangGap [2603.00592] | π0.5 etc. | 59 new tasks in the same scenes; π0.5 drops from 93.8% to 21.4% | Behavior only |
| LIBERO-Para [2603.28301] | 7 VLAs | Paraphrase costs −22 to −52 pts, driven mainly by object synonyms; per the abstract, 80–96% of failures are **planning-level trajectory divergence** | **No phase-by-phase breakdown**. Your observation that the object gets lifted but the task fails is in tension with "mostly planning-level", which needs a state-matched experiment. No control separating "meaning changed" from "input perturbation" |
| LangForce / BayesianVLA [2601.15197] | starVLA | Names the cause "information collapse"; trains to maximize conditional PMI | A training method; does not measure where existing models lose the information |
| Not All Features… [2603.19233] | 6 VLAs (π0.5, X-VLA, SmolVLA…) | Injecting task A's activations into π0.5's action expert executes A in 99.6% of episodes; goal selection and motor generation are split between the VLM and the action expert; all models encode the instruction | No quantity comparable to behavior; no time dimension |
| [Do VLAs Understand Instructions? (2610.10178v1)](https://arxiv.org/html/2610.10178v1) | π0.5, GR00T N1.7 | Offline LIBERO-10 patching and CKA; full-text notes in RESEARCH_PLAN §2.3 | No closed-loop, phase-aligned instruction-switching evaluation |
| [When Instructions Retrieve Trajectories (2609.39971v1)](https://arxiv.org/html/2609.39971v1) | π0.5, GR00T N1.7 | Behavioral and internal-state interventions; ECT; full-text notes in RESEARCH_PLAN §2.3 | Verb/object/destination changes are already tested; the remaining question here is phase-dependent redirection and matched-state default destinations |
| Scrubbing Visual Cues [2610.10912] (2026-10-07) | several | Visual shortcuts enter the action representation in early layers; later layers correct with language to different degrees; proposes the rollout-free metric action margin | Training method; abstract only |
| SwitchVLA [2506.03574] | — | Mid-execution instruction switching usually fails; adding contact-phase information raises switching success from 0% to 95.8% | A training method; supports the hypothesis that language has stopped being read after contact, but not measured on existing models |
| ProbeAct [2606.09740] | OpenVLA-OFT | Hidden-state probes predict object positions; failure detection by grasp/transport/place phase | Not about language, but its phase state machine can be reused |
| Tri-Info [2606.19998] | π0 | Information-theoretic failure prediction; names the "phantom grasp" | Not about language |

**One-sentence summary:** the behavioral level ("language is barely used") is saturated, and the mechanistic level ("which layer, which module") has just started (October 2026). **The time dimension, quantities comparable to behavior, and causal attribution of failures at matched states are all still empty.**

---

## 2. Gaps (each item is checked against §1)

| # | Gap | Why it is real | Corresponding experiment |
|---|---|---|---|
| G1 | **No common unit**. Every paper reports success-rate drops under different perturbations, so they cannot be compared across models or benchmarks | §1: all the behavioral papers use success rate; LIBERO-CF uses accuracy | E1 behavioral bits (bounded above by log₂10, with confusion matrix) |
| G2 | **Bit-scale comparison of representation and behavior** (deferred) | The two fully read papers already quantify readouts or intervention effects; see the revised scope in RESEARCH_PLAN §2.2 | E5 representation bits vs E1 behavioral bits |
| G3 | **The time dimension**: when language can still redirect behavior | Neither fully read paper measures phase-aligned instruction-switching curves; broader priority remains to be checked | E4 switching at τ; E2 as needed |
| G4 | **Is paraphrase failure about meaning or about perturbation?** And at which phase? | LIBERO-Para has no matched-distance control; it attributes failure to planning, but your data shows post-grasp failure | E3 state-branching + E6 representation invariance |
| G5 | **Default destinations after destination-word failure** | Word-category interventions are already covered; focus on matched intermediate states, timing, and comparison with empty instructions | E3 nonce-word branches; E7 deferred |
| G6 | **Data vs architecture**: is this a property of the model or of the training data? | Public checkpoints already provide a natural experiment (§4: OFT has both per-suite and four-suite-joint versions) | Every experiment runs on both |

---

## 3. How to choose a good question

A question worth doing has to pass four tests:

1. **Competing hypotheses make different predictions.** "Does the VLA use language?" fails: the answer is "a little, depending on the data". "Read once and commit, or read continuously?" passes: E4 gives opposite predictions under the two hypotheses.
2. **The answer changes what people do.** If language is only read at the start, a method like CFG/CAG that strengthens language throughout the episode is wasting compute after the commitment point, and the fix belongs in the decision before commitment. If post-grasp failure comes from perturbation, the fix is input robustness, not language understanding.
3. **It can be done with public resources.** No training, done within 1000 SU (§6).
4. **Nobody has done it.** It is positioned against the "what it does not do" column in §1.

Candidate questions, ranked by these four tests:

| Rank | Question | Novelty | Cost | Risk |
|---|---|---|---|---|
| **1** | **When language is read (commitment time) + attributing post-grasp failure at matched states** | High (G3, G4) | Medium | Low: some result comes out either way |
| 2 | Which word is the retrieval key (compositional new goals) | High (G5) | Low | Medium: new goals need to be physically feasible, which has to be checked first |
| 3 | Behavioral bits vs representation bits across 4 architecture families | Medium (G1, G2); 2603.19233 / 2610.10178 are close | Low | Low |
| 4 | Is paraphrase invariance lost at a specific layer? | Medium (G4) | Very low (forward passes only) | Low |

**Recommendation:** make 1 the main line, with 2 and 3 as supporting evidence (they share the same rollout infrastructure). 4 is nearly free, so do it as well.

---

## 4. Hypotheses and competing predictions

| Hypothesis | Prediction if it holds | Prediction if it fails |
|---|---|---|
| **H1 Commitment**: language is read before the grasp to pick a trajectory; afterwards actions barely depend on it | S(t) drops sharply around the grasp moment; switching the instruction after the grasp leaves the executed task unchanged (follows the original) | S(t) stays high; after a switch the robot changes target (put back down / carry to the new location) |
| **H2 Post-grasp failure is a perturbation effect**: the paraphrase acts as an out-of-distribution input, not as a change of meaning | At the same grasp state, failure rates for paraphrase ≈ meaning-irrelevant perturbation ≈ (perhaps) empty instruction, all > original instruction | Paraphrase failure rate > meaning-irrelevant perturbation at matched distance |
| **H3 Retrieval key**: a new instruction is executed as the training task closest in its "key word" | For compositional new goals the confusion matrix concentrates on a training task, determined by the object word (consistent with LIBERO-Para) | The robot executes the new goal, or picks the target by verb or location word |
| **H4 Present but unused**: action-query positions can decode the task while behavioral bits are low | I_repr(action query) ≫ I_behav | Both are low: the information is already lost in the representation |

H1 and H2 can hold together: post-grasp behavior does not read the meaning but is knocked off course by out-of-distribution tokens. **E3 + E4 are designed to separate these combinations.**

---

## 5. Public resources

### 5.1 Checkpoints (2026-10-13, existence confirmed via the HF API)

Covers three action-generation families. **Tier 1** is required; Tier 2 is added if budget allows.

| Tier | Model | HF repo | Family | Why |
|---|---|---|---|---|
| 1 | VLA-Adapter (Qwen2.5-0.5B) | `VLA-Adapter/LIBERO-Goal-Pro` @2ae84b8 | Parallel L1 regression, bridge attention | Already wired in; small and fast; deterministic |
| 1 | OpenVLA-OFT (Llama2-7B), **Goal only** | `moojink/openvla-7b-oft-finetuned-libero-goal` @c2d0f9f | Parallel L1, bidirectional attention | VLA-Adapter derives from the OFT code, so the adapter is nearly reused |
| 1 | OpenVLA-OFT, **four suites joint** | `moojink/openvla-7b-oft-finetuned-libero-spatial-object-goal-10` | Same as above | **Natural experiment for G6**: same architecture, different language entropy in training data |
| 1 | π0.5 (LeRobot PyTorch) | `lerobot/pi05_libero_finetuned_v044` | VLM + flow-matching action expert | Shared model family with the two fully read papers; checkpoint, suite, and metric differences prevent direct score comparison; stochastic sampling yields distributions |
| 2 | SmolVLA | `HuggingFaceVLA/smolvla_libero` | Small VLM + flow expert | Small and cheap |
| 2 | GR00T N1.7 | `nvidia/gr00t17-lerobot-libero_goal-640` | Cross-attention action module | Also studied by 2610.10178 / 2609.39971 |
| 2 | X-VLA | `lerobot/xvla-libero` | Soft prompt + flow | Also studied by 2603.19233 |
| Control | VLA-Adapter Spatial-Pro / OFT spatial | `VLA-Adapter/LIBERO-Spatial-Pro`, `moojink/openvla-7b-oft-finetuned-libero-spatial` | — | The H≈0 control |

Not chosen: original OpenVLA (autoregressive, one step at a time, slow; weaker on Goal). Before each Tier 2 model is used, confirm that the repo really is a LIBERO fine-tune and record its evaluation protocol.

### 5.2 Benchmarks and assets

- **LIBERO-Goal** (main setting): the 10 tasks share objects and initialization regions. Objects: bowl, cream cheese, wine bottle, plate; fixtures: cabinet (drawers / top), stove, wine rack.
- **LIBERO-Spatial** (H≈0 control).
- **LIBERO-Para** (github.com/cau-hai-lab/LIBERO-Para): use its paraphrases directly, to avoid writing our own and introducing bias. Its paraphrases are built on the 10 LIBERO-Goal scenes (4092 entries), and its repo already integrates eval scripts for VLA-Adapter, OFT (Goal / Mixed), π0.5 and X-VLA. **Protocol limit**: one episode per paraphrase, always from init state 0; on deterministic models each (scene, paraphrase) is one trajectory, so our E1 paraphrase mode varies the init state (see DECISIONS 2026-10-14).
- LIBERO-CF (vla-cf.github.io), LangGap (99 tasks / 59 in the same scenes) and LIBERO-PRO: as external validation if their code is public. Do not depend on them.

### 5.3 Engineering consequences

- Bridges-2 is x86, so the upstream environments install as is. Two environments are needed: `vla-oft` (VLA-Adapter + OFT, torch 2.2, transformers fork) and `lerobot` (π0.5, SmolVLA, X-VLA, GR00T). The `PolicyAdapter` interface in IMPLEMENTATION_PLAN §4.1 exists exactly for this.
- Evaluation uses **L40S (1 SU/GPU-hour) + the `GPU-shared` partition**. H100 costs twice as much, and rollouts are bottlenecked by CPU simulation.
- **Determinism**: VLA-Adapter / OFT are deterministic regressors. Restoring the same state with the same instruction gives the same trajectory, so repeat counts must come from different states, and statistics are bootstrapped over states. π0.5 / SmolVLA sample, so each state can be rolled out several times with fixed seeds.

---

## 6. Experiments

Shared infrastructure, built once:
- `goal_matrix`: an executed-task classifier (IMPLEMENTATION_PLAN §4.3–4.4) that decides which of the 10 goals was achieved, or none.
- `phase`: phase labels. Approach → grasp (target lifted ≥ h cm and gripper in contact, held for k steps) → transport → place. These are kinematics plus object state, borrowing ProbeAct's state machine.
- `branch`: save and restore full sim state, `env.sim.get_state().flatten()` → `env.set_init_state()`. On restore, clear the action queue. **Self-check: same state + same instruction + deterministic model, run twice → trajectories bit-identical.**
- Failure taxonomy: success / dropped / misplaced / stalled / wrong object / timeout, judged automatically by predicates.

### E0 Reproduction and throughput calibration (must come first)
- Run the official protocol on Goal for each Tier 1 model (500 rollouts) and compare with the officially reported success rate. If it does not match within the CI, debug before moving on.
- Measure rollouts/GPU-hour per model and the best number of processes per card. **Every cost figure in this document is then rescaled by this number.**

### E1 Behavioral bits and confusion matrix (G1, G6)
- 4 models × instruction modes {orig, empty, other, shuffle, paraphrase (LIBERO-Para)} × 10 tasks × 20 initial states.
- Metrics: confusion matrix C[i,k], follow_rate, I_behav (with CI), default_mass (which task the off-diagonal mass concentrates on, i.e. the "default task" in LIBERO-CF).
- **Key comparison**: OFT Goal-only vs OFT four-suite joint. Same architecture; does training data with more language entropy raise I_behav?

### E2 Time-resolved sensitivity S(t) (G3, H1)
- On orig rollouts, at each inference step t, freeze the observation and feed in each of the 10 instructions; S(t) = mean pairwise distance between the action chunks, normalized.
- **Align to phase**: plot S against time relative to the grasp moment, not absolute t. The commitment point t* is where S first stays below threshold.
- Forward passes only; cost is close to that of a normal rollout.

### E3 State-branching attribution of post-grasp failure (G4, H2), the core experiment for your phenomenon

**Design: a 2×2 plus controls.**
1. **Collect grasp states**: on LIBERO-Goal (main) and Spatial task 2 (where the phenomenon was first seen), run rollouts with the original instruction **and** with the paraphrase. Save the sim state at the moment of stable grasp. About 20 states per (model, task, source instruction).
2. **Branch**: from each state, continue with each of the following instructions:

| Continuation instruction | Purpose |
|---|---|
| orig | Baseline |
| 2–3 paraphrases (one phrase changed each; tag verb / object / location) | The effect of interest |
| A meaning-irrelevant perturbation at matched distance (insert a filler word, a typo, swap an article) | Separates meaning from perturbation (H2) |
| empty | Whether post-grasp behavior needs language at all |
| A different task's instruction | Can language still redirect the plan after the grasp (H1) |

3. **2×2 on source**: grasp state comes from {orig rollout, paraphrase rollout} × continuation instruction {orig, paraphrase}.
   - Paraphrase-sourced state continued with orig still fails → the grasp pose is the problem.
   - Orig-sourced state continued with paraphrase fails → the post-grasp phase is the problem.
4. **"Distance matching"** is done in the model's own input-embedding space: the mean token-embedding distance between perturbation and original instruction should be comparable to the paraphrase's. Report it alongside the results.

Reading the results:
- Paraphrase ≈ perturbation > orig → H2 (perturbation effect). The follow-up then becomes "robustness of post-grasp control to input perturbation", not language understanding.
- Paraphrase > perturbation → post-grasp behavior really reads meaning. This contradicts H1 and would be the headline finding.
- A different task's instruction can change where the object is placed after the grasp → language is still read late.

### E4 Switching the instruction mid-episode (H1)
- From orig rollouts, switch from i to j at τ ∈ {before approach, at contact, after grasp, during transport} (aligned by phase).
- The `goal_matrix` decides whether the robot ended up executing i or j. Plot "fraction following the new instruction vs τ". Under H1 this curve should fall to zero before the grasp.
- Only pairs that share the same object are informative (e.g. bowl → plate vs bowl → stove). Pairs with different objects confound "replanning" with "already holding the wrong thing".

### E5 Representation bits (G2, H4)
- IMPLEMENTATION_PLAN §6.2: GroupKFold probes by token position class (vision / language / action query), with T0 = 5 frames as the main analysis plus a time curve, and shuffled-label baselines.
- Also on rollout frames **aligned by phase**: before vs after grasp, can action-query positions still decode the task? If the representation keeps the task after the grasp while behavior no longer responds (E2), that is direct evidence of "present but unused".

### E6 Layer-wise paraphrase invariance (G4), forward passes only, nearly free
- For each layer and position class, compute R = d(h(orig), h(paraphrase)) / d(h(orig), h(other task)) on the same frame.
- R ≈ 0: the model treats the paraphrase as the same task; R ≈ 1: it treats it as a different task. Plot R by layer and see where "synonyms become different".
- The object-synonym vs verb split matches LIBERO-Para, which makes cross-checking easy.

### E7 Compositional new goals: what the retrieval key is (G5, H3)
- In LIBERO-Goal's fixed scene, write **evaluation-only** bddl files that combine seen objects with seen targets in unseen pairings. For example:
  - `put the cream cheese on the plate` (object from "cheese→bowl", target from "bowl→plate")
  - `put the cream cheese on top of the cabinet`
  - `put the wine bottle on the plate` / `put the cream cheese on the stove`
- Check feasibility first: place the object at the target with a scripted state and confirm the predicate can be satisfied. Drop any goal that cannot.
- The executed-task classifier records which of the 10 training tasks gets executed, or the new goal, or none. Then regress: is the executed task determined by the **object word** (cheese → "put cheese in bowl") or by the **target word** (plate → "bowl to plate")?
- No training, and directly addresses 2609.39971's "keying" and LIBERO-Para's "object words matter most".

### Optional: E8 Inference-time language guidance (CAG/CFG, ω sweep)
Use as an intervention test of H1. If H1 holds, raising ω only before the grasp should be about as effective as raising it throughout. Do this only after E2 gives t*.

---

## 7. Budget and order (Bridges-2, 1000 SU)

Cost is measured in rollouts and converted with E0's measured rate r (rollouts per GPU-hour). The Pion README reports about 500 rollouts per hour on a single H100/A100 for VLA-Adapter. π0.5 / OFT-7B are expected to be several times slower, and L40S figures **have not been measured**.

| Step | Experiment | Rollouts (Tier 1, 4 models) | Notes |
|---|---|---|---|
| 1 | E0 reproduction + throughput | 4 × 500 = 2,000 | Go/no-go gate |
| 2 | E6 paraphrase invariance | 0 (forward passes) | Can run in parallel with E0 |
| 3 | E1 confusion matrix | 4 × 5 × 200 = 4,000 | |
| 4 | E3 state branching | Collection ≈ 4 × 10 × 2 × 20 = 1,600; branches ≈ 4 × 10 × 40 × 7 ≈ 11,200 (post-grasp segment only, about half length) | Core |
| 5 | E2 S(t) | ≈ 4 × 200 rollouts with 10× forward passes per step | |
| 6 | E4 switching | ≈ 4 × 4 τ × 20 pairs × 10 ≈ 3,200 | |
| 7 | E7 compositional goals | ≈ 4 × 4 goals × 50 = 800 | Feasibility check first |
| 8 | E5 representation bits | Forward passes + storage | |

**Total about 25k rollouts.** If the average throughput on L40S is 100–300 rollouts per card-hour, that is roughly 80–250 SU, plus failed reruns and the forward-pass experiments, keeping total usage within a few hundred SU. This is an estimate and is recalculated after E0. **Tier 2 models are added only after the Tier 1 conclusions are clear.**

Gates:
- If E0 cannot reproduce official numbers → stop and debug.
- If E3's "same-state reproducibility" self-check fails → stop. No conclusion from E3 or E4 is valid without it.

---

## 8. Risks and what we are not doing

- **Selection bias in grasp states**: states are taken only from successful grasps. E3's 2×2 on source mitigates this. Also report the grasp success rate itself.
- **Absolute-coordinate leakage**: 2603.19233 finds injected programs are bound to absolute workspace coordinates. In a fixed LIBERO-Goal scene, "executing task k" may simply mean "running coordinate trajectory k". E7 must hold the scene fixed, which is consistent with this.
- **Paraphrase quality**: use LIBERO-Para's paraphrases. When writing our own, change only one phrase each time and keep a single fixed file.
- **Multiple comparisons**: before running, write H1–H4 and their decision rules into `docs/findings.md` (preregistration). Exploratory findings beyond that are labeled as such.
- **Not doing**: any training (that goes to DeltaAI later); OpenVLA original; per-model hyperparameter search.

---

## 9. Relation to IMPLEMENTATION_PLAN.md

- E0 / E1 / E5 / E8 are already IMPLEMENTATION_PLAN step 3–7; nothing changes there.
- New: `langeval/phase.py` (phase labels), `langeval/branch.py` (state save/restore and branching, E3/E4), `instructions.py` gains `paraphrase` / `perturb` modes (reading a fixed file), and the evaluation-only bddl files for E7.
- New adapters: `adapters/oft.py` (nearly a copy of VLA-Adapter's) and `adapters/lerobot.py` (π0.5 / SmolVLA / X-VLA / GR00T).
- The cluster entry point gets a new `bridges2/` (x86, L40S, `GPU-shared`), parallel to `deltaai/`.

---

## Sources

- LIBERO-PRO https://arxiv.org/abs/2510.03827
- LIBERO-Plus https://arxiv.org/abs/2510.13626
- LIBERO-CF / CAG https://arxiv.org/abs/2602.17659 , https://vla-cf.github.io
- LangGap https://arxiv.org/abs/2603.00592
- LIBERO-Para https://arxiv.org/abs/2603.28301 , https://github.com/cau-hai-lab/LIBERO-Para
- LangForce / BayesianVLA https://arxiv.org/abs/2601.15197
- Not All Features Are Created Equal https://arxiv.org/abs/2603.19233
- Do VLAs Understand Instructions? https://arxiv.org/abs/2610.10178
- When Instructions Retrieve Trajectories https://arxiv.org/abs/2609.39971
- Scrubbing Visual Cues https://arxiv.org/abs/2610.10912
- SwitchVLA https://arxiv.org/abs/2506.03574
- ProbeAct https://arxiv.org/abs/2606.09740
- Tri-Info https://arxiv.org/html/2606.19998v2
- Mechanistic Interpretability for Steering VLAs https://arxiv.org/abs/2509.00328
- Embodied Interpretability (ISS/NMR) https://arxiv.org/abs/2605.00321
- LIBERO-VIFO https://arxiv.org/abs/2608.17600
- Bridges-2 GPU https://support.access-ci.org/documentation/resources/bridges-2-gpu
