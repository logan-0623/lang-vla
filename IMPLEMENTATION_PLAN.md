# lang-vla：VLA 语言依赖的统一评测与可解释性实验 —— 实现计划

> 本文件是交给 Claude Code CLI 的实现说明。请按阶段顺序实现，每个阶段完成后先跑该阶段的验收检查，再进入下一阶段。
> 代码标识符、文件名、日志一律用英文；注释可以用英文。

---

## 0. 给实现者（Claude CLI）的总规则

1. **不要编造任何结果或数字。** 所有表格和图只能来自实际运行产生的文件。某一步无法验证时，停下来，在 `docs/STATUS.md` 里写明卡在哪里。
2. **不要改动 Pion 上游代码的无关部分。** 对 `Pion/` 的修改只能出现在本文件"允许修改的文件"清单里，并且全部提交在 `Pion` 仓库的 `lang-spectral` 分支上。
3. **本机（macOS）只做开发和 CPU 单元测试；训练和 LIBERO rollout 在 NCSA DeltaAI（aarch64 Grace + H100-96GB，每节点 4 卡，Slurm）上跑，所有集群入口都在 `deltaai/`（见 §3.1）。** 所有 GPU 脚本都必须能用 `--dry_run` 或极小规模参数在单卡上冒烟测试。
4. **每个新功能都要有 CPU 可跑的单元测试**（`pytest -q tests/`），不能依赖 GPU、flash-attn 或 LIBERO 仿真。需要 LIBERO 的测试用 `pytest.mark.libero` 标记，默认跳过。
5. **每次运行都要落盘完整配置**（JSON，含 git commit、所有参数、随机种子），输出目录不得覆盖已存在的结果。
6. **评测循环里的异常不能被静默吞掉。** 上游 `run_libero_eval.py` 的 `run_episode` 会捕获异常并返回失败（`except Exception: log_message(...)`）。我们的评测必须把"出错的 episode"单独计数并报告，不能算作普通失败。
7. 遇到本文件没有覆盖的设计决定，选最简单、最容易验证的做法，并在 `docs/DECISIONS.md` 里记一行。

---

## 1. 研究目标（实现者需要理解的背景）

VLA 在 LIBERO 上成功率很高，但大量证据表明它们主要靠视觉捷径，很少真正使用语言指令。根本原因是训练数据里 H(指令 | 画面) ≈ 0：画面已经决定了任务，语言不提供额外信息。

本项目做两件事：

1. **统一评测**：把现有的"让语言更有用"的方法，放在同一个语言必需的设定（LIBERO-Goal）和同一套指标下比较。
2. **可解释性测量**：对每个方法，同时测
   - **行为比特**：模型的行为里实际用了多少语言信息；
   - **表征比特**：模型内部各层、各类 token 位置上保留了多少语言信息。

两者的差距用来区分"信息在但没被用"和"信息本身丢了"。

**为什么用 LIBERO-Goal**：它的 10 个任务的物体集合和初始化区域完全相同（已核对 bddl 文件），只有目标谓词不同。所以：
- 只看画面无法确定任务，H(指令 | 画面) = log₂10 ≈ 3.32 bit；
- 任何一条 rollout 都可以同时判定 10 个目标各自是否完成，从而得到"实际执行了哪个任务"。

LIBERO-Spatial 作为对照（每个布局对应一个任务，H ≈ 0）。

---

## 2. 仓库结构

当前状态：

```
lang-vla/
  Pion/                     # git clone of https://github.com/OPTML-Group/Pion （上游 commit 8796501）
  vla-language-gap.html
```

目标结构：

```
lang-vla/
  README.md
  pyproject.toml            # 只声明 langeval 包；不安装 Pion 的依赖
  patches/
    pion-lang-spectral.patch      # 已提供，见 §3.2
  Pion/                     # 保持独立 git 仓库；在 lang-spectral 分支上工作
  langeval/                 # 本项目自己的代码，与具体模型解耦
    __init__.py
    instructions.py         # 指令变换：orig / empty / other / shuffle（以及后续的 cipher / key）
    goal_predicates.py      # LIBERO-Goal 10 个目标谓词的读取与判定
    goal_matrix.py          # 方向 A：混淆矩阵 rollout 主循环
    metrics.py              # 行为比特、指令遵循率、bootstrap CI
    guidance.py             # 推理时语言引导（CFG 形式）
    probes.py               # 表征比特（线性探针）
    patching.py             # 语言 token 激活替换
    sensitivity.py          # 逐时间步的语言敏感度
    attention_check.py      # 判定视觉 token 能否"看到"语言
    adapters/
      __init__.py
      base.py               # PolicyAdapter 抽象接口
      vla_adapter.py        # VLA-Adapter 后端（第一个也是本阶段唯一的后端）
    io.py                   # 运行配置、结果 JSON 的读写与 schema
  scripts/
    eval_goal_matrix.py
    collect_hidden_states.py
    run_probes.py
    run_patching.py
    run_sensitivity.py
    aggregate.py            # 汇总所有结果，输出表格和图
    train_arm.sh            # 训练矩阵入口（包装 Pion/VLA/VLAAdapter/scripts/run_lang_matrix.sh）
  deltaai/                  # 集群专用：config.sh / setup_env.sh / download_assets.sh / *.sbatch（已提供）
  configs/
    arms.yaml               # 所有训练组的定义
    eval.yaml               # 评测参数
  tests/
  docs/
    STATUS.md               # 实现进度与卡点
    DECISIONS.md            # 设计决定记录
    findings.md             # 只记录有结果文件支撑的结论
    vla-language-gap.html   # 从根目录移过来
  results/                  # .gitignore；每次运行一个子目录
```

`langeval` 只依赖 numpy / scipy / scikit-learn / torch；与 VLA-Adapter 相关的导入只出现在 `langeval/adapters/vla_adapter.py` 和 `scripts/` 里，并且是延迟导入（函数内部 import），保证 CPU 单元测试不需要安装 Pion 的环境。

---

## 3. 阶段 0：环境与补丁

### 3.1 环境（NCSA DeltaAI）

**DeltaAI 是 aarch64（ARM Grace CPU）**，上游 README 的版本组合在这里装不上：torch 2.2 没有 aarch64+CUDA 的 pip wheel，flash-attn 2.5.5 只有 x86 wheel。因此统一改用：

| 组件 | 上游 | DeltaAI | 说明 |
|---|---|---|---|
| Python | 3.10 | 3.10 | 不变 |
| torch / torchvision | 2.2.0 / 0.17.0 | 2.7.1+cu128 / 0.22.1 | 第一个有 aarch64 CUDA wheel 的是 2.6.0+cu126；选 2.7.1 是为了配 flash-attn |
| flash-attn | 2.5.5 | 2.6.3（mjun0812 预编译 aarch64 wheel） | |
| tensorflow | 2.15.0 | 2.15.0 → 实际装 `tensorflow-cpu-aws` | 只在 CPU 上读 RLDS，足够 |
| tensorflow_graphics | 2021.12.3 | 同版本 `--no-deps` | 它的 OpenEXR 依赖没有 aarch64 wheel，只用到 `geometry.transformation` |

`deltaai/setup_env.sh` 一次装好全部依赖（包括 LIBERO，并预写 `~/.libero/config.yaml`，否则 LIBERO 首次导入会在批处理作业里卡在 `input()`）。**版本变了，所以阶段 1 的第 1 条验收（复现官方 `env_success`）同时也是环境正确性检查，不能跳过。** transformers fork 在 torch 2.7 上是否正常、`flash_attention_2` 能否加载，以 `deltaai/smoke.sbatch` 的结果为准，失败时写进 `STATUS.md`。

存储布局（`deltaai/config.sh`）：
- 代码：`/projects/<proj>/$USER/lang-vla`（不清理）；
- conda 环境、数据、权重、训练输出、评测结果：`/work/nvme/<proj>/$USER/lang-vla`（`$STORE`）。conda 环境不能放 `$HOME`（100 GB / 75 万文件配额）；`HF_HOME` 也指到 `$STORE`。
- `/tmp` 是节点本地 3.9 TB，作业结束即清空，只放临时文件。

作业规格：`ghx4` 分区最长 48 h；`ghx4-interactive` 最长 2 h、计费 ×2、每人只能有一个作业，只用于调试。GPU 计算节点不保证外网，W&B 默认 `WANDB_MODE=offline`，之后在登录节点 `wandb sync`。

本机只装 `langeval` 的轻量依赖用于单元测试。

使用顺序：
```bash
# 登录节点，仓库根目录
export ACCOUNT=<accounts 命令显示的项目名，形如 bxxx-dtai-gh>
mkdir -p logs
srun -A $ACCOUNT -p ghx4-interactive --gpus=1 --cpus-per-task=16 --mem=64g -t 1:00:00 bash deltaai/setup_env.sh
bash deltaai/download_assets.sh
sbatch -A $ACCOUNT deltaai/smoke.sbatch          # EGL 渲染 + LIBERO + 官方 checkpoint 端到端
```

### 3.2 应用已有补丁

> 已完成（2026-10-10）：`Pion/` 换成上游 8796501 的 git clone，补丁在 `lang-spectral` 分支上提交，`check_math_numpy.py` 通过；见 `docs/DECISIONS.md`。

`patches/pion-lang-spectral.patch` 应基于上游 commit 8796501，内容：
- `VLA/VLAAdapter/lang_spectral/`：
  - `token_types.py`：给 LLM 序列每个位置打 vision / lang / action / other 标签；
  - `grad_probe.py`：按 token 类型分解权重梯度和实际更新（credit / budget / amplification / erank / top_overlap），以及各类 token 表征的 stable rank、erank；
  - `optim_lang.py`：`LangSubspaceTracker` 和 `LangGuard` 优化器；
  - `check_math_numpy.py`：不依赖 torch 的数学自检。
- `vla-scripts/finetune_pion.py`：新增配置项，`optimizer_L` 可选 `LangGuardMuon` / `LangGuardPion`，训练循环接入梯度探针。
- `experiments/robot/libero/run_libero_eval.py`：新增 `--instruction_mode {orig,empty,other,shuffle}`。
- `scripts/run_lang_matrix.sh`、`scripts/eval_lang_modes.sh`、`README_lang_spectral.md`。

```bash
cd Pion
git checkout -b lang-spectral 8796501
git apply --index ../patches/pion-lang-spectral.patch
git commit -m "Apply lang_spectral patch"
python VLA/VLAAdapter/lang_spectral/check_math_numpy.py   # 应输出 "checks passed"
```

### 3.3 资产下载（服务器）

- 数据：`openvla/modified_libero_rlds`，目录按 Pion README 重命名为 `libero_goal_no_noops` 等。
- VLM：`Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b` 放到 `Pion/VLA/VLAAdapter/pretrained_models/`。
- 公开 checkpoint（2026-10-10 从 HF API 核对，`deltaai/download_assets.sh` 按这些 revision 下载；写进 `configs/eval.yaml`）：

| 仓库 | revision | `--use_pro_version` |
|---|---|---|
| `VLA-Adapter/LIBERO-Goal-Pro` | `2ae84b8275a0c35c1b6751573a71468e1c080825` | True（自带日志 `Inference--Goal_Pro--98.2.log`） |
| `VLA-Adapter/LIBERO-Goal` | `d1aa0654ccac22cb7e45a55f21692433e9ae7e63` | False |
| `VLA-Adapter/LIBERO-Spatial-Pro` | `cb023c6771fc62ea729d218f09ac0583c14f59d4` | True |
| `VLA-Adapter/LIBERO-Spatial` | `45caa9bca50d3ea6f8e804fe04c65abac1b69717` | False |
| `openvla/modified_libero_rlds`（数据） | `6ce6aaaaabdbe590b1eef5cd29c0d33f14a08551` | 目录已是 `libero_*_no_noops`，无需改名 |
| `Stanford-ILIAD/prism-qwen25-extra-dinosiglip-224px-0_5b` | `5cfd2cc6da00c06e0be7abf35d43ec792d8e9498` | |

主分析用 Goal-Pro（与训练脚本的 `--use_pro_version True` 一致）。

### 3.4 冒烟测试

按 `Pion/VLA/VLAAdapter/README_lang_spectral.md` 的冒烟测试跑 30 步，确认：
- 日志里出现 `# lang_probe: N linears`；
- `run_dir/lang_probe.jsonl` 存在；
- 同一层内 `lang/budget + vision/budget + action/budget + other/budget ≈ 1`。

**验收**：补丁应用成功；numpy 自检通过；冒烟训练能跑完且生成探针日志。

---

## 4. 阶段 1：评测基础设施（方向 A：混淆矩阵 + 行为比特）

这是后续所有比较的基础，必须最先完成并在公开 checkpoint 上跑通。

### 4.1 `PolicyAdapter` 接口（`langeval/adapters/base.py`）

```python
class PolicyAdapter(Protocol):
    def reset(self) -> None: ...
    def act(self, obs: dict, instruction: str) -> np.ndarray:
        """Return an action chunk of shape (H, 7) in the env's action space (unnormalized),
        before LIBERO's gripper post-processing."""
    def act_with_guidance(self, obs: dict, instruction: str, omega: float,
                          null_instruction: str = "") -> np.ndarray: ...
    def hidden_states(self, obs: dict, instruction: str,
                      layers: list[int]) -> dict:
        """Per-layer hidden states for every LLM position plus a token-type array
        (vision / lang / action / other). No pooling."""
```

`vla_adapter.py` 的实现基于 `Pion/VLA/VLAAdapter/experiments/robot/openvla_utils.py` 的 `get_vla_action` 和 `prismatic/extern/hf/modeling_prismatic.py` 的 `predict_action`。注意：
- Qwen（`use_minivlm=True`）的推理 prompt 是 `get_vla_action` 里拼的 Qwen 聊天模板，指令会被 `.lower()`；训练时的 prompt 在 `prismatic/vla/datasets/datasets.py` 中是 `What action should the robot take to {lang}?`。两边必须一致，不要改动模板本身，只替换 `{lang}` 部分。
- token 类型标签复用补丁中 `lang_spectral/token_types.py` 的 `build_token_types`。

### 4.2 指令变换（`langeval/instructions.py`）

```python
def transform(mode: str, instruction: str, *, task_id: int, all_instructions: list[str],
              shift: int = 1, seed: int = 0) -> str
```
- `orig`：原指令；
- `empty`：空串（模板保留，只把任务描述换成空）；
- `other`：`all_instructions[(task_id + shift) % n]`；
- `shuffle`：同一句话的单词随机重排（固定种子）。

预留 `cipher` 和 `key` 两个模式的接口（阶段 5 才实现），本阶段对它们抛出 `NotImplementedError`。

### 4.3 目标谓词（`langeval/goal_predicates.py`）

- 对 `libero_goal` 的 10 个任务，各自构造一次环境，读取 `env.env.parsed_problem["goal_state"]`，然后关闭环境。缓存成 `list[list[tuple]]`。
- `satisfied(env) -> list[bool]`：对当前仿真状态调用 `env.env._eval_predicate(state)`，对每个任务的所有谓词取合取。
- 写一个启动时的自检：断言 10 个任务的 bddl 文件中 `(:objects ...)` 到 `(:obj_of_interest` 之间以及 `(:init ...)` 部分逐字相同；不同则报错退出。
- 生成一张"目标蕴含表"（一个目标是否可能是另一个目标的中间状态），写入结果目录，供后续解释用。

### 4.4 rollout 主循环（`langeval/goal_matrix.py`）

对每个 (instructed_task i, init_state s)：
1. 用任务 i 的环境和初始状态集合初始化（10 个任务的初始化分布相同）；
2. 给策略的指令是 `transform(mode, instruction_i, ...)`，`orig` 模式下就是指令 i；
3. **关闭环境自带的 done 判定作为终止条件**（上游循环在 `done` 时 break，`done` 只对应环境自己的任务）。每一步调用 `satisfied(env)`；
4. 某个任务 k 的谓词**连续成立 `hold_steps`（默认 10）步**，就记"执行了 k"，然后终止；到达最大步数（Goal 为 300 步，加上开头 `num_steps_wait` 步）仍没有任何任务成立，记为 `none`；
5. 同时记录：最终状态下成立的任务集合、每个任务第一次成立的时间、是否出现异常。

动作处理（归一化、夹爪符号、执行前 `num_open_loop_steps` 步）保持和上游 `run_episode` 完全一致，直接复用上游函数，不要重写。

**输出**（`results/<run_id>/goal_matrix.json`）：
```json
{
  "schema": "goal_matrix_v1",
  "checkpoint": "...", "checkpoint_sha256": "...", "git": {...},
  "instruction_mode": "orig", "guidance": {"omega": 1.0},
  "tasks": ["open the middle layer of the drawer", "..."],
  "episodes": [
    {"instructed": 0, "init_id": 3, "policy_prompt": "...", "executed": 0,
     "first_hold_step": {"0": 112}, "final_satisfied": [0], "error": null}
  ]
}
```

### 4.5 指标（`langeval/metrics.py`）

由 `episodes` 计算：
- 混淆矩阵 C[i, k]，k ∈ {0..9, none}，每行归一化；
- `follow_rate` = 对角线均值（指令 i 时执行了 i）；
- `env_success`：保留和上游一致的成功率，以便和论文数字对照；
- `default_mass`：非对角线概率质量中，最大的那一列占比（是否集中在某个"默认任务"）；
- **行为比特** `I_behav = Σ_i p(i) Σ_k C[i,k] log2(C[i,k]/p(k))`，p(i) 均匀，p(k) = Σ_i p(i) C[i,k]；上限 log2(10)；
- 所有指标都按 episode 做 bootstrap（默认 2000 次）给出 95% CI；`I_behav` 另报 Miller–Madow 偏差校正值。

单元测试至少覆盖：单位阵 C → 3.32 bit；所有行相同 → 0 bit；手工小例子与 scipy 计算一致。

### 4.6 脚本与验收

`scripts/eval_goal_matrix.py --checkpoint ... --mode orig --n_init 20 --shard i/N`
- 并行单位是 (instructed_task, init_id) 的分片，每个分片独立写一个 JSON，最后合并。不要在脚本里自己管理多 GPU：在 DeltaAI 上用 Slurm job array（`--array=0-N`）每个任务 1 卡；
- **每张卡起多个进程**：模型只有 0.5B，rollout 的瓶颈是 CPU 上的 MuJoCo 仿真，而每张 H100 配 72 个 Grace 核。先在冒烟作业里测"每卡 1/2/4/8 个进程"的总吞吐，取最好的写进 `configs/eval.yaml`；
- 支持 `--n_init` 缩小规模（完整规模是 10 指令 × 50 初态 = 500 次 rollout）。

**验收**：
1. 在 VLA-Adapter 官方 LIBERO-Goal checkpoint 上，`orig` 模式下的 `env_success` 与官方报告的数字在 CI 内一致（若不一致，先排查，不要继续）；
2. 输出 `orig / empty / other / shuffle` 四种模式的混淆矩阵热力图和行为比特；
3. `empty` 模式下预期行为比特接近 0（若明显大于 0，说明实现有问题或存在信息泄漏，需排查）。

---

## 5. 阶段 2：基线方法

所有基线都在 LIBERO-Goal 上训练（除非注明），并用阶段 1 的评测打分。训练统一通过 `Pion/VLA/VLAAdapter/vla-scripts/finetune_pion.py`，训练组定义写在 `configs/arms.yaml`。

### 5.1 训练组总表

| arm | V / L / A 优化器 | 额外设置 | 来源 |
|---|---|---|---|
| `adamw` | AdamW / AdamW / AdamW | 学习率扫 {5e-5, 1e-4, 2e-4} | 基线 |
| `muon` | Muon / Muon / Muon | | Pion 论文 |
| `pion` | Muon / Muon / DefaultPion | k_p 扫 {0, 1, 2} | Pion 论文推荐配方 |
| `pion_L` | Muon / DefaultPion / DefaultPion | | 补丁 |
| `lg_muon` | Muon / LangGuardMuon / DefaultPion | γ 扫 {0.25, 0.5, 1}，mode ∈ {ns, linear} | 补丁 |
| `lg_pion` | Muon / LangGuardPion / DefaultPion | 同上 | 补丁 |
| `lg_off` | Muon / LangGuardMuon(off) / DefaultPion | 必须和 `pion` 结果一致 | 正确性检查 |
| `sam_adamw` | AdamW 全部 + SAM | ρ 扫 {0.01, 0.05, 0.1} | 平坦性论文 |
| `film` | 与 `pion` 相同 | `--use_film True` | VLA-Adapter 自带 |
| `film_adamw` | 与 `adamw` 相同 | `--use_film True` | 同上 |
| `langforce` | 与 `adamw` 相同 | 见 §5.5 | LangForce |

推理时的语言引导（§5.2）**不是一个训练组**，而是对上面每个 checkpoint 都额外评测的一个维度。

默认训练参数：15000 步，`--use_fz True`，`--use_pro_version True`。DeltaAI 每节点 4 卡，用 **4 卡 × batch 16**（全局 64，与上游 8 × 8 相同；0.5B 模型在 96 GB 卡上预计放得下，以冒烟作业的显存峰值为准，OOM 就设 `BATCH_SIZE=8 GRAD_ACCUM=2`，但 SAM 组不能用梯度累积），只跑单节点，不做多节点。入口是 `deltaai/train.sbatch`（`ARM=adamw|muon|pion` 内置，其余组用 `OPTS='V L A O'` + `EXTRA_ARGS`）；`configs/arms.yaml` 只是把这些环境变量列成表。`SAVE_FREQ`：`adamw`/`muon`/`pion` 默认 1500（各先跑一个 seed，用于随训练步的曲线），其余组默认 5000（上游每 500 步存一个完整快照，15000 步会存 30 个）。

**seed**：已给 `finetune_pion.py` 加 `--seed`（默认 7）；torch/numpy/random 用 `seed`，TF 用 `seed + rank`（RLDS 管线不按 rank 分片，同种子会让所有卡拿到相同 batch），见 `docs/DECISIONS.md`。待验证：同 seed 两次跑，前 10 步 loss 是否逐位相同。

**算力预算（先算再申请）**：上表展开后约 27 个配置，×2–3 seed 就是 60–80 次 15000 步训练，再加每个 checkpoint 的评测。做法：
1. 冒烟作业实测 `秒/步`（4 卡），据此算每次训练的 GPU 小时，写进 `STATUS.md`，再决定 ACCESS 申请额度；
2. 两阶段：所有超参扫描先只跑 1 个 seed；每个方法只把最好的一个超参补到 3 个 seed；
3. `sam_*` 每步两次前向反向，预算 ×2。

### 5.2 推理时语言引导（`langeval/guidance.py`）

CAG（LIBERO-CF 论文的免训练版本）和 RAS 在本项目的 L1 回归动作头上是**同一个公式**：

```
a = a_null + ω · (a_cond − a_null)
a_cond = policy(obs, instruction)
a_null = policy(obs, null_instruction)     # null_instruction 默认为空任务描述
```

ω = 1 等价于原模型；ω > 1 放大语言带来的差异。实现要求：
- 在 `PolicyAdapter.act_with_guidance` 里做两次前向（可以把 cond / null 拼成 batch=2 一次完成），在**反归一化之前或之后均可**（归一化是仿射变换，结果相同），但必须在 LIBERO 的夹爪后处理**之前**组合；
- 组合后把每一维裁剪到训练数据统计给出的动作范围内，并在结果中记录被裁剪的比例；
- 评测扫 ω ∈ {1, 1.5, 2, 3}，**只在 `orig` 模式下扫**；`empty/other/shuffle` 只在 ω = 1 和最佳 ω 下跑。扫描阶段用 `--n_init 20`，最终数字用 50。

在 `docs/DECISIONS.md` 里说明：由于动作头是确定性回归，CAG-TF 与 RAS 在这里不可区分，统一称为 `lang_cfg`。CAG 论文里的"训练一个独立视觉—动作模型"版本放到阶段 5（可选）。

### 5.3 SAM（`Pion/VLA/VLAAdapter/lang_spectral/sam.py`）

新增配置 `sam_rho: float = 0.0`（0 表示关闭）和 `sam_adaptive: bool = False`。

每个优化步：
1. 正常前向与反向，得到梯度 g；
2. 计算扰动 e = ρ · g / ‖g‖（adaptive 版本对每个参数乘以 |w|），**对所有可训练参数**（包括 action head 和 proprio projector）原地加上 e；
3. 梯度清零，在扰动后的参数上再做一次前向与反向（DDP 会正常同步）；
4. 还原参数（减去 e），然后对所有优化器调用 `step()`。

要求：
- `grad_accumulation_steps` 必须为 1，否则报错；
- 梯度探针（`lang_probe`）只记录第二次反向的梯度，并在日志里注明；
- 单元测试：在一个两层小 MLP 上验证"扰动—还原"后参数逐位不变，以及 ρ = 0 时与不开 SAM 完全一致。

### 5.4 FiLM

VLA-Adapter 自带 `--use_film`：用平均语言嵌入对 ViT 各 block 做 FiLM 调制（`prismatic/models/film_vit_wrapper.py`）。训练和评测都要传 `--use_film True`。

- 先用 30 步冒烟测试确认 FiLM 与 `use_minivlm=True`、`use_pro_version=True` 组合能跑通、能存取 checkpoint；不能跑通就记录在 `STATUS.md`，并跳过这两个组。
- 在可解释性分析中，FiLM 是关键对照：它让视觉 token 显式依赖语言（见 §6.2 的预测）。

### 5.5 LangForce（移植，成本最高，放在本阶段最后）

LangForce 的官方实现基于 starVLA（Qwen3-VL），仓库 https://github.com/ZGC-EmbodyAI/LangForce 。**实现前必须先读它的源码**（`LangForce.py` 以及 starVLA 中被调用的部分），确认损失函数的准确形式，然后再移植。不要根据本文件的描述自行发明损失。

已知结构（来自论文）：
- 后验分支：序列 `[v, ℓ, Q]`，Q 是 latent action query，能看到视觉和语言 → π(a | v, ℓ)。VLA-Adapter 现有序列就是这种排列，`action_queries` 就是 Q。
- 先验分支：序列 `[v, Q, ℓ]`。在因果注意力下，Q 只能看到视觉 → p(a | v)；同时 ℓ 位于 Q 之后，可以在 ℓ 上计算语言模型对数似然 log p(ℓ | v, Q)。
- 目标：最大化 log p(ℓ | a, v) − log p(ℓ | v)（对数似然比 / 条件 PMI）。README 提醒：如果在指令外面包了额外模板，`kl_loss` 可能静默变为 0，移植时要专门检查这一点。

移植要求：
- 新文件 `Pion/VLA/VLAAdapter/lang_spectral/langforce.py`，在 `finetune_pion.py` 里用 `--objective langforce` 开启；
- 先验分支和后验分支共享权重，额外的计算开销要在日志里报告；
- 推理时只用后验分支（与原论文一致），因此评测接口不变；
- 单元测试：用一个极小的因果 Transformer 验证先验分支中 Q 的输出对 ℓ 的扰动不敏感，后验分支中则敏感。

如果移植中发现 VLA-Adapter 的结构与 LangForce 的假设不兼容（例如注意力其实是双向的，见 §6.1），停下来并在 `STATUS.md` 里说明。

### 5.6 验收

- 每个训练组都有 checkpoint、训练配置和 `lang_probe.jsonl`；
- `lg_off` 与 `pion` 在 `env_success` 和行为比特上一致（CI 重叠），否则 LangGuard 的实现有问题；
- `aggregate.py` 输出主表：每行一个 (arm, ω)，列为 `env_success`、`follow_rate`、`I_behav`（含 CI），以及 Spatial 对照上的 `env_success`。

---

## 6. 阶段 3：可解释性测量

阶段 3 的所有测量都对阶段 2 的每个 checkpoint 运行，且只需要前向计算（除了 §6.4 的在线替换）。

### 6.1 前提检查：视觉 token 能否"看到"语言（`langeval/attention_check.py`）

VLA-Adapter 的 LLM 序列是 `[BOS][视觉 patch][文本][动作 query]`。依赖里的 transformers fork 是为 OpenVLA-OFT 的并行解码改成双向注意力的，但它是否对 Qwen2 生效，需要实际验证。

**测试方法**（不依赖 attention 权重输出，flash-attn 下也能用）：同一张观测，分别用两条不同的指令做前向，比较每一层视觉 token 位置的隐藏状态。
- 如果所有层的差都在数值误差以内 → 视觉 token 看不到语言（因果注意力）；
- 如果有差异 → 存在从文本到视觉的信息流，记录从哪一层开始出现。

把结论写进 `docs/findings.md`。**§6.2 的预测取决于这个结论。** FiLM 组单独再测一次（预期一定有差异，因为 FiLM 在 ViT 内部注入了语言）。

### 6.2 表征比特（`langeval/probes.py`、`scripts/collect_hidden_states.py`、`scripts/run_probes.py`）

**采集**：
- 观测来自 LIBERO 演示数据（RLDS），按 episode 记录。
- **只取每条演示的前 `T0` 帧（默认 5 帧，机器人还没开始明显运动）作为主分析**。之后的帧里，机器人的运动方向本身就会泄漏任务身份，必须单独报告"随时间 t 的曲线"，不能混在一起。
- 对每帧保存选定层（默认每 2 层一个）所有位置的隐藏状态和 token 类型标签。**不要做任何跨 token 的平均池化。**
- 存 fp16 到 `$STORE`（不是 `$HOME`）。粗估：约 600 个位置 × 896 维 × 12 层 × 2 字节 ≈ 13 MB/帧，Goal 500 条演示 × 5 帧 ≈ 30 GB/checkpoint；实际大小以第一次采集为准写进 `STATUS.md`，随时间 t 的曲线另外按需采样。

**探针**：目标是 10 类任务身份。
- 每个 (层, 位置类别) 训练一个 L2 正则的多类逻辑回归。位置类别包括：每个动作 query 位置、视觉 token（逐 patch 训练，再把各 patch 的对数概率求和作为汇总预测，同时报告单个 patch 的最好结果）、语言 token。
- **按 episode 分组做交叉验证**（GroupKFold），同一条演示的帧不能同时出现在训练集和测试集。
- 表征比特 = log2(10) − 测试集交叉熵（单位 bit），这是互信息的下界；同时报告准确率。用温度缩放在验证集上校准概率。
- 对照：把标签随机打乱后重复，作为偏差基线。

**2×2 设计**（核心实验）：

| | 推理时给正确指令 | 推理时给空指令 |
|---|---|---|
| **LIBERO-Spatial**（布局不同） | 参照 | 视觉 token 能读出任务 → "布局记忆"（机制 a） |
| **LIBERO-Goal**（布局相同） | 视觉 token 能读出任务 → "语言被复制进视觉 token"（机制 b） | 应接近 0（否则有泄漏） |

Spatial 部分需要 Spatial 上训练的 checkpoint：用官方 Spatial checkpoint，再加一个用 `pion` 配方在 Spatial 上训练的 checkpoint。

预测：
- 若 §6.1 判定为因果注意力，Goal + 正确指令下视觉 token 位置的比特应接近 0，语言信息只能经由动作 query 进入；
- FiLM 组在 Goal + 正确指令下，视觉 token 位置应有明显的比特。

### 6.3 行为比特与表征比特的对照（`scripts/aggregate.py`）

对每个 checkpoint 画一张图：横轴是层，纵轴是动作 query 位置的表征比特；用一条水平线标出阶段 1 测得的行为比特。按方法类别着色（推理引导 / 训练目标 / 优化器 / 架构）。

需要输出的结论表：
- `I_repr_max`（各层动作 query 位置的最大表征比特）、`I_behav`、两者之差；
- 对每个 `lang_cfg` 的 ω：`I_behav` 是否上升、`I_repr` 不变（预期如此，因为推理引导不改变表征）。

### 6.4 因果替换（`langeval/patching.py`、`scripts/run_patching.py`）

在第 l 层，把语言 token 位置的隐藏状态替换成"同一观测、指令 j"时的隐藏状态，其他位置不变。

- **离线版**：在采集的观测上，测动作块向"指令 j 下的动作"移动的比例：`shift = 1 − ‖a_patched − a_j‖ / ‖a_i − a_j‖`，按层画曲线；
- **在线版**（只对 3 个代表层运行，成本较高）：在 rollout 中每次推理都施加替换，用阶段 1 的混淆矩阵测"执行了指令 j 对应任务"的比例。

实现方式：在 Qwen decoder layer 上注册 forward hook，只修改语言位置；用一个正向对照（替换全部位置）验证 hook 生效。

### 6.5 逐时间步语言敏感度（`langeval/sensitivity.py`、`scripts/run_sensitivity.py`）

在正确指令的 rollout 上，对每个推理时刻 t，固定观测，分别输入 10 条指令，计算 `S(t)` = 动作块之间的平均成对距离（除以同一时刻动作的典型幅度做归一化）。

- 输出 S(t) 曲线和"承诺时刻" t*：S(t) 持续低于阈值（默认取 t=0 时的 20%）的起点；
- 附加实验（可选，在线）：在时刻 τ 把指令从 i 换成 j，用混淆矩阵测最终执行的是 i 还是 j，画"τ — 跟随新指令比例"曲线。

### 6.6 梯度预算（已由补丁实现）

训练时的 `lang_probe.jsonl` 已经记录每层语言 token 的 `credit / budget / amplification`。`aggregate.py` 需要把它们按训练步画出来，并与最终的行为比特一起列表：训练前 20% 步的平均 `lang/budget` 与最终 `I_behav` 的秩相关。

### 6.7 验收

- §6.1 有明确结论；
- 每个 checkpoint 都有表征比特曲线（含打乱标签基线）和 2×2 表；
- 行为比特 vs 表征比特对照图和结论表生成；
- 所有结论写入 `docs/findings.md`，每条都链接到对应的结果文件。

---

## 7. 阶段 4：汇总

`scripts/aggregate.py` 读取 `results/` 下所有结果，输出到 `results/summary/`：
1. 主表（§5.6）；
2. 每个方法的混淆矩阵热力图；
3. 行为比特 vs 表征比特对照图（§6.3）；
4. 2×2 视觉 token 探针表（§6.2）；
5. 梯度预算随训练步的曲线（§6.6）；
6. 一份自动生成的 `summary.md`，只陈述数字，不写解释。

图用 matplotlib，颜色和标注要在黑白打印下也能区分。

---

## 8. 阶段 5（后续，可选，本轮不实现）

记录在这里，接口在前面阶段预留：
- **语言是钥匙还是程序**：`cipher`（每个单词固定替换成随机罕见 token 的双射）和 `key`（每个任务一个互不共享片段的随机串）两种训练指令；配合改写指令测试和只用于评测的组合新目标 bddl。
- **CAG 的独立视觉—动作模型版本**：训练一个空指令的同架构模型作为 `a_null`。
- **SmolVLA 后端**：`langeval/adapters/smolvla.py`，用于按 flow 步替换语言 KV；hook 写法参考 gripper-mujoco 仓库的 `interaction_vla/representation_study/libero/flow_trace.py`。

---

## 9. 允许修改的文件

`Pion/` 内（仅限 `lang-spectral` 分支）：
- `VLA/VLAAdapter/lang_spectral/**`（新增 `sam.py`、`langforce.py`）
- `VLA/VLAAdapter/vla-scripts/finetune_pion.py`
- `VLA/VLAAdapter/experiments/robot/libero/run_libero_eval.py`
- `VLA/VLAAdapter/experiments/robot/openvla_utils.py`（只在需要暴露隐藏状态或支持引导时，且保持原有函数签名的默认行为不变）
- `VLA/VLAAdapter/scripts/*.sh`（新增）

`Pion/` 外：本文件 §2 列出的所有路径。

---

## 10. 实现顺序与检查点

| 顺序 | 内容 | 检查点 |
|---|---|---|
| 0 | DeltaAI：`setup_env.sh`、`download_assets.sh`、`smoke.sbatch`；测训练 `秒/步` 和每卡最佳 rollout 进程数 | 冒烟作业通过，数字写进 `STATUS.md` |
| 1 | 阶段 0：Pion 建 git 基线、补丁（缺失则先补）、冒烟测试 | §3.4 验收 |
| 2 | `langeval` 骨架、`metrics.py`、`instructions.py` 及其单元测试 | `pytest -q` 全绿 |
| 3 | `goal_predicates.py`、`goal_matrix.py`、`vla_adapter.py` | 官方 Goal checkpoint 的 `env_success` 与官方数字一致 |
| 4 | 四种指令模式的混淆矩阵 | §4.6 验收 |
| 5 | `attention_check.py` | §6.1 结论写入 findings |
| 6 | `guidance.py`，在官方 checkpoint 上扫 ω | 主表第一版 |
| 7 | `collect_hidden_states.py`、`probes.py`，在官方 checkpoint 上跑 2×2 | 第一版表征比特图 |
| 8 | 给 `finetune_pion.py` 加 `--seed`；训练组：adamw、muon、pion、lg_*、sam、film（先 1 seed 扫超参） | §5.6 验收 |
| 9 | 对所有 checkpoint 跑阶段 1 与阶段 3 | §6.7 验收 |
| 10 | LangForce 移植 | 单元测试 + 训练跑通 |
| 11 | `aggregate.py` | §7 全部输出 |

步骤 3–7 只需要公开 checkpoint，不需要训练，应当先完成并汇报，再开始步骤 8 的训练。
