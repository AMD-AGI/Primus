# Primus Pilot — 补充章节（v2 supplements）

**Status**: Draft，待并入 `README.cn.md`

> 本文件起草 4 个 v2 主文档里点到但未展开的关键章节。命名 §S1–§S4 表示 supplement，定稿后建议位置见每节开头的 "Insert at" 提示。

| § | 主题 | 优先级 | 拟插入位置 |
|---|------|--------|-----------|
| S1 | Execution Model Calibration | P0 | §6 之后，作 §6.x 子节 / 或独立 §6A |
| S2 | Correctness Reference & Scale-aware Tolerance | P0 | §3.2 CORRECTNESS 的展开，作 §3.4 |
| S3 | Parallel Execution & Resource Protocol | P1 | §11 与 §12 之间，作新 §11A |
| S4 | LEARN Governance & Knowledge Lifecycle | P1 | §3.2 LEARN 的展开，作 §11B（与现有系统集成相邻） |

---

## §S1. Execution Model Calibration（解决 `expected_gain` / `confidence` 的来源）

> **问题**：§7.4 priority 公式依赖 `expected_gain(c)` 与 `confidence(c)`，§6 的 `T_step = T_comp + T_comm + T_bubble - T_overlap` 是结构性公式，包含若干自由参数（`η_comp`、`BW_eff`、`α_overlap`、`β_bubble`），单凭 ClusterProfile 的硬件 peak 无法填实。如果这些参数靠拍脑袋，§7 的搜索结构就退化为"带元数据的随机游走"。
>
> **本节做什么**：定义 calibration 参数集合、在线更新协议、置信度计算、以及失效检测。

### S1.1 参数化的 Execution Model

把 §6 的公式补上 calibration 参数（**bold** 为待学习量）：

```
T_comp_pred(plan)    = (model_flops / num_gpus) / (peak_tflops × η_comp)
T_comm_pred(plan)    = Σ_i  msg_size_i / (BW_peak_i × η_comm_i)
T_bubble_pred(plan)  = (pp - 1) / (pp - 1 + M) × T_comp_pred × β_bubble
T_overlap_pred(plan) = α_overlap × min(T_comm_overlappable, T_comp_spare)
T_step_pred          = T_comp_pred + T_comm_pred + T_bubble_pred - T_overlap_pred

Mem_pred(plan)       = M_param + M_grad + M_optim + γ_act × M_act_theory + δ_buffer

calibration_params = {
    η_comp,                       # 实际计算效率（含 kernel / mem 限制）
    η_comm[allreduce|alltoall|allgather|reduce_scatter],
    α_overlap,                    # overlap 的实际兑现率（理论上限的多少）
    β_bubble,                     # bubble 的修正项（warmup / 不均衡 stage 等）
    γ_act,                        # 实际 activation 显存 / 公式估算
    δ_buffer,                     # workspace / fragmentation 常数项
}
```

**为什么这么拆**：每个参数对应一类已知的"理论 vs 实际"差距，且彼此正交（基本上）——更新时不会互相污染。

**作用域**：calibration_params 绑定到 `(cluster_id, model_family, framework_version)` 三元组。换集群、换模型族、换 backend 都重新 calibrate；同集群跨 model_size 共享。

### S1.2 Calibration State（State Layer 持久化）

在 §8 schema 集合里新增（建议放在 ClusterProfile 旁边作独立文件，避免污染）：

```yaml
# state/calibration_state.yaml
calibration_id: mi300x-16node__llama__megatron-0.7
binding:
  cluster_id: mi300x-16node
  cluster_profile_version: mi300x-16node-v3
  model_family: llama_dense        # 同 family 共享
  framework: megatron
  framework_version: "0.7.x"

params:                            # 后验均值
  eta_comp: 0.71
  eta_comm:
    allreduce: 0.62
    alltoall:  0.48
    allgather: 0.55
    reduce_scatter: 0.58
  alpha_overlap: 0.73
  beta_bubble: 1.08
  gamma_act: 1.12
  delta_buffer_gb: 4.5

posterior:                         # 用于 confidence 计算
  kind: gaussian                   # gaussian / bootstrap
  cov_diag:                        # 各参数后验方差
    eta_comp: 0.0009
    alpha_overlap: 0.0036
    ...
  effective_n: 47                  # 有效观测数（带衰减）

residuals:                         # 最近 K=20 次预测误差，用于失效检测
  - {plan_id: r2_p4, predicted_tps: 17600, observed_tps: 17820, rel_err: +0.012}
  - {plan_id: r3_p1, predicted_tps: 18800, observed_tps: 17400, rel_err: -0.074}
  ...

freshness:
  last_updated: 2026-04-21T15:32:00Z
  observations_since_reset: 47
  drift_alarm: false               # 见 S1.5
```

### S1.3 在线更新协议（每次 Execute 完调用）

**触发点**：每个 plan Execute 完成、Snapshot 写入后（§3.3 内层泳道的 [Observe] 之后）。**不**在 EnvSweep 短跑里更新（30-50 step 信噪比太低）。

```python
# tools/calibrate.py（新增）
def calibrate.update(snapshot: Snapshot, plan: Plan,
                     state_path: str = 'state/calibration_state.yaml') -> CalibrationState:
    """
    用一次 (predicted, observed) 对更新 calibration_params。
    采用对数空间的加权递推最小二乘（WRLS）：
      - 对数空间：保证 η/α/β 正、可乘
      - 加权：近期观测权重高（指数衰减 λ=0.9）
      - 递推：O(1) 更新，不存全部历史
    """

    # 1) 残差分解：把总 step_time 误差归因到各参数
    components = decompose_step_time(snapshot, plan)
    # components = {
    #     T_comp_obs, T_comm_obs[type], T_bubble_obs, T_overlap_obs
    # }
    # 来源：profiler trace（trace.md 规定的事件分类）；
    # 若无 profile（短跑），降级为整体 T_step 残差，只更新 eta_comp + delta_buffer。

    # 2) 逐参数 WRLS 更新（带衰减）
    for param, observed_ratio in residuals_per_param(components, plan):
        state.params[param] = wrls_update(
            prior_mean = state.params[param],
            prior_var  = state.posterior.cov_diag[param],
            observation = observed_ratio,
            obs_noise  = SENSOR_NOISE[param],   # 由 profiling/trace.md 给出
            decay      = 0.9,
        )

    # 3) Append 到 residuals（环形 buffer，cap=20）
    state.residuals.append({...}); state.residuals = state.residuals[-20:]

    # 4) 漂移检测（见 S1.5）
    state.freshness.drift_alarm = detect_drift(state.residuals)

    state.checkpoint()
    return state
```

**降级路径**（无 trace 数据的情况）：

| 数据可得性 | 能更新的参数 | 不能更新的 |
|-----------|------------|-----------|
| 完整 profile trace | 全部 | — |
| 仅 step time + mem peak | `η_comp`（与 `η_comm` 混淆，归到 effective comp） + `γ_act` + `δ_buffer` | `α_overlap`、`β_bubble` 单独不可分 |
| 仅 step time | `η_comp_aggregate`（不再按通信类型拆） | 其余冻结 |
| OOM / failed | 不更新参数；`γ_act` 标记为 underestimated，下次保守 |

### S1.4 `expected_gain` / `confidence` 的具体计算

```python
# tools/predict.py（新增）
def predict.gain(parent_plan, candidate_plan, calibration_state):
    """
    返回：(predicted_tps, expected_gain_pct, confidence)
    """
    mu_step_parent = T_step_pred(parent_plan,    calibration_state.params)
    mu_step_cand   = T_step_pred(candidate_plan, calibration_state.params)

    predicted_tps = global_batch_tokens / mu_step_cand
    expected_gain = (mu_step_parent - mu_step_cand) / mu_step_parent

    # confidence: 通过参数后验方差 + delta-method 传播到 T_step
    var_step_cand = sum(
        (∂T_step / ∂param)**2 × posterior.cov_diag[param]
        for param in calibration_params
    )
    sigma_step = sqrt(var_step_cand)
    cv_step    = sigma_step / mu_step_cand     # coefficient of variation

    # confidence ∈ [0,1]：低 CV 高置信
    confidence = 1.0 / (1.0 + 5.0 × cv_step)

    # 离最近一次观测越远的"轴空间"，confidence 进一步打折
    novelty_penalty = axis_distance(candidate_plan, recent_observations)
    confidence *= max(0.3, 1 - 0.4 × novelty_penalty)

    return predicted_tps, expected_gain, confidence
```

**`confidence` 的语义**：用作 §7.4 priority 的乘子时，落在三档：

| confidence 区间 | 语义 | Re-Plan 行为 |
|-----------------|------|-------------|
| > 0.7 | 模型可信 | 正常进 priority |
| 0.4 – 0.7 | 模型借鉴 | 走 Successive Halving，多候选并跑 |
| < 0.4 | 模型不可信 | **退回 Champion-Challenger 物理实验**，预测仅作 OOM 安全网 |

`execution_strategy.md` 的策略选择规则按这个区间表落实。

### S1.5 漂移检测与失效

Calibration 不是一次到位、永久使用——集群升级、驱动变化、模型族切换都会让旧参数失效。检测协议：

| 信号 | 阈值 | 动作 |
|------|------|------|
| 最近 5 次 `rel_err` 同号 | mean(|rel_err|) > 10% | `drift_alarm = true`，本次 Re-Plan **不再消费 confidence**，相当于退化到 strategy=Champion-Challenger |
| `cluster_profile.version` 变更 | 任何变更 | 整份 calibration_state 标记 stale，`effective_n` 衰减到 5 重新 warmup |
| `framework_version` major bump | major 段变 | 同上 |
| 新模型族 | binding 不命中 | 起新文件；首次 Re-Plan 时 confidence 强制设为 0.3（保守 baseline） |
| `observations_since_reset > 200` 且最大子节点子树 dead_rate > 50% | 复合条件 | 触发 Re-Calibrate 子流程：用最近 BASELINE 跑一次 micro-bench refresh |

漂移触发后的 fallback：**`expected_gain` 仍按公式给，但 `confidence < 0.4`**——让 §7.4 的 priority 自然降权，搜索退化为更保守的物理实验为主。**不要直接关掉 Execution Model**——它仍在做 OOM 预估这件不可替代的事。

### S1.6 Cold-start：第一次怎么办

新 (cluster, model_family, framework) 三元组没有任何观测时：

1. **PROJECTION 阶段执行 single-node profiling**（已在 §3.2），把 single-node 实测的 `T_comp` / `T_comm`（若 ≥ 2 GPU）回填，得到 `η_comp` / `η_comm[*]` 的 zero-shot 估计
2. **其余参数**用 cluster 类型默认值（`alpha_overlap=0.7`，`beta_bubble=1.05`，`gamma_act=1.10`，`delta_buffer_gb=4.0`，由 `skills/execution-model/cluster_priors.md` 给出，**新增 Skill**）
3. **首次 BASELINE 跑完后立即 calibrate.update()**，把 `effective_n` 从 0 拉到 1
4. **前 3 轮**：所有候选 confidence cap 在 0.6（避免 cold-start 误信任）

### S1.7 评估与回归

`§10 评估指标`新增三行：

| 维度 | 指标 | 目标 |
|------|------|------|
| 模型质量 | `T_step_pred` vs `T_step_obs` 中位相对误差 | ≤ 15% |
| 模型质量 | `Mem_pred` vs `Mem_obs` 上分位（p90）相对误差 | ≤ 20% |
| 模型质量 | calibration 漂移误报率（drift_alarm 误触发 / 全部） | ≤ 10% |

回归方法：用历史 session 的 (predicted, observed) pair 当 hold-out test。

### S1.8 与现有 §X 的交叉引用

- `predict.gain()` 替代 §7.4 priority 公式里"假设给定"的 `expected_gain` / `confidence`
- `calibration_state.yaml` 进 §4.1 目录树（与 `tuning_state.yaml` 同级）
- §5 Tool 接口表新增：`calibrate.update`, `predict.gain`, `predict.mem`
- §8 schema 集合新增 `calibration_state.schema.json`
- §13.2 subagent 边界表：**Calibrate 不必做 subagent**（O(1) 数学操作，无大段 Skill 阅读）

---

## §S2. Correctness Reference & Scale-aware Tolerance（解决"reference 从哪来"）

> **问题**：§3 / §12.1 把"对齐 reference loss curve"放在 BASELINE 后，但 (1) 第一次 bring-up 时 reference 不存在；(2) loss curve 本来就随 mbs / gas / scale 变形——直接比对会把"正常的统计噪声"误判成"数值 bug"。
>
> **本节做什么**：定义 reference 的来源分层、scale-aware tolerance 公式、以及不同 round 应触发哪一档 gate。

### S2.1 三层 reference

| Tier | 来源 | 信任度 | 何时建立 | 何时使用 |
|------|------|--------|----------|---------|
| **T0 Anchor** | 单节点 / 双节点 deterministic FP32 / BF16 跑 200 step，固定 seed，固定数据 | 最高（视为 ground truth） | bring-up 第一次进 PROJECTION 之前 | 用于校验 Tier-1 |
| **T1 Reference** | BASELINE 跑过 CORRECTNESS gate 的曲线 | 高 | BASELINE 阶段，通过 T0 校验后晋升 | OPTIMIZE_LOOP 内 CORRECTNESS-LITE 抽查 |
| **T2 Local** | 当前 round champion 的最近 100 step 平滑曲线 | 中 | OPTIMIZE_LOOP 滚动维护 | 检测 dramatic regression（强 gate 之外的弱信号） |

**晋升规则**：T0 → T1 通过 §S2.3 的"等价归一化"测试；T1 不能自我晋升（防止漂移累积）。

**Bring-up 第一次的协议**：

```
PROJECTION
  └─ 启动前必须已存在 T0 Anchor（通过 cli flag 指向 / 自动小规模跑）
       │
       ▼
  single-node profiling
       │
       ▼
SMOKE (tiny scale × 100 step)
       │
       ├── correctness_lite_gate(against=T0)   ← 第一次数值闸门
       ▼
BASELINE
       │
       ▼
CORRECTNESS (full scale，对齐 T0 → 晋升到 T1)
       │
       ▼
OPTIMIZE_LOOP（之后所有 round 用 T1）
```

T0 跑一次的成本是固定开销（典型 < 1 GPU·h），跨 session 可复用，按 `(model_family, dataset_id, seed)` 索引。

### S2.2 Scale-aware Tolerance：噪声模型

**核心观察**：不同并行配置下 loss 不同，主要来源是**有效 batch size 变化导致的 grad noise scale**变化，而**不是**数值 bug。

```
σ_loss(scale) ≈ σ_loss(reference) × sqrt(EBS_ref / EBS(scale))

其中 EBS = global_batch_tokens × num_dp_replicas (effective batch size)
```

step-wise loss 的容差按 σ 的倍数给：

```
tolerance_step(s) = k_step × σ_loss(scale at step s) + ε_systematic
window_tolerance(s, w) = k_window × σ_loss(scale) / sqrt(w) + ε_systematic
                         （w-step rolling mean，标准误按 1/√w 缩）
```

| 参数 | 默认值 | 来源 |
|------|--------|------|
| `k_step` | 4.0（4σ） | step-wise 严格 gate |
| `k_window` | 3.0（3σ） | window-mean，更紧 |
| `w` | 50（50-step rolling） | 同 §3.1 SMOKE 的 step 量级 |
| `ε_systematic` | 0.005 | 浮点精度差异、kernel 数值路径差异等系统性偏差容忍 |

`σ_loss(reference)` 由 T0 Anchor 在建立时**实测得到**（同一 seed 跑 3 次，取 cross-run std），写入 reference 文件。

### S2.3 等价归一化（让不同 scale 可比）

直接比 raw loss 不对（不同 EBS 下绝对值就不同）。归一化到"per-token NLL"+ "EBS 调整后的预期均值"：

```
loss_per_token = loss / log(vocab_size)             # 归一到 [0,1] 区间附近
expected_mean(s | scale)  = T1_mean(s) + Δ_EBS(EBS_ref → EBS_scale)

其中 Δ_EBS 来自 grad noise scale 论文（Smith et al. / OpenAI scaling laws）的
经验关系：相同 token 数下，EBS 翻倍 → 等效 step 数 × ~1/√2 的位置上 loss 一致

实现：以 "tokens consumed" 为 x 轴对齐，而不是以 "step idx" 对齐
```

这一条实操上的影响：**比对横坐标用 tokens consumed，不用 step idx**。这把 mbs/gas 的影响自动对齐了。

### S2.4 Gate 的层次（每种触发哪个 tier）

| Gate | 在哪触发 | 用哪个 Tier | tolerance | 通过失败时 |
|------|---------|------------|-----------|-----------|
| **smoke_correctness** | SMOKE 后 | T0 | step-wise k=5σ（最松，因为 100 step 噪声大） | 回 PROJECTION 重建初始 plan |
| **baseline_correctness** | BASELINE 后 | T0 | tokens-aligned window-mean k=3σ + grad_norm 范围 | ABORT + escalate（数值正确性破坏） |
| **lite_correctness** | OPTIMIZE_LOOP 每 N 轮 / dramatic 变化时 | T1 | window-mean k=3σ | mark plan dead，回 RE_PLAN（不 abort 整个 session） |
| **regression_signal** | 每个 Snapshot | T2 | window-mean k=2σ（最敏感） | 仅 warn，不阻塞——作为 §S1 漂移检测的输入 |

**N 的取值**：默认每 3 round 跑一次 lite_correctness（与 §7.6 的 explore round 周期对齐，分摊成本）。dramatic 变化（mbs / pp / recompute 改变）时强制触发。

### S2.5 Schema 增量

**新增 ReferenceCurve schema**（§8.x）：

```yaml
# state/references/<model_family>__<dataset>__<seed>.yaml
reference_id: llama_dense_8b__c4_subset__seed42
tier: T0                           # T0 / T1
created_at: 2026-04-15T10:00:00Z
binding:
  model_family: llama_dense
  model_size_b: 8
  dataset_id: c4_subset_v3
  seed: 42
  precision: bf16

config:                            # 建立时的最小配置
  parallelism: {tp: 1, pp: 1, dp: 8, ep: 1}
  mbs: 4
  gbs: 256
  recompute: none

trajectory:                        # tokens-aligned (不是 step-aligned)
  - {tokens: 1.0e6, loss_per_token: 0.842, grad_norm: 0.51}
  - {tokens: 2.0e6, loss_per_token: 0.781, grad_norm: 0.48}
  ...

noise:                             # 同 seed 跑 3 次得到的 cross-run std
  loss_per_token_std_at_1e7_tokens: 0.0042
  grad_norm_std_at_1e7_tokens: 0.018

promoted_to_t1:                    # T0 -> T1 晋升记录
  - {at: 2026-04-15T11:30:00Z, by_session: pilot_run_20260415_a1, baseline_plan: r0_p0}
```

**FailureReport 扩展**（§8.8）：`failure_kind=NUMERICAL` 时追加 `gate_tier` 与 `tokens_at_failure`，让 escalate 时人能立刻定位到哪个比对失败。

### S2.6 边界情况

| 情况 | 处理 |
|------|------|
| 模型 / 数据 / seed 完全自定义，无 T0 可用 | Pilot 拒绝进入 OPTIMIZE_LOOP，要求用户先 commission 一份 T0（cli `pilot reference build`） |
| MoE 路由 stochastic（即使同 seed 也有差异） | T0 用 deterministic routing 跑一次；T1 用 stochastic routing 跑，noise 自然变大被 σ 吸收 |
| 用户故意改了 model arch（如 head_dim） | binding 不命中，强制重建 T0 |
| Loss spike 且明显非数值原因（如 LR warmup） | tolerance 公式不适用，靠 `skills/correctness/known_artifacts.md`（**新增 Skill**）显式跳过 first N tokens |

### S2.7 与现有章节的交叉引用

- §3.1 块状主图：CORRECTNESS 节点拆为 "BASELINE_CORRECTNESS (T0)" 与 "LITE_CORRECTNESS (T1)" 两个状态机节点
- §4.2 skills/ 新增 `correctness/` 子目录：`SKILL.md` / `tolerance.md` / `tier_promotion.md` / `known_artifacts.md`
- §5 Tool 接口新增：`reference.build`、`reference.compare`（替代当前 `observe.compare_loss` 的实现，签名升级）
- §10 评估指标新增："numerical regression 漏报率 ≤ 5%、误报率 ≤ 10%"
- §12.2 失败回路：`NUMERICAL` 行细化为 4 行（每个 gate 一行），不同 tier 失败走不同的 transition

---

## §S3. Parallel Execution & Resource Protocol（解决 §3.1 "3 个 plan 并行"的资源协议缺口）

> **问题**：§3.1 提到 Champion-Challenger / Successive Halving 时"K 个 plan 并行跑"，§9 例子里也假设 P1/P2/P3 并行 50 step。但：(1) 16 节点怎么切？(2) 一个 plan OOM 把节点搞崩，会不会污染其他 plan？(3) Slurm/k8s 上具体怎么分配？(4) 短跑得到的指标如何外推到 full scale？
>
> **本节做什么**：定义资源分片模型、隔离契约、调度协议、外推规则。

### S3.1 三种执行模式

```
        cluster_size = N nodes
        ┌──────────────────────────────────────────┐
        │ A) FullScale     ─ 1 plan × N nodes      │  BASELINE / final validation
        │ B) Sharded       ─ K plan × N/K nodes    │  Tuning Loop short runs
        │ C) TimeMux       ─ K plan 串行 × N nodes │  长 step / 强结构差异 plan
        └──────────────────────────────────────────┘
```

| 模式 | 适用 | 优势 | 代价 |
|------|------|------|------|
| **FullScale** | BASELINE、CORRECTNESS、final config 验证 | 数据真实、无外推 | 串行慢 |
| **Sharded** | Tuning Loop 内的 50 step short runs（多数情况） | 并行快 | 需要外推 + 形状约束（见 S3.4）|
| **TimeMux** | 候选间结构差异大（pp / world_size 改变）；或 候选数 < 3 不值得切片 | 简单、无外推 | 总 wall time 长 |

模式选择规则（落在 `skills/workflow/execute.md`）：

```python
def choose_execution_mode(candidates, cluster_size):
    if stage in ['BASELINE', 'CORRECTNESS', 'FINAL_VALIDATION']:
        return 'FullScale'

    # 候选间 world_size 不能整除 → 无法 sharded
    target_ws = candidates[0].world_size
    if not all(c.world_size == target_ws for c in candidates):
        return 'TimeMux'

    # 候选数太少：sharded 切片浪费节点
    if len(candidates) < 3:
        return 'TimeMux'

    # 单 shard 节点数 < 最小可信 shard
    shard_size = cluster_size // len(candidates)
    if shard_size < min_credible_shard(candidates[0]):  # 见 S3.4
        return 'TimeMux'

    return 'Sharded'
```

### S3.2 Sharded 模式的资源契约

把 cluster 切成 K 个**拓扑感知 shard**，每 shard 跑一个 candidate plan：

```
Shard 划分原则（按优先级）：
  1. IB 局部性：尽量同一 leaf switch 下的节点一组（避免跨 spine 通信影响）
  2. 同质性：同一 shard 内 GPU 型号、NIC 型号一致
  3. 大小相等：节点数差不超过 1
  4. 故障隔离：避开 §S3.5 黑名单节点
```

**`submit.run()` 升级签名**（§5 Tool 接口扩展）：

```python
submit.run(
    plans: List[Plan],
    mode: Literal['FullScale', 'Sharded', 'TimeMux'],
    shard_strategy: ShardStrategy = TopologyAware(),
    isolation: IsolationLevel = 'node',  # 'node' / 'rack' / 'cluster'
    timeout_per_plan_s: int = 1800,
    early_stop: EarlyStopPolicy = ...,
) -> List[RunResult]
```

**Slurm 实现**：

```bash
# Sharded：用 heterogeneous job step
sbatch \
  --job-name=pilot_r3 \
  --het-group=0 --nodes=4 --nodelist=node[01-04] : \
  --het-group=1 --nodes=4 --nodelist=node[05-08] : \
  --het-group=2 --nodes=4 --nodelist=node[09-12] : \
  --het-group=3 --nodes=4 --nodelist=node[13-16] \
  pilot_shard_runner.sh
```

**k8s 实现**：每 shard 一个 PodGroup（gang scheduling），不同 shard 通过 nodeAffinity 锚定。

### S3.3 故障隔离契约

| 失败类型 | 影响范围 | 其他 shard 行为 | 节点黑名单动作 |
|---------|---------|----------------|--------------|
| OOM | 仅自身 shard 进程 | 不受影响，继续 | 不拉黑 |
| HANG（NCCL timeout） | 仅自身 shard 通信组 | 不受影响 | 临时拉黑该 shard 节点 30min（保护下次） |
| 节点掉线 / GPU ECC | 该 shard 整体失败 | 不受影响 | 永久拉黑该节点（写 `state/blacklist.yaml`） |
| 跨 shard 干扰（fabric 抖动） | 多 shard 同时退化 | 检测到 ≥ 2 shard 同步退化 → ABORT 整个 batch | 触发 `PREFLIGHT` 复检 |

**检测跨 shard 干扰**：

```python
def cross_shard_anomaly(shard_results):
    # 至少 2 shard 的 tps 偏离自身预测 > 15%（同方向）
    deviations = [(r.tps - r.predicted_tps) / r.predicted_tps for r in shard_results]
    same_sign_outliers = sum(1 for d in deviations if d < -0.15)
    return same_sign_outliers >= 2
```

命中 → 写 `FailureReport(failure_kind=CLUSTER, root_cause=cross_shard_interference)`，进 §12.2 的 PREFLIGHT 回路；本批 round 不计配额。

### S3.4 Sharded 的外推问题（最容易踩坑）

**核心矛盾**：sharded 跑得到的是"N/K 节点上"的 tps，但 settle 决策需要的是"N 节点上"的 tps。两者不一定线性。

文档约定 **3 类外推规则**（落在 `skills/workflow/execute_extrapolation.md`，**新增 Skill**）：

| Plan 改变的轴 | 跨 scale 行为 | sharded → full 的外推可信度 | 建议 |
|--------------|-------------|---------------------------|------|
| 仅 env / mbs / recompute / bucket（**结构不变**） | 通信模式不变，scale 几乎线性 | 高（误差 < 5%） | sharded OK，直接外推 |
| TP / VPP / EP（**通信结构变**） | 跨节点流量随 world_size 变化 | 中（误差 5–15%） | sharded 跑筛选 + top-2 进 FullScale 复跑 |
| PP / DP（**通信域大小变**） | bubble 与 allreduce 都强依赖 world_size | 低（误差可超 20%） | 直接 TimeMux，不要 sharded |

**`min_credible_shard()`**（决定能不能 sharded）：

```python
def min_credible_shard(plan):
    # 至少要能保留原始 plan 的最小通信组
    return max(plan.tp * plan.pp,  # 一个完整 model parallel 组
               2)                   # 至少 2 节点保留 IB 通信
```

**预测一致性自检**：sharded 跑完后用 §S1 calibration 反算"如果换成 full scale 应该是多少 tps"，若与"该 plan 在 round 0 BASELINE 时的对应预测"差异 > 20%，则**强制 top-2 进 FullScale 复跑**——否则 settle 决策不可信。

### S3.5 节点健康与黑名单

```yaml
# state/blacklist.yaml
nodes:
  - id: node07
    reason: ECC_uncorrectable_at_2026-04-20
    severity: permanent             # permanent / temporary
    expires_at: null
  - id: node12
    reason: NCCL_hang_repeated_3x
    severity: temporary
    expires_at: 2026-04-21T16:00:00Z
```

**集成点**：`preflight.run()` 启动时读 blacklist，把对应节点排除出 ClusterProfile.nodes。`submit.run()` 的 shard 划分尊重 blacklist。

**自动入黑**条件（任一）：

- 单 session 内 NCCL hang ≥ 3 次（temporary，4h）
- ECC uncorrectable / GPU not present（permanent）
- 跨 session 累积失败率 > 30%（temporary，24h）

入黑名单是 §12.2 `CLUSTER` 失败的副作用，不消耗 round 配额。

### S3.6 TimeMux 模式的 checkpoint 协议

时分复用同一组节点跑 K 个 plan 时，要避免每个 plan 都重新 warmup（compile / shape 缓存丢失）。约定：

| 项 | 是否跨 plan 复用 | 备注 |
|---|----------------|-----|
| Python 进程 | **不复用**（保证隔离） | 每 plan 独立进程 |
| Megatron / TorchTitan compile cache | 同 backend / 同 model_arch 时复用 | 通过 `TORCH_INDUCTOR_CACHE_DIR` 共享 |
| Optimizer state | 不复用 | 每 plan 重新 init |
| Dataloader prefetch buffer | 不复用 | — |

实测 compile cache 复用能节省 30–60s/plan 的 warmup 时间，对 50 step short run 是可观比例。

### S3.7 评估指标增量

`§10` 新增：

| 维度 | 指标 | 目标 |
|------|------|------|
| 并行效率 | Sharded wall time / TimeMux wall time | ≤ 0.4（即 ≥ 2.5× 加速） |
| 隔离质量 | 跨 shard 干扰误判率（FullScale 复跑后翻案率） | ≤ 5% |
| 节点健康 | 黑名单 false-positive（误拉黑后通过 PREFLIGHT 复活） | ≤ 10% |

### S3.8 与现有章节的交叉引用

- §3.1 主图：`Execute` 节点内部分流到三种模式
- §5 Tool 接口：`submit.run()` 签名升级（mode / shard_strategy / isolation）
- §8 schema 新增：`shard_plan.schema.json`、`blacklist.schema.json`
- §12.1 Guardrails 新增：节点黑名单、跨 shard 干扰检测、shard 隔离契约
- §13.2 subagent 边界：Execute 仍**不做** subagent，但单个 shard 的 observe 可独立 spawn

---

## §S4. LEARN Governance & Knowledge Lifecycle（解决"LLM 自写自读"的 governance 缺口）

> **问题**：§3.1 / §4.2 把 LEARN 写成"best/失败 case 回写 `skills/knowledge/`"——这是 Skill ← State 的唯一反向流。如果让 LLM 自动写、下次自动读，无人审核，3 个月后 `knowledge/cases.md` 会塞满低质量观察、相互矛盾的 hint、过期的"经验"。
>
> **本节做什么**：把 LEARN 阶段拆成"草稿 → 审核 → 入库"三步流水线，定义草稿 schema、冲突检测、老化与撤回机制。

### S4.1 三阶段流水线

```
┌─────────────────┐    ┌──────────────────┐    ┌──────────────────┐
│  Stage A: Draft │ →  │  Stage B: Triage │ →  │  Stage C: Merge  │
│  (auto, by LLM) │    │  (human / curator)│    │  (git commit)    │
│                 │    │                  │    │                  │
│ state/knowledge_│    │ scripts/curate.py│    │ skills/knowledge/│
│   drafts/<sid>/ │    │ + manual review  │    │   *.md (versioned│
└─────────────────┘    └──────────────────┘    └──────────────────┘
                                ▲
                                │
                       ┌──────────────────┐
                       │  conflict / age  │
                       │  signals         │
                       └──────────────────┘
```

**关键约束**：`skills/knowledge/` 永远只通过 git PR 改动；运行时进程**没有写权限**——LLM 只能写 `state/knowledge_drafts/`。

### S4.2 Stage A：Draft（LLM 自动产出）

**触发**：每个 session 进入 LEARN stage 时，对以下情况各产出一个 draft：

| Draft 种类 | 触发条件 | 模板路径 |
|-----------|---------|---------|
| `final_best_case` | session 正常 STOP 且 `final_tps / baseline_tps ≥ 1.2` | templates/case.md |
| `failure_pattern` | dead 率 > 30% 的 axis 子树 | templates/anti_pattern.md |
| `env_recipe` | EnvSweep 找到 ≥ 5% 提升的 env diff | templates/env_recipe.md |
| `model_calibration_drift` | §S1 报告 drift_alarm 后人工确认是真漂移 | templates/calibration_note.md |

**Draft schema**（统一 envelope，§8 schema 新增）：

```yaml
# state/knowledge_drafts/<session_id>/<draft_id>.yaml
draft_id: drft_20260421_a3_b2
draft_kind: final_best_case        # / failure_pattern / env_recipe / calibration_note
created_at: 2026-04-21T18:00:00Z
created_by:
  session_id: pilot_run_20260418_a3
  agent: claude-3.7-sonnet
  pilot_version: 0.4.2

binding:                           # 关联 (集群, 模型族, framework)
  cluster_class: mi300x_8gpu       # cluster_class 比 cluster_id 更通用，便于跨集群迁移
  model_family: llama_dense
  model_size_range: [70, 140]      # billions
  framework: megatron
  framework_version_range: ["0.6", "0.8"]

claim:                             # 这条 draft 想沉淀的具体规律
  headline: "Llama 70B+ on MI300X-8GPU class, prefer pp=4 over pp=8 (bubble dominates after pp>4)"
  detail: |
    在 16 节点（128 GPU）规模下，pp=8 的 bubble_ratio 中位 0.21，
    pp=4 + vpp=2 的 bubble_ratio 0.09，net tps 高 14–18%。
  applicability:
    - "node_count ∈ [8, 32]"
    - "global_batch_tokens ≥ 4M"
  not_applicable_when:
    - "model_size_b > 200"          # 没有数据
    - "framework=torchtitan"        # 已知行为不同

evidence:                          # 必须有证据，不能凭 LLM 猜
  - kind: round_result
    ref: state/checkpoints/r2/plan_graph.yaml#nodes.r2_p4
    summary: "pp=4 vpp=2 配置 tps=17600 vs pp=8 配置 tps=15400"
    cost_gpu_h: 1.2
  - kind: cross_session
    ref: knowledge/cases.md#llama70b-mi300x-2026-04-15  # 历史 case 印证
    summary: "上次 session 同模型上同结论"

confidence:                        # LLM 自评（Stage B 会再判一次）
  self_assessed: 0.72
  signals:
    - "single session evidence"
    - "consistent with one prior case"
    - "no contradicting case"

conflicts_with:                    # LLM 主动标注的潜在冲突
  - knowledge_id: case-llama70b-2026-03-12
    note: "上次结论是 pp=8 + ep_overlap，但当时是 v0.5 backend，已不适用"

review_status: pending             # pending / accepted / rejected / superseded
```

**反模式（draft 拒收）**：

| 反模式 | Stage A 即拒 | 理由 |
|--------|------------|-----|
| `evidence` 为空或全部 `kind: llm_inference` | 自动拒 | 必须有 round_result 或 cross_session 引用 |
| `claim.headline` 长度 > 200 字符 | 自动拒 | 鼓励原子化 claim，复杂结论拆多个 draft |
| `binding` 留空或全 `*` | 自动拒 | 无适用边界的"普适规律"基本是噪声 |
| 与 `skills/knowledge/anti-patterns.md` 任一条直接矛盾且无新证据 | 自动拒 | 防止覆盖已知反例 |

### S4.3 Stage B：Triage（curator + 人工）

`curator` 是一个独立的 agent / CLI 工具（**不是** Pilot 主流程的一部分），定期扫描 `state/knowledge_drafts/`，输出**审核队列**。

**自动信号**（curator 给每个 draft 打分）：

| 信号 | 加分/减分 |
|------|----------|
| `confirmation_count`（多少独立 session 出现同 claim） | +0.2/次，cap +0.6 |
| 与现有 knowledge 的 `applicability` 完全重叠 | -0.3（重复） |
| 与现有 knowledge 矛盾（除非显式 `supersedes`） | -0.5 |
| 来自 final_best_case 且 final_tps / baseline > 1.5 | +0.2 |
| 来自 failure_pattern 且 dead 子树 > 5 个节点 | +0.2 |
| `evidence` 跨 ≥ 2 集群 | +0.3 |
| `evidence` 仅 1 个 session | -0.2 |

`triage_score = self_assessed_confidence + sum(signals)`

**分级**：

| triage_score | 处理 |
|--------------|------|
| ≥ 1.0 | 标 `auto_acceptable`，可一键 PR（仍需 reviewer 点 approve） |
| 0.5 – 1.0 | 进**人工审核队列**（一次最多 10 条） |
| < 0.5 | 标 `low_signal`，归档不入主库（保留在 drafts/，30 天后清理） |

**Curator CLI**（`tools/curate.py`）：

```bash
# 列出待审核
pilot curate list --since 2026-04-15

# 批量自动接受高分草稿
pilot curate auto-accept --threshold 1.2 --dry-run
pilot curate auto-accept --threshold 1.2 --apply  # 生成 PR

# 单条审核
pilot curate show drft_20260421_a3_b2
pilot curate accept drft_20260421_a3_b2 --merge-into knowledge/cases.md
pilot curate reject drft_20260421_a3_b2 --reason "evidence too thin"
pilot curate supersede drft_20260421_a3_b2 \
    --replaces case-llama70b-2026-03-12 \
    --merge-into knowledge/cases.md
```

### S4.4 Stage C：Merge（git 化）

接受的 draft 通过 PR 合并到 `skills/knowledge/`。**强约束**：

1. **PR 只能由 curator CLI 生成**（带签名 `Pilot-Curated-By: <user>`），人手 commit 不接受（防止绕过流程）
2. **每条 knowledge 必须包含**：
   - `knowledge_id`（不可变 ULID）
   - `applicability` / `not_applicable_when`
   - `evidence_refs`（指回原 draft 的 frozen 副本）
   - `created_at` / `last_confirmed_at`
   - `confirmation_count`
3. **PR 必须 ≥ 1 reviewer approve**（不论是否 auto_acceptable）
4. **CI 检查**（`scripts/lint_knowledge.py`）：
   - applicability 与现有 knowledge 不全等（否则要求 supersede）
   - 没有 broken evidence_refs
   - claim 与 anti-patterns 不矛盾

**Knowledge entry 最终形态**（`skills/knowledge/cases.md` 单条）：

```yaml
# skills/knowledge/cases.md（YAML front-matter blocks）
---
knowledge_id: 01HV9X8K7N2QD3F4G5H6J7K8L9
title: "Llama 70B+ on MI300X-8GPU: prefer pp=4 over pp=8"
applicability:
  - "model_family=llama_dense, model_size_b∈[70,140]"
  - "cluster_class=mi300x_8gpu, node_count∈[8,32]"
  - "framework=megatron, version∈[0.6, 0.8]"
not_applicable_when:
  - "model_size_b > 200"
  - "framework=torchtitan"
recommendation:
  hint: "默认起点 pp=4 + vpp=2"
  rationale_skill: skills/optimization/pipeline/vpp.md
created_at: 2026-04-21T18:30:00Z
last_confirmed_at: 2026-04-21T18:30:00Z
confirmation_count: 1
evidence_refs:
  - state/archive/drafts/drft_20260421_a3_b2.yaml
status: active                     # active / superseded / retired
---

详细 narrative ...
```

### S4.5 老化与撤回

**自动老化扫描**（每周一次 CI job）：

| 条件 | 动作 |
|------|------|
| `last_confirmed_at` > 90 天 且 `confirmation_count = 1` | 标 `stale`，下次 LEARN 不读取，挂在主库底部 |
| 同一 `binding` 出现新 knowledge 的 `applicability` 完全覆盖旧条 | 提示 reviewer 把旧条 `supersede` |
| 连续 3 次 session 在该 binding 下 evidence 与 knowledge 矛盾 | 自动开 issue，reviewer 决定 retire |
| `framework_version` 已超出 `applicability` range（外部升级） | 标 `version_drift`，等待 reconfirm |

**撤回（retire）**：knowledge 不删除，只标 `status: retired` + 保留原文，便于 audit 回放。

### S4.6 LLM 读取协议（防止用到坏知识）

LEARN 是写入侧；**Diagnose / Re-Plan 读取 `skills/knowledge/` 时**也要协议化（`skills/knowledge/SKILL.md` 给出）：

1. **优先级**：active > stale；不读 retired / superseded
2. **Binding 命中要求**：所有 `applicability` 子句必须命中当前 (cluster, model, framework)；命中部分但 `not_applicable_when` 击中 → 跳过
3. **多条相关时**：按 `confirmation_count desc, last_confirmed_at desc` 排序，取 top-3
4. **冲突处理**：若 top-3 内出现 `recommendation.hint` 互斥（同一轴方向相反），**全部丢弃 + 记录 conflict**——退化到不用 knowledge 的纯模型决策

### S4.7 跨集群迁移（cluster_class 抽象）

`cluster_id` 是具体集群（mi300x-16node、mi300x-32node），`cluster_class` 是同代 GPU + 同代 NIC + 同 GPU/节点数的等价类（mi300x_8gpu）。

**为什么**：knowledge 大多数对 GPU/节点数和 NIC 类型敏感，但**对节点总数的弹性远高于对单节点配置**。绑定 `cluster_class` 让 knowledge 在同 class 不同规模间复用。

**`cluster_class` 的判定规则**（`skills/knowledge/cluster_class.md`，**新增 Skill**）：

```python
def cluster_class(profile: ClusterProfile) -> str:
    return f"{profile.gpu_arch}_{profile.gpus_per_node}gpu_{profile.nic_class}"
    # 例：mi300x_8gpu_cx7
```

不同 `cluster_class` 之间 knowledge **不自动迁移**；可由人工 reviewer 显式 promote。

### S4.8 Multi-tenancy 协调（多 session 同时跑）

| 场景 | 协议 |
|------|------|
| 多 session 同时产 draft | `state/knowledge_drafts/<session_id>/` 天然隔离，无冲突 |
| 多 session 读同一份 `skills/knowledge/` | 只读，无冲突 |
| Curator 合并 PR 时上游已被另一 PR 改动 | git 标准 rebase 流程；CI 重跑 lint |
| 同周期内多 draft 主张相同 claim | curator 识别相似度（embedding + headline lev distance），合并为 `confirmation_count=N` 的单条 |

### S4.9 评估指标增量

`§10` 新增：

| 维度 | 指标 | 目标 |
|------|------|------|
| Knowledge 质量 | 入库 6 个月内被 supersede / retire 的比例 | ≤ 20% |
| Knowledge 质量 | 读 knowledge 后 Re-Plan 的命中率（按建议跑出 ≥ predicted gain 的比例） | ≥ 60% |
| 治理效率 | 平均 draft → merge 时长 | ≤ 7 天 |
| 治理效率 | reviewer 审核负担（人工审核条数 / 周） | ≤ 20 |

### S4.10 与现有章节的交叉引用

- §3.2 LEARN：LEARN 不再"直接回写 skills/knowledge/"，而是"写 draft + 提交审核队列"
- §4.1 目录树：新增 `state/knowledge_drafts/`、`state/archive/drafts/`、`scripts/curate.py`
- §4.2 skills/：knowledge/ 目录下新增 `SKILL.md`（读取协议）、`cluster_class.md`、`drafts_lifecycle.md`
- §5 Tool 接口：`knowledge.write` **只能写 drafts/**，不能写 skills/
- §8 schema 新增：`knowledge_draft.schema.json`、`knowledge_entry.schema.json`
- §12.1 Guardrails 新增："skills/knowledge/ 写权限隔离"、"draft schema 校验"、"冲突检测"

---

## 跨章节的统一交叉引用表

把 4 个补充章节对原文档的修改集中在一处，便于 review 与合并。

| 章节 | 修改类型 | 原文档位置 |
|------|---------|----------|
| §S1 / §S3 | 新增 Tool 接口 | §5 Tool 接口表 |
| §S1 / §S2 / §S3 / §S4 | 新增 schema | §8 数据结构 |
| §S2 | 拆分状态机节点 | §3.1 主图 + §3.2 流程说明 |
| §S1 / §S2 / §S4 | 新增 Skill 子树 | §4.2 skills/ 目录 |
| §S1 / §S2 / §S3 / §S4 | 新增评估指标 | §10 评估指标 |
| §S1 / §S2 / §S3 / §S4 | 新增 Guardrail / 失败回路 | §12.1 / §12.2 |
| §S3 | Subagent 边界微调 | §13.2 边界表 |

## 落地优先级（建议）

| 阶段 | 必须落地 | 可后置 |
|------|---------|-------|
| MVP（5–8 round 短任务） | §S2（reference + tolerance）、§S4 部分（draft schema + 写权限隔离） | §S1（cold-start 默认值即可）、§S3（用 TimeMux）、§S4 完整流水线 |
| 核心（10+ round / 多模型） | §S1（calibration 完整）、§S3（Sharded + 黑名单）、§S4 完整流水线 | §S4 跨集群迁移 |
| 完整（多天 / 多团队） | 全部 | — |

每个阶段都不阻塞后一阶段的扩展点（schema 设计预留 backward compatibility）。

---

## 一句话定位

§S1 给搜索装上**校准过的引擎**（让 priority 公式不只是漂亮的形式），§S2 给数值闸门装上**有据可查的 tolerance**（让 CORRECTNESS 不再误报与漏报），§S3 给并行执行装上**资源 + 隔离契约**（让 §3.1 那句"3 个 plan 并行"真正落地），§S4 给经验沉淀装上**门禁**（让 LEARN 不会随时间退化为知识噪音库）。
