# Primus Pilot — 训练调优系统

**Status**: Design spec

> 训练任务的自动调优系统。  
> Agent（任意支持工具调用的 LLM，如 Cursor / Claude / Codex）读取 Skills 中的知识，调用 Tools 执行操作，完成建模→搜索→收敛的闭环。

---

## Scope & Positioning

> **Pilot 是调优领域的知识 + 工具包，不是 agent runtime。**  
> 凡是"怎么推理、怎么隔离 context、怎么派生 subagent"这类问题，都交给具体的 agent 框架（Claude Code / Cursor / Codex / 自研 harness）。Pilot 只提供"调这件事该知道什么、该做什么、状态长什么样"。

### 分工表

| 关切 | Pilot 负责 | Agent 框架负责 |
|------|------------|----------------|
| 调优领域知识（瓶颈分类、优化策略、env catalog、...） | ✓ `skills/*.md` | — |
| Orchestrator / Stage Worker 的**角色 prompt** | ✓ `prompts/` | 负责注入 |
| 业务动作（preflight / submit / observe / constraint / ...） | ✓ `tools/`（CLI 或 MCP） | 负责调用 |
| 数据契约（TuningState / PlanGraph / SubagentResult / ...） | ✓ `schemas/` | 负责解析与校验 |
| 状态持久化（YAML/JSON 落盘、checkpoint 目录布局） | ✓ `state/` + `tools/state/*` | 负责读写 |
| LLM 调用、tool_use 解析、retry、rate limit | — | ✓ |
| **Subagent 隔离**（独立 context、单独 tool scope） | — | ✓ Claude Code Task / Cursor Task / ... |
| **Context 管理**（压缩、窗口、handoff） | — | ✓（Pilot 只提供策略规则 Skill，§13.2 策略 A） |
| 工具协议（MCP / function calling / shell） | — | ✓ |

### 核心守则

1. **Pilot 的核心目录（skills / prompts / tools / schemas / state）不 import 任何 agent SDK。**  
   anthropic / cursor-client / openai / ... 这类依赖只允许出现在 `integrations/` 下（如果有）。
2. **Tools 以进程或 MCP 边界暴露，不以 Python 函数边界暴露。**  
   CLI（`python -m pilot.tools.submit --plan ...`）或 MCP server 方法；任何框架都能通过 shell/MCP 接入，不必绑定 SDK。
3. **Prompts 是 framework-agnostic Markdown。**  
   不写 API 格式（`input_schema` / `tool_choice` / ...）。工具注册由 integrations 层做。
4. **Schemas 以 JSON Schema 为源**；对 Python 方便则生成 Pydantic mirror。
5. **State 就是文件系统**。任何 agent 通过 shell 或 `tools/state/*` CLI 读写。

### 角色与职责的归属

- **§2.2 的 Orchestrator / Stage Worker 是"角色"（roles）**，不是 Pilot 自己实现的组件——由 agent 框架的主会话承担 Orchestrator，由框架原生 subagent 机制（Claude Code Task / Cursor subagent）承担 Stage Worker。Pilot 负责写清楚两个角色的 prompt、可用工具、输出契约。
- **§13 三层 context 策略的归属**：
  - **策略 A（State-first protocol）**：Pilot 主责——写进 `skills/workflow/orchestration.md` 作为 Agent 必须遵守的规则 + `schemas/` 约束 `SubagentResult.summary` 大小
  - **策略 B（Subagent isolation）**：**框架主责**——Pilot 只给出"哪些 stage 该派生 subagent"的边界表（§13.2）；具体 spawn 机制交给框架
  - **策略 C（Session handoff）**：**框架主责**——Pilot 只规定 handoff 的 state 文件格式（§8.7）

### 关于 `pilot/agent/`

`pilot/agent/` 是**可选的 Python 参考实现 / fallback**，展示"若要自己搭 harness 该怎么写"。  
**生产路径推荐**：直接让 agent 框架消费 `pilot/skills` + `pilot/prompts` + `pilot/tools`；框架侧的薄适配层放在 `pilot/integrations/<framework>/`。

---

## 目录

0. [Scope & Positioning](#scope--positioning)
0.1 [前置条件 — 调用 Pilot 之前需要准备什么](#前置条件)
1. [问题与边界](#1-问题与边界)
2. [系统架构](#2-系统架构)
3. [系统流程](#3-系统流程)
4. [目录结构](#4-目录结构)
5. [Tool 接口](#5-tool-接口)
6. [Execution Model（核心知识）](#6-execution-model核心知识)
7. [搜索空间维护与解保证](#7-搜索空间维护与解保证)
8. [数据结构（Schema）](#8-数据结构schema)
9. [完整迭代示例](#9-完整迭代示例)
10. [评估指标](#10-评估指标)
11. [与现有系统的集成](#11-与现有系统的集成)
12. [Guardrails](#12-guardrails)
13. [Context Management & Multi-Agent Orchestration](#13-context-management--multi-agent-orchestration)
14. [一句话总结](#14-一句话总结)

---

## 前置条件

Pilot 是**调优系统**，不是 cluster manager。在执行任何 `python -m pilot.tools.*` 命令之前，使用者（用户 / SRE）必须先把下面两种运行环境之一准备好：

| 模式 | 你来准备什么 | Pilot 然后做什么 |
|------|--------------|------------------|
| **`single`** | 本机一个 container，N 张 GPU 在容器内可见 | 所有事情都在该 container 内做；不跨节点 fan-out |
| **`slurm`**  | 一个已分配的 SLURM 资源（`salloc -N k`）+ 每台节点上的 container 接入方式 | 用 `srun --jobid=<id>` 接入既有分配；**不**自己创建或销毁 allocation |

你通过一份小小的 `cluster.yaml` 来声明当前是哪种模式（schema：`schemas/cluster_config.schema.json`）。每个 Pilot 工具都把这份文件当作**唯一的环境输入**。**没有任何回退路径**：`cluster.yaml` 缺失或过期 → 每个工具都立即以 `failure.kind=CLUSTER` 退出，并指出修复方法。

→ 详细操作指南请看 **`SETUP.md`**（docker run / salloc 命令、container 接入方式、健康检查、常见坑）。  
→ Agent 视角的同一份契约请看 **`AGENTS.md` §4**。

这种边界设定是**有意为之**：让 Pilot 保持小、可移植、可复现。所有原本会以 "ssh fan-out 配置"、"docker compose"、"k8s job manifest" 形态存在的事情，统统留在 Pilot **之外**。

---

## 1. 问题与边界

| 挑战 | 具体问题 |
|------|---------|
| **参数空间爆炸** | DP / TP / PP / EP / VPP / CP × MBS × recompute × 通信参数，组合 10⁴+ |
| **瓶颈定位困难** | compute / comm / memory / bubble 混合交织，每次从零排查 |
| **试错成本高** | 一次多节点实验数百 GPU·h，坏实验跑完才知道 |
| **经验碎片化** | best-known config 散落在 Slack / wiki / 脑中，不可复用 |

**做什么**：Dense / MoE bring-up、scaling 退化诊断、并行 + 通信联合调参。

**不做什么**：自动改 kernel / 模型结构 / 通信库实现。

---

## 2. 系统架构

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Agent Layer                                │
│              (任意工具调用 LLM，如 Cursor / Claude / Codex)          │
│                                                                     │
│   - 读取 Skills 获取知识（workflow / execution-model / 优化策略）     │
│   - 读写 State 维护工作记忆（PlanGraph / TuningState / ...）          │
│   - 调用 Tools 执行操作（preflight / submit / observe / env_probe）  │
│   - 按 state_machine.md 的转移规则驱动 Tuning Loop                   │
└────────────────────┬─────────────┬────────────────────┬─────────────┘
                     │             │                    │
                     ▼             ▼                    ▼
┌───────────────────────┐ ┌──────────────────┐ ┌───────────────────────┐
│     Skill Layer       │ │   State Layer    │ │      Tool Layer       │
│   (知识 / Markdown)   │ │ (工作记忆/YAML)  │ │   (执行 / Python)     │
│                       │ │                  │ │                       │
│  skills/              │ │ state/           │ │  tools/               │
│  ├── workflow/        │ │ ├── cluster_     │ │  ├── preflight.py     │
│  │   (state_machine,  │ │ │   profile.yaml │ │  ├── env_probe.py     │
│  │    plan_graph,     │ │ ├── tuning_      │ │  ├── profiler.py      │
│  │    replan, settle, │ │ │   state.yaml   │ │  ├── submit.py        │
│  │    smoke,          │ │ ├── plan_        │ │  ├── observe.py       │
│  │    correctness …)  │ │ │   graph.yaml   │ │  ├── constraint.py    │
│  ├── execution-model/ │ │ ├── candidate_   │ │  ├── state.py         │
│  ├── optimization/    │ │ │   pool.yaml    │ │  └── knowledge.py     │
│  ├── env/             │ │ └── checkpoints/ │ │                       │
│  ├── profiling/       │ │   r0/, r1/, ...  │ │  Agent 通过 function  │
│  ├── constraints/     │ │                  │ │  call 调用这些函数；   │
│  └── knowledge/       │ │ 每 stage 出口    │ │  函数读写 State Layer  │
│                       │ │ checkpoint，可中 │ │                       │
│  Agent 读取 .md 获取   │ │ 断续起 / 回放    │ │                       │
│  领域知识和规则        │ │                  │ │                       │
└───────────────────────┘ └──────────────────┘ └───────────────────────┘
         ▲                          ▲                       ▲
         │                          │                       │
         └────────── LEARN 阶段把 best/失败 case ────────────┘
                    回写 skills/knowledge/
                    （唯一的 Skill ← State 反向流）
```

**四层职责切分**：

| 层 | 形式 | 谁写 | 谁读 | 例子 |
|----|------|------|------|------|
| **Agent** | LLM 推理 | — | Skills + State | "comm_ratio=0.35，按 skills/optimization/comm/SKILL.md 应试 overlap" |
| **Skill** | Markdown | 人类（少数 LEARN 阶段由系统） | Agent | `T_bubble = (pp-1)/(pp-1+M) × T_comp` |
| **State** | YAML / JSON | Agent 调 `state.checkpoint()` | Agent + 审计/回放 | `PlanGraph.champion = r2_p4` |
| **Tool** | Python 函数 | 人类 | Agent (function call) | `preflight.run()` 返回 ClusterProfile，写入 State |

**关键设计原则**：
- **Skill ↔ State 单向**：Agent 读 Skill 决定怎么做、读 State 决定下一步；只有 LEARN 阶段会反向写 Skill（沉淀经验）
- **State 是单一真相源**：所有跨 stage 的工作记忆都在 State Layer；Tool 是无状态函数（输入 → 输出 + State 更新）
- **Skill 是知识，不是逻辑**：所有"if X then Y"的规则也写在 Markdown 里（如 `state_machine.md` 的转移表）；Agent 是规则的执行者，不是规则的拥有者
- **可观测 = 可回放**：State 全程持久化，任何一次决策都能从 `state/checkpoints/rN/` 完整重现
- **Agent 是两层结构**：Orchestrator 只拿指针，Stage Worker 吞细节；详见 §2.2 与 §13

### 2.2 Agent Orchestration Model

> 本节描述的 **Orchestrator / Stage Worker 是两种"角色"（roles），不是 Pilot 自己实现的组件**。它们由具体的 agent 框架（Claude Code 的主会话 + Task 派生子 agent、Cursor 的 Agent + subagent、自研 harness、...）承担。Pilot 只负责定义这两个角色的职责、可见工具集、context 预算、输出契约——具体"怎么派生、怎么隔离、怎么管 context"交给框架。参见 **Scope & Positioning** 一节。
>
> **为什么拆成两层**：如果让单一会话同时承担"状态机推进"（小而长）和"每 stage 的具体推理"（大而短），context 会随 round 线性膨胀到注意力稀释区；把角色拆开，让框架原生的 subagent 机制兑现隔离，Orchestrator 的稳态 context 才能保持 O(1)。

```
┌───────────────────────────────────────────────────────────────────┐
│                   Orchestrator Agent (长生命周期)                  │
│                                                                   │
│  持有（稳态 < 2K tokens）：                                        │
│    - session_id, current_stage, round_id                          │
│    - champion_id（PlanGraph 的指针，不是节点详情）                  │
│    - budget_used, budget_remaining                                │
│    - last_decision（一行摘要）                                     │
│                                                                   │
│  职责：                                                           │
│    - 读 skills/workflow/state_machine.md + orchestration.md       │
│    - 按转移规则决定下一 stage                                      │
│    - 调 subagent.spawn() 派生 Stage Worker                         │
│    - 收回 SubagentResult（< 200 tokens），更新指针，落 checkpoint   │
│    - 绝不 re-load Snapshot / CandidatePool / Skill 细节            │
└──────┬────────┬─────────────┬──────────────┬─────────────┬────────┘
       │        │             │              │             │ spawn
       ▼        ▼             ▼              ▼             ▼
   ┌───────┐┌─────────┐  ┌──────────┐  ┌─────────────┐┌──────────┐
   │Diagnose││Re-Plan │  │EnvSweep  │  │Correctness- ││Preflight │
   │ Worker ││ Worker │  │ Worker   │  │Lite Worker  ││ Worker   │
   │(一次性)││(一次性)│  │(一次性)  │  │(一次性)     ││(一次性)  │
   └───┬───┘└───┬────┘  └────┬─────┘  └──────┬──────┘└────┬─────┘
       │        │             │              │             │
       │        │   - 只读自己那一片 Skill 子树              │
       │        │   - 只读 State 的相关切片                  │
       │        │   - 调业务 Tool（submit / observe / ...）  │
       │        │   - 写回 State Layer                      │
       │        │   - 返回 SubagentResult                   │
       ▼        ▼             ▼              ▼             ▼
  ┌──────────────────────────────────────────────────────────────┐
  │                    State Layer (共享记忆)                     │
  │   PlanGraph / CandidatePool / TuningState / checkpoints/     │
  │   Worker 之间 *不* 直接通信，全部走这里（隔离性 + 可审计）      │
  └──────────────────────────────────────────────────────────────┘
```

**两层契约**：

| 维度 | Orchestrator | Stage Worker |
|------|--------------|--------------|
| 生命周期 | 整个 session（可跨天） | 单 stage 执行，完成即销毁 |
| Context 预算 | 稳态 < 2K tokens / round | 单次峰值 < 30K tokens |
| 读 Skill | 仅 `workflow/state_machine.md` + `orchestration.md` | 仅该 stage 相关子树（如 `optimization/comm/*`） |
| 读 State | 仅 `TuningState.summary` | 按需读 PlanGraph / Snapshot 等切片 |
| 写 State | 仅更新指针类字段 | 写入该 stage 的完整产物（DiagnosisReport / CandidatePool / ...） |
| 调 Tool | 仅 `state.*` / `subagent.spawn()` | 所有业务 Tool（submit / observe / constraint / ...） |
| 能看到历史 | 看不到 Worker 的推理 trace | 看不到其他 Worker 的 context |

**这个拆分带来的性质**：
- Orchestrator 只持指针，细节留在 State Layer；Stage Worker 每次新生，单次用完即丢，不累积 context。
- 单 round 内 Orchestrator 增量 < 500 tokens，长循环（20+ round）下 context 不线性膨胀。
- 副作用：多 plan 的 Observe / Diagnose 可以并行 spawn subagent，天然并行化。

**谁来真正派生 subagent**：

| 框架 | Orchestrator 承载物 | Stage Worker 承载物 |
|------|-------------------|---------------------|
| Claude Code | 主会话 | Task tool 派生的 subagent |
| Cursor | Agent 主会话 | Task tool 派生的 subagent |
| OpenAI Codex | 主会话 | 其 subagent 机制 |
| 自研 harness（`pilot/agent/`）| `Orchestrator` Python 类 | `StageWorker` Python 类 + 独立 Claude API 会话 |

Pilot 的角色是**给所有这些承载方式写同一套 prompt + tool scope + 输出契约**，使得不同框架下 Orchestrator 表现一致。

---

## 3. 系统流程

> 三层视图：**§3.1 块状主图** 给 stage 输入/输出概览（最易懂） → **§3.2 流程说明** 用文字串起每个 stage 的职责与转移规则 → **§3.3 内部泳道** 展开 Tuning Loop 的角色分工（Agent / Skill / Tool）。

### 3.1 块状主图（主干）

```
                         ┌──────────────────────────────┐
                         │        User Input            │
                         │  - Model Spec                │
                         │  - Cluster Size              │
                         │  - TargetVector              │
                         │    (primary / constraints /  │
                         │     budget)                  │
                         └──────────────┬───────────────┘
                                        │
                                        ▼
┌──────────────────────────────────────────────────────────────────────┐
│                        1. PREFLIGHT++                                │
│                                                                      │
│  Collect:                                                            │
│    - GEMM / MFMA peak                                                │
│    - IB / XGMI bandwidth                                             │
│    - AllReduce / All2All baseline                                    │
│    - env probe (NCCL/HSA/alloc 连通性 + micro-bench)                 │
│                                                                      │
│  Output:                                                             │
│    ClusterProfile = {                                                │
│        compute_peak, comm_bw, latency, overlap_capability,           │
│        env_baseline (cluster_shared 默认)                            │
│    }                                                                 │
│  （按 version + age 跨任务复用；过期则自动重跑）                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                     2. PROJECTION / Modeling                         │
│                                                                      │
│  Input: Model Spec + ClusterProfile                                  │
│                                                                      │
│  Step 1: Single-node profiling                                       │
│    (layers, mbs, recompute) → T_comp, Mem_peak                       │
│                                                                      │
│  Step 2: Build Execution Model                                       │
│    T_comp(l, mbs) / Mem(l, mbs) / T_comm / Bubble(P, M)              │
│                                                                      │
│  Step 3: Generate Initial Plans                                      │
│    Plan = { parallel, partition, mbs, recompute,                     │
│             env.diff (scale-aware), expected: {tps, bottleneck} }    │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                          3. SMOKE                                    │
│  Tiny scale × 100 step：验证可起 / 无静默 hang / 无显存溢出           │
│  失败 → 回 PROJECTION                                                │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                         4. BASELINE                                  │
│  完整规模跑起点；记入 history[0]                                      │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                       5. CORRECTNESS                                 │
│  loss curve / grad norm vs reference 对齐                            │
│  失败 → ABORT + escalate（数值正确性破坏，停下问人）                  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
        ┌────────────────────────────────────────────────────┐
        │            Tuning Loop（核心 / 双层）               │
        └────────────────────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                      6. 外层 · 结构循环                              │
│                                                                      │
│   Execute → CORRECTNESS-LITE → Observe → Diagnose → Re-Plan          │
│                                                  │                   │
│                                                  ▼                   │
│   ┌───────────────────────────────────────────────────────────────┐  │
│   │  Re-Plan 子流程                                                │  │
│   │                                                                │  │
│   │  ① 从 PlanGraph 选派生源（默认 champion；stagnation→shelved） │  │
│   │  ② Skill Mapping by bottleneck:                                │  │
│   │      COMM    → bucket / overlap                                │  │
│   │      PIPELINE→ vpp / microbatch                                │  │
│   │      MEMORY  → recompute / offload                             │  │
│   │      COMPUTE → mbs / parallel                                  │  │
│   │  ③ 生成 CandidatePool（每候选含                                │  │
│   │      predicted_gain × confidence / cost = priority）           │  │
│   │  ④ Constraint Check (OOM / invalid / env 不兼容)              │  │
│   │  ⑤ exhausted_neighborhoods 去重                                │  │
│   │  ⑥ Strategy Select：                                           │  │
│   │      cluster_shared+weakly_local 为主 → Champion-Challenger    │  │
│   │      strongly_local + 模型可信        → Per-Plan + Pruning     │  │
│   │      预算宽裕 / 不确定                → Successive Halving     │  │
│   │  ⑦ 输出 top-K plan 进 Execute                                  │  │
│   └───────────────────────────────────────────────────────────────┘  │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼ (条件触发：env_suspect 命中 且 结构稳定)
┌──────────────────────────────────────────────────────────────────────┐
│                   7. 内层 · EnvSweep（可选）                         │
│                                                                      │
│   - 锁定外层 best plan 的结构                                        │
│   - 扫 weakly_local env axes（NCCL_BUFFSIZE / alloc_conf / ...）     │
│   - 30-50 step 并行短跑，挑 best env diff 合并进 baseline            │
│   - 单次 cap：≤ 5 flag、≤ 8 组合                                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
                               ▼
┌──────────────────────────────────────────────────────────────────────┐
│                    8. Settle / Convergence                           │
│                                                                      │
│   维护 PlanGraph (树 + frontier + exhausted_neighborhoods):           │
│     - new_best.tps > champion × (1+ε_promote=2%) → 升 new champion   │
│         旧 champion 进 shelved（不丢，留作 backtrack 候选）           │
│     - 微弱提升 → champion 不变，新 best 进 shelved                    │
│     - 子树连续 2 轮 dead 率 > 50% → backtrack 到 frontier 次优        │
│     - 每 K=3 轮强制一次 explore round（候选只从 shelved 派生）        │
│                                                                      │
│   Stop Conditions:                                                   │
│       · TargetVector.constraints 全满足 且 primary 不再显著提升       │
│       · gain < 2% (连续两轮) 且 frontier 中无高 priority 候选         │
│       · max_rounds / budget 触达                                     │
└──────────────────────────────┬───────────────────────────────────────┘
                               │
              不收敛 ◄─────────┴──────────► 收敛
                  │                            │
       回到 6. 外层下一轮                       ▼
                              ┌───────────────────────────────────────┐
                              │             9. REPORT                 │
                              │   Final Best Config + decision trace  │
                              └────────────────┬──────────────────────┘
                                               │
                                               ▼
                              ┌───────────────────────────────────────┐
                              │             10. LEARN                 │
                              │   best / 失败 case 回写               │
                              │   skills/knowledge/                   │
                              └───────────────────────────────────────┘

【Reentry edges】（任意阶段触发回跳，不计 round 配额）
  Diagnose 发现 ClusterProfile 过期    → PREFLIGHT
  Re-Plan 发现结构性失效（规格变更）    → PROJECTION
  HANG (NCCL/IB timeout)                → PREFLIGHT (env_probe)
  CLUSTER (节点掉线 / 驱动错误)         → PREFLIGHT
  OOM / INVALID_CONFIG                  → Re-Plan（标记 dead）
  NUMERICAL drift (CORRECTNESS-LITE)    → ABORT + escalate
```

### 3.2 流程说明（状态机驱动，每个 stage 出口持久化 `TuningState`，支持中断续起）


1. **需求收集**：用户输入 Model Spec / Cluster Size / **TargetVector**（primary + constraints + budget，见 §8.6）
2. **读 workflow Skill**：Agent 读取 `skills/workflow/SKILL.md` + `state_machine.md`，了解阶段集合与转移规则
3. **项目信息收集**：Agent 调用 Tool 采集集群与模型配置
4. **进入状态机驱动的 workflow**：
   - **PREFLIGHT**：硬件基线 + 集群级 env baseline → `ClusterProfile`（任务间复用，按 `version`+`age` 决定是否重跑）
   - **PROJECTION**：单机 profiling + Execution Model + 初始 Plans（含 scale-aware env diff）
   - **SMOKE**：全配置 + tiny scale（如 1 节点 / 100 step）；验证可起、无静默 hang；失败回 PROJECTION
   - **BASELINE**：完整规模跑起点
   - **CORRECTNESS**：loss curve / grad norm 与 reference 对齐（数值闸门，失败则 ABORT 上报）
   - **OPTIMIZE_LOOP（双层 + 状态机）**：
     - 外层（结构）：`Observe → Diagnose → Re-Plan → Execute → CORRECTNESS-LITE → Settle`
     - 内层（EnvSweep，可选）：在 Settle 后触发，锁结构扫 env diff
     - 任意阶段可基于 `reentry_when` 回跳（如 Diagnose 发现 ClusterProfile 过期 → PREFLIGHT；Re-Plan 发现结构性失效 → PROJECTION）
     - Guardrail 触发走显式失败回路（见 §12），不消耗 round 配额
   - **REPORT**：输出 Final Config + 决策 trace
   - **LEARN**：best config 与失败 case 回写 `skills/knowledge/`，供下次复用
5. **验收**：跑完整测试，审计日志，确认 commit

> **为什么外层/内层分开**：env 调参不改变形状（无 OOM 风险）、单次成本低（30-50 step 即可分辨），适合在每个外层 baseline 稳定后做一次"安全 sweep"；env 的最优值依赖结构（mbs / world_size），所以**不能一次跑完一劳永逸**，必须嵌在外层每轮里。
>
> **为什么用状态机而不是线性管线**：让"加新阶段 / 加跳转 / 加运行模式（如 RL post-training 双 loop）"变成"加节点 + 加边"，不需要改主泳道图；同时显式声明 `reentry_when`（回跳条件）与 `on_fail`（失败转移），让 Agent 的决策有规则可循。

### 3.3 Tuning Loop 内部泳道（角色分工详细展开）

> 双层视图：**外层 Orchestrator 只推状态机 + spawn subagent**，**内层 Stage Worker（即下面泳道中的 "Agent" 列）承担该 stage 的 Skill 读取与 Tool 调用**。Orchestrator 在两个 Worker 之间只看到 `SubagentResult` 的一行摘要，不吸收 Worker 的推理 trace。

**外层（Orchestrator 视角，每 stage 约 < 500 tokens 新增 context）**：

```
[loop over rounds]
  1. 读 TuningState.summary（state.resume 产生，~200 tokens）
  2. 按 state_machine.md 决定 next_stage
  3. worker_result = subagent.spawn(
        stage = next_stage,
        input_refs = {snapshot_id, plan_graph_ref, ...},
        skill_scope = ["workflow/<stage>.md", "optimization/<bottleneck>/*"])
  4. 更新指针：champion_id / round_id / budget_used
  5. state.checkpoint() + state.trim()  ← 丢弃 Worker trace
  6. 判断 Stop / Continue / Handoff
```

**内层（Stage Worker 视角，单次 < 30K tokens，用完即销毁）**：

```
┌─────────────────────┬─────────────────────────┬─────────────────────────┐
│   Stage Worker      │        Skill            │        Tool             │
│  (one-shot agent)   │   (knowledge)           │   (execution)           │
├─────────────────────┼─────────────────────────┼─────────────────────────┤
│                     │                         │                         │
│ [Execute]           │                         │                         │
│     │               │                         │                         │
│     ├── 读 execute.md ► 执行规则              │                         │
│     │               │   - early_stop 阈值     │                         │
│     │               │   - 串行/并行策略       │                         │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► submit.run(plan)       │
│                     │                         │   └► 返回 run_id        │
│                     │                         │                         │
│ [CORRECTNESS-LITE]  │                         │                         │
│  (Execute 后随机抽查) │                         │                         │
│     │               │                         │                         │
│     ├── 读 correctness.md ► 数值闸门          │                         │
│     │               │   - loss_delta_pct 阈值 │                         │
│     │               │   - grad_norm 范围      │                         │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► observe.compare_loss() │
│                     │                         │   └► pass / drift       │
│                     │                         │                         │
│ [Observe]           │                         │                         │
│     │               │                         │                         │
│     ├── 读 observe.md ► snapshot schema       │                         │
│     │               │   - tps, bubble_ratio   │                         │
│     │               │   - comm_ratio, mem     │                         │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► observe.snapshot(run_id)
│                     │                         │   └► 返回 Snapshot      │
│                     │                         │                         │
│ [Diagnose]          │                         │                         │
│     │               │                         │                         │
│     ├── 读 diagnose.md ► 瓶颈分类规则         │                         │
│     │               │   if comm_ratio > X     │                         │
│     │               │     → COMM_BOUND        │                         │
│     │               │   if bubble > X         │                         │
│     │               │     → PIPELINE_BOUND    │                         │
│     │               │                         │                         │
│     └── Agent 推理得出 bottleneck             │                         │
│                     │                         │                         │
│ [Re-Plan]           │                         │                         │
│     │               │                         │                         │
│     ├── 读 optimization/{bottleneck}/ ────────┤                         │
│     │               │   返回优化策略:         │                         │
│     │               │   - 增大 TP             │                         │
│     │               │   - 调 bucket size      │                         │
│     │               │   - 开 overlap          │                         │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► constraint.check(plan) │
│                     │                         │   └► 返回 valid/invalid │
│                     │                         │                         │
│ [EnvSweep]          │                         │                         │
│  (条件触发：         │                         │                         │
│   bottleneck 命中    │                         │                         │
│   且结构稳定)        │                         │                         │
│     │               │                         │                         │
│     ├── 读 env/SKILL.md & env_probe.md ──────┤                          │
│     │               │   - 候选 flag 列表      │                         │
│     │               │   - 安全 sweep 协议     │                         │
│     │               │                         │                         │
│     ├── 读 optimization/{bottleneck}/env.md   │                         │
│     │               │   - 该瓶颈下相关 flag   │                         │
│     │               │                         │                         │
│     ├── 调 Tool ───────────────────────────────► constraint.check_env() │
│     │               │                         │   └► 拒绝危险组合       │
│     │               │                         │                         │
│     └── 调 Tool ───────────────────────────────► submit.run(plans, short)
│                     │                         │   └► 30-50 step 并行    │
│                     │                         │      返回 best env diff │
│                     │                         │                         │
│ [Settle]            │                         │                         │
│     │               │                         │                         │
│     ├── 读 settle.md ► 收敛规则               │                         │
│     │               │   - greedy pick best    │                         │
│     │               │   - stop conditions     │                         │
│     │               │                         │                         │
│     └── Agent 决定: continue / stop           │                         │
│           │                                   │                         │
│           ├── continue → 回到 [Execute]       │                         │
│           └── stop → 输出 Final Config        │                         │
│                     │                         │                         │
└─────────────────────┴─────────────────────────┴─────────────────────────┘
```

---

## 4. 目录结构

### 4.1 仓库整体布局

Pilot 按 Scope & Positioning 里声明的 5 类产物 + 2 类可选适配组织目录：

```
pilot/
│
├── skills/                         # 调优领域知识（Markdown；详见 §4.2）
│   ├── workflow/                   #   调优主流程（状态机 / orchestration / observe / diagnose / replan / settle / ...）
│   ├── execution-model/            #   训练建模公式（T_comp / Mem / T_comm / Bubble）
│   ├── optimization/               #   按瓶颈组织的优化策略（comm / pipeline / memory / compute / moe）
│   ├── env/                        #   env 调参 catalog（rccl / hsa / alloc / threading / presets）
│   ├── profiling/                  #   数据采集方法（preflight / env_probe / trace / ...）
│   ├── constraints/                #   安全约束（OOM 预估 / 配置合法性 / env 不兼容矩阵）
│   └── knowledge/                  #   LEARN 阶段写入目标（patterns / cases / anti-patterns）
│
├── prompts/                        # 角色 prompt（framework-agnostic Markdown）
│   ├── orchestrator.md             #   Orchestrator 角色：状态机推进 + spawn subagent
│   └── worker/                     #   每种 Stage Worker 的角色 prompt
│       ├── diagnose.md
│       ├── replan.md
│       ├── envsweep.md
│       ├── observe.md
│       ├── correctness_lite.md
│       └── preflight.md
│
├── tools/                          # 业务动作（Python；以 CLI / MCP 为边界暴露）
│   ├── preflight.py                #   preflight.run / env_probe.run / env_probe.sweep
│   ├── profiler.py
│   ├── submit.py                   #   submit.run / submit.cancel
│   ├── observe.py                  #   observe.snapshot / observe.compare_loss
│   ├── constraint.py               #   constraint.check / check_env / estimate_mem / diagnose_failure
│   ├── state.py                    #   state.checkpoint / resume / trim / handoff
│   ├── subagent.py                 #   subagent.spawn 的协议抽象层（具体实现由 integrations 注入）
│   └── knowledge.py                #   knowledge.write
│
├── schemas/                        # 数据契约（JSON Schema 为源 + Pydantic mirror）
│   ├── cluster_profile.schema.json
│   ├── plan.schema.json
│   ├── snapshot.schema.json
│   ├── diagnosis_report.schema.json
│   ├── env_sweep_result.schema.json
│   ├── target_vector.schema.json
│   ├── tuning_state.schema.json
│   ├── plan_graph.schema.json
│   ├── candidate_pool.schema.json
│   ├── subagent_result.schema.json
│   ├── failure_report.schema.json
│   └── pydantic/                   #   由 JSON Schema 生成，Python 消费方便
│
├── state/                          # 运行时产物目录（gitignore）
│   ├── cluster_profile.yaml        #   跨任务复用（按 version + age）
│   ├── tuning_state.yaml           #   每 stage 出口 checkpoint 的入口
│   ├── plan_graph.yaml
│   ├── candidate_pool.yaml
│   └── checkpoints/
│       ├── r0/, r1/, r2/, ...      #   每 round 的完整快照（可回放）
│       └── handoff/                #   Orchestrator 自我接力的落地点
│
├── integrations/                   # 薄适配层（可选，每个框架一个；不强依赖）
│   ├── claude-code/
│   │   ├── README.md               #   告诉 Claude Code 怎么跑 Pilot
│   │   ├── CLAUDE.md               #   入口 prompt，引用 skills / prompts / tools
│   │   └── mcp-server.py           #   MCP 包装 pilot/tools 的可选适配
│   ├── cursor/
│   │   ├── AGENTS.md               #   Cursor agent 的入口
│   │   └── rules/                  #   .cursor/rules/ 模板
│   └── codex/
│       └── README.md
│
└── agent/                          # 可选的 Python 参考 harness（fallback）
    │                               #   生产路径推荐走 integrations/<framework>/；
    │                               #   本目录适用于：无原生 subagent 的 LLM / headless 常驻 / 参考实现
    ├── orchestrator.py
    ├── subagent.py
    ├── schemas.py                  #   对 schemas/pydantic/ 的本地 mirror
    ├── state.py
    ├── skills.py
    ├── orchestrator_tools.py
    ├── worker_tools.py
    └── prompts/
```

**分层职责速记**：

| 目录 | 形态 | 时态 | 谁写 | 谁读 |
|------|------|------|------|------|
| `skills/`       | Markdown（知识） | 现在时（规则） | 人 / LEARN 阶段 | Agent / Worker |
| `prompts/`      | Markdown（角色） | 现在时（身份） | 人 | Agent 框架注入 |
| `tools/`        | Python（动作） | 命令式 | 人 | Agent（经 CLI / MCP）|
| `schemas/`      | JSON Schema（契约） | 静态 | 人 | 双方校验 |
| `state/`        | YAML（工作记忆） | 过去+现在 | Agent（经 `state.*` tool）| 全链路 / 审计 |
| `integrations/` | 各框架原生入口 | 静态粘合 | 人 | 框架 runtime |
| `agent/`        | Python（兜底 harness） | 命令式 | 人 | `python -m pilot.agent` |

**版本化与 gitignore**：
- `skills/` `prompts/` `tools/` `schemas/` `integrations/` `agent/` 全部进 git
- `state/` 整个目录进 `.gitignore`；运行产物按 `session_id` 归档到外部存储
- 回归 / CI 用的 fixture `state/` 放到 `tests/fixtures/state/`，与运行时目录分离

### 4.2 `skills/` 详细目录

```
skills/                                 # 唯一知识源（Agent 读取）
│
├── workflow/                           # 调优主流程（状态机驱动）
│   ├── SKILL.md                        # tuning loop 总体说明（含外/内层切换条件）
│   ├── state_machine.md                # 状态集合 / 转移规则 / reentry_when / on_fail
│   ├── orchestration.md                # Orchestrator ↔ Stage Worker 协议
│   │                                   #         + context hygiene 规则 + handoff 协议
│   ├── projection.md                   # 建模阶段
│   ├── smoke.md                        # 起前自检（tiny scale × 100 step）
│   ├── correctness.md                  # 数值闸门（loss/grad vs reference）
│   ├── observe.md                      # 观测数据定义（snapshot schema）
│   ├── diagnose.md                     # 瓶颈分类逻辑（含 env_suspect / reentry 触发）
│   ├── plan.md                         # plan 结构定义（含 env.diff）
│   ├── plan_graph.md                   # 解空间维护（树 + frontier + exhausted_neighborhoods）
│   ├── replan.md                       # 候选生成 + priority 公式 + 派生源选择
│   ├── axis_taxonomy.md                # 轴分类：cluster_shared / weakly_local / strongly_local
│   ├── execution_strategy.md           # Champion-Challenger / Per-Plan / Successive Halving 选择规则
│   ├── execute.md                      # 执行与 early stop
│   ├── envsweep.md                     # 内层 EnvSweep 协议（触发条件 / 候选 / 收敛）
│   ├── settle.md                       # 收敛逻辑（贪心 + 软回滚 + explore round）
│   └── learn.md                        # best/失败 case 回写 knowledge/ 的协议
│
├── execution-model/                    # 训练建模（核心知识）
│   ├── SKILL.md                        # 总体说明
│   ├── compute.md                      # T_comp(layers, mbs)
│   ├── memory.md                       # Mem(layers, mbs)
│   ├── communication.md                # T_comm / allreduce / alltoall
│   ├── pipeline.md                     # Bubble(pp, M)
│   ├── partition.md                    # layer partition / stage balance
│   └── examples.md                     # 建模示例（Dense / MoE）
│
├── optimization/                       # 按瓶颈类型组织的优化策略
│   ├── SKILL.md                        # 总体策略
│   │
│   ├── comm/                           # 通信瓶颈
│   │   ├── SKILL.md                    # reduce_comm_pressure
│   │   ├── bucket.md                   # bucket tuning
│   │   ├── overlap.md                  # overlap 优化
│   │   ├── topology.md                 # 跨节点 vs 单节点
│   │   └── env.md                      # COMM_BOUND 候选 env（→ env/rccl.md）
│   │
│   ├── pipeline/                       # pipeline 瓶颈
│   │   ├── SKILL.md                    # pipeline 优化策略
│   │   ├── vpp.md                      # VPP tuning
│   │   ├── microbatch.md               # MBS / GAS
│   │   └── balance.md                  # stage balance
│   │
│   ├── memory/                         # 显存瓶颈
│   │   ├── SKILL.md                    # memory 优化策略
│   │   ├── recompute.md                # activation recompute
│   │   ├── offload.md                  # CPU / NVMe offload
│   │   ├── fragmentation.md            # 内存碎片
│   │   └── env.md                      # MEMORY_BOUND 候选 env（→ env/alloc.md）
│   │
│   ├── compute/                        # 计算瓶颈
│   │   ├── SKILL.md                    # compute 利用率优化
│   │   ├── mbs.md                      # mbs scaling
│   │   ├── parallel.md                 # dp/tp 调整
│   │   ├── kernel.md                   # kernel-level hint
│   │   └── env.md                      # COMPUTE_BOUND 候选 env（→ env/threading.md, hsa.md）
│   │
│   └── moe/                            # MoE 专项
│       ├── SKILL.md
│       ├── routing.md
│       ├── dispatch.md
│       └── load_balance.md
│
├── env/                                # env 调参 catalog（事实源 / 单点维护）
│   ├── SKILL.md                        # env 调参总论 / 双层循环触发条件
│   ├── rccl.md                         # NCCL_*/RCCL_* 全集（默认 / 范围 / 已知坑）
│   ├── hsa.md                          # HSA_*/HIP_*/GPU_MAX_HW_QUEUES
│   ├── alloc.md                        # PYTORCH_HIP_ALLOC_CONF / MALLOC_*
│   ├── threading.md                    # OMP_*/MKL_*/numactl
│   └── presets.md                      # per-cluster-type 已验证预设组合
│
├── profiling/                          # 数据采集方法
│   ├── SKILL.md
│   ├── preflight.md                    # cluster baseline
│   ├── gpu.md                          # GPU metrics
│   ├── network.md                      # IB / RCCL
│   ├── trace.md                        # timeline 分析
│   └── env_probe.md                    # env 安全探测协议（连通性 → micro-bench → 多节点）
│
├── constraints/                        # 安全约束
│   ├── SKILL.md
│   ├── oom.md                          # OOM 预估规则
│   ├── config.md                       # 配置合法性
│   ├── validation.md                   # 验证方法
│   └── env.md                          # env 不兼容矩阵（互斥 / 危险组合）
│
└── knowledge/                          # 经验沉淀（LEARN stage 写入目标）
    ├── SKILL.md                        # 检索 / 写入协议
    ├── patterns.md                     # 通用规律（"MoE > 16 节点必开 alltoall overlap"）
    ├── cases.md                        # 历史 best config 案例库（按模型 × 集群索引）
    └── anti-patterns.md                # 失败 case / 已知坑
```

**env 知识的组织原则**：
- `skills/env/*.md` 是**唯一 catalog**（每个 flag 只在这里完整定义一次）
- `skills/optimization/{bottleneck}/env.md` 只列**「该瓶颈下应优先看哪些 flag」**，引用 catalog
- 这样新增 flag 只改 catalog 一处，避免知识散落

**Skill 的作用**：Agent（Cursor / Claude / Codex 等任意工具调用 LLM）读取这些 Markdown 文件获取领域知识，而不是把知识硬编码在代码里。这样：
- 知识可以独立迭代，不需要改代码
- 不同的 Agent runtime 共用同一套知识
- 新人可以直接阅读 Skills 了解系统的调优逻辑

---

## 5. Tool 接口

Tools 是 Agent 通过 function calling 调用的 Python 函数：

| Tool | 功能 | 输入 | 输出 |
|------|------|------|------|
| `preflight.run()` | 采集集群硬件基线 | cluster_id | ClusterProfile |
| `env_probe.run()` | 探测/校验集群级 env baseline（连通性 + RCCL micro-bench） | cluster_id, candidate_envs | EnvBaseline (写入 ClusterProfile) |
| `env_probe.sweep()` | 内层 EnvSweep：固定结构，扫描 env diff | base_plan, candidate_envs, max_steps | best_env_diff, results |
| `profiler.run()` | 单机 profiling | model_spec, configs | ProfilingResult |
| `submit.run()` | 提交训练任务 | plan, cluster | job_id |
| `submit.cancel()` | 取消任务 | job_id | status |
| `observe.snapshot()` | 采集运行时数据 | job_id | Snapshot |
| `observe.compare_loss()` | CORRECTNESS 闸门：与 reference loss 对齐 | job_id, reference_curve | pass / drift, delta_pct |
| `constraint.check()` | 验证配置合法性 | plan, cluster | valid, violations |
| `constraint.check_env()` | 验证 env 组合（互斥 / 危险） | env_diff, baseline | valid, violations |
| `constraint.estimate_mem()` | 估算显存 | plan | mem_gb |
| `constraint.diagnose_failure()` | 失败归因（OOM/hang/invalid → reason） | snapshot/error | failure_reason, suggested_transition |
| `state.checkpoint()` | 持久化 TuningState（每 stage 出口调用） | tuning_state | path |
| `state.resume()` | 从 checkpoint 续起 | path | tuning_state |
| `state.trim()` | Orchestrator 每 stage 出口丢弃细节，仅保留指针类字段 | tuning_state, keep_fields | trimmed_state |
| `state.handoff()` | Orchestrator 自我接力（context 临近上限时） | session_id | handoff_path |
| `subagent.spawn()` | 派生 Stage Worker，隔离 context；返回结构化结果 | stage, input_refs, skill_scope | SubagentResult |
| `knowledge.write()` | LEARN：回写 best/失败 case | report, kind | written_paths |

Agent 根据 Skill 中的知识决定"做什么"，然后调用 Tool 执行。

---

## 6. Execution Model（核心知识）

定义在 `skills/execution-model/*.md` 中，Agent 读取这些公式进行预估。

**Step Time 分解**：

```
T_step = T_comp + T_comm + T_bubble - T_overlap

T_comp = model_flops / (num_gpus × peak_tflops × efficiency)
  - efficiency 从 ClusterProfile + profiling 数据得到

T_comm = AllReduce(grad_size/dp) + AllToAll(moe_msg) + AllGather(zero_shard)
  - bandwidth 从 ClusterProfile.rccl_baseline.<scope>.collectives.<coll>.roll_up.median_bw_gbs 查表
  - <scope> ∈ {intra_node, inter_node, world}，由并行形态决定（TP/EP→intra_node，DP/FSDP→world）

T_bubble = (pp-1) / (pp-1 + M) × T_comp   # M = num_microbatch

T_overlap = min(T_comm_overlappable, T_comp_spare)
```

**显存估算**：

```
Mem = M_param + M_grad + M_optim + M_act + M_buffer

M_param = params / (tp × pp) × bytes_per_param
M_act   = f(seq, hidden, mbs, layers/pp, recompute)
```

---

## 7. 搜索空间维护与解保证

> 本节回答："为什么这套贪心循环不会陷在局部最优、不会重复搜索、不会漏掉真正的最优解？"
> 这是 Pilot 收敛性的核心设计。Schema 落地见 §8.9 PlanGraph 与 §8.10 CandidatePool；
> 这里只讲**心智模型与机制**。

### 7.1 朴素贪心的两个坑

最直觉的做法是：Settle 维护一个 baseline（当前最优），每轮 Re-Plan 在它之上派生候选、
跑完取本轮 best 当下一轮 baseline。但裸贪心容易掉进两个坑：

| 坑 | 表现 | 后果 |
|----|------|------|
| **过早收敛** | 每轮选 best 就丢掉次优；但次优可能开了另一种瓶颈的门（例如 P_b 当前略差，但开启了 PIPELINE_BOUND，下一轮潜力更大） | 永远停在某个局部最优 |
| **重复搜索** | history 只去重已跑过的 plan id，但**派生关系丢失**；无法回答"这个配置是从哪个 baseline 演化来的、为什么没继续" | 反复在已耗尽的邻域试探，浪费预算 |

Pilot 的应对是：**把"解空间"显式化为一棵带评分的搜索树（PlanGraph），把"候选"显式化为带优先级的池（CandidatePool），并在贪心之上叠加 3 种探索机制。**

### 7.2 心智模型：搜索 = 维护一棵带评分的树

Re-Plan / Settle / Execute 三者协同维护的不是一条链，而是一棵树：

```
                       root_plan (BASELINE)
                       tps=12000
                       /        |          \
                   P1            P2           P3
                tps=14200     tps=15800    tps=13100   ← Round 1 候选
                              ★ champion       │
                              /     \          ↓
                            P4       P5      shelved
                          tps=17600  OOM    （留在候选池，可复活）
                          ★ champion │
                          /  \       ↓
                        ...  ...   dead       ← Round 2
```

节点状态四分类：

| 状态 | 语义 | 是否可派生新候选 |
|------|------|-----------------|
| **champion** | 当前 baseline，沿粗箭头延伸 | 是（默认派生源） |
| **shelved** | 本轮没赢但活着；可能后续复活做 backtrack | 是（探索时） |
| **dead** | OOM / invalid / 数值失败，永久剪枝 | 否 |
| **running** | 正在 Execute 中 | — |

派生不一定从 champion 走——某些条件下 Re-Plan **从 shelved 派生**（见 §7.5 Backtrack）。
这一点是跳出局部最优的关键。

### 7.3 PlanGraph：解空间维护

PlanGraph 是 State 层的持久化结构，每个 stage 出口写入。只列与"解保证"相关的关键字段
（完整 schema 见 §8.9）：

| 字段 | 作用 |
|------|------|
| `champion` | 当前 baseline pointer |
| `champion_history` | 历任冠军（连续多次冠军 → 高稳定性奖励） |
| `nodes[*].parent` | 派生关系；让审计能回放"从哪来、为什么走到这" |
| `nodes[*].status` | completed / shelved / dead / running |
| `nodes[*].derived_axis` | 该节点相对父节点动了哪根轴；novelty / 邻域剪枝都看它 |
| `frontier` | 当前可派生节点集合（champion + shelved） |
| `exhausted_neighborhoods` | 已探索半径；新候选若落在这里直接 reject，防重复搜索 |
| `metadata.rounds_since_promotion` | stagnation 检测计数 |
| `metadata.rounds_since_explore` | 距上次强制探索轮的距离 |
| `metadata.dead_rate_in_subtree` | 各节点子树失败率，> 50% 触发 backtrack |

### 7.4 候选生成：带优先级的池，不是固定的 K 个

每轮 Re-Plan 输出的不是"要跑的 K 个 plan"，而是**带优先级的候选池**，由 Strategy
Select（见 §8.10 selection）做最终裁决：

```
priority(c) = expected_gain(c) × confidence(c) / est_cost(c)
            × novelty_bonus(c)              # 探索未走过的轴 +20%
            × parent_stability_bonus(c)     # 从多次冠军派生 +10%
```

候选池里同时混合两类来源：

- **exploit 候选**：从 `champion` 派生，沿当前最优继续微调。
- **explore 候选**：从 `shelved` 派生，复活之前没赢但 viable 的分支。

被拒的候选（命中 `exhausted_neighborhoods` / 触发 `constraint.estimate_mem` /
违反 `constraint.check`）也写进 `candidate_pool.rejected[]`，理由可追溯。

### 7.5 Settle：贪心 + 软回滚

Settle 不是简单"选 best 当 baseline"。它在升降 champion 之外，还要决定 shelved 是否
复活、是否进入 stagnation/explore 模式：

```python
def settle(round_results, plan_graph):
    new_best = max(round_results, key=score)
    cur     = plan_graph.champion

    # 规则 1：显著提升 → 升新 champion
    if new_best.tps > cur.tps * (1 + ε_promote):     # ε_promote ≈ 2%
        plan_graph.champion = new_best.id
        plan_graph.set_status(cur, 'shelved')        # 旧 champion 不丢，进 shelved
    # 规则 2：微弱提升 → 保留旧 champion，best 进 shelved 待后续复活
    elif new_best.tps > cur.tps:
        plan_graph.set_status(new_best.id, 'shelved')
    # 规则 3：全部退化 → 触发 backtrack（见 §7.6）
    # 规则 4：连续 N 轮 gain < ε_stop → 进入 stagnation 模式
    if recent_gain_pct(plan_graph, n=2) < ε_stop:
        switch_to_explore_mode()                     # 下一轮 Re-Plan 从 shelved 派生
```

ε_promote / ε_stop 默认值在 `skills/pilot/workflow/settle.md` 给出，具体数值随
TargetVector 的紧迫度调整（budget 越紧，ε_promote 越大、越保守）。

### 7.6 跳出局部最优的 3 种探索机制

| 机制 | 触发条件 | 动作 | 防止的失败模式 |
|------|----------|------|----------------|
| **Backtrack（回退到次优分支）** | 当前 champion 子树连续 2 轮 dead 率 > 50%；或 stagnation 持续 2 轮 | 从 frontier 里挑 priority 第二高的节点作为新 champion，重新派生 | 卡在死胡同里 |
| **Diversification Bonus** | Re-Plan 每次评分都生效 | 给"覆盖未探索轴"的候选加 priority 权重，避免一直在一根轴上 nudge | 一根轴反复微调、忽略其他轴 |
| **Periodic Exploration Round** | 每 K 轮强制一次（默认 K=3） | 候选池只从 `shelved` + 未探索轴生成，不从 champion 派生；这一轮可能不涨甚至跌 | 长期局部最优 |

3 种机制不互斥。Backtrack 应对"突然跑死"，Diversification 应对"温水青蛙式停滞"，
Periodic Exploration 是兜底。

### 7.7 收敛性保证

在上面的搜索结构之上，再叠加 4 条静态保证：

| 机制 | 作用 |
|------|------|
| **Execution Model 筛选** | Plan 进入 Execute 前已经过模型预估，质量高（confidence < 阈值的候选直接被剪） |
| **PlanGraph + champion_history** | champion 单调或经审计的回退；不会退化无据 |
| **`exhausted_neighborhoods`** | Re-Plan 排除已尝试邻域，搜索空间逐轮缩小 |
| **三重终止条件** | TargetVector 达成 / gain < ε_stop 连续两轮 / 触达 `max_rounds` 或 `budget.total_gpu_h` |

成本上界（典型 Dense / MoE bring-up，single-node 量级；多节点按比例放大）：

| 阶段 | 成本 |
|------|------|
| Preflight | ~30 min（首次；同集群跨任务复用） |
| Projection | ~1 GPU·h（含 single-node profiling） |
| Tuning Loop | ~3-5 GPU·h（含 SMOKE / CORRECTNESS-LITE / 可选 EnvSweep） |
| **总计** | ~5-7 GPU·h |

> **设计交叉引用**：
> - PlanGraph 落地 schema → §8.9
> - CandidatePool 落地 schema → §8.10
> - 升降规则与 stagnation 阈值 → `skills/pilot/workflow/settle.md`
> - 候选生成 7 步 → `skills/pilot/workflow/replan.md`
> - 探索/利用切换策略 → `skills/pilot/workflow/execution_strategy.md`
> - 轴的探索半径与剪枝定义 → `skills/pilot/workflow/axis_taxonomy.md`

---

## 8. 数据结构（Schema）

Agent 与 Tool 之间通过结构化数据交换。下面是几个核心 schema。

### 8.1 ClusterProfile（Preflight 输出）

Schema 2.0：`rccl_baseline` 拆成三个 scope —— `intra_node`（每节点一条曲线，节点间并行测量）、`inter_node`（1 GPU/节点的单环，隔离 IB/RoCE）、`world`（完整 N×gpus_per_node 单环，对应实际训练拓扑）。单节点模式只填 `intra_node`，`inter_node`/`world` 留 `null`。权威 schema：`pilot/schemas/cluster_profile.schema.json`。

```yaml
schema_version: "2.0"
cluster_id: mi300x-16node
version: mi300x_8gpu-16node-v1
cluster_class: mi300x_8gpu
collected_at: 2026-04-15T10:00:00Z
nodes_total: 16
nodes_healthy: 16
gpus_per_node: 8

compute:
  peak_tflops_bf16: 1300         # 实测 GEMM peak（节点间中位数）
  peak_tflops_fp8: 2600
  hbm_bandwidth_gbs: 5300
  hbm_capacity_gb: 192
  per_node_variance_pct: 1.4

interconnect:
  intra_node:
    type: xgmi
    bandwidth_gbs: 800            # 回填自 rccl_baseline.intra_node 最大 size 的 AR
  inter_node:
    type: ib
    bandwidth_gbs: 400            # 单卡有效带宽
    uniformity: uniform

rccl_baseline:
  intra_node:                     # 每节点一条曲线（每个节点跑独立的本地 8 卡 PG，并行）
    world_size: 8
    nnodes_measured: 16
    collectives:
      allreduce:
        sizes_mb: [1, 16, 64, 256]
        per_node_bw_gbs:
          smc01: [25.1, 182.2, 306.3, 377.2]
          smc02: [24.9, 181.0, 305.5, 374.0]
          # ... 其余 14 个节点
          smc16: [25.0, 181.5, 305.8, 376.1]
        per_node_latency_us:
          smc01: [73.1, 161.1, 383.4, 1245.3]
          # ...
        roll_up:
          median_bw_gbs: [25.0, 181.5, 305.8, 376.1]
          min_bw_gbs:    [22.3, 175.0, 290.0, 281.5]   # smc04 慢
          max_bw_gbs:    [25.2, 182.2, 306.3, 377.2]
          stddev_pct:    [3.2,  1.1,   1.5,   9.8]
          slow_nodes_at_max_size: [smc04]
      allgather:      { ... 同样形状 ... }
      reduce_scatter: { ... }
      broadcast:      { ... }
      alltoall:       { ... }

  inter_node:                     # 1 GPU/节点 × 16 节点单环（隔离 IB）
    world_size: 16
    nnodes: 16
    nproc_per_node: 1
    collectives:
      allreduce:
        sizes_mb:   [1, 16, 64, 256]
        bw_gbs:     [3.2, 18.5, 23.1, 25.4]
        latency_us: [580, 1580, 5060, 18400]
      # allgather / reduce_scatter / broadcast 类似；alltoall 在此 scope 通常省略

  world:                          # 完整 128 卡环（实际训练拓扑）
    world_size: 128
    nnodes: 16
    nproc_per_node: 8
    collectives:
      allreduce:
        sizes_mb:   [1, 16, 64, 256]
        bw_gbs:     [1.8, 11.4, 19.7, 22.3]
        latency_us: [1020, 2580, 5940, 20800]
      # allgather / reduce_scatter / broadcast / alltoall 类似

env_baseline:                      # 集群级 env 黄金默认（一次探测 / 多任务复用）
  version: mi300x_8gpu-16node-v1
  status: validated                # validated / tentative
  rccl:
    NCCL_IB_HCA: "mlx5_0:1,mlx5_1:1,mlx5_2:1,mlx5_3:1"
    NCCL_NET_GDR_LEVEL: 4
    NCCL_IB_GID_INDEX: 3
    NCCL_SOCKET_IFNAME: bond0
  hsa:
    HSA_FORCE_FINE_GRAIN_PCIE: 1
    GPU_MAX_HW_QUEUES: 2
  alloc:
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True"
  threading:
    OMP_NUM_THREADS: 8
```

**消费方读表指引**（PROJECTION / SCHEDULER 用）：

- `intra_node` 一律读 `roll_up.median_bw_gbs`（典型情况）或 `roll_up.min_bw_gbs`（最坏情况下界）。单节点退化下 `min == median == max`、`stddev_pct == 0`。
- `inter_node` 与 `world` 直接读 `bw_gbs`（单环，无需聚合）。
- `slow_nodes_at_max_size` 是 blacklist 提案的标准来源；PREFLIGHT 只标记，不自动拉黑。

### 8.2 Plan（待执行的配置）

```yaml
plan_id: r3_p2
parent_baseline: r2_p1            # 上一轮的最优 plan
parallelism:
  tp: 4
  pp: 2
  dp: 16
  ep: 8
  vpp: 2
runtime:
  mbs: 2
  gbs: 1024
  recompute: selective
comm:
  bucket_size_mb: 64
  overlap: true
env:                               # 仅记录相对 env_baseline 的差异
  baseline_ref: mi300x-16node-v3
  diff:
    NCCL_MIN_NCHANNELS: 16
    NCCL_BUFFSIZE: 16777216
    PYTORCH_HIP_ALLOC_CONF: "expandable_segments:True,max_split_size_mb:512"
    RCCL_MSCCL_ENABLE: 1
predicted:                         # Execution Model 预估值
  tps: 18500
  mem_peak_gb: 165
  comm_ratio: 0.22
generated_by:
  bottleneck: PIPELINE_BOUND
  strategy: skills/optimization/pipeline/vpp.md
```

### 8.3 Snapshot（Observe 输出）

```yaml
run_id: job_8842
plan_id: r3_p2
collected_at: 2026-04-15T11:23:00Z
metrics:
  tps: 17800
  step_time_ms: 412
  comm_ratio: 0.18
  bubble_ratio: 0.09
  overlap_ratio: 0.61
  mem_peak_gb: 158
  gpu_util_avg: 0.74
status: completed                  # / early_stopped / oom / failed
warnings: []
```

### 8.4 DiagnosisReport（Diagnose 输出）

```yaml
snapshot_id: job_8842
bottleneck: COMPUTE_BOUND          # COMM / PIPELINE / MEMORY / COMPUTE
confidence: 0.85
evidence:
  - "comm_ratio=0.18 < threshold 0.25"
  - "bubble_ratio=0.09 < threshold 0.15"
  - "gpu_util=0.74 vs baseline peak 0.92 → headroom"
recommended_skills:
  - skills/optimization/compute/mbs.md
  - skills/optimization/compute/parallel.md
env_suspect:                       # 若有 → 触发 EnvSweep（先于结构调整）
  - flag: NCCL_BUFFSIZE
    reason: "comm_ratio=0.32 但 msg_size_p95=4MB，buffer 偏小"
    hint: skills/env/rccl.md#buffsize
  - flag: PYTORCH_HIP_ALLOC_CONF
    reason: "mem_reserved/mem_alloc=1.45 → 碎片偏高"
    hint: skills/env/alloc.md#expandable-segments
candidate_axes:                    # Re-Plan 用：每个轴的类型决定如何搜
  - {axis: vpp,                  type: structural,     candidates: [1, 2, 4]}
  - {axis: mbs,                  type: structural,     candidates: [1, 2]}
  - {axis: NCCL_BUFFSIZE,        type: strongly_local, candidates: [8M, 16M, 32M]}
  - {axis: PYTORCH_HIP_ALLOC_CONF, type: weakly_local, candidates: [seg, no-seg]}
  - {axis: NCCL_IB_HCA,          type: cluster_shared, candidates: [baseline]}
suggested_strategy: Per-Plan       # Champion-Challenger / Per-Plan / Successive_Halving
  rationale: "strongly_local axis NCCL_BUFFSIZE present, model confidence>0.7"
```

### 8.5 EnvSweepResult（内层循环输出）

```yaml
sweep_id: r3_envsweep_1
parent_plan: r3_p2                 # 锁定的结构 baseline
trigger: COMM_BOUND
candidates:                        # 本轮试过的 env 组合
  - env_diff: {NCCL_BUFFSIZE: 8388608}
    tps: 17900
    delta_pct: +0.6
  - env_diff: {NCCL_BUFFSIZE: 16777216, NCCL_MIN_NCHANNELS: 16}
    tps: 18650
    delta_pct: +4.8                # ← best
  - env_diff: {NCCL_BUFFSIZE: 33554432}
    tps: 17600
    delta_pct: -1.1
selected_diff:                     # 合并进 baseline.env.diff
  NCCL_BUFFSIZE: 16777216
  NCCL_MIN_NCHANNELS: 16
cost_gpu_h: 0.3
```

### 8.6 TargetVector（用户输入 / Settle 判定依据）

把"达标"从单 TPS 升级为多目标。Settle 的停止条件 = 全部 `constraints` 满足且 `primary` 不再显著提升 / 触达 `budget`。

```yaml
target:
  primary: tps                     # 主目标（也支持 per_token_cost_usd / time_to_loss）
  constraints:                     # 必须满足的硬约束
    - mem_peak_gb <= 180
    - per_token_cost_usd <= 1.2e-7
    - correctness.loss_delta_pct <= 1.0
  budget:
    total_gpu_h: 10
    max_rounds: 5
    wallclock_h: 24
  preferences:                     # 软偏好（同分时打破平局）
    prefer_lower_pp: true          # 减少 bubble 风险
    prefer_known_env_presets: true # 优先用已沉淀的 presets
```

### 8.7 TuningState（每个 stage 出口持久化）

让 Agent 中断后可从任意 stage 续起；同时是审计 / 回放的基本单元。

```yaml
session_id: pilot_run_20260418_a3
current_stage: OPTIMIZE_LOOP.OBSERVE   # 状态机定位
stage_history:
  - {stage: PREFLIGHT,   exit: success, at: ...}
  - {stage: PROJECTION,  exit: success, at: ...}
  - {stage: SMOKE,       exit: success, at: ...}
  - {stage: BASELINE,    exit: success, at: ...}
  - {stage: CORRECTNESS, exit: success, at: ...}
  - {stage: OPTIMIZE_LOOP.SETTLE, exit: continue, round: 2, at: ...}

cluster_profile_ref: mi300x-16node-v3   # 指向 ClusterProfile 版本
target: <TargetVector>
baseline: <Plan>                         # 当前最优
history:                                 # Re-Plan 去重用
  - {plan_id: r0_p0, tps: 12000, status: completed}
  - {plan_id: r1_p1, tps: 14200, status: completed}
  - {plan_id: r2_p3, tps: 0,     status: oom}

budget_used:
  gpu_h: 4.2
  rounds: 2
  wallclock_h: 6.1

reentry_log:                             # 跳回记录（审计用）
  - {from: DIAGNOSE, to: PREFLIGHT, reason: cluster_profile.age>7d, at: ...}
```

### 8.8 FailureReport（Guardrail 触发时的归因输出）

由 `constraint.diagnose_failure()` 生成，决定 Guardrail 走哪条恢复路径（见 §12）。

```yaml
run_id: job_8842
failure_kind: OOM                  # OOM / HANG / INVALID_CONFIG / NUMERICAL / CLUSTER
root_cause: "act_mem_underestimated"
evidence:
  - "predicted mem_peak=170GB, actual=192GB at step 47"
  - "no recompute on layers 12-23"
suggested_transition:              # 给状态机的回路提示
  to: REPLAN
  hint: "mark plan dead, try recompute=full on rear half"
counts_against_budget: false       # OOM/invalid 不消耗 round 配额
```

### 8.9 PlanGraph（解空间维护）

把 `history` 从平铺 list 升级为带派生关系的树。Re-Plan 选派生源、Settle 决定 champion 升降、防局部最优都依赖这个结构。

```yaml
plan_graph:
  champion: r2_p4                  # 当前 baseline 的 plan_id
  champion_history:                # 历任冠军（连续多次冠军 → 高稳定性）
    - {round: 0, id: r0_p0}
    - {round: 1, id: r1_p2}
    - {round: 2, id: r2_p4}

  nodes:                           # 每个尝试过的 plan 都是一个节点
    r0_p0:
      parent: null
      status: completed            # completed / shelved / dead / running
      tps: 12000
      bottleneck: COMM_BOUND
      ★champion_at: [0]
      derived_axis: null           # root
    r1_p2:
      parent: r0_p0
      status: completed
      tps: 15800
      bottleneck: PIPELINE_BOUND
      ★champion_at: [1]
      derived_axis: {NCCL_overlap: enable}
    r1_p3:
      parent: r0_p0
      status: shelved              # 没赢但活着，下一轮可被复活
      tps: 13100
      reason: "ep 减小损害容量，但仍 viable，留作 backtrack 候选"
      derived_axis: {ep: 8→4}
    r2_p4:
      parent: r1_p2
      status: completed
      tps: 17600
      ★champion_at: [2]
      derived_axis: {vpp: 1→2}
    r2_p5:
      parent: r1_p2
      status: dead                 # 永久剪枝，不再考虑
      reason: OOM
      predicted_mem: 165
      actual_mem: 192
      derived_axis: {mbs: 1→2}

  frontier:                        # 当前可派生的活跃节点集合（champion + shelved）
    - r2_p4    # champion
    - r1_p3    # shelved
    - r1_p1    # shelved

  exhausted_neighborhoods:         # 已探索半径，新候选若落在这里直接 reject
    - {around: r0_p0, axis: bucket_size_mb, tried: [16, 32, 64, 128]}
    - {around: r1_p2, axis: vpp,            tried: [1, 2]}
    - {around: r2_p4, axis: NCCL_BUFFSIZE,  tried: [8M, 16M]}

  metadata:
    rounds_since_promotion: 0      # 用于 stagnation 检测
    rounds_since_explore: 2        # 距上次 explore round 的轮数（>=K=3 触发）
    dead_rate_in_subtree:          # 各节点子树的失败率（>50% 触发 backtrack）
      r2_p4: 0.50                  # P4 子树 2 个孩子里 1 dead
```

### 8.10 CandidatePool（Re-Plan 输出）

Re-Plan 不再直接输出"要跑的 K 个 plan"，而是输出一个**带优先级的候选池**，让 Strategy Select 做最终裁决。

```yaml
candidate_pool:
  generated_at: round_3
  derived_from:                    # 派生源（默认 champion；stagnation 时从 shelved）
    primary: r2_p4
    secondary: [r1_p3]             # 若启用 explore round
  policy: explore_exploit          # exploit / explore / explore_exploit

  candidates:
    # exploit 候选（从 champion 派生）
    - id: r3_p1
      parent: r2_p4
      axis_change: {mbs: 1→2}
      predicted_tps: 18800
      predicted_gain_pct: 6.8
      confidence: 0.82             # Execution Model 对该预测的置信度
      est_cost_gpu_h: 0.4
      novelty_bonus: 1.0           # 该轴未在 r2_p4 周围探索过
      stability_bonus: 1.10        # 派生自连续 2 轮冠军
      priority: 1.55               # = gain × confidence × bonuses / cost

    - id: r3_p2
      parent: r2_p4
      axis_change: {NCCL_BUFFSIZE: 8M→16M}
      predicted_tps: 18200
      confidence: 0.65
      est_cost_gpu_h: 0.3
      priority: 0.92

    # explore 候选（从 shelved 派生，避免局部最优）
    - id: r3_p3
      parent: r1_p3
      axis_change: {ep: 4→8, with: alltoall_overlap}
      predicted_tps: 17900
      confidence: 0.55
      est_cost_gpu_h: 0.5
      novelty_bonus: 1.20          # 复活 shelved 的探索奖励
      priority: 0.78
      tag: explore

  selection:                       # Strategy Select 的最终输出
    strategy: Champion-Challenger  # 来自 axis_taxonomy 决策
    pick_top_k: 3
    selected: [r3_p1, r3_p2, r3_p3]
    rejected:                      # 被拒的候选也记录，供审计
      - {id: r3_p_x, axis_change: {bucket_size_mb: 32→64},
         reason: "exhausted_neighborhoods 命中 (around r0_p0)"}
      - {id: r3_p_y, axis_change: {pp: 2→4},
         reason: "constraint.estimate_mem 超阈值 (210GB > 192GB cap)"}

  priority_formula: |
    priority(c) = expected_gain(c) × confidence(c) / est_cost(c)
                × novelty_bonus(c)              # 探索未走过的轴 +20%
                × parent_stability_bonus(c)     # 从多次冠军派生 +10%
```

### 8.11 SubagentResult（Stage Worker → Orchestrator）

Stage Worker 执行完毕后返回给 Orchestrator 的**唯一结构化载荷**。Orchestrator 只解析这个对象来更新指针，不吸收 Worker 的任何中间 trace。原始细节（Snapshot / DiagnosisReport / CandidatePool / EnvSweepResult）都已由 Worker 写入 State Layer，这里只给引用。

```yaml
subagent_result:
  stage: RE_PLAN                   # DIAGNOSE / RE_PLAN / ENV_SWEEP / CORRECTNESS_LITE / ...
  worker_id: sw_r3_replan_20260421_112345
  status: success                  # success / failed / escalate

  # 结构化产物的 State Layer 引用（Orchestrator 不读内容）
  artifacts:
    - kind: CandidatePool
      ref: state/round_3/candidate_pool.yaml
      size_bytes: 4280

  # 给 Orchestrator 的一行摘要（必须 < 200 tokens）
  summary:
    headline: "COMPUTE_BOUND, 3 exploit + 1 explore candidates, top priority=1.55"
    key_metrics:
      selected_count: 3
      rejected_count: 2
      top_priority: 1.55
      est_cost_gpu_h: 1.2

  # 建议状态机转移（Orchestrator 做最终裁决）
  suggested_transition:
    to: EXECUTE
    reason: "candidate_pool.selected non-empty, budget ok"

  # 资源占用（用于评估与 budget 追踪）
  cost:
    wallclock_s: 12.4
    tokens_used: 18420           # Worker 单次消耗的 LLM tokens
    tool_calls: 7

  # 若 status=failed 才出现
  failure:
    kind: SKILL_MISSING          # / TOOL_ERROR / CONSTRAINT_VIOLATION / ...
    message: "optimization/moe/dispatch.md not found"
    escalate_to_orchestrator: true
```

**Orchestrator 侧的处理协议**（落在 `skills/workflow/orchestration.md`）：

1. 只读 `summary` + `suggested_transition` + `status`
2. 根据 `suggested_transition` 参照 `state_machine.md` 决策
3. 把 `summary.headline` 追加到 `TuningState.stage_history`（单行日志）
4. 丢弃 `SubagentResult` 对象本身（不保留完整 YAML 进入 context）
5. 立即 `state.checkpoint()` + `state.trim()`

---

## 9. 完整迭代示例

一次 Tuning Loop 的真实数据流（以 MoE 16 节点为例）：

```
Round 0 (Baseline)
  Plan:     {tp:2, pp:4, ep:8, mbs:1, recompute:full}
  Snapshot: tps=12000, comm_ratio=0.38, bubble=0.12, mem=140GB
  Diagnose: COMM_BOUND (alltoall 占 28%)
  Re-Plan:  读 skills/optimization/comm/ → 生成 3 候选
            P1: bucket_size 16→64 MB
            P2: 启用 alltoall overlap
            P3: ep 8→4（减少跨节点通信）

Round 1 (外层)
  Execute:  3 个 plan 并行跑（每个 50 step early-stop）
  Results:
    P1: tps=14200 (+18%)
    P2: tps=15800 (+32%)  ← best
    P3: tps=13100 (+9%, ep 减小导致 expert 容量下降)
  Settle:   选 P2 作为新 baseline

Round 1' (内层 EnvSweep，因 Diagnose 输出 env_suspect=NCCL_BUFFSIZE)
  Sweep:    锁定 P2 结构，扫 3 个候选（30 step）
            E1: NCCL_BUFFSIZE=8M             → tps=15900 (+0.6%)
            E2: NCCL_BUFFSIZE=16M+MIN_NCH=16 → tps=16550 (+4.7%)  ← best
            E3: NCCL_BUFFSIZE=32M            → tps=15600 (-1.3%)
  Merge:    P2.env.diff += {NCCL_BUFFSIZE:16M, NCCL_MIN_NCHANNELS:16}
            new baseline tps=16550
  Cost:     0.3 GPU·h
  Diagnose: PIPELINE_BOUND (bubble 升至 0.18)
  Re-Plan:  读 skills/optimization/pipeline/ → 生成 2 候选
            P4: vpp 1→2
            P5: mbs 1→2

Round 2
  Execute:  P4, P5
  Results:
    P4: tps=17600 (+11%)  ← best
    P5: OOM → mark dead (predicted_mem 误差超 15%)
  Settle:   升 P4 为新 champion；P2(旧 champion) 进 shelved
  PlanGraph:
    champion: P4
    shelved:  [P1, P3, P2]
    dead:     [P5]
    exhausted_neighborhoods: {around=P2, axis=NCCL_BUFFSIZE, tried=[8M,16M,32M]}
  Diagnose: COMPUTE_BOUND

Round 3 (CandidatePool 演示)
  Re-Plan:  从 P4 派生 + 1 个 explore 候选（rounds_since_explore=2）
    Pool:
      - P6: P4 → mbs:2→3            priority=1.42
      - P7: P4 → tp:2→4             priority=0.95
      - P8: P3 复活 → ep:4→8 + AT-overlap  priority=0.78 (tag=explore)
    Strategy Select: Champion-Challenger, top-3 → [P6, P7, P8]
    Rejected: P_x (vpp:2→4, exhausted around P4)
  Execute (50 step):
    P6: tps=18100 (+2.8%)  ← best
    P7: OOM (tp 大幅增加突破 act mem) → dead
    P8: tps=15200 → 不及 champion，回 shelved
  Settle:   微弱提升 (+2.8% < ε_promote=2%×1.5)，champion 仍是 P4，P6 进 shelved
  rounds_since_promotion: 1

Round 4
  Re-Plan:  rounds_since_promotion=1，仍 exploit；从 P4 派生 P9 (recompute=selective)
  Execute:  P9: tps=18400 (+1.7%)
  Settle:   gain<2%，连续 2 轮无显著提升 → 进 stagnation
  Stop check: frontier 中无 priority>1.0 候选 → STOP

Final: champion=P9, tps=18400 (1.53× over baseline)
       PlanGraph 完整回放可还原所有派生路径
耗时:  ~4.5 GPU·h (Tuning Loop 部分)
```

> **PlanGraph 的价值**：搜索是显式的树结构而非黑箱，每一步都可审计、可回放、可在不同任务间迁移（同模型同集群下次直接载入 `exhausted_neighborhoods`，避免重复探索）。

---

## 10. 评估指标

如何判断 Pilot 系统本身好不好用：

| 维度 | 指标 | 目标 |
|------|------|------|
| **效果** | 最终 TPS / baseline TPS | ≥ 1.3× |
| **效果** | 与人工 best-known 配置差距 | ≥ 90% |
| **效率** | 总 GPU·h 成本 | ≤ 10 GPU·h |
| **效率** | 收敛轮数 | ≤ 5 轮 |
| **可靠性** | 成功率（无 OOM / 任务挂掉） | ≥ 95% |
| **可靠性** | Cost Model 预估误差 | ≤ 20% |
| **可解释性** | 每步决策可追溯到 Skill | 100% |
| **可扩展性** | Orchestrator 稳态 context / round | ≤ 2K tokens |
| **可扩展性** | Stage Worker 单次峰值 context | ≤ 30K tokens |
| **可扩展性** | 不触发 handoff 的最大 round 数 | ≥ 20 |
| **可扩展性** | SubagentResult 对 Orchestrator 的 context 注入 | ≤ 200 tokens / stage |

**评估方法**：
1. **回归测试集**：维护 5-10 个典型场景（Dense 7B/70B、MoE 8x7B 等），每次系统升级后自动跑一遍
2. **对照实验**：同一模型 + 集群，分别由 Pilot 和资深工程师调优，对比结果
3. **消融实验**：关掉 Execution Model / 关掉 Cost Model 筛选，看 Tuning Loop 退化多少

---

## 11. 与现有系统的集成

Pilot 不重新发明轮子，而是组合现有能力：

| 现有系统 | 集成方式 |
|---------|---------|
| **Primus** | `submit.run()` 调 Primus CLI 提交任务；Preflight 复用 Primus 的硬件检测 |
| **Megatron / TorchTitan** | 通过 Tuning IR 生成各 backend 的 config，Agent 不需要感知具体框架 |
| **WandB / Prometheus** | `observe.snapshot()` 从 WandB API 拉取 metrics |
| **rocprof / RCCL profiler** | `profiler.run()` 包装 rocprof 命令，输出结构化 trace |
| **Slurm** | `submit.run()` 内部生成 sbatch 脚本，复用现有调度 |

这种集成方式的好处：**Pilot 死掉后，所有底层工具仍然可用**；现有工程师的 muscle memory 不被破坏。

---

## 12. Guardrails

### 12.1 预防性约束

| 机制 | 位置 | 作用 |
|------|------|------|
| OOM 预估 | `constraint.estimate_mem()` | 自动过滤高风险配置 |
| 数值闸门 | `correctness.md` + `observe.compare_loss()` | BASELINE 后 / 每 N 轮抽查 loss curve 与 reference 对齐 |
| Smoke 起前自检 | `smoke.md` + `submit.run(tiny)` | tiny scale × 100 step 验证可起，失败回 PROJECTION |
| early stop | `observe.snapshot()` | OOM / 吞吐退化 → 立即终止 |
| history 去重 | Settle 逻辑 | 不重复尝试已失败配置 |
| max_rounds / budget | Settle + TargetVector.budget | 总成本硬约束（GPU·h / wallclock） |
| env 连通性 fail-fast | `env_probe.run()` | 错误 NCCL_IB_HCA 等 30s 内炸掉，不进 baseline |
| env 不兼容矩阵 | `constraint.check_env()` | 拒绝危险/互斥组合（如 MSCCL × 某些 GDR 模式） |
| EnvSweep 单次 cap | inner loop | 每次 ≤ 5 flag、≤ 8 组合、≤ 50 step |
| env baseline 版本化 | `ClusterProfile.env_baseline.version` | 集群升级 / 驱动变更触发重新探测 |
| env diff-only 记录 | `Plan.env.diff` | 审计、复现、回滚都基于 diff |
| 审计日志 + 持久化 | `state.checkpoint()` 每 stage 出口 | 可回放、可中断续起 |
| **Context hygiene** | `state.trim()` 每 stage 退出 | Orchestrator context 稳态；细节只留在 State Layer |
| **Subagent 隔离** | `subagent.spawn()` + `orchestration.md` | Stage 级 context 不污染 Orchestrator |
| **SubagentResult 强约束** | `orchestration.md` 校验 | 返回对象 `summary` 段 ≤ 200 tokens，超限拒收 |
| **Context overflow 回路** | Orchestrator 自检 `ctx_tokens > 0.5×window` | 自动 `state.handoff()`，不丢进度 |
| **Worker budget cap** | `subagent.spawn(max_tokens=30K)` | Worker 超预算强制 early-return，标 escalate |

### 12.2 失败回路（状态机的 `on_fail` / `reentry_when` 实现）

Guardrail 触发后由 `constraint.diagnose_failure()` 输出 `FailureReport`（§8.8），驱动状态机走以下转移之一。**`counts_against_budget=false` 表示不消耗 round 配额**。

| 失败类别 | 归因 | 转移目标 | 说明 |
|---------|------|---------|------|
| `OOM` | 显存超出预估 | → `REPLAN` | 标记 plan dead，强制 recompute / 减 mbs；不计配额 |
| `HANG` (NCCL/IB) | 通信卡住 > timeout | → `PREFLIGHT` (env_probe) | 怀疑 env baseline 失效，重新探测；不计配额 |
| `INVALID_CONFIG` | constraint.check 失败 | → `REPLAN` | 仅丢弃该 plan，不进 Execute；不计配额 |
| `NUMERICAL` | loss drift / NaN | → `ABORT` + escalate | 数值正确性破坏，停下问人 |
| `CLUSTER` | 节点掉线 / 驱动错误 | → `PREFLIGHT` | ClusterProfile 标记为 stale，重探 |
| `BUDGET_EXCEEDED` | gpu_h / wallclock 超限 | → `REPORT` | 提前结束，输出当前最优 |
| `STRUCTURAL_INVALIDATION` | 模型/数据规格变 | → `PROJECTION` | 重新建模，history 失效 |
| **`CONTEXT_OVERFLOW`** | Orchestrator context > 0.5 × window | → `HANDOFF` | `state.handoff()` 写入接力点，fresh Orchestrator `state.resume()` 继续；不计配额 |
| **`SUBAGENT_FAILED`** | Worker status=failed / 超 budget | → `REPLAN` or `ABORT` | 看 `SubagentResult.failure.kind`；可恢复的回 REPLAN，否则 escalate |
| `UNKNOWN` | 兜底 | → `ABORT` + escalate | 不要乱猜，交给人判断 |

---

## 13. Context Management & Multi-Agent Orchestration

> 回答一个具体问题：为什么让单一 Agent 跑完整个 Tuning Loop 会在 5 轮以后开始"失忆"或"重复试错"？根因不是 LLM 窗口不够大，而是"状态机推进"（小而长）与"stage 推理"（大而短）挤在同一个会话，context 按 round 线性膨胀到注意力稀释区。本章给出解决协议。
>
> **本章的归属分工**（见 Scope & Positioning）：
>
> | 策略 | Pilot 主责 | Agent 框架主责 |
> |------|------------|----------------|
> | A · State-first protocol | ✓（`skills/workflow/orchestration.md` 规则 + schema 大小约束） | 负责执行 |
> | B · Subagent isolation   | 只给边界表（§13.2 表格） | ✓（Task tool / subagent 派生机制） |
> | C · Session handoff      | 只给 handoff 文件格式（§8.7） | ✓（进程管理 / 会话重启） |
>
> 本章描述的"Orchestrator / Stage Worker 行为"全部是**角色规范**，落在 `prompts/` 与 `skills/workflow/orchestration.md`，由框架的原生会话/subagent 机制兑现。

### 13.1 问题：为什么单 Agent 会在长循环中失效

一个典型 round 的 context 增量（单 Agent 跑完整 Tuning Loop 的估算）：

| 来源 | 增量（tokens） | 说明 |
|------|---------------|------|
| Skill 回忆（optimization/comm/* + env/*） | ~3–5K | 每轮按需引用，即使系统里已加载 |
| Snapshot YAML（§8.3） | ~0.5K | 每个 plan 一个 |
| DiagnosisReport（§8.4） | ~1K | 每轮一个 |
| CandidatePool（§8.10） | ~2K | 每轮一个，含 rejected 记录 |
| EnvSweepResult（§8.5，可选） | ~0.8K | 触发才有 |
| Agent 自由推理 trace | ~3–5K | 越复杂越多 |
| **单轮小计** | **10–15K** | — |

按 5 轮 = 50–75K，10 轮 = 100–150K。在 200K context 上看似还有余量，但：
- LLM 注意力质量在 60% 占用后明显下降
- 老 round 的 Snapshot 与 CandidatePool 对当下决策几乎无用，却持续占位
- 一次 re-read Skill 会把整份 md 贴回 context，再次放大

**结论**：这是架构问题，不是窗口问题。扩大窗口只是延迟发作，不解决。

### 13.2 三层解决策略（由轻到重）

#### 策略 A：State-first protocol（必须做，零新增组件）

**原则**：State Layer（§2 已经是"单一真相源"）才是 Agent 的工作记忆；context 只存指针。

强制规则（落在 `skills/workflow/orchestration.md`）：

1. **每个 stage 出口必须 `state.checkpoint()` + `state.trim()`**
2. **进入下一 stage 前 Orchestrator context 仅保留**：
   ```yaml
   {session_id, current_stage, round_id,
    champion_id, budget_used, last_decision_summary}
   ```
   上轮 Snapshot / CandidatePool / DiagnosisReport 全部丢弃，需要时按引用从 State Layer 读回局部切片。
3. **Skill 不 re-load**：同一 session 内已加载过的 Skill 文件，Orchestrator 不主动重读；Stage Worker 按需读自己 scope 内的 Skill。

**效果**：Orchestrator 稳态 context 固定 < 2K tokens，不随 round 增长。

#### 策略 B：Subagent isolation（推荐做，核心机制）

**原则**：把 context-heavy 的 stage 做成独立 subagent，用完即销毁；Orchestrator 只拿 `SubagentResult` 摘要。

**Subagent 边界表**（哪些 stage 必须做 subagent、哪些不必）：

| Stage | 做 subagent？ | 理由 | 典型输入 | 典型输出 |
|-------|--------------|------|---------|---------|
| **Preflight** | 是 | 首次 ~30min，micro-bench 数据量大；跨任务复用时其实只要 ClusterProfile | `cluster_id` | `ClusterProfile` |
| **Projection** | 是 | 要读 execution-model/* 整个子树 + profiling 数据 | `model_spec, cluster_profile_ref` | `InitialPlans` 引用 |
| **Observe** | 是（每 plan 一个） | 多 plan 并行跑时天然可并行 observe | `run_id` | `Snapshot` 引用 |
| **Diagnose** | 是 | 要读 diagnose.md + execution-model/* + 可能的 profiling trace | `snapshot_id` | `DiagnosisReport` 引用 |
| **Re-Plan** | 是（最重的一个） | optimization/* 整个子树 + env catalog + axis_taxonomy | `diagnosis_report_ref, plan_graph_ref` | `CandidatePool` 引用 |
| **EnvSweep（内层）** | 是 | 天然独立子循环，单次 ≤ 8 组合 × 50 step | `base_plan_ref, candidate_envs` | `EnvSweepResult` 引用 |
| **Correctness-Lite** | 是 | 要拉 reference curve 做比对 | `run_id, reference_ref` | `{pass, delta_pct}` |
| **Execute** | 否 | 主要是 `submit.run()` + 轮询，逻辑轻 | — | — |
| **Settle** | 否 | 纯数值判断 + 规则匹配，很小 | — | — |
| **状态机推进** | 否（Orchestrator 本职） | — | — | — |

**边界设计原则**：

- **在 Skill 跨度最大的地方切一刀**：Re-Plan / Diagnose / EnvSweep 是"读多份 Skill + 写一份产物"的典型，最适合 subagent
- **粒度上限**：不要"每个 Skill 一个 subagent"——RPC 灾难
- **粒度下限**：不要"整个 OPTIMIZE_LOOP 一个 subagent"——那就退化回单 Agent 吞全部细节的反模式

**调用协议**（`orchestration.md` 中规定）：

```python
def orchestrator_step():
    state = state.resume(checkpoint_path)           # < 500 tokens
    next_stage = decide_next_stage(state)           # by state_machine.md

    result = subagent.spawn(
        stage = next_stage,
        input_refs = state.relevant_refs(next_stage),  # 只给指针
        skill_scope = SKILL_SCOPES[next_stage],        # 限制 Worker 能读什么
        max_tokens = 30_000,                           # Worker 预算上限
    )  # 返回 SubagentResult，§8.11

    if result.status == 'failed':
        return handle_failure(result.failure)

    state.apply(result.summary, result.suggested_transition)
    state.checkpoint()
    state.trim(keep=['session_id', 'current_stage', 'round_id',
                     'champion_id', 'budget_used'])
    return state
```

#### 策略 C：Session handoff（兜底，长时调优必备）

**原则**：Orchestrator 本身也可以被替换。即使策略 A + B 都生效，跑 20+ round 或多天任务时仍可能累积；handoff 让 Orchestrator 自己"换班"。

**触发条件**：
- `context_tokens > 0.5 × window`（预警）
- `context_tokens > 0.75 × window`（强制）
- 主动：每 K 轮（K=10）周期性 handoff，保证长期稳定

**协议**：

```python
def handoff_if_needed():
    if self.ctx_tokens < 0.5 * window:
        return
    handoff_path = state.handoff(
        session_id = self.session_id,
        reason = 'context_pressure',
        next_action_hint = self.pending_decision,
    )
    # 新 Orchestrator 被拉起，读 handoff_path 恢复
    spawn_new_orchestrator(resume_from=handoff_path)
    sys.exit(0)
```

**和 §12.2 的关系**：`CONTEXT_OVERFLOW` 走这条回路；不消耗 round 配额。

### 13.3 三层策略如何协同

```
┌─────────────────────────────────────────────────────────────────┐
│ 策略 A  State-first protocol                                     │
│   每 stage 必 checkpoint + trim → Orchestrator context 稳态      │
│   不做：单轮 context 稳不住，B 再多 subagent 也救不回来          │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ 前置条件
                            │
┌─────────────────────────────────────────────────────────────────┐
│ 策略 B  Subagent isolation                                       │
│   Re-Plan / Diagnose / EnvSweep 拆出 → Worker context 用完即丢   │
│   不做：Orchestrator 被迫吸收 Worker trace，A 的指针被污染       │
└─────────────────────────────────────────────────────────────────┘
                            ▲
                            │ 兜底补充
                            │
┌─────────────────────────────────────────────────────────────────┐
│ 策略 C  Session handoff                                          │
│   Orchestrator 自我接力，极长任务下最后防线                      │
│   不做：20+ round / 多天任务仍可能因杂项（错误、诊断循环）撞墙   │
└─────────────────────────────────────────────────────────────────┘
```

### 13.4 Context budget 总账

| 组件 | 稳态预算 | 峰值预算 | 违反处理 |
|------|----------|----------|----------|
| Orchestrator | < 2K tokens | < 10K tokens | 触发 `state.handoff()` |
| Stage Worker（Diagnose） | n/a（一次性） | < 20K tokens | `subagent.spawn(max_tokens=30K)` early-return |
| Stage Worker（Re-Plan） | n/a | < 30K tokens | 同上 |
| Stage Worker（EnvSweep） | n/a | < 15K tokens | 同上 |
| SubagentResult.summary | — | < 200 tokens | `orchestration.md` 规则拒收 |
| State Layer | 不限（磁盘） | — | 只受磁盘容量 |

### 13.5 反模式（明确不做什么）

| 反模式 | 为什么不 |
|--------|---------|
| "每个 Skill 文件一个 subagent" | Skill 间需交叉引用（COMPUTE_BOUND 也要看 comm/overlap.md），RPC 灾难；启动开销 > 收益 |
| "按 round 拆 Agent，一 round 一 Agent" | 跨 round 的 PlanGraph / exhausted_neighborhoods 演化强耦合，不如用策略 A 存到 State Layer |
| "用 subagent 当'更大的 context'用" | 如果 subagent 内部也膨胀（读了全部 optimization/**），问题只是推迟；Worker 必须短命且读片 |
| "Orchestrator 把 Worker 的推理 trace 也吸进来审计" | 审计靠 State Layer + `SubagentResult.artifacts[]`，不靠 context 回显 |
| "用函数调用嵌套替代 subagent" | 没有 context 隔离，等于让单一会话吞下所有 Worker trace |

### 13.6 框架兼容性

Pilot 的主干（Skills + State + Tools + Schemas）与具体 agent runtime 解耦，允许不同框架按各自能力兑现不同层级的 context 管理：

| 框架能力 | Pilot 的要求 | 如果框架不支持 |
|---------|--------------|----------------|
| 原生 subagent 派生 | 推荐：可兑现策略 B 完整 | 降级：让"Stage Worker"退化为同会话内的子对话段，出段手动清上下文——策略 B 失效，仅策略 A 生效，协议仍正确 |
| 原生 handoff / 会话重启 | 推荐：可兑现策略 C | 降级：依赖框架的 context 压缩；手动 Stop/Resume |
| MCP / function calling | 两者选一即可 | 降级：Pilot 的 `tools/` 是 CLI，`shell` 调用就够用 |
| 长时常驻（>1 天） | 推荐：用于 >20 round 任务 | 降级：多次分段运行，每次 `state.resume()` 续起 |

**降级路径始终可用**：即使所有高级能力都没有，只要框架能读 .md + 调 shell + 写 .yaml，Pilot 的主干就能用，只是 context 管理能力会相应弱化。

**工具抽象**：`subagent.*` / `state.handoff()` 这类工具是**协议抽象**——Pilot 规定输入/输出契约和调用时机，具体实现由框架提供（Claude Code / Cursor 的 Task tool、自研 harness 的进程派生、...）。

### 13.7 可分层落地

三层策略不必一次性全部到位；按集成的框架能力与任务长度逐层启用即可：

| 层 | 动作 | 适用场景 | 前置依赖 |
|----|------|----------|----------|
| 最小可用 | 只做策略 A：落 `state.trim()` + `orchestration.md` 的 hygiene 规则 | 5–8 round 短任务 | 无 |
| 核心 | 增加策略 B：Re-Plan / Diagnose / EnvSweep 三个 subagent | 10+ round 中等任务 | 框架支持 subagent 派生 |
| 完整 | 增加策略 C：`state.handoff()` + `CONTEXT_OVERFLOW` 回路 | 多天 / 20+ round 长时任务 | 框架支持会话重启 / resume |

每一层都可独立部署并带来对应收益；上层失效不影响下层的正确性。

---

## 14. 一句话总结

Primus Pilot = 调优领域的**知识 + 工具包**（`skills/` 知识 + `prompts/` 角色 prompt + `tools/` 业务动作 + `schemas/` 数据契约 + `state/` 持久化），**runtime 交给 agent 框架**（Claude Code / Cursor / 自研 harness）——框架用自己的主会话承担 **Orchestrator** 角色、用原生 subagent 机制派生 **Stage Worker**，按 Pilot 规定的 state-first protocol 推进状态机，让长循环下 context 保持 O(1) 稳态，完成 Preflight → Projection → Tuning Loop 的闭环。
