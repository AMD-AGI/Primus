# 任务：把 pilot/ 从"自造 agent runtime"重写成"调优 skill 集合"

## 背景

`pilot/README.md` 当前是 1789 行的设计文档，把 Pilot 设计成了一个跨框架的自治调优系统，
自己造了 Orchestrator / Stage Worker、PlanGraph、CandidatePool、TuningState、
SubagentResult 等一堆 schema，以及 `prompts/ tools/ schemas/ integrations/ agent/`
五个目录。这是过度设计：

- 用户就在 Cursor 里跑，**Cursor 主 session 就是 Orchestrator**，
  **Cursor 的 Task subagent 就是 Stage Worker**，**chat 历史就是 state**。
  完全不需要 Pilot 自己再造一遍。
- 真正稀缺、Pilot 必须自己提供的只有**调优领域知识本身**：execution model 公式、
  按瓶颈分类的优化策略、env catalog、profiling / preflight 经验、constraint。
  这些应该全部以 **SKILL.md** 形式存在，对齐 `.cursor/skills/` 里的习惯。

## 你的任务

把 `pilot/` 重写成一个简洁的 skill 集合，约定一条规则：
**每次"提交训练 + 收集 profile"都在 subagent 里执行**，主 agent 读 skill 去 think，
不要再自己造 state machine / schema / orchestrator。

## 第一步：必读文件（先读完再动手）

1. `pilot/README.md` — 当前过度设计的版本，**你要从这里抽取领域知识**
   （execution model 公式、瓶颈分类、env 列表、优化策略、preflight 怎么做等），
   抽完之后这些知识要重新组织进新的 SKILL.md 里。
2. `.cursor/skills/slurm-idle-node-check/SKILL.md` — **skill 格式范本**，
   你的所有新 SKILL.md 都要严格对齐这个格式（frontmatter + Workflow + 可执行步骤）。
3. `.cursor/skills/slurm-training-node-validation/SKILL.md` — 另一个 skill 范本，
   特别看它怎么用 subagent 做并行执行。
4. `.cursor/skills/slurm-xiaoming-dev-container/SKILL.md` — 看它怎么描述
   "进入容器跑东西"的约定，run-and-profile skill 会用到类似套路。
5. `examples/README.md` 和 `examples/run_local_pretrain_cli.sh` —
   了解 Primus 训练是怎么实际启动的，run-and-profile 和 preflight skill
   需要知道真正的命令行长什么样，不要凭空发明。

## 目标目录结构

```
pilot/
├── README.legacy.md                   # 把当前 README 原样改名保留（作为知识源参考）
├── README.md                          # 新的，一页：定位 + 用法 + subagent 约定
└── skills/
    ├── tuning-loop/
    │   └── SKILL.md                   # 主入口 skill：think → run → diagnose → 下一轮
    │                                  # 描述整个调优循环的"think 流程"，是其他 skill 的总调度
    ├── execution-model/
    │   └── SKILL.md                   # T_step / T_comp / T_comm / T_bubble / Mem 公式
    ├── bottleneck-diagnose/
    │   └── SKILL.md                   # 看到 snapshot 后如何判断 COMM / PIPELINE / MEMORY / COMPUTE
    ├── optimize-comm/SKILL.md         # 通信瓶颈策略 + 相关 env 候选
    ├── optimize-pipeline/SKILL.md     # bubble / vpp / mbs / stage balance
    ├── optimize-memory/SKILL.md       # recompute / offload / 碎片
    ├── optimize-compute/SKILL.md      # mbs / parallel / kernel hint
    ├── optimize-moe/SKILL.md          # routing / dispatch / load balance
    ├── env-catalog/
    │   └── SKILL.md                   # rccl / hsa / alloc / threading 完整 env 字典（单点维护）
    ├── run-and-profile/
    │   └── SKILL.md                   # 约定：每次跑训练 + 收 profile = 一个 subagent
    │                                  # 必须明确写出：subagent 入参格式、subagent 出参格式
    └── preflight/
        └── SKILL.md                   # 集群 baseline + env_probe（首次或 cluster 变更后跑）
```

## 严格不要做的事（很重要）

绝对**不要**创建以下任何东西，全部是当前 README 的过度设计，新版要彻底删掉：

- ❌ `pilot/prompts/` 目录 — skill 本身就是 prompt
- ❌ `pilot/tools/` 目录 — 不要写 `state.checkpoint / state.trim / state.handoff /
  subagent.spawn / knowledge.write` 这些 Python 文件，Cursor 自带这些能力
- ❌ `pilot/schemas/` 目录 — 不要为 PlanGraph / CandidatePool / TuningState /
  SubagentResult / FailureReport / DiagnosisReport / EnvSweepResult 写 JSON Schema
- ❌ `pilot/state/` 目录 — chat 历史就是 state，最多在 skill 里建议
  "需要长期保存的最佳配置写到 `output/pilot/<session>.md`"
- ❌ `pilot/integrations/` 目录 — 用户就在 Cursor 里用，不需要做框架适配
- ❌ `pilot/agent/` 目录 — 不要写 Python 版的 Orchestrator / StageWorker fallback
- ❌ 不要设计 state machine（on_fail / reentry_when / state transition 表）
- ❌ 不要设计 Strategy A / B / C 三层 context management
- ❌ 不要把"Orchestrator vs Stage Worker"写成两个角色规范，
  让 LLM 自己读 skill 自己 think 即可

## 新 README.md 应该包含什么（一页内）

只需要：

1. **What**：Pilot 是给训练调优用的 skill 集合（一句话）。
2. **Who reads it**：Cursor 主 agent 读 skill 去 think，subagent 跑实际训练。
3. **Convention**：唯一一条硬约定 ——
   "每次提交训练 + 收 profile 必须在 subagent 里执行，
    subagent 返回一段 markdown 摘要给主 agent，不要把 profile 原文塞回主 chat"。
4. **入口 skill**：`pilot/skills/tuning-loop/SKILL.md`，从这里开始读。
5. **Skill 索引表**：列出所有 skill 名 + 一句话描述 + 什么时候触发。
6. **完**。不要再写 §1 Problem & boundary、§2 Architecture、§3 System flow、
   §13 Context Management 这些章节，全删。

## 每个 SKILL.md 的格式硬要求

严格对齐 `.cursor/skills/slurm-idle-node-check/SKILL.md`：

```markdown
---
name: <skill-name-kebab-case>
description: <一句话说明这个 skill 是干什么的，以及"什么时候用"。
             触发词要写清楚，因为 Cursor 用这段决定要不要加载这个 skill。>
---

# <Title>

<2-3 句概述>

## Workflow

### Step 1: ...
### Step 2: ...
...

## Important Notes
...
```

具体要求：
- 全文 **英文**，对齐 `.cursor/skills/` 里现有 skill。
- description 字段必须能让 Cursor 在"用户提到 X / Y / Z"时正确触发，
  例如 `optimize-comm/SKILL.md` 的 description 应包含
  "communication bottleneck, NCCL/RCCL tuning, allreduce/alltoall, bucket size, overlap" 等关键词。
- 优先给"判断逻辑 / 公式 / 阈值 / 决策表"，不要写大段散文。
- 用 markdown 表格组织"瓶颈 → 候选策略"、"症状 → env flag"这种映射。

## `run-and-profile/SKILL.md` 必须明确的契约

这是**唯一**需要约定接口的 skill，必须写清楚两件事：

### Subagent 入参（主 agent 调用 subagent 时给什么）

```yaml
plan:
  parallelism: {tp, pp, dp, ep, vpp, ...}
  runtime: {mbs, gbs, recompute, ...}
  comm: {bucket_size_mb, overlap, ...}
  env_diff: {NCCL_*: ..., HSA_*: ..., ...}     # 只写相对 baseline 的 diff
cluster:
  nodes: N
  gpus_per_node: M
purpose: <baseline | smoke | candidate-eval | env-sweep>
max_steps: <int>
```

### Subagent 出参（subagent 跑完返回给主 agent）

固定一段 markdown，主 agent 只看这段，不读 profile 原文：

```markdown
## Run Result (run_id: <id>)

- **plan**: <一行 yaml summary>
- **status**: completed | early_stopped | oom | hang | failed
- **metrics**: tps=<>, step_time_ms=<>, comm_ratio=<>, bubble_ratio=<>, mem_peak_gb=<>
- **bottleneck hint**: COMM_BOUND | PIPELINE_BOUND | MEMORY_BOUND | COMPUTE_BOUND | MIXED
- **one-line summary**: <一句话，<200 字>
- **artifacts**:
  - profile: <path>
  - log: <path>
  - snapshot.yaml: <path>     # 主 agent 需要时再用 Read 工具读
```

`tuning-loop/SKILL.md` 里要明确写：
"主 agent 每轮的 chat 里只保留这段 markdown 摘要，
 不要把 profile / log 原文 paste 回 chat"。

## `tuning-loop/SKILL.md` 应该包含什么

这是核心入口 skill，要写清楚 think 流程，包括：

1. **怎么开始一次调优** —— 用户给了 model + cluster + 目标，主 agent 该做什么
2. **PREFLIGHT / SMOKE / BASELINE / LOOP 的顺序**（保留 README §3.1 的主流程经验）
3. **每一轮 LOOP 的 think 步骤**：
   - 决定下一轮 plan（参考 bottleneck-diagnose + optimize-* skill）
   - 调 run-and-profile subagent
   - 看摘要，更新自己的"已试过 / 已淘汰 / 当前最佳"心智模型
   - 判断收敛 / 继续
4. **think tips**（淡化版的 README §7）：
   - 已经试过的参数组合不要再试
   - 连续 2 轮 gain<2% 就停
   - 偶尔从次优分支再 fork 一次，避免局部最优
   - 不需要 PlanGraph schema，自己在 chat 里维护即可
5. **何时升级到长期记忆** —— 出最终 report 时把最佳配置写到
   `output/pilot/<session_id>.md`，方便下次同模型同集群参考

## 知识从哪里来

`pilot/README.md` 里这些章节是真正的领域知识，重写时**不要丢**，按下面的对应关系搬：

| 旧 README 章节 | 搬去新的哪个 SKILL.md |
|---|---|
| §6 Execution Model 全部公式 | `execution-model/SKILL.md` |
| §3.1 主流程图 / §3.2 Flow narrative 的阶段定义 | `tuning-loop/SKILL.md` |
| §4.2 `optimization/comm/*` 的策略要点 | `optimize-comm/SKILL.md` |
| §4.2 `optimization/pipeline/*` | `optimize-pipeline/SKILL.md` |
| §4.2 `optimization/memory/*` | `optimize-memory/SKILL.md` |
| §4.2 `optimization/compute/*` | `optimize-compute/SKILL.md` |
| §4.2 `optimization/moe/*` | `optimize-moe/SKILL.md` |
| §4.2 `env/{rccl,hsa,alloc,threading,presets}.md` | `env-catalog/SKILL.md` |
| §4.2 `profiling/*` + §4.2 `constraints/*` | 拆进 `preflight/SKILL.md` 和各 optimize-* |
| §8.4 DiagnosisReport 里的判断阈值 | `bottleneck-diagnose/SKILL.md`（作为决策表，不是 schema）|
| §8.1 ClusterProfile 字段定义 | `preflight/SKILL.md`（作为"采集什么"清单，不是 JSON Schema）|
| §9 Full iteration example | `tuning-loop/SKILL.md` 末尾作为一个 worked example |

§2.2 / §13 / §12.2 这些纯 orchestration / context / state machine 的章节
**全部不要搬**，新设计里这些事 Cursor 自己就做了。

## 验收标准

1. `pilot/README.legacy.md` 存在，是当前 1789 行 README 的 git mv 改名。
2. 新 `pilot/README.md` ≤ 200 行，一页讲清定位 + 用法 + 约定。
3. `pilot/skills/` 下 11 个 SKILL.md 全部建好，每个都有正确的 frontmatter
   和可触发的 description。
4. `pilot/` 根目录下**没有** `prompts/ tools/ schemas/ state/ integrations/ agent/`
   任何一个目录。
5. 没有任何 JSON Schema 文件、没有任何 Python 文件。
6. `run-and-profile/SKILL.md` 明确定义了 subagent 入参 / 出参格式（markdown，非 schema）。
7. 旧 README 里所有领域知识都按上表搬到了对应的新 SKILL.md，没有丢知识。

## 工作建议

1. 先把 `pilot/README.md` git mv 到 `pilot/README.legacy.md`。
2. 通读一遍 `README.legacy.md`，把领域知识按上表分类标记。
3. 先写 `tuning-loop/SKILL.md` 和 `run-and-profile/SKILL.md`（骨架先确定）。
4. 再写其他 9 个 SKILL.md，分批写，每写完一批用 `ReadLints` 和重新读一遍校对。
5. 最后写新的 `pilot/README.md`（一页）。
6. 自检：`ls pilot/` 应该只有 `README.md` `README.legacy.md` `skills/` 三项。

开工吧。遇到拿不准的设计问题，**优先选更简单的那个方案**，
不要往 README.legacy.md 那个复杂方向回退。
