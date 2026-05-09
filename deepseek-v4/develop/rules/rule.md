# DeepSeek-V4 Development Rules

> 本文件汇总 `wen_xie_qle` 在 DeepSeek-V4 集成开发过程中沉淀下来的
> 工作习惯 / 工程规则 / 标准做法。新规则一律追加到本文件，按照下面
> 的分节归类；废止的规则用 `~~strikethrough~~` 标记并保留原文 + 废止
> 日期。
>
> All rules are written in English; section titles are bilingual for
> findability. Rules are atomic and self-contained — pick the section
> first, then read the rule.
>
> Last updated: 2026-05-09 (P30 perf-table rule).

---

## §1. Code-review workflow / 代码评审流程

### R1.1 — Stop before commit for review
**STANDING RULE since plan-3 P23 (commit `6d185003`).**
After finishing the work for a phase / task, **DO NOT commit
automatically**. Stop and present a one-page summary of what changed,
what passed gates, and what was de-scoped. Wait for the user to
review and explicitly say "commit" before running `git add` /
`git commit`.

### R1.2 — Push only on explicit prompt
Even after a commit, **DO NOT** `git push` until the user says
something like "push to origin", "push origin", or "push". Treat the
local commit as the natural pause point.

### R1.3 — Status-pin commit pattern
Every feature commit (`feat(deepseek-v4)[P<id>]: ...`) is followed
**immediately** by a docs-only commit pinning the corresponding
`status.md` rows to the feature SHA:

```
docs(deepseek-v4)[P<id>]: pin status.md P<id> cells to the P<id> SHA (<feature-sha>)
```

The pin commit ONLY touches `develop/progress/status.md` (and
`p<id>-summary.md` when bumping the `TBD-p<id>` placeholder). Never
fold the pin into the feature commit — the split is what lets a
future reader follow the audit trail without scanning the whole
status diff.

### R1.4 — Commit message format
- Subject prefix: `feat(deepseek-v4)[P<id>]:` for feature work,
  `refactor(deepseek-v4)[P<id>]:` for behaviour-preserving
  refactors, `docs(deepseek-v4)[P<id>]:` for docs-only commits.
- When a commit closes a plan or opens a plan, add the plan tag too:
  `feat(deepseek-v4)[plan-5][P28]: ...`,
  `docs(deepseek-v4)[plan-5]: open plan-5 ...`.
- Subject ≤ 100 chars; HEREDOC for the body when there is one.
- See git log of `2ec7e40c`, `afd7ea59`, `da6f48bc` for canonical
  examples.

---

## §2. Phase deliverables / 每个阶段必交的产物

### R2.1 — Per-phase summary file (NEW since P29)
Every phase that ships work — even pure-doc rescopes — closes with
a one-page summary at `develop/progress/p<id>/p<id>-summary.md`.

The 8 required sections are pinned in
`develop/plan-5/01-roadmap.md` ("Per-phase deliverable convention").
`p29-summary.md` is the canonical example.

The summary lands in the **same commit as the status-pin commit**
(R1.3), not the feature commit, so a future reader can read the
summary first and follow links into the feature commit.

### R2.2 — Status row format
Every status.md row has **5 columns**: `[ ] | task | commit | date |
note`.
- `[x]` complete, `[ ]` pending, `[-] ~~task~~` de-scoped (preserve
  the strike-through original text).
- `commit` placeholder before the actual commit: `TBD-p<id>`.
  Filled in by the pin commit (R1.3).
- `date` is the **work-completion date**, not the pin-commit date.
- `note` is the running blocker / decision / numerics record. Keep
  it long-form when it contains forensic data; short when not.

### R2.3 — Trace + report layout
Performance traces and reports follow a fixed layout:

| artefact | path |
|---|---|
| trace-capture script | `develop/progress/p<id>/run_*_trace_*.sh` |
| smoke script | `develop/progress/p<id>/run_smoke_*.sh` |
| raw chrome trace JSON | `output/...` (gitignored — see R6.1) |
| baseline / after-phase report | `develop/profile/profile-{baseline,after-p<id>}-ep<N>-<YYYYMMDD>.{md,html}` |
| report renderer | `develop/profile/_tools/render_baseline_report.py` |

### R2.4 — Forensic helper scripts
One-off forensic Python scripts (e.g. `_forensics{,2,3}.py`) live
under `progress/p<id>/` with a leading underscore to mark them as
non-shipping helpers. Their generated text outputs (`forensics*_output.txt`)
are gitignored; **only the helper script and the curated
attribution table inside `refinement.md` get committed**.

### R2.5 — Performance table upkeep
When a task optimises V4 attention kernels, update
`develop/perf/attention_perf.md` after the attention operator unit tests
pass. The table entry must record the new FWD / BWD TFLOP/s for every
affected attention family (`compress_ratio == 0`, `4`, and / or `128`)
using the table's documented shape and FLOP-counting convention.

When a task runs the EP8 proxy and end-to-end performance improves,
update `develop/perf/proxy_ep8.md` with the new steady iter time and
TFLOP/s/GPU. Keep intermediate rows when they explain an optimisation
step or regression (for example, dense-only P30 before HCA split-mask).

---

## §3. Documentation conventions / 文档约定

### R3.1 — English in dev docs
All `develop/plan-*/`, `develop/progress/`, `develop/profile/`,
`develop/techblog/` documentation is written in English.
**The bilingual exception is allowed only inside `develop/rules/`
and `develop/notes/`** (this file is the example).
Plan-4 P25 (`f467a739`) explicitly translated the Chinese parts of
the plan-4 README to English; that is the canonical decision.

### R3.2 — Plan layout
Every plan ships under `develop/plan-<n>/` with three required files
plus an optional README:

| file | purpose |
|---|---|
| `01-roadmap.md` | phase overview table, dependency graph, milestones, top risks |
| `02-phase-details.md` | per-phase task breakdown, design notes, edge cases |
| `03-test-strategy.md` | gate matrix, correctness ratchet, perf-budget contract |
| `README.md` (optional) | high-level pitch + links |

### R3.3 — Gate naming
Numerical equivalence + smoke + perf gates are named `G<N>` where
`<N>` is monotonically increasing across the project lifetime
(plan-3 owns G1..G10, plan-4 owns G11..G30, plan-5 starts at G31).
**Never reuse a gate number.** Sub-letters (`G33a` / `G33b`) are
allowed when the parent gate has multiple sub-criteria captured by
distinct scripts.

---

## §4. Test conventions / 测试约定

### R4.1 — Test file naming
`tests/unit_tests/megatron/transformer/deepseek_v4/test_v4_p<id>_<short_name>.py`.
The phase id in the filename is the phase that introduced the test;
later phases that extend it keep the original filename.

### R4.2 — `pytest.mark.slow` for release-tier shapes
Numerical equivalence tests that hit production-scale shapes
(V4-Flash widths: `H=64, head_dim=512, S=4096`, etc.) are marked
`pytest.mark.slow` and only run under `pytest -m slow` or
`--run-slow`.
Fast-tier (small-shape) parametrisations run by default in CI / dev
loops; release-tier is opt-in.

### R4.3 — Real shape > mocked shape
**STANDING RULE since P27 (commit `e19663f7`).**
When introducing a new operator unit test, add **real** V4-Flash /
V4-Pro shape parametrisations from day one (release-tier behind
`pytest.mark.slow`). VRAM is not a blocker for op-only tests.

### R4.4 — Banned-warning ratchet
Smoke runs must `grep` clean against the running banned-warning set.
Each plan extends the set; the **current set is pinned in
`develop/plan-<n>/03-test-strategy.md`** under the "banned warnings"
heading. Categories so far:
- plan-3: `c10d::allreduce` autograd fallback
- plan-4: V4 attention fallback / nn.Linear fallback
- plan-5: `sinkhorn_normalize compile error`

### R4.5 — Correctness ratchet
Every phase must keep prior plan's gates green. Plan-5 P29 still
runs plan-4 G23..G30 before declaring close. Pre-existing failures
unrelated to the phase (e.g. `test_v4_mtp.py::test_helper_pulls_norm_and_linear_from_v4_provider`)
get documented in the summary file's "known follow-ups" section
**after** verification by stash + re-run.

---

## §5. Run scripts and env-var conventions / 启动脚本与环境变量

### R5.1 — `${VAR:-DEFAULT}` guards
Every override-able variable in `run_deepseek_v4.sh`,
`run_deepseek_v4_flash_proxy.sh`, and the `progress/p<id>/run_*.sh`
scripts uses `${VAR:-DEFAULT}` so callers can flip any knob from the
command line without editing the script.

### R5.2 — `PRIMUS_*` namespace for proxy overrides
Proxy / model-shape overrides exposed to callers use the
`PRIMUS_*` env-var prefix (e.g. `PRIMUS_SEQ_LENGTH`,
`PRIMUS_TOTAL_LAYERS`, `PRIMUS_NUM_EXPERTS`).
Feature-flag knobs use the matching un-prefixed name in
`UPPER_SNAKE_CASE` (e.g. `USE_V4_TRITON_ATTENTION`,
`USE_V4_COMPILED_SINKHORN`, `USE_TURBO_DEEPEP`).

### R5.3 — V4 perf-knob inventory (current)
| env var | yaml field | default in `run_deepseek_v4.sh` | default in proxy |
|---|---|---:|---:|
| `USE_V4_TRITON_ATTENTION` | `use_v4_triton_attention` | True | True |
| `USE_V4_TRITON_CSA_ATTENTION` | `use_v4_triton_csa_attention` | True | True |
| `USE_TURBO_DEEPEP` | `use_turbo_deepep` | True | True |
| `TURBO_USE_GROUPED_MLP` | `turbo_use_grouped_mlp` | True | True |
| `USE_V4_COMPILED_SINKHORN` | `use_v4_compiled_sinkhorn` | False | True |
| `USE_TURBO_ATTENTION` | `use_turbo_attention` | False | False (precedence rule, R8.2) |

### R5.4 — DeepEP best-practice config (when `use_turbo_deepep=True`)
The following knobs MUST be set together when `use_turbo_deepep` is
True:

| knob | value |
|---|---|
| `moe_shared_expert_overlap` | `false` (DeepEP forbids) |
| `moe_router_dtype` | `fp32` |
| `turbo_deepep_use_comm_stream` | `false` |
| `turbo_deepep_num_cu` | `80` for EP < 16, `32` for EP ∈ [16, 64] |

This is encoded in `run_deepseek_v4.sh:46..72`. Source: P23 user
note + DeepEP best-practices.

---

## §6. Git hygiene / Git 习惯

### R6.1 — Don't commit smoke logs / traces / heavy artefacts
**STANDING RULE since plan-3 P23.**
The following NEVER get committed:
- training stdout / stderr (`*.log`, `log_*.txt`, `debug.log`)
- chrome traces (`*.json` under `output/.../tensorboard/`,
  `*.tfevents*`, `trace_*.txt`)
- compressed traces (`*.tgz`)
- forensic helper outputs (`forensics_output.txt`,
  `forensics2_output.txt`, `forensics3_output.txt`, `_forensics_*.txt`)

Each `progress/p<id>/.gitignore` enumerates the patterns explicitly.
The canonical example is `progress/p29/.gitignore`.

### R6.2 — Don't commit vendor clones
Vendored upstream clones live under `deepseek-v4/<repo>/` (e.g.
`aiter/`, `Primus-Turbo/`, `TransformerEngine/`, `NVIDIA-NeMo/`,
`transformers/`, `deepseek-ai/`). They are reference-only; **never
add them to git**. The top-level `.gitignore` keeps them out.

### R6.3 — Don't commit local scaffolding
`bak/`, `output/`, `ut_out/`, `run_alloc.sh`, `pr-*-body.md`,
`tools/docker/start_container_*.sh`, and other host-specific
scaffolding stay local. If a script is genuinely shared,
add it deliberately under `tools/` / `examples/` / `scripts/`.

### R6.4 — No interactive git
Never use `git rebase -i`, `git add -i`, `git commit --amend`
(unless explicitly asked + the strict rules in CLAUDE / the system
prompt are met). Never `git push --force` to `main` / `master` /
`origin/dev/wenx/deepseek-v4` without the user's explicit consent.

---

## §7. Code style / 代码风格

### R7.1 — No narrative / explanatory comments
Comments only for non-obvious intent, trade-offs, or constraints
the code itself can't convey. Avoid:
- `# Import the module`
- `# Define the function`
- `# Increment the counter`
- `// Return the result`
- `// Handle the error`

Comments explaining "why this design over the obvious alternative"
are encouraged (see `hyper_connection.py:60..86` for the
`dynamic=True` rationale).

### R7.2 — bf16 tensor-core matmul + fp32 softmax (V4 attention)
**STANDING DTYPE CONTRACT since P24 (commit `38ef526c`).**
For the V4 attention reference / Triton paths:
- **All matmuls run on bf16 tensor cores** (input + weight + accum
  staying in bf16 except where MFMA forces fp32 accum).
- **Only softmax / Sinkhorn run in fp32** (scale + max + exp + sum
  + cast back).

Anything else (e.g. dropout, masking, RoPE) follows the input dtype
unless the call site explicitly says fp32.

### R7.3 — Imports at top of file, not inline
No mid-function `import ...`. Any deferred / lazy import gets a
named helper function near the top of the file with a docstring
explaining why the deferral is necessary.

---

## §8. Architecture / dispatch precedence rules / 调度优先级

### R8.1 — Switch flag namespace (`use_v4_triton_*`)
**STANDING NAMING DECISION since plan-4 (commit `f467a739`).**
In-tree V4-specific Triton kernels are gated behind switches named
`use_v4_triton_<kernel_name>` (e.g. `use_v4_triton_attention`,
`use_v4_triton_csa_attention`).
The earlier name `use_v4_attention` was retired during plan-4 plan
authoring; current callers must use the `use_v4_triton_*` form.

### R8.2 — V4 attention dispatch precedence
**STANDING DECISION since plan-4 P27 (commit `e19663f7`).**
For `compress_ratio ∈ {0, 128}` layers (dense / HCA), the dispatch
order in `DeepseekV4Attention.forward` is:

1. Turbo (`use_turbo_attention=True`) — currently broken on aiter
   for `head_dim=512` + sink, kept for future re-enable.
2. V4 Triton (`use_v4_triton_attention=True`) — current production
   path.
3. Eager reference (`v4_attention_kernels/reference.py`) —
   numerical baseline.

For `compress_ratio == 4` layers (CSA), Turbo is skipped:

1. V4 Triton CSA (`use_v4_triton_csa_attention=True`).
2. Eager CSA reference.

### R8.3 — TFLOPs counting rule
**STANDING DECISION since P20 (commit `4c27787d`).**
Megatron `num_floating_point_operations` for V4 counts ONLY:
- GEMM-type ops (linear, MoE expert linear, output projection,
  dense FFN, gated MLP)
- attention-type ops (Q×K, attn-weights × V, including SWA
  and CSA variants)

It does NOT count:
- softmax / sigmoid / RMSNorm / Sinkhorn / layernorm
- elementwise ops (mul, add, rsqrt, GELU)
- reductions (sum, mean) outside attention

This is the contract for the TFLOP/s/GPU number reported in every
plan-5 trace report.

---

## §9. Performance analysis / 性能分析

### R9.1 — 10 % de-scope rule
**STANDING DECISION since plan-5 P28.**
A bottleneck row in a `develop/profile/profile-*.md` ranked-bottleneck
table that accounts for **< 10 %** of steady iter wall time triggers
de-scope of the corresponding plan-5 phase (or sub-task).
The rule is applied at phase-close (read the trace numbers, then
update the next phase's tasks accordingly). De-scoped tasks are
kept in tree as plan follow-ups, not deleted.

### R9.2 — Multi-stream overlap factor
The chrome trace's `Σ kernel duration` across streams divided by
the wall-clock GPU active time gives the **multi-stream overlap
factor** (1.0× = sequential, > 1.0× = parallel streams).
A change in this factor between two traces is a primary explanation
candidate when wall-time delta does not match Σ kernel-time delta
(see `profile-after-p29-ep8-20260509.md` for the canonical worked
example: P29 dropped Σ kernel-time by ~7 s but wall-time by only
0.2 s because the overlap factor collapsed from 1.87× to 1.00×).

### R9.3 — Forensic attribution before fix
**STANDING DECISION since plan-5 P29 rescope.**
Before designing a perf fix, attribute the bottleneck to a single
Python source line via the chrome trace's `External id` correlation
table. The forensic helper script gets committed (under
`progress/p<id>/_forensics*.py`) so the attribution can be re-run
on a future trace.

---

## §10. Dev environment / 开发环境

### R10.1 — Dev machine + container
- Default dev host: `mi355-gpu-14` (8× MI355X).
- Default training container: `dev_primus_wenx_693`
  (entered via `podman exec`).
- SSH directly into the host; no jump-box / proxy required.

### R10.2 — Workspace path
Workspace lives at
`/shared/amdgpu/home/wen_xie_qle/workspace/Primus-deepseek-v4`
and is shared across hosts. Output (`output/`) is per-host and
gitignored.

### R10.3 — Branch naming
Active dev branch: `dev/wenx/deepseek-v4`. PRs land into Primus
`main` from this branch.

---

## §11. House-keeping / 维护本文件

### R11.1 — Adding a new rule
- Pick the smallest matching `§N` section. Create a new `§` only
  when no existing section fits.
- Use the `R<N.M> — <one-line title>` format for the heading.
- First paragraph: when the rule started + the source commit /
  conversation if known.
- Keep each rule self-contained; **no cross-rule dependencies**
  inside a single rule body.

### R11.2 — Retiring a rule
Mark the rule heading and body with `~~strike-through~~` and add
a final line `**Retired YYYY-MM-DD: <one-line reason>**`. **Do
not delete retired rules** — they are the audit trail.

### R11.3 — Bumping `Last updated`
Update the date at the top of this file every time a rule is added,
amended, or retired.
