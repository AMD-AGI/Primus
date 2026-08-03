# Primus Monthly Engineering Report — 2026-08

## Time window

- Timezone: Asia/Shanghai (GMT+8)
- Start: **2026-06-02 17:53 (GMT+8)**
- End: **2026-08-03 17:03 (GMT+8)**
- Span: **~62 days**

> **Extended (backfill) window — gap self-healed.** This report covers
> **2026-06-02 → 2026-08-03**, not just calendar August, because the previous
> monthly run for `2026-08` used a too-narrow window (only ~8 hours on
> 2026-08-01) and no report was published for July. Under the continuity rule,
> the `2026-08` metadata being replaced is excluded from the window scan, so
> `window_start` falls back to the end of the last contiguous coverage group
> (the `2026-06` monthly report, which ended 2026-06-02 17:53 GMT+8). All merged
> PRs and pin changes from that ~2-month gap are included below so no history is
> lost.
>
> **Sanity flag:** the resulting window spans **~62 days**, which is well beyond
> a healthy ~30-day monthly window (> 45-day threshold). This is expected for a
> backfill run and is called out here for human sanity-checking; the window was
> **not** truncated.

## Executive summary

- **123 PRs** were merged to `main` in this extended window (2026-06-02 →
  2026-08-03). This is a **backfill** run that recovers merged-PR and pin-change
  history from June and July, which the earlier narrow `2026-08` run had
  silently dropped.
- **Gap detected: yes.** `window_start` (2026-06-02) is earlier than the first
  day of the current calendar month (2026-08-01), so this run re-covers the
  June→August gap. The extended span (~62 days > 45 days) is flagged above.
- Engineering activity was dominated by the **Flux / diffusion backend**
  (a large multi-PR series adding a diffusion training stack), **DeepSeek-V4**
  training support, **MLPerf** examples, and broad **CI / test-coverage
  hardening**. `Other` (largely feature work) and `Bug Fix` are the two largest
  categories.
- **Backend pin change inside the window:** `third_party/torchtitan` was bumped
  from `5fb7cc2e` (v0.1.0) to `73a0e697` (v0.2.2 tag) in **PR #871** (merged
  2026-07-17). Because this submodule SHA change falls **inside** the window,
  the torchtitan backend-gap report set was **regenerated** this run.
- **No Megatron-LM submodule change** in the window (`d3528a21` unchanged), so
  the Megatron backend-gap report was **not** regenerated. Megatron-LM upstream
  drift, however, has now crossed **1050 commits** behind upstream `main`.
- **Primus-Turbo** CI/benchmark/AITER/TRITON pins all advanced during the window
  (Q3 bumps in early July); Primus-Turbo has no standalone backend-gap report,
  so its drift is captured in the quarterly-drift section below.

## Monthly PR update table

| PR | Merged Time (GMT+8) | Category | Key Update |
| --- | --- | --- | --- |
| [#740](https://github.com/AMD-AGI/Primus/pull/740) | 2026-06-03 09:06 | Performance Optimization | feat(torchtitan): SDMA copy-engine all-gather for FSDP |
| [#744](https://github.com/AMD-AGI/Primus/pull/744) | 2026-06-03 19:28 | Bug Fix | fix(runner): run direct launcher via array instead of eval |
| [#745](https://github.com/AMD-AGI/Primus/pull/745) | 2026-06-04 09:39 | Bug Fix | [Megatron-LM] fix primus turbo attn memory footprint |
| [#743](https://github.com/AMD-AGI/Primus/pull/743) | 2026-06-04 10:47 | Bug Fix | [Megatron-LM] fix primus turbo grouped linear weight layout |
| [#587](https://github.com/AMD-AGI/Primus/pull/587) | 2026-06-04 17:21 | Other | [Megatron-Bridge] Mamba SFT posttraining support |
| [#747](https://github.com/AMD-AGI/Primus/pull/747) | 2026-06-05 15:01 | Other | feat(megatron): forward last-rank training_log to rank 0 |
| [#749](https://github.com/AMD-AGI/Primus/pull/749) | 2026-06-06 06:14 | Turbo/Dependency Version Update | [Fix] pin down dependencies for megatron |
| [#750](https://github.com/AMD-AGI/Primus/pull/750) | 2026-06-08 16:12 | CI/Infra | feat(packaging): build & publish Primus wheel with bundled primus-cli |
| [#756](https://github.com/AMD-AGI/Primus/pull/756) | 2026-06-09 08:57 | Bug Fix | [Fix] unsupported moe_use_legacy_grouped_gemm arg in Mamba models |
| [#742](https://github.com/AMD-AGI/Primus/pull/742) | 2026-06-10 07:57 | Performance Optimization | [Megatron-LM] feat: use_turbo_permute_padding to reduce d2h |
| [#748](https://github.com/AMD-AGI/Primus/pull/748) | 2026-06-10 08:57 | Other | Add llama3.1 8b mxfp4 support |
| [#746](https://github.com/AMD-AGI/Primus/pull/746) | 2026-06-10 10:33 | Other | Add llama3.1 8b mxfp8 example |
| [#751](https://github.com/AMD-AGI/Primus/pull/751) | 2026-06-10 16:48 | Other | chore(skills): consolidate agent skills under skills/ |
| [#701](https://github.com/AMD-AGI/Primus/pull/701) | 2026-06-10 19:02 | Other | feat(megatron): support sft native |
| [#737](https://github.com/AMD-AGI/Primus/pull/737) | 2026-06-10 19:10 | Turbo/Dependency Version Update | chore(docker): bump base image rocm/primus v26.2 -> v26.3 |
| [#757](https://github.com/AMD-AGI/Primus/pull/757) | 2026-06-11 09:09 | CI/Infra | feat(packaging): third_party pip extras + fix dashboard refresh |
| [#763](https://github.com/AMD-AGI/Primus/pull/763) | 2026-06-12 07:49 | Performance Optimization | feat(megatron): SDMA copy-engine all-gather for FSDP2 |
| [#762](https://github.com/AMD-AGI/Primus/pull/762) | 2026-06-13 01:02 | Bug Fix | fix PYTHONPATH for pip-installed primus |
| [#767](https://github.com/AMD-AGI/Primus/pull/767) | 2026-06-16 07:12 | CI/Infra | [OOB Release] add jax v26.4 dockerfile |
| [#755](https://github.com/AMD-AGI/Primus/pull/755) | 2026-06-16 07:16 | Other | Tuning agent, memory-based benchmarking support plus fixes |
| [#764](https://github.com/AMD-AGI/Primus/pull/764) | 2026-06-16 13:57 | Bug Fix | fix(v263): root-fix gfx942 hd128 backward crash in image + CI |
| [#760](https://github.com/AMD-AGI/Primus/pull/760) | 2026-06-16 14:00 | Bug Fix | [projection] add unit tests; fix single-node DP & dense MoE-spec bugs |
| [#765](https://github.com/AMD-AGI/Primus/pull/765) | 2026-06-16 14:01 | CI/Infra | Add core unit tests (pipeline_parallel, utils) + UT coverage in CI |
| [#768](https://github.com/AMD-AGI/Primus/pull/768) | 2026-06-16 15:23 | Bug Fix | fix(megatron): extend aiter DEEPBIND hd128 backward fix to gfx950 |
| [#769](https://github.com/AMD-AGI/Primus/pull/769) | 2026-06-16 19:20 | Refactor | refactor(turbo): rename deprecated turbo flags + hard-assert |
| [#771](https://github.com/AMD-AGI/Primus/pull/771) | 2026-06-17 09:03 | Docs | docs: add CI status badges to README |
| [#772](https://github.com/AMD-AGI/Primus/pull/772) | 2026-06-19 20:23 | Bug Fix | fix(torchtitan): correct MFU on MI350X/MI355X via peak FLOPS patch |
| [#775](https://github.com/AMD-AGI/Primus/pull/775) | 2026-06-23 09:54 | Bug Fix | fix(megatron): don't log from spawned async-checkpoint worker |
| [#782](https://github.com/AMD-AGI/Primus/pull/782) | 2026-06-25 05:56 | CI/Infra | [OOB Release] add primus v26.4 dockerfile |
| [#781](https://github.com/AMD-AGI/Primus/pull/781) | 2026-06-25 15:48 | Bug Fix | fix(runner): honor PRIMUS_LOG_LEVEL in prepare/hook scripts |
| [#783](https://github.com/AMD-AGI/Primus/pull/783) | 2026-06-26 00:51 | CI/Infra | Merge releases/primus-v26.4 back to main |
| [#780](https://github.com/AMD-AGI/Primus/pull/780) | 2026-06-26 02:04 | CI/Infra | ci: supply-chain hardening (pin actions, Dependabot, Scorecard) |
| [#773](https://github.com/AMD-AGI/Primus/pull/773) | 2026-06-26 02:04 | CI/Infra | Add unit-vs-E2E coverage comparison, patch tests, JUnit reports |
| [#800](https://github.com/AMD-AGI/Primus/pull/800) | 2026-06-26 18:38 | CI/Infra | ci: limit Dependabot version updates to GitHub Actions |
| [#826](https://github.com/AMD-AGI/Primus/pull/826) | 2026-06-29 15:41 | Other | chore: update GitHub org refs AMD-AIG-AIMA -> AMD-AGI |
| [#784](https://github.com/AMD-AGI/Primus/pull/784) | 2026-06-30 09:34 | Turbo/Dependency Version Update | Upgrade to FLA 0.5.1 |
| [#799](https://github.com/AMD-AGI/Primus/pull/799) | 2026-06-30 10:15 | Turbo/Dependency Version Update | build(deps): bump mlflow 3.4.0 -> 3.11.1 |
| [#804](https://github.com/AMD-AGI/Primus/pull/804) | 2026-06-30 10:16 | Turbo/Dependency Version Update | build(deps): bump github-actions group (11 updates) |
| [#801](https://github.com/AMD-AGI/Primus/pull/801) | 2026-06-30 10:17 | CI/Infra | Add CPU unit tests for tuning agent/CLI/launcher + benchmark fix |
| [#787](https://github.com/AMD-AGI/Primus/pull/787) | 2026-06-30 10:18 | CI/Infra | Add CPU unit tests for backends and projection |
| [#827](https://github.com/AMD-AGI/Primus/pull/827) | 2026-06-30 10:18 | Turbo/Dependency Version Update | build(deps): bump fastmcp 2.14.0 -> 3.2.0 |
| [#828](https://github.com/AMD-AGI/Primus/pull/828) | 2026-06-30 14:40 | Bug Fix | fix: padding bug and remove use_turbo_permute_padding flag |
| [#802](https://github.com/AMD-AGI/Primus/pull/802) | 2026-07-01 14:05 | CI/Infra | ci: targeted PR tests, runtime summary, config-consistency check |
| [#830](https://github.com/AMD-AGI/Primus/pull/830) | 2026-07-01 17:06 | Turbo/Dependency Version Update | chore: bump primus-turbo version |
| [#836](https://github.com/AMD-AGI/Primus/pull/836) | 2026-07-02 08:19 | Turbo/Dependency Version Update | ci: revert to fla 0.4.x |
| [#806](https://github.com/AMD-AGI/Primus/pull/806) | 2026-07-06 10:07 | Turbo/Dependency Version Update | ci: bump Primus-Turbo/AITER pins for Flux (mxfp4) + skip draft CI |
| [#841](https://github.com/AMD-AGI/Primus/pull/841) | 2026-07-07 09:47 | Other | add mixtral-8x22B config files for maxtext backend |
| [#842](https://github.com/AMD-AGI/Primus/pull/842) | 2026-07-07 09:48 | Other | improve the logging format for the megatron backend |
| [#856](https://github.com/AMD-AGI/Primus/pull/856) | 2026-07-07 16:27 | Other | feat(flux): core runtime + Megatron adapter scaffolding |
| [#858](https://github.com/AMD-AGI/Primus/pull/858) | 2026-07-07 16:27 | Docs | docs(flux): diffusion training documentation |
| [#808](https://github.com/AMD-AGI/Primus/pull/808) | 2026-07-08 08:01 | Other | feat(flux): FSDP2 fp32/bf16 optimizers + fp8 all-gather |
| [#821](https://github.com/AMD-AGI/Primus/pull/821) | 2026-07-08 08:01 | Other | feat(flux): curated diffusion example/model/data configs |
| [#847](https://github.com/AMD-AGI/Primus/pull/847) | 2026-07-08 09:31 | Other | feat(megatron): migrate MLPerf GPT-OSS-20B pretrain trainer |
| [#849](https://github.com/AMD-AGI/Primus/pull/849) | 2026-07-08 09:43 | Other | feat: add AITER_LOG_LEVEL to suppress log |
| [#850](https://github.com/AMD-AGI/Primus/pull/850) | 2026-07-08 09:45 | Bug Fix | [Megatron-LM] fix duplicated memory footprint w/ turbo grouped gemm |
| [#859](https://github.com/AMD-AGI/Primus/pull/859) | 2026-07-08 09:50 | Performance Optimization | opt: remove grouped mlp d2h sync |
| [#779](https://github.com/AMD-AGI/Primus/pull/779) | 2026-07-08 13:27 | Other | Add diffusion backend & Wan training support |
| [#860](https://github.com/AMD-AGI/Primus/pull/860) | 2026-07-08 18:16 | Bug Fix | fix: remove duplicated flag use_turbo_fp4_autocast |
| [#851](https://github.com/AMD-AGI/Primus/pull/851) | 2026-07-08 19:04 | Refactor | refactor: remove primus/modules, migrate code into core/backends |
| [#810](https://github.com/AMD-AGI/Primus/pull/810) | 2026-07-08 23:29 | Other | feat(flux): common diffusion module (embeddings, norm, DiT block) |
| [#809](https://github.com/AMD-AGI/Primus/pull/809) | 2026-07-09 08:39 | Other | feat(flux): Primus-Turbo float8 + local-spec extensions |
| [#867](https://github.com/AMD-AGI/Primus/pull/867) | 2026-07-09 16:48 | Bug Fix | fix(diffusion): repoint module_utils import to primus.core |
| [#811](https://github.com/AMD-AGI/Primus/pull/811) | 2026-07-09 21:37 | Other | feat(flux): Flux DiT model, layers, attention, checkpoint converter |
| [#814](https://github.com/AMD-AGI/Primus/pull/814) | 2026-07-10 15:53 | Other | feat(flux): mxfp4 local-spec extension + fp4 utils/enums |
| [#813](https://github.com/AMD-AGI/Primus/pull/813) | 2026-07-13 15:41 | Other | feat(flux): delayed fp8 scaling + TE DPA prologue patches |
| [#872](https://github.com/AMD-AGI/Primus/pull/872) | 2026-07-14 08:46 | Other | Auto benchmark tool refinement |
| [#815](https://github.com/AMD-AGI/Primus/pull/815) | 2026-07-14 08:51 | Other | feat(flux): torch.compile + DDP-overlap compile patches |
| [#812](https://github.com/AMD-AGI/Primus/pull/812) | 2026-07-14 15:28 | Other | feat(flux): diffusion data pipeline (energon/synthetic, encoders) |
| [#816](https://github.com/AMD-AGI/Primus/pull/816) | 2026-07-14 21:52 | Other | feat(flux): diffusion training primitives (forward step, schedulers) |
| [#824](https://github.com/AMD-AGI/Primus/pull/824) | 2026-07-15 09:22 | Docs | Dev/production doc |
| [#854](https://github.com/AMD-AGI/Primus/pull/854) | 2026-07-15 11:08 | Other | Add MLPerf examples for llama3.1 8b and gpt-oss 20B |
| [#818](https://github.com/AMD-AGI/Primus/pull/818) | 2026-07-15 13:48 | Other | feat(flux): Flux HF->Primus checkpoint conversion tools |
| [#877](https://github.com/AMD-AGI/Primus/pull/877) | 2026-07-15 13:53 | Other | Add MLPerf Training 6.0 Llama2-70B LoRA post-training on MI355X |
| [#817](https://github.com/AMD-AGI/Primus/pull/817) | 2026-07-15 20:16 | Other | feat(flux): diffusion data preprocessing pipelines + data CLI |
| [#705](https://github.com/AMD-AGI/Primus/pull/705) | 2026-07-16 08:30 | Other | feat(runner): add run_preflight_direct.sh for non-container preflight |
| [#869](https://github.com/AMD-AGI/Primus/pull/869) | 2026-07-16 08:31 | Other | feat(maxtext): support MaxText v26.4 with v26.3 back-compat |
| [#870](https://github.com/AMD-AGI/Primus/pull/870) | 2026-07-16 09:01 | Bug Fix | fix(megatron): ROCm-safe attention_backend + Mamba/SFT E2E cleanup |
| [#875](https://github.com/AMD-AGI/Primus/pull/875) | 2026-07-16 09:41 | Other | feat: add moe_router_force_load_balancing_type |
| [#819](https://github.com/AMD-AGI/Primus/pull/819) | 2026-07-16 14:05 | Other | feat(flux): diffusion + Flux pretrain trainers |
| [#832](https://github.com/AMD-AGI/Primus/pull/832) | 2026-07-16 16:43 | Other | Add flux.1 to diffusion backend |
| [#880](https://github.com/AMD-AGI/Primus/pull/880) | 2026-07-17 11:37 | Other | feat: use_turbo_autotune flag + refine force-load-balancing flag |
| [#878](https://github.com/AMD-AGI/Primus/pull/878) | 2026-07-17 11:55 | Performance Optimization | feat: remove extra htod when enable turbo grouped gemm |
| [#879](https://github.com/AMD-AGI/Primus/pull/879) | 2026-07-17 11:57 | Bug Fix | fix(mlperf): dataset prep no longer deletes sibling files |
| [#882](https://github.com/AMD-AGI/Primus/pull/882) | 2026-07-17 16:43 | Other | Add DeepSeek-V4 training support (model, kernels, Muon, FP8/FP4) |
| [#885](https://github.com/AMD-AGI/Primus/pull/885) | 2026-07-17 17:27 | Other | Support Crusoe (Spur) cluster for DeepSeek-V4 multi-node runs |
| [#871](https://github.com/AMD-AGI/Primus/pull/871) | 2026-07-17 17:54 | Turbo/Dependency Version Update | feat(torchtitan): upgrade to v0.2.2 for torch 2.12 + GPT-OSS |
| [#884](https://github.com/AMD-AGI/Primus/pull/884) | 2026-07-17 17:55 | Bug Fix | fix(fsdp2): skip explicit forward prefetch w/ activation recompute |
| [#820](https://github.com/AMD-AGI/Primus/pull/820) | 2026-07-18 08:44 | Other | feat(flux): MLPerf logging/warmup/lr-schedule patches |
| [#864](https://github.com/AMD-AGI/Primus/pull/864) | 2026-07-20 10:52 | Other | feat: odc adapt |
| [#886](https://github.com/AMD-AGI/Primus/pull/886) | 2026-07-20 16:55 | Bug Fix | fix(config): coerce true/false env interpolation to bool in yaml |
| [#887](https://github.com/AMD-AGI/Primus/pull/887) | 2026-07-20 18:08 | Other | Adapt Primus launch to the spur (amd-spur) cluster |
| [#883](https://github.com/AMD-AGI/Primus/pull/883) | 2026-07-21 08:56 | Docs | docs: editorial improvements from user review pass |
| [#892](https://github.com/AMD-AGI/Primus/pull/892) | 2026-07-21 08:56 | Other | Dev/version number update |
| [#822](https://github.com/AMD-AGI/Primus/pull/822) | 2026-07-21 09:13 | CI/Infra | test(flux): diffusion integration tests |
| [#895](https://github.com/AMD-AGI/Primus/pull/895) | 2026-07-21 11:21 | Docs | docs: strip trailing whitespace |
| [#903](https://github.com/AMD-AGI/Primus/pull/903) | 2026-07-22 08:41 | Docs | docs: enable generating llms.txt and llms-full.txt |
| [#904](https://github.com/AMD-AGI/Primus/pull/904) | 2026-07-22 08:43 | Docs | update git clone url in the docs |
| [#906](https://github.com/AMD-AGI/Primus/pull/906) | 2026-07-23 07:06 | Other | Support maxtext v26.5 release |
| [#898](https://github.com/AMD-AGI/Primus/pull/898) | 2026-07-23 16:04 | Other | Update turbo flydsl sparse attn |
| [#907](https://github.com/AMD-AGI/Primus/pull/907) | 2026-07-23 16:23 | Other | [Docker Release] Update MI300X configs for release |
| [#900](https://github.com/AMD-AGI/Primus/pull/900) | 2026-07-23 16:23 | Other | update mi325x config files for primus-v26.5 |
| [#897](https://github.com/AMD-AGI/Primus/pull/897) | 2026-07-23 16:25 | Other | Update batch size for mbridge qwen3-32B on MI300X |
| [#910](https://github.com/AMD-AGI/Primus/pull/910) | 2026-07-23 16:26 | Other | [v25.5 Docker Release] Update configs to enable DeepEP |
| [#912](https://github.com/AMD-AGI/Primus/pull/912) | 2026-07-23 16:28 | Bug Fix | fix(maxtext): handle v26.5 2-value initialize()/run() API |
| [#893](https://github.com/AMD-AGI/Primus/pull/893) | 2026-07-23 16:30 | Other | update MI355 yaml for v26.5 |
| [#868](https://github.com/AMD-AGI/Primus/pull/868) | 2026-07-23 16:51 | Other | feat(skills): port-validation-guide + backend-patch-explorer |
| [#890](https://github.com/AMD-AGI/Primus/pull/890) | 2026-07-23 16:56 | Docs | docs(odc): add examples/odc reproduction guide |
| [#855](https://github.com/AMD-AGI/Primus/pull/855) | 2026-07-23 17:02 | Bug Fix | fix(projection): correct MI355X/gfx950 peak TFLOPS and XCD count |
| [#913](https://github.com/AMD-AGI/Primus/pull/913) | 2026-07-23 17:42 | CI/Infra | ci: chown root-owned E2E leftovers back to runner user |
| [#915](https://github.com/AMD-AGI/Primus/pull/915) | 2026-07-24 07:45 | CI/Infra | [OOB Release] add primus and jax v26.5 dockerfiles |
| [#917](https://github.com/AMD-AGI/Primus/pull/917) | 2026-07-27 11:09 | CI/Infra | fix(ci): stop anchore/sbom-action double-uploading SBOM |
| [#896](https://github.com/AMD-AGI/Primus/pull/896) | 2026-07-27 11:11 | CI/Infra | ci: bump base image/runner to v26.4, drop v26.3 workarounds |
| [#888](https://github.com/AMD-AGI/Primus/pull/888) | 2026-07-27 11:11 | CI/Infra | feat(dashboard): decouple Pages sections + single deployer |
| [#891](https://github.com/AMD-AGI/Primus/pull/891) | 2026-07-28 10:28 | Other | [Megatron-LM] feat: grouped gemm fp4 support, skip cache trans weight |
| [#927](https://github.com/AMD-AGI/Primus/pull/927) | 2026-07-29 13:35 | Other | [Megatron-LM] feat(moe): FlyDSL fused MegaMoE layer (EP-only/TP=1/bf16) |
| [#931](https://github.com/AMD-AGI/Primus/pull/931) | 2026-07-29 14:03 | Bug Fix | fix(runner): pass runner dir to slurm entry so sbatch works on spur |
| [#676](https://github.com/AMD-AGI/Primus/pull/676) | 2026-07-29 15:09 | Other | feat(megatron): Gated Delta Net (GDN) & Kimi Delta Attention (KDA) |
| [#932](https://github.com/AMD-AGI/Primus/pull/932) | 2026-07-30 09:42 | Bug Fix | fix(megatron): honor MXFP4 gradient SR setting |
| [#929](https://github.com/AMD-AGI/Primus/pull/929) | 2026-07-30 10:27 | Docs | Add third-party attribution headers for copied/adapted code |
| [#936](https://github.com/AMD-AGI/Primus/pull/936) | 2026-07-30 12:55 | Turbo/Dependency Version Update | [Hybrid Models] Upgrade FLA version to 0.5.1 |
| [#934](https://github.com/AMD-AGI/Primus/pull/934) | 2026-07-30 16:53 | Other | feat(deepseek-v4): enable fused MegaMoE expert path on DeepSeek-V4 |
| [#921](https://github.com/AMD-AGI/Primus/pull/921) | 2026-07-31 09:26 | Bug Fix | Fix maxtext mixtral sharding to resolve perf drop |
| [#937](https://github.com/AMD-AGI/Primus/pull/937) | 2026-07-31 14:05 | Bug Fix | fix(megatron): patch upstream GatedDeltaNet gate for ROCm NaN |

**Category breakdown (123 PRs):** Bug Fix 25 · Performance Optimization 5 ·
Turbo/Dependency Version Update 11 · CI/Infra 18 · Refactor 2 · Docs 9 ·
Other 53.

## Megatron-LM drift overview

- Submodule: `third_party/Megatron-LM`
- Upstream: `https://github.com/NVIDIA/Megatron-LM.git` (`main`)
- Pinned SHA in Primus `main`: `d3528a21` (2026-03-06) — **unchanged in this
  window** (last bumped pre-window in PR #654, 2026-04-10).
- Upstream `main` HEAD: `df2da78b` (2026-08-03)
- Upstream ahead count: **1050 commits**
- Recommendation: **plan sync**

The Megatron-LM submodule SHA did **not** change inside this window, so no
Megatron backend-gap regeneration was triggered. The upstream gap has, however,
crossed 1000 commits, which is worth planning for.

### Megatron-LM upstream feature delta table

| Area | Notable upstream areas that have moved since the pin |
| --- | --- |
| Megatron-FSDP (MFSDP) | - **A2A / EP overlap**: A2A overlap for Megatron-FSDP<br>- **HybridEP**: high-priority A2A stream + HybridEP preprocessing SMs<br>- **NCCL symmetric memory**: symmetric-memory staging in experimental FSDP<br>- **Compat**: `is_torch_min_version` gating in FSDP source |
| Optimizers | - **CUDA graph**: enable CUDA graph for the ADAM optimizer<br>- **Emerging Optimizers**: integration refactor + more optimizers added<br>- **Muon**: route non-Muon params through `DistributedOptimizer`<br>- **MIMO**: distributed-checkpoint save/load fixes for non-colocated MIMO |
| Checkpoint & inference | - **Async save**: DCP and FSDP async save support<br>- **Inference harness**: inference performance test harness for GPT/hybrid models<br>- **Mamba EP inference**: eager-fallback + mixed CUDA graph test<br>- **Dataset**: inter-document attention masking in `GPTDataset` |
| MoE & precision | - **MoE routing**: routing analysis and metrics capture<br>- **MoE grad**: thread custom process groups through MoE grad finalization<br>- **MXFP8**: configurable fine-grained param gather + asymmetric transpose buffer<br>- **CUDA graph training**: CUDA-graph training-iteration test |

## TorchTitan drift overview

- Submodule: `third_party/torchtitan`
- Upstream: `https://github.com/pytorch/torchtitan.git` (`main`)
- Pinned SHA in Primus `main`: `73a0e697` (**v0.2.2** tag, upstream-dated
  2026-02-20) — **changed inside this window** (bumped from `5fb7cc2e`/v0.1.0 in
  PR #871, merged 2026-07-17).
- Upstream `main` HEAD: `681fd4b5` (2026-08-01)
- Upstream ahead count: **741 commits** (diff `698 files, +123580 / -35965`)
- Recommendation: **plan sync**

Because the torchtitan submodule SHA changed inside the window, the torchtitan
backend-gap report set (`docs/backend-gap/reports/torchtitan/upstream-main/`)
was **regenerated** this run. The bump moved Primus onto the tagged **v0.2.2**
release (a maintained anchor) instead of an unreleased mainline commit, but
upstream `main` has since advanced 741 commits past v0.2.2.

### TorchTitan upstream feature delta table

| Area | Notable upstream areas that have moved since the pin |
| --- | --- |
| Models | - **New families**: Kimi K2 (`kimi_k2_7`, #3532) and Qwen 3.5 added<br>- **GPT-OSS**: GPT-OSS enablement under `spmd_types`<br>- **Reorg**: shared `models/common` layer; `flux` moved into `models/flux`<br>- **DeepSeek-V3**: mxfp8 debug config + compile fixes |
| experiments/ | - **graph_trainer**: GraphPP runner, EP overlap/chunking passes, FSDP collective splitting, dI/dW backward splitting<br>- **rl**: GRPO/DAPO examples, cudagraph knobs, `DPRequestRouter`, entropy metrics<br>- **transformers_modeling_backend**: MoE support (#2679) and SFT (#3243)<br>- **torchft**: fault-tolerant training experiment added |
| distributed/ | - **DeepEP v2**: upgrade to DeepEP v2 APIs enabling cudagraphable mode (#3808)<br>- **minimal_async_ep**: new async-EP kernels (int32 overflow fix)<br>- **New modules**: `compile.py`, `fsdp.py`, `full_dtensor.py`, `spmd_types.py`<br>- **Removed**: legacy `expert_parallel.py` / `dual_pipe_v.py` |
| Precision, checkpoint & CI | - **mxfp8 MoE**: enable `ep=1` (#3935)<br>- **Checkpoint**: remote fsspec paths (e.g. `gs://`) via filesystem helpers (#3887)<br>- **Tokenizer**: support transformers 5.9.0 / hub 1.24.0 downloads<br>- **CI**: AMD 8-GPU-feature CI + AutoParallel device-agnostic tests |

## Primus-Turbo quarterly drift overview

- Component: `third_party/Primus-Turbo` (pinned via CI/benchmark workflow env
  vars, not a git submodule)
- Drift type: current version vs **quarter-start** version on Primus `main`
- `quarter_start_ts`: **2026-07-01 00:00 (GMT+8)** (Q3 anchor)
- CI pin (`.github/workflows/ci.yaml`, `PRIMUS_TURBO_COMMIT`): `3c39ef25` (#355)
  → `9b5d3092` (#429) — **+67 commits**
- Benchmark pin (`.github/workflows/benchmark.yaml`, `PRIMUS_TURBO_COMMIT`):
  `3c39ef25` (#355) → `a04a233c` (#386) — **+35 commits**
- CI pin is **+32 commits ahead** of the benchmark pin.
- AITER pin: `b5e03ed1` (#3070) → `0f3c58e6` (**v0.1.14.post1** tag)
- TRITON pin: `88b227e2` → `09500db9`
- UCCL pin: `5afb4117` (#710) — unchanged
- Recommendation: **monitor**

All Primus-Turbo pin movement landed during this window (early-July Q3 bumps in
PRs #830 and #806). Primus-Turbo has no standalone backend-gap report set, so it
is tracked here in the quarterly-drift section rather than via a backend-gap
regeneration.

### Primus-Turbo quarterly drift table

| Pin | Quarter-start (2026-07-01) | Current | Delta / notable changes |
| --- | --- | --- | --- |
| CI `PRIMUS_TURBO_COMMIT` | `3c39ef25` (#355, moe_permute probs layout) | `9b5d3092` (#429, megakernel gfx950 correctness) | +67 commits |
| Benchmark `PRIMUS_TURBO_COMMIT` | `3c39ef25` (#355) | `a04a233c` (#386, nt-layout bwd GEMM) | +35 commits; CI is +32 ahead of benchmark |
| `PRIMUS_TURBO_AITER_COMMIT` | `b5e03ed1` (#3070) | `0f3c58e6` (v0.1.14.post1) | AITER moved to a tagged release |
| `TRITON_COMMIT` | `88b227e2` | `09500db9` | Triton source pin advanced |

Notable Primus-Turbo changes since quarter start (`3c39ef25` → `9b5d3092`):

- **Grouped/blockwise FP8 GEMM**: MI300/MI355 grouped blockwise FP8 GEMM + HIP fused quant (#358).
- **JAX MoE support**: JAX vmap batching rules for MoE/grouped-GEMM primitives for pipeline-parallel (#350).
- **Attention correctness**: qkv_format stride fixes + torch.compile stride tests (#362).
- **Packaging**: self-hosted PEP 503 release pipeline; AITER made an optional, lazily-imported dependency.

## Source links

- Merged PRs (this window): [Primus PRs merged 2026-06-02 → 2026-08-03](https://github.com/AMD-AGI/Primus/pulls?q=is%3Apr+is%3Amerged+base%3Amain+merged%3A2026-06-02T09%3A53%3A00Z..2026-08-03T09%3A03%3A09Z)
- torchtitan pin bump: [PR #871](https://github.com/AMD-AGI/Primus/pull/871)
- Megatron-LM upstream: [NVIDIA/Megatron-LM `main`](https://github.com/NVIDIA/Megatron-LM/commits/main)
- Megatron-LM pin `d3528a21`: [compare vs upstream main](https://github.com/NVIDIA/Megatron-LM/compare/d3528a21301db2d12e92912b3ec025dc8a2ed4d6...main)
- TorchTitan upstream: [pytorch/torchtitan `main`](https://github.com/pytorch/torchtitan/commits/main)
- TorchTitan pin `73a0e697` (v0.2.2): [compare vs upstream main](https://github.com/pytorch/torchtitan/compare/73a0e6979dd10b6b1904098eb3c8f62c18ab87ce...main)
- Primus-Turbo: [AMD-AGI/Primus-Turbo](https://github.com/AMD-AGI/Primus-Turbo)
- Primus-Turbo CI drift: [compare `3c39ef25...9b5d3092`](https://github.com/AMD-AGI/Primus-Turbo/compare/3c39ef259aa6d724c77c481e926466e7a167e938...9b5d3092efcbc087657b233d8e9ae662cee6ec6b)

---

_Generated 2026-08-03 17:03 (GMT+8). Facts read from `origin/main`, upstream
Git repositories, and the GitHub API. This run is a full replacement of the
earlier `2026-08` report and backfills the June→July coverage gap._
