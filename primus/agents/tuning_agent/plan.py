"""Systematic seed planner.

Before the LLM ever proposes anything, we evaluate a small structured set
of legal configurations so the agent starts informed. The seed strategy
follows the *Optimization Exploration Guidelines* in the Primus Projection
skill (``Primus/.skills/primus-projection.md``) — i.e. high-leverage
optimizations first, qualitative parallelism reshapes last.

Order of returned candidates (small ``--seed-budget`` runs the first N):

  1. **Baseline** — the workload's existing config (anchor / sanity).
  2. **Memory levers** — ``recompute_granularity`` ∈ {none, selective}
     against the baseline parallelism. Workload YAMLs commonly ship
     ``full`` recompute defaults that cost 20–30% throughput when memory
     headroom exists.
  3. **Tier-A MoE comm** (``use_turbo_deepep`` + ``sync_free_stage``)
     — per the skill, DeepEP is "the largest single-optimization
     throughput gain (35%+ for DeepSeek V2)" and SyncFree stage 3 takes
     A2A overlap from 65% → 85%.
  4. **Tier-A precision** (``fp8: hybrid``) — ~2× compute speedup for
     linear layers per the skill.
  5. **Combined best Tier-A** — DeepEP + SyncFree=3 + FP8 + recompute=none
     (matches the skill's Step 8 "Combine best options").
  6. **Pipeline schedule sweep** — Zero-Bubble (VPP=1) and ZBV-formatted
     / greedy-half (VPP=2) per the skill's Step 3.
  7. **Baseline-neighbor (VPP × MBS × recompute) sweep** — the standard
     manual-tuning levers (e.g. published Mixtral 8x22B uses VPP=2 + MBS=1).
  8. **Tier-B/C alternatives** — ``use_distributed_optimizer`` /
     ``use_torch_fsdp2`` (DP-sharding) and ``decoder_first/last_pipeline_num_layers``
     where applicable.
  9. **Coarse parallelism grid** — sparse sweep over (TP, PP, EP, CP) to
     surface qualitatively different parallelisations.

The intent: catch the *highest-impact* wins deterministically (DeepEP, FP8,
recompute) and let the LLM spend its budget on the harder polish.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass

from .config import AgentConfig
from .legality import AxisLegality, TrialConfig, derive_legality, validate, fill_defaults_from_baseline
from .workload import ArchitectureRecord


@dataclass
class SeedPlan:
    candidates: list[TrialConfig]
    rationale: str


def build_seed_plan(arch: ArchitectureRecord, agent_cfg: AgentConfig,
                    max_candidates: int = 12) -> SeedPlan:
    """Generate the deterministic seed plan.

    Order of returned candidates (small ``--seed-budget`` runs the first N):

      1. **Baseline** — the workload's existing config (known-working anchor).
      2. **Baseline neighbors** — the same (TP, PP, EP, CP) as the baseline,
         varying only the cheap memory levers (VPP, MBS, recompute). These
         are the most common manual tuning hits (e.g. published Mixtral 8x22B
         uses baseline + VPP=2 + MBS=1) and *cannot be missed* by the coarse
         grid. Sweeping ``recompute_granularity ∈ {none, selective, full}``
         lets the agent exploit HBM headroom — workload YAMLs commonly ship
         full-recompute defaults that cost ~30% throughput when not needed.
      3. **Coarse grid** — a sparse sweep over (TP, PP, EP, CP, MBS, VPP)
         to surface qualitatively different parallelisations.

    Each coarse axis is anchored on the baseline value so configurations like
    "PP=4 (=baseline)" are never dropped by the 3-point reduction.
    """
    cluster = agent_cfg.target_cluster
    legality = derive_legality(arch, cluster)
    world_size = cluster.num_nodes * cluster.gpus_per_node

    # ── coarse axis reduction with baseline anchoring ─────────────────────
    def _coarse(values: list[int], cap: int, anchor: int | None = None) -> list[int]:
        """Pick ≤4 representative values <= cap, ALWAYS keeping ``anchor``.

        Failing to keep the workload's own value here is the #1 bug we hit
        in practice — e.g. on 8×MI355X for Mixtral 8x22B, ``legality.pp =
        [1, 2, 4, 7, 8, 14, 28, 56]`` collapsed to first/middle/last
        ``[1, 8, 56]`` and the baseline PP=4 was unreachable from the grid.
        """
        kept = sorted({v for v in values if v <= cap})
        if not kept:
            kept = sorted(set(values))
        if anchor is not None and anchor in kept:
            picks: set[int] = {anchor}
        else:
            picks = set()
        if len(kept) <= 4:
            picks.update(kept)
        else:
            picks.update([kept[0], kept[len(kept) // 2], kept[-1]])
        # Always include the smallest legal value so we have a "no comm"
        # reference point.
        picks.add(kept[0])
        return sorted(picks)

    tp_set = _coarse(legality.tp, cluster.gpus_per_node, anchor=arch.tensor_model_parallel_size)
    # Pipeline parallel can legitimately span the whole cluster.
    pp_set = _coarse(legality.pp, world_size, anchor=arch.pipeline_model_parallel_size)
    # Context parallel only helps when sequences are long enough that the
    # per-rank attention payload offsets the all-gather/ring overhead. The
    # rule of thumb (cf. Primus Projection skill, Megatron CP guidance) is
    # ``seq_length > ~32k``; for shorter sequences CP just shrinks the
    # data-parallel dim and adds comm. Workloads that *baseline* with CP>1
    # are honored as-is so the agent can still tune them.
    CP_SEQ_THRESHOLD = 32 * 1024
    seq_len = int(arch.seq_length or 0)
    cp_baseline = int(arch.context_parallel_size or 1)
    cp_allowed = (seq_len > CP_SEQ_THRESHOLD) or (cp_baseline > 1)

    if arch.is_moe:
        ep_set = _coarse(legality.ep, cluster.gpus_per_node, anchor=arch.expert_model_parallel_size)
        if cp_allowed and cluster.num_nodes > 1:
            cp_set = _coarse(legality.cp, cluster.gpus_per_node, anchor=cp_baseline)
        else:
            cp_set = [cp_baseline] if cp_baseline in legality.cp else [1]
    else:
        ep_set = [1]
        if cp_allowed:
            cp_set = _coarse(legality.cp, cluster.gpus_per_node, anchor=cp_baseline)
        else:
            cp_set = [cp_baseline] if cp_baseline in legality.cp else [1]

    # MBS: baseline ± halved/doubled. The cheapest memory & GEMM lever; many
    # workload YAMLs ship a conservative MBS that's not optimal for tps.
    base_mbs = arch.micro_batch_size or 1
    mbs_set = sorted({1, max(1, base_mbs // 2), base_mbs, base_mbs * 2})

    # VPP only matters when PP>1. ``legality.vpp`` already returns the union
    # over all legal PP; per-trial ``validate()`` rejects (PP, VPP) combos
    # where VPP doesn't divide num_layers/PP.
    vpp_set: list = list(legality.vpp) if legality.vpp else [None]

    def _recompute_layers(pp: int) -> int:
        """Pick a sensible ``recompute_num_layers`` for full-stage recompute.

        Resolution order (highest priority first):
          1. ``len(recompute_layer_ids)`` from the workload yaml — this is
             what the published DeepSeek V3 reference uses (recomputes a
             specific 21-layer subset; the agent expresses the same total
             via ``recompute_num_layers`` since ``recompute_method=block``
             only takes a count, not a list).
          2. ``recompute_num_layers`` from the workload yaml (when > 0).
          3. ``num_layers // pp`` — fallback to "every layer in the stage".
        """
        layer_id_count = len(arch.recompute_layer_ids or [])
        existing = int(arch.recompute_num_layers or 0)
        if (arch.recompute_granularity or "") != "full":
            return layer_id_count or existing
        if not arch.num_layers or pp <= 0:
            return layer_id_count or existing or 1
        stage_layers = max(1, arch.num_layers // pp)
        return max(layer_id_count, existing, stage_layers)

    _SENTINEL = object()  # sentinel meaning "use the workload's value"

    def _make(tp: int, pp: int, ep: int, cp: int, mbs: int,
              vpp: int | None,
              *,
              recompute=_SENTINEL,
              pp_schedule: str = "auto",
              enable_zero_bubble: bool | None = _SENTINEL,
              use_turbo_deepep: bool | None = _SENTINEL,
              sync_free_stage: int | None = _SENTINEL,
              fp8: str | None = _SENTINEL,
              cross_entropy_loss_fusion: bool | None = _SENTINEL,
              use_torch_fsdp2: bool | None = _SENTINEL,
              use_distributed_optimizer: bool | None = _SENTINEL,
              target_ep_size: int | None = _SENTINEL,
              ) -> TrialConfig | None:
        """Build a TrialConfig, fill defaults, validate; return None if illegal.

        Each Tier-A/B knob (DeepEP, SyncFree, FP8, FSDP2, etc.) defaults to
        ``_SENTINEL`` meaning "leave at None / inherit yaml". Pass an
        explicit value to enable that override for this seed.
        """
        rec = arch.recompute_granularity if recompute is _SENTINEL else recompute
        rcl = 0 if rec in (None, "none") else _recompute_layers(pp)

        def _resolve(v, default=None):
            return default if v is _SENTINEL else v

        cfg = TrialConfig(
            tp=tp, pp=pp, ep=ep, cp=cp,
            mbs=mbs,
            gbs=arch.global_batch_size or 1,
            vpp=vpp,
            pp_schedule=pp_schedule,
            enable_zero_bubble=_resolve(enable_zero_bubble),
            recompute_granularity=rec,
            recompute_num_layers=rcl,
            cross_entropy_loss_fusion=_resolve(cross_entropy_loss_fusion),
            use_torch_fsdp2=_resolve(use_torch_fsdp2),
            use_distributed_optimizer=_resolve(use_distributed_optimizer),
            use_turbo_deepep=_resolve(use_turbo_deepep),
            sync_free_stage=_resolve(sync_free_stage),
            target_ep_size=_resolve(target_ep_size),
            fp8=_resolve(fp8),
            overlap_grad_reduce=arch.overlap_grad_reduce,
        )
        cfg = fill_defaults_from_baseline(cfg, arch)
        ok, _ = validate(cfg, arch, cluster, legality)
        return cfg if ok else None

    baseline = TrialConfig(
        tp=arch.tensor_model_parallel_size,
        pp=arch.pipeline_model_parallel_size,
        ep=arch.expert_model_parallel_size,
        cp=arch.context_parallel_size,
        mbs=arch.micro_batch_size,
        gbs=arch.global_batch_size,
        vpp=arch.virtual_pipeline_model_parallel_size,
        pp_schedule="auto",
        recompute_granularity=arch.recompute_granularity,
        recompute_num_layers=_recompute_layers(arch.pipeline_model_parallel_size),
        overlap_grad_reduce=arch.overlap_grad_reduce,
    )

    candidates: list[TrialConfig] = [baseline]
    seen: set[str] = {baseline.signature()}

    def _push(cfg: TrialConfig | None) -> None:
        if cfg is None:
            return
        sig = cfg.signature()
        if sig in seen:
            return
        seen.add(sig)
        candidates.append(cfg)

    def _full() -> bool:
        return len(candidates) >= max_candidates

    bp = baseline.pp
    bep = baseline.ep
    bcp = baseline.cp
    btp = baseline.tp
    bmbs = baseline.mbs
    bvpp = baseline.vpp

    # Recompute strategies, ordered by expected throughput (fastest first):
    # "none" > "selective" > "full". The memory projection rejects ones that
    # don't fit; this gives the seed phase a chance to find configurations
    # that exploit available HBM headroom (the dominant gap vs published
    # benchmarks where memory headroom is large but the workload yaml ships
    # full-recompute defaults).
    recompute_strategies: list[str | None]
    if arch.recompute_granularity in ("full", "selective"):
        recompute_strategies = ["none", "selective", arch.recompute_granularity]
    else:
        recompute_strategies = [arch.recompute_granularity, "none"]
    seen_rc: set = set()
    recompute_strategies = [r for r in recompute_strategies
                            if not (r in seen_rc or seen_rc.add(r))]

    # ── 2. Memory levers at baseline parallelism ──────────────────────────
    # Sweep recompute first: it's the cheapest way to recover throughput
    # when the workload yaml ships ``full`` defaults but HBM has headroom.
    for rec in recompute_strategies:
        _push(_make(btp, bp, bep, bcp, bmbs, bvpp, recompute=rec))
        if _full():
            break

    # ── 2b. Layout-aware PP × VPP exploration ────────────────────────────
    # For workloads whose num_layers is non-divisor-friendly (DeepSeek V3 =
    # 61, prime), the Primus runtime distributes layers via an explicit
    # ``pipeline_model_parallel_layout`` string. The published DSv3 ref uses
    # PP=8 / VPP=2 (16 stages of 3-4 layers each) — *not* PP=16. Without
    # this step the agent would never see the published reference because
    # PP=8 isn't a divisor of 61. ``write_trial_yaml`` regenerates a layout
    # for the new (PP, VPP) so this is fully representable.
    #
    # We seed these *before* the Tier-A comm/precision sweeps because they
    # change the parallelism shape — a different shape can dramatically
    # alter what's optimal for memory + comm overlap, and they're the
    # observed delta vs the published reference for non-divisor models.
    # Within layout-aware combos we prefer larger PP first (closer to the
    # workload's total stages = balanced compute/comm) and VPP=2 over
    # VPP=1 (matches DSv3's published reference of 16 stages).
    if not _full() and arch.pipeline_model_parallel_layout and arch.num_layers:
        candidates_pp_vpp: list[tuple] = []
        for cand_pp in (2, 4, 8):
            if cand_pp not in legality.pp:
                continue
            if cand_pp == bp:  # baseline already covered
                continue
            for cand_vpp in (2, 1):
                if cand_vpp not in legality.vpp:
                    continue
                if cand_pp * cand_vpp > arch.num_layers:
                    continue
                # Sort key: lexicographic on (-pp, -vpp) so the order is
                # PP=8/VPP=2, PP=8/VPP=1, PP=4/VPP=2, PP=4/VPP=1, ...
                candidates_pp_vpp.append((-cand_pp, -cand_vpp, cand_pp, cand_vpp))
        candidates_pp_vpp.sort()
        for _k1, _k2, cand_pp, cand_vpp in candidates_pp_vpp:
            if _full():
                break
            # Primary: match the workload's recompute knob (full + layer-id
            # count) to mirror the published reference as closely as possible.
            _push(_make(btp, cand_pp, bep, bcp, bmbs, cand_vpp,
                        recompute=arch.recompute_granularity or "full"))

        # Secondary: for the most-impactful (PP, VPP) pairs (those that match
        # the workload's total stage count, e.g. PP=8/VPP=2 ≡ PP=16/VPP=1
        # mapping for DSv3), also probe MBS=1 and selective recompute. These
        # are the configurations most likely to fit when the primary
        # recompute=full + bmbs combination either OOMs or hits a sub-node
        # benchmark crash (observed for v26.2 on PP=8/VPP=2 + FP8 hybrid).
        for _k1, _k2, cand_pp, cand_vpp in candidates_pp_vpp:
            if _full():
                break
            # Only the PP=8 / VPP=2 mirror gets the deeper variant probe;
            # smaller PP (4, 2) is unlikely to outperform PP=8 once memory
            # fits, so we don't burn budget on their variants here.
            if cand_pp != 8:
                continue
            # MBS=1 with same recompute=full — halves activation per stage,
            # often dodges the v26.2 FP8 sub-node bench crash.
            if 1 != bmbs:
                _push(_make(btp, cand_pp, bep, bcp, 1, cand_vpp,
                            recompute=arch.recompute_granularity or "full"))
                if _full():
                    break
            # Selective recompute (saves attention activations only — ~45%
            # cheaper than full) — common winner when HBM has headroom.
            _push(_make(btp, cand_pp, bep, bcp, bmbs, cand_vpp,
                        recompute="selective"))

    # ── 3. Tier-A MoE communication (DeepEP + SyncFree) ───────────────────
    # Per the skill, DeepEP is "the largest single-optimization throughput
    # gain (35%+ for DeepSeek V2)". SyncFree stages 2/3 auto-enable DeepEP
    # per `projection performance --sync-free-stage` semantics.
    #
    # We test standalone DeepEP first, then DeepEP stacked with the *known
    # winner* (recompute=none), to give the agent both an isolated DeepEP
    # measurement AND the most promising combination. SyncFree=2/3 require
    # `moe_use_legacy_grouped_gemm=True` which depends on Megatron-LM
    # exposing `TEGroupedMLPSubmodules` — that's not available in every
    # build, so we keep these as last-priority probes that the runtime can
    # cheaply reject.
    deepep_already_on = bool(arch.use_turbo_deepep)
    sync_free_already = int(arch.sync_free_stage or 0)
    fp8_already_on = arch.fp8 not in (None, "", "null", False) or arch.precision == "fp8"
    if arch.is_moe and not _full():
        moe_combos: list[tuple[str, dict]] = []
        if not deepep_already_on:
            # Standalone DeepEP first (clean isolation of the +35% per-skill).
            moe_combos.append(("DeepEP", {"use_turbo_deepep": True}))
            # Then DeepEP + recompute=none — the highest-leverage achievable
            # combo on environments without TEGroupedMLP.
            moe_combos.append(
                ("DeepEP+recompute=none",
                 {"use_turbo_deepep": True, "recompute": "none"}))
        for _name, kw in moe_combos:
            _push(_make(btp, bp, bep, bcp, bmbs, bvpp, **kw))
            if _full():
                break

    # ── 4. Tier-A precision (FP8) ─────────────────────────────────────────
    # ~2× compute speedup for linear-layer GEMMs per the skill. Test
    # standalone first, then stacked with recompute=none and DeepEP.
    if not _full() and not fp8_already_on:
        _push(_make(btp, bp, bep, bcp, bmbs, bvpp, fp8="hybrid"))
        if not _full():
            _push(_make(btp, bp, bep, bcp, bmbs, bvpp,
                        fp8="hybrid", recompute="none"))
        if not _full() and arch.is_moe and not deepep_already_on:
            _push(_make(btp, bp, bep, bcp, bmbs, bvpp,
                        fp8="hybrid", use_turbo_deepep=True,
                        recompute="none"))

    # ── 5. Combined-best Tier-A (DeepEP + SyncFree=3 + FP8 + recompute=none)
    # SyncFree=3 is best-case overlap per the skill but requires the
    # `moe_use_legacy_grouped_gemm=True` runtime path; we add it last so
    # earlier (achievable) combos are evaluated first. The required
    # couplings (`moe_router_dtype=fp32`, `moe_use_legacy_grouped_gemm=True`)
    # are auto-applied by `evaluator.write_trial_yaml`.
    if not _full() and arch.is_moe and sync_free_already < 3:
        combined_kw: dict = {"recompute": "none", "sync_free_stage": 3}
        if not deepep_already_on:
            combined_kw["use_turbo_deepep"] = True
        if not fp8_already_on:
            combined_kw["fp8"] = "hybrid"
        _push(_make(btp, bp, bep, bcp, bmbs, bvpp, **combined_kw))

    # SyncFree=2 as a separate fallback (smaller blast radius if SF=3 fails).
    if not _full() and arch.is_moe and sync_free_already < 2:
        sf2_kw: dict = {"recompute": "none", "sync_free_stage": 2}
        if not deepep_already_on:
            sf2_kw["use_turbo_deepep"] = True
        if not fp8_already_on:
            sf2_kw["fp8"] = "hybrid"
        _push(_make(btp, bp, bep, bcp, bmbs, bvpp, **sf2_kw))

    # ── 6. Pipeline-schedule sweep ────────────────────────────────────────
    # ZB and ZBV variants reduce pipeline bubble. ZBV-formatted typically
    # wins for MoE with VPP=2; ZB (VPP=1) is the dense default. The
    # skill's "compare every algorithm" pattern.
    #
    # We exercise schedules at *two* pipeline points:
    #   (a) the baseline PP (when baseline already uses pipelines), and
    #   (b) a "natural" PP for large dense models (e.g. Llama-70B with
    #       baseline PP=1) that have to break into stages just to fit
    #       memory. This catches workloads where the YAML ships PP=1 but
    #       the cluster math forces PP>=4.
    schedule_pp_targets: list[int] = []
    if bp > 1:
        schedule_pp_targets.append(bp)
    # For layout-aware workloads, also schedule-sweep at the most-impactful
    # alternative PP depth (PP=8 for DSv3) — historical agent runs showed
    # ``PP=16 + zerobubble`` was the previous best (1045 tps) by reducing
    # pipeline bubble; the same trick at PP=8 can offset the slightly
    # higher per-stage compute cost from running fewer pipeline ranks.
    if (arch.pipeline_model_parallel_layout and arch.num_layers and 8 in legality.pp
            and 8 != bp and 8 not in schedule_pp_targets):
        schedule_pp_targets.append(8)
    # Heuristic: when baseline PP=1 but model is large enough that DP-only
    # is unlikely to fit, also try a sensible PP (largest legal PP that is
    # ≤ gpus_per_node and divides num_layers). Common choice: 4 or 8.
    if bp == 1 and arch.num_layers:
        for cand in (8, 4, 2):
            if cand in legality.pp and arch.num_layers % cand == 0 and cand not in schedule_pp_targets:
                schedule_pp_targets.append(cand)
                break
    # Some workloads (notably DSv3 with 61 layers) encode interleaved staging
    # via ``pipeline_model_parallel_layout`` where the raw ``num_layers // pp``
    # parity check is misleading. Example: PP=8 with a 16-stage layout implies
    # VPP=2 and supports ZBV even though ``61 // 8`` is odd.
    layout_stage_count: int | None = None
    if arch.pipeline_model_parallel_layout:
        normalized = str(arch.pipeline_model_parallel_layout).strip()
        if len(normalized) >= 2 and normalized[0] == normalized[-1] and normalized[0] in ("'", '"'):
            normalized = normalized[1:-1].strip()
        n = sum(1 for part in normalized.split("|") if part.strip())
        if n > 0:
            layout_stage_count = n

    def _layout_implies_vpp(pp: int, vpp: int) -> bool:
        if not layout_stage_count or pp <= 0:
            return False
        return layout_stage_count % pp == 0 and (layout_stage_count // pp) == vpp

    for spp in schedule_pp_targets:
        if _full():
            break
        # VPP=1 candidates: zero-bubble + heuristic.
        for sched in ("zerobubble", "zerobubble-heuristic"):
            if sched in legality.pp_schedules_by_vpp.get(1, []):
                # Split-wgrad schedules cannot pair with activation recompute
                # (projection assert + runtime wgrad cache pins inputs).
                _push(_make(btp, spp, bep, bcp, bmbs, 1,
                            pp_schedule=sched, enable_zero_bubble=True,
                            recompute="none"))
                if _full():
                    break
        # VPP=2 candidates: ZBV variants.
        #
        # Historically we gated this by ``(num_layers // pp) % 2 == 0``.
        # That misses valid layout-driven interleaving cases (DSv3 PP=8 with
        # 16-stage layout). Allow ZBV when either check passes.
        can_try_zbv = (
            not _full()
            and (
                (arch.num_layers and (arch.num_layers // spp) % 2 == 0)
                or _layout_implies_vpp(spp, 2)
            )
        )
        if can_try_zbv:
            for sched in ("zbv-formatted", "zbv-greedy-half"):
                if sched in legality.pp_schedules_by_vpp.get(2, []):
                    _push(_make(btp, spp, bep, bcp, bmbs, 2,
                                pp_schedule=sched, recompute="none"))
                    if _full():
                        break

    # ── 7. Baseline-neighbor sweep over (VPP, MBS, recompute) ─────────────
    # The standard manual-tuning levers; complements the Tier-A wins above.
    bvpp_options: list = []
    if arch.num_layers and bp:
        stage_layers = max(1, arch.num_layers // bp)
        for v in (None, *legality.vpp):
            if v is None:
                bvpp_options.append(None)
            elif v == 1 or stage_layers % v == 0:
                bvpp_options.append(v)
    else:
        bvpp_options = [None, 1]
    seen_vpp: set = set()
    bvpp_options = [v for v in bvpp_options if not (v in seen_vpp or seen_vpp.add(v))]

    if not _full():
        for vpp in bvpp_options:
            mbs_neighbors: list[int] = []
            for m in (bmbs, max(1, bmbs // 2), 1, bmbs * 2):
                if m not in mbs_neighbors:
                    mbs_neighbors.append(m)
            for rec in recompute_strategies:
                for mbs in mbs_neighbors:
                    _push(_make(btp, bp, bep, bcp, mbs, vpp, recompute=rec))
                    if _full():
                        break
                if _full():
                    break
            if _full():
                break

    # ── 8. Tier-B/C DP-sharding alternatives ──────────────────────────────
    # Distributed optimizer (ZeRO-1-like) is the cheap memory win; FSDP2 is
    # heavier but unlocks more headroom for very large models. Both are
    # alternatives — `validate()` already rejects enabling them simultaneously.
    if not _full():
        _push(_make(btp, bp, bep, bcp, bmbs, bvpp,
                    use_distributed_optimizer=True))
    if not _full():
        # FSDP2 with no recompute: tests whether the optimizer/grad sharding
        # buys enough memory headroom to drop recompute entirely.
        _push(_make(btp, bp, bep, bcp, bmbs, bvpp,
                    recompute="none",
                    use_torch_fsdp2=True))

    # ── 9. Coarse parallelism grid ────────────────────────────────────────
    if not _full():
        for tp, pp, ep, cp, mbs, vpp in itertools.product(
                tp_set, pp_set, ep_set, cp_set, mbs_set, vpp_set):
            if pp == 1 and vpp not in (None, 1):
                continue
            _push(_make(tp, pp, ep, cp, mbs, vpp))
            if _full():
                break

    rationale = (
        f"Baseline (TP={baseline.tp}, PP={baseline.pp}, EP={baseline.ep}, "
        f"CP={baseline.cp}, MBS={baseline.mbs}, VPP={baseline.vpp}) "
        f"-> recompute sweep ({recompute_strategies}) "
        f"-> Tier-A MoE comm (DeepEP, SyncFree=2/3) "
        f"-> FP8 hybrid -> combined-best (DeepEP+SyncFree=3+FP8+recompute=none) "
        f"-> pipeline-schedule sweep (zerobubble, zbv-formatted, zbv-greedy-half) "
        f"-> baseline-neighbor (vpp={bvpp_options}, mbs neighbors) "
        f"-> distributed_optimizer / FSDP2 "
        f"-> coarse grid TP×PP×EP×CP "
        f"(tp={tp_set}, pp={pp_set}, ep={ep_set}, cp={cp_set}"
        + (f", CP gated by seq_length={seq_len}<={CP_SEQ_THRESHOLD}" if not cp_allowed else "")
        + "). "
        f"Generated {len(candidates)} legal seeds. Order follows the Primus "
        f"Projection skill's optimization priority (DeepEP > SyncFree > FP8 "
        f"> recompute > schedule)."
    )
    return SeedPlan(candidates=candidates, rationale=rationale)
