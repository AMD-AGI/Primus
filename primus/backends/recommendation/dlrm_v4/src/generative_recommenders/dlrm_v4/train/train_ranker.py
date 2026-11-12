# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# pyre-strict
import argparse
import logging

logging.basicConfig(level=logging.INFO)
import os
import sys
import time
import traceback

import gin
import torch
from torch import multiprocessing as mp
from torchrec.test_utils import get_free_port

# NOTE: heavy imports of generative_recommenders.dlrm_v4.* are deferred to
# inside _main_func so that gin-driven env-var bootstrap (see
# _env_bootstrap.apply_env_bootstrap) can run BEFORE the triton kernel
# modules evaluate their `@triton.autotune` decorators at module-load time.

logger: logging.Logger = logging.getLogger(__name__)


SUPPORTED_CONFIGS = {
    "debug": "debug.gin",
    "kuairand-1k": "kuairand_1k.gin",
    "movielens-1m": "movielens_1m.gin",
    "movielens-20m": "movielens_20m.gin",
    "movielens-13b": "movielens_13b.gin",
    "movielens-18b": "movielens_18b.gin",
    "streaming-400m": "streaming_400m.gin",
    "streaming-200b": "streaming_200b.gin",
    "streaming-100b": "streaming_100b.gin",
    "yambda-5b": "yambda_5b.gin",
}


def _main_func(
    local_rank: int,
    world_size: int,
    node_rank: int,
    gpus_per_node: int,
    master_addr: str,
    master_port: int,
    gin_file: str,
    mode: str,
) -> None:
    # `local_rank` is the index handed out by mp.start_processes (0..gpus_per_node-1)
    # and indexes this node's GPUs. The GLOBAL rank is what every downstream
    # consumer wants (data sharding via StreamingWindowSampler, checkpoint I/O,
    # metrics), so derive it once and pass it through as `rank`. Only the CUDA
    # device must be node-local. Single-node (node_rank=0) → rank == local_rank,
    # exactly as before.
    rank = node_rank * gpus_per_node + local_rank
    device = torch.device(f"cuda:{local_rank}")
    logger.info(
        f"rank: {rank} (node_rank={node_rank} local_rank={local_rank}), "
        f"world_size: {world_size}, device: {device}"
    )
    # Ship-12 item (34) persistent kernel cache (scripts/launch_slurm.sh
    # KCACHE_DIR, default off): the launcher seeded this container's Triton
    # cache before the trainer started and exported what it seeded (unset =
    # cache off). Rank 0 reports it in the trainer log.
    _kcache = os.environ.get("KCACHE_SEEDED", "")
    if rank == 0 and _kcache:
        _kc_entries = 0
        for _kc_field in _kcache.split():
            if _kc_field.startswith("entries="):
                try:
                    _kc_entries = int(_kc_field[len("entries="):])
                except ValueError:
                    _kc_entries = 0
        logger.info(
            "%s [kcache] seeded %s TRITON_CACHE_DIR=%s",
            "armed OK:" if _kc_entries > 0 else "NOT armed:",
            _kcache,
            os.environ.get("TRITON_CACHE_DIR", "<unset>"),
        )
    # Phase 1: parse gin early with skip_unknown=True so env-bootstrap
    # bindings take effect BEFORE any module-level @gin.configurable
    # discovers itself. This is required because triton @triton.autotune
    # decorators in generative_recommenders.ops.triton.* read env vars at
    # module import time, and the heavy imports below pull those in.
    from generative_recommenders.dlrm_v4.train._env_bootstrap import apply_env_bootstrap
    from generative_recommenders.dlrm_v4.train.mlperf_logging_utils import (
        get_mlperf_logger,
        mlperf_checkpoint_present,
    )

    gin.parse_config_file(gin_file, skip_unknown=True)
    apply_env_bootstrap()

    # SHIP-13 CANDIDATE (dlstart) TSPEC, dlrm_v4/train/triton_spec.py: the
    # recorder (TRITON_SPEC_RECORD=<dir>, non-scored record runs) and the
    # post-RUN_START compile monitor (TRITON_SPEC_MONITOR=1) wrap
    # JITFunction._do_compile at class level. Installed here, right after the env
    # bootstrap (so they are gin-bindable like the other TRITON_* knobs) and
    # before the Phase-2 imports below: no kernel module is imported yet, so
    # nothing has compiled. Both off = no triton import, no output. Never raises
    # (a failure logs "[triton-spec] NOT armed" and the run continues).
    from generative_recommenders.dlrm_v4.train import triton_spec as _tspec

    _tspec.install_from_env(rank=rank, local_rank=local_rank, world_size=world_size)

    # Cold-start vs resume, decided from the on-disk checkpoint BEFORE setup so
    # the one-time INIT/RUN markers fire on a genuine cold start only and are NOT
    # re-emitted on a resume relaunch — the MLPerf run (run_start..run_stop) spans
    # the resume as a single coherent event stream in one appended log file.
    mlperf_resume = mlperf_checkpoint_present(os.environ.get("CKPT_PATH", ""))
    # Rank-0-gated MLPerf logger, only for the streaming-train-eval path. `fresh`
    # truncates the log on cold start (one run_start per file) but appends on a
    # resume so the pre-crash events are preserved and continued.
    mlperf_logger = (
        get_mlperf_logger(rank=rank, fresh=not mlperf_resume)
        if mode == "streaming-train-eval"
        else None
    )
    # INIT_START fires before setup on a cold start only (resume continues the
    # already-open run, whose markers were emitted by the original process).
    mlperf_cold_start = mlperf_logger is not None and not mlperf_resume
    if mlperf_cold_start:
        # cache_clear reports whether the launcher really flushed the page cache
        # (scripts/launch_slurm.sh, CLEAR_CACHES); the checker accepts any value.
        cache_cleared = os.environ.get("MLPERF_CACHE_CLEARED", "0") == "1"
        if not cache_cleared:
            logger.warning(
                "MLPerf cache_clear=False: the page cache was not flushed before "
                "this run (launch with CLEAR_CACHES=1); not a compliant timed run"
            )
        mlperf_logger.event(
            key=mlperf_logger.constants.CACHE_CLEAR, value=cache_cleared
        )
        mlperf_logger.start(key=mlperf_logger.constants.INIT_START)

    # Phase 2: heavy imports. Triton kernel modules evaluate their autotune
    # decorators here, using the env vars set above.
    from generative_recommenders.dlrm_v4.checkpoint import load_dmp_checkpoint
    from generative_recommenders.dlrm_v4.train.utils import (
        cleanup,
        decorrelate_runtime_rng,
        eval_loop,
        make_model,
        make_optimizer_and_shard,
        make_train_test_dataloaders,
        seed_everything,
        setup,
        streaming_train_eval_loop,
        train_eval_loop,
        train_loop,
    )
    from generative_recommenders.dlrm_v4.utils import (
        MetricsLogger,
        get_gpu_peak_flops,
    )
    # Imported here (not lazily) so the Phase-3 strict gin parse sees the
    # run_blas_warmup configurable bound in yambda_5b.gin ($BLAS_WARMUP).
    from generative_recommenders.dlrm_v4.train.blas_warmup import (
        run_blas_warmup,
        run_startup_preload,
    )
    # FP8 first-use warmup (gin run_fp8_first_use_warmup.enabled = $FP8_WARMUP,
    # ship-12 item (35)); imported here for the same strict-parse reason.
    from generative_recommenders.dlrm_v4.train.fp8_warmup import (
        run_fp8_first_use_warmup,
    )
    # Robust Triton autotune pins (gin apply_env_bootstrap.DLRM_HSTU_FP8_AUTOTUNE_PINS,
    # default on; opt out with DLRM_HSTU_FP8_AUTOTUNE_PINS=0). Seeds the
    # Autotuner caches of the kernel modules imported above before any launch.
    from generative_recommenders.ops.triton.autotune_pin_table import (
        apply_autotune_pins,
    )

    apply_autotune_pins()

    # Ship6 kernel switches that have no launch-time arming line of their own:
    # report the flags the loaded modules actually use (one line per rank).
    # wln_bwd_flydsl (DLRM_HSTU_FP8_WLN_BWD_FLYDSL) is the requested switch only; which variants
    # actually run FlyDSL is in its per-variant "[ln-bwd]" lines at first use.
    from generative_recommenders.modules import stu as _stu
    from generative_recommenders.ops import position as _position
    from generative_recommenders.ops.triton import (
        triton_hstu_linear as _thl,
        triton_hstu_preprocess_and_attention as _thpa,
        triton_layer_norm as _tln,
    )

    logger.info(
        "armed OK: ship6 kernel switches: wln_bwd_v2=%d fast_sigmoid=%d "
        "lnmd_bwd_v2=%d save_normed_x=%d save_targets_vk=%d posenc_fast=%d "
        "wln_bwd_flydsl=%d",
        int(_tln._wln_bwd_v2_enabled()),
        int(_tln._wln_bwd_v2_fast_sigmoid()),
        int(_thl._use_lnmd_bwd_v2()),
        int(_thpa._SAVE_BF16_NORMED_X),
        int(_stu._SAVE_TARGETS_VK),
        int(_position._POSENC_FAST),
        int(_tln._wln_bwd_flydsl_enabled()),
    )
    # Item (20) R16 LayerNorm-family folds (yambda_5b.gin): the flags as the
    # loaded modules read them. posenc_concat_fold acts only with posenc_fast,
    # dropout_mscale_fold only with wln_bwd_v2 (each also logs its own ACTIVE
    # line at first use).
    from generative_recommenders.ops import input_dropout_fold as _idf

    logger.info(
        "armed OK: r16 LN-fold switches: wln_fwd_v2=%d lnmd_fwd_v2=%d "
        "posenc_concat_fold=%d dropout_mscale_fold=%d tgt_sparse_resid=%d",
        int(_tln._wln_fwd_v2_enabled()),
        int(os.environ.get("DLRM_HSTU_FP8_LNMD_FWD_V2", "0") != "0"),
        int(_position._POSENC_CONCAT_FOLD),
        int(_idf.input_dropout_fold_enabled()),
        int(_stu._TGT_SPARSE_RESID),
    )
    # Weighted-LN backward (v2 kernel) exact pow2 1/D (bitwise; default off in
    # code, gin turns it on, kill switch DLRM_HSTU_FP8_WLN_BWD_POW2=0): see
    # ops/triton/triton_layer_norm.py _wln_bwd_pow2_enabled.
    logger.info(
        "armed OK: wln_bwd_pow2=%d", int(_tln._wln_bwd_pow2_enabled())
    )

    setup(
        rank=rank,
        world_size=world_size,
        master_addr=master_addr,
        master_port=master_port,
        device=device,
    )
    # SHIP-13 CANDIDATE (dlstart) TSPEC replay (gin apply_env_bootstrap.
    # TRITON_SPEC_REPLAY, default "0" = off): compile the recorded Triton
    # specializations from the current source and load them on this GPU, so the
    # first call after RUN_START is an in-process cache hit (no kernel launch, no
    # tensor, no model / optimizer / RNG state). Earliest point that has both
    # what replay needs: setup() above did torch.cuda.set_device (the kernel
    # caches are per device) and created the process group, and the Phase-2
    # imports loaded the kernel modules (rows resolve by module.qualname; a
    # missing generative_recommenders.* module is imported, which loads no
    # FlyDSL). Before the BLAS warmup, the FP8 first-use warmup (35) and the
    # startup preload below, which then find their Triton kernels already
    # compiled. The node's ranks split the compiles (LPT shard + file barrier,
    # no collective). Never raises: "armed OK: [triton-spec] replay ..." or
    # "[triton-spec] NOT armed: ..." on every rank. Not on an MLPerf resume
    # relaunch (no RUN_START there: the clock is already running).
    _tspec.replay_from_env(
        rank=rank,
        local_rank=local_rank,
        local_world=gpus_per_node,
        world_size=world_size,
        resume=bool(mlperf_resume and mlperf_logger is not None),
    )
    # Phase 3: re-parse to bind the @gin.configurables now that they are
    # registered. The earlier skip_unknown pass already consumed the
    # env-bootstrap binding, but bindings are idempotent so re-applying is
    # fine, and this pass is the one that actually wires up make_model,
    # make_train_test_dataloaders, etc.
    gin.parse_config_file(gin_file)

    # Seed all RNGs (gin-configurable $SEED) BEFORE make_model() so weight init
    # is reproducible run-to-run. Must follow the full parse above so the binding
    # is wired, and precede make_model() below.
    seed_everything(rank=rank)

    # Closed-division learning rate: DENSE_LR / SPARSE_LR unset -> the reference
    # value for this global batch size, 1e-6 * GBS / 8192 (the midpoint of the
    # allowed [0.5x, 1.5x] band); the sparse rate must equal the dense rate, so a
    # single explicit value is applied to both. Resolved before the optimizers
    # read the env (gin dlr/slr env_float, default -1 = unset).
    _gbs = world_size * int(os.environ.get("BATCH_SIZE", "1024"))
    _dlr, _slr = os.environ.get("DENSE_LR", ""), os.environ.get("SPARSE_LR", "")
    if not _dlr and not _slr:
        _dlr = _slr = repr(1e-6 * _gbs / 8192)
    elif not _slr:
        _slr = _dlr
    elif not _dlr:
        _dlr = _slr
    os.environ["DENSE_LR"], os.environ["SPARSE_LR"] = _dlr, _slr
    if rank == 0:
        logger.info(
            "learning rate: dense=%s sparse=%s (global batch %d)", _dlr, _slr, _gbs
        )

    model, model_configs, embedding_table_configs = make_model()
    model, optimizer = make_optimizer_and_shard(
        model=model,
        device=device,
        world_size=world_size,
        local_world_size=gpus_per_node,
        embedding_table_configs=embedding_table_configs,
    )
    # Decorrelate forward-time stochasticity (HSTU dropout) per data-parallel
    # rank. MUST run after make_model() + make_optimizer_and_shard() so the
    # replicated dense weights and sharded embeddings stay init-identical across
    # ranks; this only offsets the global RNG by rank so dropout masks differ.
    decorrelate_runtime_rng(rank=rank)
    train_dataloader, test_dataloader = make_train_test_dataloaders(
        hstu_config=model_configs,
        embedding_table_configs=embedding_table_configs,
    )
    # TFLOPS/MFU reporting: query the model's static dense estimate +
    # current GPU's peak FLOPS. Both default to 0 if the model doesn't
    # expose get_num_flops_per_sample, in which case MetricsLogger silently
    # drops the tflops fields from the perf line.
    inner_model = model.module if hasattr(model, "module") else model
    num_flops_per_sample = (
        float(inner_model.get_num_flops_per_sample())
        if hasattr(inner_model, "get_num_flops_per_sample")
        else 0.0
    )
    gpu_peak_flops = get_gpu_peak_flops(
        "bf16" if getattr(model_configs, "bf16_training", True) else "fp32"
    )
    # Streaming fixed-holdout eval uses the dual fresh/cumulative metric sets:
    # window_* = fresh per-pass full-holdout, lifetime_* = cumulative across
    # passes (AUC via O(bins) histogram). Other modes keep the legacy single set.
    metrics = MetricsLogger(
        multitask_configs=model_configs.multitask_configs,
        batch_size=train_dataloader.batch_size,
        window_size=2500,
        device=device,
        rank=rank,
        # Pass the live world_size so metric normalization is correct at any
        # node count; the gin's MetricsLogger.world_size default (=8) is only a
        # single-node fallback and would mis-normalize a multi-node run.
        world_size=world_size,
        num_flops_per_sample=num_flops_per_sample,
        gpu_peak_flops=gpu_peak_flops,
        model=model,
        eval_cumulative=(mode == "streaming-train-eval"),
        # Lifetime-AUC backend + bins/window come from gin (see yambda_5b.gin:
        # MetricsLogger.{train,eval}_lifetime_auc_mode / cumulative_auc_bins /
        # lifetime_auc_window), env-overridable. eval_cumulative stays explicit
        # because it is runtime-mode dependent, not a config knob.
    )
    # Capture streaming resume hint (None for cold start / non-streaming
    # checkpoints). For the streaming-train-eval mode, we forward this into
    # streaming_train_eval_loop so it can advance past the last completed
    # window OR re-enter the partial window and skip already-trained batches.
    resume_train_ts, resume_batch_idx_in_window, resume_split_contract, resume_cold_start = (
        load_dmp_checkpoint(
            model=model,
            optimizer=optimizer,
            metric_logger=metrics,
            device=device,
            rank=rank,
        )
    )

    # Runtime C++ extension prebuild (gin apply_env_bootstrap.DLRM_HSTU_FP8_PREBUILD,
    # default on; opt out with DLRM_HSTU_FP8_PREBUILD=0). Compilation belongs in
    # the untimed initialization phase: it runs before INIT_STOP/RUN_START
    # below, touches no data, and builds exactly the module the first FP8 step
    # would otherwise build lazily inside the timed region (~14 s).
    if os.environ.get("DLRM_HSTU_FP8_PREBUILD", "0") == "1":
        import time as _time

        from generative_recommenders.ops.triton import triton_fp8 as _tf8

        _t0 = _time.perf_counter()
        _built = _tf8.prebuild_extensions()
        logger.info(
            "armed OK: prebuild before run_start: residual_fused_ext=%d %.1fs",
            int(_built),
            _time.perf_counter() - _t0,
        )
        # yambda_5b.gin item (49): the onesweep sort extension (no-op unless
        # DLRM_HSTU_FP8_POSENC_TS_ONESWEEP or DLRM_HSTU_FP8_EC_JOINT_ONESWEEP
        # is on; logs "armed OK: prebuild ... onesweep_sort_ext=1", or a
        # "NOT armed ... onesweep_sort_ext=0" warning when the build fails).
        from generative_recommenders.ops import onesweep_sort as _osort

        _osort.prebuild()

    from generative_recommenders.dlrm_v4.train import dl_prestart as _dlstart_guard

    # MLPerf cold start: from here on no sample may be handed to a worker or
    # prefetched until RUN_START is logged (dl_prestart.require_run_start guards
    # the sampler release and the cold prefetch).
    if mlperf_cold_start and not metrics.mlperf_run_started:
        _dlstart_guard.expect_run_start()

    # BLAS first-use warmup (gin run_blas_warmup.enabled = $BLAS_WARMUP; on via
    # yambda_5b.gin item (18), absent binding = off). Library/handle/TunableOp-operator init and kernel code-object
    # load for every GEMM family of the first forward+backward, on synthetic
    # tensors at a tiny M, in the untimed initialization phase like the prebuild
    # above. Touches no module, parameter, optimizer, FP8 cache, pin table or
    # RNG stream (dlrm_v4/train/blas_warmup.py). Logs "armed OK: [blas-warmup]".
    # TSPEC globals guard first: the warmups below launch replayed kernels (no-op
    # without a replay).
    _tspec.recheck_globals("pre_warmup")
    run_blas_warmup(device=device)

    # DataLoader pre-start (gin prestart_streaming_loaders.enabled = $DL_PRESTART;
    # item (18), level 2 adds the post-RUN_START cold first-batch prefetch):
    # build the streaming dataset object and fork the train/eval worker pools in
    # the untimed initialization phase, with EMPTY sampler windows -- no sample
    # is read, prefetched or collated before RUN_START (the [dlstart] timeline
    # line after the first step stamps worker-ready and first-read times rel.
    # RUN_START as the compliance proof). Initialization only, like the prebuild.
    from generative_recommenders.dlrm_v4.train import dl_prestart as _dlstart

    def _b1pp_first_use_warmup() -> None:
        # B1'' first-use warmup (item (64);
        # dlrm_v4/b1pp.py): compiles the Triton kernels of the B1''
        # parts that are switched on (DLRM_HSTU_FP8_KJT_RECAT_FUSED /
        # _TBE_REBAG / _TBE_BOUNDS_TRITON) and the fused seq-a2a K1-K4
        # (DLRM_HSTU_FP8_SEQ_A2A_FUSED) of this rank's sequence all-to-alls
        # (D, local features and world size read from the sharded model; the
        # per-rank batch sizes below) on synthetic tensors, in the untimed
        # initialization phase like the FP8 warmup. No data, no RNG, no
        # collective, no model/optimizer state written. A no-op when every
        # part is off, and once it has run. Logs "armed OK: [b1pp-warmup]". An
        # ordinary failure is logged and that part runs stock; a sticky device
        # fault (a failed synchronize: the HIP context is unusable) RAISES
        # here, before RUN_START, as the BLAS warmup and the startup preload do.
        try:
            from generative_recommenders.dlrm_v4 import b1pp as _b1pp
        except Exception as _b1pp_err:  # noqa: BLE001 -- an import failure must not kill the run
            logger.warning(
                "DLRMV4 b1pp [b1pp-warmup] skipped: %s: %s",
                type(_b1pp_err).__name__,
                _b1pp_err,
            )
            return
        _b1pp.run_first_use_warmup(
            device,
            model=model,
            batch_sizes=(
                getattr(train_dataloader, "batch_size", None),
                getattr(test_dataloader, "batch_size", None),
            ),
        )

    if mode == "streaming-train-eval":
        from generative_recommenders.dlrm_v4.train.utils import (
            prestart_streaming_loaders,
        )

        _prestarted = prestart_streaming_loaders(
            hstu_config=model_configs,
            embedding_table_configs=embedding_table_configs,
            rank=rank,
            world_size=world_size,
        )
        # FP8 first-use warmup (ship-12 item (35); gin
        # run_fp8_first_use_warmup.enabled = $FP8_WARMUP, default 0 = off): the
        # JIT/compile of the kernels the FP8 switch (and, tier 2, the bf16
        # calibration steps) would otherwise compile inside the timed region, on
        # synthetic tensors only (dlrm_v4/train/fp8_warmup.py; every latch and
        # cache it touches is restored). Only when the pools above are forked
        # (FlyDSL loads after the fork), and before the startup preload's
        # gc.freeze below so its garbage is collected first. Never raises.
        run_fp8_first_use_warmup(
            device=device,
            model=inner_model,
            batch_size=train_dataloader.batch_size,
            prestarted=bool(_prestarted),
        )
        # B1'' first-use warmup, here so that, like the FP8 warmup above, its
        # compile garbage is collected and its objects are frozen by the startup
        # preload's gc.freeze below (the in-loop full GC at the first window
        # then does not walk them inside the timed region).
        _b1pp_first_use_warmup()
        # Startup preload (gin run_startup_preload.enabled = $BLAS_WARMUP):
        # FlyDSL kernel-module import, the JIT build of the WLN_BWD_FLYDSL FlyDSL
        # LayerNorm-backward kernels on synthetic zero/one tensors, and one
        # gc.collect()/freeze(), AFTER every pool above is forked.
        # Initialization only: no data, no RNG, no training state.
        run_startup_preload(prestarted=bool(_prestarted))
        # Pinned-pool pre-grow (ttt-fix, utils.pin_pool_prewarm; kill switch DL_PIN_PREWARM=0): the train and eval
        # pools' pinned blocks, allocated here after every pool above is forked instead of by the pools' pin threads
        # inside the timed region (first steps / first eval). Host memory only; no data, no RNG.
        if _prestarted:
            from generative_recommenders.dlrm_v4.train.utils import pin_pool_prewarm

            pin_pool_prewarm(rank)

    # B1'' first-use warmup for the other modes (a no-op when it already ran in
    # the streaming block above); before the pre-grow so it sees the final free
    # memory.
    _b1pp_first_use_warmup()

    # Caching-allocator pre-grow (gin apply_env_bootstrap.DLRM_HSTU_FP8_ALLOC_PREGROW,
    # default on; opt out with DLRM_HSTU_FP8_ALLOC_PREGROW=0). Maps the train
    # pipeline's stream pools once, here in the untimed initialization phase
    # (model and optimizer are already allocated), so no hipMemMap device drain
    # runs mid-step. Allocates and frees device memory only; touches no data.
    from generative_recommenders.dlrm_v4 import alloc_pregrow as _alloc_pregrow

    _alloc_pregrow.pregrow_before_run_start()

    # Eval embedding lookahead side stream (gin apply_env_bootstrap.
    # DLRM_HSTU_FP8_EVAL_EMB_LOOKAHEAD, default False = off; a pre-set env var
    # wins, =0 is the kill switch; dlrm_v4/train/eval_lookahead.py). Knob off:
    # returns at once (no stream, no allocation, no collective). Knob on: on
    # every rank, here in the untimed initialization and after the pre-grow
    # above sized main=rest, it creates the eval pipeline's dedicated side
    # stream, places it off the main stream's hardware queue (host_sync_free.
    # shares_hw_queue, at most 8 tries, fail closed), pre-grows its pool from
    # the margin, and arms it only if every rank did (one all_reduce). Logs
    # "armed OK: eval emb lookahead stream ready ..." or "... DECLINED ...".
    from generative_recommenders.dlrm_v4.train import eval_lookahead as _evla

    _evla.prepare_before_run_start()

    # S14-d targeted HIP code-object preload (gin apply_env_bootstrap.
    # DLRM_HSTU_FP8_HIP_PRELOAD, yambda_5b.gin item (70) S14-d: gin binds it
    # ON; code default unset = off; opt out with
    # DLRM_HSTU_FP8_HIP_PRELOAD=0, a pre-set env var wins over gin;
    # dlrm_v4/train/hip_preload.py). Reads the env that the Phase-1 env
    # bootstrap at the top of this function wrote. ROCm
    # 7.2.1 loads libtorch_hip/fbgemm kernels lazily: the first launch of each
    # translation unit pays its fat-binary digest + code-object load (~1 ms per
    # uncompressed MiB), and step 1 (with its once-per-process checks) launches
    # ~40-50 such TUs for the first time (v2 table: 46 rows at 1n, 48 at 2n/4n).
    # hipFuncGetAttributes on one host stub per TU of the checked-in table
    # (hip_preload_table.jsonl) does that load here instead. Initialization only
    # (MLPerf training_rules L545-549, library init before the clock touches no
    # dataset): no kernel launch, no tensor, RNG, model or optimizer state. Here,
    # after setup()'s set_device, the imports, every warmup and the allocator
    # pre-grow (which therefore sizes its pools exactly as without the preload),
    # on every rank. Fail-open per entry, never raises; arming line on rank 0
    # "armed OK: [hip-preload] ...".
    from generative_recommenders.dlrm_v4.train import hip_preload as _hip_preload

    _hip_preload.preload_from_env(
        rank=rank, world_size=world_size, local_world_size=gpus_per_node
    )

    # THP NOHUGEPAGE sweep (gin item (95); code default off, apply_env_bootstrap.
    # DLRM_HSTU_FP8_THP_NOHUGE_SWEEP = True turns it on; env "0" is the kill
    # switch, a pre-set env var wins; dlrm_v4/train/thp_sweep.py). Once, after
    # the DataLoader pre-start fork and every warmup above, MADV_NOHUGEPAGE on
    # this rank's huge-page-eligible private anonymous mappings that hold 4 KiB-
    # mapped pages (exact whole-VMA bounds: a vm_flags change, no page-table
    # change). The GPU driver's restore after the fork re-faults its registered
    # pages for write, which splits their huge pages by copy-on-write; without
    # the sweep khugepaged re-collapses them inside the timed region and each
    # collapse evicts this rank's GPU queues (KFD USERPTR). Reads /proc/self/
    # smaps, /proc/self/maps and the THP policy in sysfs only. Knob off: one env
    # lookup, no import, no line. Never raises: "armed OK: [thp-sweep] ..." /
    # "PARTIAL: [thp-sweep] ..." / "NOT armed: [thp-sweep] ...".
    if os.environ.get("DLRM_HSTU_FP8_THP_NOHUGE_SWEEP", "").strip().lower() in (
        "1", "true", "on", "yes"
    ):
        try:
            from generative_recommenders.dlrm_v4.train import thp_sweep as _thp_sweep

            _thp_sweep.sweep_from_env(rank=rank)
        except Exception as _thp_sweep_err:  # noqa: BLE001 -- never raise: the run continues
            print(
                "NOT armed: [thp-sweep] rank=%s: %s: %s"
                % (rank, type(_thp_sweep_err).__name__, _thp_sweep_err),
                flush=True,
            )

    # TSPEC globals guard, last check before the clock starts (so it never runs
    # inside the timed region; mark_run_start below is only a timestamp). No-op
    # without a replay.
    _tspec.recheck_globals("run_start")

    # MLPerf run markers: open the run exactly once. On a cold start emit
    # submission info + hyperparameters + INIT_STOP/RUN_START and mark the run as
    # started (persisted in the checkpoint via metrics.mlperf_run_started). On a
    # resume, load_dmp_checkpoint restored mlperf_run_started=True, so we skip the
    # markers and just continue the stream. `metrics.mlperf_run_started` guards a
    # double-emit even if cold/resume detection and the checkpoint ever disagree.
    if mlperf_cold_start and not metrics.mlperf_run_started:
        # Submission info + hyperparameters + INIT_STOP/RUN_START, all emitted by
        # the logger (optimizer names/LRs read from gin internally). Seed is the
        # value setup() resolved and exported to $SEED.
        # Evidence for the cache-flush / clock rules: page-cache residency of the
        # reference dataset .npy files opened before RUN_START (explicit list,
        # no glob, so no non-reference file is mapped), one line per node
        # (mincore, reads nothing; sample pages should be zero after the flush).
        if rank % max(1, torch.cuda.device_count()) == 0 and os.environ.get(
            "DLRM_DATA_PATH"
        ):
            _dlstart_guard.page_cache_snapshot(os.environ["DLRM_DATA_PATH"], rank)
        mlperf_logger.log_run_start(
            global_batch_size=world_size * int(train_dataloader.batch_size),
            seed=int(os.environ.get("SEED", "1")),
            # MLPerf 6.1 disclosure keys: per-GPU batch and the gin file in use.
            micro_batch_size=int(train_dataloader.batch_size),
            config_filename=gin_file,
        )
        # Per-node RUN_START marker for the launcher's in-run page-cache warm
        # (Primus DlrmV4Trainer, WARM_CACHE_AFTER_RUN_START): only global rank 0
        # writes the MLPerf log, so a node other than 0 cannot see run_start
        # there. log_run_start ends with a barrier, so every rank is past
        # RUN_START here; each node's local rank 0 writes the node's marker.
        _marker = os.environ.get("DLRM_RUN_START_MARKER")
        if _marker and rank % max(1, torch.cuda.device_count()) == 0:
            try:
                with open(_marker, "w") as _f:
                    _f.write("%.3f\n" % time.time())
            except OSError as _e:
                logger.warning("run_start marker %s not written: %s", _marker, _e)
        # TSPEC: Triton compiles from here on are "post" RUN_START (recorder
        # phase, monitor count).
        _tspec.mark_run_start()
        # Reference for the [dlstart] startup timeline / compliance check.
        _dlstart.set_run_start()
        metrics.mlperf_run_started = True
    else:
        # Resume or no MLPerf logger: the run is already open (or untimed); the
        # same TSPEC boundary for the recorder phase and the monitor.
        _tspec.mark_run_start()
    # Pass the logger to the loop whenever MLPerf logging is enabled, so block /
    # eval / train_loss / run_stop events emit on BOTH a cold start and a resume.
    mlperf_run_active = mlperf_logger is not None and metrics.mlperf_run_started

    # train loop
    failed = False
    try:
        if mode == "train":
            train_loop(
                rank=rank,
                model=model,
                dataloader=train_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "eval":
            # reinit metrics logger for eval
            metrics = MetricsLogger(
                multitask_configs=model_configs.multitask_configs,
                batch_size=train_dataloader.batch_size,
                window_size=1000,
                device=device,
                rank=rank,
                world_size=world_size,
            )
            eval_loop(
                rank=rank,
                model=model,
                dataloader=test_dataloader,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "train-eval":
            train_eval_loop(
                rank=rank,
                model=model,
                train_dataloader=train_dataloader,
                eval_dataloader=test_dataloader,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
            )
        elif mode == "streaming-train-eval":
            # Teardown after RUN_STOP ($DL_TEARDOWN_CLOSE_POOLS, default on;
            # dlrm_v4/train/pool_close.py): the loop records its persistent
            # input pools in this list, and they are closed as soon as it
            # returns (RUN_STOP already logged), on every rank, before the
            # barrier and destroy_process_group below. 0 keeps the old exit.
            from generative_recommenders.dlrm_v4.train import pool_close as _pool_close

            _input_pools = [] if _pool_close.enabled() else None
            streaming_train_eval_loop(
                rank=rank,
                model=model,
                optimizer=optimizer,
                metric_logger=metrics,
                device=device,
                hstu_config=model_configs,
                embedding_table_configs=embedding_table_configs,
                resume_train_ts=resume_train_ts,
                resume_batch_idx_in_window=resume_batch_idx_in_window,
                resume_split_contract=resume_split_contract,
                resume_cold_start=resume_cold_start,
                # Only pass the logger when run boundaries were emitted, so the
                # loop never produces orphan block/eval events.
                mlperf_logger=mlperf_logger if mlperf_run_active else None,
                teardown_pools=_input_pools,
            )
            if _input_pools is not None:
                _pool_close.close_pools(_input_pools, rank=rank, log=logger.warning)
            elif rank == 0:
                logger.warning(_pool_close.OFF_LINE)
    except Exception as e:
        failed = True
        logger.info(traceback.format_exc())
        raise Exception(e)
    finally:
        # Fail fast. A rank that raised must not wait on its peers: they are
        # usually inside a collective this rank will never join, so the
        # teardown barrier below never returns and the whole job sat until
        # the launcher's 15-minute stall watchdog (runner plans/dlrmv4 09:
        # after the eval-time tag mismatch, rank 16's next call was this
        # barrier). Exit at once instead; mp.start_processes then stops the
        # node's other ranks, and the launcher sees the trainer fail and
        # aborts every node within seconds.
        if failed:
            logging.shutdown()
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(1)
        # Graceful distributed teardown on success: barrier so all ranks
        # finish in lockstep, then destroy the process group (best-effort) to
        # avoid noisy TCPStore/NCCL shutdown warnings at exit.
        if torch.distributed.is_initialized():
            try:
                torch.distributed.barrier()
            except Exception:
                logger.info("teardown barrier failed (non-fatal)")
            try:
                cleanup()
            except Exception:
                logger.info("teardown destroy_process_group failed (non-fatal)")
    # Clean exit ($DL_TEARDOWN_HARD_EXIT, default on; dlrm_v4/train/
    # clean_exit.py). Reached only when the try above completed: after the
    # loop (RUN_STOP), the pool close, the barrier and destroy_process_group.
    # An exception never gets here, so a failure keeps its traceback and
    # nonzero exit code. Runs the trainer's exit hooks, SIGKILLs the rank's
    # remaining DataLoader workers (the normal exit terminates and joins
    # them), flushes the logs and calls os._exit(0), so no daemon thread
    # (pool-close helper, pin thread) can be inside torch while the
    # interpreter finalizes (SIGABRT at exit); a TunableOp tuning session
    # keeps the normal exit (clean_exit.py). streaming-train-eval only: the
    # mode with the persistent pinned pools. 0 keeps the interpreter exit.
    if mode == "streaming-train-eval":
        from generative_recommenders.dlrm_v4.train import clean_exit as _clean_exit

        if _clean_exit.enabled():
            _clean_exit.hard_exit(rank, log=logger.warning)
        elif rank == 0:
            logger.warning(_clean_exit.OFF_LINE)


def get_args():  # pyre-ignore [3]
    """Parse commandline."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", default="debug", choices=SUPPORTED_CONFIGS.keys(), help="dataset"
    )
    parser.add_argument(
        "--mode",
        default="train",
        choices=["train", "eval", "train-eval", "streaming-train-eval"],
        help="mode",
    )
    args, unknown_args = parser.parse_known_args()
    logger.warning(f"unknown_args: {unknown_args}")
    return args


def main() -> None:
    args = get_args()
    logger.info(args)
    launch(args.dataset, args.mode)


def launch(dataset: str, mode: str, gin_file: str = "") -> None:
    """Spawn this node's ranks and train. `gin_file` replaces the dataset's
    gin/<config> (Primus passes a generated file that includes it and adds
    bindings); empty keeps the dataset's file."""
    assert dataset in SUPPORTED_CONFIGS, f"Unsupported dataset: {dataset}"
    assert mode in [
        "train",
        "eval",
        "train-eval",
        "streaming-train-eval",
    ], f"Unsupported mode: {mode}"
    # Distributed topology (single-node defaults reproduce the legacy behavior):
    #   GPUS_PER_NODE  local procs to spawn on THIS node (default: all visible GPUs)
    #   NNODES/NODE_RANK  multi-node fan-out, set by the SLURM launcher
    #   WORLD_SIZE     global rank count = NNODES * GPUS_PER_NODE
    #   MASTER_ADDR/PORT  rank-0 rendezvous; the port MUST match across nodes, so
    #                     honor it from the env when set and only fall back to a
    #                     random free port for the standalone single-node path.
    GPUS_PER_NODE = int(os.environ.get("GPUS_PER_NODE", 0)) or torch.cuda.device_count()
    NNODES = int(os.environ.get("NNODES", 1))
    NODE_RANK = int(os.environ.get("NODE_RANK", 0))
    WORLD_SIZE = NNODES * GPUS_PER_NODE
    MASTER_ADDR = os.environ.get("MASTER_ADDR", "localhost")
    MASTER_PORT = str(os.environ.get("MASTER_PORT") or get_free_port())
    gin_path = gin_file or f"{os.path.dirname(__file__)}/gin/{SUPPORTED_CONFIGS[dataset]}"
    logger.info(
        f"launching: nnodes={NNODES} node_rank={NODE_RANK} "
        f"gpus_per_node={GPUS_PER_NODE} world_size={WORLD_SIZE} "
        f"master={MASTER_ADDR}:{MASTER_PORT}"
    )
    # torch.profiler on ROCm >= 10 (kineto on rocprofiler-sdk): the imports above (fbgemm_gpu,
    # via torchrec) start the HIP runtime in this launcher, which initializes rocprofiler-sdk
    # here, and rocprofiler-sdk exports ROCPROFILER_REGISTER_LIBRARY. Inherited by the spawned
    # ranks, it makes rocprofiler-register initialize rocprofiler-sdk during each rank's
    # `import torch`, before kineto's tool (rocprofiler_configure in libtorch_cpu.so) is
    # visible; kineto then takes the initialized SDK as configured and every trace holds CPU
    # events only (docs/04-technical-guides/recommendation/dlrm_v4/rocm-port.md U6). Traced runs drop that self-export (a value
    # already in os.environ, i.e. set before this process started, is kept); untraced runs
    # are unchanged. os.unsetenv: the export is a C-level setenv that os.environ never saw.
    if (os.environ.get("OUTPUT_TRACE", "0").strip() not in ("", "0")
            and "ROCPROFILER_REGISTER_LIBRARY" not in os.environ):
        os.unsetenv("ROCPROFILER_REGISTER_LIBRARY")

    mp.start_processes(
        _main_func,
        args=(
            WORLD_SIZE,
            NODE_RANK,
            GPUS_PER_NODE,
            MASTER_ADDR,
            MASTER_PORT,
            gin_path,
            mode,
        ),
        nprocs=GPUS_PER_NODE,
        join=True,
        start_method="spawn",
    )


if __name__ == "__main__":
    main()
