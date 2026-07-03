###############################################################################
# ODC (On-Demand Communication) integration into Megatron's PyTorch-FSDP2 path.
#
# Gated by env var ODC_ENABLE=1 (so it is a no-op unless explicitly turned on)
# AND requires use_torch_fsdp2=true.
#
# Strategy (see odc_rocm_dev/ROCM_ADAPTATION_REPORT.md for context):
#   Megatron's TorchFullyShardedDataParallel wraps the model with the STANDARD
#   PyTorch `fully_shard` API. ODC's odc/fsdp/fsdp2.py monkey-patches the same
#   torch.distributed.fsdp._fully_shard internals (foreach_all_gather,
#   FSDPParamGroup.post_backward, ...), so ODC's comm replacement applies
#   transparently to Megatron's FSDP2 modules.
#
#   Integration points:
#     patch_fsdp2()       -> BEFORE the first fully_shard call
#     patch_lazy_init(m)  -> AFTER fully_shard, for every wrapped module
#     pre_minibatch_start -> start of each train_step       (PHASE 2)
#     pre_optimizer_step  -> before optimizer.step()        (PHASE 2)
#
# ODC_PHASE env var controls how far we wire in:
#   "1" (default): only __init__ hook (patch_fsdp2 + patch_lazy_init).
#                  Verifies model construction + first forward/backward do not
#                  crash (i.e. ODC's symm-buffer replacement is compatible with
#                  Megatron's FSDP2 params). Gradients are NOT yet routed.
#   "2": also hook train_step / optimizer.step for full grad routing.
###############################################################################

import os

from primus.core.patches import PatchContext, get_args, register_patch
from primus.modules.module_utils import log_rank_0, warning_rank_0

_ODC_READY = False
_FSDP2_PATCHED = False


def _ensure_odc_ready():
    """Initialize MORI-SHMEM and apply ODC's module-level FSDP2 patches.

    Must run BEFORE the first fully_shard() call. Idempotent.
    """
    global _ODC_READY, _FSDP2_PATCHED
    import odc
    from odc.fsdp import fsdp2 as odc_fsdp2

    if not _ODC_READY:
        os.environ.setdefault("MORI_SHMEM_HEAP_SIZE", "8G")
        if "MORI_SOCKET_IFNAME" not in os.environ and "NCCL_SOCKET_IFNAME" in os.environ:
            os.environ["MORI_SOCKET_IFNAME"] = os.environ["NCCL_SOCKET_IFNAME"].lstrip("=")
        odc.init_nvshmem()
        _ODC_READY = True
        log_rank_0("[ODC.torch_fsdp2] init_nvshmem (MORI) done")

    if not _FSDP2_PATCHED:
        odc_fsdp2.patch_fsdp2(enable_hpz=False)
        _FSDP2_PATCHED = True
        log_rank_0("[ODC.torch_fsdp2] patch_fsdp2() applied (foreach_all_gather / post_backward replaced)")


def _apply_patch_lazy_init(root_module):
    """Call ODC patch_lazy_init on every fully_shard-ed submodule that owns a
    parameter group.

    A module is fully_shard-ed iff FSDP2 attached the ``_get_fsdp_state``
    accessor. We additionally require ``state._fsdp_param_group is not None``:
    container wrappers (e.g. the root) have no param group of their own, and ODC
    itself skips them everywhere (replace_sharded_param_with_symm_buffer /
    pre_minibatch_start / pre_optimizer_step all filter ``_fsdp_param_group is
    None``). Patching such a module would just raise AttributeError, so we skip
    it explicitly instead of catching the error.
    """
    from odc.fsdp import fsdp2 as odc_fsdp2

    patched, skipped = 0, 0
    for m in root_module.modules():
        if not hasattr(m, "_get_fsdp_state"):
            continue
        state = m._get_fsdp_state()
        if state is None or state._fsdp_param_group is None:
            skipped += 1
            continue
        try:
            odc_fsdp2.patch_lazy_init(m)
            patched += 1
        except Exception as e:  # noqa: BLE001
            warning_rank_0(
                f"[ODC.torch_fsdp2] patch_lazy_init failed on a param-group module: "
                f"{type(e).__name__}: {e}"
            )
    log_rank_0(
        f"[ODC.torch_fsdp2] patch_lazy_init applied to {patched} param-group modules "
        f"({skipped} container module(s) without param group skipped)"
    )


@register_patch(
    "megatron.fsdp.odc_torch_fsdp2",
    backend="megatron",
    phase="before_train",
    description="Integrate ODC on-demand communication into Megatron's PyTorch FSDP2 path (ROCm/MORI).",
    condition=lambda ctx: os.environ.get("ODC_ENABLE", "0") == "1"
    and getattr(get_args(ctx), "use_torch_fsdp2", False),
)
def patch_odc_torch_fsdp2(ctx: PatchContext):
    import megatron.core.distributed.torch_fully_sharded_data_parallel as tfsdp_mod

    TorchFSDP = tfsdp_mod.TorchFullyShardedDataParallel
    if getattr(TorchFSDP.__init__, "_odc_hooked", False):
        log_rank_0("[ODC.torch_fsdp2] __init__ already hooked, skip")
        return

    orig_init = TorchFSDP.__init__
    phase = os.environ.get("ODC_PHASE", "1")

    def odc_init(self, *args, **kwargs):
        # 1) MORI init + ODC module-level FSDP2 patch, BEFORE any fully_shard.
        _ensure_odc_ready()
        # 2) Megatron's original __init__ runs all fully_shard() calls.
        orig_init(self, *args, **kwargs)
        # 3) AFTER fully_shard: install ODC's lazy_init hook (symm-buffer replace)
        #    on every wrapped module.
        _apply_patch_lazy_init(self.module)
        log_rank_0(f"[ODC.torch_fsdp2] TorchFSDP wrapped with ODC (ODC_PHASE={phase})")

    odc_init._odc_hooked = True
    TorchFSDP.__init__ = odc_init
    log_rank_0("[ODC.torch_fsdp2] hooked TorchFullyShardedDataParallel.__init__")

    if phase == "2":
        _install_train_loop_hooks()


def _find_gpt_model(root):
    """Locate the GPTModel (owns output_layer + compute_language_model_loss)."""
    m = getattr(root, "module", root)  # Float16Module -> GPTModel
    if hasattr(m, "output_layer") and hasattr(m, "compute_language_model_loss"):
        return m
    for sub in root.modules():
        if hasattr(sub, "output_layer") and hasattr(sub, "compute_language_model_loss"):
            return sub
    return None


def _fused_ce_hooked_on(gpt):
    pp = getattr(type(gpt), "_postprocess", None)
    return bool(getattr(pp, "_fused_ce_hooked", False))


def _odc_gather_output_weight(root):
    """At the minibatch boundary (a SYNC point: all ranks enter train_step
    together), all-gather the sharded output weight ONCE and cache the full
    tensor as a grad-tracking leaf. fused CE then reuses it for every micro
    batch WITHOUT per-micro collectives (which would deadlock under DiffMicro).
    Only active when fused CE has patched GPTModel._postprocess.
    """
    gpt = _find_gpt_model(root)
    if gpt is None or not _fused_ce_hooked_on(gpt):
        return
    try:
        if getattr(gpt, "share_embeddings_and_output_weights", False):
            ow = gpt.shared_embedding_or_output_weight()
        else:
            ow = gpt.output_layer.weight
        if hasattr(ow, "full_tensor"):
            gpt._odc_cached_output_weight = ow.full_tensor().detach().requires_grad_(True)
    except Exception as e:  # noqa: BLE001
        warning_rank_0(f"[ODC.fusedce] gather output weight failed: {type(e).__name__}: {e}")


def _odc_reduce_output_grad(root):
    """At pre_optimizer_step (minibatch end, also a SYNC point): all-reduce the
    cached full-weight grad across DP (AVG), scatter it back to the sharded
    param's .grad (DTensor), and drop the cache. One collective per minibatch
    per rank -> DiffMicro-safe.
    """
    import torch.distributed as dist

    gpt = _find_gpt_model(root)
    if gpt is None:
        return
    cached = getattr(gpt, "_odc_cached_output_weight", None)
    if cached is None:
        return
    _dbg_grad = os.environ.get("ODC_DEBUG_GRAD", "0") == "1"
    try:
        if cached.grad is not None:
            full_grad = cached.grad
            if getattr(gpt, "share_embeddings_and_output_weights", False):
                ow = gpt.shared_embedding_or_output_weight()
            else:
                ow = gpt.output_layer.weight
            _pre = full_grad.norm().item() if _dbg_grad else None
            dist.all_reduce(full_grad, op=dist.ReduceOp.AVG)
            if _dbg_grad:
                warning_rank_0(
                    f"[ODC.dbg] output_weight grad norm: pre-AVG(rank0)={_pre:.2f} "
                    f"post-AVG={full_grad.norm().item():.2f}"
                )
            from torch.distributed.tensor import DTensor, distribute_tensor

            if isinstance(ow, DTensor):
                scattered = distribute_tensor(full_grad, ow.device_mesh, ow.placements)
                ow.grad = scattered if ow.grad is None else (ow.grad + scattered)
            elif ow.grad is None:
                ow.grad = full_grad
            else:
                ow.grad = ow.grad + full_grad
    except Exception as e:  # noqa: BLE001
        warning_rank_0(f"[ODC.fusedce] reduce output grad failed: {type(e).__name__}: {e}")
    finally:
        gpt._odc_cached_output_weight = None


def _odc_grad_spike_guard(root):
    """Skip the optimizer step when the global grad norm spikes abnormally.

    WHY: ODC's asynchronous P2P/MORI gradient reduction is numerically
    NON-deterministic -- the same forward (identical loss) can occasionally
    produce a huge grad spike (observed: same iter, loss 11.17512 vs 11.17508,
    but grad norm 9.7 vs 442415). Most spikes are small and clip_grad absorbs
    them, but an extreme one pushes params into a persistent divergent state
    (clip only bounds magnitude, not the already-corrupted direction). This is
    most visible under token-imbalanced SAME_MICRO (arm2); LB-Mini (token
    balanced) rarely triggers it.

    FIX: at the minibatch sync point (optimizer.step), compute the global grad
    norm; if it exceeds ODC_GRAD_SPIKE_THRESHOLD (default 1000, <=0 disables),
    zero ALL grads so the ensuing step is a no-op -- i.e. SKIP this bad iter,
    exactly like Megatron skips nan/inf iterations. Normal grad norm here is
    ~3-42, recoverable spikes ~100-200, pathological ones ~1e5+, so 1000 cleanly
    separates them.

    The all_reduce below is safe under DiffMicro: optimizer.step runs once per
    minibatch at a sync point where all ranks are aligned.
    """
    import torch
    import torch.distributed as dist

    thr = float(os.environ.get("ODC_GRAD_SPIKE_THRESHOLD", "1000"))
    if thr <= 0:
        return
    local_sq = None
    for p in root.parameters():
        g = getattr(p, "grad", None)
        if g is None:
            continue
        if hasattr(g, "to_local"):  # DTensor -> this rank's shard
            g = g.to_local()
        s = g.detach().float().pow(2).sum()
        local_sq = s if local_sq is None else (local_sq + s)
    if local_sq is None:
        return
    if dist.is_initialized():
        dist.all_reduce(local_sq, op=dist.ReduceOp.SUM)
    gnorm = local_sq.sqrt().item()
    if gnorm > thr:
        for p in root.parameters():
            if getattr(p, "grad", None) is not None:
                p.grad.zero_()
        warning_rank_0(
            f"[ODC.spike_guard] grad norm {gnorm:.1f} > {thr} -> SKIP step "
            f"(grads zeroed; non-deterministic ODC async-reduce spike, params unchanged)"
        )


def _install_train_loop_hooks():
    """PHASE 2: wire pre_minibatch_start / pre_optimizer_step into the loop."""
    import megatron.training.training as mt_training
    from odc.fsdp import fsdp2 as odc_fsdp2

    if getattr(mt_training.train_step, "_odc_hooked", False):
        return

    orig_train_step = mt_training.train_step

    def _find_fsdp_root(model):
        # model is a list of model_chunks (TorchFSDP instances). The fully_shard
        # root is chunk.module.
        chunk = model[0] if isinstance(model, (list, tuple)) else model
        root = getattr(chunk, "module", chunk)
        return root

    def odc_train_step(forward_step_func, data_iterator, model, optimizer, *a, **kw):
        root = _find_fsdp_root(model)
        # pre_minibatch_start: clear ODC accumulations at the start of the step.
        try:
            odc_fsdp2.pre_minibatch_start(root)
        except Exception as e:  # noqa: BLE001
            warning_rank_0(f"[ODC.torch_fsdp2] pre_minibatch_start failed: {type(e).__name__}: {e}")
        # fused CE: all-gather full output weight ONCE per minibatch (sync point),
        # so per-micro-batch fused CE never issues a DiffMicro-unsafe collective.
        _odc_gather_output_weight(root)

        # Hook optimizer.step once to inject pre_optimizer_step before it.
        if not getattr(optimizer.step, "_odc_hooked", False):
            orig_step = optimizer.step

            def odc_step(*sa, **sk):
                try:
                    odc_fsdp2.pre_optimizer_step(root)
                except Exception as e:  # noqa: BLE001
                    warning_rank_0(
                        f"[ODC.torch_fsdp2] pre_optimizer_step failed: {type(e).__name__}: {e}"
                    )
                # fused CE: reduce-scatter cached full output-weight grad back to
                # the sharded param (sync point, one collective per minibatch).
                _odc_reduce_output_grad(root)
                # guard against ODC async-reduce's occasional non-deterministic
                # grad spike (can push params into a divergent state, esp. under
                # token-imbalanced SAME_MICRO): skip the step if grad norm spikes.
                _odc_grad_spike_guard(root)
                return orig_step(*sa, **sk)

            odc_step._odc_hooked = True
            optimizer.step = odc_step
            log_rank_0("[ODC.torch_fsdp2] hooked optimizer.step (pre_optimizer_step injected)")

        return orig_train_step(forward_step_func, data_iterator, model, optimizer, *a, **kw)

    odc_train_step._odc_hooked = True
    mt_training.train_step = odc_train_step
    log_rank_0("[ODC.torch_fsdp2] hooked train_step (pre_minibatch_start injected)")


@register_patch(
    "megatron.fsdp.odc_torch_fsdp2_teardown",
    backend="megatron",
    phase="after_train",
    description="Stop ODC reduction watcher after training so the GPU is released cleanly (ROCm/MORI).",
    condition=lambda ctx: os.environ.get("ODC_ENABLE", "0") == "1"
    and getattr(get_args(ctx), "use_torch_fsdp2", False),
)
def patch_odc_torch_fsdp2_teardown(ctx: PatchContext):
    """Gracefully stop the ODC reduction watcher once training is done.

    ODC's ReductionWatcher runs in a spawned subprocess whose server_loop spins
    until it receives an explicit "stop". Megatron/Primus never invoke ODC's
    teardown, so without this hook the watcher stays alive after training, blocks
    the parent rank on join() and holds the GPU. We call ONLY
    ReductionService.stop() (sends "stop" + barrier + cuda sync) and deliberately
    skip finalize_distributed() / SymmBufferRegistry.finalize() so we do not tear
    down the process group before Primus' own cleanup runs. The daemon=True flag
    on the watcher process is the crash-path backstop; this is the graceful path.
    """
    from odc.fsdp import fsdp2 as odc_fsdp2

    rs = odc_fsdp2.get_reduction_service()
    if rs is None:
        log_rank_0("[ODC.torch_fsdp2] teardown: no reduction_service, skip")
        return
    try:
        rs.stop()
        log_rank_0("[ODC.torch_fsdp2] reduction watcher stopped at after_train (GPU releases cleanly)")
    except Exception as e:  # noqa: BLE001
        warning_rank_0(
            f"[ODC.torch_fsdp2] teardown stop() failed (non-fatal, daemon will reap): "
            f"{type(e).__name__}: {e}"
        )
