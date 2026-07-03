import logging

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp._flat_param import FlatParamHandle, HandleShardingStrategy
from torch.distributed.fsdp._runtime_utils import _div_if_needed, _FSDPState

import odc

logger = logging.getLogger(__name__)

reduction_service = None
gather_service = None


def get_reduction_service():
    return reduction_service


def get_gather_service():
    return gather_service


def custom_get_reduce_scatter_tensors(
    state: _FSDPState, unsharded_grad: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns the input and output tensors to reduce-scatter, respectively.
    """
    chunks = list(unsharded_grad.chunk(state.world_size))
    numel_to_pad = state.world_size * chunks[0].numel() - unsharded_grad.numel()
    padded_unsharded_grad = (
        F.pad(unsharded_grad, [0, numel_to_pad]) if numel_to_pad > 0 else unsharded_grad
    )
    # new_sharded_grad = torch.empty_like(chunks[0])  # padded
    # return padded_unsharded_grad, new_sharded_grad
    return padded_unsharded_grad


def _reduce_grad(state, handle) -> None:
    """
    For sharded strategies, this runs gradient reduction, sharded gradient
    accumulation if needed, and the post-reduction callback.
    """

    flat_param = handle.flat_param
    uses_hybrid_sharded_strategy = handle._sharding_strategy in (
        HandleShardingStrategy.HYBRID_SHARD,
        HandleShardingStrategy._HYBRID_SHARD_ZERO2,
    )
    # We clear `.grad` to permit multiple backwards. This avoids a race where
    # the second backward pass computation precedes ahead of the first backward
    # pass reduction, which is possible since the reduction is issued in a
    # separate stream and is async and would result in reducing the wrong
    # gradient.
    unsharded_grad = flat_param.grad.data
    flat_param.grad = None
    # print(f"unsharded_grad shape: {unsharded_grad.shape} dtype {unsharded_grad.dtype}")
    # padded_unsharded_grad, new_sharded_grad = _get_reduce_scatter_tensors(state, unsharded_grad)
    padded_unsharded_grad = custom_get_reduce_scatter_tensors(state, unsharded_grad)
    # print(state._comm_hook)
    assert state._comm_hook is None, "ODC does not support comm hook"
    if state._comm_hook is None:  # default path
        _div_if_needed(padded_unsharded_grad, state._gradient_predivide_factor)
        pg = handle._fake_process_group if handle._use_fake_reduce else state.process_group
        # dist.reduce_scatter_tensor(
        #     new_sharded_grad,
        #     padded_unsharded_grad,
        #     group=pg,
        # )

        rs_func = get_reduction_service().scatter_accumulate
        rs_func(id(handle.flat_param), padded_unsharded_grad, pg)
        handle.flat_param._saved_grad_shard = get_reduction_service().get_accumulation(
            id(handle.flat_param)
        )

        assert not uses_hybrid_sharded_strategy, "ODC does not support hybrid sharded strategy"
        # if uses_hybrid_sharded_strategy:
        #     # Don't wait during trace
        #     if not torch.distributed._functional_collectives.is_torchdynamo_compiling():
        #         state._all_reduce_stream.wait_stream(state._post_backward_stream)
        #     with state._device_handle.stream(state._all_reduce_stream):
        #         # Since the new sharded gradient is produced in the post-
        #         # backward stream and consumed in the all-reduce stream,
        #         # inform the caching allocator
        #         _no_dispatch_record_stream(new_sharded_grad, state._all_reduce_stream)
        #         dist.all_reduce(new_sharded_grad, group=state._inter_node_pg)
        #         _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
        #         grad_to_offload = _accumulate_sharded_grad(
        #             state, handle, new_sharded_grad
        #         )
        #         _post_reduce_grad_callback(state, handle, grad_to_offload)
        #         return
        # _div_if_needed(new_sharded_grad, state._gradient_postdivide_factor)
    else:
        pass
        # state._comm_hook(state._comm_hook_state, padded_unsharded_grad, new_sharded_grad)
        # NOTE: HSDP variants do not support communication hook.

    # Already set below: handle.flat_param._saved_grad_shard = xxx
    # grad_to_offload = _accumulate_sharded_grad(state, handle, new_sharded_grad)
    # Not supported by ODC
    assert not handle._offload_params, "ODC does not support offloading"
    assert not handle._use_orig_params, "ODC does not support using original parameters"
    # _post_reduce_grad_callback(state, handle, grad_to_offload)


def prepare_gradient_for_optim(self):
    """Prepare the gradient for optimizer computation by moving the sharded gradient to the ``.grad`` attribute."""
    from torch.distributed.utils import _p_assert

    def cast_grad_to_param_dtype_if_needed(flat_param):
        # TODO (rohan-varma): test for full precision with keep_low_precision_grads
        if not self._force_full_precision and self._keep_low_precision_grads:
            _p_assert(flat_param.grad is not None, "Unexpected None grad!")
            if flat_param.grad.dtype != self._fwd_bwd_param_dtype:
                flat_param.grad.data = flat_param.grad.to(self._fwd_bwd_param_dtype)
                if self._use_orig_params:
                    self._use_sharded_grad_views()

    flat_param = self.flat_param
    # TODO (awgu): We should replace these conditional checks to encode
    # the logical intention more directly.
    if hasattr(flat_param, "_cpu_grad"):
        # NOTE: This branch includes `NO_SHARD`.
        self._check_sharded(flat_param)
        self._check_on_cpu(flat_param)
        flat_param.grad = flat_param._cpu_grad  # type: ignore[attr-defined]
        cast_grad_to_param_dtype_if_needed(flat_param)
    elif hasattr(flat_param, "_saved_grad_shard"):
        self._check_sharded(flat_param)
        self._check_on_compute_device(flat_param)
        if flat_param._saved_grad_shard is not None:
            self._check_on_compute_device(flat_param._saved_grad_shard)  # type: ignore[attr-defined]
        # If no sharded gradient was computed this iteration, then there is
        # no need to forward `_saved_grad_shard` to `grad`
        if flat_param._post_backward_called:  # type: ignore[attr-defined]
            # print(f"Rank {dist.get_rank()}: flat_param shape: {flat_param.shape}")
            flat_param.grad = None  # type: ignore[attr-defined]
            if flat_param.grad is not None:
                cast_grad_to_param_dtype_if_needed(flat_param)
    else:
        _p_assert(
            not self.uses_sharded_strategy or not flat_param._post_backward_called,  # type: ignore[attr-defined]
            "All sharded parameters that received a gradient in the "
            "post-backward should use `_saved_grad_shard`",
        )
    # Delete `_saved_grad_shard` since its existence indicates a previous
    # gradient to accumulate with in the post-backward hook
    if hasattr(flat_param, "_saved_grad_shard"):
        delattr(flat_param, "_saved_grad_shard")


def all_gather_flat_param(self, padded_unsharded_flat_param):
    """
    All-gather the handle's flat parameter to the destination ``padded_unsharded_flat_param``.

    Then switch to use the all-gathered tensor.
    """
    from torch.distributed.fsdp._common_utils import _no_dispatch_record_stream
    from torch.distributed.utils import _p_assert

    _p_assert(
        hasattr(self, "process_group") and hasattr(self, "world_size"),
        "Expects a process group and world size to have been set via `shard()`",
    )
    sharded_flat_param = self.flat_param.data
    # print(f"sharded_flat_param.dtype: {sharded_flat_param.dtype}")
    expected_numel = sharded_flat_param.numel() * self.world_size
    _p_assert(
        padded_unsharded_flat_param.numel() == expected_numel,
        f"Expects {expected_numel} numel but got {padded_unsharded_flat_param.numel()}",
    )

    pg = self._fake_process_group if self._use_fake_all_gather else self.process_group

    # HACK this should be handled by C10D
    if sharded_flat_param.is_cpu:  # type: ignore[attr-defined]
        tensor_list = list(
            torch.chunk(
                padded_unsharded_flat_param,
                dist.get_world_size(pg),  # type: ignore[arg-type]
            )
        )
        dist.all_gather(tensor_list, sharded_flat_param, group=pg)
    else:
        # padded_unsharded_flat_param output torch.bfloat16 sharded_flat_param input torch.float32
        assert padded_unsharded_flat_param.dtype == self._fwd_bwd_param_dtype
        assert sharded_flat_param.dtype == self._orig_param_dtype
        # Handle dtype conversion for mixed precision
        # We need to convert to output to orig_dtype for all-gather
        # as the sharded_flat_param is in orig_dtype for all ranks, then convert back
        # This is because when using ODC, the peer may have not reached here and be able to
        # convert input to _fwd_bwd_param_dtype.
        from torch.distributed.utils import _free_storage

        needs_dtype_conversion = padded_unsharded_flat_param.dtype != sharded_flat_param.dtype
        if needs_dtype_conversion:
            assert self._uses_param_mixed_precision
            # Convert output to orig_dtype for all-gather (creates new tensor)
            padded_unsharded_flat_param_orig_dtype = padded_unsharded_flat_param.to(
                self._orig_param_dtype
            )
        else:
            padded_unsharded_flat_param_orig_dtype = padded_unsharded_flat_param

        ag_func = gather_service.gather_into_tensor
        ag_func(
            padded_unsharded_flat_param_orig_dtype,  # Could be fp32
            # padded_unsharded_flat_param,
            sharded_flat_param,
            pg,
        )

        # Convert back to fwd_bwd_param_dtype and free the temporary buffer
        if needs_dtype_conversion:
            # Convert the dtype conversion buffer back to fwd_bwd_param_dtype in-place
            # copy_ will do the implicit conversion.
            padded_unsharded_flat_param.copy_(padded_unsharded_flat_param_orig_dtype)
            # Free the temporary buffer
            _free_storage(padded_unsharded_flat_param_orig_dtype)

    if self._offload_params:
        # In case of offloading, `flat_param.data` (i.e. sharded param) is
        # created on the pre-unshard stream. We need to hand it over to the
        # unshard stream for all-gather
        _no_dispatch_record_stream(
            sharded_flat_param,
            self._device_handle.current_stream(),  # unshard_stream
        )
    return padded_unsharded_flat_param


old_get_shard = FlatParamHandle._get_shard


def custom_get_shard(tensor, rank, world_size):
    from torch.distributed.fsdp._flat_param import FlatParameter

    sharded, padded = old_get_shard(tensor, rank, world_size)
    assert isinstance(tensor, FlatParameter)
    sharded_in_nvshmem = odc.SymmBufferRegistry.get_instance().update_symm_buffer(
        id(tensor), sharded
    )
    return sharded_in_nvshmem, padded


def _use_low_precision_shard(self):
    """
    Allocate on the compute device and switch to using the low precision sharded flat parameter.
    call path:
        _runtime_utils.py:_unshard()
            -> pre_unshard()
                -> self._use_low_precision_shard()
    """
    self._check_low_precision_shard()
    # flat_param = self.flat_param
    # _alloc_storage(
    #     flat_param._mp_shard,
    #     flat_param._local_shard.size(),  # type: ignore[attr-defined]
    # )
    # # `copy_()` implicitly casts to the low precision
    # flat_param._mp_shard.copy_(  # type: ignore[attr-defined]
    #     flat_param._local_shard.to(  # type: ignore[attr-defined]
    #         self.device, non_blocking=True
    #     )
    # )
    # # Invariant: `_mp_shard` is always on the compute device.
    # flat_param.data = flat_param._mp_shard  # type: ignore[attr-defined]
    # flat_param.data = flat_param._local_shard.to(self.device, non_blocking=True)

    # todo: need to check
    # flat_param.data.copy_(flat_param._local_shard.to(self.device, non_blocking=True))


def _free_low_precision_sharded_param(self):
    """
    Frees the low precision sharded flat parameter.

    call path:
        _runtime_utils.py:_unshard()
            -> handle.post_unshard()
                -> self._free_low_precision_sharded_param()
    """
    self._check_low_precision_shard()
    # `_mp_shard` is allocated in the pre-unshard stream, consumed in the
    # unshard stream for sharded strategies, and consumed in both the
    # unshard and default streams for `NO_SHARD`. For sharded strategies,
    # the current stream here is the unshard stream, and for `NO_SHARD`,
    # it is the default stream. For `NO_SHARD`, only recording for the
    # default stream suffices since the default stream waits for the
    # unshard stream.
    # _no_dispatch_record_stream(
    #     self.flat_param._mp_shard,
    #     self._device_handle.current_stream(),  # type: ignore[attr-defined]
    # )
    # _free_storage(self.flat_param._mp_shard)  # type: ignore[attr-defined]


def patch_fsdp1(reduce_dtype=None):
    global reduction_service, gather_service
    reduction_service = odc.ReductionService(accumulation_dtype=reduce_dtype)
    gather_service = odc.GatherService()
    from torch.distributed.fsdp import _runtime_utils

    _runtime_utils._reduce_grad = _reduce_grad

    FlatParamHandle.prepare_gradient_for_optim = prepare_gradient_for_optim
    FlatParamHandle._get_shard = custom_get_shard
    FlatParamHandle._all_gather_flat_param = all_gather_flat_param
    FlatParamHandle._use_low_precision_shard = _use_low_precision_shard
    FlatParamHandle._free_low_precision_sharded_param = _free_low_precision_sharded_param


def pre_optimizer_step(fsdp_module):

    assert isinstance(fsdp_module, _FSDPState)
    with torch.cuda.nvtx.range("scatter_accumulate_sync"):
        get_reduction_service().sync(fsdp_module.process_group)

    # time.sleep(1)
    for acc in get_reduction_service().accumulations:
        if hasattr(fsdp_module, "_inter_node_pg"):
            dist.all_reduce(acc, group=fsdp_module._inter_node_pg)
        _div_if_needed(acc, fsdp_module._gradient_postdivide_factor)
    # print(f"Model parameters: {[p.numel()/ 1e6 for p in fsdp_module.parameters()]}")
    for handle in fsdp_module._all_handles:
        # print(f"Rank {dist.get_rank()}: cast_grad_to_param_dtype_if_needed shape: {handle.flat_param.shape}")
        handle.flat_param.grad = (
            get_reduction_service()
            .get_accumulation(id(handle.flat_param))
            .to(handle.flat_param.dtype)
        )


def pre_minibatch_start(_fsdp_module):
    get_reduction_service().clear_accumulations()

    # Make sure optimizer updates are visible to all ranks
    dist.barrier()


def stop():
    get_reduction_service().stop()
    odc.SymmBufferRegistry.get_instance().finalize()
    odc.finalize_distributed()
