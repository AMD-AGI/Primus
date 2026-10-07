"""torch's fused AdamW and the grad-clip multiply, run on each DTensor's local shard.

Under FSDP2 every param, grad and Adam moment is a DTensor, so `_fused_adamw_` and the clip's
`_foreach_mul_` go through DTensor dispatch (sharding propagation over ~1000 tensors per call).
That is CPU time the GPU waits out at the end of every step: in a 4-node A6W4 trace the CPU
spent ~20 ms in AdamW.step (14.7 ms of it in DTensor's `_fused_adamw_`) with the GPU mostly
idle. Both ops are elementwise per tensor on Shard placements, so DTensor only runs the same
local op on the local shards; calling it on those directly is byte-identical.

The grad norm itself stays on DTensor: its cross-rank reduction order is DTensor's to define.
"""

import torch
from torch.optim import AdamW
from torch.optim.adam import adam

try:
    from torch.distributed.tensor import DTensor
except ImportError:  # pragma: no cover
    DTensor = None


def _local(ts):
    if DTensor is None:
        return ts
    return [t._local_tensor if isinstance(t, DTensor) else t for t in ts]


def clip_grads_with_norm_local_(parameters, max_norm, total_norm):
    """torch.nn.utils.clip_grads_with_norm_(parameters, max_norm, total_norm, foreach=True) on the
    grads' local shards. ``total_norm`` must be a plain (full) tensor."""
    grads = [p.grad for p in parameters if p.grad is not None]
    if not grads:
        return
    clip_coef = float(max_norm) / (total_norm + 1e-6)
    clip_coef_clamped = torch.clamp(clip_coef, max=1.0)
    grouped = torch.utils._foreach_utils._group_tensors_by_device_and_dtype([_local(grads)])
    for (device, _), ([device_grads], _) in grouped.items():
        torch._foreach_mul_(device_grads, clip_coef_clamped.to(device))


class LocalShardAdamW(AdamW):
    """AdamW(fused=True) whose step runs the fused kernel on local shards. State stays keyed by,
    and shaped like, the DTensor params, so checkpoints are unchanged."""

    def __init__(self, params, **kwargs):
        kwargs["fused"] = True
        super().__init__(params, **kwargs)

    @torch.no_grad()
    def step(self, closure=None):
        if closure is not None:
            return super().step(closure)
        self._accelerator_graph_capture_health_check()
        for group in self.param_groups:
            params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps = [], [], [], [], [], []
            beta1, beta2 = group["betas"]
            has_complex = self._init_group(
                group, params, grads, exp_avgs, exp_avg_sqs, max_exp_avg_sqs, state_steps
            )
            adam(
                _local(params),
                _local(grads),
                _local(exp_avgs),
                _local(exp_avg_sqs),
                _local(max_exp_avg_sqs),
                state_steps,
                amsgrad=group["amsgrad"],
                has_complex=has_complex,
                beta1=beta1,
                beta2=beta2,
                lr=group["lr"],
                weight_decay=group["weight_decay"],
                eps=group["eps"],
                maximize=group["maximize"],
                foreach=group["foreach"],
                capturable=group["capturable"],
                differentiable=group["differentiable"],
                fused=group["fused"],
                grad_scale=getattr(self, "grad_scale", None),
                found_inf=getattr(self, "found_inf", None),
                decoupled_weight_decay=group["decoupled_weight_decay"],
            )
        return None
