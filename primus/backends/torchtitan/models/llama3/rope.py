import torch

import transformer_engine
import transformer_engine_torch as tex


if torch.__version__ >= "2.4.0":
    _torch_custom_op_wrapper = torch.library.custom_op
    _torch_register_fake_wrapper = torch.library.register_fake
else:
    # TODO
    pass


@_torch_custom_op_wrapper("te::_rope_fwd", mutates_args=(), device_types="cuda")
def _rope_fwd(
    t: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    output = tex.fused_rope_forward(t.transpose(0, 1), freqs, True).transpose(0, 1)
    return output


@_torch_register_fake_wrapper("te::_rope_fwd")
def _rope_fwd_fake(
    t: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    return torch.empty_like(t, device=t.device)


@_torch_custom_op_wrapper("te::_rope_bwd", mutates_args=("dx",), device_types="cuda")
def _rope_bwd(
    dx: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    # dx: [B, S, H, D]
    dx = tex.fused_rope_backward(dx.transpose(0, 1), freqs, True).transpose(0, 1)
    return dx


@_torch_register_fake_wrapper("te::_rope_bwd")
def _rope_bwd_fake(
    dx: torch.Tensor,
    freqs: torch.Tensor,
) -> torch.Tensor:
    # assert dx.ndim == 4 and freqs.ndim == 4
    # assert dx.shape[1] == freqs.shape[0]
    # assert dx.shape[-1] == freqs.shape[-1]
    return torch.empty_like(dx, device=dx.device)


if torch.__version__ >= "2.4.0":
    _wrapped_rope_fwd = torch.ops.te._rope_fwd
    _wrapped_rope_bwd = torch.ops.te._rope_bwd
else:
    # TODO:
    pass


class TEFusedRoPEFunc(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        t: torch.Tensor,
        freqs: torch.Tensor,
    ) -> torch.Tensor:
        if freqs.dtype != torch.float32:
            freqs = freqs.float()
        output = _wrapped_rope_fwd(t, freqs)
        ctx.save_for_backward(freqs)
        return output

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        (freqs,) = ctx.saved_tensors
        grad_input = _wrapped_rope_bwd(grad_output, freqs)
        return grad_input, None