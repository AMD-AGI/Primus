###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Gradient-norm clipping without a device-to-host sync.

Megatron's ``get_grad_norm_fp32`` returns ``total_norm.item() ** 0.5``: the host waits for the norm's all-reduce,
then computes the clip coefficient ``max_norm / (norm + 1e-6)`` as a Python float and launches the scale with it.
The host is blocked for the whole backward tail, and the GPU idles between the all-reduce and the optimizer while
the host wakes up and launches the clip.

Here the squared norm stays on the device (:class:`DeviceGradNorm`) and the coefficient is computed there in fp64,
then cast to fp32 -- the precision at which every scale kernel applies it. The gradients are always scaled, by 1.0
when no clipping is needed, which leaves them unchanged bit for bit.

Bitwise notes (checked on the GPU this targets):

* fp64 ``sqrt``, ``+`` and ``/`` on the GPU are correctly rounded. The host path's ``x ** 0.5`` (libm ``pow``) is
  not always: for a small fraction of inputs it is one ulp away from the correctly rounded square root. The fp32
  coefficient derived from either is the same in practice (a one-ulp fp64 difference survives the fp32 cast only when
  it straddles an fp32 rounding boundary). The *reported* norm is still computed on the host
  with the original expression, from the same fp32 squared norm, so it is identical to the original.
* A bf16 tensor multiplied by an fp32 0-dim CUDA tensor (``mul_`` or ``_foreach_mul_``) rounds the multiplier to
  bf16 first, so it does not reproduce ``mul_(python_float)``. The gradients are therefore scaled by a small
  Triton kernel that multiplies in fp32 and rounds once, as the eager kernel does. (For fp32 gradients
  ``torch._foreach_mul_`` with the device scalar would be exact too, but it always reads and writes every byte.)
* The kernel masks all loads and stores off when the coefficient is exactly 1.0, so a step that does not clip costs
  one near-empty launch per contiguous run instead of a full read and write of the gradients.
"""

import operator

import torch
import triton
import triton.language as tl


class DeviceGradNorm:
    """A gradient 2-norm whose value is still on the device.

    ``sq`` is the all-reduced squared norm (fp32, one element), exactly the tensor Megatron calls ``.item()`` on.
    Converting to a host float (``float()``, formatting, arithmetic, comparisons) syncs once and caches the result;
    the expression is the original ``sq.item() ** (1.0 / 2.0)``, so the host value is identical to Megatron's.
    """

    __slots__ = ("sq", "_host")

    def __init__(self, sq: torch.Tensor):
        self.sq = sq
        self._host = None

    def set_host_squared(self, sq_value: float) -> None:
        """Provide the squared norm read back asynchronously elsewhere, so ``float()`` need not sync."""
        if self._host is None:
            self._host = sq_value ** (1.0 / 2.0)

    def __float__(self) -> float:
        if self._host is None:
            self._host = self.sq.item() ** (1.0 / 2.0)
        return self._host

    def __format__(self, spec: str) -> str:
        return format(float(self), spec)

    def __repr__(self) -> str:
        return f"DeviceGradNorm({float(self)!r})" if self._host is not None else "DeviceGradNorm(<on device>)"

    def __bool__(self) -> bool:
        return bool(float(self))


def _delegate(op, reflected=False):
    if reflected:
        return lambda self, other: op(other, float(self))
    return lambda self, other: op(float(self), other)


# Any consumer that treats the norm as a number gets the host value (correct, at the cost of one sync).
for _name, _op in (
    ("add", operator.add),
    ("sub", operator.sub),
    ("mul", operator.mul),
    ("truediv", operator.truediv),
    ("pow", operator.pow),
):
    setattr(DeviceGradNorm, f"__{_name}__", _delegate(_op))
    setattr(DeviceGradNorm, f"__r{_name}__", _delegate(_op, reflected=True))
for _name, _op in (
    ("lt", operator.lt),
    ("le", operator.le),
    ("gt", operator.gt),
    ("ge", operator.ge),
    ("eq", operator.eq),
    ("ne", operator.ne),
):
    setattr(DeviceGradNorm, f"__{_name}__", _delegate(_op))
DeviceGradNorm.__hash__ = None


def clip_coefficient(total_norm: DeviceGradNorm, max_norm: float) -> torch.Tensor:
    """fp32 0-dim device tensor: ``max_norm / (norm + 1e-6)`` computed in fp64, or 1.0 where that is not < 1.

    ``tensor / tensor`` rather than ``max_norm / tensor``: the latter is evaluated as ``reciprocal() * max_norm``,
    which rounds twice.
    """
    norm = total_norm.sq.reshape(()).double().sqrt()
    coeff = norm.new_full((), float(max_norm)) / (norm + 1.0e-6)
    one = torch.ones_like(coeff)
    # NaN < 1 is False, so a NaN norm scales by 1.0 -- the original also skips the scale then.
    return torch.where(coeff < 1.0, coeff, one).float()


@triton.jit
def _scale_by_device_scalar_kernel(x_ptr, n_elements, coeff_ptr, BLOCK: tl.constexpr):
    pid = tl.program_id(0).to(tl.int64)
    offs = pid * BLOCK + tl.arange(0, BLOCK)
    coeff = tl.load(coeff_ptr)
    # A coefficient of exactly 1.0 leaves every element unchanged: mask everything off, so a step that does not
    # clip moves no gradient bytes.
    mask = (offs < n_elements) & (coeff != 1.0)
    x = tl.load(x_ptr + offs, mask=mask)
    y = x.to(tl.float32) * coeff
    tl.store(x_ptr + offs, y.to(x_ptr.dtype.element_ty), mask=mask)


_BLOCK = 4096


def scale_flat_(flat: torch.Tensor, coeff: torch.Tensor) -> None:
    """``flat *= coeff`` for a contiguous 1-D tensor, multiplying in fp32 and rounding once to ``flat.dtype``."""
    n = flat.numel()
    if n == 0:
        return
    grid = (triton.cdiv(n, _BLOCK),)
    _scale_by_device_scalar_kernel[grid](flat, n, coeff, BLOCK=_BLOCK, num_warps=8)


def scale_grads_(grads, coeff: torch.Tensor, runs=None) -> None:
    """Scale every gradient in ``grads`` by the device scalar ``coeff`` (fp32, 0-dim).

    ``runs`` -- optional ``(base, offset, numel)`` contiguous runs covering ``grads`` exactly once (as produced by the
    flat-clip coalescer); each is scaled with one launch. Otherwise each gradient is scaled with one launch.
    """
    if runs is not None:
        for base, offset, numel in runs:
            scale_flat_(torch.as_strided(base, (numel,), (1,), offset), coeff)
        return
    for g in grads:
        if g.is_contiguous():
            scale_flat_(g.view(-1), coeff)
        else:
            tmp = g.contiguous()
            scale_flat_(tmp.view(-1), coeff)
            g.copy_(tmp)
