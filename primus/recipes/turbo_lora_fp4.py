"""LoRA qkv/proj adapters folded into the frozen MXFP4 base GEMM.

Production runs each LoRALinear as the frozen base GEMM followed by Bridge's
ParallelLinearAdapter: dropout, two bf16 hipBLASLt GEMMs, the alpha/rank scale and an
add, each its own pass over [M, N] or [M, K]. ``lora_frozen_linear`` (Primus-Turbo
``flydsl/gemm/lora_fp4_frozen.py``, overlaid by run_and_time.sh from ``turbo_lora/``)
does the rank-16 product inside the base GEMM's epilogue in forward and inside the dgrad
accumulate in backward.

The adapter input is the activation the base GEMM already receives (the norm output for
qkv, x for proj), so the hook does not reimplement the base linear. LoRALinear.forward
leaves the adapter in ``turbo_fp4_frozen.LORA_SLOT``; the frozen gemm_fp4 wrapper takes
it for that one call. When that GEMM is not on the frozen MXFP4 path (after healing) or
the shape is not one the kernel was validated on, the slot is declined and Bridge's own
adapter runs, unchanged.

Numerics differ from production by one rounding: production rounds the adapter product
to bf16 before the add, the fused epilogue adds it in fp32 and rounds once. The dropout
mask comes from the same generator but not the same draw as nn.Dropout.
"""

from __future__ import annotations

import math
import os

import torch

import turbo_fp4_frozen

#: (N, K) of qkv and proj at M = 8192 tokens, the shapes the campaign validated
#: (0 bad / 6000 repeated launches on both).
_SHAPES = {(8192, 10240, 8192), (8192, 8192, 8192)}
_RANK = 16

_STATE = {"installed": False, "fused": 0, "declined": 0, "fallback": 0}
_ONES = {}
_KIND = {10240: "qkv", 8192: "proj"}

#: LORA_FP4_FUSED_SCOPE=qkv|proj fuses only that projection; the other runs Bridge's adapter.
_SCOPE = os.environ.get("LORA_FP4_FUSED_SCOPE", "all")

#: LORA_FP4_PARITY=1 checks the fused path against production numerics inside the run:
#: every LORA_FP4_PARITY_EVERY-th call's actual output against the frozen base GEMM plus
#: Bridge's bf16 adapter, and every LORA_FP4_PARITY_BWD_EVERY-th call's backward (the
#: champion's dgrad kernels re-run on the real grad_out, and the actual LoRA weight
#: grads) against the same reference. The backward sample holds that call's input until
#: its backward, so it is kept sparse. Lines go to LORA_FP4_PARITY_LOG: stdout is not
#: kept after Primus init.
_PARITY = os.environ.get("LORA_FP4_PARITY", "0") == "1"
_PARITY_EVERY = max(1, int(os.environ.get("LORA_FP4_PARITY_EVERY", "1")))
_PARITY_BWD_EVERY = max(1, int(os.environ.get("LORA_FP4_PARITY_BWD_EVERY", "40")))
_PARITY_FLAG_DB = float(os.environ.get("LORA_FP4_PARITY_FLAG_DB", "30"))
_PARITY_SUMMARY = 1600
_PARITY_LOG = os.environ.get("LORA_FP4_PARITY_LOG", "/results/lora_parity.log")
_PSTATE = {"calls": {}, "stats": {}, "handles": []}

#: LORA_FP4_PIN_CFG=1 pins the qkv/proj GEMM configs to the fixed default (the one used
#: under graph capture: no TACCW or coop twin, 10/9 pipeline) instead of FlyDSL's
#: in-process timed autotune; =all pins every MXFP4 GEMM. PRIMUS_TURBO_AUTO_TUNE does not
#: reach this, it only picks the backend. The rank epilogue reuses the plain GEMM's
#: autotuned config, and only the ruler's own winner was validated with it.
_PIN_CFG = os.environ.get("LORA_FP4_PIN_CFG", "0")
_PIN_CFG = "all" if _PIN_CFG == "all" else ("lora" if _PIN_CFG == "1" else "")


def _rank0() -> bool:
    return not torch.distributed.is_initialized() or torch.distributed.get_rank() == 0


def _note(key: str, msg: str) -> None:
    _STATE[key] += 1
    if _STATE[key] == 1 and _rank0():
        print(f"[LoRAFP4] {msg}", flush=True)
        if _PARITY:
            _plog(f"[LoRAFP4] {msg}")


def _keep_mask(a: torch.Tensor, p: float) -> torch.Tensor:
    if p > 0.0:
        return torch.empty_like(a, dtype=torch.bool).bernoulli_(1.0 - p)
    key = (tuple(a.shape), a.device)
    ones = _ONES.get(key)
    if ones is None:
        ones = _ONES[key] = torch.ones(a.shape, device=a.device, dtype=torch.bool)
    return ones


def _rank() -> int:
    return torch.distributed.get_rank() if torch.distributed.is_initialized() else 0


def _plog(msg: str) -> None:
    try:
        with open(_PARITY_LOG, "a") as fh:
            fh.write(f"[r{_rank()}] {msg}\n")
    except OSError:
        pass


def _snr_db(ref: torch.Tensor, out: torch.Tensor) -> float:
    ref = ref.float()
    err = out.float() - ref
    num = ref.pow(2).sum().item()
    den = err.pow(2).sum().item()
    if not math.isfinite(den):
        return float("-inf")
    if den == 0.0:
        return float("inf")
    return 10.0 * math.log10(max(num, 1e-30) / den)


def _record(kind: str, idx: int, metric: str, db: float, extra: str = "") -> None:
    st = _PSTATE["stats"].setdefault((kind, metric), [0, 0, float("inf")])
    st[0] += 1
    bad = not db >= _PARITY_FLAG_DB
    st[1] += bad
    st[2] = min(st[2], db)
    if idx < 3 or bad:
        _plog(f"{'FLAG ' if bad else ''}{kind} call={idx} {metric} snr={db:.2f}dB {extra}".rstrip())


def _gemm_cfgs(m: int, n: int, k: int) -> str:
    """The cached MXFP4 GEMM configs for (m, n, k): (gm, gn, xcd, wlv, elgk, taccw, coop)."""
    try:
        import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as km

        return str({key[3:]: cfg for key, cfg in km._MXFP4_CFG_CACHE.items() if tuple(key[:3]) == (m, n, k)})
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must not stop the run
        return f"<unavailable: {exc!r}>"


def _pin_gemm_cfgs() -> None:
    import primus_turbo.flydsl.gemm.gemm_mxfp4_kernel as km

    autotune = km._autotune_mxfp4_config
    km._MXFP4_CFG_CACHE.clear()

    def pinned(M, N, K, args, out_fp16=False, k_real=None, row_bytes=None, prepacked=False, **kw):
        if _PIN_CFG != "all" and (M, N, K) not in _SHAPES:
            return autotune(M, N, K, args, out_fp16, k_real, row_bytes, prepacked, **kw)
        key = (M, N, K, row_bytes, out_fp16)
        cfg = km._MXFP4_CFG_CACHE.get(key)
        if cfg is None:
            cfg = km._MXFP4_CFG_CACHE[key] = (*km._mxfp4_nt_config(M, N, K), 10, 9, False, False)
        return cfg

    km._autotune_mxfp4_config = pinned


def _summary(kind: str, idx: int) -> None:
    parts = [
        f"{metric} n={n} flagged={nbad} min={lo:.2f}dB"
        for (k, metric), (n, nbad, lo) in sorted(_PSTATE["stats"].items())
        if k == kind
    ]
    _plog(f"SUMMARY {kind} calls={idx + 1} " + " | ".join(parts))


def _reference_forward(a, row, mask, lora_a, lora_b, p, scale, config):
    """Production numerics: frozen base GEMM, then Bridge's adapter in bf16."""
    act_row, _, dropped = turbo_fp4_frozen._recipes(config.use_gradient_sr)
    a_q, a_s = turbo_fp4_frozen._quantize_row(a, act_row, dropped, config)
    base = turbo_fp4_frozen._gemm_nt(a_q, a_s, row.qdata, row.scale_inv, torch.bfloat16, config)
    xd = torch.ops.aten.native_dropout_backward(a, mask, 1.0 / (1.0 - p))
    h = xd @ lora_a.t()
    y = (h @ lora_b.t()) * scale
    return base, y, a_q, a_s


def _dequant(q, s):
    """fp32 values of a row-quantized 1x32 MXFP4 operand, off Turbo's C++ dequantize (not FlyDSL)."""
    from primus_turbo.pytorch.kernels.quantization.quantization_impl import dequantize_mxfp4_impl

    if q.dtype != torch.float4_e2m1fn_x2:
        q = q.view(torch.float4_e2m1fn_x2)
    if s.dim() != q.dim():
        s = s.reshape(q.shape[0], -1)
    return dequantize_mxfp4_impl(q.contiguous(), torch.float32, 1, 32, s.contiguous())


def _dq_record(kind, idx, metric, fn) -> None:
    """Records snr(*fn()); fp4 x e8m0 values are exact in fp32, so the reference is exact."""
    try:
        ref, out = fn()
        _record(kind, idx, metric, _snr_db(ref, out))
    except Exception as exc:  # noqa: BLE001 -- a diagnostic must not stop the run
        if not _PSTATE.setdefault("dq_failed", False):
            _PSTATE["dq_failed"] = True
            _plog(f"{kind} call={idx} {metric} unavailable: {exc!r}")


def _check_forward(kind, idx, a, row, mask, pending, config, out):
    with torch.no_grad():
        base, y, a_q, a_s = _reference_forward(
            a, row, mask, pending.lora_a, pending.lora_b, pending.p, pending.scale, config
        )
        ref = base + y
        nonfinite = int((~torch.isfinite(out)).sum().item())
        # What fwd_out would read if the LoRA term were dropped: out - base is not
        # comparable to y itself, the bf16 rounding of base swamps it.
        extra = f"nonfinite={nonfinite} if_lora_dropped={_snr_db(ref, base):.2f}dB"
        if idx < 3:
            extra += (
                f" p={pending.p} scale={pending.scale} keep_frac={mask.float().mean().item():.4f}"
                f" |base|max={base.abs().max().item():.4g} |y|max={y.abs().max().item():.4g}"
                f" A={tuple(pending.lora_a.shape)}/{pending.lora_a.dtype} B={tuple(pending.lora_b.shape)}"
            )
        _record(kind, idx, "fwd_out", _snr_db(ref, out), extra)
        # fwd_out's base comes off the same (overlaid) FlyDSL GEMM as `out`; these do not.
        if idx < 3 or idx % _PARITY_BWD_EVERY == 0:
            def dq_base():
                exact = _dequant(a_q, a_s) @ _dequant(row.qdata, row.scale_inv).t()
                _PSTATE["dq_base"] = exact
                return exact, base

            _dq_record(kind, idx, "fwd_base_vs_dq", dq_base)
            exact = _PSTATE.pop("dq_base", None)
            if exact is not None:
                _dq_record(kind, idx, "fwd_out_vs_dq", lambda: (exact + y.float(), out))


def _check_backward(kind, idx, a, mask, pending, out, col, config):
    """Reference grads on the real grad_out; the champion's dgrad kernels re-run on it."""
    lora_a, lora_b, p, scale = pending.lora_a, pending.lora_b, pending.p, pending.scale
    keep = 1.0 / (1.0 - p)
    ref = {}

    def on_grad_out(g):
        from primus_turbo.flydsl.gemm import lora_fp4_frozen as champ

        with torch.no_grad():
            g2 = g.reshape(g.shape[0], -1).contiguous()
            xd = torch.ops.aten.native_dropout_backward(a, mask, keep)
            h = xd @ lora_a.t()
            gs = g2 * scale
            grad_b = gs.t() @ h
            dh = gs @ lora_b
            grad_a = dh.t() @ xd
            dx_lora = torch.ops.aten.native_dropout_backward(dh @ lora_a, mask, keep)
            dh_f, grad_b_f = champ._g_dual(g2, lora_b, h * scale, scale)
            dx_f = champ._rank_acc_mask(torch.zeros_like(a), dh_f, lora_a, mask, keep)
            _record(kind, idx, "bwd_dh", _snr_db(dh, dh_f))
            _record(kind, idx, "bwd_grad_b_kernel", _snr_db(grad_b, grad_b_f))
            _record(kind, idx, "bwd_dx_lora_kernel", _snr_db(dx_lora, dx_f))
            ref["grad_a"], ref["grad_b"] = grad_a, grad_b
            if a.requires_grad:
                try:
                    _, grad_row, dropped = turbo_fp4_frozen._recipes(config.use_gradient_sr)
                    g_q, g_s = turbo_fp4_frozen._quantize_row(g2, grad_row, dropped, config)
                    ref["dx"] = _dequant(g_q, g_s) @ _dequant(col.qdata, col.scale_inv).t() + dx_lora.float()
                except Exception as exc:  # noqa: BLE001
                    _plog(f"{kind} call={idx} act_dx reference unavailable: {exc!r}")
        return None

    def dx_hook(grad):
        r = ref.pop("dx", None)
        if r is not None:
            _record(kind, idx, "act_dx_vs_dq", _snr_db(r, grad), f"nonfinite={int((~torch.isfinite(grad)).sum().item())}")
        return None

    def param_hook(name, key):
        def hook(grad):
            r = ref.pop(key, None)
            if r is not None:
                _record(kind, idx, f"act_{name}", _snr_db(r, grad), f"nonfinite={int((~torch.isfinite(grad)).sum().item())}")
            return None

        return hook

    handles = _PSTATE["handles"]
    handles.append(out.register_hook(on_grad_out))
    if a.requires_grad:
        handles.append(a.register_hook(dx_hook))
    handles.append(lora_a.register_hook(param_hook("grad_a", "grad_a")))
    handles.append(lora_b.register_hook(param_hook("grad_b", "grad_b")))


def log_step(step, loss_dict, grad_norm) -> None:
    """Per-step loss and grad norm into the parity log; drops the last step's hooks."""
    if not _PARITY:
        return
    for h in _PSTATE["handles"]:
        h.remove()
    _PSTATE["handles"].clear()
    if _rank() != 0:
        return
    losses = " ".join(
        f"{k}={(v.item() if isinstance(v, torch.Tensor) else v):.5f}" for k, v in (loss_dict or {}).items()
    )
    gn = grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm
    _plog(f"STEP {step} {losses} grad_norm={gn}")


def _run_forward_pre_hooks(mod: torch.nn.Module) -> None:
    """Fires the hooks mod(...) would, without its forward.

    Megatron DDP's distributed optimizer all-gathers each updated parameter shard from a
    forward pre-hook on the module that owns it (Bridge turns overlap_param_gather on at
    DP > 1). The fused path reads linear_in/linear_out weights without calling either
    module, so skipping this leaves every rank on its own optimizer shard's update only.
    """
    for hook_id, hook in list(mod._forward_pre_hooks.items()):
        if hook_id in mod._forward_pre_hooks_with_kwargs:
            hook(mod, (), {})
        else:
            hook(mod, ())


class _Pending:
    __slots__ = ("lora_a", "lora_b", "p", "scale", "taken", "mods")

    def __init__(self, lora_a, lora_b, p, scale, mods=()):
        self.lora_a, self.lora_b, self.p, self.scale = lora_a, lora_b, p, scale
        self.mods = mods
        self.taken = False

    def __call__(self, a, row, col, config):
        m, k = a.shape
        n = row.qdata.shape[0]
        kind = _KIND.get(n, str(n))
        if (
            (m, n, k) not in _SHAPES
            or tuple(self.lora_a.shape) != (_RANK, k)
            or tuple(self.lora_b.shape) != (n, _RANK)
            or _SCOPE not in ("all", kind)
        ):
            _note("declined", f"declined x={tuple(a.shape)} n={n} scope={_SCOPE}; Bridge adapter runs")
            return None
        from primus_turbo.flydsl.gemm.lora_fp4_frozen import lora_frozen_linear

        for mod in self.mods:
            _run_forward_pre_hooks(mod)
        mask = _keep_mask(a, self.p)
        out = lora_frozen_linear(
            a,
            row.qdata,
            row.scale_inv,
            col.qdata,
            col.scale_inv,
            self.lora_a,
            self.lora_b,
            mask,
            p=self.p,
            scale=self.scale,
            config=config,
        )
        self.taken = True
        _note("fused", f"fused x={tuple(a.shape)} n={n} p={self.p} scale={self.scale}")
        if _PARITY:
            idx = _PSTATE["calls"].get(kind, 0)
            _PSTATE["calls"][kind] = idx + 1
            if idx == 0:
                _plog(f"CFG {kind} pinned={_PIN_CFG or 'no'} {_gemm_cfgs(m, n, k)}")
            if idx % _PARITY_EVERY == 0:
                _check_forward(kind, idx, a, row, mask, self, config, out)
            if idx % _PARITY_BWD_EVERY == 0 and out.requires_grad:
                _check_backward(kind, idx, a, mask, self, out, col, config)
            if (idx + 1) % _PARITY_SUMMARY == 0:
                _summary(kind, idx)
                _plog(f"CFG {kind} pinned={_PIN_CFG or 'no'} {_gemm_cfgs(m, n, k)}")
        return out


def _adapter_args(adapter):
    """(lora_a, lora_b, p, scale) when the adapter is the plain rank-16 pre-dropout form."""
    from megatron.bridge.peft.utils import ParallelLinearAdapter
    from megatron.core import parallel_state

    if not isinstance(adapter, ParallelLinearAdapter):
        return None
    if (
        adapter.dropout_position != "pre"
        or not isinstance(adapter.activation, torch.nn.Identity)
        or getattr(adapter, "is_expert", False)
        or adapter.dim != _RANK
        or parallel_state.get_tensor_model_parallel_world_size() != 1
        or getattr(adapter.linear_in, "bias", None) is not None
        or getattr(adapter.linear_out, "bias", None) is not None
    ):
        return None
    drop = adapter.dropout
    if isinstance(drop, torch.nn.Dropout):
        p = float(drop.p) if drop.training else 0.0
    elif isinstance(drop, torch.nn.Identity):
        p = 0.0
    else:
        return None
    return adapter.linear_in.weight, adapter.linear_out.weight, p, adapter.alpha / adapter.dim


def install() -> bool:
    if _STATE["installed"]:
        return True
    from megatron.bridge.peft.lora_layers import LoRALinear

    orig_forward = LoRALinear.forward
    if _PIN_CFG:
        _pin_gemm_cfgs()

    def forward(self, x, *args, **kwargs):
        args_ = _adapter_args(self.adapter) if self._adapter_enabled else None
        if args_ is None:
            return orig_forward(self, x, *args, **kwargs)
        pending = _Pending(*args_, mods=(self.adapter.linear_in, self.adapter.linear_out))
        slot = turbo_fp4_frozen.LORA_SLOT
        slot[0] = pending
        try:
            linear_output, bias, layernorm_output = self.base_linear_forward(x, *args, **kwargs)
        finally:
            slot[0] = None
        if pending.taken:
            return linear_output, bias
        _note("fallback", "base GEMM not on the frozen MXFP4 path; Bridge adapter runs")
        adapter_output = self.adapter(layernorm_output.contiguous())
        adapter_output = adapter_output.reshape(linear_output.shape)
        return linear_output + adapter_output, bias

    forward.__wrapped__ = orig_forward
    LoRALinear.forward = forward
    _STATE["installed"] = True
    return True
