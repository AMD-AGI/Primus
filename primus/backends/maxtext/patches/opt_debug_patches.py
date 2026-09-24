###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText optimizer-state debug patch (opt-in).

Wraps ``training_loop_iteration`` in MaxText's NNX pre-train loop so the first
few steps log -- and optionally dump to ``.npy`` -- selected parameters and
their optimizer leaves (``mu``/``nu``/``count``), plus every non-finite
parameter leaf. Also appends ``grad_norm``/``raw_grad_norm``/``param_norm`` to
the per-step training log line.

  PRIMUS_MAXTEXT_OPT_DEBUG=1           enable
  PRIMUS_MAXTEXT_OPT_DEBUG_KEYS=a,b    param path components to track (default wi_0,wi_1)
  PRIMUS_MAXTEXT_OPT_DEBUG_STEPS=N     number of leading steps to inspect (default 3)
  PRIMUS_MAXTEXT_OPT_DEBUG_DUMP=<dir>  also save each tracked leaf as float32 .npy

Every inspected step forces a device sync and host transfers, so this is a
debugging aid only.
"""

import functools
import os

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import log_rank_0, warning_rank_0

_ENV = "PRIMUS_MAXTEXT_OPT_DEBUG"
_FLOAT32_MIN_NORMAL = 1.1754944e-38
_MAX_BAD_LEAVES_LOGGED = 24


def _enabled(_ctx=None) -> bool:
    return os.getenv(_ENV) == "1"


def _log(msg: str) -> None:
    from maxtext.utils import max_logging

    max_logging.log(msg)


def _leaf_stats(x):
    import jax.numpy as jnp

    f = x.astype(jnp.float32)
    fin = jnp.isfinite(f)
    a = jnp.abs(f)
    nz = fin & (a > 0)
    stats = (
        f"dtype={x.dtype} shape={x.shape} "
        f"nan={int(jnp.isnan(f).sum())} inf={int(jnp.isinf(f).sum())} zero={int((f == 0).sum())} "
        f"denorm={int((nz & (a < _FLOAT32_MIN_NORMAL)).sum())} "
        f"absmax={float(jnp.where(fin, a, 0).max()):.4e} "
        f"absmin_nz={float(jnp.where(nz, a, jnp.inf).min()):.4e}"
    )
    return f, stats


def _inspect(step: int, state, tag: str) -> None:
    import jax
    import numpy as np
    from flax import nnx

    keys = os.getenv(f"{_ENV}_KEYS", "wi_0,wi_1").split(",")
    dump_dir = os.getenv(f"{_ENV}_DUMP")
    trees = (
        ("param", nnx.to_pure_dict(nnx.state(state.model, nnx.Param))),
        ("opt", nnx.to_pure_dict(nnx.state(state.optimizer))),
    )
    for label, tree in trees:
        for path, x in jax.tree_util.tree_leaves_with_path(tree):
            if not isinstance(x, jax.Array):
                continue
            ks = jax.tree_util.keystr(path)
            if label == "opt" and "count" in ks:
                _log(f"OPTDBG {tag} step={step} {label}{ks} = {jax.device_get(x)}")
                continue
            if not any(f"'{k}'" in ks for k in keys):
                continue
            f, stats = _leaf_stats(x)
            _log(f"OPTDBG {tag} step={step} {label}{ks} {stats}")
            if dump_dir:
                os.makedirs(dump_dir, exist_ok=True)
                name = "".join(c if c.isalnum() or c in "_-" else "_" for c in ks).strip("_")
                np.save(
                    os.path.join(dump_dir, f"{tag}_step{step}_{label}_{name}.npy"),
                    np.asarray(jax.device_get(f)),
                )


def _scan_nonfinite_params(step: int, state) -> None:
    import jax
    import jax.numpy as jnp
    from flax import nnx

    n_bad = 0
    for path, x in jax.tree_util.tree_leaves_with_path(nnx.state(state.model, nnx.Param)):
        if not isinstance(x, jax.Array) or not jnp.issubdtype(x.dtype, jnp.inexact):
            continue
        nan_c = int(jnp.isnan(x).sum())
        inf_c = int(jnp.isinf(x).sum())
        if nan_c or inf_c:
            n_bad += 1
            if n_bad <= _MAX_BAD_LEAVES_LOGGED:
                _log(
                    f"OPTDBG BAD step={step} param{jax.tree_util.keystr(path)} nan={nan_c} inf={inf_c} "
                    f"dtype={x.dtype} shape={x.shape}"
                )
    if n_bad:
        _log(f"OPTDBG BAD step={step} non-finite param leaves={n_bad}")


def _wrap_iteration(orig):
    max_steps = int(os.getenv(f"{_ENV}_STEPS", "3"))
    first_call = [True]

    @functools.wraps(orig)
    def training_loop_iteration(jax_device_state, python_vars, immutable_data):
        step = int(python_vars["step"])
        active = step < max_steps and hasattr(jax_device_state["state"], "optimizer")
        if active and first_call[0]:
            _inspect(step, jax_device_state["state"], "pre")
        first_call[0] = False
        orig(jax_device_state, python_vars, immutable_data)
        if active:
            _inspect(step, jax_device_state["state"], "post")
            _scan_nonfinite_params(step, jax_device_state["state"])

    return training_loop_iteration


def _wrap_log_training_metrics(orig):
    @functools.wraps(orig)
    def _log_training_metrics(self, metrics, step):
        orig(self, metrics, step)
        scalars = metrics.get("scalar", {})
        parts = [
            f"{name}: {float(scalars[key]):.6g}"
            for name, key in (
                ("grad_norm", "learning/grad_norm"),
                ("raw_grad_norm", "learning/raw_grad_norm"),
                ("param_norm", "learning/param_norm"),
            )
            if key in scalars
        ]
        if parts:
            _log(f"OPTDBG norms step={step} " + ", ".join(parts))

    return _log_training_metrics


@register_patch(
    patch_id="maxtext.opt_debug",
    backend="maxtext",
    phase="setup",
    description="Opt-in per-step param/optimizer-state inspection and dumps (PRIMUS_MAXTEXT_OPT_DEBUG=1)",
    condition=_enabled,
)
def patch_opt_debug(ctx: PatchContext) -> None:
    log_rank_0("[Patch:maxtext.opt_debug] Patching MaxText training loop...")
    try:
        from maxtext.common import metric_logger
        from maxtext.trainers.pre_train import train as maxtext_train
    except ImportError as e:
        warning_rank_0(f"[Patch:maxtext.opt_debug] MaxText v26.4+ layout not found; skipping: {e}")
        return

    maxtext_train.training_loop_iteration = _wrap_iteration(maxtext_train.training_loop_iteration)
    metric_logger.MetricLogger._log_training_metrics = _wrap_log_training_metrics(
        metric_logger.MetricLogger._log_training_metrics
    )
    warning_rank_0("[Patch:maxtext.opt_debug] MaxText training loop patched.")
