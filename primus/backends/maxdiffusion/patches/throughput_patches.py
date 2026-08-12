###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
MaxDiffusion Throughput Patch

MaxDiffusion's per-step line reports step time, FLOP rate and loss, but nothing
that says how much *data* a step moved::

    completed step: 18, seconds: 11.722, TFLOP/s/device: 348.548, loss: 1.540

This patch adds throughput to that line and to the metrics stream::

    completed step: 18, seconds: 11.722, TFLOP/s/device: 348.548,
    Tokens/s/device: 6756.526, Samples/s/device: 0.0853,
    Frames/s/device: 7.251, loss: 1.540

Everything is per device, like the ``TFLOP/s/device`` it sits next to:

``Samples/s/device``
    Videos (or images) per second per GPU. Defined for every model family.
``Frames/s/device``
    Video frames per second per GPU, i.e. ``Samples/s/device * num_frames``.
    Only emitted for families whose config describes frames.
``Tokens/s/device``
    Patchified latent tokens per second per GPU: 79,200 tokens for a
    720x1280x85 WAN video, 1,024 for a 512px FLUX image. Derived from the same
    shape ``calculate_{wan,flux}_tflops`` feeds its FLOP count, so token rate
    and FLOP rate always describe the same work, and unlike samples or frames
    it stays comparable across resolutions and clip lengths. Omitted for the
    UNet families (SD 1.x/2.x, SDXL, DreamBooth), which have no equivalent
    notion of a token.

The ``Tokens/s/device`` name matches MaxText's step line on purpose:
``parse_metrics_for_maxtext`` in ``tools/daily/daily_report.py`` then works
unchanged on MaxDiffusion logs.

The same numbers are written into ``metrics["scalar"]`` as ``perf/*`` entries
before any consumer runs, so they also reach TensorBoard and ``metrics_file``
alongside the upstream perf scalars, together with the cluster-wide
``perf/samples_per_second``, ``perf/frames_per_second`` and
``perf/tokens_per_second`` totals.
"""

from typing import Any, Dict, Optional

from primus.core.patches import PatchContext, register_patch
from primus.core.utils.module_utils import error_rank_0, log_rank_0, warning_rank_0

# WAN and FLUX both encode with a VAE that downsamples 8x in H and W and then
# patchify the latents 2x2 before the transformer blocks; WAN additionally
# compresses 4x along time.
_VAE_SCALE_FACTOR_SPATIAL = 8
_WAN_VAE_SCALE_FACTOR_TEMPORAL = 4
_TRANSFORMER_PATCH_SIZE = 2

# Model families whose sample -> token expansion is known. UNet families
# (SD 1.x/2.x, SDXL, DreamBooth) have no comparable token count, and LTX-Video
# uses a different VAE geometry, so they report samples only.
_WAN_MODEL_NAMES = frozenset({"wan2.1", "wan2.2"})
_FLUX_MODEL_NAMES = frozenset({"flux", "flux-dev", "flux-schnell"})

_work_per_step: Dict[str, float] = {}
_enabled = True
_tensorboard_hint_logged = False


def _config_keys(config: Any) -> Dict[str, Any]:
    """Return the config as a plain dict.

    ``HyperParameters.__getattr__`` raises ``ValueError`` (not
    ``AttributeError``) for unknown keys, so ``getattr(config, k, default)``
    is not safe here; MaxDiffusion configs differ per model family.
    """
    return config.get_keys()


def _effective_video_dims(keys: Dict[str, Any]) -> Dict[str, Any]:
    """Frame geometry the data iterator actually produces.

    ``get_wan_dimension`` honours ``synthetic_override_{height,width,num_frames}``
    -- and only those three -- when the synthetic iterator builds a batch, while
    ``WanTrainer.calculate_tflops`` always reads the plain config keys. Rates are
    reported for the shape that is really trained on, and a divergence is worth
    flagging because it means the upstream TFLOP/s number describes a different
    video than the one in the batch.
    """
    synthetic = keys.get("dataset_type") == "synthetic"
    dims = {}
    mismatched = []
    for name in ("height", "width", "num_frames"):
        configured = keys.get(name)
        override = keys.get(f"synthetic_override_{name}") if synthetic else None
        dims[name] = override or configured
        if override and configured and override != configured:
            mismatched.append(f"{name}: data={override} vs config={configured}")

    if mismatched:
        warning_rank_0(
            "[Patch:maxdiffusion.throughput] Synthetic data shape differs from the config shape "
            f"({', '.join(mismatched)}). Throughput follows the data; TFLOP/s/device does not."
        )
    return dims


def _tokens_per_sample(keys: Dict[str, Any], dims: Dict[str, Any]) -> Optional[int]:
    """Transformer tokens one sample expands into, or None where undefined.

    Both branches rebuild the latent sequence length that the data iterator and
    ``calculate_{wan,flux}_tflops`` use, so Tokens/s and TFLOP/s describe the
    same work. Text-conditioning tokens are excluded: they are a fixed
    per-sample cost that does not scale with resolution or clip length.
    """
    if keys.get("model_name") in _WAN_MODEL_NAMES:
        height, width, num_frames = dims["height"], dims["width"], dims["num_frames"]
        if not height or not width or not num_frames:
            return None

        latent_frames = (num_frames - 1) // _WAN_VAE_SCALE_FACTOR_TEMPORAL + 1
        latent_positions = (
            (height // _VAE_SCALE_FACTOR_SPATIAL) * (width // _VAE_SCALE_FACTOR_SPATIAL) * latent_frames
        )
        return int(latent_positions // _TRANSFORMER_PATCH_SIZE**2)

    if keys.get("flux_name") in _FLUX_MODEL_NAMES:
        resolution = keys.get("resolution")
        if not resolution:
            return None

        latent_side = resolution // (_VAE_SCALE_FACTOR_SPATIAL * _TRANSFORMER_PATCH_SIZE)
        return int(latent_side * latent_side)

    return None


def _resolve_work_per_step(config: Any) -> Dict[str, float]:
    """Samples/tokens/frames one training step consumes. Constant, so cached."""
    if _work_per_step:
        return _work_per_step

    import jax

    keys = _config_keys(config)
    per_device_samples = float(keys.get("per_device_batch_size") or 0.0)
    if per_device_samples <= 0:
        # Raise rather than report zeros; the caller disables the patch.
        raise ValueError(f"unusable per_device_batch_size={keys.get('per_device_batch_size')!r}")

    # data_sharding spreads the batch over every mesh axis, so a step trains
    # per_device_batch_size samples on each of jax.device_count() devices --
    # the same normalization calculate_tflops uses for TFLOP/s/device.
    global_samples = float(keys.get("global_batch_size_to_train_on") or per_device_samples * jax.device_count())

    work = {"per_device_samples": per_device_samples, "global_samples": global_samples}

    dims = _effective_video_dims(keys)
    tokens_per_sample = _tokens_per_sample(keys, dims)
    if tokens_per_sample:
        work["per_device_tokens"] = tokens_per_sample * per_device_samples
        work["global_tokens"] = tokens_per_sample * global_samples

    num_frames = dims["num_frames"]
    if num_frames:
        work["per_device_frames"] = float(num_frames) * per_device_samples
        work["global_frames"] = float(num_frames) * global_samples

    _work_per_step.update(work)
    log_rank_0(
        f"[Patch:maxdiffusion.throughput] Per step: {global_samples:g} samples "
        f"({per_device_samples:g}/device), {work.get('per_device_frames', 0):g} frames/device, "
        f"{work.get('per_device_tokens', 0):g} tokens/device"
    )
    return _work_per_step


def _throughput_scalars(config: Any, step_time_seconds: float) -> Dict[str, float]:
    if step_time_seconds <= 0:
        return {}

    work = _resolve_work_per_step(config)
    scalars = {
        "perf/samples_per_second": work["global_samples"] / step_time_seconds,
        "perf/samples_per_second_per_device": work["per_device_samples"] / step_time_seconds,
    }
    if "per_device_tokens" in work:
        scalars["perf/tokens_per_second"] = work["global_tokens"] / step_time_seconds
        scalars["perf/tokens_per_second_per_device"] = work["per_device_tokens"] / step_time_seconds
    if "per_device_frames" in work:
        scalars["perf/frames_per_second"] = work["global_frames"] / step_time_seconds
        scalars["perf/frames_per_second_per_device"] = work["per_device_frames"] / step_time_seconds
    return scalars


def _format_step_line(scalars: Dict[str, Any], step: Any) -> str:
    """Upstream's step line, widened with whichever throughput metrics exist."""
    message = "completed step: {}, seconds: {:.3f}, TFLOP/s/device: {:.3f}".format(
        step,
        scalars["perf/step_time_seconds"],
        scalars["perf/per_device_tflops_per_sec"],
    )
    if "perf/tokens_per_second_per_device" in scalars:
        message += ", Tokens/s/device: {:.3f}".format(scalars["perf/tokens_per_second_per_device"])
    if "perf/samples_per_second_per_device" in scalars:
        message += ", Samples/s/device: {:.4f}".format(scalars["perf/samples_per_second_per_device"])
    if "perf/frames_per_second_per_device" in scalars:
        message += ", Frames/s/device: {:.3f}".format(scalars["perf/frames_per_second_per_device"])
    return message + ", loss: {:.3f}".format(float(scalars["learning/loss"]))


@register_patch(
    patch_id="maxdiffusion.throughput",
    backend="maxdiffusion",
    phase="before_train",
    description="Report sample/token throughput in MaxDiffusion step metrics",
    condition=lambda ctx: True,  # Always enabled
)
def patch_maxdiffusion_throughput(ctx: PatchContext) -> None:
    """Add throughput to ``train_utils`` metric recording and step logging."""
    del ctx

    try:
        from maxdiffusion import max_logging, pyconfig, train_utils
    except ImportError as exc:  # noqa: BLE001 - never abort a run over metrics
        error_rank_0(f"[Patch:maxdiffusion.throughput] Failed to import maxdiffusion.train_utils: {exc!r}")
        return

    # Reset in case a previous run in this process left state behind.
    global _enabled, _tensorboard_hint_logged
    _enabled = True
    _tensorboard_hint_logged = False
    _work_per_step.clear()

    upstream_record_scalar_metrics = train_utils.record_scalar_metrics
    upstream_write_metrics_to_tensorboard = train_utils.write_metrics_to_tensorboard

    def record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr):
        upstream_record_scalar_metrics(metrics, step_time_delta, per_device_tflops, lr)

        # pyconfig.config is only populated once pyconfig.initialize() has run,
        # which happens after this patch is installed.
        global _enabled
        if not _enabled or pyconfig.config is None:
            return
        try:
            metrics["scalar"].update(_throughput_scalars(pyconfig.config, step_time_delta.total_seconds()))
        except Exception as exc:  # noqa: BLE001 - a metric must not kill training
            _enabled = False
            warning_rank_0(f"[Patch:maxdiffusion.throughput] Disabled after error: {exc!r}")

    def write_metrics_to_tensorboard(writer, metrics, step, config):
        """Upstream ``train_utils.write_metrics_to_tensorboard`` with a wider step line."""
        import jax
        import numpy as np

        try:
            step_line = _format_step_line(metrics["scalar"], step)
        except (KeyError, TypeError, ValueError) as exc:  # unexpected metric set
            warning_rank_0(f"[Patch:maxdiffusion.throughput] Falling back to upstream step line: {exc!r}")
            upstream_write_metrics_to_tensorboard(writer, metrics, step, config)
            return

        global _tensorboard_hint_logged
        if jax.process_index() == 0:
            max_logging.log(step_line)
            for metric_name in metrics.get("scalar", []):
                writer.add_scalar(metric_name, np.array(metrics["scalar"][metric_name]), step)
            for metric_name in metrics.get("scalars", []):
                writer.add_scalars(metric_name, metrics["scalars"][metric_name], step)

            # Upstream reprints the tensorboard hint on every log_period, which
            # at log_period=1 doubles the length of the step log to repeat a
            # string that never changes.
            if not _tensorboard_hint_logged:
                _tensorboard_hint_logged = True
                max_logging.log(f"To see full metrics 'tensorboard --logdir={config.tensorboard_dir}'")

            if step % config.log_period == 0:
                writer.flush()

    train_utils.record_scalar_metrics = record_scalar_metrics
    train_utils.write_metrics_to_tensorboard = write_metrics_to_tensorboard

    log_rank_0("[Patch:maxdiffusion.throughput] MaxDiffusion throughput metrics patched successfully.")
