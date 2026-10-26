###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""Native Megatron runtime for HyLo Enhanced-ILD and pure-KL SFT.

The first implementation intentionally supports TP=PP=1. Data parallel ranks
each own a frozen teacher, which keeps the layer mapping identical to the
reference Hugging Face implementation and avoids sharding ambiguities.
"""

from __future__ import annotations

import glob
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Optional

import torch
import torch.nn.functional as F

from primus.backends.megatron.training.hylo_order_dataset import unset

_KL_CHUNK = 4096
_RUNTIME: Optional["HyLoRuntime"] = None


def _unwrap(model: Any) -> Any:
    seen = set()
    while hasattr(model, "module") and id(model) not in seen:
        seen.add(id(model))
        model = model.module
    return model


def _tensor(output: Any) -> torch.Tensor:
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)) and output:
        return _tensor(output[0])
    raise TypeError(f"cannot extract tensor from {type(output).__name__}")


def _token_mean(distance: torch.Tensor, token_mask: Optional[torch.Tensor]) -> torch.Tensor:
    if token_mask is None:
        return distance.mean()
    if tuple(token_mask.shape) != tuple(distance.shape):
        raise ValueError(
            f"token_mask {tuple(token_mask.shape)} does not match the per-token distances "
            f"{tuple(distance.shape)} (sequence, batch)"
        )
    weights = token_mask.to(device=distance.device, dtype=torch.float32)
    total = (distance.float() * weights).sum()
    return (total / weights.sum().clamp_min(1.0)).to(distance.dtype)


def enhanced_ild_loss(
    student_hidden: Iterable[torch.Tensor],
    teacher_hidden: Iterable[torch.Tensor],
    student_mixer: Iterable[torch.Tensor],
    teacher_mixer: Iterable[torch.Tensor],
    token_mask: Optional[torch.Tensor] = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """HyLo Enhanced-ILD: sum layer hidden L2 and token-mixer-output L2.

    Each term is a mean over tokens. ``token_mask`` (sequence, batch) limits those means to real
    tokens; without it every position counts, as on packed data.
    """
    sh, th = list(student_hidden), list(teacher_hidden)
    sm, tm = list(student_mixer), list(teacher_mixer)
    if not sh or len(sh) != len(th) or len(sm) != len(tm) or len(sh) != len(sm):
        raise RuntimeError(
            "Enhanced-ILD capture mismatch: "
            f"student_hidden={len(sh)}, teacher_hidden={len(th)}, "
            f"student_mixer={len(sm)}, teacher_mixer={len(tm)}"
        )
    hidden_loss = sum(
        _token_mean(torch.norm(a - b.detach(), p=2, dim=-1), token_mask) for a, b in zip(sh, th)
    )
    mixer_loss = sum(_token_mean(torch.norm(a - b.detach(), p=2, dim=-1), token_mask) for a, b in zip(sm, tm))
    return hidden_loss + mixer_loss, {
        "hylo hidden ild": hidden_loss.detach(),
        "hylo mixer ild": mixer_loss.detach(),
    }


def masked_chunked_kl(
    student_logits: torch.Tensor,
    teacher_logits: torch.Tensor,
    loss_mask: Optional[torch.Tensor] = None,
    *,
    chunk_size: int = _KL_CHUNK,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute summed KL(teacher || student) without materializing softmaxes globally."""
    student = student_logits.reshape(-1, student_logits.shape[-1])
    teacher = teacher_logits.reshape(-1, teacher_logits.shape[-1])
    if student.shape != teacher.shape:
        raise ValueError(f"student logits {tuple(student.shape)} != teacher {tuple(teacher.shape)}")
    if loss_mask is None:
        mask = torch.ones(student.shape[0], device=student.device, dtype=torch.float32)
    else:
        mask = loss_mask.reshape(-1).to(device=student.device, dtype=torch.float32)
    total = torch.zeros((), device=student.device, dtype=torch.float32)
    for start in range(0, student.shape[0], chunk_size):
        stop = min(start + chunk_size, student.shape[0])
        s = student[start:stop].float()
        t = teacher[start:stop].float()
        per_token = F.kl_div(
            F.log_softmax(s, dim=-1),
            F.softmax(t, dim=-1),
            reduction="none",
        ).sum(dim=-1)
        total = total + (per_token * mask[start:stop]).sum()
    return total, mask.sum().detach().to(torch.int)


@dataclass
class _Capture:
    layer_inputs: list[torch.Tensor] = field(default_factory=list)
    hidden: list[torch.Tensor] = field(default_factory=list)
    mixer: list[torch.Tensor] = field(default_factory=list)
    final: list[torch.Tensor] = field(default_factory=list)

    def clear(self) -> None:
        self.layer_inputs.clear()
        self.hidden.clear()
        self.mixer.clear()
        self.final.clear()

    def ild_hidden(self) -> list[torch.Tensor]:
        """Layer outputs, the last after the final norm, like Hugging Face ``hidden_states[1:]``."""
        if not self.final:
            return list(self.hidden)
        return [*self.hidden[:-1], self.final[-1]]


class HyLoRuntime:
    """Own the frozen teacher and hook state for one Megatron process."""

    def __init__(self, teacher: torch.nn.Module, stage: str):
        self.teacher = teacher
        self.stage = stage
        self.teacher_capture = _Capture()
        self.student_capture = _Capture()
        self._student_hooked: set[int] = set()
        self._handles: list[Any] = []
        self._install_teacher_hooks()

    def _install_teacher_hooks(self) -> None:
        decoder = _unwrap(self.teacher).decoder
        self._hook_final(getattr(decoder, "final_layernorm", None), self.teacher_capture)
        for layer in decoder.layers:
            self._handles.append(
                layer.register_forward_pre_hook(
                    lambda _m, args, kwargs, cap=self.teacher_capture: self._capture_input(cap, args, kwargs),
                    with_kwargs=True,
                )
            )
            self._handles.append(
                layer.register_forward_hook(
                    lambda _m, _args, out, cap=self.teacher_capture: cap.hidden.append(_tensor(out))
                )
            )
            # Without grad and in eval mode, Megatron's bias-dropout-add adds the residual into
            # the attention output in place, so keep a copy of the attention output alone.
            self._handles.append(
                layer.self_attention.register_forward_hook(
                    lambda _m, _args, out, cap=self.teacher_capture: cap.mixer.append(_tensor(out).clone())
                )
            )

    @staticmethod
    def _capture_input(cap: _Capture, args: tuple, kwargs: dict):
        value = kwargs.get("hidden_states", args[0] if args else None)
        if value is not None:
            cap.layer_inputs.append(value)
        return args, kwargs

    def _hook_final(self, norm: Optional[torch.nn.Module], cap: _Capture) -> None:
        if norm is not None:
            self._handles.append(
                norm.register_forward_hook(lambda _m, _args, out: cap.final.append(_tensor(out)))
            )

    def ild_captures(self) -> tuple[list, list, list, list]:
        """``enhanced_ild_loss`` arguments: student and teacher hidden, then mixer outputs."""
        if bool(self.student_capture.final) != bool(self.teacher_capture.final):
            raise RuntimeError(
                "Enhanced-ILD needs the final norm output of both models or neither, got "
                f"student={len(self.student_capture.final)}, teacher={len(self.teacher_capture.final)}"
            )
        return (
            self.student_capture.ild_hidden(),
            self.teacher_capture.ild_hidden(),
            self.student_capture.mixer,
            self.teacher_capture.mixer,
        )

    def install_student_hooks(self, model: torch.nn.Module) -> None:
        base = _unwrap(model)
        if id(base) in self._student_hooked:
            return
        self._student_hooked.add(id(base))
        stack = base.decoder
        self._hook_final(getattr(stack, "final_norm", None), self.student_capture)
        mixer_slots = []
        mlp_slots = []
        for idx, (symbol, layer) in enumerate(zip(stack.layer_type_list, stack.layers)):
            value = getattr(symbol, "value", str(symbol))
            if value in ("M", "*"):
                mixer_slots.append((idx, layer, value))
            elif value == "-":
                mlp_slots.append((idx, layer))
        if len(mixer_slots) != len(mlp_slots):
            raise RuntimeError(
                f"HyLo requires paired mixer/MLP slots, got {len(mixer_slots)} and {len(mlp_slots)}"
            )
        for dense_idx, (_idx, layer, value) in enumerate(mixer_slots):
            self._handles.append(
                layer.register_forward_pre_hook(self._student_teacher_force_hook(dense_idx), with_kwargs=True)
            )
            mixer_module = layer.self_attention if value == "*" else getattr(layer, "mixer", layer)
            self._handles.append(
                mixer_module.register_forward_hook(
                    lambda _m, _args, out, cap=self.student_capture: cap.mixer.append(_tensor(out))
                )
            )
        for _idx, layer in mlp_slots:
            self._handles.append(
                layer.register_forward_hook(
                    lambda _m, _args, out, cap=self.student_capture: cap.hidden.append(_tensor(out))
                )
            )

    def _student_teacher_force_hook(self, dense_idx: int):
        def hook(_module, args, kwargs):
            if self.stage != "ild":
                return args, kwargs
            if dense_idx >= len(self.teacher_capture.layer_inputs):
                raise RuntimeError(f"missing teacher layer input {dense_idx}")
            forced = self.teacher_capture.layer_inputs[dense_idx].detach()
            if "hidden_states" in kwargs:
                kwargs = dict(kwargs)
                kwargs["hidden_states"] = forced
                return args, kwargs
            if not args:
                raise RuntimeError("student mixer received no hidden_states")
            return (forced, *args[1:]), kwargs

        return hook

    def clear(self) -> None:
        self.teacher_capture.clear()
        self.student_capture.clear()


def _resolve_legacy_checkpoint(path: str) -> str:
    root = Path(path)
    if root.is_file():
        return str(root)
    tracker = root / "latest_checkpointed_iteration.txt"
    if tracker.is_file():
        value = tracker.read_text().strip()
        iteration = "release" if value == "release" else f"iter_{int(value):07d}"
        root = root / iteration
    elif not root.name.startswith(("iter_", "release")):
        candidates = sorted(glob.glob(str(root / "iter_*")))
        if candidates:
            root = Path(candidates[-1])
    candidates = sorted(root.glob("mp_rank_00*/model_optim_rng.pt"))
    if not candidates:
        raise FileNotFoundError(f"no TP=1 legacy checkpoint below {path}")
    return str(candidates[0])


def _load_teacher_envelope(checkpoint: str) -> dict:
    envelope = torch.load(_resolve_legacy_checkpoint(checkpoint), map_location="cpu", weights_only=False)
    if not isinstance(envelope, dict):
        raise ValueError(f"teacher checkpoint {checkpoint} is not a mapping")
    return envelope


def _load_teacher_state(teacher: torch.nn.Module, envelope: dict) -> None:
    state = envelope.get("model", envelope)
    result = teacher.load_state_dict(state, strict=False)
    bad_missing = [key for key in result.missing_keys if not key.endswith("_extra_state")]
    bad_unexpected = [key for key in result.unexpected_keys if not key.endswith("_extra_state")]
    if bad_missing or bad_unexpected:
        raise RuntimeError(
            f"teacher checkpoint mismatch; missing={bad_missing[:8]}, unexpected={bad_unexpected[:8]}"
        )


def set_teacher_yarn(config: Any, factor: float, original_max_position_embeddings: int) -> None:
    """Hugging Face ``rope_type: yarn``, which HyLo's ``train_distill.py`` gives the teacher.

    With ``yarn_mscale`` 1 and ``yarn_mscale_all_dim`` 0, Megatron's attention
    scales cos and sin by ``0.1 * ln(factor) + 1``, Hugging Face's attention factor.
    """
    config.yarn_rotary_scaling_factor = factor
    config.yarn_original_max_position_embeddings = original_max_position_embeddings
    config.yarn_beta_fast = 32.0
    config.yarn_beta_slow = 1.0
    config.yarn_mscale = 1.0
    config.yarn_mscale_all_dim = 0.0
    config.yarn_correction_range_round_to_int = True


def _build_teacher(args, pre_process, post_process, config, pg_collection):
    from megatron.core.models.gpt import GPTModel
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.transformer.module import Float16Module
    from megatron.training.arguments import core_transformer_config_from_args

    envelope = _load_teacher_envelope(args.hylo_teacher_checkpoint)
    teacher_args = envelope.get("args")
    if teacher_args is None:
        raise ValueError("teacher checkpoint has no saved Megatron args")
    teacher_config = core_transformer_config_from_args(teacher_args)
    requested_layers = getattr(args, "hylo_teacher_num_layers", None)
    if requested_layers is not None and int(requested_layers) != teacher_config.num_layers:
        raise ValueError(
            f"teacher checkpoint has {teacher_config.num_layers} layers, "
            f"recipe requests {requested_layers}"
        )
    teacher_config.multi_latent_attention = False
    rope_scaling = getattr(args, "hylo_teacher_use_rope_scaling", None)
    if rope_scaling is None:
        rope_scaling = teacher_args.use_rope_scaling
    position_embedding_type = teacher_args.position_embedding_type
    yarn_factor = getattr(args, "hylo_teacher_yarn_factor", None)
    if not unset(yarn_factor):
        set_teacher_yarn(
            teacher_config,
            float(yarn_factor),
            int(args.hylo_teacher_yarn_original_max_position_embeddings),
        )
        position_embedding_type, rope_scaling = "yarn", False
    spec = get_gpt_layer_with_transformer_engine_spec(
        num_experts=None,
        moe_grouped_gemm=False,
        qk_layernorm=False,
        multi_latent_attention=False,
    )
    teacher = GPTModel(
        config=teacher_config,
        transformer_layer_spec=spec,
        vocab_size=teacher_args.padded_vocab_size,
        max_sequence_length=teacher_args.max_position_embeddings,
        pre_process=pre_process,
        post_process=post_process,
        fp16_lm_cross_entropy=teacher_args.fp16_lm_cross_entropy,
        parallel_output=False,
        share_embeddings_and_output_weights=not teacher_args.untie_embeddings_and_output_weights,
        position_embedding_type=position_embedding_type,
        rotary_percent=teacher_args.rotary_percent,
        rotary_base=teacher_args.rotary_base,
        rope_scaling=bool(rope_scaling),
        rope_scaling_factor=getattr(teacher_args, "rope_scaling_factor", 8.0),
        pg_collection=pg_collection,
    )
    _load_teacher_state(teacher, envelope)
    teacher.requires_grad_(False)
    if torch.cuda.is_available():
        teacher.cuda(torch.cuda.current_device())
    if args.fp16 or args.bf16:
        teacher = Float16Module(teacher_config, teacher)
    teacher.eval()
    return teacher


def wrap_model_provider(base_provider: Callable, args: Any) -> Callable:
    """Build the student normally and create one frozen dense teacher beside it."""

    def provider(*provider_args, **provider_kwargs):
        global _RUNTIME
        student = base_provider(*provider_args, **provider_kwargs)
        if _RUNTIME is None:
            config = provider_kwargs.get("config") or getattr(student, "config", None)
            if config is None:
                raise RuntimeError("HyLo could not resolve the student TransformerConfig")
            pre_process = provider_kwargs.get("pre_process", provider_args[0] if provider_args else True)
            post_process = provider_kwargs.get(
                "post_process", provider_args[1] if len(provider_args) > 1 else True
            )
            pg_collection = provider_kwargs.get("pg_collection")
            teacher = _build_teacher(args, pre_process, post_process, config, pg_collection)
            stage = str(args.hylo_distill_stage).lower()
            _RUNTIME = HyLoRuntime(teacher, stage)
        return student

    return provider


def validate_hylo_args(args: Any) -> None:
    if int(getattr(args, "tensor_model_parallel_size", 1)) != 1:
        raise ValueError("native HyLo currently requires tensor_model_parallel_size=1")
    if int(getattr(args, "pipeline_model_parallel_size", 1)) != 1:
        raise ValueError("native HyLo currently requires pipeline_model_parallel_size=1")
    stage = str(getattr(args, "hylo_distill_stage", "")).lower()
    if stage not in ("ild", "sft"):
        raise ValueError("hylo_distill_stage must be 'ild' or 'sft'")
    if not getattr(args, "hylo_teacher_checkpoint", None):
        raise ValueError("hylo_teacher_checkpoint is required")
    if not unset(getattr(args, "hylo_teacher_yarn_factor", None)) and unset(
        getattr(args, "hylo_teacher_yarn_original_max_position_embeddings", None)
    ):
        raise ValueError("hylo_teacher_yarn_factor needs hylo_teacher_yarn_original_max_position_embeddings")
    if stage == "sft" and float(getattr(args, "hylo_ce_weight", 0.0)) != 0.0:
        raise ValueError("the native 4MLA12GDN recipe requires hylo_ce_weight=0")


def _log_loss(metrics: dict[str, torch.Tensor]) -> None:
    from primus.core.utils.module_utils import log_rank_0

    parts = ", ".join(f"{name}={value.detach().float().item():.6f}" for name, value in metrics.items())
    log_rank_0(f"[HyLo] {parts}")


def sft_real_tokens(loss_mask: torch.Tensor) -> torch.Tensor:
    """Real-token mask, (sequence, batch), of an SFT micro-batch's ``loss_mask`` (batch, sequence).

    The SFT dataset pads each sample at its end and supervises it through its last real token, so
    a position is real when it is at or before the sample's last supervised position. Prompt
    tokens count, as they do on packed data.
    """
    supervised_from_here = torch.flip(torch.cumsum(torch.flip(loss_mask, dims=[-1]), dim=-1), dims=[-1])
    return (supervised_from_here > 0).transpose(0, 1)


def _ild_count(loss: torch.Tensor) -> torch.Tensor:
    # Enhanced-ILD is already a per-token mean, and Megatron divides each
    # micro-batch loss by this count, so one micro-batch counts once.
    return torch.ones((), dtype=torch.int, device=loss.device)


def _loss_result(loss: torch.Tensor, tokens: torch.Tensor, metrics: dict[str, torch.Tensor]):
    _log_loss(metrics)
    report = {
        name: torch.cat([value.detach().float().view(1), tokens.detach().float().view(1)])
        for name, value in metrics.items()
    }
    return loss, tokens, report


def create_pretrain_forward_step():
    from megatron.training import get_args
    from pretrain_mamba import get_batch

    def forward_step(data_iterator, model, return_schedule_plan=False):
        if return_schedule_plan:
            raise NotImplementedError("HyLo distillation does not support schedule plans")
        assert _RUNTIME is not None
        tokens, labels, loss_mask, attention_mask, position_ids, *_ = get_batch(data_iterator)
        _RUNTIME.install_student_hooks(model)
        _RUNTIME.clear()
        with torch.no_grad():
            teacher_logits = _RUNTIME.teacher(tokens, position_ids, attention_mask, labels=None)
        student_logits = model(tokens, position_ids, attention_mask, labels=None)

        def loss_func(_output):
            if _RUNTIME.stage == "ild":
                loss, parts = enhanced_ild_loss(*_RUNTIME.ild_captures())
                return _loss_result(loss, _ild_count(loss), {"hylo ild": loss.detach(), **parts})
            loss, tokens_count = masked_chunked_kl(
                student_logits,
                teacher_logits.detach(),
                loss_mask,
                chunk_size=int(getattr(get_args(), "hylo_kl_chunk_size", _KL_CHUNK)),
            )
            weighted_loss = loss * float(getattr(get_args(), "hylo_kl_weight", 1.0))
            return _loss_result(weighted_loss, tokens_count, {"hylo kl": loss.detach()})

        return student_logits, loss_func

    return forward_step


def create_sft_forward_step():
    """SFT forward step for the exact HyLo pure teacher-KL objective."""

    def forward_step(data_iterator, model, return_schedule_plan=False):
        if return_schedule_plan:
            raise NotImplementedError("HyLo SFT does not support schedule plans")
        assert _RUNTIME is not None
        if data_iterator is None:
            return None, lambda _output: _loss_result(torch.tensor(0.0), torch.tensor(0, dtype=torch.int), {})
        try:
            batch = next(data_iterator)
        except StopIteration:
            return None, lambda _output: _loss_result(torch.tensor(0.0), torch.tensor(0, dtype=torch.int), {})
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokens = batch["input_ids"].to(device=device, dtype=torch.long)
        loss_mask = batch["loss_mask"].to(device=device, dtype=torch.float32)
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)
            loss_mask = loss_mask.unsqueeze(0)
        position_ids = torch.arange(tokens.shape[1], device=device).unsqueeze(0).expand_as(tokens)
        _RUNTIME.install_student_hooks(model)
        _RUNTIME.clear()
        with torch.no_grad():
            teacher_logits = _RUNTIME.teacher(tokens, position_ids, None, labels=None)
        student_logits = model(tokens, position_ids, None, labels=None)

        def loss_func(_output):
            if _RUNTIME.stage == "ild":
                loss, parts = enhanced_ild_loss(
                    *_RUNTIME.ild_captures(), token_mask=sft_real_tokens(loss_mask)
                )
                return _loss_result(loss, _ild_count(loss), {"hylo ild": loss.detach(), **parts})
            from megatron.training import get_args

            loss, count = masked_chunked_kl(
                student_logits,
                teacher_logits.detach(),
                loss_mask,
                chunk_size=int(getattr(get_args(), "hylo_kl_chunk_size", _KL_CHUNK)),
            )
            weighted_loss = loss * float(getattr(get_args(), "hylo_kl_weight", 1.0))
            return _loss_result(weighted_loss, count, {"hylo kl": loss.detach()})

        return student_logits, loss_func

    return forward_step


def reset_runtime_for_tests() -> None:
    global _RUNTIME
    _RUNTIME = None
