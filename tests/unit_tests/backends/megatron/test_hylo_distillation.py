###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from primus.backends.megatron.training import hylo_distillation
from primus.backends.megatron.training.hylo_distillation import (
    HyLoRuntime,
    create_sft_forward_step,
    enhanced_ild_loss,
    masked_chunked_kl,
    set_teacher_yarn,
    validate_hylo_args,
)
from primus.core.config.primus_config import load_primus_config
from primus.core.utils import file_utils

ROOT = Path(__file__).resolve().parents[4]


def test_enhanced_ild_matches_hylo_equation():
    teacher_hidden = [torch.zeros(2, 3, 4), torch.ones(2, 3, 4)]
    student_hidden = [torch.ones(2, 3, 4), torch.full((2, 3, 4), 3.0)]
    teacher_mixer = [torch.full((2, 3, 4), 2.0), torch.full((2, 3, 4), 4.0)]
    student_mixer = [torch.full((2, 3, 4), 5.0), torch.full((2, 3, 4), 8.0)]
    loss, parts = enhanced_ild_loss(student_hidden, teacher_hidden, student_mixer, teacher_mixer)
    expected_hidden = sum(
        torch.norm(s - t, p=2, dim=-1).mean() for s, t in zip(student_hidden, teacher_hidden)
    )
    expected_mixer = sum(torch.norm(s - t, p=2, dim=-1).mean() for s, t in zip(student_mixer, teacher_mixer))
    assert torch.equal(loss, expected_hidden + expected_mixer)
    assert torch.equal(parts["hylo hidden ild"], expected_hidden)
    assert torch.equal(parts["hylo mixer ild"], expected_mixer)


def test_masked_chunked_kl_matches_dense_reference_and_has_grad():
    generator = torch.Generator().manual_seed(7)
    student = torch.randn(2, 5, 11, generator=generator, requires_grad=True)
    teacher = torch.randn(2, 5, 11, generator=generator)
    mask = torch.tensor([[1, 1, 0, 1, 0], [0, 1, 1, 1, 1]], dtype=torch.float32)
    actual, count = masked_chunked_kl(student, teacher, mask, chunk_size=3)
    per_token = F.kl_div(
        F.log_softmax(student.float(), dim=-1),
        F.softmax(teacher.float(), dim=-1),
        reduction="none",
    ).sum(dim=-1)
    expected = (per_token * mask).sum()
    torch.testing.assert_close(actual, expected)
    assert count.item() == int(mask.sum())
    actual.backward()
    assert student.grad is not None
    assert torch.count_nonzero(student.grad[mask == 0]) == 0


def test_validate_hylo_args_rejects_unsupported_parallelism():
    base = dict(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        hylo_distill_stage="sft",
        hylo_teacher_checkpoint="/tmp/teacher",
        hylo_ce_weight=0.0,
    )
    validate_hylo_args(Namespace(**base))
    with pytest.raises(ValueError, match="tensor_model_parallel_size=1"):
        validate_hylo_args(Namespace(**{**base, "tensor_model_parallel_size": 2}))
    with pytest.raises(ValueError, match="pipeline_model_parallel_size=1"):
        validate_hylo_args(Namespace(**{**base, "pipeline_model_parallel_size": 2}))
    with pytest.raises(ValueError, match="hylo_ce_weight=0"):
        validate_hylo_args(Namespace(**{**base, "hylo_ce_weight": 1.0}))
    with pytest.raises(ValueError, match="hylo_distill_stage"):
        validate_hylo_args(Namespace(**{**base, "hylo_distill_stage": "pretrain"}))
    with pytest.raises(ValueError, match="hylo_teacher_checkpoint"):
        validate_hylo_args(Namespace(**{**base, "hylo_teacher_checkpoint": None}))
    with pytest.raises(ValueError, match="original_max_position_embeddings"):
        validate_hylo_args(Namespace(**{**base, "hylo_teacher_yarn_factor": 32.0}))


def test_enhanced_ild_rejects_mismatched_captures():
    hidden = [torch.zeros(1, 2)]
    with pytest.raises(RuntimeError, match="capture mismatch"):
        enhanced_ild_loss(hidden, [], hidden, hidden)


def test_masked_chunked_kl_rejects_shape_mismatch():
    student = torch.zeros(2, 3)
    teacher = torch.zeros(2, 4)
    with pytest.raises(ValueError, match="student logits"):
        masked_chunked_kl(student, teacher)


class _Symbol:
    def __init__(self, value):
        self.value = value


class _Attention(nn.Module):
    def forward(self, hidden_states):
        return hidden_states


class _TeacherLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _Attention()

    def forward(self, hidden_states):
        return hidden_states


class _Teacher(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_TeacherLayer()])


class _Mixer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = nn.Identity()
        self.seen = None

    def forward(self, hidden_states=None):
        self.seen = hidden_states.detach().clone()
        return hidden_states


class _MLP(nn.Module):
    def forward(self, hidden_states=None):
        return hidden_states


class _Student(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.mixer = _Mixer()
        self.decoder.mlp = _MLP()
        self.decoder.layer_type_list = [_Symbol("M"), _Symbol("-")]
        self.decoder.layers = nn.ModuleList([self.decoder.mixer, self.decoder.mlp])


def test_teacher_forcing_replaces_only_ild_mixer_inputs():
    runtime = HyLoRuntime(_Teacher(), "ild")
    forced = torch.arange(4, dtype=torch.float32).view(1, 1, 4)
    runtime.teacher_capture.layer_inputs.append(forced)
    student = _Student()
    runtime.install_student_hooks(student)
    original = torch.zeros(1, 1, 4)

    student.decoder.layers[0](hidden_states=original)
    assert torch.equal(student.decoder.mixer.seen, forced)

    runtime.stage = "sft"
    student.decoder.layers[0](hidden_states=original)
    assert torch.equal(student.decoder.mixer.seen, original)


class _Scale(nn.Module):
    def __init__(self, factor):
        super().__init__()
        self.factor = factor

    def forward(self, hidden_states):
        return hidden_states * self.factor


class _ResidualTeacherLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.self_attention = _Attention()

    def forward(self, hidden_states):
        return hidden_states + self.self_attention(hidden_states)


class _TeacherGPT(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layers = nn.ModuleList([_ResidualTeacherLayer()])
        self.decoder.final_layernorm = _Scale(3.0)

    def forward(self, tokens, position_ids, attention_mask, labels=None):
        hidden = tokens.float().unsqueeze(-1).repeat(1, 1, 4)
        for layer in self.decoder.layers:
            hidden = layer(hidden)
        return self.decoder.final_layernorm(hidden)


class _ResidualMixerLayer(nn.Module):
    def __init__(self):
        super().__init__()
        self.mixer = _Scale(5.0)

    def forward(self, hidden_states=None):
        return hidden_states + self.mixer(hidden_states)


class _StudentHybrid(nn.Module):
    def __init__(self):
        super().__init__()
        self.decoder = nn.Module()
        self.decoder.layer_type_list = [_Symbol("M"), _Symbol("-")]
        self.decoder.layers = nn.ModuleList([_ResidualMixerLayer(), _MLP()])
        self.decoder.final_norm = _Scale(7.0)

    def forward(self, tokens, position_ids, attention_mask, labels=None):
        hidden = torch.zeros(*tokens.shape, 4)
        for layer in self.decoder.layers:
            hidden = layer(hidden_states=hidden)
        return self.decoder.final_norm(hidden)


def test_ild_step_scores_last_layer_after_final_norm_once_per_micro_batch(monkeypatch):
    monkeypatch.setattr(hylo_distillation, "_RUNTIME", HyLoRuntime(_TeacherGPT(), "ild"))
    monkeypatch.setattr(hylo_distillation, "_log_loss", lambda metrics: None)
    tokens = torch.tensor([[1, 2]])
    batch = {"input_ids": tokens, "loss_mask": torch.ones(1, 2)}
    _output, loss_func = create_sft_forward_step()(iter([batch]), _StudentHybrid())
    loss, count, report = loss_func(_output)

    # Teacher-forced input x: the teacher's layer gives 2x and its final norm 6x;
    # the student's mixer gives 5x, its layer 6x, and its final norm 42x.
    width = torch.norm(torch.ones(4)) * tokens.float().abs().mean()
    torch.testing.assert_close(report["hylo hidden ild"][0].cpu(), (42 - 6) * width)
    torch.testing.assert_close(report["hylo mixer ild"][0].cpu(), (5 - 1) * width)
    torch.testing.assert_close(loss.cpu(), 40 * width)
    assert count.item() == 1
    assert report["hylo ild"][1].item() == 1


def test_ild_requires_final_norm_on_both_models():
    runtime = HyLoRuntime(_TeacherGPT(), "ild")
    runtime.teacher_capture.final.append(torch.zeros(1))
    with pytest.raises(RuntimeError, match="final norm"):
        runtime.ild_captures()


class _InPlaceResidualTeacherLayer(nn.Module):
    """Adds the residual into the attention output in place, like Megatron's eval-mode bias-dropout-add."""

    def __init__(self):
        super().__init__()
        self.self_attention = _Scale(2.0)

    def forward(self, hidden_states):
        return self.self_attention(hidden_states).add_(hidden_states)


def test_teacher_mixer_capture_is_attention_output_without_residual():
    teacher = nn.Module()
    teacher.decoder = nn.Module()
    teacher.decoder.layers = nn.ModuleList([_InPlaceResidualTeacherLayer()])
    runtime = HyLoRuntime(teacher, "ild")
    with torch.no_grad():
        out = teacher.decoder.layers[0](torch.ones(2, 1, 4))
    assert torch.equal(out, torch.full((2, 1, 4), 3.0))
    assert torch.equal(runtime.teacher_capture.mixer[0], torch.full((2, 1, 4), 2.0))


def test_teacher_yarn_matches_hugging_face_llama_yarn():
    yarn = pytest.importorskip("megatron.core.models.common.embeddings.yarn_rotary_pos_embedding")
    rope_utils = pytest.importorskip("transformers.modeling_rope_utils")
    if not torch.cuda.is_available():
        pytest.skip("YarnRotaryEmbedding builds its frequencies on the GPU")
    from transformers import LlamaConfig

    config = SimpleNamespace()
    set_teacher_yarn(config, 32.0, 8192)
    rope = yarn.YarnRotaryEmbedding(
        kv_channels=64,
        rotary_base=500000.0,
        scaling_factor=config.yarn_rotary_scaling_factor,
        original_max_position_embeddings=config.yarn_original_max_position_embeddings,
        beta_fast=config.yarn_beta_fast,
        beta_slow=config.yarn_beta_slow,
        mscale=config.yarn_mscale,
        mscale_all_dim=config.yarn_mscale_all_dim,
        correction_range_round_to_int=config.yarn_correction_range_round_to_int,
    )
    emb, _ = rope.get_emb(16)
    # train_distill.py also raises the teacher's max_position_embeddings to factor * original.
    hf = LlamaConfig(
        hidden_size=2048,
        num_attention_heads=32,
        head_dim=64,
        rope_theta=500000.0,
        max_position_embeddings=32 * 8192,
        rope_scaling={"rope_type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192},
    )
    inv_freq, attention_factor = rope_utils.ROPE_INIT_FUNCTIONS["yarn"](hf, "cpu")
    expected = torch.outer(torch.arange(16, dtype=torch.float32), inv_freq)
    torch.testing.assert_close(emb[:, 0, 0, :32].cpu(), expected)
    concentration = yarn._yarn_get_concentration_factor_from_config(config)
    assert concentration == pytest.approx(attention_factor)


def test_hylo_recipes_match_stage_contract(monkeypatch):
    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
    recipes = {
        "examples/megatron/configs/MI300X/hylo_gdn_1B-enhanced-ild.yaml": ("ild", 4096, 0.0),
        "examples/megatron/configs/MI300X/hylo_mla_1B-enhanced-ild.yaml": ("ild", 4096, 1.0),
        "examples/megatron/configs/MI300X/hylo_m2_1B-enhanced-ild.yaml": ("ild", 4096, 0.0),
        "examples/megatron/configs/MI300X/hylo_4MLA12GDN_1B-8K-sft.yaml": ("sft", 8192, 0.25),
    }
    for relative, (stage, seq_length, attention_ratio) in recipes.items():
        module = load_primus_config(ROOT / relative, None).modules[0]
        params = module.params
        assert params.hylo_distill_stage == stage
        assert params.hylo_kl_weight == 1.0
        assert params.hylo_ce_weight == 0.0
        assert params.seq_length == seq_length
        assert params.tensor_model_parallel_size == 1
        assert params.pipeline_model_parallel_size == 1
        assert params.hybrid_attention_ratio == attention_ratio
        assert params.finetune is True
        assert params.no_load_optim is True
        assert params.no_load_rng is True


def test_hylo_m2_recipes_follow_zebra_m2_ild(monkeypatch):
    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
    configs = ROOT / "examples/megatron/configs/MI300X"
    ild = load_primus_config(configs / "hylo_m2_1B-enhanced-ild.yaml", None).modules[0].params
    constructor = load_primus_config(configs / "hylo_m2_1B-constructor.yaml", None).modules[0].params
    for params in (ild, constructor):
        assert params.spec[1] == "hylo_m2_hybrid_stack_spec"
        assert params.hybrid_override_pattern == "M-" * 16
        assert params.hybrid_linear_type == "mamba"
        assert (params.hidden_size, params.mamba_expand, params.mamba_d_xb) == (2048, 1, 512)
        assert (params.mamba_state_dim, params.mamba_head_dim, params.mamba_num_groups) == (64, 64, 32)

    assert ild.hylo_teacher_use_rope_scaling is True
    assert (ild.lr, ild.lr_warmup_fraction, ild.lr_decay_style) == (2.0e-4, 0.01, "cosine")
    assert (ild.micro_batch_size, ild.global_batch_size) == (12, 96)
    assert (ild.seed, ild.weight_decay, ild.adam_beta1, ild.adam_beta2) == (42, 0.0, 0.9, 0.999)
    assert (ild.adam_eps, ild.clip_grad) == (1.0e-8, 1.0)


def test_hylo_mla_recipes_follow_zebra_mla_ild(monkeypatch):
    monkeypatch.setattr(file_utils, "create_path_if_not_exists", lambda *args, **kwargs: None)
    configs = ROOT / "examples/megatron/configs/MI300X"
    ild = load_primus_config(configs / "hylo_mla_1B-enhanced-ild.yaml", None).modules[0].params
    constructor = load_primus_config(configs / "hylo_mla_1B-constructor.yaml", None).modules[0].params
    for params in (ild, constructor):
        assert params.spec[1] == "hylo_grouped_mla_hybrid_stack_spec"
        assert params.hybrid_override_pattern == "*-" * 16
        assert (params.num_attention_heads, params.hylo_mla_kv_heads) == (32, 8)
        assert (params.q_lora_rank, params.kv_lora_rank) == (1344, 128)
        assert (params.qk_head_dim, params.qk_pos_emb_head_dim, params.v_head_dim) == (32, 32, 64)

    assert (ild.rotary_scaling_factor, ild.original_max_position_embeddings) == (32.0, 8192)
    # HyLo's MLA rope_scaling has no mscale_all_dim, so the softmax scale stays 1/sqrt(q_head_dim).
    assert (ild.mscale, ild.mscale_all_dim) == (1.0, 0.0)
    assert ild.hylo_teacher_use_rope_scaling is True
    assert getattr(ild, "hylo_teacher_yarn_factor", None) is None
    assert (ild.lr, ild.lr_warmup_fraction, ild.lr_decay_style) == (2.0e-4, 0.01, "cosine")
    assert (ild.micro_batch_size, ild.global_batch_size) == (12, 96)
    assert (ild.seed, ild.weight_decay, ild.adam_beta1, ild.adam_beta2) == (42, 0.0, 0.9, 0.999)
    assert (ild.adam_eps, ild.clip_grad) == (1.0e-8, 1.0)
