###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Mistral Large 4: CPU tests of the patches, the shared layout helpers, the PP latent carrier,
the posttrain hook's family resolution and the model-provider registration."""

import importlib.util
import json
import re
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
    kv_carrier_needed,
    kv_sharing_active,
    moe_layer_freq_list,
)
from primus.backends.megatron.patches import (
    mistral_large4_args_patches,
    mistral_large4_pp_shape_patches,
)
from primus.core.patches.context import PatchContext

RELEASE_KV = [*range(39), 35, 40, 41, 42, 35] + [x for i in range(44, 72, 4) for x in (i, i + 1, i + 2, 35)]
REPO = Path(__file__).resolve().parents[5]


def _ctx(phase, backend_args=None, params=None):
    return PatchContext(
        backend="megatron",
        phase=phase,
        extra={"backend_args": backend_args, "module_config": SimpleNamespace(params=params)},
    )


# ---- shared layout helpers ------------------------------------------------------------------
@pytest.mark.parametrize(
    "k,n,experts,want",
    [
        (1, 4, 8, [0, 1, 1, 1]),
        (0, 3, 8, [1, 1, 1]),
        (6, 4, 8, [0, 0, 0, 0]),
        (None, 3, 8, [0, 1, 1]),
        (1, 3, 0, [0, 0, 0]),
        (1, 3, None, [0, 0, 0]),
    ],
)
def test_moe_layer_freq_list(k, n, experts, want):
    assert moe_layer_freq_list(k, n, experts) == want


@pytest.mark.parametrize("value", ["3", 3, "[1, ", "[None, 1]", [], [1.7, 0.2], [True, False]])
def test_normalize_layer_list_refuses(value):
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
        normalize_layer_list,
    )

    with pytest.raises(ValueError, match="no_rope_layers"):
        normalize_layer_list(value, 2, "no_rope_layers", int)


def test_moe_layer_freq_list_refuses_negative_k():
    with pytest.raises(ValueError):
        moe_layer_freq_list(-2, 4, 8)


def test_kv_sharing_helpers():
    assert len(RELEASE_KV) == 72
    assert kv_sharing_active(RELEASE_KV, 72) and not kv_sharing_active(RELEASE_KV, 39)
    assert kv_sharing_active(str(RELEASE_KV), 40)
    assert not kv_sharing_active(None, 4)
    assert kv_carrier_needed(RELEASE_KV, 72, 6) and not kv_carrier_needed(RELEASE_KV, 72, 1)
    assert not kv_carrier_needed(RELEASE_KV, 8, 2)


# ---- args patches ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "n,k_args,k_params,want",
    [
        (4, 1, None, [0, 1, 1, 1]),
        (8, None, 1, [0] + [1] * 7),
        (3, 5, None, [0, 0, 0]),
        (2, None, None, [0, 1]),
    ],
)
def test_moe_layer_freq_patch(n, k_args, k_params, want):
    # build_args: model_type and first_k_dense_replace are Primus keys, only in the module params
    args = SimpleNamespace(num_layers=n, num_experts=16, moe_layer_freq=1)  # the preset's value
    if k_args is not None:
        args.first_k_dense_replace = k_args
    params = SimpleNamespace(model_type="mistral_large4")
    if k_params is not None:
        params.first_k_dense_replace = k_params
    ctx = _ctx("build_args", args, params)
    assert mistral_large4_args_patches._is_mistral_large4(ctx)
    mistral_large4_args_patches.patch_mistral_large4_moe_layer_freq(ctx)
    assert args.moe_layer_freq == want


@pytest.fixture
def megatron_moe_freq_type(monkeypatch):
    """Megatron's ``moe_freq_type``. Importing ``megatron.training.arguments`` pulls in GPU-only
    packages (fla, mamba_ssm), so unless it is already loaded a stand-in with Megatron's own code
    (the character check and ``eval`` of ``_eval_pattern``) is installed for the patch to import."""
    import sys
    import types

    if "megatron.training.arguments" in sys.modules:
        return  # the real one (a GPU job that has imported it)

    def moe_freq_type(x):
        if isinstance(x, int):
            return x
        if "[" in x:
            if re.search(r"[^,\d\[\]\(\)\+\*]", x):
                raise ValueError(f"Invalid pattern: {x}")
            return eval(x)
        return int(x)

    module = types.ModuleType("megatron.training.arguments")
    module.moe_freq_type = moe_freq_type
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", module)


@pytest.mark.parametrize(
    "given,k,want",
    [
        (1, 1, [0, 1, 1, 1]),  # the preset (Megatron's default): derived
        (1, 2, [0, 0, 1, 1]),  # a non-default k (user or checkpoint) needs no moe_layer_freq override
        ("([0]*1+[1]*3)", 1, [0, 1, 1, 1]),
        ("[0, 0, 1, 1]", 2, [0, 0, 1, 1]),
        ([0, 1, 1, 1], 1, [0, 1, 1, 1]),
        ("1", 1, [0, 1, 1, 1]),  # the CLI / env spelling of the default
        (" 1 ", 2, [0, 0, 1, 1]),
        ("[1]*4", 0, [1, 1, 1, 1]),  # Megatron's other expression spellings
        ("[0]+[1]*3", 1, [0, 1, 1, 1]),
    ],
)
def test_moe_layer_freq_patch_accepts(given, k, want, megatron_moe_freq_type):
    params = SimpleNamespace(model_type="mistral_large4")
    args = SimpleNamespace(num_layers=4, num_experts=16, first_k_dense_replace=k, moe_layer_freq=given)
    mistral_large4_args_patches.patch_mistral_large4_moe_layer_freq(_ctx("build_args", args, params))
    assert args.moe_layer_freq == want


DISAGREES = "disagrees with first_k_dense_replace"
NOT_SUPPORTED = "is not supported"
NOT_01 = "is not a list of 0 / 1"
NOT_EXPR = "Megatron list expression"


@pytest.mark.parametrize(
    "given,match",
    [
        ([1, 1, 1, 1], DISAGREES),
        ("([0]*1+[1]*1)*2", DISAGREES),  # evaluates, but to a different layout
        ("([0]*1+[1]*71)", DISAGREES),  # wrong length
        ([0, 1, 1, 1, 1], DISAGREES),
        ("([0]*3+[1]*N)", NOT_EXPR),
        ("garbage", NOT_EXPR),
        ("__import__('os')", NOT_EXPR),
        ("[1]*'4'", NOT_EXPR),
        (2, NOT_SUPPORTED),
        ("2", NOT_SUPPORTED),
        ([False, True, True, True], NOT_01),
        ([None, 1, 1, 1], NOT_01),
    ],
)
def test_moe_layer_freq_patch_refuses(given, match, megatron_moe_freq_type):
    """A list that disagrees with first_k_dense_replace, a wrong length or an unparsable value
    raises (never silently replaced), each with the message of its branch."""
    params = SimpleNamespace(model_type="mistral_large4")
    args = SimpleNamespace(num_layers=4, num_experts=16, first_k_dense_replace=1, moe_layer_freq=given)
    with pytest.raises(ValueError, match=re.escape(match)):
        mistral_large4_args_patches.patch_mistral_large4_moe_layer_freq(_ctx("build_args", args, params))


def test_moe_layer_freq_patch_other_models():
    ctx = _ctx(
        "build_args", SimpleNamespace(num_layers=2, moe_layer_freq=1), SimpleNamespace(model_type="gpt")
    )
    assert not mistral_large4_args_patches._is_mistral_large4(ctx)


@pytest.mark.parametrize(
    "tp,etp,ep,deepep,hit",
    [
        (1, None, 1, True, True),
        (2, None, 1, True, False),
        (2, 1, 1, True, True),  # TP2 with expert-TP 1 and EP1: one expert rank (the proxy's TP2 layout)
        (1, None, 8, True, False),
        (2, 1, 4, True, False),
        (1, None, 1, False, False),
    ],
)
def test_deepep_single_rank(tp, etp, ep, deepep, hit):
    params = SimpleNamespace(
        model_type="mistral_large4",
        tensor_model_parallel_size=tp,
        expert_tensor_parallel_size=etp,
        expert_model_parallel_size=ep,
        use_turbo_deepep=deepep,
        moe_token_dispatcher_type="flex",
    )
    ctx = _ctx("setup", params=params)
    assert mistral_large4_args_patches._deepep_unusable(ctx) is hit
    params.model_type = "deepseek_v3"  # the same layout on another model: never selected
    assert not mistral_large4_args_patches._deepep_unusable(ctx)
    params.model_type = "mistral_large4"
    if hit:
        mistral_large4_args_patches.patch_mistral_large4_deepep_single_rank(ctx)
        assert params.use_turbo_deepep is False and params.moe_token_dispatcher_type == "alltoall"


@pytest.mark.parametrize(
    "overrides,hit",
    [
        ({}, False),  # plain training: Primus-Turbo grouped GEMM
        ({"use_megatron_fsdp": True}, True),
        ({"ckpt_format": "torch_dist", "save": "/ckpt"}, True),
        ({"ckpt_format": "torch_dist", "load": "/ckpt"}, True),
        ({"ckpt_format": "torch_dist", "pretrained_checkpoint": "/ckpt"}, True),  # SFT from a checkpoint
        ({"ckpt_format": "torch_dist", "save": "/ckpt", "use_distributed_optimizer": True}, False),
        ({"ckpt_format": "torch", "save": "/ckpt"}, False),
        ({"ckpt_format": "torch_dist"}, False),  # nothing saved or loaded: Turbo grouped GEMM kept
        ({"use_megatron_fsdp": True, "use_turbo_grouped_gemm": False}, False),
        ({"tensor_model_parallel_size": 8, "expert_tensor_parallel_size": 8}, True),
        ({"tensor_model_parallel_size": 8, "expert_tensor_parallel_size": 1}, False),
        ({"tensor_model_parallel_size": 2, "expert_tensor_parallel_size": None}, True),  # ETP defaults to TP
    ],
)
def test_turbo_grouped_gemm_fallback(overrides, hit):
    params = SimpleNamespace(
        model_type="mistral_large4",
        use_turbo_grouped_gemm=True,
        moe_grouped_gemm=True,
        use_megatron_fsdp=False,
        use_distributed_optimizer=False,
        ckpt_format="torch",
        save=None,
        load=None,
        pretrained_checkpoint=None,
        tensor_model_parallel_size=1,
        expert_tensor_parallel_size=1,
    )
    for k, v in overrides.items():
        setattr(params, k, v)
    ctx = _ctx("setup", params=params)
    assert mistral_large4_args_patches._turbo_grouped_gemm_unusable(ctx) is hit
    params.model_type = "deepseek_v3"  # the same settings on another model: never selected
    assert not mistral_large4_args_patches._turbo_grouped_gemm_unusable(ctx)
    params.model_type = "mistral_large4"
    if hit:
        mistral_large4_args_patches.patch_mistral_large4_turbo_grouped_gemm_fallback(ctx)
        assert params.use_turbo_grouped_gemm is False and params.moe_grouped_gemm is False


# ---- PP shape patch -------------------------------------------------------------------------
@pytest.mark.parametrize(
    "model,pp,n,want",
    [
        ("mistral_large4", 6, 72, True),
        ("mistral_large4", 1, 72, False),
        ("mistral_large4", 2, 8, False),
        ("gpt", 6, 72, False),
    ],
)
def test_pp_shape_condition(model, pp, n, want):
    params = SimpleNamespace(
        model_type=model, pipeline_model_parallel_size=pp, num_layers=n, kv_source_layers=str(RELEASE_KV)
    )
    assert (
        mistral_large4_pp_shape_patches._wants_mistral_large4_pp_shape_patch(
            _ctx("before_train", params=params)
        )
        is want
    )


def test_pp_shape_patch_adds_the_latent_rows(monkeypatch):
    """The P2P shape gets ceil(s * kv_lora_rank / h) rows for the packed latent (here s 1024,
    r 16, h 64: 256 rows), on the 1F1B path (stage-local s) and the interleaved path (its
    seq_length, divided by CP and, with SP, TP inside the schedule)."""
    import megatron.core.pipeline_parallel.schedules as sched

    monkeypatch.setattr(sched, "get_tensor_shapes", lambda *a, **k: [(1024, 1, 64), ()])
    monkeypatch.setattr(sched, "forward_backward_pipelining_with_interleaving", lambda **k: k)
    params = SimpleNamespace(
        model_type="mistral_large4",
        pipeline_model_parallel_size=2,
        num_layers=4,
        kv_source_layers=[0, 1, 2, 1],
        kv_lora_rank=16,
        hidden_size=64,
        sequence_parallel=True,
    )
    pgs = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2), tp=SimpleNamespace(size=lambda: 2))
    mistral_large4_pp_shape_patches.patch_mistral_large4_pp_tensor_shape(_ctx("before_train", params=params))
    assert sched.get_tensor_shapes() == [(1024 + 256, 1, 64), ()]
    # 4096 // (cp 2 * tp 2) = 1024 local rows -> 1280; the schedule divides 5120 by 4 again
    out = sched.forward_backward_pipelining_with_interleaving(seq_length=4096, pg_collection=pgs)
    assert out["seq_length"] == 1280 * 4
    # applied again in the same process: not sized twice
    mistral_large4_pp_shape_patches.patch_mistral_large4_pp_tensor_shape(_ctx("before_train", params=params))
    assert sched.get_tensor_shapes() == [(1280, 1, 64), ()]
    assert (
        sched.forward_backward_pipelining_with_interleaving(seq_length=4096, pg_collection=pgs)["seq_length"]
        == 5120
    )
    # another model's wrapper is refused, not taken as already sized
    from primus.backends.megatron.patches.kimi_k3_pp_shape_patches import (
        _make_k3_get_tensor_shapes,
    )

    monkeypatch.setattr(sched, "get_tensor_shapes", _make_k3_get_tensor_shapes(lambda: [(8, 1, 4)], 2))
    with pytest.raises(RuntimeError, match="another patch"):
        mistral_large4_pp_shape_patches.patch_mistral_large4_pp_tensor_shape(
            _ctx("before_train", params=params)
        )


@pytest.mark.parametrize("sequence_parallel,want", [(True, 1282 * 4), (False, 2563 * 2)])
def test_pp_shape_interleaved_divides_by_tp_only_with_sp(monkeypatch, sequence_parallel, want):
    """The interleaved schedule's seq_length is divided by cp x tp with sequence parallelism and by
    cp alone without it (TP 2, CP 2). seq_length 4100 makes the two differ: with SP the stage-local
    s is 1025 (+ ceil(1025 * 16 / 64) = 257 rows), without SP 2050 (+ 513 rows)."""
    import megatron.core.pipeline_parallel.schedules as sched

    monkeypatch.setattr(sched, "get_tensor_shapes", lambda *a, **k: [(1024, 1, 64)])
    monkeypatch.setattr(sched, "forward_backward_pipelining_with_interleaving", lambda **k: k)
    params = SimpleNamespace(
        model_type="mistral_large4",
        pipeline_model_parallel_size=2,
        num_layers=4,
        kv_source_layers=[0, 1, 2, 1],
        kv_lora_rank=16,
        hidden_size=64,
        sequence_parallel=sequence_parallel,
    )
    pgs = SimpleNamespace(cp=SimpleNamespace(size=lambda: 2), tp=SimpleNamespace(size=lambda: 2))
    mistral_large4_pp_shape_patches.patch_mistral_large4_pp_tensor_shape(_ctx("before_train", params=params))
    out = sched.forward_backward_pipelining_with_interleaving(seq_length=4100, pg_collection=pgs)
    assert out["seq_length"] == want


def _benign_wraps(fn):
    """A decorator that does not change the shape (profiling / NVTX style)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*a, **k):
        return fn(*a, **k)

    return wrapper


def test_pp_shape_patch_wrapper_ownership(monkeypatch):
    """A benign functools.wraps wrapper (no wire mark) does not stop the patch; our own wrapper
    under a benign one is recognised (not sized twice); a wire-changing wrapper of another model on
    the interleaved schedule is refused like one on get_tensor_shapes."""
    import megatron.core.pipeline_parallel.schedules as sched

    m = mistral_large4_pp_shape_patches
    params = SimpleNamespace(
        model_type="mistral_large4",
        pipeline_model_parallel_size=2,
        num_layers=4,
        kv_source_layers=[0, 1, 2, 1],
        kv_lora_rank=16,
        hidden_size=64,
        sequence_parallel=False,
    )
    ctx = _ctx("before_train", params=params)
    pgs = SimpleNamespace(cp=SimpleNamespace(size=lambda: 1), tp=SimpleNamespace(size=lambda: 1))

    # benign wrappers on both functions: the patch applies on top
    monkeypatch.setattr(sched, "get_tensor_shapes", _benign_wraps(lambda *a, **k: [(1024, 1, 64)]))
    monkeypatch.setattr(sched, "forward_backward_pipelining_with_interleaving", _benign_wraps(lambda **k: k))
    m.patch_mistral_large4_pp_tensor_shape(ctx)
    assert sched.get_tensor_shapes() == [(1280, 1, 64)]
    assert getattr(sched.get_tensor_shapes, m.WIRE_OWNER_ATTR) == "mistral_large4"

    # a benign wrapper over our wrappers: recognised as ours, not sized twice
    monkeypatch.setattr(sched, "get_tensor_shapes", _benign_wraps(sched.get_tensor_shapes))
    monkeypatch.setattr(
        sched,
        "forward_backward_pipelining_with_interleaving",
        _benign_wraps(sched.forward_backward_pipelining_with_interleaving),
    )
    m.patch_mistral_large4_pp_tensor_shape(ctx)
    assert sched.get_tensor_shapes() == [(1280, 1, 64)]
    assert (
        sched.forward_backward_pipelining_with_interleaving(seq_length=1024, pg_collection=pgs)["seq_length"]
        == 1280
    )

    # another wire-changing patch on the interleaved schedule only: refused
    def foreign(**k):
        return k

    setattr(foreign, m.WIRE_OWNER_ATTR, "another_model")
    monkeypatch.setattr(sched, "get_tensor_shapes", lambda *a, **k: [(1024, 1, 64)])
    monkeypatch.setattr(sched, "forward_backward_pipelining_with_interleaving", foreign)
    with pytest.raises(RuntimeError, match="forward_backward_pipelining_with_interleaving"):
        m.patch_mistral_large4_pp_tensor_shape(ctx)


@pytest.mark.parametrize("mark", ["_k3_pp_seq_mult", "_v4_pp_shape_patched", "_v4_pp_interleaved_patched"])
def test_pp_wire_foreign_wrapper_over_ours_is_refused(mark):
    """Another model's wire patch that wraps ours (functools.wraps copies our marks onto it) is
    foreign, not ours: its mark is checked before ours."""
    import functools

    from primus.backends.megatron.patches import mistral_large4_pp_shape_patches as m

    ours = m.make_get_tensor_shapes(lambda *a, **k: [(8, 1, 8)], 4)
    assert m._wire_state(ours) == "ours"
    assert m._wire_state(_benign_wraps(ours)) == "ours"

    @functools.wraps(ours)
    def theirs(*a, **k):
        return ours(*a, **k)

    setattr(theirs, mark, True)
    assert m._wire_state(theirs) == "foreign"
    assert m._wire_state(_benign_wraps(theirs)) == "foreign"


@pytest.mark.parametrize(
    "s,r,h,want", [(2048, 512, 6144, 171), (4096, 512, 6144, 342), (3, 4, 8, 2), (1, 1, 8, 1)]
)
def test_kv_carrier_rows(s, r, h, want):
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
        kv_carrier_rows,
    )

    assert kv_carrier_rows(s, r, h) == want


@pytest.mark.parametrize("flag", ["patch_zero_bubble", "patch_primus_pipeline"])
def test_pp_shape_patch_refuses_zero_bubble(flag):
    params = SimpleNamespace(
        model_type="mistral_large4",
        pipeline_model_parallel_size=2,
        num_layers=4,
        kv_source_layers=[0, 1, 2, 1],
    )
    setattr(params, flag, True)
    with pytest.raises(NotImplementedError, match=flag):
        mistral_large4_pp_shape_patches.patch_mistral_large4_pp_tensor_shape(
            _ctx("before_train", params=params)
        )


def test_patch_modules_import_without_megatron():
    """Primus imports every patch module before Megatron is on sys.path."""
    import subprocess
    import sys

    code = (
        "import sys; sys.path = [p for p in sys.path if 'Megatron-LM' not in p]; "
        "import primus.backends.megatron.patches.mistral_large4_args_patches, "
        "primus.backends.megatron.patches.mistral_large4_pp_shape_patches, "
        "primus.backends.megatron.patches.mistral_large4_training_log_patches; "
        "assert 'megatron.core' not in sys.modules"
    )
    env = {"PYTHONPATH": str(REPO), "PATH": "/usr/bin:/bin"}
    subprocess.run([sys.executable, "-c", code], check=True, cwd=str(REPO), env=env)


# ---- config validation ----------------------------------------------------------------------
def _cfg(**kw):
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_transformer_config import (
        MistralLarge4TransformerConfig,
    )

    base = dict(
        num_layers=4,
        hidden_size=64,
        num_attention_heads=4,
        kv_channels=16,
        q_lora_rank=None,
        kv_lora_rank=16,
        qk_head_dim=8,
        qk_pos_emb_head_dim=8,
        v_head_dim=16,
        mscale_all_dim=0.0,
        use_cpu_initialization=True,
    )
    base.update(kw)
    return MistralLarge4TransformerConfig(**base)


@pytest.mark.parametrize(
    "srcs,rope,exc,match",
    [
        ([0, 1, 2, -1], [0, 0, 0, 0], ValueError, "breaks the rule 0 <= s"),
        ([0, 1, 3, 3], [0, 0, 0, 0], ValueError, "breaks the rule 0 <= s"),
        ([0, 0, 2, 2], [0, 0, 0, 0], NotImplementedError, "more than one source layer"),
        ([0, 1, 2, 1], [1, 1, 1, 0], ValueError, "rope/NoPE kinds differ"),
    ],
)
def test_kv_source_validation(srcs, rope, exc, match):
    with pytest.raises(exc, match=match):
        _cfg(kv_source_layers=srcs, no_rope_layers=rope)


def test_single_kv_source():
    cfg = _cfg(kv_source_layers=[0, 1, 2, 1], no_rope_layers=[1, 0, 1, 0])
    assert cfg.kv_source_layer == 1 and cfg.stores_kv(1) and cfg.multi_latent_attention


# ---- posttrain hook family resolution and the model-provider registration ---------------------
def _hook():
    path = REPO / "runner/helpers/hooks/train/posttrain/megatron/01_convert_checkpoints.py"
    spec = importlib.util.spec_from_file_location("ml4_convert_hook", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_hook_hands_the_conversion_to_build_args(tmp_path, capsys):
    """The ordering posttrain_needs_checkpoint relies on: the posttrain hook prints
    ``extra.pretrained_checkpoint=<converted dir>`` (execute_hooks.sh turns it into
    ``--pretrained_checkpoint``, the trainer's build_args then sees it), and with it the refusal
    does not fire, whatever hf_path holds."""
    from primus.core.patches.patch_registry import PatchRegistry

    conv = tmp_path / "conv"  # a finished conversion: Megatron's tracker file beside release/
    conv.mkdir()
    (conv / "latest_checkpointed_iteration.txt").write_text("release")
    _hook()._emit_native_checkpoint_args(conv)
    lines = dict(
        line.split("=", 1) for line in capsys.readouterr().out.splitlines() if line.startswith("extra.")
    )
    assert lines["extra.pretrained_checkpoint"] == str(tmp_path / "conv")
    patch = PatchRegistry.get("megatron.mistral_large4.posttrain_needs_checkpoint")
    params = SimpleNamespace(
        model_type="mistral_large4",
        stage="sft",
        hf_path="mistralai/Mistral-Large-4-1T-A52B",
        pretrained_checkpoint=lines["extra.pretrained_checkpoint"],
    )
    assert not patch.condition(_ctx("build_args", SimpleNamespace(), params))


@pytest.mark.parametrize(
    "cfg",
    [
        {"architectures": ["MistralLarge4ForConditionalGeneration"]},
        {"model_type": "mistral_large4"},
        {"model_type": "mistral_large4_text"},
    ],
)
def test_hook_family(tmp_path, cfg):
    (tmp_path / "config.json").write_text(json.dumps(cfg))
    assert _hook()._detect_native_family(str(tmp_path)) == "mistral_large4"


def test_registry_selects_the_family_for_the_release_config():
    """native_families/mistral_large4.py registers the family for the release's config.json
    (top-level architectures / model_type, and the text decoder's model_type alone); the
    converter is built with the mistral_large4 model provider."""
    from primus.backends.megatron.checkpoint import native_registry

    release = {
        "architectures": ["MistralLarge4ForConditionalGeneration"],
        "model_type": "mistral_large4",
        "dtype": "bfloat16",
        "text_config": {"model_type": "mistral_large4_text"},
    }
    for cfg in (release, {"model_type": "mistral_large4_text"}):
        fam = native_registry.family_for_config(cfg)
        assert fam is not None and fam.name == "mistral_large4"
    assert fam.converter == "primus.backends.megatron.checkpoint.native_loader_mistral_large4_text"
    assert native_registry.model_type_for("mistral_large4") == "mistral_large4"
    assert fam.options == ()  # no native_ckpt_* options: convert() gets only the hook's layout keys
    # the release's dtype is a top-level "dtype" (no torch_dtype): the family's hook reads it
    assert native_registry.dtype_name(fam.hook("dtype_from_config")(release)) == "bf16"
    assert fam.hook("dtype_from_config")({"text_config": {"dtype": "float32"}}) == "float32"


def test_hook_dtype_from_the_release_config(tmp_path):
    (tmp_path / "config.json").write_text(
        json.dumps({"architectures": ["MistralLarge4ForConditionalGeneration"], "dtype": "float32"})
    )
    hook = _hook()
    assert hook._detect_native_family(str(tmp_path)) == "mistral_large4"
    assert hook._native_dtype_from_config(str(tmp_path), None, "mistral_large4") == "fp32"
    assert hook._native_dtype_from_config(str(tmp_path), "bf16", "mistral_large4") == "bf16"


def test_hook_dispatches_to_the_mistral_loader(tmp_path, monkeypatch):
    """native_convert_checkpoint's family branch imports native_loader_mistral_large4_text (not
    the DeepSeek fallback) and calls its convert with the hook's options."""
    import sys
    import types

    from primus.backends.megatron.checkpoint import native_convert_common as common

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral_large4"}))
    hook = _hook()
    calls = []
    fake = types.ModuleType("primus.backends.megatron.checkpoint.native_loader_mistral_large4_text")
    fake.convert = lambda hf, out, **kw: calls.append(("mistral", hf, out, kw))
    other = types.ModuleType("primus.backends.megatron.checkpoint.native_loader_deepseek")
    other.convert = lambda *a, **kw: calls.append(("deepseek",))
    import primus.backends.megatron.checkpoint as ckpt_pkg

    for name, mod in ((fake.__name__, fake), (other.__name__, other)):
        monkeypatch.setitem(sys.modules, name, mod)
        monkeypatch.setattr(ckpt_pkg, name.rsplit(".", 1)[1], mod, raising=False)
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: "")
    monkeypatch.setattr(common, "apply_convert_patches", lambda **kw: 0)
    monkeypatch.setattr(hook, "_native_dtype_from_config", lambda path, override, family=None: "bf16")
    hook.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), {})
    assert calls == [
        (
            "mistral",
            str(tmp_path),
            str(tmp_path / "out"),
            {
                "dtype": "bf16",
                "tensor_parallel_size": 1,
                "pipeline_parallel_size": 1,
                "expert_parallel_size": 1,
                "ckpt_format": "torch_dist",
            },
        )
    ]


def test_model_provider_registration():
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_builders as b,
    )
    from primus.core.utils.import_utils import get_model_provider

    provider = get_model_provider("mistral_large4")
    assert provider.func is b.model_provider and provider.args == (b.mistral_large4_builder,)


@pytest.mark.parametrize(
    "model_type",
    ["mistral_large4_text", "mistral-large4", "mistral_large_4_1t_a52b", "mistral", "gpt", "llama", None],
)
def test_model_type_other_than_mistral_large4_with_its_layout_is_refused(model_type):
    """Any model_type other than mistral_large4 (a typo, the HF name, the repo id, None) with this
    model's per-layer lists is refused at build_args, before the GPT provider would be chosen;
    without the lists (another model) nothing is refused, and mistral_large4 itself passes."""
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("megatron.mistral_large4.model_type_check")
    for k in ("no_rope_layers", "indexer_types", "kv_source_layers"):
        params = SimpleNamespace(model_type=model_type, **{k: "[1, 1, 1, 0]"})
        c = _ctx("build_args", SimpleNamespace(), params)
        assert patch.condition(c)
        with pytest.raises(ValueError, match="model_type 'mistral_large4'"):
            patch.handler(c)
    assert not patch.condition(_ctx("build_args", SimpleNamespace(), SimpleNamespace(model_type=model_type)))
    ok = SimpleNamespace(model_type="mistral_large4", kv_source_layers="[0, 1, 2, 1]")
    assert not patch.condition(_ctx("build_args", SimpleNamespace(), ok))


@pytest.mark.parametrize(
    "flags,want",
    [({}, True), ({"use_distributed_optimizer": True}, False), ({"use_megatron_fsdp": True}, False)],
)
def test_experts_marked_expert_parallel(flags, want):
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_builders as b,
    )

    args = SimpleNamespace(use_distributed_optimizer=False, use_megatron_fsdp=False, use_torch_fsdp2=False)
    for k, v in flags.items():
        setattr(args, k, v)
    assert b._experts_on_expert_groups(args) is want


def test_mark_experts_expert_parallel():
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_builders as b,
    )

    model = torch.nn.Module()
    model.layer = torch.nn.Module()
    model.layer.mlp = torch.nn.Module()
    model.layer.mlp.experts = torch.nn.Linear(2, 2, bias=False)
    model.layer.mlp.shared_experts = torch.nn.Linear(2, 2, bias=False)
    model.layer.mlp.router = torch.nn.Linear(2, 2, bias=False)
    model.layer.mlp.experts.weight.allreduce = True
    model.decoder = model.layer  # parameters are named once
    assert b.mark_experts_expert_parallel(model) == 1
    assert model.layer.mlp.experts.weight.allreduce is False
    assert getattr(model.layer.mlp.shared_experts.weight, "allreduce", True)
    assert getattr(model.layer.mlp.router.weight, "allreduce", True)


def test_freeze_routed_experts():
    """Only the routed experts are frozen; shared expert, router, attention and head stay trainable."""
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_builders as b,
    )

    model = torch.nn.Module()
    model.decoder = torch.nn.Module()
    model.decoder.layers = torch.nn.ModuleList()
    for _ in range(2):
        layer = torch.nn.Module()
        layer.mlp = torch.nn.Module()
        layer.mlp.experts = torch.nn.Module()
        layer.mlp.experts.local_experts = torch.nn.ModuleList([torch.nn.Linear(2, 2, bias=False)] * 1)
        layer.mlp.experts.local_experts.append(torch.nn.Linear(2, 2, bias=False))
        layer.mlp.shared_experts = torch.nn.Linear(2, 2, bias=False)
        layer.mlp.router = torch.nn.Linear(2, 2, bias=False)
        layer.self_attention = torch.nn.Linear(2, 2, bias=False)
        model.decoder.layers.append(layer)
    model.output_layer = torch.nn.Linear(2, 2, bias=False)
    assert b.freeze_routed_experts(model) == 4  # 2 layers x 2 local experts
    for name, p in model.named_parameters():
        assert p.requires_grad is (".mlp.experts." not in name), name
    assert any(p.requires_grad for n, p in model.named_parameters() if "shared_experts" in n)
    assert all(p.requires_grad for n, p in model.named_parameters() if ".router." in n)


@pytest.mark.parametrize(
    "flags,raises",
    [
        ({}, False),
        ({"reset_position_ids": True}, True),
        ({"reset_attention_mask": True}, True),
        ({"reset_position_ids": True, "use_packed_attention": True}, False),
        ({"reset_attention_mask": True, "use_packed_attention": True}, False),
    ],
)
def test_check_position_layout(flags, raises):
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_builders as b,
    )

    args = SimpleNamespace(**flags)
    if raises:
        with pytest.raises(NotImplementedError):
            b.check_position_layout(args)
    else:
        b.check_position_layout(args)


def test_first_iteration_log_reset():
    """is_first_iteration reaches Megatron's training_log unchanged (its timers and log decision
    are Megatron's); only on a first iteration that is a log step are the loss accumulators
    cleared afterwards, as a normal log step clears them."""
    from primus.backends.megatron.patches import (
        mistral_large4_training_log_patches as tl,
    )

    seen = []

    def original(loss_dict, total_loss_dict, learning_rate, iteration, *rest, is_first_iteration=False):
        seen.append(is_first_iteration)
        total_loss_dict["lm loss"] = total_loss_dict.get("lm loss", torch.zeros(1)) + 2.0
        total_loss_dict["advanced iterations"] = total_loss_dict.get("advanced iterations", 0) + 1

    log = tl.make_training_log(original, lambda: 1)
    acc = {}
    log({}, acc, 1e-5, 5, is_first_iteration=True)  # resumed run, first step is a log step
    assert seen == [True]  # flag passed unchanged
    assert float(acc["lm loss"]) == 0.0 and acc["advanced iterations"] == 0  # reset afterwards
    log({}, acc, 1e-5, 6, is_first_iteration=False)
    assert float(acc["lm loss"]) == 2.0  # not touched on later steps (Megatron resets them itself)
    seen.clear()
    log = tl.make_training_log(original, lambda: 10)
    acc = {}
    log({}, acc, 1e-5, 5, is_first_iteration=True)  # not a log step: Megatron's behaviour kept
    assert float(acc["lm loss"]) == 2.0 and acc["advanced iterations"] == 1
    acc = {}
    log({}, acc, 1e-5, iteration=10, is_first_iteration=True)  # by keyword
    assert float(acc["lm loss"]) == 0.0
    assert seen == [True, True]
    # a non-tensor accumulator is zeroed too, as Megatron's own reset zeroes every non-counter key
    acc = {"aux": 3.0, "lm loss": torch.ones(1), "skipped iterations": 2}
    tl.reset_loss_accumulators(acc)
    assert acc["aux"] == 0.0 and float(acc["lm loss"]) == 0.0 and acc["skipped iterations"] == 0
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("megatron.mistral_large4.first_iteration_log")
    assert patch is not None
    # installed before the other training_log wrappers, whose (*args, **kwargs) signatures it
    # could not bind (mlperf_logging priority 15, training_log.unified_patch 50)
    import primus.backends.megatron.patches.mlperf_logging_patches  # noqa: F401
    import primus.backends.megatron.patches.training_log.print_rank_last_patches  # noqa: F401

    others = [
        PatchRegistry.get(pid)
        for pid in ("megatron.training.mlperf_logging", "megatron.training_log.unified_patch")
    ]
    assert all(other is not None and patch.priority < other.priority for other in others)
    for model, want in (("mistral_large4", True), ("deepseek_v3", False)):
        assert patch.condition(_ctx("before_train", params=SimpleNamespace(model_type=model))) is want


def test_first_iteration_log_keeps_time_per_iteration():
    """Megatron's training_log on a first iteration does not reset its interval timer; the wrapper
    clears the iteration counters, so it resets the timer too: the next log line divides the time
    of its own iterations by their count (each iteration takes 1 s here: 1.0 s per iteration)."""
    from primus.backends.megatron.patches import (
        mistral_large4_training_log_patches as tl,
    )

    timer = {"elapsed": 0.0}
    per_iter = []

    def megatron_like(loss_dict, total_loss_dict, learning_rate, iteration, *rest, is_first_iteration=False):
        # a 1 s iteration, then Megatron's log step (log_interval 1)
        timer["elapsed"] += 1.0
        total_loss_dict["advanced iterations"] = total_loss_dict.get("advanced iterations", 0) + 1
        should_reset = not is_first_iteration
        elapsed = timer["elapsed"]
        if should_reset:
            timer["elapsed"] = 0.0
        per_iter.append(elapsed / total_loss_dict["advanced iterations"])
        if should_reset:
            total_loss_dict["advanced iterations"] = 0

    def reset_timer():
        timer["elapsed"] = 0.0

    log = tl.make_training_log(megatron_like, lambda: 1, reset_timer)
    acc = {}
    log({}, acc, 1e-5, 5, is_first_iteration=True)
    log({}, acc, 1e-5, 6, is_first_iteration=False)
    log({}, acc, 1e-5, 7, is_first_iteration=False)
    assert per_iter == [1.0, 1.0, 1.0]
    # without the timer reset (counters cleared, timer not) the second line would be 2.0
    per_iter.clear()
    timer["elapsed"] = 0.0
    log = tl.make_training_log(megatron_like, lambda: 1)
    acc = {}
    log({}, acc, 1e-5, 5, is_first_iteration=True)
    log({}, acc, 1e-5, 6, is_first_iteration=False)
    assert per_iter == [1.0, 2.0]


# ---- PP carrier of the shared latent (split / join of the decoder block) ----------------------
def _stage(layers, pre, post, kv=(0, 1, 2, 1)):
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_block import (
        MistralLarge4TransformerBlock,
    )

    blk = MistralLarge4TransformerBlock.__new__(MistralLarge4TransformerBlock)
    blk.config = SimpleNamespace(kv_lora_rank=4, num_layers=4, kv_source=lambda i: kv[i])
    blk.carrier, blk.kv_source, blk.layer_indices = True, 1, list(layers)
    blk.pre_process, blk.post_process = pre, post
    return blk


@pytest.mark.parametrize("s,b,r,h", [(3, 1, 4, 8), (5, 3, 4, 8), (7, 2, 3, 8), (4, 1, 8, 8)])
def test_pp_carrier_packs_densely_and_passes_gradients(s, b, r, h):
    """Batch > 1, a latent width that does not divide h: each sample's latent round-trips exactly,
    the wire has s + ceil(s * r / h) rows, and gradients reach both the hidden states and the latent."""
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
        kv_carrier_rows,
    )

    first, second = _stage([0, 1], True, False), _stage([2, 3], False, True)
    for blk in (first, second):
        blk.config.kv_lora_rank = r
    hid = torch.randn(s, b, h, requires_grad=True)
    lat = torch.randn(s, b, r, requires_grad=True)
    wire = first._join_output(hid, lat)
    assert wire.shape == (s + kv_carrier_rows(s, r, h), b, h)
    h2, lat2 = second._split_input(wire)
    assert torch.equal(h2, hid) and torch.equal(lat2, lat)
    (h2.sum() + (2 * lat2).sum()).backward()
    assert torch.equal(hid.grad, torch.ones_like(hid)) and torch.equal(lat.grad, 2 * torch.ones_like(lat))


def test_pp_carrier_checks_the_wire_against_seq_length():
    """With the sequence length known, a P2P tensor without the latent's rows is refused instead of
    being split at an s that happens to fit (5 rows = 3 + ceil(3 * 4 / 8) without the check)."""
    h, lat = torch.randn(5, 1, 8), torch.randn(5, 1, 4)
    first, second = _stage([0, 1], True, False), _stage([2, 3], False, True)
    second.config.pp_carrier_seq_length = 5
    wire = first._join_output(h, lat)
    h2, lat2 = second._split_input(wire)  # the sized wire: 5 + 3 rows
    assert torch.equal(h2, h) and torch.equal(lat2, lat)
    with pytest.raises(RuntimeError, match="pp_tensor_shape patch did not size"):
        second._split_input(h)  # hidden states only: no carrier rows
    # sequence parallelism: the stage-local s is seq_length // tp
    second.config.pp_carrier_seq_length, second.config.sequence_parallel = 10, True
    second.pg_collection = SimpleNamespace(
        cp=SimpleNamespace(size=lambda: 1), tp=SimpleNamespace(size=lambda: 2)
    )
    h2, lat2 = second._split_input(wire)
    assert torch.equal(h2, h) and torch.equal(lat2, lat)
    # variable_seq_lengths: no fixed length, the split is derived from the received rows
    second.config.variable_seq_lengths = True
    second.config.pp_carrier_seq_length = 999
    assert torch.equal(second._split_input(wire)[0], h)


def test_pp_carrier_before_source_and_missing_latent():
    h = torch.randn(3, 1, 8)
    before = _stage([0], True, False)  # layer 1 (the source) runs on a later stage
    assert torch.count_nonzero(before._join_output(h, None)[3:]) == 0
    assert _stage([1], False, False)._split_input(torch.zeros(5, 1, 8))[1] is None
    with pytest.raises(RuntimeError):  # the source ran on this stage but no latent reached the carrier
        _stage([0, 1], True, False)._join_output(h, None)


def test_pp_carrier_empty_stage_forwards_the_latent():
    """A stage that builds no layers (e.g. a pipeline_model_parallel_layout entry with none)
    passes the received latent on unchanged instead of sending zeros."""
    h = torch.randn(3, 1, 8)
    lat = torch.randn(3, 1, 4)
    wire = _stage([0, 1], True, False)._join_output(h, lat)  # the source (layer 1) ran first
    empty = _stage([], False, False)
    h2, lat2 = empty._split_input(wire)
    assert torch.equal(h2, h) and torch.equal(lat2, lat)
    out = empty._join_output(h2, lat2)
    assert torch.equal(out, wire)
    h3, lat3 = _stage([2, 3], False, True)._split_input(out)  # the consumer stage still gets it
    assert torch.equal(lat3, lat)


# ---- converter (CPU): fields, single-rank check, fp32-kept tensors -----------------------------
def _release_text_config(**over):
    t = {
        "num_hidden_layers": 4,
        "hidden_size": 64,
        "intermediate_size": 96,
        "num_attention_heads": 4,
        "head_dim": 8,
        "qk_rope_head_dim": 8,
        "qk_nope_head_dim": 8,
        "v_head_dim": 16,
        "kv_lora_rank": 16,
        "rms_norm_eps": 1e-5,
        "vocab_size": 128,
        "rope_parameters": {
            "rope_type": "yarn",
            "factor": 2048,
            "original_max_position_embeddings": 512,
            "rope_theta": 10000.0,
            "attention_factor": 1.0,
        },
        "n_routed_experts": 4,
        "moe_intermediate_size": 24,
        "num_experts_per_tok": 2,
        "n_shared_experts": 1,
        "no_rope_layers": [1, 1, 1, 0],
        "indexer_types": ["full"] * 4,
        "kv_source_layers": [0, 1, 2, 3],
        "index_n_heads": 2,
        "index_head_dim": 16,
        "index_topk": 8,
        "first_k_dense_replace": 1,
    }
    t.update(over)
    return {"text_config": t}


@pytest.mark.parametrize("value,want", [(None, [0, 1, 1, 1]), (0, [1, 1, 1, 1]), (2, [0, 0, 1, 1])])
def test_converter_first_k_dense_replace(value, want):
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    cfg = _release_text_config(first_k_dense_replace=value)
    if value is None:
        del cfg["text_config"]["first_k_dense_replace"]  # missing: the model's default (layer 0 dense)
    f = conv.derive_megatron_fields(cfg)
    assert f["moe_layer_freq"] == want


def test_converter_accepts_sharded_layouts_for_torch_dist():
    """torch_dist is written unsharded and resharded by Megatron at load (AMD-AGI/Primus#1400): a
    TP 4 / EP 2 (or PP 3) fine-tune is converted; only legacy torch needs TP = PP = EP = 1."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    def args(fmt, tp=1, pp=1, ep=1):
        return SimpleNamespace(
            tensor_parallel_size=tp, pipeline_parallel_size=pp, expert_parallel_size=ep, ckpt_format=fmt
        )

    for layout in ({"tp": 4, "ep": 2}, {"tp": 8, "pp": 3, "ep": 4}, {}):
        conv._check_layout(args("torch_dist", **layout))
    conv._check_layout(args("torch"))
    for k in ("tp", "pp", "ep"):
        with pytest.raises(ValueError, match="legacy torch checkpoint loads only at TP = PP = EP = 1"):
            conv._check_layout(args("torch", **{k: 2}))


def test_hook_converts_a_tp4_ep2_experiment(tmp_path, monkeypatch):
    """A TP 4 / EP 2 torch_dist experiment is accepted: the posttrain hook resolves
    torch_dist for it (AMD-AGI/Primus#1400) and the Mistral Large 4 converter's layout check, run
    with the options the hook passes, accepts it."""
    import sys
    import types

    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    hook = _hook()
    experiment = {"tp": 4, "pp": 1, "ep": 2, "ckpt_format": "torch_dist"}
    assert hook._resolve_native_ckpt_format(None, experiment) == "torch_dist"

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral_large4"}))
    seen = []
    fake = types.ModuleType(conv.__name__)
    fake.convert = lambda hf, out, **kw: (conv._check_layout(SimpleNamespace(**kw)), seen.append(kw))
    import primus.backends.megatron.checkpoint as ckpt_pkg

    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    monkeypatch.setattr(ckpt_pkg, fake.__name__.rsplit(".", 1)[1], fake, raising=False)
    monkeypatch.setattr(common, "ensure_megatron_on_path", lambda: "")
    monkeypatch.setattr(common, "apply_convert_patches", lambda **kw: 0)
    monkeypatch.setattr(hook, "_native_dtype_from_config", lambda path, override, family=None: "bf16")
    opts = {"tensor_parallel_size": 4, "expert_parallel_size": 2, "ckpt_format": "torch_dist"}
    hook.native_convert_checkpoint(str(tmp_path), str(tmp_path / "out"), opts)
    assert seen and seen[0]["tensor_parallel_size"] == 4 and seen[0]["expert_parallel_size"] == 2


def test_converter_keeps_fp32_tensors_exact():
    """lerope.alpha, the attention and indexer SSMax and the indexer weights_proj land in the
    model's fp32 tensors with their exact fp32 values; bf16 tensors are rounded once."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    exact = torch.tensor([3.1415927, 1.0001234567, -2.7182817, 0.3333333], dtype=torch.float32)
    W = {
        "tok_embeddings.weight": torch.randn(8, 4),
        "norm.weight": torch.full((4,), 1.0001234567),
        "output.weight": torch.randn(8, 4),
        "lerope.alpha": exact.clone(),
        "layers.0.attention_norm.weight": torch.ones(4),
        "layers.0.ffn_norm.weight": torch.ones(4),
        "layers.0.attention.wq.weight": torch.randn(4, 4),
        "layers.0.attention.wkv_a_with_mqa.weight": torch.randn(4, 4),
        "layers.0.attention.kv_a_norm.weight": torch.ones(4),
        "layers.0.attention.wkv_b.weight": torch.randn(4, 4),
        "layers.0.attention.wo.weight": torch.randn(4, 4),
        "layers.0.attention.ssmax_scaling.s": exact.clone(),
        "layers.0.attention.ssmax_scaling.b": exact.clone(),
        "layers.0.feed_forward.w1.weight": torch.randn(2, 4),
        "layers.0.feed_forward.w3.weight": torch.randn(2, 4),
        "layers.0.feed_forward.w2.weight": torch.randn(4, 2),
    }
    idx = {
        "q_input_norm.weight": torch.ones(4),
        "q_input_norm.bias": torch.zeros(4),
        "wq_b.weight": torch.randn(4, 4),
        "wk.weight": torch.randn(4, 4),
        "k_norm.weight": torch.ones(4),
        "k_norm.bias": torch.zeros(4),
        "weights_proj.weight": exact.view(1, 4).repeat(4, 1).clone(),
        "ssmax_scaling.s": exact.clone(),
        "ssmax_scaling.b": exact.clone(),
    }
    W.update({f"layers.0.attention.indexer.{k}": v for k, v in idx.items()})
    fp32_keys = {
        "decoder.rotary.alpha",
        "decoder.layers.0.self_attention.indexer.weights_proj",
        "decoder.layers.0.self_attention.indexer.ssmax_s",
        "decoder.layers.0.self_attention.indexer.ssmax_b",
        "decoder.layers.0.self_attention.ssmax_s",
        "decoder.layers.0.self_attention.ssmax_b",
    }
    shapes = {
        "embedding.word_embeddings.weight": (8, 4),
        "decoder.final_layernorm.weight": (4,),
        "output_layer.weight": (8, 4),
        "decoder.rotary.alpha": (4,),
        "decoder.layers.0.input_layernorm.weight": (4,),
        "decoder.layers.0.pre_mlp_layernorm.weight": (4,),
        "decoder.layers.0.self_attention.linear_q_proj.weight": (4, 4),
        "decoder.layers.0.self_attention.linear_kv_down_proj.weight": (4, 4),
        "decoder.layers.0.self_attention.kv_layernorm.weight": (4,),
        "decoder.layers.0.self_attention.linear_kv_up_proj.weight": (4, 4),
        "decoder.layers.0.self_attention.linear_proj.weight": (4, 4),
        "decoder.layers.0.self_attention.ssmax_s": (4,),
        "decoder.layers.0.self_attention.ssmax_b": (4,),
        "decoder.layers.0.mlp.linear_fc1.weight": (4, 4),
        "decoder.layers.0.mlp.linear_fc2.weight": (4, 2),
    }
    for mine, theirs in (
        ("q_norm_weight", (4,)),
        ("q_norm_bias", (4,)),
        ("wq_b", (4, 4)),
        ("wk", (4, 4)),
        ("k_norm_weight", (4,)),
        ("k_norm_bias", (4,)),
        ("weights_proj", (4, 4)),
        ("ssmax_s", (4,)),
        ("ssmax_b", (4,)),
    ):
        shapes[f"decoder.layers.0.self_attention.indexer.{mine}"] = theirs
    own = {
        k: torch.zeros(v, dtype=torch.float32 if k in fp32_keys else torch.bfloat16)
        for k, v in shapes.items()
    }

    class _Model:
        config = SimpleNamespace(
            kv_source=lambda i: i, index_source_layer=lambda i: i, uses_rope=lambda i: True
        )

        def state_dict(self):
            return own

    class _Store:
        def get(self, name, dtype):
            assert dtype == torch.float32  # every tensor is read in fp32, never through bf16
            return W[name].to(dtype)

        def all_keys(self):
            return set(W)

    margs = SimpleNamespace(params_dtype=torch.bfloat16, num_layers=1, num_experts=0, moe_layer_freq=[0])
    sd, consumed = conv.build_mcore_state_dict(_Model(), _Store(), margs, {}, lambda *a: None)
    assert consumed == set(W)
    for k in fp32_keys:
        assert sd[k] is own[k] and sd[k].dtype == torch.float32
        assert torch.equal(sd[k].reshape(-1, 4)[0], exact), k
    assert sd["decoder.final_layernorm.weight"].dtype == torch.bfloat16
    assert torch.equal(sd["decoder.final_layernorm.weight"], W["norm.weight"].to(torch.bfloat16))


def test_block_position_and_mask_checks():
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_block import (
        MistralLarge4TransformerBlock as B,
    )

    pos = torch.arange(6)
    B._check_position_ids(None, pos)
    B._check_position_ids(torch.arange(6)[None].expand(2, -1), pos)
    with pytest.raises(NotImplementedError):  # reset_position_ids: positions restart mid-sequence
        B._check_position_ids(torch.tensor([[0, 1, 2, 0, 1, 2]]), pos)
    B._check_sparse_mask(None, 6)
    B._check_sparse_mask(torch.ones(6, 6, dtype=torch.bool).triu(1)[None, None], 6)
    reset = torch.ones(6, 6, dtype=torch.bool).triu(1)
    reset[3:, :3] = True  # reset_attention_mask: a second document
    with pytest.raises(NotImplementedError):
        B._check_sparse_mask(reset[None, None], 6)
    # every sample is checked, not only the first
    causal = torch.ones(6, 6, dtype=torch.bool).triu(1)
    with pytest.raises(NotImplementedError):
        B._check_sparse_mask(torch.stack([causal, reset])[:, None], 6)
    with pytest.raises(NotImplementedError):
        B._check_position_ids(torch.stack([torch.arange(6), torch.tensor([0, 1, 2, 0, 1, 2])]), pos)
    # the O(s) row test agrees with a full [s, s] comparison on near-causal masks
    g = torch.Generator().manual_seed(0)
    for s in (1, 2, 5, 9):
        causal = torch.ones(s, s, dtype=torch.bool).triu(1)
        variants = [causal, ~causal, torch.zeros(s, s, dtype=torch.bool), torch.ones(s, s, dtype=torch.bool)]
        for _ in range(40):
            m = causal.clone()
            flips = torch.randint(0, s * s, (int(torch.randint(1, 3, (1,), generator=g)),), generator=g)
            m.view(-1)[flips] ^= True
            variants.append(m)
        for m in variants:
            want = bool((m == causal).all())
            try:
                B._check_sparse_mask(m[None, None], s)
                got = True
            except NotImplementedError:
                got = False
            assert got == want, (s, m)


def test_check_indexer_source_refuses_random_sparse_selection(tmp_path):
    """seq_length > index_topk selects keys with the frozen indexer: refused without a checkpoint
    (it would stay random), accepted with load / pretrained_checkpoint, for a model whose weights
    are filled after the build, and at seq_length <= index_topk (dense causal)."""
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_builders as b,
    )

    cfg = SimpleNamespace(index_topk=2048)
    b.check_indexer_source(SimpleNamespace(seq_length=2048), cfg)
    with pytest.raises(ValueError, match="frozen indexer"):
        b.check_indexer_source(SimpleNamespace(seq_length=4096, load=None), cfg)
    ckpt = tmp_path / "ckpt"
    ckpt.mkdir()
    # a load / pretrained_checkpoint that holds no checkpoint (no tracker file: the load == save
    # auto-resume pattern on its first run, a stale path) is no loaded indexer
    for key in ("load", "pretrained_checkpoint"):
        with pytest.raises(ValueError, match="frozen indexer"):
            b.check_indexer_source(SimpleNamespace(seq_length=4096, **{key: str(ckpt)}), cfg)
    (ckpt / "latest_checkpointed_iteration.txt").write_text("release")
    b.check_indexer_source(SimpleNamespace(seq_length=4096, load=str(ckpt)), cfg)
    b.check_indexer_source(SimpleNamespace(seq_length=4096, pretrained_checkpoint=str(ckpt)), cfg)
    import sys
    import types

    fake = types.ModuleType("megatron.training")
    logged = []
    fake.print_rank_0 = logged.append
    old = sys.modules.get("megatron.training")
    sys.modules["megatron.training"] = fake
    try:
        # weights filled after the build: accepted with a log line (the forward check follows)
        b.check_indexer_source(SimpleNamespace(seq_length=4096, perform_initialization=False), cfg)
        assert logged and "perform_initialization" in logged[-1]
        b.check_indexer_source(SimpleNamespace(seq_length=4096, allow_untrained_indexer=True), cfg)
    finally:
        if old is None:
            del sys.modules["megatron.training"]
        else:
            sys.modules["megatron.training"] = old


def test_sparse_forward_refuses_an_indexer_at_init():
    """The forward half of the guard: an indexer still at its init values (never loaded: also a
    run resumed from its own from-scratch checkpoint) is refused at the first sparse forward;
    loaded weights, or allow_untrained_indexer, pass. Checked once per layer."""
    from primus.backends.megatron.core.transformer.mistral_large4.attention import (
        MistralLarge4Attention,
        MistralLarge4Indexer,
    )

    # only the tensors at_init reads, at the values __init__ gives them
    idx = MistralLarge4Indexer.__new__(MistralLarge4Indexer)
    torch.nn.Module.__init__(idx)
    for name, shape, value in (
        ("q_norm_weight", (8,), 1.0),
        ("k_norm_weight", (4,), 1.0),
        ("ssmax_s", (2,), 0.0),
    ):
        idx.register_buffer(name, torch.full(shape, value))
    assert idx.at_init()
    att = SimpleNamespace(indexer=idx, layer_idx=3, config=SimpleNamespace(allow_untrained_indexer=False))
    with pytest.raises(RuntimeError, match="initial weights"):
        MistralLarge4Attention._check_indexer_loaded(att)
    att.config.allow_untrained_indexer = True
    MistralLarge4Attention._check_indexer_loaded(att)
    idx.ssmax_s.fill_(0.25)  # loaded weights
    assert not idx.at_init()
    att.config.allow_untrained_indexer = False
    MistralLarge4Attention._check_indexer_loaded(att)
    assert att._indexer_checked


def test_check_stage_starts_at_build():
    """A stage starting on a 'shared' indexer layer is refused at build when seq_length > index_topk."""
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_builders as b,
    )

    blk = torch.nn.Module()
    blk._stage_starts_on_shared = True
    blk.layer_indices = [17, 18]
    blk.config = SimpleNamespace(index_topk=2048)
    model = torch.nn.Sequential(blk)
    b.check_stage_starts(model, SimpleNamespace(seq_length=2048))  # dense path only: allowed
    with pytest.raises(ValueError, match="starts on layer 17"):
        b.check_stage_starts(model, SimpleNamespace(seq_length=4096))
    blk._stage_starts_on_shared = False
    b.check_stage_starts(model, SimpleNamespace(seq_length=4096))


def test_layer_lists_from_checkpoint_refuses_user_overrides(tmp_path, monkeypatch):
    """A user value (not the preset's) that disagrees with the checkpoint's config.json is refused;
    the preset's own value, or one that agrees, is replaced / kept."""
    cut = {
        "text_config": {
            "num_hidden_layers": 4,
            "no_rope_layers": [1, 1, 1, 0],
            "indexer_types": ["full", "full", "full", "full"],
            "kv_source_layers": [0, 1, 2, 3],
            "first_k_dense_replace": 1,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(cut))
    preset = {
        "no_rope_layers": "[1, 0, 1, 0, 1, 0]",
        "indexer_types": None,
        "kv_source_layers": None,
        "first_k_dense_replace": 2,
    }
    monkeypatch.setattr(mistral_large4_args_patches, "_preset_values", lambda params: dict(preset))

    def run(**given):
        args = SimpleNamespace(num_layers=4, **given)
        params = SimpleNamespace(model_type="mistral_large4", hf_path=str(tmp_path), model="x.yaml")
        mistral_large4_args_patches.patch_mistral_large4_layer_lists_from_checkpoint(
            _ctx("build_args", args, params)
        )
        return args

    # the preset's values: replaced by the checkpoint's
    a = run(no_rope_layers=preset["no_rope_layers"], first_k_dense_replace=2)
    assert a.no_rope_layers == [1, 1, 1, 0] and a.first_k_dense_replace == 1
    # a user value that agrees with the checkpoint: kept
    assert run(no_rope_layers=[1, 1, 1, 0, 1]).no_rope_layers == [1, 1, 1, 0]
    # user values that disagree: refused
    with pytest.raises(ValueError, match="no_rope_layers"):
        run(no_rope_layers=[0, 1, 1, 1])
    with pytest.raises(ValueError, match="first_k_dense_replace"):
        run(first_k_dense_replace=3)
    with pytest.raises(ValueError, match="kv_source_layers"):
        run(kv_source_layers="[0, 1, 2, 1]")


def test_layer_lists_from_checkpoint(tmp_path, monkeypatch):
    cut = {
        "text_config": {
            "num_hidden_layers": 4,
            "no_rope_layers": [1, 1, 0, 0],
            "indexer_types": ["full", "full", "shared", "full"],
            "kv_source_layers": [0, 1, 2, 2],
            "first_k_dense_replace": 1,
        }
    }
    (tmp_path / "config.json").write_text(json.dumps(cut))
    # the preset's own value (the release's first entries): replaced by the checkpoint's
    preset = "[0, 1, 2, 3, 4, 5]"
    monkeypatch.setattr(
        mistral_large4_args_patches,
        "_preset_values",
        lambda params: {
            "kv_source_layers": preset,
            "no_rope_layers": None,
            "indexer_types": None,
            "first_k_dense_replace": 1,
        },
    )
    args = SimpleNamespace(num_layers=4, kv_source_layers=preset)
    params = SimpleNamespace(
        model_type="mistral_large4", stage="sft", hf_path=str(tmp_path), kv_source_layers=preset
    )
    ctx = _ctx("build_args", args, params)
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("megatron.mistral_large4.layer_lists_from_checkpoint")
    assert patch.condition(ctx)
    mistral_large4_args_patches.patch_mistral_large4_layer_lists_from_checkpoint(ctx)
    assert args.kv_source_layers == [0, 1, 2, 2] and params.kv_source_layers == [0, 1, 2, 2]
    assert args.indexer_types == ["full", "full", "shared", "full"] and args.first_k_dense_replace == 1
    args.num_layers = 2  # the checkpoint's first 2 layers: lists taken, cut to 2 by the config
    args.kv_source_layers = args.indexer_types = args.no_rope_layers = None
    params.first_k_dense_replace = args.first_k_dense_replace = None
    mistral_large4_args_patches.patch_mistral_large4_layer_lists_from_checkpoint(ctx)
    assert args.kv_source_layers == [0, 1, 2, 2] and args.no_rope_layers == [1, 1, 0, 0]
    assert args.indexer_types == ["full", "full", "shared", "full"] and args.first_k_dense_replace == 1
    assert params.first_k_dense_replace == 1
    args.num_layers = 6
    with pytest.raises(ValueError):  # the checkpoint has 4 layers
        mistral_large4_args_patches.patch_mistral_large4_layer_lists_from_checkpoint(ctx)
    # no local config.json (a Hub id, or no hf_path): the preset's lists stay
    assert not patch.condition(
        _ctx("build_args", args, SimpleNamespace(model_type="mistral_large4", hf_path="org/x"))
    )
    assert not patch.condition(
        _ctx(
            "build_args", args, SimpleNamespace(model_type="deepseek_v3", stage="sft", hf_path=str(tmp_path))
        )
    )
    # pretraining (no stage) with a local hf_path (e.g. for the tokenizer) keeps the preset's layout
    assert not patch.condition(
        _ctx("build_args", args, SimpleNamespace(model_type="mistral_large4", hf_path=str(tmp_path)))
    )
    # a user value that disagrees with the checkpoint is refused, also without a readable preset
    monkeypatch.setattr(mistral_large4_args_patches, "_preset_values", lambda params: None)
    args.num_layers, args.kv_source_layers = 4, "[0, 1, 2, 1]"
    with pytest.raises(ValueError, match="disagrees with the checkpoint"):
        mistral_large4_args_patches.patch_mistral_large4_layer_lists_from_checkpoint(ctx)
    # a checkpoint whose KV sharing breaks the rule is refused before anything is set
    args.kv_source_layers = None
    for bad, match in (
        ([0, 1, 5, 1], "breaks the rule"),  # out of range / forward
        ([0, 1, 2, 1], "rope/NoPE kinds differ"),  # NoPE layer 3 would reuse rope layer 1's KV
    ):
        cut["text_config"]["kv_source_layers"] = bad
        cut["text_config"]["no_rope_layers"] = [1, 1, 1, 0]
        (tmp_path / "config.json").write_text(json.dumps(cut))
        with pytest.raises(ValueError, match=match):
            mistral_large4_args_patches.patch_mistral_large4_layer_lists_from_checkpoint(ctx)


def test_layer_lists_order_and_hf_defaults(tmp_path, monkeypatch):
    """build_args order: the checkpoint's lists (and first_k_dense_replace) before moe_layer_freq,
    by explicit priorities; a checkpoint without a list gets the HF default (no NoPE, no sharing),
    not the preset's release list."""
    from primus.core.patches.patch_registry import PatchRegistry

    P = mistral_large4_args_patches
    lists = PatchRegistry.get("megatron.mistral_large4.layer_lists_from_checkpoint")
    freq = PatchRegistry.get("megatron.mistral_large4.moe_layer_freq")
    assert lists.priority == P.PRIORITY_LAYER_LISTS < freq.priority == P.PRIORITY_MOE_LAYER_FREQ
    refuse = PatchRegistry.get("megatron.mistral_large4.posttrain_needs_checkpoint")
    assert refuse.priority == P.PRIORITY_NEEDS_CHECKPOINT < lists.priority  # refused before any read
    assert freq.priority < 50  # explicit: not the registry default

    (tmp_path / "config.json").write_text(
        json.dumps({"text_config": {"num_hidden_layers": 4, "first_k_dense_replace": 2}})
    )
    release = "[0, 1, 2, 1]"
    args = SimpleNamespace(num_layers=4, kv_source_layers=release, num_experts=4, moe_layer_freq=1)
    params = SimpleNamespace(
        model_type="mistral_large4", stage="sft", hf_path=str(tmp_path), kv_source_layers=release
    )
    # the preset's own value (not a user override): replaced by the checkpoint's
    monkeypatch.setattr(
        P,
        "_preset_values",
        lambda params: {
            "kv_source_layers": release,
            "no_rope_layers": None,
            "indexer_types": None,
            "first_k_dense_replace": None,
        },
    )
    ctx = _ctx("build_args", args, params)
    for patch in sorted((lists, freq), key=lambda p: p.priority):
        patch.handler(ctx)
    assert args.kv_source_layers == [0, 1, 2, 3]
    assert args.no_rope_layers == [1, 1, 1, 1] and args.indexer_types == ["full"] * 4
    assert args.moe_layer_freq == [0, 0, 1, 1]  # the checkpoint's first_k_dense_replace

    # first_k_dense_replace missing in the checkpoint: the converter's default (1), with a warning
    (tmp_path / "config.json").write_text(json.dumps({"text_config": {"num_hidden_layers": 4}}))
    args.moe_layer_freq, args.first_k_dense_replace = 1, None
    params.first_k_dense_replace = None
    args.kv_source_layers = release
    for patch in sorted((lists, freq), key=lambda p: p.priority):
        patch.handler(ctx)
    assert args.first_k_dense_replace == 1 and args.moe_layer_freq == [0, 1, 1, 1]


def test_posttrain_without_checkpoint_is_refused(tmp_path):
    """SFT (stage set) with no pretrained_checkpoint / load: refused at argument time with what to
    set (the hook would take the Bridge path or train from random weights). With the converted
    checkpoint the hook passes, or for pretraining (no stage), nothing is refused."""
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("megatron.mistral_large4.posttrain_needs_checkpoint")

    def ctx(**kw):
        params = SimpleNamespace(model_type="mistral_large4", **kw)
        return _ctx("build_args", SimpleNamespace(), params)

    c = ctx(stage="sft", hf_path=None, pretrained_checkpoint=None, load=None)
    assert patch.condition(c)
    with pytest.raises(ValueError, match="no hf_path is set. Pass --hf_path <local snapshot"):
        patch.handler(c)
    # a Hub id: the hook cannot detect the family there; the message names it and does not ask
    # for --hf_path again
    c = ctx(stage="sft", hf_path="mistralai/Mistral-Large-4-1T-A52B", pretrained_checkpoint=None, load=None)
    assert patch.condition(c)
    with pytest.raises(ValueError, match="'mistralai/Mistral-Large-4-1T-A52B' is not a local directory") as e:
        patch.handler(c)
    assert "Pass --hf_path" not in str(e.value)
    # a local snapshot that no conversion reached (trainer run without the posttrain hook)
    snap = tmp_path / "snap"
    snap.mkdir()
    (snap / "config.json").write_text("{}")
    c = ctx(stage="sft", hf_path=str(snap), pretrained_checkpoint=None, load=None)
    assert patch.condition(c)
    with pytest.raises(ValueError, match="is a local checkpoint, but no conversion of it reached this run"):
        patch.handler(c)
    # what the hook passes after converting hf_path (extra.pretrained_checkpoint): not refused,
    # with a local or a Hub hf_path alike
    conv = tmp_path / "conv"
    conv.mkdir()
    (conv / "latest_checkpointed_iteration.txt").write_text("release")
    for hf in (str(snap), "mistralai/Mistral-Large-4-1T-A52B"):
        assert not patch.condition(ctx(stage="sft", hf_path=hf, pretrained_checkpoint=str(conv)))
    assert not patch.condition(ctx(stage="sft", pretrained_checkpoint=str(conv)))
    # pretrained_checkpoint is held to the same rule as load: a mistyped or stale path, or the
    # empty output directory of a failed conversion, is refused (Megatron would only warn and
    # start from random weights)
    for bad in ("/nonexistent/ckpt", str(tmp_path / "empty")):
        c = ctx(stage="sft", hf_path=None, pretrained_checkpoint=bad, load=None)
        assert patch.condition(c)
        with pytest.raises(ValueError, match="no pretrained_checkpoint or load directory"):
            patch.handler(c)
    (tmp_path / "empty").mkdir()
    assert patch.condition(ctx(stage="sft", pretrained_checkpoint=str(tmp_path / "empty"), load=None))
    # load: a checkpoint only when it holds Megatron's tracker file (the load == save auto-resume
    # pattern names an empty or missing directory on the first run)
    resume = tmp_path / "resume"
    assert patch.condition(ctx(stage="sft", load=str(resume)))
    resume.mkdir()
    assert patch.condition(ctx(stage="sft", load=str(resume)))
    (resume / "latest_checkpointed_iteration.txt").write_text("release")
    assert not patch.condition(ctx(stage="sft", load=str(resume)))
    assert not patch.condition(ctx(finetune=True))  # pretraining: pre_trainer.yaml defaults finetune true
    # an explicit pretrain stage is pretraining, not post-training
    for stage in ("pretrain", "Pretrain", ""):
        assert not patch.condition(ctx(stage=stage, load=None))
    assert not patch.condition(
        _ctx("build_args", SimpleNamespace(), SimpleNamespace(model_type="deepseek_v3", stage="sft"))
    )


def test_preset_values_errors():
    """A model preset that does not exist reads as None (the caller then refuses a disagreeing
    value); other errors raise."""
    P = mistral_large4_args_patches
    assert P._preset_values(SimpleNamespace(model="no_such_model_preset.yaml")) is None
    assert P._preset_values(SimpleNamespace(model=None)) is None
    with pytest.raises(ValueError, match="path traversal"):
        P._preset_values(SimpleNamespace(model="../../../../etc/passwd"))


def test_block_context_shares_keep_with_moved_layer_copies():
    """A full-indexer layer on another device writes its keep mask through _local_ctx's copy;
    a following shared layer (also on a moved device) reads it from the same dict."""
    from primus.backends.megatron.core.transformer.mistral_large4.attention import (
        MistralLarge4Attention as A,
    )

    ctx = {
        "positions": torch.arange(4),
        "freqs": torch.zeros(4, 1, 1, 2),
        "topk": {},
        "keep": {},
        "sparse": True,
    }
    local = A._local_ctx(ctx, torch.device("meta"))  # a different device: a copy of the dict
    assert local is not ctx and local["positions"].device.type == "meta"
    local["keep"][0] = "mask of layer 0"
    local["topk"][0] = "topk of layer 0"
    shared = A._local_ctx(ctx, torch.device("meta"))
    assert shared["keep"][0] == "mask of layer 0" and shared["topk"][0] == "topk of layer 0"


def test_block_context_creates_keep(monkeypatch):
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_block import (
        MistralLarge4TransformerBlock,
    )

    blk = MistralLarge4TransformerBlock.__new__(MistralLarge4TransformerBlock)
    torch.nn.Module.__init__(blk)
    one = SimpleNamespace(size=lambda: 1)
    blk.config = SimpleNamespace(sequence_parallel=False, index_topk=8)
    blk.pg_collection = SimpleNamespace(tp=one, cp=one)
    blk.rotary = SimpleNamespace(dim=0)
    blk._stage_starts_on_shared = False
    ctx = blk._context(torch.zeros(4, 1, 2), None, None, None)
    assert ctx["keep"] == {} and ctx["topk"] == {} and ctx["sparse"] is False
    # packed THD: the longest document decides dense vs sparse, not the pack length
    packed = SimpleNamespace(
        qkv_format="thd", max_seqlen_q=4, cu_seqlens_q=torch.tensor([0, 4, 8, 12]), cu_seqlens_q_padded=None
    )
    ctx = blk._context(torch.zeros(12, 1, 2), packed, None, None)
    assert ctx["sparse"] is False
    # rope / SSMax positions restart per document
    assert ctx["positions"].tolist() == [0, 1, 2, 3] * 3
    assert torch.allclose(ctx["log_pos"], (torch.tensor([0, 1, 2, 3] * 3).float() + 1).log())
    # uneven documents
    packed.cu_seqlens_q, packed.max_seqlen_q = torch.tensor([0, 3, 4, 12]), 8
    ctx = blk._context(torch.zeros(12, 1, 2), packed, None, None)
    assert ctx["positions"].tolist() == [0, 1, 2, 0, 0, 1, 2, 3, 4, 5, 6, 7]
    # padded documents: cu_seqlens_q_padded (the token layout) wins over cu_seqlens_q
    packed.cu_seqlens_q = torch.tensor([0, 3, 7, 10])
    packed.cu_seqlens_q_padded = torch.tensor([0, 4, 8, 12])
    packed.max_seqlen_q = 4
    ctx = blk._context(torch.zeros(12, 1, 2), packed, None, None)
    assert ctx["positions"].tolist() == [0, 1, 2, 3] * 3


def test_consolidated_store_refuses_other_layouts(tmp_path):
    """Without the consolidated index (the default name, or the one config.json's
    transformers_weights names) the store refuses the directory with a clear error: an HF-layout
    copy has none of the Mistral-native names the converter maps."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    (tmp_path / "config.json").write_text(json.dumps({"model_type": "mistral_large4"}))
    torch.save({"model.norm.weight": torch.ones(4)}, tmp_path / "pytorch_model.bin")
    with pytest.raises(FileNotFoundError, match="consolidated.safetensors.index.json .*the default"):
        conv.ConsolidatedStore(str(tmp_path))
    (tmp_path / "config.json").write_text(
        json.dumps({"model_type": "mistral_large4", "transformers_weights": "other.index.json"})
    )
    with pytest.raises(FileNotFoundError, match="other.index.json .*transformers_weights"):
        conv.ConsolidatedStore(str(tmp_path))


def test_layer_lists_from_config_are_validated():
    """config.json lists go through the model config's parsing and checks (one policy): a string is
    a list literal (not split into characters), a list shorter than num_hidden_layers and a value
    outside the allowed set are refused; the model config refuses no_rope_layers other than 0 / 1."""
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
        hf_layer_lists,
    )

    lists, defaulted = hf_layer_lists({"num_hidden_layers": 4, "no_rope_layers": "[1, 0, 1, 0]"})
    assert lists["no_rope_layers"] == [1, 0, 1, 0] and "no_rope_layers" not in defaulted
    for bad, match in (
        ({"no_rope_layers": [1, 0]}, "fewer than num_layers"),
        ({"indexer_types": "full"}, "not a list literal"),
        ({"indexer_types": ["full", "shard", "full", "full"]}, r"indexer_types\[1\] = 'shard'"),
        ({"no_rope_layers": [1, 2, 1, 0]}, r"no_rope_layers\[1\] = 2"),
        ({"kv_source_layers": [0, 1, "2", 3]}, "not all int"),
    ):
        with pytest.raises(ValueError, match=match):
            hf_layer_lists({"num_hidden_layers": 4, **bad})
    with pytest.raises(ValueError, match=r"no_rope_layers\[0\] = -1"):
        _cfg(no_rope_layers=[-1, 1, 1, 0])


def test_converter_layer_lists_default_like_training():
    """A config.json without the per-layer lists converts with the HF defaults (all rope, own KV,
    own indexer): the same policy as the training-side patch (one shared function)."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
        hf_layer_lists,
    )

    cfg = _release_text_config()
    for k in ("no_rope_layers", "indexer_types", "kv_source_layers"):
        del cfg["text_config"][k]
    f = conv.derive_megatron_fields(cfg)
    assert f["no_rope_layers"] == [1, 1, 1, 1]
    assert f["kv_source_layers"] == [0, 1, 2, 3] and f["indexer_types"] == ["full"] * 4
    lists, defaulted = hf_layer_lists(cfg["text_config"])
    assert {k: f[k] for k in lists} == lists and len(defaulted) == 3


def test_block_checks_each_run_once_when_they_apply():
    """A dense first micro-batch does not consume the sparse-mask check; a later sparse one runs it.
    Each check runs once per block (no host sync on every micro-batch)."""
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_block import (
        MistralLarge4TransformerBlock,
    )

    blk = MistralLarge4TransformerBlock.__new__(MistralLarge4TransformerBlock)
    torch.nn.Module.__init__(blk)
    one = SimpleNamespace(size=lambda: 1)
    blk.config = SimpleNamespace(sequence_parallel=False, index_topk=4)
    blk.pg_collection = SimpleNamespace(tp=one, cp=one)
    blk.rotary = SimpleNamespace(dim=0)
    blk._stage_starts_on_shared = False
    blk._context(torch.zeros(3, 1, 2), None, None, None)  # dense (3 <= 4), no position ids
    reset = torch.ones(6, 6, dtype=torch.bool).triu(1)
    reset[3:, :3] = True
    with pytest.raises(NotImplementedError):  # sparse (6 > 4) with a non-causal mask: still checked
        blk._context(torch.zeros(6, 1, 2), None, None, reset[None, None])
    with pytest.raises(NotImplementedError):  # position ids still checked on their first appearance
        blk._context(torch.zeros(3, 1, 2), None, torch.tensor([[0, 1, 0]]), None)
    # once a check passed, later micro-batches are not re-checked (no per-micro-batch host sync):
    # the guarantee is check_position_layout's build-time refusal (test_check_position_layout)
    blk2 = MistralLarge4TransformerBlock.__new__(MistralLarge4TransformerBlock)
    torch.nn.Module.__init__(blk2)
    blk2.config, blk2.pg_collection, blk2.rotary = blk.config, blk.pg_collection, blk.rotary
    blk2._stage_starts_on_shared = False
    causal = torch.ones(6, 6, dtype=torch.bool).triu(1)
    blk2._context(torch.zeros(6, 1, 2), None, torch.arange(6)[None], causal[None, None])
    synced = []
    real = MistralLarge4TransformerBlock._assert
    MistralLarge4TransformerBlock._assert = staticmethod(lambda ok, msg: synced.append(msg))
    try:
        blk2._context(torch.zeros(6, 1, 2), None, torch.arange(6)[None], causal[None, None])
    finally:
        MistralLarge4TransformerBlock._assert = real
    assert synced == []


def test_converter_save_retries_only_write_failures(tmp_path, monkeypatch):
    """A started write that fails (CheckpointException or Megatron's async 'Worker failure'
    RuntimeError) is retried after removing only the directory it wrote (release/ for torch_dist); a mapping error (nothing written)
    is not; an earlier conversion is never overwritten and no write is attempted then."""
    import shutil

    from torch.distributed.checkpoint.api import CheckpointException

    from primus.backends.megatron.checkpoint import native_convert_common as common
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    keep = tmp_path / "other.txt"
    keep.write_text("user data")
    iter_dir = tmp_path / "release"  # torch_dist is saved straight into release/
    margs = SimpleNamespace(iteration=1, ckpt_format="torch_dist")
    save = lambda: conv._save_with_retry(None, {}, set(), None, margs, str(tmp_path), lambda *a: None)

    for err in (
        CheckpointException("write", {0: (RuntimeError("unexpected pos"), None)}),
        RuntimeError("Worker failure: unexpected pos"),
    ):
        calls = []

        def flaky(*a, _err=err, **kw):
            calls.append(1)
            assert not iter_dir.exists()  # the failed attempt's directory was removed
            iter_dir.mkdir()
            if len(calls) < 2:
                (iter_dir / "partial").write_text("x")
                raise _err

        monkeypatch.setattr(common, "load_validate_save", flaky)
        save()
        assert len(calls) == 2 and keep.read_text() == "user data"
        shutil.rmtree(iter_dir)

    calls = []

    def always_fails(*a, **kw):
        calls.append(1)
        iter_dir.mkdir(exist_ok=True)
        raise RuntimeError("Worker failure")

    monkeypatch.setattr(common, "load_validate_save", always_fails)
    with pytest.raises(RuntimeError, match="Worker failure"):
        save()
    assert len(calls) == conv.SAVE_ATTEMPTS and not iter_dir.exists()  # no partial dir left

    calls = []

    def bad_mapping(*a, **kw):
        calls.append(1)
        raise RuntimeError("native conversion mapping incomplete/incorrect")

    monkeypatch.setattr(common, "load_validate_save", bad_mapping)
    with pytest.raises(RuntimeError, match="mapping"):
        save()
    assert len(calls) == 1  # a mapping error is not retried

    # a started write that fails deterministically (full disk, validation) is not retried
    for err in (
        OSError(28, "No space left on device"),
        RuntimeError("Worker failure: No space left on device"),
        ValueError("validation failed"),
    ):
        calls = []

        def started_then_fails(*a, _err=err, **kw):
            calls.append(1)
            iter_dir.mkdir(exist_ok=True)
            raise _err

        monkeypatch.setattr(common, "load_validate_save", started_then_fails)
        with pytest.raises(type(err)):
            save()
        assert len(calls) == 1 and not iter_dir.exists()

    # an interrupted release/ (no tracker file) is set aside before the first attempt (kept as
    # release.incomplete-*): a failing write is then retried and its partial release/ removed
    iter_dir.mkdir()
    (iter_dir / "old").write_text("x")
    calls = []
    monkeypatch.setattr(common, "load_validate_save", always_fails)
    with pytest.raises(RuntimeError, match="Worker failure"):
        save()
    aside = [p for p in tmp_path.iterdir() if p.name.startswith("release.incomplete-")]
    assert len(calls) == conv.SAVE_ATTEMPTS and not iter_dir.exists()
    assert len(aside) == 1 and (aside[0] / "old").read_text() == "x"
    shutil.rmtree(aside[0])

    # same leftover, the first write fails once and the retry succeeds
    iter_dir.mkdir()
    (iter_dir / "old").write_text("x")
    calls = []

    def fails_once(*a, **kw):
        calls.append(1)
        assert not iter_dir.exists()
        iter_dir.mkdir()
        if len(calls) < 2:
            raise RuntimeError("Worker failure unexpected pos")

    monkeypatch.setattr(common, "load_validate_save", fails_once)
    save()
    assert len(calls) == 2
    for p in tmp_path.iterdir():
        if p.name.startswith("release"):
            shutil.rmtree(p)

    # an earlier complete conversion (tracker file) is refused before any write
    calls = []
    monkeypatch.setattr(common, "load_validate_save", lambda *a, **kw: calls.append(1))
    for earlier in ("release", "1"):
        (tmp_path / common.TRACKER_FILENAME).write_text(earlier)
        with pytest.raises(RuntimeError, match="refusing to overwrite"):
            save()
        assert calls == []
    (tmp_path / common.TRACKER_FILENAME).unlink()

    # torch format writes iter_<n>: an existing one is refused
    margs.ckpt_format = "torch"
    (tmp_path / "iter_0000001").mkdir()
    with pytest.raises(RuntimeError, match="refusing to overwrite"):
        save()
    assert calls == [] and (tmp_path / "iter_0000001").is_dir()


def test_full_recompute_backward_order_restores_shared_topk(monkeypatch):
    """Full recompute (block method): chunks are recomputed last-first in backward. A shared
    layer whose source full layer sits in an earlier chunk must still find that layer's top-k
    after a later full layer replaced ctx's single entry."""
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_block as blkmod,
    )
    from primus.backends.megatron.core.transformer.mistral_large4.attention import (
        MistralLarge4Attention as A,
    )

    s = 6
    topks = {i: torch.randint(0, s, (1, s, 2)) for i in (0, 2)}

    def attention(idx, src):
        att = A.__new__(A)
        torch.nn.Module.__init__(att)
        att.config = SimpleNamespace(sequence_parallel=False)
        att._indexer_checked = True  # a stub indexer (the loaded-weights check has its own test)
        att.layer_idx, att.index_source_idx = idx, src
        att.consumes_kv = att.stores_kv = False
        att.kv_source_idx = idx
        att.indexer = (lambda *a, _i=idx: topks[_i]) if src == idx else None
        return att

    atts = [attention(0, 0), attention(1, 0), attention(2, 2)]  # full, shared, full
    layers = [SimpleNamespace(self_attention=t) for t in atts]
    seen = []

    blk = blkmod.MistralLarge4TransformerBlock.__new__(blkmod.MistralLarge4TransformerBlock)
    torch.nn.Module.__init__(blk)
    blk.layers = layers
    blk.pre_process = True
    blk.final_layernorm = None
    blk.config = SimpleNamespace(
        sequence_parallel=False,
        recompute_granularity="full",
        recompute_method="block",
        recompute_num_layers=3,
        distribute_saved_activations=False,
    )
    blk.training = True
    ctx = {
        "positions": torch.arange(s),
        "freqs": None,
        "topk": {},
        "keep": {},
        "causal_keep": torch.ones(s, s, dtype=torch.bool).tril_(),
    }
    monkeypatch.setattr(blk, "_split_input", lambda h: (h, None))
    monkeypatch.setattr(blk, "_join_output", lambda h, lat: h)
    monkeypatch.setattr(blk, "_context", lambda *a: ctx)

    def run_layer(layer, h, latent, c, *a):
        keep = layer.self_attention._keep(h, c)
        seen.append((layer.self_attention.layer_idx, keep.clone()))
        return h, None

    monkeypatch.setattr(blk, "_run_layer", run_layer)
    runs = []

    def checkpoint(fn, distribute, *args):  # record the chunk to replay it in backward order
        runs.append((fn, args))
        return fn(*args)

    monkeypatch.setattr(blkmod.tensor_parallel, "checkpoint", checkpoint)
    blk.forward(torch.zeros(s, 1, 2), None)
    forward = dict(seen)
    assert torch.equal(forward[1], forward[0])  # the shared layer used layer 0's mask
    seen.clear()
    for fn, args in reversed(runs):  # backward: last chunk first
        fn(*args)
    recomputed = dict(seen)
    for i in range(3):
        assert torch.equal(recomputed[i], forward[i]), i


def test_converter_checks_vocab_against_tokenizer():
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    conv.check_vocab(204800, {"vocab_size": 204800}, "hf")
    with pytest.raises(ValueError, match="fewer rows"):
        conv.check_vocab(131072, {"vocab_size": 204800}, "hf")


def test_consolidated_store_dequantises_fp8_blocks(tmp_path):
    """The release layout: consolidated.safetensors.index.json (named by config.json's
    transformers_weights), FP8 weights with a 128x128 qscale_weight; read_weight dequantises and
    marks both names consumed; an unquantised tensor is read as stored."""
    from safetensors.torch import save_file

    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    w = (torch.randn(256, 384) * 4).to(torch.float8_e4m3fn)
    scale = torch.tensor([[2.0**-3, 2.0**-2, 2.0**-1], [1.0, 2.0, 4.0]], dtype=torch.float32)
    norm = torch.randn(384, dtype=torch.bfloat16)
    save_file(
        {"layers.0.attention.wq.weight": w, "layers.0.attention.wq.qscale_weight": scale},
        str(tmp_path / "consolidated-00001-of-00002.safetensors"),
    )
    save_file({"norm.weight": norm}, str(tmp_path / "consolidated-00002-of-00002.safetensors"))
    index = {
        "weight_map": {
            "layers.0.attention.wq.weight": "consolidated-00001-of-00002.safetensors",
            "layers.0.attention.wq.qscale_weight": "consolidated-00001-of-00002.safetensors",
            "norm.weight": "consolidated-00002-of-00002.safetensors",
        }
    }
    (tmp_path / "consolidated.safetensors.index.json").write_text(json.dumps(index))
    (tmp_path / "config.json").write_text(
        json.dumps(
            {"model_type": "mistral_large4", "transformers_weights": "consolidated.safetensors.index.json"}
        )
    )
    store = conv.ConsolidatedStore(str(tmp_path))
    assert store.all_keys() == set(index["weight_map"])
    consumed = set()
    got = conv.read_weight(store, "layers.0.attention.wq.weight", torch.float32, consumed)
    want = w.float() * scale.repeat_interleave(128, 0).repeat_interleave(128, 1)
    assert torch.equal(got, want)
    assert consumed == {"layers.0.attention.wq.weight", "layers.0.attention.wq.qscale_weight"}
    assert torch.equal(conv.read_weight(store, "norm.weight", torch.bfloat16, consumed), norm)
    assert consumed == set(index["weight_map"])

    # an FP8 tensor whose scale is missing (renamed key, missing shard) is refused, not cast unscaled
    save_file({"layers.0.attention.wo.weight": w}, str(tmp_path / "consolidated-00003.safetensors"))
    index["weight_map"]["layers.0.attention.wo.weight"] = "consolidated-00003.safetensors"
    (tmp_path / "consolidated.safetensors.index.json").write_text(json.dumps(index))
    store = conv.ConsolidatedStore(str(tmp_path))
    with pytest.raises(RuntimeError, match="stored in FP8 but has no block scale"):
        conv.read_weight(store, "layers.0.attention.wo.weight", torch.float32, set())
    # the same through the plain HF-layout store (build_mcore_state_dict reads every store this way)
    plain = tmp_path / "plain"
    plain.mkdir()
    save_file({"w.weight": w}, str(plain / "model.safetensors"))
    with pytest.raises(RuntimeError, match="stored in FP8"):
        conv.read_weight(conv.common.SafetensorsStore(str(plain)), "w.weight", torch.float32, set())


@pytest.mark.parametrize("case", ["shape", "missing_key"])
def test_state_dict_mapping_errors_fail_fast(case):
    """A mapped tensor whose shape differs from the model's, or a key the model lacks, is raised by
    the first assignment (layer 0), before any later layer is read."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    own = {"embedding.word_embeddings.weight": torch.zeros(8, 4)}
    reads = []

    class _Own(dict):  # the embedding row count is read first; a missing key is then refused
        def __contains__(self, k):
            return case == "shape" and dict.__contains__(self, k)

    class _Model:
        config = SimpleNamespace()

        def state_dict(self):
            return _Own(own)

    class _Store:
        def get(self, name, dtype):
            reads.append(name)
            return torch.zeros(8, 3 if case == "shape" else 4)

        def has(self, name):
            return False

        def all_keys(self):
            return set()

    margs = SimpleNamespace(params_dtype=torch.float32, num_layers=4, num_experts=0, moe_layer_freq=[0] * 4)
    match = "does not match the model" if case == "shape" else "the model has no such tensor"
    with pytest.raises(RuntimeError, match=match):
        conv.build_mcore_state_dict(_Model(), _Store(), margs, {}, lambda *a: None)
    assert reads == ["tok_embeddings.weight"]


def test_config_from_args_keeps_args_and_sets_yarn_and_carrier(monkeypatch):
    """mistral_large4_config_from_args: Megatron gets a copy of args with MLA off and no rope_type
    (the caller's args unchanged); the result has rope_type yarn and pp_carrier_seq_length from
    seq_length (None when unset); another rope_type, or another config class, is refused."""
    import sys
    import types

    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_builders as builders,
    )

    Cfg = builders.MistralLarge4TransformerConfig
    seen = []
    result = {"cls": Cfg}

    class Other:  # a config class other than the requested one
        pass

    def fake(args, config_class=None):
        seen.append((args, config_class))
        return object.__new__(result["cls"])

    mod = types.ModuleType("megatron.training.arguments")
    mod.core_transformer_config_from_args = fake
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", mod)

    args = SimpleNamespace(multi_latent_attention=True, rope_type="yarn", seq_length=128)
    cfg = builders.mistral_large4_config_from_args(args)
    local, cls = seen[-1]
    assert cls is Cfg and local is not args
    assert local.multi_latent_attention is False and local.rope_type is None
    assert args.multi_latent_attention is True and args.rope_type == "yarn"  # caller's args unchanged
    assert cfg.rope_type == "yarn" and cfg.pp_carrier_seq_length == 128

    cfg = builders.mistral_large4_config_from_args(
        SimpleNamespace(multi_latent_attention=True, seq_length=None)
    )
    assert cfg.pp_carrier_seq_length is None

    with pytest.raises(ValueError, match="YaRN"):
        builders.mistral_large4_config_from_args(SimpleNamespace(rope_type="rope", seq_length=8))

    result["cls"] = Other
    with pytest.raises(TypeError, match="not MistralLarge4TransformerConfig"):
        builders.mistral_large4_config_from_args(SimpleNamespace(seq_length=8))


@pytest.mark.parametrize(
    "model,cp,turbo,hit",
    [
        ("mistral_large4", 2, True, True),
        ("mistral_large4", 1, True, False),
        ("mistral_large4", 2, False, False),
        ("deepseek_v3", 2, True, False),
    ],
)
def test_turbo_attention_off_with_cp(model, cp, turbo, hit):
    params = SimpleNamespace(model_type=model, context_parallel_size=cp, use_turbo_attention=turbo)
    ctx = _ctx("setup", params=params)
    assert mistral_large4_args_patches._turbo_attention_with_cp(ctx) is hit
    if hit:
        mistral_large4_args_patches.patch_mistral_large4_turbo_attention_cp(ctx)
        assert params.use_turbo_attention is False


def test_layer_lists_from_preset_warns_for_a_hub_id(tmp_path, monkeypatch):
    from primus.core.patches.patch_registry import PatchRegistry

    patch = PatchRegistry.get("megatron.mistral_large4.layer_lists_from_preset")
    args = SimpleNamespace(num_layers=4)
    hub = _ctx(
        "build_args", args, SimpleNamespace(model_type="mistral_large4", stage="sft", hf_path="mistralai/X")
    )
    local = _ctx(
        "build_args", args, SimpleNamespace(model_type="mistral_large4", stage="sft", hf_path=str(tmp_path))
    )
    (tmp_path / "config.json").write_text("{}")
    assert patch.condition(hub) and not patch.condition(local)
    # no hf_path (an SFT run from an already converted pretrained_checkpoint): also warned
    assert patch.condition(
        _ctx("build_args", args, SimpleNamespace(model_type="mistral_large4", stage="sft"))
    )
    # pretraining reads no checkpoint: no warning about its layout
    assert not patch.condition(
        _ctx("build_args", args, SimpleNamespace(model_type="mistral_large4", hf_path="mistralai/X"))
    )
    seen = []
    monkeypatch.setattr(mistral_large4_args_patches, "log_rank_0", seen.append)
    mistral_large4_args_patches.patch_mistral_large4_layer_lists_from_preset(hub)
    assert "mistralai/X" in seen[0] and "WARNING" in seen[0]


def test_sparse_core_attention_recompute_passes_only_tensors(monkeypatch):
    """Selective core_attn recompute on the sparse path: Megatron's CheckpointFunction saves its
    positional args with save_for_backward (tensors only), so softmax_scale must be bound, not
    passed. The stand-in checkpoint saves its args the same way and runs a backward."""
    from primus.backends.megatron.core.transformer.mistral_large4 import (
        attention as attn_mod,
    )

    class Saving(torch.autograd.Function):
        @staticmethod
        def forward(ctx, fn, *args):
            ctx.fn = fn
            ctx.save_for_backward(*args)  # TypeError for a float, as in Megatron
            with torch.no_grad():
                return fn(*args)

        @staticmethod
        def backward(ctx, grad):
            args = [t.detach().requires_grad_(t.is_floating_point()) for t in ctx.saved_tensors]
            with torch.enable_grad():
                out = ctx.fn(*args)
            torch.autograd.backward(out, grad)
            return (None,) + tuple(t.grad if t.is_floating_point() else None for t in args)

    calls = []

    def checkpoint(fn, distribute, *args):
        calls.append(args)
        return Saving.apply(fn, *args)

    monkeypatch.setattr(attn_mod.tensor_parallel, "checkpoint", checkpoint)
    s, b, n, d = 6, 1, 2, 4
    att = attn_mod.MistralLarge4Attention.__new__(attn_mod.MistralLarge4Attention)
    torch.nn.Module.__init__(att)
    att.softmax_scale = 0.5
    att.config = SimpleNamespace(attention_dropout=0.0)
    att.checkpoint_core_attention = True
    att.recompute_up_proj = False
    att.training = True
    q, k, v = (torch.randn(s, b, n, d, requires_grad=True) for _ in range(3))
    keep = torch.ones(b, s, s, dtype=torch.bool).tril_()
    att._ctx = {"sparse": True}
    monkeypatch.setattr(att, "_local_ctx", lambda ctx, device: ctx, raising=False)
    monkeypatch.setattr(att, "_qkv", lambda h, ctx: (q, k, v), raising=False)
    monkeypatch.setattr(att, "_keep", lambda h, ctx: keep, raising=False)
    att.linear_proj = lambda core: (core, None)
    out, _ = att.forward(torch.zeros(s, b, 8), None)
    assert all(torch.is_tensor(x) for x in calls[0])  # no float among the saved args
    out.sum().backward()
    ref = attn_mod.sparse_topk_attention(q.detach(), k.detach(), v.detach(), keep, 0.5)
    assert torch.allclose(out, ref) and q.grad is not None


def test_attention_ssmax_is_tp_sharded_in_dist_checkpoints(monkeypatch):
    """ssmax_s / ssmax_b hold this TP rank's heads: the attention's sharded_state_dict gives them
    the TP axis 0 (rank 1 of 2 writes the second half), not Megatron's TP-replicated default."""
    from megatron.core.transformer.multi_latent_attention import MLASelfAttention

    from primus.backends.megatron.core.transformer.mistral_large4.attention import (
        MistralLarge4Attention,
    )

    class Group:
        def __init__(self, rank, size):
            self._rank, self._size = rank, size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(MLASelfAttention, "sharded_state_dict", lambda self, prefix, offsets, metadata: {})
    attn = torch.nn.Module.__new__(MistralLarge4Attention)
    torch.nn.Module.__init__(attn)
    attn.ssmax_s = torch.nn.Parameter(torch.zeros(4))
    attn.ssmax_b = torch.nn.Parameter(torch.ones(4))
    attn.tp_group = Group(1, 2)
    sd = MistralLarge4Attention.sharded_state_dict(
        attn, "decoder.layers.0.self_attention.", (), {"dp_cp_group": Group(0, 1)}
    )
    for name in ("ssmax_s", "ssmax_b"):
        st = sd[f"decoder.layers.0.self_attention.{name}"]
        assert tuple(st.global_shape) == (8,) and tuple(st.global_offset) == (4,), (name, st)


@pytest.mark.parametrize(
    "pp_rank,pp_size,vp_stage,main",
    [(0, 1, None, True), (0, 3, 0, True), (1, 3, None, False), (2, 3, 1, False), (0, 3, 1, False)],
)
def test_block_rotary_alpha_has_one_main_replica_over_pp(pp_rank, pp_size, vp_stage, main, monkeypatch):
    """Every pipeline stage holds the same LeRoPE ``rotary.alpha``: only the first stage's copy is the
    main replica of the torch_dist checkpoint (else PP > 1 fails sharding validation)."""
    from megatron.core.dist_checkpointing.mapping import ShardedTensor, is_main_replica
    from megatron.core.transformer.transformer_block import TransformerBlock

    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_block import (
        MistralLarge4TransformerBlock,
    )

    class Group:
        def __init__(self, rank, size):
            self._rank, self._size = rank, size

        def rank(self):
            return self._rank

        def size(self):
            return self._size

    alpha = torch.zeros(4)
    st = ShardedTensor.from_rank_offsets("decoder.rotary.alpha", alpha, replica_id=(0, 0, 0))
    monkeypatch.setattr(torch.distributed, "is_initialized", lambda: True)
    monkeypatch.setattr(
        TransformerBlock,
        "sharded_state_dict",
        lambda self, prefix, offsets, metadata: {f"{prefix}rotary.alpha": st},
    )
    blk = torch.nn.Module.__new__(MistralLarge4TransformerBlock)
    torch.nn.Module.__init__(blk)
    blk.pg_collection = SimpleNamespace(pp=Group(pp_rank, pp_size))
    blk.vp_stage = vp_stage
    sd = MistralLarge4TransformerBlock.sharded_state_dict(blk, "decoder.", (), None)
    rid = sd["decoder.rotary.alpha"].replica_id
    assert is_main_replica(rid) is main, rid
    assert rid[1:] == (0, 0)


@pytest.mark.parametrize("tp,want", [(1, 128), (2, 256), (8, 1024), (None, 128)])
def test_vocab_divisor_takes_the_training_tp(tp, want, monkeypatch):
    """The conversion runs at TP 1 but pads the vocab as training at tensor_parallel_size pads it:
    build_margs returns margs whose make_vocab_size_divisible_by is the divisor x TP."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    args = SimpleNamespace(
        make_vocab_size_divisible_by=128,
        tensor_parallel_size=tp,
        hf_dir="/nonexistent",
        save_dir="/nonexistent",
        seq_length=64,
        dtype="bf16",
        ckpt_format="torch_dist",
    )
    assert conv.vocab_divisor(args) == want
    # Megatron's argument module imports GPU-only packages on import: a stand-in parser with the
    # attributes build_margs reads, and validate_args as identity (build_margs' own output is checked)
    import sys
    import types

    def add_megatron_arguments(p):
        p.set_defaults(make_vocab_size_divisible_by=128)
        return p

    fake = types.ModuleType("megatron.training.arguments")
    fake.add_megatron_arguments = add_megatron_arguments
    fake.validate_args = lambda m, *a, **kw: m
    monkeypatch.setitem(sys.modules, "megatron.training.arguments", fake)
    margs = conv.build_margs(conv.derive_megatron_fields(_release_text_config()), args)
    assert margs.make_vocab_size_divisible_by == want
    assert margs.tensor_model_parallel_size == 1  # converted unsharded


def test_converter_refuses_yarn_scaling_the_model_does_not_apply():
    """transformers' YaRN scales cos / sin by attention_factor, which it derives when the config
    omits it (0.1 ln(factor) + 1, ~1.76 for 2048) or from mscale / mscale_all_dim. The model applies
    none, so only configs whose scale resolves to 1.0 (the release: attention_factor 1.0) convert."""
    import math

    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    def cfg(**rope):
        c = _release_text_config()
        c["text_config"]["rope_parameters"] = {
            "rope_type": "yarn",
            "factor": 2048,
            "original_max_position_embeddings": 512,
            "rope_theta": 10000.0,
            **rope,
        }
        return c

    conv.derive_megatron_fields(cfg(attention_factor=1.0))  # the release
    assert conv.yarn_attention_factor({"factor": 2048}) == pytest.approx(0.1 * math.log(2048) + 1)
    with pytest.raises(ValueError, match="attention_factor 1.76"):
        conv.derive_megatron_fields(cfg())  # omitted: transformers' default, not 1.0
    with pytest.raises(ValueError, match="attention_factor"):
        conv.derive_megatron_fields(cfg(attention_factor=1.2))
    with pytest.raises(ValueError, match="mscale_all_dim"):
        conv.derive_megatron_fields(cfg(mscale=1.0, mscale_all_dim=1.0))  # ratio 1 but softmax mscale
    with pytest.raises(ValueError, match="attention_factor"):
        conv.derive_megatron_fields(cfg(mscale=2.0, mscale_all_dim=1.0))


def test_converter_rope_theta_layouts():
    """rope_theta is read from the rope dict or the top-level text config (rope_scaling layout);
    both missing, or both set and different, is an error, never a silent 10000."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    cfg = _release_text_config()
    assert conv.derive_megatron_fields(cfg)["rotary_base"] == 10000.0
    legacy = _release_text_config(rope_theta=1e6)
    rope = legacy["text_config"].pop("rope_parameters")
    del rope["rope_theta"]
    legacy["text_config"]["rope_scaling"] = rope
    assert conv.derive_megatron_fields(legacy)["rotary_base"] == 1e6
    same = _release_text_config(rope_theta=10000.0)
    assert conv.derive_megatron_fields(same)["rotary_base"] == 10000.0
    with pytest.raises(ValueError, match="!= top-level"):
        conv.derive_megatron_fields(_release_text_config(rope_theta=1e6))
    del legacy["text_config"]["rope_theta"]
    with pytest.raises(ValueError, match="no rope_theta"):
        conv.derive_megatron_fields(legacy)


def test_patches_install_their_wrappers_once(monkeypatch):
    """Applying the first-iteration-log patch replaces Megatron's training.training_log with
    the wrapper, once."""
    import sys
    import types

    from primus.backends.megatron.patches import (
        mistral_large4_training_log_patches as logp,
    )

    def training_log(loss_dict, total_loss_dict, learning_rate, iteration, *rest, is_first_iteration=False):
        return "logged"

    # stand-ins for Megatron's modules (the real training module imports GPU-only packages)
    megatron_global_vars = types.ModuleType("megatron.training.global_vars")
    megatron_global_vars.get_args = lambda: SimpleNamespace(log_interval=1)
    megatron_training = types.ModuleType("megatron.training.training")
    megatron_training.training_log = training_log
    package = types.ModuleType("megatron.training")
    package.__path__ = []
    package.global_vars, package.training = megatron_global_vars, megatron_training
    # the parent package too: run alone (or first in a worker) nothing has imported megatron yet
    parent = sys.modules.get("megatron") or types.ModuleType("megatron")
    monkeypatch.setitem(sys.modules, "megatron", parent)
    monkeypatch.setitem(sys.modules, "megatron.training", package)
    monkeypatch.setattr(parent, "training", package, raising=False)
    for name, mod in (("global_vars", megatron_global_vars), ("training", megatron_training)):
        monkeypatch.setitem(sys.modules, f"megatron.training.{name}", mod)
    ctx = _ctx("before_train", params=SimpleNamespace(model_type="mistral_large4"))
    for _ in range(2):  # a second application does not wrap twice
        logp.patch_mistral_large4_first_iteration_log(ctx)
    wrapped_log = megatron_training.training_log
    assert wrapped_log is not training_log and getattr(wrapped_log, logp._WRAPPED_MARK)
    assert wrapped_log.__wrapped__ is training_log and wrapped_log.__name__ == "training_log"

    # a training_log without the parameters the wrapper binds is not wrapped (no silent no-op)
    def other(*a, **kw):
        return None

    megatron_training.training_log = other
    assert logp.missing_params(other) == list(logp.REQUIRED_PARAMS)
    logp.patch_mistral_large4_first_iteration_log(ctx)
    assert megatron_training.training_log is other


@pytest.mark.parametrize("shape", [(256, 384), (130, 200), (130, 260), (128, 128)])
def test_dequant_block_fp8_fast_path_matches_the_formula(shape):
    """The broadcast path (dims multiples of 128) and the general path give W = fp8 * scale per tile."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    torch.manual_seed(0)
    rows, cols = shape
    w = torch.randn(rows, cols)
    scale = torch.rand((rows + 127) // 128, (cols + 127) // 128) + 0.5
    want = torch.empty(rows, cols)
    for i in range(rows):
        for j in range(0, cols, 128):
            want[i, j : j + 128] = w[i, j : j + 128] * scale[i // 128, j // 128]
    w_in = w.clone()
    got = conv.dequant_block_fp8(w_in, scale)
    assert torch.equal(got, want)
    assert torch.equal(w_in, w)  # the input (possibly a store's cached tensor) is not modified


@pytest.mark.parametrize(
    "shape,scale_shape", [((256, 256), (4, 4)), ((130, 200), (1, 2)), ((256, 256), (2, 1))]
)
def test_dequant_block_fp8_refuses_a_scale_on_another_grid(shape, scale_shape):
    """A scale whose grid is not [ceil(r/128), ceil(c/128)] is refused on both paths, not truncated."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    with pytest.raises(ValueError, match="block-FP8 scale shape"):
        conv.dequant_block_fp8(torch.ones(*shape), torch.ones(*scale_shape))


def test_converter_dtype_validated():
    """bf16 / fp16 / fp32 and their torch / HF names are accepted; anything else is refused."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    for raw, want in (
        ("bf16", "bf16"),
        ("bfloat16", "bf16"),
        ("torch.bfloat16", "bf16"),
        ("fp16", "fp16"),
        ("float16", "fp16"),
        ("FP32", "fp32"),
        ("float32", "fp32"),
    ):
        assert conv.conversion_dtype(raw) == want
    for bad in ("bf61", "int8", "", None):
        with pytest.raises(ValueError, match="unknown dtype"):
            conv.conversion_dtype(bad)
    with pytest.raises(ValueError, match="unknown dtype"):
        conv.convert("/nonexistent", "/nonexistent-out", dtype="bfloat61")


@pytest.mark.parametrize("value", [3, "3", "{[1]: 0}", "abc"])
def test_layer_list_value_not_a_list_names_the_key(value):
    """A scalar or a non-list literal for a per-layer list raises ValueError naming the key."""
    with pytest.raises(ValueError, match="no_rope_layers .* is not a list literal"):
        mistral_large4_args_patches._comparable(value, "no_rope_layers", 4)


def _without_rope_key(*keys):
    def edit(cfg):
        for k in keys:
            del cfg["text_config"]["rope_parameters"][k]

    return edit


def _set_text(**kw):
    def edit(cfg):
        cfg["text_config"].update(kw)

    return edit


@pytest.mark.parametrize(
    "edit,match",
    [
        (_set_text(n_group=8), "n_group=8"),
        (_set_text(topk_group=4), "topk_group=4"),
        (_set_text(scoring_func="sigmoid"), "scoring_func 'sigmoid'"),
        (_set_text(norm_topk_prob=False), "norm_topk_prob false"),
        (_set_text(topk_method="noaux_tc"), "topk_method 'noaux_tc'"),
        (_set_text(tie_word_embeddings=True), "tied embeddings"),
        (lambda cfg: cfg.update(tie_word_embeddings=True), "tied embeddings"),  # top-level key
        (lambda cfg: cfg["text_config"].pop("rope_parameters"), "expects YaRN rope parameters"),
        (
            _set_text(rope_parameters={"rope_type": "default", "rope_theta": 1e4}),
            "expects YaRN rope, got 'default'",
        ),
        (_without_rope_key("factor"), "lack \\['factor'\\]"),
        (
            _without_rope_key("original_max_position_embeddings"),
            "lack \\['original_max_position_embeddings'\\]",
        ),
    ],
)
def test_converter_refuses_configs_it_does_not_implement(edit, match):
    """Each routing, embedding and rope setting the model does not implement is refused with its
    own message (never converted with other semantics)."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    conv.derive_megatron_fields(_release_text_config())  # the release's values convert
    cfg = _release_text_config()
    edit(cfg)
    with pytest.raises(ValueError, match=match):
        conv.derive_megatron_fields(cfg)


@pytest.mark.parametrize("topk_method", [None, "greedy"])
def test_converter_accepts_release_routing_values(topk_method):
    """n_group / topk_group 1, softmax, norm_topk_prob true and greedy (or absent) top-k convert."""
    from primus.backends.megatron.checkpoint import (
        native_loader_mistral_large4_text as conv,
    )

    conv.derive_megatron_fields(
        _release_text_config(
            n_group=1, topk_group=1, scoring_func="softmax", norm_topk_prob=True, topk_method=topk_method
        )
    )


def test_kv_sharing_predicates_refuse_invalid_sources():
    """kv_sharing_active / kv_carrier_needed (which size the PP wire) refuse a kv_source_layers that
    points forward, out of range, or at a layer that does not own its KV."""
    from primus.backends.megatron.core.models.mistral_large4.mistral_large4_layout import (
        kv_carrier_needed,
        kv_sharing_active,
    )

    assert kv_carrier_needed([0, 1, 2, 1], 4, 2) is True
    assert kv_sharing_active([0, 1, 2, 3], 4) is False
    for bad in ([0, 1, 5, 1], [0, 2, 2, 3], [0, 0, 1, 3], [-1, 1, 2, 3]):
        with pytest.raises(ValueError, match="breaks the rule"):
            kv_carrier_needed(bad, 4, 2)
    # with no_rope_layers (as the decoder block and the PP shape patch pass it), the rope / NoPE
    # kind rule of the model config too: layer 3 (rope) cannot reuse layer 1's NoPE latent
    assert kv_carrier_needed([0, 1, 2, 1], 4, 2, [1, 0, 1, 0]) is True
    with pytest.raises(ValueError, match="rope/NoPE kinds differ"):
        kv_carrier_needed([0, 1, 2, 1], 4, 2, [1, 0, 1, 1])


def test_turbo_modules_follow_enable_primus_turbo(monkeypatch):
    """The layer spec's TE provider takes a Primus-Turbo module only with enable_primus_turbo on
    (and primus_turbo installed), as the shared megatron.turbo patches do; otherwise the use_turbo_*
    switches are read as off, so attention and grouped experts are TE's. The provider module is a
    stand-in (the real one imports Transformer Engine, which needs a GPU)."""
    import sys
    import types

    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_layer_specs as specs,
    )
    from primus.backends.megatron.core.models.mistral_large4 import (
        mistral_large4_layout as layout,
    )

    on = dict(
        use_turbo_gemm=True,
        use_turbo_norm_te_linear=True,
        use_turbo_attention=True,
        use_turbo_grouped_gemm=True,
        moe_use_legacy_grouped_gemm=False,
    )
    primus_args = {}

    class Provider:
        def __init__(self):
            self.cfg = primus_args["cfg"]

    fake = types.ModuleType("primus.backends.megatron.core.extensions.transformer_engine_spec_provider")
    fake.PrimusTurboSpecProvider = Provider
    monkeypatch.setitem(sys.modules, fake.__name__, fake)
    cfg = SimpleNamespace(transformer_impl="transformer_engine")

    primus_args["cfg"] = SimpleNamespace(enable_primus_turbo=False, **on)
    p = specs._backend(cfg)
    assert not any(getattr(p.cfg, k) for k in specs.TURBO_MODULE_SWITCHES)
    assert primus_args["cfg"].use_turbo_attention  # the global args are not modified

    primus_args["cfg"] = SimpleNamespace(enable_primus_turbo=True, **on)
    monkeypatch.setattr(layout.importlib.util, "find_spec", lambda name: None)  # not installed
    assert not specs._backend(cfg).cfg.use_turbo_grouped_gemm
    monkeypatch.setattr(layout.importlib.util, "find_spec", lambda name: object())
    p = specs._backend(cfg)
    assert p.cfg is primus_args["cfg"] and p.cfg.use_turbo_attention and p.cfg.use_turbo_grouped_gemm
