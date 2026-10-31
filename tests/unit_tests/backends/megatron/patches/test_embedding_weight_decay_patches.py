###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Unit tests for the opt-in skip_embedding_weight_decay optimizer patch
(primus/backends/megatron/patches/embedding_weight_decay_patches.py) and its core helper
(primus/backends/megatron/core/optimizer/embedding_weight_decay.py).

Built through Megatron's own _get_param_groups and through the registered patch: the
word-embedding weight (and the tied copy) gets wd_mult 0, the standard bias/norm
overrides stay, linear weights keep weight decay. The _get_param_groups wrapper acts
only inside a setup_model_and_optimizer call (outside one it calls the original
unchanged), where the flag off or missing args is an error. Muon and the FSDP2 custom
optimizers stop the run through run_patches (SystemExit). An optimizer built without
the check fails the build. A frozen embedding is accepted. The after_train patch
(uninstall()) restores Megatron's functions.

Megatron-dependent tests use ``tests.utils.require_megatron`` (MEGATRON_PATH aware).
"""

from types import SimpleNamespace

import pytest

pytest.importorskip("torch")

import torch  # noqa: E402

from tests.utils import require_megatron  # noqa: E402


def _installed(patch):
    """``{attribute name: wrapper}`` for the wrappers the patch installed and has not removed."""
    return {key.rsplit(".", 1)[-1]: entry[3] for key, entry in patch._installed.items()}


# --------------------------------------------------------------------------- embedding no-wd
class _TinyGPT(torch.nn.Module):
    """Parameter names as in Megatron's GPTModel (tied embeddings, last PP stage copy)."""

    def __init__(self, tied_copy=True):
        super().__init__()
        self.embedding = torch.nn.Module()
        self.embedding.word_embeddings = torch.nn.Embedding(8, 4)
        self.decoder = torch.nn.Module()
        self.decoder.linear_qkv = torch.nn.Linear(4, 4, bias=True)
        self.decoder.final_layernorm = torch.nn.LayerNorm(4)
        self.output_layer = torch.nn.Linear(4, 8, bias=False)
        if tied_copy:
            self.output_layer.weight.shared_embedding = True


@pytest.fixture
def single_rank_group(tmp_path):
    """_get_param_groups all-gathers the group keys, so it needs a (1-rank, gloo) process group."""
    import torch.distributed as dist

    if dist.is_initialized():
        yield
        return
    dist.init_process_group("gloo", init_method=f"file://{tmp_path}/pg", rank=0, world_size=1)
    try:
        yield
    finally:
        dist.destroy_process_group()


def _wd_by_name(model, groups):
    by_id = {id(p): g.get("wd_mult", 1.0) for g in groups for p in g["params"]}
    return {n: by_id[id(p)] for n, p in model.named_parameters()}


@pytest.mark.parametrize("tied_copy", [True, False])
def test_embedding_no_wd_param_groups(single_rank_group, tied_copy):
    require_megatron()
    from megatron.core.optimizer import _get_param_groups
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    from primus.backends.megatron.core.optimizer.embedding_weight_decay import (
        add_embedding_no_wd,
        embedding_wd_mults,
    )

    config = OptimizerConfig(weight_decay=0.1)
    model = _TinyGPT(tied_copy)
    # None = "use Megatron's standard overrides": they must be kept, not replaced.
    groups = _get_param_groups([model], config, add_embedding_no_wd(None, config))
    wd = _wd_by_name(model, groups)
    assert wd["embedding.word_embeddings.weight"] == 0.0
    assert wd["output_layer.weight"] == (0.0 if tied_copy else 1.0)
    assert wd["decoder.linear_qkv.weight"] == 1.0
    assert wd["decoder.linear_qkv.bias"] == 0.0  # standard override kept
    assert wd["decoder.final_layernorm.weight"] == 0.0  # standard override kept
    assert all(w == 0.0 for _, w in embedding_wd_mults([model], groups))

    # Without the flag: Megatron's default leaves weight decay on the embedding.
    default = _wd_by_name(model, _get_param_groups([model], config, None))
    assert default["embedding.word_embeddings.weight"] == 1.0


def test_add_embedding_no_wd_none_needs_config():
    from primus.backends.megatron.core.optimizer.embedding_weight_decay import (
        add_embedding_no_wd,
    )

    with pytest.raises(ValueError):
        add_embedding_no_wd(None)


def test_add_embedding_no_wd_merges_same_key():
    require_megatron()
    from primus.backends.megatron.core.optimizer.embedding_weight_decay import (
        add_embedding_no_wd,
        embedding_no_wd_overrides,
    )

    (key,) = embedding_no_wd_overrides()
    merged = add_embedding_no_wd({key: {"max_lr": 1e-3}})
    assert merged[key] == {"max_lr": 1e-3, "wd_mult": 0.0}
    # {} is the caller turning the standard overrides off: kept as is.
    assert list(add_embedding_no_wd({})) == [key]


def test_check_embedding_no_wd_flags_missing_and_ungrouped(single_rank_group):
    from primus.backends.megatron.core.optimizer.embedding_weight_decay import (
        check_embedding_no_wd,
    )

    model = _TinyGPT()
    others = [p for n, p in model.named_parameters() if "word_embeddings" not in n]
    with pytest.raises(RuntimeError, match="not in any param group"):
        check_embedding_no_wd([model], [{"params": others, "wd_mult": 0.0}])
    # A group without wd_mult is Megatron's default weight decay, not a pass.
    with pytest.raises(RuntimeError, match="kept weight decay"):
        check_embedding_no_wd([model], [{"params": list(model.parameters())}])


def test_check_embedding_no_wd_frozen_and_first_stage():
    require_megatron()
    from primus.backends.megatron.core.optimizer.embedding_weight_decay import (
        check_embedding_no_wd,
    )

    # A frozen embedding needs no override: no error on a first-stage chunk.
    model = _TinyGPT(tied_copy=False)
    model.pre_process = True
    model.embedding.word_embeddings.weight.requires_grad_(False)
    trainable = [p for p in model.parameters() if p.requires_grad]
    assert check_embedding_no_wd([model], [{"params": trainable, "wd_mult": 1.0}]) == []
    # A first-stage chunk without any embedding parameter is still an error, per chunk.
    stage = torch.nn.Module()
    stage.pre_process = True
    stage.linear = torch.nn.Linear(2, 2)
    model.embedding.word_embeddings.weight.requires_grad_(True)
    groups = [{"params": [model.embedding.word_embeddings.weight], "wd_mult": 0.0}]
    with pytest.raises(RuntimeError, match="chunk 1 has no parameter matching"):
        check_embedding_no_wd([model, stage], groups)


def _ctx(**params):
    from primus.core.patches import PatchContext

    defaults = {"skip_embedding_weight_decay": True, "optimizer": "adam", "bf16": True}
    module_config = SimpleNamespace(params=SimpleNamespace(**{**defaults, **params}))
    return PatchContext(backend="megatron", phase="before_train", extra={"module_config": module_config})


def test_patch_condition_is_the_flag():
    from primus.backends.megatron.patches.embedding_weight_decay_patches import (
        skip_embedding_wd_enabled,
        skip_embedding_wd_in_args,
    )

    assert skip_embedding_wd_enabled(_ctx(skip_embedding_weight_decay=False)) is False
    assert skip_embedding_wd_enabled(_ctx()) is True
    # Strings are parsed (YAML env substitution / CLI): "false" is off, not bool("false").
    assert skip_embedding_wd_enabled(_ctx(skip_embedding_weight_decay="false")) is False
    assert skip_embedding_wd_enabled(_ctx(skip_embedding_weight_decay="true")) is True
    for bad in ("", "maybe", 2):
        with pytest.raises(ValueError, match="skip_embedding_weight_decay must be a boolean"):
            skip_embedding_wd_enabled(_ctx(skip_embedding_weight_decay=bad))
    # The flag on Megatron's args is parsed the same way.
    assert skip_embedding_wd_in_args(SimpleNamespace(skip_embedding_weight_decay="false")) is False
    assert skip_embedding_wd_in_args(SimpleNamespace(skip_embedding_weight_decay="true")) is True
    assert skip_embedding_wd_in_args(SimpleNamespace()) is False
    assert skip_embedding_wd_in_args(None) is False
    with pytest.raises(ValueError):
        skip_embedding_wd_in_args(SimpleNamespace(skip_embedding_weight_decay="maybe"))


@pytest.mark.parametrize(
    "args,unsupported",
    [
        # Same case-sensitive test as Megatron's setup_model_and_optimizer ('muon' in args.optimizer).
        ({"optimizer": "muon"}, True),
        ({"optimizer": "dist_muon"}, True),
        ({"optimizer": "adam_muon"}, True),
        ({"optimizer": "Muon"}, False),
        ({"optimizer": "adam"}, False),
        ({"optimizer": "sgd"}, False),
        # The Primus FSDP2 custom optimizers drop config_overrides (bf16 only).
        ({"use_torch_fsdp2": True, "use_fsdp2_fp32_param_optimizer": True}, True),
        ({"use_torch_fsdp2": True, "use_fsdp2_bf16_master_weight_optimizer": True}, True),
        ({"use_torch_fsdp2": True, "use_fsdp2_fp32_param_optimizer": True, "bf16": False}, False),
    ],
)
def test_unsupported_reason(args, unsupported):
    from primus.backends.megatron.patches.embedding_weight_decay_patches import (
        unsupported_reason,
    )

    reason = unsupported_reason(SimpleNamespace(**{"optimizer": "adam", "bf16": True, **args}))
    assert (reason is not None) is unsupported


@pytest.fixture
def patched_megatron(monkeypatch):
    """The real Megatron _get_param_groups, restored afterwards; ``args`` is what the
    wrapper reads as Megatron's args."""
    require_megatron()
    import megatron.core.optimizer as optimizer_module
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    from primus.backends.megatron.patches import embedding_weight_decay_patches as patch

    monkeypatch.setattr(optimizer_module, "_get_param_groups", optimizer_module._get_param_groups)
    args = SimpleNamespace(skip_embedding_weight_decay=True, optimizer="adam", bf16=True)
    monkeypatch.setattr(patch, "_megatron_args", lambda: args)
    yield SimpleNamespace(
        patch=patch, opt=optimizer_module, config=OptimizerConfig(weight_decay=0.1), args=args
    )
    # Tests that call install() leave no wrapper behind for later tests in this process.
    patch.uninstall()
    patch.uninstall_resume_check()


def _build_groups(m, model, overrides=None, in_setup=True):
    """What get_megatron_optimizer does: Megatron's own _get_param_groups_and_buffers,
    which looks up _get_param_groups in its module at call time. ``in_setup``: inside a
    setup_model_and_optimizer call (the wrapped one), where the override applies."""

    def build():
        groups, _ = m.opt._get_param_groups_and_buffers(
            [model], 0, m.config, overrides, filter_fn=lambda g: True, buffer_name="buffers"
        )
        return groups

    if not in_setup:
        return build()
    out = []
    m.patch.wrap_setup(lambda: (out.append(build()), None, None))()
    return out[0]


def test_patch_applies_through_real_megatron_param_group_chain(single_rank_group, patched_megatron):
    m = patched_megatron
    model = _TinyGPT()
    assert m.patch.install() is True
    assert m.patch.install() is False  # idempotent
    groups = _build_groups(m, model)
    wd = _wd_by_name(model, groups)
    assert wd["embedding.word_embeddings.weight"] == 0.0
    assert wd["output_layer.weight"] == 0.0  # tied copy
    assert wd["decoder.linear_qkv.weight"] == 1.0
    from primus.backends.megatron.core.optimizer.embedding_weight_decay import (
        weight_decay_summary,
    )

    assert weight_decay_summary([model], groups) == {
        "decayed_2d": 1,
        "not_decayed_2d": 0,
        "untied_output_layer": False,
    }
    assert weight_decay_summary([_TinyGPT(tied_copy=False)], [])["untied_output_layer"] is True


def test_patch_outside_setup_is_unchanged(single_rank_group, patched_megatron):
    """Outside a setup_model_and_optimizer call the installed wrapper calls Megatron's
    _get_param_groups unchanged, also with the flag off or Megatron's args unset."""
    m = patched_megatron
    model = _TinyGPT()
    m.patch.install()
    assert (
        _wd_by_name(model, _build_groups(m, model, in_setup=False))["embedding.word_embeddings.weight"] == 1.0
    )
    m.args.skip_embedding_weight_decay = False
    assert (
        _wd_by_name(model, _build_groups(m, model, in_setup=False))["embedding.word_embeddings.weight"] == 1.0
    )
    m.patch.uninstall()
    assert not m.patch._scopes


def test_patch_flag_off_at_call_time_raises(single_rank_group, patched_megatron):
    """Inside setup_model_and_optimizer, the flag off in Megatron's args is an error, not
    a silent fall-back to the original groups; the scope is closed afterwards."""
    m = patched_megatron
    model = _TinyGPT()
    m.patch.install()
    assert _wd_by_name(model, _build_groups(m, model))["embedding.word_embeddings.weight"] == 0.0
    m.args.skip_embedding_weight_decay = False
    with pytest.raises(RuntimeError, match="off in Megatron's args"):
        _build_groups(m, model)
    assert m.patch._scopes == []


def test_patch_without_megatron_args_raises(single_rank_group, monkeypatch):
    """Inside setup_model_and_optimizer, before Megatron's args are set, the wrapper raises
    instead of building the groups without the override."""
    require_megatron()
    import megatron.training.global_vars as gv
    from megatron.core.optimizer.optimizer_config import OptimizerConfig

    from primus.backends.megatron.patches import embedding_weight_decay_patches as patch

    monkeypatch.setattr(gv, "_GLOBAL_ARGS", None)
    assert patch._megatron_args() is None
    calls = []
    # default getter: Megatron's (unset) args
    wrapper = patch.wrap(lambda model_chunks, config, config_overrides: calls.append(1))
    setup = patch.wrap_setup(
        lambda: (wrapper([_TinyGPT()], OptimizerConfig(weight_decay=0.1), None), None, None)
    )
    with pytest.raises(RuntimeError, match="before Megatron's args are set"):
        setup()
    assert calls == []
    assert patch._scopes == []


def test_setup_wrapper_fails_when_param_groups_were_not_checked(single_rank_group, patched_megatron):
    """setup_model_and_optimizer fails if it built an optimizer without the wrapped
    _get_param_groups, and passes when the wrapper checked the groups in the same call.
    A build checked in an earlier call does not count for a later one."""
    m = patched_megatron
    model = _TinyGPT()
    m.patch.install()

    def through_wrapper():
        _build_groups(m, model, in_setup=False)  # inside the outer wrapped call below
        return [model], object(), None

    assert m.patch.wrap_setup(through_wrapper)()[0] == [model]

    bypass = m.patch.wrap_setup(lambda: ([model], object(), None))
    with pytest.raises(RuntimeError, match="without the wrapped _get_param_groups"):
        bypass()
    # No optimizer (skip_train): nothing to check.
    assert m.patch.wrap_setup(lambda: ([model], None, None))()[1] is None
    assert m.patch._scopes == []


def test_patch_check_raises_when_override_does_not_apply(single_rank_group, patched_megatron):
    """A model whose embedding is not matched by the override fails the build."""
    m = patched_megatron
    # An "original" that drops the overrides: the embedding keeps weight decay.
    from megatron.core.optimizer import _get_param_groups as real

    wrapper = m.patch.wrap(
        lambda model_chunks, config, config_overrides: real(model_chunks, config, None), lambda: m.args
    )
    with pytest.raises(RuntimeError, match="kept weight decay"):
        m.patch.wrap_setup(lambda: (wrapper([_TinyGPT()], m.config, None), None, None))()


def test_patch_uninstall_restores_original(patched_megatron):
    import megatron.training.training as training_module

    m = patched_megatron
    before = m.opt._get_param_groups
    before_setup = training_module.setup_model_and_optimizer
    m.patch.install()
    assert m.opt._get_param_groups is not before
    assert training_module.setup_model_and_optimizer is not before_setup
    assert m.patch.uninstall() == (
        ["megatron.training.training.setup_model_and_optimizer", "megatron.core.optimizer._get_param_groups"],
        [],
    )
    assert m.opt._get_param_groups is before
    assert training_module.setup_model_and_optimizer is before_setup
    assert m.patch.uninstall() == ([], [])  # no-op
    assert m.opt._get_param_groups is before
    assert _installed(m.patch) == {}


def test_patch_found_by_identity_not_by_attribute(single_rank_group, patched_megatron, monkeypatch):
    """A later patch's functools.wraps wrapper over ours (it copies any function
    attribute) is not taken for ours: install() does not wrap again, uninstall() leaves
    the outer wrapper in place and turns ours off (calls pass through unchanged), and a
    plain wrapper that only carries a copied attribute is not removed."""
    import functools

    import megatron.training.training as training_module

    m = patched_megatron
    before_setup = training_module.setup_model_and_optimizer
    monkeypatch.setattr(training_module, "setup_model_and_optimizer", before_setup)
    model = _TinyGPT()
    assert m.patch.install() is True
    ours = m.opt._get_param_groups
    assert _installed(m.patch)["_get_param_groups"] is ours

    @functools.wraps(ours)
    def outer(*a, **k):
        return ours(*a, **k)

    m.opt._get_param_groups = outer
    assert m.patch.install() is False and m.opt._get_param_groups is outer
    assert _wd_by_name(model, _build_groups(m, model))["embedding.word_embeddings.weight"] == 0.0
    assert m.patch.uninstall() == (
        ["megatron.training.training.setup_model_and_optimizer"],
        ["megatron.core.optimizer._get_param_groups"],
    )
    assert m.opt._get_param_groups is outer
    assert training_module.setup_model_and_optimizer is before_setup
    # Turned off: under a (now unwrapped) setup scope the original groups come back.
    groups = []
    m.patch.wrap_setup(lambda: (groups.append(_build_groups(m, model, in_setup=False)), None, None))()
    assert _wd_by_name(model, groups[0])["embedding.word_embeddings.weight"] == 1.0
    assert m.patch.uninstall() == ([], [])
    m.opt._get_param_groups = ours.__wrapped__


def _global_names(fn):
    """Names that fn looks up as module globals at call time (LOAD_GLOBAL)."""
    import dis

    return {i.argval for i in dis.get_instructions(fn) if i.opname == "LOAD_GLOBAL"}


def test_real_megatron_binds_wrapped_name_at_call_time():
    """The real optimizer chain finds _get_param_groups through its module globals at
    call time, so the wrapper applies. If upstream moves to a by-name import, this fails."""
    require_megatron()
    import megatron.core.optimizer as optimizer_module

    chain = optimizer_module._get_param_groups_and_buffers
    assert chain.__globals__ is vars(optimizer_module)
    assert "_get_param_groups" in _global_names(chain)
    assert "_get_param_groups_and_buffers" in _global_names(optimizer_module.get_megatron_optimizer)


def test_registered_patch_checks_backend_args(patched_megatron):
    """check_and_install checks the backend args (Megatron's args object): flag set, and a
    supported optimizer; only then does it install."""
    m = patched_megatron
    before = m.opt._get_param_groups
    ctx = _ctx()
    ctx.extra["backend_args"] = SimpleNamespace(skip_embedding_weight_decay=False)
    with pytest.raises(RuntimeError, match="not in the backend args"):
        m.patch.check_and_install(ctx)
    ctx.extra["backend_args"] = SimpleNamespace(skip_embedding_weight_decay=True, optimizer="dist_muon")
    with pytest.raises(NotImplementedError, match="muon"):
        m.patch.check_and_install(ctx)
    assert m.opt._get_param_groups is before
    ctx.extra["backend_args"] = SimpleNamespace(skip_embedding_weight_decay=True, optimizer="adam")
    m.patch.check_and_install(ctx)
    assert m.opt._get_param_groups is not before
    m.patch.uninstall()


def test_registered_patch_checks_without_backend_args(patched_megatron, monkeypatch):
    """Without backend_args the same checks run on Megatron's get_args(); with no args at
    all the patch fails instead of installing unchecked."""
    m = patched_megatron
    before = m.opt._get_param_groups
    ctx = _ctx()
    ctx.extra.pop("backend_args", None)
    monkeypatch.setattr(
        m.patch, "_megatron_args", lambda: SimpleNamespace(skip_embedding_weight_decay=True, optimizer="muon")
    )
    with pytest.raises(NotImplementedError, match="muon"):
        m.patch.check_and_install(ctx)
    monkeypatch.setattr(
        m.patch,
        "_megatron_args",
        lambda: SimpleNamespace(
            skip_embedding_weight_decay=True,
            use_torch_fsdp2=True,
            bf16=True,
            use_fsdp2_fp32_param_optimizer=True,
        ),
    )
    with pytest.raises(NotImplementedError, match="fsdp2"):
        m.patch.check_and_install(ctx)
    monkeypatch.setattr(m.patch, "_megatron_args", lambda: None)
    with pytest.raises(RuntimeError, match="no Megatron args"):
        m.patch.check_and_install(ctx)
    assert m.opt._get_param_groups is before


_PATCH_ID = "megatron.optimizer.skip_embedding_weight_decay"


@pytest.mark.parametrize(
    "backend_args",
    [
        SimpleNamespace(skip_embedding_weight_decay=True, optimizer="muon"),
        SimpleNamespace(
            skip_embedding_weight_decay=True,
            optimizer="adam",
            bf16=True,
            use_torch_fsdp2=True,
            use_fsdp2_fp32_param_optimizer=True,
        ),
        SimpleNamespace(skip_embedding_weight_decay=False),
    ],
)
def test_run_patches_stops_on_install_error(patched_megatron, backend_args):
    """Through run_patches with its default stop_on_error=False (as train_runtime calls
    it), an install error stops the run instead of being logged and skipped."""
    from primus.core.patches.patch_runner import run_patches

    m = patched_megatron
    before = m.opt._get_param_groups
    ctx = _ctx()
    with pytest.raises(SystemExit, match="could not be applied"):
        run_patches(
            backend="megatron",
            phase="before_train",
            enabled_ids=[_PATCH_ID],
            extra={**ctx.extra, "backend_args": backend_args},
        )
    assert m.opt._get_param_groups is before


def test_run_patches_before_and_after_train(patched_megatron):
    """before_train installs both wrappers, after_train removes them."""
    import megatron.training.training as training_module

    from primus.core.patches.patch_runner import run_patches

    m = patched_megatron
    before, before_setup = m.opt._get_param_groups, training_module.setup_model_and_optimizer
    extra = {
        **_ctx().extra,
        "backend_args": SimpleNamespace(skip_embedding_weight_decay=True, optimizer="adam"),
    }
    ids = [_PATCH_ID, _PATCH_ID + ".uninstall"]
    assert run_patches(backend="megatron", phase="before_train", enabled_ids=ids, extra=extra) == 1
    assert m.opt._get_param_groups is not before
    assert training_module.setup_model_and_optimizer is not before_setup
    assert run_patches(backend="megatron", phase="after_train", enabled_ids=ids, extra=extra) == 1
    assert m.opt._get_param_groups is before
    assert training_module.setup_model_and_optimizer is before_setup


def test_fsdp2_optimizer_rule_agrees_with_its_users():
    """core/optimizer/fsdp2_optimizer_flags.py (used by unsupported_reason) has the same
    activation rule as the registered FSDP2 optimizer patches' conditions. (Its flag names
    are the exclusivity check's own tuple, re-exported, so they cannot drift.)"""
    import primus.backends.megatron.patches.optimizer_patches  # noqa: F401
    from primus.backends.megatron.core.optimizer import fsdp2_optimizer_flags as owner
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p
    from primus.core.patches import PatchRegistry

    for flag, patch_id in (
        ("use_fsdp2_fp32_param_optimizer", "megatron.optimizer.fsdp2_fp32_param"),
        ("use_fsdp2_bf16_master_weight_optimizer", "megatron.optimizer.fsdp2_bf16_master_weight"),
    ):
        for on, fsdp2, bf16 in (
            (True, True, True),
            (True, True, False),
            (True, False, True),
            (False, True, True),
        ):
            args = SimpleNamespace(**{flag: on, "use_torch_fsdp2": fsdp2, "bf16": bf16})
            ctx = SimpleNamespace(extra={"module_config": SimpleNamespace(params=args)})
            active = owner.active_fsdp2_optimizers(args) == [flag]
            assert active is (on and fsdp2 and bf16)
            assert bool(PatchRegistry.get(patch_id).condition(ctx)) is active
            assert (p.unsupported_reason(args) is not None) is active


# --------------------------------------------------------------------------- resume check
def _ckpt_args(**kw):
    return SimpleNamespace(**kw)


@pytest.mark.parametrize(
    "saved,now,run,raises",
    [
        (True, True, {}, False),
        (False, False, {}, False),
        (None, False, {}, False),  # saved before the flag existed: off
        (None, True, {}, True),
        (True, False, {}, True),  # resume that turns the flag off
        (False, True, {}, True),  # resume that turns the flag on
        (True, False, {"no_load_optim": True}, False),
        (False, True, {"finetune": True}, False),
    ],
)
def test_check_resume_flag(saved, now, run, raises):
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    ckpt = _ckpt_args() if saved is None else _ckpt_args(skip_embedding_weight_decay=saved)
    args = SimpleNamespace(skip_embedding_weight_decay=now, **run)
    if raises:
        with pytest.raises(RuntimeError, match="optimizer state cannot be loaded"):
            p.check_resume_flag(args, ckpt)
    else:
        p.check_resume_flag(args, ckpt)
    p.check_resume_flag(args, None)  # no saved args: nothing to check


def test_check_resume_flag_with_a_non_boolean_saved_value():
    """An unreadable saved value stops the run: the param groups cannot be confirmed."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    args, ckpt = SimpleNamespace(skip_embedding_weight_decay=True), _ckpt_args(
        skip_embedding_weight_decay="maybe"
    )
    with pytest.raises(RuntimeError, match="cannot confirm"):
        p.check_resume_flag(args, ckpt)


def test_resume_check_condition_reads_the_megatron_args():
    """A load dir set only in Megatron's final args also turns the check on."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    ctx = _ctx()
    assert p.resume_check_needed(ctx) is False
    ctx.extra["backend_args"] = SimpleNamespace(load="/ckpt")
    assert p.resume_check_needed(ctx) is True


@pytest.mark.parametrize("flag", [True, False, None, "false"])
def test_resume_check_runs_for_every_load(flag):
    """Both directions are checked: the check is installed for any run that loads a
    checkpoint, whatever the flag; a run that loads nothing is never wrapped."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    assert p.resume_check_needed(_ctx(skip_embedding_weight_decay=flag, load="/ckpt")) is True
    assert p.resume_check_needed(_ctx(skip_embedding_weight_decay=flag)) is False
    assert p.resume_check_uninstall_needed(_ctx(skip_embedding_weight_decay=flag)) is False


def test_flag_off_run_stops_on_a_checkpoint_saved_with_the_flag_on():
    """The reviewer's case: a generic recipe (flag off) resumes, with the optimizer
    state, a checkpoint saved by the SmolLM3 recipe (flag on): stopped with a clear error
    before the optimizer state is loaded."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    state = {"args": _ckpt_args(skip_embedding_weight_decay=True)}
    wrapped = p.wrap_load_base_checkpoint(lambda load_dir, args, rank0=False: (state, "c", False, "torch"))
    with pytest.raises(RuntimeError, match="saved with skip_embedding_weight_decay=True and this run has"):
        wrapped("dir", SimpleNamespace(), rank0=True)  # flag absent from the run's args: off
    assert wrapped("dir", SimpleNamespace(no_load_optim=True))[0] is state


def test_flag_off_run_is_not_stopped_by_the_wrapper_itself(monkeypatch):
    """With the flag off, the check only compares values: a result it cannot read passes
    through, and an install error is logged, not fatal."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    odd = p.wrap_load_base_checkpoint(lambda load_dir, args, rank0=False: {"args": None})
    assert odd("dir", SimpleNamespace(skip_embedding_weight_decay=False)) == {"args": None}

    def boom():
        raise ImportError("no checkpointing module")

    lines = []
    monkeypatch.setattr(p, "install_resume_check", boom)
    monkeypatch.setattr(p, "log_rank_0", lines.append)
    p.patch_skip_embedding_weight_decay_resume_check(_ctx(skip_embedding_weight_decay=False, load="/ckpt"))
    assert "resume check not installed" in lines[-1]


def test_conditions_do_not_raise_on_a_bad_flag_value():
    """A value that is not a boolean: the conditions return True (they never raise, which
    would abort run_patches for every patch) and the install patch stops the run."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    bad = _ctx(skip_embedding_weight_decay="", load="/ckpt")
    assert p.skip_embedding_wd_condition(bad) is True
    assert p.skip_embedding_wd_uninstall_needed(bad) is True
    assert p.resume_check_needed(bad) is True
    with pytest.raises(SystemExit, match="skip_embedding_weight_decay must be a boolean"):
        p.patch_skip_embedding_weight_decay(bad)


def test_resume_check_install_error_stops_the_run(monkeypatch):
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    def boom():
        raise ImportError("no checkpointing module")

    monkeypatch.setattr(p, "install_resume_check", boom)
    with pytest.raises(SystemExit, match="resume check could not be installed"):
        p.patch_skip_embedding_weight_decay_resume_check(_ctx(load="/ckpt"))


def test_load_base_checkpoint_wrapper_checks_the_saved_args():
    """The wrapper checks the state dict's saved args against the ``args`` argument; a
    release checkpoint, a state dict without args and a turned-off wrapper pass."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    def load(load_dir, args, rank0=False, checkpointing_context=None):
        return state, "ckpt", release, "torch"

    run = SimpleNamespace(skip_embedding_weight_decay=False)
    state, release = {"args": _ckpt_args(skip_embedding_weight_decay=True)}, False
    on = [True]
    wrapped = p.wrap_load_base_checkpoint(load, is_active=lambda: on[0])
    with pytest.raises(RuntimeError, match="saved with skip_embedding_weight_decay=True"):
        wrapped("dir", run, rank0=True)
    with pytest.raises(RuntimeError):
        wrapped(load_dir="dir", args=run)
    release = True
    assert wrapped("dir", run)[0] is state
    release, state = False, {}
    assert wrapped("dir", run)[0] == {}
    state = {"args": _ckpt_args(skip_embedding_weight_decay=True)}
    on[0] = False
    assert wrapped("dir", run)[0] is state
    # A result that is not Megatron's tuple: with the flag on the check cannot run, so it
    # raises (with it off it is passed through: test_flag_off_run_is_not_stopped_...).
    on[0] = True
    odd = p.wrap_load_base_checkpoint(lambda load_dir, args, rank0=False: {"args": None})
    with pytest.raises(RuntimeError, match="returned dict, not"):
        odd("dir", SimpleNamespace(skip_embedding_weight_decay=True))
    # A signature without ``args``: refused at install, not a check that never runs.
    with pytest.raises(RuntimeError, match="no 'args' parameter"):
        p.wrap_load_base_checkpoint(lambda load_dir, run_args, rank0=False: None)


def test_real_load_checkpoint_reads_args_before_the_optimizer():
    """Pinned Megatron: load_checkpoint finds _load_base_checkpoint through its module
    globals (so the wrapper applies), the wrapped name has the (load_dir, args, rank0)
    arguments and returns (state_dict, checkpoint_name, release, ckpt_type), and the
    optimizer state is loaded after that call."""
    require_megatron()
    import inspect

    import megatron.training.checkpointing as ck

    assert "_load_base_checkpoint" in _global_names(ck.load_checkpoint)
    assert list(inspect.signature(ck._load_base_checkpoint).parameters)[:3] == ["load_dir", "args", "rank0"]
    src = inspect.getsource(ck.load_checkpoint)
    first = src.index("state_dict, checkpoint_name, release, ckpt_type = _load_base_checkpoint(")
    assert first < src.index("optimizer.load_state_dict")


def test_resume_check_patch_through_patch_runner(patched_megatron):
    """The resume-check patch runs for a run that loads a checkpoint, wraps Megatron's
    _load_base_checkpoint once, and its after_train patch restores it."""
    import megatron.training.checkpointing as ck

    from primus.core.patches.patch_runner import run_patches

    m = patched_megatron
    rid = _PATCH_ID + ".resume_check"
    ids = [rid, rid + ".uninstall"]
    before = ck._load_base_checkpoint
    assert m.patch.resume_check_needed(_ctx()) is False
    assert run_patches(backend="megatron", phase="before_train", enabled_ids=ids, extra=_ctx().extra) == 0
    assert ck._load_base_checkpoint is before
    extra = _ctx(load="/ckpt").extra
    assert m.patch.resume_check_needed(_ctx(pretrained_checkpoint="/p")) is True
    for _ in range(2):
        assert run_patches(backend="megatron", phase="before_train", enabled_ids=ids, extra=extra) == 1
    assert ck._load_base_checkpoint is not before and ck._load_base_checkpoint.__wrapped__ is before
    assert run_patches(backend="megatron", phase="after_train", enabled_ids=ids, extra=extra) == 1
    assert ck._load_base_checkpoint is before


def test_fsdp2_flags_have_one_definition():
    from primus.backends.megatron.core.optimizer import fsdp2_optimizer_flags as owner
    from primus.backends.megatron.patches.args import rocm_arg_validation as v

    # Each flag of the one list is reported by the helper this patch uses.
    for flag in v._FSDP2_OPTIMIZER_FLAGS:
        on = SimpleNamespace(use_torch_fsdp2=True, bf16=True, **{flag: True})
        assert owner.active_fsdp2_optimizers(on) == [flag]
    assert "use_fsdp2_fp32_param_optimizer" in v._FSDP2_OPTIMIZER_FLAGS
    # Without torch FSDP2, or without bf16, none of them is active.
    for gate in ({"use_torch_fsdp2": False, "bf16": True}, {"use_torch_fsdp2": True, "bf16": False}):
        args = SimpleNamespace(**gate, **{f: True for f in v._FSDP2_OPTIMIZER_FLAGS})
        assert owner.active_fsdp2_optimizers(args) == []


def test_param_groups_wrapper_checks_the_signature_at_install():
    """A Megatron whose _get_param_groups takes other parameters is refused at install
    (before_train), not with a TypeError in the optimizer build."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as patch

    with pytest.raises(RuntimeError, match="does not start with"):
        patch.wrap(lambda model_chunks, cfg, overrides: None)
    with pytest.raises(RuntimeError, match="does not start with"):
        patch.wrap(lambda *a, **k: None)
    patch.wrap(lambda model_chunks, config, config_overrides, extra=None: None)


def test_param_groups_wrapper_passes_on_every_argument(monkeypatch):
    """A later parameter of _get_param_groups (here ``extra``) is passed on, by keyword or
    by position, inside and outside a setup scope; only config_overrides is replaced."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as patch

    seen = []

    def original(model_chunks, config, config_overrides, extra=None):
        seen.append((config_overrides, extra))
        return []

    monkeypatch.setattr(patch, "apply_and_check", lambda call, chunks, cfg, ov: call(("no-wd", ov)))
    w = patch.wrap(original, args_getter=lambda: SimpleNamespace(skip_embedding_weight_decay=True))
    w([], None, None, extra=1)  # outside a scope: unchanged
    monkeypatch.setattr(patch, "_scopes", [0])
    w([], None, "ov", extra=2)
    w([], None, "ov", 3)
    assert seen == [(None, 1), (("no-wd", "ov"), 2), (("no-wd", "ov"), 3)]
    assert patch._scopes == [2]


def test_resume_check_skips_a_checkpoint_saved_without_the_optimizer_state():
    """A checkpoint saved with no_save_optim has no optimizer state to load, so a flag
    change is not an error."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    ckpt = _ckpt_args(skip_embedding_weight_decay=True, no_save_optim=True)
    p.check_resume_flag(SimpleNamespace(skip_embedding_weight_decay=False), ckpt)
    with pytest.raises(RuntimeError, match="optimizer state cannot be loaded"):
        p.check_resume_flag(
            SimpleNamespace(skip_embedding_weight_decay=False), _ckpt_args(skip_embedding_weight_decay=True)
        )


def test_flag_off_run_ignores_a_saved_value_that_is_not_a_boolean():
    """A generic (flag-off) run is not stopped by an odd saved value; a flag-on run is."""
    from primus.backends.megatron.patches import embedding_weight_decay_patches as patch

    saved = SimpleNamespace(skip_embedding_weight_decay="maybe")
    patch.check_resume_flag(SimpleNamespace(skip_embedding_weight_decay=False), saved)
    patch.check_resume_flag(SimpleNamespace(), saved)
    with pytest.raises(RuntimeError, match="cannot confirm"):
        patch.check_resume_flag(SimpleNamespace(skip_embedding_weight_decay=True), saved)


def test_check_and_install_with_no_extra_falls_back_to_megatron_args(monkeypatch):
    from primus.backends.megatron.patches import embedding_weight_decay_patches as patch

    monkeypatch.setattr(patch, "_megatron_args", lambda: None)
    ctx = SimpleNamespace(extra=None, config=None, params={"skip_embedding_weight_decay": True})
    monkeypatch.setattr(patch, "skip_embedding_wd_enabled", lambda c: True)
    with pytest.raises(RuntimeError, match="no Megatron args"):
        patch.check_and_install(ctx)


def test_install_records_by_identity_and_reinstall_does_not_revive_a_turned_off_wrapper():
    """_install/_uninstall on a plain module: restore when ours is the attribute; turn
    ours off (not unlink) under a later wrapper; a new install wraps the outer function
    and the old, turned-off wrapper stays off."""
    import types

    from primus.backends.megatron.patches import embedding_weight_decay_patches as p

    mod = types.ModuleType("fake_mod")
    mod.fn = lambda x: x

    def make(original, is_active):
        def wrapper(x):
            return original(x) + 1 if is_active() else original(x)

        return wrapper

    keys = ("fake_mod.fn",)
    base = mod.fn
    try:
        assert p._install(mod, "fn", make) is True and mod.fn(1) == 2
        assert p._install(mod, "fn", make) is False  # once
        assert p._uninstall(keys) == (["fake_mod.fn"], []) and mod.fn is base
        assert p._install(mod, "fn", make) is True
        ours = mod.fn
        mod.fn = lambda x: ours(x) * 10  # a later patch wraps over ours
        assert mod.fn(1) == 20
        assert p._uninstall(keys) == ([], ["fake_mod.fn"]) and mod.fn(1) == 10  # ours off
        assert p._install(mod, "fn", make) is True and mod.fn(1) == 11  # one new wrapper
        assert ours(1) == 1  # the old one stays off
        assert p._uninstall(keys) == (["fake_mod.fn"], []) and p._uninstall(keys) == ([], [])
    finally:
        p._installed.pop("fake_mod.fn", None)
