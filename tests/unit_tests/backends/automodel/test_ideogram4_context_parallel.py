###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 context-parallel plan.

WHY THESE ARE STRUCTURAL TESTS. Every way this plan can be wrong is silent. A
wrong plan does not raise: the model runs, the loss falls, and only the numbers
are wrong. There is no output to compare against without a multi-rank run, so
these assert the SHAPE OF THE PLAN and the guards around it instead, which is
cheap and catches the regressions that actually happen.

Four of them are pinned here specifically:

  1. WRAPPING THE FORWARD. diffusers resolves the root plan's entries against the
     forward SIGNATURE, so wrapping it with ``*args, **kwargs`` hides
     ``encoder_hidden_states`` and ``indicator`` and leaves them unsplit. This is
     an easy thing to reach for -- it is how the guard hook would most naturally
     be added -- and it turns the guard into the class of bug it guards against.
     ``test_root_plan_names_resolve_in_forward_signature`` is the check.

  2. SPLITTING segment_ids. The attention mask has to stay full length, because
     after the Ulysses exchange each rank holds the whole sequence with a subset
     of the heads. The absence of that entry is load-bearing, so it is asserted
     rather than left implicit. With diffusers 0.39 the wrong convention fails
     with a shape error, but that is diffusers' current behaviour rather than a
     guarantee, so the absence is asserted here.

  3. CP TOGETHER WITH THE VARIABLE-LENGTH PROCESSOR. That processor bypasses the
     attention dispatch where the all-to-all lives, so each rank would attend only
     within its own shard -- without raising. The guard has to raise instead, and
     has to key off the processor being INSTALLED rather than merely requested.

  4. MODULE KEYS THAT MATCH NOTHING. A typo in a plan key makes diffusers attach
     that hook nowhere, silently, so the keys are resolved against a real model.

No GPU and no distributed initialization: the plan is a class attribute, and the
model is built tiny on CPU purely so module names and output ranks can be
resolved. The padding, the timestep guard and their wiring need no diffusers and
are tested in ``test_ideogram4_cp_guards.py``.
"""

import inspect
import types

import pytest

torch = pytest.importorskip("torch")
# The submodules, not the package: an older diffusers (CI's image ships 0.33.1)
# imports fine but has neither, and a module-scope ImportError stops collection.
pytest.importorskip("diffusers.models._modeling_parallel")
pytest.importorskip("diffusers.models.transformers.transformer_ideogram4")

from diffusers.models._modeling_parallel import (  # noqa: E402
    ContextParallelInput,
    ContextParallelOutput,
)
from diffusers.models.transformers.transformer_ideogram4 import (  # noqa: E402
    Ideogram4Transformer2DModel,
)

from primus.backends.automodel.models.ideogram4 import adapter  # noqa: E402
from primus.backends.automodel.models.ideogram4 import (  # noqa: E402
    context_parallel as cp,
)

# Small but structurally faithful: a three-axis rotary section, and a head count
# divisible by a context-parallel degree of two.
TINY_MODEL_KWARGS = dict(
    in_channels=128,
    num_layers=1,
    attention_head_dim=32,
    num_attention_heads=4,
    intermediate_size=64,
    adaln_dim=32,
    llm_features_dim=16,
    mrope_section=(4, 2, 2),
)


_MISSING = object()

# Class attributes install() sets or wraps, on the model and on the cached adapter.
_MODEL_ATTRS = (
    "_cp_plan",
    "enable_parallelism",
    "forward",
    "_primus_cp_installed",
    "_primus_cp_enable_wrapped",
)
_ADAPTER_ATTRS = ("forward", "_primus_cp_pad_wrapped")


def _snapshot(cls, names):
    return {name: cls.__dict__.get(name, _MISSING) for name in names}


def _restore(cls, saved):
    for name, value in saved.items():
        if value is _MISSING:
            if name in cls.__dict__:
                delattr(cls, name)
        else:
            setattr(cls, name, value)


def _clear(cls, names):
    for name in names:
        if name in cls.__dict__:
            delattr(cls, name)


@pytest.fixture
def pristine():
    """The two classes as they were before any install, restored afterwards.

    ``install`` mutates class-level state on the model and on the cached adapter
    class, so without the restore one test's install would leak into every later
    test in the session, adapter tests included.
    """
    model_cls = Ideogram4Transformer2DModel
    adapter_cls = adapter.get_ideogram4_adapter_class()
    saved_model = _snapshot(model_cls, _MODEL_ATTRS)
    saved_adapter = _snapshot(adapter_cls, _ADAPTER_ATTRS)
    saved_logged = cp._pad_logged
    _clear(model_cls, ("_primus_cp_installed", "_primus_cp_enable_wrapped"))
    _clear(adapter_cls, ("_primus_cp_pad_wrapped",))
    if saved_model["_cp_plan"] is not _MISSING:
        delattr(model_cls, "_cp_plan")
    cp._pad_logged = False
    yield types.SimpleNamespace(
        model_cls=model_cls,
        adapter_cls=adapter_cls,
        forward=model_cls.forward,
        adapter_forward=adapter_cls.forward,
        enable_parallelism=model_cls.enable_parallelism,
    )
    _restore(model_cls, saved_model)
    _restore(adapter_cls, saved_adapter)
    cp._pad_logged = saved_logged


@pytest.fixture(autouse=True)
def installed_plan(pristine):
    """Install onto the real classes; ``pristine`` puts them back."""
    assert cp.install() is True
    return pristine.model_cls


@pytest.fixture
def tiny_model():
    return Ideogram4Transformer2DModel(**TINY_MODEL_KWARGS)


class TestInstall:
    def test_satisfies_the_upstream_eligibility_gate(self, installed_plan):
        """Verbatim the check upstream makes before it will enable CP. Attaching the
        plan is the whole reason this module exists: everything else CP needs is
        already there."""
        assert getattr(installed_plan, "_cp_plan", None) is not None
        assert hasattr(installed_plan, "enable_parallelism")

    def test_is_idempotent(self, installed_plan):
        """Reports success on the second call rather than failure. Patch
        installation can be reached more than once, and re-wrapping
        ``enable_parallelism`` would stack the guards."""
        before = installed_plan.enable_parallelism
        assert cp.install() is True
        assert installed_plan.enable_parallelism is before

    def test_does_not_wrap_the_forward(self, installed_plan, pristine):
        """The regression in its most direct form: the forward is the one the class
        had before install."""
        assert installed_plan.forward is pristine.forward

    def test_install_wraps_the_entry_point_and_the_adapter(self, installed_plan, pristine):
        """The guards and the padding say nothing unless install did the wrapping,
        and the fixture starts from unwrapped classes, so this one did."""
        assert installed_plan.enable_parallelism is not pristine.enable_parallelism
        assert pristine.adapter_cls.forward is not pristine.adapter_forward
        assert pristine.adapter_cls._primus_cp_pad_wrapped is True


class TestAShippedPlan:
    """Once diffusers declares its own plan, that plan wins -- but the padding and
    the guards hold whichever plan splits the sequence, so they still go in."""

    @pytest.fixture(autouse=True)
    def installed_plan(self, pristine):
        """Overrides the module fixture: the shipped plan has to be in place
        before install runs."""
        pristine.model_cls._cp_plan = {"": {}}
        return pristine

    def test_the_shipped_plan_is_kept(self, installed_plan):
        shipped = installed_plan.model_cls._cp_plan
        assert cp.install() is True
        assert installed_plan.model_cls._cp_plan is shipped

    def test_the_guards_and_the_padding_are_still_installed(self, installed_plan):
        assert cp.install() is True
        assert installed_plan.model_cls.enable_parallelism is not installed_plan.enable_parallelism
        assert installed_plan.adapter_cls.forward is not installed_plan.adapter_forward

    def test_the_varlen_combination_is_still_refused(self, installed_plan, tiny_model, set_option):
        cp.install()
        set_option("primus_ideogram4.varlen_attention", True)
        with pytest.raises(ValueError, match="variable-length"):
            tiny_model.enable_parallelism(config=None)

    def test_it_warns(self, installed_plan, caplog):
        with caplog.at_level("WARNING"):
            cp.install()
        assert any("ships a context-parallel plan" in r.getMessage() for r in caplog.records)


class TestPlanStructure:
    def test_root_splits_exactly_the_per_token_inputs(self, installed_plan):
        """This model is single-stream, so these three split together -- and no
        others. The root forward derives the indicator masks and applies them to the
        other two before the first block, so any subset would mismatch shapes."""
        assert set(installed_plan._cp_plan[""]) == {
            "hidden_states",
            "encoder_hidden_states",
            "indicator",
        }

    @pytest.mark.parametrize("name", ["segment_ids", "position_ids", "timestep"])
    def test_load_bearing_omissions_stay_out(self, installed_plan, name):
        """These absences are decisions, not oversights, and each has its own
        reason: the segment ids must stay whole so the mask is full length after the
        exchange, the position ids stay whole because the rotary OUTPUT is split
        instead, and the timestep is per sample and broadcasts."""
        assert name not in installed_plan._cp_plan[""]

    def test_root_plan_names_resolve_in_forward_signature(self, installed_plan):
        """The reason the forward must not be wrapped. diffusers matches the root
        plan's keys against the signature, so a ``*args, **kwargs`` wrapper leaves
        exactly these inputs unsplit, and says nothing about it. Resolved the way
        diffusers resolves them, following ``__wrapped__``: the real forward is
        itself decorated, and its outer signature is ``*args, **kwargs``. The
        identity check above is what catches a wrapper of ours."""
        params = set(inspect.signature(installed_plan.forward).parameters)
        missing = set(installed_plan._cp_plan[""]) - params
        assert not missing, f"plan names not visible in the forward signature: {sorted(missing)}"

    def test_rotary_outputs_are_split_not_its_inputs(self, installed_plan):
        """Rotary embeddings are applied before the all-to-all, so the cosines and
        sines have to end up local and aligned with the local tokens."""
        rope_plan = installed_plan._cp_plan["rotary_emb"]
        assert set(rope_plan) == {0, 1}, "it returns a cosine and a sine; both are split"
        for entry in rope_plan.values():
            assert isinstance(entry, ContextParallelInput)
            assert entry.split_output is True
            assert entry.split_dim == 1

    def test_output_is_gathered_at_the_final_layer(self, installed_plan):
        entry = installed_plan._cp_plan["final_layer"]
        assert isinstance(entry, ContextParallelOutput)
        assert entry.gather_dim == 1

    def test_every_split_is_on_the_sequence_axis(self, installed_plan):
        """One axis is the sequence, and splitting on any other would shard the batch
        or the channels while the mask still describes the sequence."""
        for entry in installed_plan._cp_plan[""].values():
            assert entry.split_dim == 1


class TestPlanResolvesAgainstARealModel:
    def test_module_keys_exist(self, installed_plan, tiny_model):
        """A typo in a module key attaches the hook nowhere, silently."""
        names = dict(tiny_model.named_modules())
        for key in installed_plan._cp_plan:
            if key == "":
                continue
            assert key in names, f"plan key {key!r} matches no submodule"

    def test_rotary_output_rank_matches_expected_dims(self, installed_plan, tiny_model):
        """``expected_dims`` is the only shape assertion diffusers makes, so it is
        worth keeping honest against the real output."""
        cos, sin = tiny_model.rotary_emb(torch.zeros(2, 8, 3, dtype=torch.long))
        expected = installed_plan._cp_plan["rotary_emb"][0].expected_dims
        assert cos.dim() == expected
        assert sin.dim() == expected


class TestRefusesTheVarlenCombination:
    def test_refused_when_requested_by_the_setting(self, installed_plan, tiny_model, set_option):
        """The combination is silent corruption, so switching CP on has to raise."""
        set_option("primus_ideogram4.varlen_attention", True)
        with pytest.raises(ValueError, match="primus_ideogram4.varlen_attention"):
            tiny_model.enable_parallelism(config=None)

    def test_refused_when_installed_but_no_longer_requested(self, installed_plan, tiny_model, monkeypatch):
        """The guard is about what is TRUE, not what was asked for. The processor
        patches a class default, which outlives the setting that caused it, so
        checking only the setting would let the corrupting combination through
        in exactly the case where the processor is already in place."""
        from diffusers.models.transformers.transformer_ideogram4 import (
            Ideogram4Attention,
        )

        monkeypatch.setattr(Ideogram4Attention, "_primus_varlen_installed", True, raising=False)
        with pytest.raises(ValueError, match="Restart with the setting off"):
            tiny_model.enable_parallelism(config=None)

    def test_the_stock_attention_path_reaches_the_original(self, installed_plan, tiny_model, monkeypatch):
        """The guard must not fire on the default configuration: the original entry
        point, replaced here by a recording stub before wrapping, is reached."""
        reached = []

        def original(self, *, config, cp_plan=None):
            reached.append(config)

        # Pinned off: an attention-processor test may have left the processor on
        # the class, which is the refusal's job to notice, not this test's.
        monkeypatch.setattr(cp, "_varlen_attention_in_use", lambda: None)
        monkeypatch.setattr(installed_plan, "enable_parallelism", original)
        monkeypatch.setattr(installed_plan, "_primus_cp_enable_wrapped", False)
        cp._wrap_enable_parallelism(installed_plan)
        tiny_model.enable_parallelism(config=None)
        assert reached == [None]


class TestTimestepGuardOnTheRealModel:
    def test_the_timestep_is_the_second_positional_parameter(self):
        """The hook takes ``args[1]`` as the timestep when it arrives positionally,
        which is only right while the real forward keeps that order."""
        params = list(inspect.signature(Ideogram4Transformer2DModel.forward).parameters)
        assert params[1:3] == ["hidden_states", "timestep"]

    def test_a_per_token_timestep_is_caught_through_the_hook(self, tiny_model):
        """End to end through the hook's argument extraction, which has to find the
        timestep whether it arrived positionally or by keyword."""
        with pytest.raises(ValueError, match="per-sample timestep"):
            cp._timestep_pre_hook(tiny_model, (), {"timestep": torch.rand(2, 24)})
        with pytest.raises(ValueError, match="per-sample timestep"):
            cp._timestep_pre_hook(tiny_model, (torch.rand(2, 8, 4), torch.rand(2, 24)), {})
        assert cp._timestep_pre_hook(tiny_model, (), {"timestep": torch.rand(2)}) is None
        assert cp._timestep_pre_hook(tiny_model, (torch.rand(2, 8, 4), torch.rand(2)), {}) is None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
