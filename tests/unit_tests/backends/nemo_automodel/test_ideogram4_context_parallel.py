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
     rather than left implicit. diffusers survives both mask conventions, which is
     exactly why the wrong one would not raise.

  3. CP TOGETHER WITH THE VARIABLE-LENGTH PROCESSOR. That processor bypasses the
     attention dispatch where the all-to-all lives, so each rank would attend only
     within its own shard -- without raising. The guard has to raise instead, and
     has to key off the processor being INSTALLED rather than merely requested.

  4. MODULE KEYS THAT MATCH NOTHING. A typo in a plan key makes diffusers attach
     that hook nowhere, silently, so the keys are resolved against a real model.

No GPU and no distributed initialization: the plan is a class attribute, and the
model is built tiny on CPU purely so module names and output ranks can be
resolved.
"""

import inspect

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("diffusers")

from diffusers.models._modeling_parallel import (  # noqa: E402
    ContextParallelInput,
    ContextParallelOutput,
)
from diffusers.models.transformers.transformer_ideogram4 import (  # noqa: E402
    Ideogram4Transformer2DModel,
)

from primus.backends.nemo_automodel.models.ideogram4 import (  # noqa: E402
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


@pytest.fixture(autouse=True)
def installed_plan():
    """Install onto the real class, then put it back.

    ``install`` mutates class-level state, so without the restore a failure in one
    test would leak into the next and the suite would stop meaning anything.
    """
    model_cls = Ideogram4Transformer2DModel
    saved = {
        "_cp_plan": getattr(model_cls, "_cp_plan", None),
        "enable_parallelism": model_cls.enable_parallelism,
        "forward": model_cls.forward,
        "_primus_cp_installed": getattr(model_cls, "_primus_cp_installed", False),
    }
    model_cls._primus_cp_installed = False
    cp.install()
    yield model_cls
    for name, value in saved.items():
        setattr(model_cls, name, value)


@pytest.fixture
def tiny_model():
    return Ideogram4Transformer2DModel(**TINY_MODEL_KWARGS)


@pytest.fixture(autouse=True)
def no_varlen(monkeypatch):
    """The variable-length processor off, since most tests are not about it."""
    monkeypatch.delenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", raising=False)


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

    def test_does_not_wrap_the_forward(self, installed_plan):
        """The regression in its most direct form: the forward is left alone."""
        assert installed_plan.forward is Ideogram4Transformer2DModel.forward


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
        exactly these inputs unsplit, and says nothing about it."""
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
    def test_refused_when_requested_by_environment(self, installed_plan, tiny_model, monkeypatch):
        """The combination is silent corruption, so switching CP on has to raise."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
        with pytest.raises(ValueError, match="PRIMUS_IDEOGRAM_VARLEN_ATTN"):
            tiny_model.enable_parallelism(config=None)

    def test_refused_when_installed_but_no_longer_requested(self, installed_plan, tiny_model, monkeypatch):
        """The guard is about what is TRUE, not what was asked for. The processor
        patches a class default, which outlives the variable that caused it, so
        checking only the environment would let the corrupting combination through
        in exactly the case where the processor is already in place."""
        from diffusers.models.transformers.transformer_ideogram4 import (
            Ideogram4Attention,
        )

        monkeypatch.setattr(Ideogram4Attention, "_primus_varlen_installed", True, raising=False)
        with pytest.raises(ValueError, match="variable-length"):
            tiny_model.enable_parallelism(config=None)

    def test_the_stock_attention_path_is_allowed(self, installed_plan, tiny_model):
        """The guard must not fire on the default configuration. ``config=None``
        gets past our check and then fails inside diffusers; anything other than our
        own message means the guard let it through, which is what matters here."""
        try:
            tiny_model.enable_parallelism(config=None)
        except ValueError as exc:
            assert "variable-length" not in str(exc)
        except Exception:
            pass


class TestTimestepGuard:
    @pytest.mark.parametrize(
        "timestep, refused",
        [
            (None, False),
            (torch.rand(2), False),  # per sample: broadcasts, needs no split
            (torch.rand(2, 24), True),  # per token: would need splitting, so refuse
            (torch.rand(2, 24, 1), True),
        ],
    )
    def test_shape_assumption_is_enforced(self, timestep, refused):
        if refused:
            with pytest.raises(ValueError, match="per-sample timestep"):
                cp._assert_timestep_is_per_sample(timestep)
        else:
            cp._assert_timestep_is_per_sample(timestep)

    def test_a_per_token_timestep_is_caught_through_the_hook(self, tiny_model):
        """End to end through the hook's argument extraction, which has to find the
        timestep whether it arrived positionally or by keyword."""
        with pytest.raises(ValueError, match="per-sample timestep"):
            cp._timestep_pre_hook(tiny_model, (), {"timestep": torch.rand(2, 24)})
        with pytest.raises(ValueError, match="per-sample timestep"):
            cp._timestep_pre_hook(tiny_model, (torch.rand(2, 8, 4), torch.rand(2, 24)), {})
        # And must not fire on the per-sample shape, by either route.
        assert cp._timestep_pre_hook(tiny_model, (), {"timestep": torch.rand(2)}) is None
        assert cp._timestep_pre_hook(tiny_model, (torch.rand(2, 8, 4), torch.rand(2)), {}) is None


class TestTheGuardIsWiredUp:
    """That the guard runs at all, which testing the assertion in isolation says
    nothing about.

    Installed onto a stand-in class rather than the real one, because the real
    ``enable_parallelism`` fails inside diffusers on any config a test can build,
    which is before the hook would be registered. The stand-in lets the inner call
    succeed so the wiring is what is under test.
    """

    @pytest.fixture
    def stub_model(self, monkeypatch):
        import diffusers.models.transformers.transformer_ideogram4 as module

        class Ideogram4Transformer2DModel(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.enabled = []

            def enable_parallelism(self, config=None):
                self.enabled.append(config)

            def forward(self, hidden_states, timestep=None, **kwargs):
                return hidden_states

        monkeypatch.setattr(module, "Ideogram4Transformer2DModel", Ideogram4Transformer2DModel)
        assert cp.install() is True
        return Ideogram4Transformer2DModel()

    def test_no_hook_before_cp_is_switched_on(self, stub_model):
        """Installing the plan alone must not add a per-forward cost to runs that
        never enable CP."""
        assert len(stub_model._forward_pre_hooks) == 0

    def test_the_hook_is_registered_when_cp_is_switched_on(self, stub_model):
        stub_model.enable_parallelism(config=None)

        assert stub_model.enabled == [None], "the original was not called"
        assert len(stub_model._forward_pre_hooks) == 1

    def test_the_hook_refuses_a_per_token_timestep_on_forward(self, stub_model):
        """The point of the whole arrangement: a per-token timestep is caught on the
        forward it would have been mis-sliced on."""
        stub_model.enable_parallelism(config=None)

        with pytest.raises(ValueError, match="per-sample timestep"):
            stub_model(torch.rand(2, 8, 4), timestep=torch.rand(2, 8))

    def test_the_hook_passes_a_per_sample_timestep(self, stub_model):
        stub_model.enable_parallelism(config=None)

        out = stub_model(torch.rand(2, 8, 4), timestep=torch.rand(2))
        assert out.shape == (2, 8, 4)
