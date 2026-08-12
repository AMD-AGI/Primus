###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 context-parallel plan.

WHY THIS TEST EXISTS:
  Every failure mode this plan can have is silent. Nothing about a wrong plan raises:
  the model still runs, the loss still falls, and only the numbers are wrong. So these
  tests assert the *structure* of the plan rather than any behaviour.

  Three specific regressions are pinned here, all of which were live during development:

  1. **Wrapping ``forward``.** diffusers resolves the root plan's entries against the
     forward signature, so wrapping ``forward`` with ``*args, **kwargs`` hides
     ``encoder_hidden_states`` and ``indicator`` and leaves them unsplit. The first
     implementation did exactly this. ``test_root_plan_names_resolve_in_forward_signature``
     is the guard.
  2. **Splitting ``segment_ids``.** The attention mask must stay full length, because
     after the Ulysses all-to-all each rank holds the whole sequence with a subset of
     heads. Its absence from the plan is load-bearing, so it is asserted explicitly.
  3. **CP combined with the var-len flash processor.** That processor bypasses
     ``dispatch_attention_fn``, so the all-to-all never runs and each rank attends only
     within its own shard -- without raising. The guard must raise instead.

  No GPU and no distributed init is needed; the plan is a class attribute and the model
  is instantiated tiny, on CPU, purely to resolve module names.
"""

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

# Small but structurally faithful: 3-axis MRoPE, heads divisible by a CP degree of 2.
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
    """Install onto the real class, then restore it.

    ``install`` mutates class-level state, so without this a failure in one test would
    leak into the next and the suite would stop meaning anything.
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
        if name == "_cp_plan" and value is None:
            model_cls._cp_plan = None
        else:
            setattr(model_cls, name, value)


@pytest.fixture
def tiny_model():
    return Ideogram4Transformer2DModel(**TINY_MODEL_KWARGS)


def test_install_satisfies_the_upstream_eligibility_gate(installed_plan):
    """Verbatim the check AutoModel makes before it will enable CP."""
    assert getattr(installed_plan, "_cp_plan", None) is not None
    assert hasattr(installed_plan, "enable_parallelism")


def test_install_is_idempotent(installed_plan):
    assert cp.install() is False


def test_root_plan_splits_exactly_the_per_token_inputs(installed_plan):
    """Ideogram-4 is single-stream, so these three split together -- and no others.

    The root forward derives the indicator masks and applies them to the other two
    before the first block, so a subset would mismatch shapes.
    """
    assert set(installed_plan._cp_plan[""]) == {
        "hidden_states",
        "encoder_hidden_states",
        "indicator",
    }


@pytest.mark.parametrize("name", ["segment_ids", "position_ids", "timestep"])
def test_load_bearing_omissions_stay_out_of_the_plan(installed_plan, name):
    """These absences are decisions, not oversights.

    ``segment_ids`` must stay whole so the mask is full length after the all-to-all;
    ``position_ids`` stays whole because MRoPE's *output* is split instead; ``timestep``
    is per-sample and broadcasts.
    """
    assert name not in installed_plan._cp_plan[""]


def test_root_plan_names_resolve_in_forward_signature(installed_plan):
    """Regression: never wrap ``forward`` on a CP-planned model.

    diffusers matches the root plan's keys against the forward signature, so a
    ``*args, **kwargs`` wrapper silently leaves those inputs unsplit.
    """
    import inspect

    params = set(inspect.signature(installed_plan.forward).parameters)
    missing = set(installed_plan._cp_plan[""]) - params
    assert not missing, f"plan names not visible in forward signature: {sorted(missing)}"


def test_mrope_outputs_are_split_not_its_inputs(installed_plan):
    """RoPE is applied to q/k before the all-to-all, so cos/sin must end up local."""
    rope_plan = installed_plan._cp_plan["rotary_emb"]
    assert set(rope_plan) == {0, 1}, "MRoPE returns (cos, sin); both outputs must be split"
    for entry in rope_plan.values():
        assert isinstance(entry, ContextParallelInput)
        assert entry.split_output is True
        assert entry.split_dim == 1


def test_output_is_gathered_at_final_layer(installed_plan):
    entry = installed_plan._cp_plan["final_layer"]
    assert isinstance(entry, ContextParallelOutput)
    assert entry.gather_dim == 1


def test_plan_module_names_resolve_on_a_real_model(installed_plan, tiny_model):
    """A typo in a module key would make diffusers attach the hook nowhere, silently."""
    names = dict(tiny_model.named_modules())
    for key in installed_plan._cp_plan:
        if key == "":
            continue
        assert key in names, f"plan key {key!r} matches no submodule"


def test_mrope_output_rank_matches_expected_dims(installed_plan, tiny_model):
    """``expected_dims`` is the only shape assertion diffusers makes; keep it honest."""
    cos, sin = tiny_model.rotary_emb(torch.zeros(2, 8, 3, dtype=torch.long))
    expected = installed_plan._cp_plan["rotary_emb"][0].expected_dims
    assert cos.dim() == expected
    assert sin.dim() == expected


def test_cp_refuses_the_varlen_processor(installed_plan, tiny_model, monkeypatch):
    """CP + var-len is silent corruption, so enabling CP must raise instead."""
    monkeypatch.setenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", "1")
    with pytest.raises(ValueError, match="PRIMUS_IDEOGRAM_VARLEN_ATTN"):
        tiny_model.enable_parallelism(config=None)


def test_cp_allows_the_stock_attention_path(installed_plan, tiny_model, monkeypatch):
    """The guard must not fire on the default configuration.

    ``config=None`` gets past our check and then fails inside diffusers; anything other
    than our own ValueError means the guard let it through, which is what matters here.
    """
    monkeypatch.delenv("PRIMUS_IDEOGRAM_VARLEN_ATTN", raising=False)
    try:
        tiny_model.enable_parallelism(config=None)
    except ValueError as exc:
        assert "PRIMUS_IDEOGRAM_VARLEN_ATTN" not in str(exc)
    except Exception:
        pass


@pytest.mark.parametrize(
    "timestep, should_raise",
    [
        (torch.rand(2), False),  # per-sample: broadcasts, needs no split
        (torch.rand(2, 24), True),  # per-token: would need splitting, so refuse
    ],
)
def test_timestep_shape_assumption_is_enforced(timestep, should_raise):
    if should_raise:
        with pytest.raises(ValueError, match="per-sample timestep"):
            cp._assert_timestep_is_per_sample(timestep)
    else:
        cp._assert_timestep_is_per_sample(timestep)
