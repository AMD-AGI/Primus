###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 context-parallel guards and sequence padding.

The plan itself needs a diffusers that ships the Ideogram-4 transformer, and its
tests are in ``test_ideogram4_context_parallel.py``. Everything here is the other
half -- the padding, the per-sample timestep guard, the variable-length refusal
and the wiring that puts them in place -- which is plain torch, so it is tested
against stand-in classes and runs wherever torch does.

What is pinned:

  * THE DEGREE BELONGS TO THE MODEL. It is read back from what the original
    enable_parallelism stored, through diffusers' wrapping of a bare
    context-parallel config, and lives on that instance: a second model in the
    same process is not padded.

  * THE GUARDS ARE INSTALLED ONCE. Enabling CP twice on one instance must not
    stack a second hook.

  * THE PADDING keeps the image tokens at the tail and fills the new columns
    with the padding segment and position 0.
"""

import types

import pytest

torch = pytest.importorskip("torch")

from primus.backends.automodel.models.ideogram4 import adapter  # noqa: E402
from primus.backends.automodel.models.ideogram4 import (  # noqa: E402
    context_parallel as cp,
)


@pytest.fixture(autouse=True)
def fresh_log_state(monkeypatch):
    monkeypatch.setattr(cp, "_pad_logged", False)


@pytest.fixture
def no_varlen(monkeypatch):
    """The varlen check reads process-wide state; the wiring tests pin it off."""
    monkeypatch.setattr(cp, "_varlen_attention_in_use", lambda: None)


def stub_model_class():
    """A stand-in with the real entry point's keyword-only signature. Like the
    real one, it stores a parallel config wrapping the context-parallel config."""

    class Model(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.enabled = []

        def enable_parallelism(self, *, config, cp_plan=None):
            self.enabled.append(config)
            self._parallel_config = types.SimpleNamespace(context_parallel_config=config)

        def forward(self, hidden_states, timestep=None, **kwargs):
            return hidden_states

    cp._wrap_enable_parallelism(Model)
    return Model


def cp_config(ulysses=2, ring=1):
    return types.SimpleNamespace(ring_degree=ring, ulysses_degree=ulysses)


class TestEnableParallelismWiring:
    def test_the_original_is_called_with_the_config(self, no_varlen):
        model = stub_model_class()()
        config = cp_config()
        model.enable_parallelism(config=config)
        assert model.enabled == [config]

    def test_no_hook_and_no_padding_before_cp_is_switched_on(self):
        """Installing alone must not add a per-forward cost to runs that never
        enable CP."""
        model = stub_model_class()()
        assert len(model._forward_pre_hooks) == 0
        assert cp.sequence_multiple(model) == 1

    def test_the_degree_is_recorded_on_the_instance(self, no_varlen):
        model = stub_model_class()()
        model.enable_parallelism(config=cp_config(ulysses=2))
        assert cp.sequence_multiple(model) == 2

    def test_another_instance_is_not_padded(self, no_varlen):
        """An EMA copy or an eval pipeline in the same process keeps its own."""
        Model = stub_model_class()
        parallel, other = Model(), Model()
        parallel.enable_parallelism(config=cp_config(ulysses=2))
        assert cp.sequence_multiple(other) == 1
        assert len(other._forward_pre_hooks) == 0

    def test_the_degree_is_read_through_diffusers_wrapping(self, no_varlen):
        """diffusers stores a parallel config around the context-parallel one; a
        degree read only from the top level would come out as 1."""
        model = stub_model_class()()
        model.enable_parallelism(config=cp_config(ulysses=2, ring=1))
        assert isinstance(model._parallel_config, types.SimpleNamespace)
        assert cp.sequence_multiple(model) == 2

    @pytest.mark.parametrize(
        "config, degree",
        [
            (None, 1),
            (cp_config(ulysses=2), 2),
            (cp_config(ulysses=None, ring=None), 1),
            (types.SimpleNamespace(context_parallel_config=cp_config(ulysses=4)), 4),
            (types.SimpleNamespace(context_parallel_config=None), 1),
        ],
    )
    def test_degree_of(self, config, degree):
        assert cp._degree_of(config) == degree

    def test_enabling_twice_registers_one_hook(self, no_varlen):
        model = stub_model_class()()
        model.enable_parallelism(config=cp_config())
        model.enable_parallelism(config=cp_config())
        assert len(model._forward_pre_hooks) == 1
        assert len(model.enabled) == 2

    def test_wrapping_twice_wraps_once(self, no_varlen):
        Model = stub_model_class()
        wrapped = Model.enable_parallelism
        cp._wrap_enable_parallelism(Model)
        assert Model.enable_parallelism is wrapped

    def test_a_refused_enable_leaves_the_model_untouched(self, no_varlen):
        """Recorded only after the original accepts the config, so a refused
        enable does not pad a run that is not actually split."""

        class Refusing(torch.nn.Module):
            def enable_parallelism(self, *, config, cp_plan=None):
                raise ValueError("unsupported")

        cp._wrap_enable_parallelism(Refusing)
        model = Refusing()
        with pytest.raises(ValueError, match="unsupported"):
            model.enable_parallelism(config=cp_config())
        assert cp.sequence_multiple(model) == 1
        assert len(model._forward_pre_hooks) == 0

    def test_the_hook_refuses_a_per_token_timestep_on_forward(self, no_varlen):
        """The point of the arrangement: a per-token timestep is caught on the
        forward it would have been mis-sliced on."""
        model = stub_model_class()()
        model.enable_parallelism(config=cp_config())
        with pytest.raises(ValueError, match="per-sample timestep"):
            model(torch.rand(2, 8, 4), timestep=torch.rand(2, 8))

    def test_the_hook_passes_a_per_sample_timestep(self, no_varlen):
        model = stub_model_class()()
        model.enable_parallelism(config=cp_config())
        assert model(torch.rand(2, 8, 4), timestep=torch.rand(2)).shape == (2, 8, 4)


class TestVarlenRefusal:
    def test_the_stock_attention_path_reaches_the_original(self, no_varlen):
        model = stub_model_class()()
        model.enable_parallelism(config=cp_config())
        assert len(model.enabled) == 1

    def test_refused_before_the_original_runs(self, monkeypatch):
        monkeypatch.setattr(cp, "_varlen_attention_in_use", lambda: "requested")
        model = stub_model_class()()
        with pytest.raises(ValueError, match="variable-length"):
            model.enable_parallelism(config=cp_config())
        assert model.enabled == []

    def test_a_requested_processor_says_to_turn_the_setting_off(self, monkeypatch):
        monkeypatch.setattr(cp, "_varlen_attention_in_use", lambda: "requested")
        with pytest.raises(ValueError, match="Turn primus_ideogram4.varlen_attention off") as excinfo:
            cp._refuse_varlen_combination()
        assert "Restart" not in str(excinfo.value)

    def test_an_earlier_install_says_to_restart(self, monkeypatch):
        """The setting is already off here, so telling the user to turn it off
        would send them round in a circle."""
        monkeypatch.setattr(cp, "_varlen_attention_in_use", lambda: "installed")
        with pytest.raises(ValueError, match="Restart with the setting off") as excinfo:
            cp._refuse_varlen_combination()
        assert "Turn primus_ideogram4.varlen_attention off" not in str(excinfo.value)

    def test_the_setting_is_reported_as_requested(self, set_option):
        set_option("primus_ideogram4.varlen_attention", True)
        assert cp._varlen_attention_in_use() == "requested"


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

    def test_the_hook_finds_the_timestep_by_keyword_or_position(self):
        module = torch.nn.Module()
        with pytest.raises(ValueError, match="per-sample timestep"):
            cp._timestep_pre_hook(module, (), {"timestep": torch.rand(2, 24)})
        with pytest.raises(ValueError, match="per-sample timestep"):
            cp._timestep_pre_hook(module, (torch.rand(2, 8, 4), torch.rand(2, 24)), {})
        assert cp._timestep_pre_hook(module, (), {"timestep": torch.rand(2)}) is None
        assert cp._timestep_pre_hook(module, (torch.rand(2, 8, 4), torch.rand(2)), {}) is None


def _packed(length, max_text, grid=(2, 3)):
    """A prepared batch whose tail is the image region."""
    hidden = torch.zeros(2, length, 4)
    hidden[:, max_text:] = 1
    return {
        "hidden_states": hidden,
        "encoder_hidden_states": torch.zeros(2, length, 3),
        "position_ids": torch.zeros(2, length, 3, dtype=torch.long),
        "segment_ids": torch.zeros(2, length, dtype=torch.long),
        "indicator": torch.zeros(2, length, dtype=torch.long),
        "_max_text": max_text,
        "_grid_h": grid[0],
        "_grid_w": grid[1],
    }


class TestSequencePadding:
    """The real cache packs 255 text tokens with a 16x16 grid, which is 511.
    A degree of 2 cannot shard that, and diffusers asserts in the forward."""

    def test_an_indivisible_length_is_left_padded(self):
        """The new column must carry the padding segment ID and position 0. The batch
        starts as ones, which differ from both fills, so the new column cannot pass
        by accident; filling with 1, the real tokens' segment, fails here."""
        batch = _packed(11, 5)
        batch["segment_ids"] = torch.ones_like(batch["segment_ids"])
        batch["position_ids"] = torch.ones_like(batch["position_ids"])
        out = cp.pad_prepared_inputs(batch, 2)
        assert out["_max_text"] == 6
        assert out["hidden_states"].shape[1] == 12
        assert torch.all(out["hidden_states"][:, :1] == 0)
        assert torch.all(out["hidden_states"][:, -6:] == 1)
        assert out["indicator"].shape[1] == 12
        assert torch.all(out["indicator"][:, 0] == 0)
        assert torch.all(out["segment_ids"][:, 0] == adapter._layout_constants()[3])
        assert torch.all(out["position_ids"][:, 0] == 0)
        assert torch.equal(out["segment_ids"][:, 1:], batch["segment_ids"])
        assert torch.equal(out["position_ids"][:, 1:], batch["position_ids"])

    def test_a_divisible_length_is_unchanged(self):
        out = cp.pad_prepared_inputs(_packed(10, 4), 2)
        assert out["_max_text"] == 4
        assert out["hidden_states"].shape[1] == 10

    def test_padding_is_off_when_the_degree_is_one(self):
        out = cp.pad_prepared_inputs(_packed(11, 5), 1)
        assert out["hidden_states"].shape[1] == 11


class TestAdapterForwardPadding:
    """The padding runs in the adapter's forward, which is handed the model, so
    each model is padded to its own degree."""

    @pytest.fixture
    def adapter_cls(self):
        class Adapter:
            def forward(self, model, inputs):
                self.seen = inputs
                return inputs["hidden_states"].shape[1]

        cp._install_input_padding(Adapter)
        return Adapter

    def test_a_model_with_cp_is_padded(self, adapter_cls):
        model = torch.nn.Module()
        setattr(model, cp._MULTIPLE_ATTR, 2)
        assert adapter_cls().forward(model, _packed(11, 5)) == 12

    def test_a_model_without_cp_is_not(self, adapter_cls):
        assert adapter_cls().forward(torch.nn.Module(), _packed(11, 5)) == 11

    def test_the_degree_is_found_through_a_wrapper(self, adapter_cls):
        inner = torch.nn.Module()
        setattr(inner, cp._MULTIPLE_ATTR, 2)
        wrapper = torch.nn.Module()
        wrapper.module = inner
        assert adapter_cls().forward(wrapper, _packed(11, 5)) == 12

    def test_installing_twice_wraps_once(self, adapter_cls):
        wrapped = adapter_cls.forward
        cp._install_input_padding(adapter_cls)
        assert adapter_cls.forward is wrapped


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
