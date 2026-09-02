###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the Ideogram-4 real-AC parallelization strategy.

WHAT IS BEING DEFENDED, and why it needs a test at all:

  The strategy exists because AutoModel's default one reaches for submodule
  attributes an Ideogram block does not have, making
  ``fsdp.activation_checkpointing: true`` a silent no-op. Silent is the operative
  word -- the config echo shows the setting as requested, nothing logs a problem,
  and the only evidence is that turning it on changes nothing. A test that
  asserted "checkpointing happened" by measuring memory would need a multi-GPU
  run; asserting it by recording which blocks were wrapped costs nothing and
  catches the same regression.

  The reshard-forwarding obligation is shared with every other strategy in this
  backend and has one shape, so it comes from the shared contract module rather
  than being restated here.

  The stride is Ideogram's own knob, so its parsing is tested here. It is parsed
  STRICTLY, unlike the diagnostic environment helpers: a typo that fell back to
  the default would give a run that asked for a partial stride the full
  checkpointing it was trying to avoid, and the only evidence would be a step time
  nobody had a baseline for.

No GPU and no real AutoModel: the parallelizer is stubbed and the sharding calls
are recorded rather than performed.
"""

import pytest

from tests.unit_tests.backends.nemo_automodel.parallelize._reshard_contract import (
    FakeMesh,
    assert_forwards_reshard,
    assert_root_unit_stays_unsharded,
    assert_sharding_args_are_keyword,
    install_stub_module,
    install_stub_parallelizer,
    model_with_blocks,
)

BLOCK_ATTR = "layers"
MODEL_NAME = "Ideogram4Transformer2DModel"


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    monkeypatch.delenv("PRIMUS_IDEOGRAM_REAL_AC", raising=False)
    monkeypatch.delenv("PRIMUS_IDEOGRAM_AC_EVERY", raising=False)


@pytest.fixture
def strategy(monkeypatch):
    """Install the strategy against a stubbed parallelizer, and hand it back.

    Returns ``(instance, calls)``. The reshard repair is stubbed as installed, so
    its absence warning does not fire in the tests that are not about it.
    """
    monkeypatch.setenv("PRIMUS_IDEOGRAM_REAL_AC", "1")
    module, calls = install_stub_parallelizer(monkeypatch)
    install_stub_module(
        monkeypatch,
        "primus.backends.nemo_automodel.distributed.fsdp2_reshard",
        patch_installed=True,
        applied_reshard_after_forward=True,
    )

    from primus.backends.nemo_automodel.models.ideogram4 import parallelize

    assert parallelize.install() is True
    return module.PARALLELIZATION_STRATEGIES[MODEL_NAME](), calls


# --------------------------------------------------------------------------- #
# Activation checkpointing actually happening                                 #
# --------------------------------------------------------------------------- #
class TestCheckpointingHappens:
    def test_full_ac_wraps_every_block(self, strategy, monkeypatch):
        """The whole reason the strategy exists. Under the default strategy this
        wraps nothing and says nothing about it."""
        instance, _calls = strategy
        wrapped = []
        import nemo_automodel.components.distributed.parallelizer as P

        monkeypatch.setattr(
            P, "checkpoint_wrapper", lambda block, **kw: wrapped.append(block) or ("w", block)
        )

        model = model_with_blocks(BLOCK_ATTR, blocks=list(range(6)))
        instance.parallelize(model=model, device_mesh=FakeMesh(), activation_checkpointing=True)
        assert len(wrapped) == 6
        assert all(isinstance(b, tuple) for b in getattr(model, BLOCK_ATTR))

    def test_it_wraps_before_sharding(self, strategy, monkeypatch):
        """fully_shard indexes parameters, so the module structure has to be final
        first. Wrapping afterwards would shard the unwrapped modules and leave the
        checkpointing outside the sharded units."""
        instance, calls = strategy
        order = []
        import nemo_automodel.components.distributed.parallelizer as P

        monkeypatch.setattr(
            P,
            "checkpoint_wrapper",
            lambda block, **kw: order.append("wrap") or ("w", block),
        )
        monkeypatch.setattr(
            P,
            "apply_fsdp2_sharding_recursively",
            lambda *a, **kw: order.append("shard") or calls["sharding"].append((a, kw)),
        )

        instance.parallelize(
            model=model_with_blocks(BLOCK_ATTR, blocks=[1, 2]),
            device_mesh=FakeMesh(),
            activation_checkpointing=True,
        )
        assert order == ["wrap", "wrap", "shard"]

    def test_ac_off_wraps_nothing(self, strategy, monkeypatch):
        instance, _calls = strategy
        wrapped = []
        import nemo_automodel.components.distributed.parallelizer as P

        monkeypatch.setattr(P, "checkpoint_wrapper", lambda b, **kw: wrapped.append(b) or b)

        instance.parallelize(
            model=model_with_blocks(BLOCK_ATTR, blocks=list(range(4))),
            device_mesh=FakeMesh(),
            activation_checkpointing=False,
        )
        assert wrapped == []

    @pytest.mark.parametrize("raw", ["false", "off", "0", "no", "none", ""])
    def test_false_like_strings_wrap_nothing(self, strategy, monkeypatch, raw):
        """These are all non-empty-or-empty strings that a bare truthiness test
        gets wrong in one direction or the other. Delegated to the shared helper,
        but asserted here because this is the strategy a run actually uses."""
        instance, _calls = strategy
        wrapped = []
        import nemo_automodel.components.distributed.parallelizer as P

        monkeypatch.setattr(P, "checkpoint_wrapper", lambda b, **kw: wrapped.append(b) or b)

        instance.parallelize(
            model=model_with_blocks(BLOCK_ATTR, blocks=list(range(4))),
            device_mesh=FakeMesh(),
            activation_checkpointing=raw,
        )
        assert wrapped == []

    def test_selective_does_not_take_the_full_path(self, strategy, monkeypatch):
        """'selective' is a truthy string, so a bare test makes it identical to
        full -- which is the more expensive of the two and looks like it worked."""
        instance, _calls = strategy
        import nemo_automodel.components.distributed.parallelizer as P

        wrapped, selective = [], []
        monkeypatch.setattr(P, "checkpoint_wrapper", lambda b, **kw: wrapped.append(b) or b)
        monkeypatch.setattr(
            P,
            "apply_selective_checkpointing_to_layers",
            lambda model, layers, kv, **kw: selective.append(list(layers)),
        )

        instance.parallelize(
            model=model_with_blocks(BLOCK_ATTR, blocks=list(range(4))),
            device_mesh=FakeMesh(),
            activation_checkpointing="selective",
        )
        assert wrapped == [], "selective took the full-AC path"
        assert len(selective) == 1
        assert len(selective[0]) == 4

    def test_a_model_without_the_block_list_warns_rather_than_raising(self, strategy, caplog):
        """A run is already under way by this point, so failing it would be worse
        than proceeding without checkpointing -- but it must not be silent, since
        silence is the bug this strategy fixes."""
        import types

        instance, _calls = strategy
        with caplog.at_level("WARNING"):
            instance.parallelize(
                model=types.SimpleNamespace(something_else=[1]),
                device_mesh=FakeMesh(),
                activation_checkpointing=True,
            )
        assert [r for r in caplog.records if "nothing checkpointed" in r.getMessage()]


# --------------------------------------------------------------------------- #
# The stride                                                                  #
# --------------------------------------------------------------------------- #
class TestStride:
    def test_it_reaches_the_helper(self, monkeypatch):
        instance_calls = {}
        monkeypatch.setenv("PRIMUS_IDEOGRAM_REAL_AC", "1")
        monkeypatch.setenv("PRIMUS_IDEOGRAM_AC_EVERY", "3")
        module, _calls = install_stub_parallelizer(monkeypatch)
        install_stub_module(
            monkeypatch,
            "primus.backends.nemo_automodel.distributed.fsdp2_reshard",
            patch_installed=True,
            applied_reshard_after_forward=True,
        )

        from primus.backends.nemo_automodel.distributed import (
            activation_checkpointing as ac,
        )
        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        monkeypatch.setattr(
            ac,
            "apply",
            lambda *a, **kw: instance_calls.update(kw) or ("full", 0),
        )
        parallelize.install()
        instance = module.PARALLELIZATION_STRATEGIES[MODEL_NAME]()
        instance.parallelize(
            model=model_with_blocks(BLOCK_ATTR, blocks=list(range(9))),
            device_mesh=FakeMesh(),
            activation_checkpointing=True,
        )
        assert instance_calls["stride"] == 3

    def test_it_wraps_every_nth_block_end_to_end(self, monkeypatch):
        """Through the real helper, so the strategy and the helper are checked to
        agree rather than only that a number was passed."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_REAL_AC", "1")
        monkeypatch.setenv("PRIMUS_IDEOGRAM_AC_EVERY", "4")
        module, _calls = install_stub_parallelizer(monkeypatch)
        install_stub_module(
            monkeypatch,
            "primus.backends.nemo_automodel.distributed.fsdp2_reshard",
            patch_installed=True,
            applied_reshard_after_forward=True,
        )

        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        parallelize.install()

        import nemo_automodel.components.distributed.parallelizer as P

        monkeypatch.setattr(P, "checkpoint_wrapper", lambda b, **kw: ("w", b))

        model = model_with_blocks(BLOCK_ATTR, blocks=list(range(12)))
        module.PARALLELIZATION_STRATEGIES[MODEL_NAME]().parallelize(
            model=model, device_mesh=FakeMesh(), activation_checkpointing=True
        )
        wrapped = [i for i, b in enumerate(getattr(model, BLOCK_ATTR)) if isinstance(b, tuple)]
        assert wrapped == [0, 4, 8]

    @pytest.mark.parametrize("raw,expected", [("", 0), ("1", 0), ("2", 2), ("12", 12)])
    def test_parsing(self, monkeypatch, raw, expected):
        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        if raw:
            monkeypatch.setenv("PRIMUS_IDEOGRAM_AC_EVERY", raw)
        assert parallelize.ac_stride() == expected

    def test_one_normalizes_to_zero(self, monkeypatch):
        """Because it means the same thing as not setting it, and carrying it
        through as a special case would put 'every 1' in the log."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_AC_EVERY", "1")
        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        assert parallelize.ac_stride() == 0

    @pytest.mark.parametrize("raw", ["nonsense", "2.5", "two", "0", "-3"])
    def test_a_bad_value_is_refused_rather_than_defaulted(self, monkeypatch, raw):
        """Strictly, unlike the diagnostic helpers. A silent fallback would give a
        run that asked for a partial stride the full checkpointing it was trying to
        avoid, with no evidence but a step time nobody had a baseline for."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_AC_EVERY", raw)
        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        with pytest.raises(ValueError, match="PRIMUS_IDEOGRAM_AC_EVERY"):
            parallelize.ac_stride()

    def test_a_bad_value_fails_at_install_not_mid_run(self, monkeypatch):
        """So the error names the environment variable, instead of surfacing later
        as a parallelization failure with no mention of it."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_REAL_AC", "1")
        monkeypatch.setenv("PRIMUS_IDEOGRAM_AC_EVERY", "nonsense")
        install_stub_parallelizer(monkeypatch)

        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        with pytest.raises(ValueError, match="PRIMUS_IDEOGRAM_AC_EVERY"):
            parallelize.install()


# --------------------------------------------------------------------------- #
# The shared contract                                                         #
# --------------------------------------------------------------------------- #
class TestResharadContract:
    @pytest.mark.parametrize("value", [False, True, None])
    def test_reshard_after_forward_is_forwarded(self, strategy, value):
        """None matters as much as the others: it is the 'user said nothing, keep
        the heuristic' case, and coercing it to a bool changes the default for
        everyone."""
        instance, calls = strategy
        assert_forwards_reshard(instance, calls, BLOCK_ATTR, value)

    def test_the_root_unit_stays_unsharded(self, strategy):
        instance, calls = strategy
        assert_root_unit_stays_unsharded(instance, calls, BLOCK_ATTR)

    def test_the_sharding_call_uses_keywords(self, strategy):
        """Positional passing is the mechanism of the bug the repair exists for."""
        instance, calls = strategy
        assert_sharding_args_are_keyword(instance, calls, BLOCK_ATTR)

    def test_a_missing_reshard_repair_is_reported_as_an_error(self, monkeypatch, caplog):
        """The patch runner isolates failures, so without this the symptom is one
        log line and then a whole run at ZeRO-3 traffic."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_REAL_AC", "1")
        module, _calls = install_stub_parallelizer(monkeypatch)
        install_stub_module(
            monkeypatch,
            "primus.backends.nemo_automodel.distributed.fsdp2_reshard",
            patch_installed=False,
            applied_reshard_after_forward=None,
        )

        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        parallelize.install()
        with caplog.at_level("ERROR"):
            module.PARALLELIZATION_STRATEGIES[MODEL_NAME]().parallelize(
                model=model_with_blocks(BLOCK_ATTR, blocks=[1]),
                device_mesh=FakeMesh(),
                reshard_after_forward=False,
            )
        assert [r for r in caplog.records if "reshard repair is NOT installed" in r.getMessage()]

    def test_it_does_not_raise_when_the_repair_is_missing(self, monkeypatch):
        """A run is under way by then, so failing it would be the worse outcome."""
        monkeypatch.setenv("PRIMUS_IDEOGRAM_REAL_AC", "1")
        module, _calls = install_stub_parallelizer(monkeypatch)
        install_stub_module(
            monkeypatch,
            "primus.backends.nemo_automodel.distributed.fsdp2_reshard",
            patch_installed=False,
            applied_reshard_after_forward=None,
        )

        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        parallelize.install()
        module.PARALLELIZATION_STRATEGIES[MODEL_NAME]().parallelize(
            model=model_with_blocks(BLOCK_ATTR, blocks=[1]),
            device_mesh=FakeMesh(),
            reshard_after_forward=False,
        )


# --------------------------------------------------------------------------- #
# Installation                                                                #
# --------------------------------------------------------------------------- #
class TestInstall:
    def test_it_is_a_no_op_when_not_requested(self):
        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        assert parallelize.install() is False

    def test_it_registers_under_the_model_class_name(self, strategy):
        """AutoModel dispatches by class name, so this string is the whole
        mechanism -- a typo makes the strategy silently never apply."""
        import nemo_automodel.components.distributed.parallelizer as P

        assert MODEL_NAME in P.PARALLELIZATION_STRATEGIES

    def test_it_is_idempotent(self, strategy, monkeypatch):
        import nemo_automodel.components.distributed.parallelizer as P

        from primus.backends.nemo_automodel.models.ideogram4 import parallelize

        registered = P.PARALLELIZATION_STRATEGIES[MODEL_NAME]
        assert parallelize.install() is True
        assert P.PARALLELIZATION_STRATEGIES[MODEL_NAME] is registered

    def test_a_tensor_parallel_mesh_warns(self, strategy, caplog):
        """There is no Ideogram TP plan, so a run that asked for one gets FSDP
        only. Proceeding is right; doing it quietly is not."""
        import types

        instance, _calls = strategy

        class MeshWithTP:
            mesh_dim_names = ("tp",)

            def __getitem__(self, name):
                return types.SimpleNamespace(size=lambda: 2)

            def size(self):
                return 8

        with caplog.at_level("WARNING"):
            instance.parallelize(model=model_with_blocks(BLOCK_ATTR, blocks=[1]), device_mesh=MeshWithTP())
        assert [r for r in caplog.records if "tensor parallelism" in r.getMessage()]


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
