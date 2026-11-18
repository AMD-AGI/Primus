###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""
Unit tests for the AutoModel diffusion cache-builder registry.

THE PROPERTY WORTH DEFENDING: listing the models must not import any of them.

The CLI builds its ``--model`` choices from this registry, so it consults it on
every invocation -- including ``primus data --help``. A builder pulls in an
autoencoder, a text encoder and an image library, so a registry that imported its
entries eagerly would make the help text depend on the whole encoder stack being
installed, and would let one model's missing optional dependency break every other
model's build. That is why the entries are dotted strings, and why the tests below
check it in a fresh interpreter: in this one, an earlier test may already have
imported a builder, which would make any sys.modules check pass vacuously.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path

import pytest

from primus.backends.automodel.data import registry

REPO_ROOT = Path(__file__).resolve().parents[4]

# What a builder pulls in, and what listing or registering must not.
BUILDER_MODULES = (
    "primus.backends.automodel.models.ideogram4.data.build",
    "primus.backends.automodel.models.ideogram4.processor",
)
ENCODER_STACK = ("diffusers", "transformers", "PIL", "nemo_automodel")


def imported_by(statement, baseline="import primus.backends.automodel"):
    """The modules ``statement`` imports in a fresh interpreter, beyond those the
    baseline already did. The baseline is the package itself, whose own imports
    are not what is under test."""
    code = "\n".join(
        [
            "import json, sys",
            baseline,
            "before = set(sys.modules)",
            statement,
            "print(json.dumps(sorted(set(sys.modules) - before)))",
        ]
    )
    result = subprocess.run(
        [sys.executable, "-c", code], cwd=REPO_ROOT, capture_output=True, text=True, timeout=300
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout.strip().splitlines()[-1])


def heavy(modules, names):
    return [m for m in modules if any(m == n or m.startswith(n + ".") for n in names)]


def cache_parser():
    from primus.cli.subcommands import data as data_subcommand

    parser = argparse.ArgumentParser()
    data_subcommand.register_subcommand(parser.add_subparsers())
    return parser


REQUIRED_ARGS = [
    "data",
    "automodel-cache",
    "--model",
    "ideogram4",
    "--image-dir",
    "/images",
    "--caption-dir",
    "/captions",
    "--output-dir",
    "/cache",
]


class TestListing:
    def test_the_registered_models_are_listed(self):
        assert "ideogram4" in registry.available_models()

    def test_the_listing_is_sorted(self):
        """It goes into user-facing help and argparse choices, so an order that
        depended on dictionary insertion would be gratuitously unstable."""
        models = registry.available_models()
        assert models == sorted(models)

    def test_listing_imports_nothing(self):
        """The property this design exists for."""
        modules = imported_by(
            "from primus.backends.automodel.data import registry; registry.available_models()"
        )
        assert not heavy(modules, BUILDER_MODULES + ENCODER_STACK)

    def test_the_entries_are_strings_not_callables(self):
        for model, target in registry.CACHE_BUILDERS.items():
            assert isinstance(target, str), f"{model} was registered as an import"
            assert ":" in target, f"{model} is missing the ':<callable>' part"


class TestResolution:
    def test_a_registered_model_resolves_to_a_callable(self):
        pytest.importorskip("torch")
        builder = registry.get_cache_builder("ideogram4")
        assert callable(builder)

    def test_the_resolved_builder_takes_the_arguments_the_cli_passes(self):
        """The CLI calls the builder by keyword, so a rename on either side would
        otherwise surface only when someone ran the command with real weights."""
        pytest.importorskip("torch")
        import inspect

        parameters = inspect.signature(registry.get_cache_builder("ideogram4")).parameters
        for name in (
            "image_dir",
            "caption_dir",
            "output_dir",
            "num_samples",
            "resolution",
            "max_text_tokens",
            "vae_source",
            "text_encoder_source",
            "tokenizer_source",
            "device",
            "dtype",
            "seed",
            "shuffle",
        ):
            assert name in parameters, f"the builder does not accept {name}"

    def test_every_registered_model_resolves(self):
        """A typo in a dotted string is only ever discovered at dispatch, which for
        this registry means after someone has waited for a model to load."""
        pytest.importorskip("torch")
        for model in registry.available_models():
            assert callable(registry.get_cache_builder(model))


class TestUnknownModel:
    def test_an_unknown_model_is_refused(self):
        with pytest.raises(ValueError, match="no cache builder"):
            registry.get_cache_builder("not-a-model")

    def test_the_error_lists_what_is_available(self):
        """The registry is the only place this list exists, so the error is where a
        user finds out what they could have typed."""
        with pytest.raises(ValueError) as excinfo:
            registry.get_cache_builder("not-a-model")
        for model in registry.available_models():
            assert model in str(excinfo.value)


class TestCliWiring:
    def test_the_subcommand_registers_without_the_encoder_stack(self):
        """Registering the parser calls available_models. If that import chain ever
        reached a builder, 'primus data --help' would start requiring diffusers and
        transformers."""
        modules = imported_by(
            "import argparse\n"
            "from primus.backends.automodel.data.cli import register_data_subcommands\n"
            "register_data_subcommands(argparse.ArgumentParser().add_subparsers())"
        )
        assert not heavy(modules, BUILDER_MODULES + ENCODER_STACK)

    def test_the_full_data_parser_does_not_reach_a_builder(self):
        """The whole 'primus data' parser, Megatron commands included. Those may
        import what they need, so only the AutoModel builders are checked."""
        modules = imported_by(
            "import argparse\n"
            "from primus.cli.subcommands import data\n"
            "data.register_subcommand(argparse.ArgumentParser().add_subparsers())"
        )
        assert not heavy(modules, BUILDER_MODULES)

    def test_the_defaults(self):
        parsed = cache_parser().parse_args(REQUIRED_ARGS)
        assert parsed.model == "ideogram4"
        assert parsed.num_samples == 1024
        assert parsed.resolution == 256
        assert parsed.max_text_tokens == 128
        assert parsed.shuffle is False, "sorted order is the default"

    def test_the_encoder_sources_default_to_unset(self):
        """So the builder's own defaults win, rather than being overwritten by
        None coming down from argparse."""
        parsed = cache_parser().parse_args(REQUIRED_ARGS)
        assert parsed.vae_source is None
        assert parsed.text_encoder_source is None
        assert parsed.tokenizer_source is None

    @pytest.mark.parametrize("flag", ["--resolution", "--max-text-tokens"])
    @pytest.mark.parametrize("value", ["0", "-16"])
    def test_a_non_positive_size_is_refused_at_parse_time(self, flag, value):
        """Before the encoders load, rather than after every image is encoded."""
        with pytest.raises(SystemExit):
            cache_parser().parse_args(REQUIRED_ARGS + [flag, value])

    def test_an_unregistered_model_is_rejected_by_argparse(self):
        """The choices come from the registry, so this is the registry deciding."""
        args = list(REQUIRED_ARGS)
        args[args.index("ideogram4")] = "flux"
        with pytest.raises(SystemExit):
            cache_parser().parse_args(args)

    def test_the_existing_commands_are_still_registered(self):
        """This adds a command to a parser three others share, so the check that
        it stayed additive is worth having explicitly."""
        from primus.cli.subcommands import data as data_subcommand

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        data_parser = data_subcommand.register_subcommand(subparsers)

        commands = set()
        for action in data_parser._actions:
            if isinstance(action, argparse._SubParsersAction):
                commands |= set(action.choices)

        assert {
            "diffusion-raw",
            "diffusion-encoded",
            "diffusion-ingest",
            "automodel-cache",
        } <= commands


class TestDispatch:
    """What the command hands the builder, with the builder replaced."""

    @staticmethod
    def dispatch(monkeypatch, extra=()):
        from primus.backends.automodel.data import cli

        received = {}

        def builder(**kwargs):
            received.update(kwargs)
            return {
                "num_samples": 3,
                "in_channels": 128,
                "grid_h": 16,
                "grid_w": 16,
                "llm_features_dim": 4096,
                "skipped": {"no_caption": 1, "too_long": 2, "failed": 0},
            }

        monkeypatch.setattr(registry, "get_cache_builder", lambda model: builder)
        args = cache_parser().parse_args(REQUIRED_ARGS + list(extra))
        cli._prepare_automodel_cache(args)
        return received

    def test_the_arguments_reach_the_builder(self, monkeypatch):
        received = self.dispatch(monkeypatch, ["--num-samples", "7", "--resolution", "512", "--shuffle"])
        assert received["image_dir"] == "/images"
        assert received["caption_dir"] == "/captions"
        assert received["output_dir"] == "/cache"
        assert received["num_samples"] == 7
        assert received["resolution"] == 512
        assert received["max_text_tokens"] == 128
        assert received["shuffle"] is True

    def test_unset_sources_are_left_to_the_builder(self, monkeypatch):
        received = self.dispatch(monkeypatch)
        assert not {"vae_source", "text_encoder_source", "tokenizer_source"} & set(received)

    def test_set_sources_are_passed(self, monkeypatch):
        received = self.dispatch(
            monkeypatch,
            ["--vae-source", "/vae", "--text-encoder-source", "/te", "--tokenizer-source", "/tok"],
        )
        assert (received["vae_source"], received["text_encoder_source"], received["tokenizer_source"]) == (
            "/vae",
            "/te",
            "/tok",
        )

    def test_the_skip_counts_are_reported(self, monkeypatch, caplog):
        with caplog.at_level("INFO"):
            self.dispatch(monkeypatch)
        assert any("2 over the token budget" in r.getMessage() for r in caplog.records)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
