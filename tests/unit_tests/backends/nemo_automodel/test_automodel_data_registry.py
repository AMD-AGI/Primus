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
model's build. That is why the entries are dotted strings, and why the test below
checks it by watching sys.modules rather than by reading the code.
"""

import sys

import pytest

from primus.backends.nemo_automodel.data import registry


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
        before = set(sys.modules)
        registry.available_models()
        newly_imported = set(sys.modules) - before
        assert not [m for m in newly_imported if "ideogram4" in m]

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
        import argparse

        from primus.cli.subcommands import data as data_subcommand

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        data_subcommand.register_subcommand(subparsers)

        parsed = parser.parse_args(
            [
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
        )
        assert parsed.model == "ideogram4"
        assert parsed.num_samples == 1024
        assert parsed.resolution == 256
        assert parsed.max_text_tokens == 128
        assert parsed.shuffle is False, "sorted order is the default"

    def test_the_encoder_sources_default_to_unset(self):
        """So the builder's own defaults win, rather than being overwritten by
        None coming down from argparse."""
        import argparse

        from primus.cli.subcommands import data as data_subcommand

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        data_subcommand.register_subcommand(subparsers)
        parsed = parser.parse_args(
            [
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
        )
        assert parsed.vae_source is None
        assert parsed.text_encoder_source is None
        assert parsed.tokenizer_source is None

    def test_an_unregistered_model_is_rejected_by_argparse(self):
        """The choices come from the registry, so this is the registry deciding."""
        import argparse

        from primus.cli.subcommands import data as data_subcommand

        parser = argparse.ArgumentParser()
        subparsers = parser.add_subparsers()
        data_subcommand.register_subcommand(subparsers)

        with pytest.raises(SystemExit):
            parser.parse_args(
                [
                    "data",
                    "automodel-cache",
                    "--model",
                    "flux",
                    "--image-dir",
                    "/images",
                    "--caption-dir",
                    "/captions",
                    "--output-dir",
                    "/cache",
                ]
            )

    def test_the_existing_commands_are_still_registered(self):
        """This adds a command to a parser three others share, so the check that
        it stayed additive is worth having explicitly."""
        import argparse

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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
