###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import importlib

import yaml


class PrimusPlugin:
    """Base class for all Primus plugins."""

    def on_setup(self, runtime):
        pass

    def on_train_start(self, runtime):
        pass

    def on_train_end(self, runtime):
        pass

    def on_finalize(self, runtime):
        pass


class TrainRuntime:
    """TrainRuntime manages the training lifecycle and plugin hooks."""

    def __init__(self, config_path: str):
        self.config = self._load_config(config_path)
        self.plugins = []
        self.backend = None

    def _load_config(self, config_path: str):
        with open(config_path, "r") as f:
            return yaml.safe_load(f)

    def _load_plugins(self):
        plugin_names = self.config.get("plugins", [])
        for name in plugin_names:
            module_path = f"primus.plugins.{name}.plugin"
            module = importlib.import_module(module_path)
            plugin = module.register_plugin()
            self.plugins.append(plugin)

    def _load_backend(self):
        backend_name = self.config.get("backend")
        if not backend_name:
            raise ValueError("Backend not specified in the configuration.")

        module_path = f"primus.modules.trainer.{backend_name}.pre_trainer"
        backend_module = importlib.import_module(module_path)
        self.backend = backend_module.TrainerClass()

    def setup(self):
        for plugin in self.plugins:
            if hasattr(plugin, "before_setup"):
                plugin.before_setup(self)

        for plugin in self.plugins:
            plugin.on_setup(self)

        for plugin in self.plugins:
            if hasattr(plugin, "after_setup"):
                plugin.after_setup(self)

    def launch(self):
        for plugin in self.plugins:
            plugin.on_train_start(self)

        if self.backend:
            self.backend.launch()

        for plugin in self.plugins:
            plugin.on_train_end(self)

    def finalize(self):
        for plugin in self.plugins:
            if hasattr(plugin, "before_finalize"):
                plugin.before_finalize(self)

        for plugin in self.plugins:
            plugin.on_finalize(self)

        for plugin in self.plugins:
            if hasattr(plugin, "after_finalize"):
                plugin.after_finalize(self)

    def run(self):
        self._load_plugins()
        self._load_backend()
        self.setup()
        self.launch()
        self.finalize()
