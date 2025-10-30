import argparse
import os
import sys
from pathlib import Path
 
from primus.core.launcher.parser import PrimusParser
from primus.core.projection.training_config import convert_primus_config_to_projection_config
from primus.core.projection.module_profilers.language_model import build_profiler, get_language_model_profiler_spec


def launch_projection_from_cli(args, overrides):
    """
    Entry point for the 'projection' subcommand.

    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Projection] Config file '{cfg_path}' not found.")

    config_parser = PrimusParser()
    primus_config = config_parser.parse(args)
    training_config = convert_primus_config_to_projection_config(primus_config)
    print(training_config)

    model_profiler_spec = get_language_model_profiler_spec(training_config)
    model_profiler = build_profiler(model_profiler_spec)