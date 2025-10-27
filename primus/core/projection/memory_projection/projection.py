import argparse
import os
import sys
from pathlib import Path

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import add_pretrain_parser, load_primus_config


def launch_projection_from_cli(args, overrides):
    """
    Entry point for the 'projection' subcommand.

    """
    cfg_path = Path(args.config)
    if not cfg_path.exists():
        raise FileNotFoundError(f"[Primus:Projection] Config file '{cfg_path}' not found.")
    print("HELLO WORLD FROM PROJECTION")
    