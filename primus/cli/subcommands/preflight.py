###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import logging
import os
import platform
import sys
from typing import List, Optional

from primus.cli.base import CommandBase
from primus.cli.utils import print_error, print_info, print_success, print_warning

logger = logging.getLogger(__name__)


class PreflightCommand(CommandBase):
    """Command for environment and configuration checks.

    Performs various checks to ensure the environment is properly configured
    for running Primus training and benchmarking workloads.
    """

    @classmethod
    def name(cls) -> str:
        return "preflight"

    @classmethod
    def help(cls) -> str:
        return "Environment and configuration checks"

    @classmethod
    def description(cls) -> str:
        return """
Environment validation and configuration checks.

This command performs various checks to ensure your environment is properly
configured for Primus workloads, including:
  - Python version and dependencies
  - GPU availability and configuration
  - ROCM/HIP installation
  - Network configuration for distributed training
  - File system permissions

Examples:
  # Run all checks
  primus preflight --check-all

  # Run only GPU checks
  primus preflight --check-gpu

  # Run checks with verbose output
  primus preflight --check-all --verbose
        """

    @classmethod
    def register_arguments(cls, parser: argparse.ArgumentParser) -> None:
        """Register preflight-specific arguments."""
        parser.description = cls.description()

        parser.add_argument(
            "--check-all",
            action="store_true",
            help="Run all checks (default if no specific checks specified)",
        )
        parser.add_argument(
            "--check-python",
            action="store_true",
            help="Check Python version and dependencies",
        )
        parser.add_argument(
            "--check-gpu",
            action="store_true",
            help="Check GPU availability and configuration",
        )
        parser.add_argument(
            "--check-rocm",
            action="store_true",
            help="Check ROCM/HIP installation",
        )
        parser.add_argument(
            "--check-network",
            action="store_true",
            help="Check network configuration for distributed training",
        )
        parser.add_argument(
            "--check-filesystem",
            action="store_true",
            help="Check file system permissions and paths",
        )

    @classmethod
    def run(cls, args: argparse.Namespace, unknown_args: Optional[List[str]] = None) -> None:
        """Execute the preflight checks.

        Args:
            args: Parsed arguments.
            unknown_args: Unknown arguments (ignored).
        """
        print_info("Starting Primus environment preflight checks...")
        print()

        # Determine which checks to run
        run_all = args.check_all or not any(
            [
                args.check_python,
                args.check_gpu,
                args.check_rocm,
                args.check_network,
                args.check_filesystem,
            ]
        )

        checks = {
            "python": run_all or args.check_python,
            "gpu": run_all or args.check_gpu,
            "rocm": run_all or args.check_rocm,
            "network": run_all or args.check_network,
            "filesystem": run_all or args.check_filesystem,
        }

        all_passed = True

        # Run checks
        if checks["python"]:
            all_passed &= cls._check_python()

        if checks["gpu"]:
            all_passed &= cls._check_gpu()

        if checks["rocm"]:
            all_passed &= cls._check_rocm()

        if checks["network"]:
            all_passed &= cls._check_network()

        if checks["filesystem"]:
            all_passed &= cls._check_filesystem()

        # Summary
        print()
        print("=" * 70)
        if all_passed:
            print_success("All checks passed! Environment is ready for Primus.")
        else:
            print_warning("Some checks failed. Please review the output above.")
            sys.exit(1)

    @classmethod
    def _check_python(cls) -> bool:
        """Check Python version and key dependencies."""
        print("🐍 Python Environment Check")
        print("-" * 70)

        all_passed = True

        # Check Python version
        version = sys.version_info
        version_str = f"{version.major}.{version.minor}.{version.minor}"
        print(f"  Python version: {version_str}")

        if version.major == 3 and version.minor >= 8:
            print_success("  Python version is compatible (>= 3.8)")
        else:
            print_error("  Python version must be >= 3.8")
            all_passed = False

        # Check key dependencies
        dependencies = ["torch", "yaml", "numpy"]
        for dep in dependencies:
            try:
                __import__(dep)
                print_success(f"  {dep} is installed")
            except ImportError:
                print_error(f"  {dep} is not installed")
                all_passed = False

        print()
        return all_passed

    @classmethod
    def _check_gpu(cls) -> bool:
        """Check GPU availability."""
        print("🎮 GPU Check")
        print("-" * 70)

        try:
            import torch

            if torch.cuda.is_available():
                num_gpus = torch.cuda.device_count()
                print_success(f"  CUDA/ROCm available with {num_gpus} GPU(s)")

                for i in range(num_gpus):
                    gpu_name = torch.cuda.get_device_name(i)
                    print(f"    GPU {i}: {gpu_name}")

                # Check GPU memory
                if num_gpus > 0:
                    mem_total = torch.cuda.get_device_properties(0).total_memory / (1024**3)
                    print(f"    GPU 0 memory: {mem_total:.2f} GB")

                print()
                return True
            else:
                print_warning("  No GPUs detected. Some functionality may be limited.")
                print()
                return True  # Warning, not error
        except ImportError:
            print_error("  PyTorch is not installed")
            print()
            return False
        except Exception as e:
            print_error(f"  Error checking GPUs: {e}")
            print()
            return False

    @classmethod
    def _check_rocm(cls) -> bool:
        """Check ROCM/HIP installation."""
        print("🔧 ROCM/HIP Check")
        print("-" * 70)

        all_passed = True

        # Check ROCM_HOME
        rocm_home = os.getenv("ROCM_HOME") or os.getenv("ROCM_PATH")
        if rocm_home and os.path.exists(rocm_home):
            print_success(f"  ROCM_HOME: {rocm_home}")

            # Check for key ROCM tools
            tools = ["rocm-smi", "hipcc"]
            for tool in tools:
                tool_path = os.path.join(rocm_home, "bin", tool)
                if os.path.exists(tool_path):
                    print_success(f"  {tool} found")
                else:
                    print_warning(f"  {tool} not found")
        else:
            print_warning("  ROCM_HOME not set or invalid (may be using CUDA instead)")

        # Check PyTorch ROCM support
        try:
            import torch

            if hasattr(torch.version, "hip"):
                print_success(f"  PyTorch built with ROCm: {torch.version.hip}")
            else:
                print_info("  PyTorch built with CUDA (not ROCm)")
        except ImportError:
            pass

        print()
        return all_passed

    @classmethod
    def _check_network(cls) -> bool:
        """Check network configuration."""
        print("🌐 Network Check")
        print("-" * 70)

        all_passed = True

        # Check hostname
        hostname = platform.node()
        print(f"  Hostname: {hostname}")

        # Check common environment variables for distributed training
        env_vars = ["MASTER_ADDR", "MASTER_PORT", "RANK", "WORLD_SIZE", "LOCAL_RANK"]
        any_set = False
        for var in env_vars:
            value = os.getenv(var)
            if value:
                print(f"  {var}: {value}")
                any_set = True

        if not any_set:
            print_info("  No distributed training environment variables set")
            print_info("  (This is normal for single-node training)")

        print()
        return all_passed

    @classmethod
    def _check_filesystem(cls) -> bool:
        """Check file system permissions."""
        print("📁 File System Check")
        print("-" * 70)

        all_passed = True

        # Check current directory
        cwd = os.getcwd()
        print(f"  Current directory: {cwd}")

        if os.access(cwd, os.W_OK):
            print_success("  Current directory is writable")
        else:
            print_error("  Current directory is not writable")
            all_passed = False

        # Check common paths
        paths_to_check = {
            "Home": os.path.expanduser("~"),
            "Temp": "/tmp",
        }

        for name, path in paths_to_check.items():
            if os.path.exists(path):
                if os.access(path, os.W_OK):
                    print_success(f"  {name} directory ({path}) is writable")
                else:
                    print_warning(f"  {name} directory ({path}) is not writable")

        print()
        return all_passed
