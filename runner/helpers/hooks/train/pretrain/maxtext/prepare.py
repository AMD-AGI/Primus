###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc.
#
# See LICENSE for license information.
###############################################################################

"""
MaxText pre-train preparation hook for primus-cli direct.

This script is invoked by:

    runner/helpers/hooks/train/pretrain/prepare_experiment.sh

via:

    python maxtext/prepare.py \
        --config <exp.yaml> \
        --data_path <data_root> \
        --primus_path <primus_root> \
        --patch_args <patch_args.txt> \
        [--backend_path <override>] \
        [<extra CLI args>...]

It is responsible for:
  - Resolving the MaxText backend path (CLI > env > default)
  - Optionally preparing datasets (currently a no-op hook)
  - Emitting generic `extra.*=...` lines on stdout so that
    `primus-cli direct` can append the corresponding CLI args
    to the final training command:

        extra.backend_path=/path/to/maxtext   →  --backend_path /path/to/maxtext
"""

import argparse
import subprocess
from pathlib import Path
from typing import Optional

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import load_primus_config
from runner.helpers.hooks.train.pretrain.utils import (
    get_env_case_insensitive,
    log_error_and_exit,
    log_info,
)


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare Primus MaxText environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Path to write additional args (kept for compatibility; not used by this hook)",
    )
    parser.add_argument(
        "--backend_path",
        type=str,
        default=None,
        help="Optional override for MaxText path; takes precedence over env and default.",
    )
    return parser.parse_known_args()


def resolve_backend_path(
    cli_path: Optional[str], env_var: str, default_subdir: str, primus_path: Path, name: str
) -> Path:
    if cli_path:
        path = Path(cli_path).resolve()
        log_info(f"Using {name} path from CLI: {path}")
    else:
        env_value = get_env_case_insensitive(env_var)
        if env_value:
            path = Path(env_value).resolve()
            log_info(f"{env_var.upper()} found in environment: {path}")
        else:
            path = primus_path / default_subdir
            log_info(f"{env_var.upper()} not found, falling back to: {path}")
    return path


def prepare_dataset_if_needed(
    primus_config: PrimusConfig, primus_path: Path, data_path: Path, patch_args: Path, env=None
):
    """
    Placeholder for future MaxText dataset preparation logic.

    This hook is intentionally a no-op because current MaxText pre-train
    experiments in this integration either use synthetic data or datasets
    that are prepared outside of Primus, so no additional preprocessing
    is required here.

    Dataset preparation logic should be implemented in this function when
    non-synthetic or external datasets are introduced that require
    on-the-fly preprocessing or conversion, following the patterns used
    by other backends.
    """

    # No-op for now; datasets are prepared outside of Primus for MaxText.
    return


def install_maxtext_dependencies() -> None:
    """
    Ensure required RDMA / InfiniBand system packages for multi-node
    JAX/MaxText training are present.

    The package list is exclusively RDMA / IB / verbs / netlink related
    (``librdmacm-dev``, ``rdmacm-utils``, ``infiniband-diags``, ``perftest``,
    ``libibverbs-*``, ``libibumad*``, ``libnl-*`` ...). These are only needed
    for cross-node communication, so this step is intentionally skipped on
    single-node runs (``NNODES <= 1``), which makes single-node MaxText
    launches work cleanly even on minimal images with no apt connectivity.

    Behaviour (when ``NNODES > 1``):
      - If all packages are already installed (typical for pre-built JAX
        multi-node images), this is a fast no-op (no apt invocation).
      - Only actually-missing packages are installed; ``apt-get update``
        runs first so this works in containers whose apt index has not
        been populated yet.
      - apt failures are logged as warnings rather than fatal errors: a
        missing optional system library should not bring down the whole
        training pipeline. The user can bake the packages into the image
        or rerun once apt connectivity is restored.

    Note on package names:
      ``libibnetdisc5`` was renamed to ``libibnetdisc5t64`` starting from
      Ubuntu 24.04 (noble) due to the time_t transition. We list the new
      name; on older releases dpkg-query will simply report it as missing
      and the warning path will be taken.
    """
    import os

    try:
        nnodes = int(os.environ.get("NNODES", "1"))
    except ValueError:
        nnodes = 1

    if nnodes <= 1:
        log_info(
            f"NNODES={nnodes}, single-node run: skipping RDMA/IB system package "
            "install (these packages are only needed for multi-node training)."
        )
        return

    log_info(
        f"NNODES={nnodes}, multi-node run: ensuring RDMA/IB system packages "
        "for Jax/MaxText are installed."
    )

    pkgs = [
        "iproute2",
        "libelf-dev",
        "gcc",
        "make",
        "libtool",
        "autoconf",
        "librdmacm-dev",
        "rdmacm-utils",
        "infiniband-diags",
        "ibverbs-utils",
        "perftest",
        "ethtool",
        "libibverbs-dev",
        "rdma-core",
        "strace",
        "libibmad5",
        "libibnetdisc5t64",
        "ibverbs-providers",
        "libibumad-dev",
        "libibumad3",
        "libibverbs1",
        "libnl-3-dev",
        "libnl-route-3-dev",
    ]

    def _missing(pkg_list):
        out = []
        for p in pkg_list:
            r = subprocess.run(
                ["dpkg-query", "-W", "-f=${Status}", p],
                capture_output=True,
                text=True,
            )
            if "install ok installed" not in r.stdout:
                out.append(p)
        return out

    missing = _missing(pkgs)
    if not missing:
        log_info("All MaxText system packages already installed; skipping apt.")
        return

    log_info(f"Missing system packages, attempting apt install: {' '.join(missing)}")

    steps = [
        ["apt-get", "update"],
        ["apt-get", "install", "-y", "--no-install-recommends", *missing],
    ]
    for cmd in steps:
        try:
            subprocess.run(cmd, check=True)
        except subprocess.CalledProcessError as exc:
            log_info(
                f"WARNING: '{' '.join(cmd)}' failed (exit {exc.returncode}). "
                "MaxText system-package install was not completed. If you are "
                "using a pre-built image, consider baking these packages into "
                "the image. Continuing with training launch."
            )
            return
        except FileNotFoundError:
            log_info(
                f"WARNING: '{cmd[0]}' not found. Skipping system-package install."
            )
            return

    still_missing = _missing(missing)
    if still_missing:
        log_info(
            "WARNING: The following packages are still missing after apt install: "
            f"{' '.join(still_missing)}. Continuing with training launch."
        )


# ---------- Main ----------
def main():
    args, unknown = parse_args()

    primus_path = Path(args.primus_path).resolve()
    data_path = Path(args.data_path).resolve()
    exp_path = Path(args.config).resolve()
    patch_args_file = Path(args.patch_args).resolve()

    log_info(f"PRIMUS_PATH: {primus_path}")
    log_info(f"DATA_PATH: {data_path}")
    log_info(f"EXP: {exp_path}")
    log_info(f"BACKEND_PATH: {args.backend_path}")
    log_info(f"PATCH-ARGS: {patch_args_file}")

    if not exp_path.is_file():
        log_error_and_exit(f"EXP file not found: {exp_path}")

    primus_config, _ = load_primus_config(args, unknown)

    maxtext_path = resolve_backend_path(
        args.backend_path, "MAXTEXT_PATH", "third_party/maxtext", primus_path, "MaxText"
    )

    try:
        pre_trainer_cfg = primus_config.get_module_config("pre_trainer")
    except Exception:
        log_error_and_exit("Missing required module config: pre_trainer")

    if not hasattr(pre_trainer_cfg, "dataset_type") or pre_trainer_cfg.dataset_type is None:
        log_error_and_exit("Missing required field: pre_trainer.dataset_type")

    dataset_type = pre_trainer_cfg.dataset_type

    # Then install required MaxText/JAX system packages on every node.
    install_maxtext_dependencies()

    if dataset_type == "synthetic":
        log_info("'dataset_type: synthetic', Skipping dataset preparation.")
    else:
        prepare_dataset_if_needed(
            primus_config=primus_config,
            primus_path=primus_path,
            data_path=data_path,
            patch_args=patch_args_file,
            env=None,
        )

    # Expose resolved backend path to the caller (e.g., primus-cli direct)
    # via a generic extra.* line on stdout, which will be converted to:
    #   --backend_path <maxtext_path>
    log_info(f"Exposing resolved backend path via extra.backend_path={maxtext_path}")
    print(f"extra.backend_path={maxtext_path}")

    # Expose JAX coordinator environment variables for distributed training
    # These will be exported by execute_hooks.sh
    import os

    master_addr = os.getenv("MASTER_ADDR", "localhost")
    master_port = os.getenv("MASTER_PORT", "1234")

    log_info(
        f"Exposing JAX coordinator: JAX_COORDINATOR_IP={master_addr}, JAX_COORDINATOR_PORT={master_port}"
    )
    print(f"env.JAX_COORDINATOR_IP={master_addr}")
    print(f"env.JAX_COORDINATOR_PORT={master_port}")

    # Expose MaxText/JAX performance tuning environment variables
    # These mirror the settings from examples/run_pretrain.sh
    log_info("Exposing MaxText performance tuning environment variables")

    # XLA/JAX settings
    dump_hlo_dir = os.getenv("DUMP_HLO_DIR", f"{primus_path}/output/xla_dump_hlo")
    dump_hlo = os.getenv("DUMP_HLO", "0")
    print(f"env.DUMP_HLO_DIR={dump_hlo_dir}")
    print(f"env.DUMP_HLO={dump_hlo}")
    print("env.NVTE_ALLOW_NONDETERMINISTIC_ALGO=1")
    # set XLA_PYTHON_CLIENT_MEM_FRACTION to 0.93
    # to avoid HSA_STATUS_ERROR_OUT_OF_RESOURCES error during multi-node training
    xla_python_client_mem_fraction = os.getenv("XLA_PYTHON_CLIENT_MEM_FRACTION", ".97")
    print(f"env.XLA_PYTHON_CLIENT_MEM_FRACTION={xla_python_client_mem_fraction}")
    print("env.NVTE_USE_HIPBLASLT=1")

    xla_flags = "--xla_gpu_memory_limit_slop_factor=95 --xla_gpu_reduce_scatter_combine_threshold_bytes=8589934592 --xla_gpu_enable_command_buffer='' --xla_gpu_enable_latency_hiding_scheduler=true --xla_gpu_all_gather_combine_threshold_bytes=8589934592 --xla_gpu_enable_triton_gemm=false --xla_gpu_enable_cublaslt=true --xla_gpu_autotune_level=0 --xla_gpu_enable_all_gather_combine_by_dim=false"
    if dump_hlo == "1":
        xla_flags += f" --xla_dump_to={dump_hlo_dir}"
        log_info(f"XLA HLO dumping enabled, output directory: {dump_hlo_dir}")
    print(f"env.XLA_FLAGS={xla_flags}")
    # set TF_CPP_MIN_LOG_LEVEL=2 to suppress the error messages at the end of JAX/MaxText training
    print(f"env.TF_CPP_MIN_LOG_LEVEL=2")

    # AMD GPU optimizations
    print("env.HIP_FORCE_DEV_KERNARG=1")
    print("env.HSA_FORCE_FINE_GRAIN_PCIE=1")

    # Transformer Engine settings for MaxText
    print("env.NVTE_FUSED_ATTN=1")
    print("env.NVTE_CK_USES_BWD_V3=1")
    print("env.NVTE_CK_USES_FWD_V3=1")
    print("env.NVTE_CK_IS_V3_ATOMIC_FP32=0")
    print("env.NVTE_CK_HOW_V3_BF16_CVT=2")
    print("env.NVTE_FUSED_ATTN_CK=1")
    print("env.NVTE_FUSED_ATTN_AOTRITON=0")

    # Expose run mode: MaxText uses single mode (plain python instead of torchrun)
    # This will be exported as RUN_MODE env var by execute_hooks.sh
    log_info("Exposing run mode via env.RUN_MODE=single")
    print("env.RUN_MODE=single")


if __name__ == "__main__":
    log_info("========== Prepare MaxText Env (pre-train hook) ==========")
    main()
