###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Backend-path / environment helpers shared by the training and projection
entry points.

Training is driven entirely by the core runtime
(:mod:`primus.core.runtime.train_runtime`). Only the backend-path resolution
utilities remain here, still used by the projection subcommand, runner hooks
and examples.
"""

import os
import sys
from pathlib import Path


def _info_enabled() -> bool:
    """True when PRIMUS_LOG_LEVEL permits INFO-level chatter (DEBUG/INFO).

    Mirrors the level gating in runner/lib/common.sh so informational prints are
    silenced when the user sets PRIMUS_LOG_LEVEL=WARN/ERROR.
    """
    return os.environ.get("PRIMUS_LOG_LEVEL", "INFO").upper() in ("DEBUG", "INFO")


def setup_backend_path(framework: str, backend_path=None, verbose: bool = True):
    """
    Setup Python path for backend modules.

    Priority order:
    1. --backend-path from CLI
    2. BACKEND_PATH from environment
    3. Source tree fallback: <primus>/../../third_party/{framework}

    Returns:
        str: The first valid backend path inserted into sys.path.
    """
    candidate_paths = []

    # 1) From CLI
    if backend_path:
        if isinstance(backend_path, str):
            backend_path = [backend_path]
        candidate_paths.extend(backend_path)

    # 2) From environment variable
    env_path = os.getenv("BACKEND_PATH")
    if env_path:
        candidate_paths.append(env_path)

    # 3) Fallback: source-tree third_party/<name> first, then the deps-sync
    #    location used by installed wheels (`primus-cli deps sync`):
    #    $PRIMUS_THIRDPARTY_DIR or ~/.cache/Primus/third_party.
    fallback_name_map = {
        "megatron": "Megatron-LM",
        "torchtitan": "torchtitan",
        "maxtext": "maxtext",
    }
    mapped_name = fallback_name_map.get(framework, framework)

    source_tree_path = Path(__file__).resolve().parent.parent / "third_party" / mapped_name
    if framework == "maxtext" and (source_tree_path / "src").exists():
        source_tree_path = source_tree_path / "src"
    candidate_paths.insert(0, str(source_tree_path))

    tp_root = os.getenv("PRIMUS_THIRDPARTY_DIR") or str(Path.home() / ".cache" / "Primus" / "third_party")
    deps_sync_path = Path(tp_root) / mapped_name
    if framework == "maxtext" and (deps_sync_path / "src").exists():
        deps_sync_path = deps_sync_path / "src"
    candidate_paths.append(str(deps_sync_path))
    if verbose and _info_enabled():
        print(f"[Primus] candidate_paths: {candidate_paths}")

    # Normalize & deduplicate
    candidate_paths = list(dict.fromkeys(os.path.normpath(os.path.abspath(p)) for p in candidate_paths))

    # Insert the first existing path into sys.path
    for path in candidate_paths:
        if os.path.exists(path):
            if path not in sys.path:
                sys.path.insert(0, path)
                if verbose and _info_enabled():
                    print(f"[Primus] sys.path.insert: {path}")
            return path  # Return the first valid path

    # None of the candidate paths exist
    raise FileNotFoundError(
        f"[Primus] backend_path not found for framework '{framework}'. "
        f"Tried paths: {candidate_paths}. "
        f"Hint: run `primus-cli deps sync` to populate the deps-sync dir, "
        f"install the backend under third_party/{mapped_name}, "
        f"or provide a valid --backend_path/BACKEND_PATH."
    )


def setup_env(data_path: str):
    if "HF_HOME" not in os.environ:
        hf_home = os.path.join(data_path, "huggingface")
        os.environ["HF_HOME"] = hf_home
        print(f"[Primus CLI] HF_HOME={hf_home}")
    else:
        hf_home = os.environ["HF_HOME"]
        print(f"[Primus CLI] HF_HOME already set: {hf_home}")
