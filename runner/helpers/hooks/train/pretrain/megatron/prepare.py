###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

import argparse
import hashlib
import json
import os
import subprocess
import time
from pathlib import Path
from time import sleep

import nltk
from datasets import load_dataset

from primus.core.launcher.config import PrimusConfig
from primus.core.launcher.parser import load_primus_config
from primus.pretrain import setup_backend_path
from runner.helpers.hooks.train.pretrain.utils import (
    default_backend_path,
    get_env_case_insensitive,
    get_node_rank,
    log_error_and_exit,
    log_info,
)

# The split seed is deliberately not spelled --seed. primus-cli forwards one
# argument list to both this hook and training, and `seed` is already a Megatron
# training parameter (trainer_base.yaml, default 1234). While the two shared a
# flag, retuning the training RNG changed the split cache key and so
# re-downloaded BookCorpus and re-tokenized the corpus from scratch.
DEFAULT_SPLIT_SEED = 42


# ---------- Helpers ----------
def check_dir_nonempty(path: Path, name: str):
    if not path.is_dir() or not any(path.iterdir()):
        log_error_and_exit(
            f"{name} ({path}) does not exist or is empty.\n"
            "Please ensure Primus is properly initialized.\n"
            "If not yet cloned, run:\n"
            "    git clone --recurse-submodules git@github.com:AMD-AGI/Primus.git\n"
            "Or if already cloned, initialize submodules with:\n"
            "    git submodule update --init --recursive"
        )


def tokenized_files_exist(tokenized_data_path: Path) -> bool:
    return Path(f"{tokenized_data_path}.bin").exists() and Path(f"{tokenized_data_path}.idx").exists()


def remove_tokenized_files(tokenized_data_path: Path):
    for suffix in (".bin", ".idx"):
        Path(f"{tokenized_data_path}{suffix}").unlink(missing_ok=True)


def split_cache_key(test_size: float, seed: int) -> str:
    # The payload key stays "seed" even though the flag is --split_seed: it is
    # hashed into raw-split filenames and written to .split.json, so renaming it
    # would invalidate every cache already on disk.
    payload = json.dumps({"seed": seed, "test_size": test_size}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode()).hexdigest()[:12]


def split_metadata_path(tokenized_eval_data_path: Path) -> Path:
    return Path(f"{tokenized_eval_data_path}.split.json")


def split_metadata_matches(metadata_path: Path, test_size: float, seed: int) -> bool:
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, ValueError, TypeError):
        return False
    return metadata == {"seed": seed, "test_size": test_size}


def atomic_write(path: Path, content: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def write_split_metadata(metadata_path: Path, test_size: float, seed: int):
    atomic_write(
        metadata_path,
        json.dumps({"seed": seed, "test_size": test_size}, sort_keys=True) + "\n",
    )


def run_preprocess(
    dataset_json: Path,
    output_prefix: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    env=None,
):
    # Use preprocess_data.py from the current script's directory
    preprocess_script = Path(__file__).parent / "preprocess_data.py"
    output_prefix.parent.mkdir(parents=True, exist_ok=True)

    # preprocess_data.py appends ``_text_sentence`` to --output-prefix. Treat
    # output_prefix here as the final Megatron prefix exposed to training, so a
    # custom TOKENIZED_*_DATA_PATH is honored exactly.
    generated_suffix = "_text_sentence"
    output_prefix_text = str(output_prefix)
    needs_rename = not output_prefix_text.endswith(generated_suffix)
    if needs_rename:
        preprocess_prefix = output_prefix.with_name(f".{output_prefix.name}.primus_preprocess")
    else:
        preprocess_prefix = Path(output_prefix_text[: -len(generated_suffix)])

    start = time.time()
    subprocess.run(
        [
            "python3",
            str(preprocess_script),
            "--input",
            str(dataset_json),
            "--tokenizer-type",
            tokenizer_type,
            "--tokenizer-model",
            tokenizer_model,
            "--output-prefix",
            str(preprocess_prefix),
            "--workers",
            str(os.cpu_count()),
            "--split-sentences",
            "--partitions",
            "2",
        ],
        check=True,
        env=env,
    )
    if needs_rename:
        generated_prefix = Path(f"{preprocess_prefix}{generated_suffix}")
        for suffix in (".bin", ".idx"):
            generated_file = Path(f"{generated_prefix}{suffix}")
            if not generated_file.exists():
                raise RuntimeError(f"Preprocessing did not create expected file: {generated_file}")
            generated_file.replace(Path(f"{output_prefix}{suffix}"))
    log_info(f"Preprocessing of {output_prefix.name} completed in {int(time.time() - start)} s")


def download_bookcorpus(dataset_json: Path):
    if dataset_json.exists():
        log_info(f"Found dataset file: {dataset_json}, skipping download.")
        return
    log_info(f"Downloading and saving BookCorpus dataset to {dataset_json} ...")
    nltk.download("punkt")
    dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
    dataset.to_json(str(dataset_json))
    log_info("Download and save completed.")


def download_bookcorpus_split(train_json: Path, valid_json: Path, test_size: float, seed: int):
    if train_json.exists() and valid_json.exists():
        log_info(f"Found dataset files: {train_json} and {valid_json}, skipping download.")
        return
    log_info(f"Downloading BookCorpus and splitting into {train_json} / {valid_json} ...")
    nltk.download("punkt")
    dataset = load_dataset("bookcorpus", split="train", trust_remote_code=True)
    splits = dataset.train_test_split(test_size=test_size, seed=seed)
    splits["train"].to_json(str(train_json))
    splits["test"].to_json(str(valid_json))
    log_info("Download and split completed.")


def prepare_dataset(
    data_path: Path,
    tokenizer_type: str,
    tokenizer_model: str,
    tokenized_data_path: Path,
    tokenized_eval_data_path: Path | None = None,
    test_size: float = 0.005,
    seed: int = 42,
    env=None,
):
    dataset_path = data_path / "bookcorpus"
    output_path = dataset_path / tokenizer_type
    hf_home = Path(os.environ.get("HF_HOME", data_path / "huggingface"))
    os.environ["HF_HOME"] = str(hf_home)

    train_exists = tokenized_files_exist(tokenized_data_path)
    eval_exists = tokenized_eval_data_path is not None and tokenized_files_exist(tokenized_eval_data_path)
    metadata_path = (
        split_metadata_path(tokenized_eval_data_path) if tokenized_eval_data_path is not None else None
    )
    split_matches = metadata_path is not None and split_metadata_matches(metadata_path, test_size, seed)

    if train_exists and (tokenized_eval_data_path is None or (eval_exists and split_matches)):
        log_info("All required tokenized files exist, skipping preprocessing.")
        return

    output_path.mkdir(parents=True, exist_ok=True)
    log_info(f"Preprocessing dataset with tokenizer {tokenizer_type} / {tokenizer_model}")

    if tokenized_eval_data_path is None:
        dataset_json = dataset_path / "bookcorpus_megatron.json"
        download_bookcorpus(dataset_json)
        run_preprocess(dataset_json, tokenized_data_path, tokenizer_type, tokenizer_model, env)
        return

    # Validation is requested: BookCorpus ships a single 'train' split, so carve
    # out a held-out slice here. Megatron's --split cannot do it for us because
    # Primus always passes an explicit --train_data_path, and Megatron rejects
    # --split together with per-split blends (blended_megatron_dataset_config.py).
    cache_key = split_cache_key(test_size, seed)
    train_json = dataset_path / f"bookcorpus_train_{cache_key}.json"
    valid_json = dataset_path / f"bookcorpus_valid_{cache_key}.json"
    download_bookcorpus_split(train_json, valid_json, test_size, seed)

    if not split_matches:
        if train_exists or eval_exists:
            log_info(
                "Existing validation tokenized files use different or unknown split metadata; "
                "regenerating train and evaluation files."
            )
        remove_tokenized_files(tokenized_data_path)
        remove_tokenized_files(tokenized_eval_data_path)
        train_exists = False
        eval_exists = False

    if train_exists:
        log_info("Train tokenized files already exist, only generating the evaluation files.")
    else:
        run_preprocess(train_json, tokenized_data_path, tokenizer_type, tokenizer_model, env)
    if not eval_exists:
        run_preprocess(valid_json, tokenized_eval_data_path, tokenizer_type, tokenizer_model, env)
    write_split_metadata(metadata_path, test_size, seed)


EVAL_DATA_PATH_KEYS = ("valid_data_path", "test_data_path")


def configured_eval_data_path_keys(pre_trainer_cfg) -> set[str]:
    """Evaluation data keys the experiment config already points at itself."""
    return {key for key in EVAL_DATA_PATH_KEYS if getattr(pre_trainer_cfg, key, None) is not None}


def validation_dataset_requested(pre_trainer_cfg) -> bool:
    """Whether the run needs a held-out dataset that Primus has to build itself."""
    # Either key means the user brought their own evaluation data, and the
    # generated split would end up overriding it on the command line.
    if configured_eval_data_path_keys(pre_trainer_cfg):
        return False
    if getattr(pre_trainer_cfg, "eval_interval", 0) <= 0:
        return False
    return (
        bool(getattr(pre_trainer_cfg, "full_validation", False))
        or (getattr(pre_trainer_cfg, "eval_iters", 0) or 0) > 0
    )


def dataset_coordination_paths(
    tokenized_data_path: Path,
    tokenized_eval_data_path: Path | None,
    tokenizer_type: str,
    tokenizer_model: str,
    test_size: float,
    seed: int,
) -> tuple[Path, Path]:
    marker_prefix = tokenized_eval_data_path or tokenized_data_path
    payload = {
        "eval_path": str(tokenized_eval_data_path) if tokenized_eval_data_path else None,
        "seed": seed if tokenized_eval_data_path else None,
        "test_size": test_size if tokenized_eval_data_path else None,
        "tokenizer_model": str(tokenizer_model) if tokenizer_model is not None else None,
        "tokenizer_type": tokenizer_type,
        "train_path": str(tokenized_data_path),
    }
    key = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()[:12]
    marker_base = Path(f"{marker_prefix}.primus_prepare_{key}")
    return Path(f"{marker_base}.done"), Path(f"{marker_base}.failed")


def required_tokenized_files_exist(tokenized_data_path: Path, tokenized_eval_data_path: Path | None) -> bool:
    return tokenized_files_exist(tokenized_data_path) and (
        tokenized_eval_data_path is None or tokenized_files_exist(tokenized_eval_data_path)
    )


def dataset_cache_ready(
    tokenized_data_path: Path,
    tokenized_eval_data_path: Path | None,
    test_size: float,
    seed: int,
) -> bool:
    if not required_tokenized_files_exist(tokenized_data_path, tokenized_eval_data_path):
        return False
    return tokenized_eval_data_path is None or split_metadata_matches(
        split_metadata_path(tokenized_eval_data_path), test_size, seed
    )


def wait_for_dataset(done_flag: Path, failed_flag: Path):
    try:
        timeout = float(os.environ.get("PRIMUS_DATA_PREP_TIMEOUT_SECONDS", "3600"))
        poll_interval = float(os.environ.get("PRIMUS_DATA_PREP_POLL_SECONDS", "30"))
    except ValueError:
        log_error_and_exit(
            "PRIMUS_DATA_PREP_TIMEOUT_SECONDS and PRIMUS_DATA_PREP_POLL_SECONDS must be numeric."
        )
    if timeout <= 0 or poll_interval <= 0:
        log_error_and_exit(
            "PRIMUS_DATA_PREP_TIMEOUT_SECONDS and PRIMUS_DATA_PREP_POLL_SECONDS must be positive."
        )

    log_info(
        f"Waiting up to {timeout:g}s for rank 0 dataset preparation marker {done_flag}. "
        "The tokenized output directory must be on storage shared by every node."
    )
    deadline = time.monotonic() + timeout
    while True:
        if done_flag.exists():
            return
        if failed_flag.exists():
            try:
                failure = failed_flag.read_text(encoding="utf-8").strip()
            except OSError:
                failure = "rank 0 reported an unspecified failure"
            log_error_and_exit(f"Rank 0 dataset preparation failed: {failure}")
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            log_error_and_exit(
                f"Timed out waiting for dataset preparation after {timeout:g}s: {done_flag}. "
                "Verify that rank 0 is running and TOKENIZED_*_DATA_PATH points to shared storage."
            )
        sleep(min(poll_interval, remaining))


def prepare_dataset_if_needed(
    primus_config: PrimusConfig,
    data_path: Path,
    test_size: float = 0.005,
    seed: int = 42,
    env=None,
):
    pre_trainer_cfg = primus_config.get_module_config("pre_trainer")

    # Skip dataset preparation if train_data_path is explicitly set
    if pre_trainer_cfg.train_data_path is not None:
        return

    # SFT configs only declare post_trainer, but get_module_config() aliases
    # pre_trainer to it, so `train pretrain --config <sft.yaml>` lands here too.
    # SFT builds its dataset inside the trainer (HF datasets / sft_dataset_name),
    # so the bookcorpus flow below would only demand a pointless HF_TOKEN and
    # download a corpus that never gets read.
    if getattr(pre_trainer_cfg, "stage", None) == "sft":
        log_info(
            "stage=sft detected, skipping bookcorpus preparation "
            "(the SFT dataset is loaded inside the trainer)."
        )
        return

    # Check if this is a diffusion model (uses Energon datasets, not tokenized datasets)
    model_type = getattr(pre_trainer_cfg, "model_type", None)
    trainer_class = getattr(pre_trainer_cfg, "trainer_class", None)
    data_path_config = getattr(pre_trainer_cfg, "data_path", None)

    # Determine if this is a diffusion model
    is_diffusion = False
    if model_type == "diffusion_model":
        is_diffusion = True
    elif trainer_class and ("Flux" in str(trainer_class) or "Diffusion" in str(trainer_class)):
        is_diffusion = True
    elif hasattr(model_type, "name") and "DIFFUSION" in model_type.name:
        is_diffusion = True

    # For diffusion models with data_path set, skip tokenization (they use Energon)
    if is_diffusion and data_path_config:
        log_info("=" * 80)
        log_info("Diffusion model detected with data_path configured.")
        log_info("Skipping tokenization (diffusion models use Energon datasets).")
        log_info(f"Data will be loaded from: {data_path_config}")
        log_info("=" * 80)
        return

    # For language models, proceed with tokenization
    tokenizer_type = getattr(pre_trainer_cfg, "tokenizer_type", None)
    if not tokenizer_type:
        log_info("No tokenizer_type found, skipping dataset preparation.")
        return

    tokenizer_model = getattr(pre_trainer_cfg, "tokenizer_model", None)
    bookcorpus_root = Path(data_path) / f"bookcorpus/{tokenizer_type}"

    # Without validation the whole corpus is the training set, and the tokenized
    # prefix stays 'bookcorpus_text_sentence' so existing caches keep working.
    # With validation the corpus is split, so train/eval get their own prefixes.
    tokenized_eval_data_path = None
    if validation_dataset_requested(pre_trainer_cfg):
        default_tokenized_path = bookcorpus_root / "bookcorpus_train_text_sentence"
        tokenized_data_path = Path(
            os.environ.get(
                "TOKENIZED_TRAIN_DATA_PATH",
                os.environ.get("TOKENIZED_DATA_PATH", str(default_tokenized_path)),
            )
        )
        tokenized_eval_data_path = Path(
            os.environ.get("TOKENIZED_EVAL_DATA_PATH", str(bookcorpus_root / "bookcorpus_eval_text_sentence"))
        )
    else:
        default_tokenized_path = bookcorpus_root / "bookcorpus_text_sentence"
        tokenized_data_path = Path(os.environ.get("TOKENIZED_DATA_PATH", str(default_tokenized_path)))

    done_flag, failed_flag = dataset_coordination_paths(
        tokenized_data_path,
        tokenized_eval_data_path,
        tokenizer_type,
        tokenizer_model,
        test_size,
        seed,
    )
    node_rank = get_node_rank()

    if node_rank == 0:
        failed_flag.unlink(missing_ok=True)
        cache_ready = dataset_cache_ready(tokenized_data_path, tokenized_eval_data_path, test_size, seed)
        if done_flag.exists() and not cache_ready:
            log_info(f"Ignoring stale dataset completion marker: {done_flag}")
            done_flag.unlink()
        if not done_flag.exists():
            if cache_ready:
                atomic_write(done_flag, "ok\n")
                log_info("All required tokenized files already exist; recording completion marker.")
            else:
                try:
                    hf_token = os.environ.get("HF_TOKEN")
                    if not hf_token:
                        log_error_and_exit("Environment variable HF_TOKEN must be set.")

                    if not tokenizer_model:
                        log_error_and_exit(
                            "tokenizer_model not found in configuration. "
                            "This is required for language model tokenization."
                        )

                    log_info(f"TOKENIZED_TRAIN_DATA_PATH is {tokenized_data_path}")
                    if tokenized_eval_data_path is not None:
                        log_info(f"TOKENIZED_EVAL_DATA_PATH is {tokenized_eval_data_path}")

                    prepare_dataset(
                        data_path=data_path,
                        tokenizer_type=tokenizer_type,
                        tokenizer_model=tokenizer_model,
                        tokenized_data_path=tokenized_data_path,
                        tokenized_eval_data_path=tokenized_eval_data_path,
                        test_size=test_size,
                        seed=seed,
                        env=env,
                    )
                    if not dataset_cache_ready(
                        tokenized_data_path, tokenized_eval_data_path, test_size, seed
                    ):
                        raise RuntimeError(
                            "Dataset preparation completed without all expected files or split metadata."
                        )
                except BaseException as error:
                    atomic_write(failed_flag, f"{type(error).__name__}: {error}\n")
                    raise
                atomic_write(done_flag, "ok\n")
                log_info("Dataset preparation completed.")
    else:
        wait_for_dataset(done_flag, failed_flag)
        if not dataset_cache_ready(tokenized_data_path, tokenized_eval_data_path, test_size, seed):
            log_error_and_exit(
                f"Dataset completion marker exists but expected tokenized files are missing: {done_flag}"
            )

    # Expose the resolved dataset paths to the caller (e.g., primus-cli direct)
    # via generic extra.* lines on stdout, which will be converted to:
    #   --train_data_path <tokenized_data_path> [--valid_data_path ...]
    print(f"extra.train_data_path={tokenized_data_path}")
    if tokenized_eval_data_path is not None:
        # extra.* lines become CLI arguments, which outrank the YAML, so only
        # advertise the keys the config left unset.
        configured = configured_eval_data_path_keys(pre_trainer_cfg)
        for key in EVAL_DATA_PATH_KEYS:
            if key not in configured:
                print(f"extra.{key}={tokenized_eval_data_path}")


def resolve_megatron_path_for_helper(primus_path: Path, backend_path: str | None) -> Path:
    """Resolve Megatron path for C++ helper build: CLI > BACKEND_PATH > MEGATRON_PATH > default."""
    if backend_path:
        path = Path(backend_path).resolve()
        log_info(f"Using backend_path from argument: {path}")
        return path

    env_backend = get_env_case_insensitive("BACKEND_PATH")
    if env_backend:
        path = Path(env_backend).resolve()
        log_info(f"Using backend_path from BACKEND_PATH environment: {path}")
        return path

    env_backend = get_env_case_insensitive("MEGATRON_PATH")
    if env_backend:
        path = Path(env_backend).resolve()
        log_info(f"Using backend_path from MEGATRON_PATH environment: {path}")
        return path

    path = default_backend_path(primus_path, "Megatron-LM")
    log_info(f"No backend_path provided, falling back to: {path}")
    return path


def build_megatron_helper(megatron_path: Path):
    """Build Megatron's helper C++ dataset library."""
    # Expose resolved backend_path to the caller (e.g., primus-cli direct)
    # via a generic extra.* line on stdout, which will be converted to:
    #   --backend_path <megatron_path>
    print(f"extra.backend_path={megatron_path}")

    check_dir_nonempty(megatron_path, "megatron")

    # build C++ helper
    dataset_cpp_dir = megatron_path / "megatron/core/datasets"
    log_info(f"Building Megatron dataset helper in {dataset_cpp_dir}")

    ret = subprocess.run(["make"], cwd=dataset_cpp_dir)
    if ret.returncode != 0:
        log_error_and_exit("Building Megatron C++ helper failed.")


# ---------- Main ----------
def parse_args():
    """Parse the hook's own arguments, leaving everything else for training.

    Unknown arguments are returned rather than rejected because primus-cli hands
    this hook the full training argument list. Anything named here is consumed and
    stops reaching the config overrides, which is why no argument may share a name
    with a training parameter.
    """
    parser = argparse.ArgumentParser(description="Prepare Primus environment")
    parser.add_argument("--primus_path", type=str, required=True, help="Root path to the Primus project")
    parser.add_argument("--data_path", type=str, required=True, help="Path to data directory")
    parser.add_argument("--config", type=str, required=True, help="Path to experiment YAML config")
    parser.add_argument(
        "--test_size", type=float, default=0.005, help="Held-out fraction of the train/valid split"
    )
    parser.add_argument(
        "--split_seed",
        type=int,
        default=DEFAULT_SPLIT_SEED,
        help=(
            f"Seed for the train/valid split (default {DEFAULT_SPLIT_SEED}). "
            "Distinct from the Megatron --seed training parameter, which this "
            "hook does not read."
        ),
    )
    parser.add_argument(
        "--patch_args",
        type=str,
        default="/tmp/primus_patch_args.txt",
        help="Reserved for runner hook interface compatibility",
    )
    parser.add_argument(
        "--backend_path",
        type=str,
        default=None,
        help="Optional path to backend (e.g., Megatron), will be added to PYTHONPATH",
    )
    return parser.parse_known_args()


def main():
    args, unknown = parse_args()

    log_info(f"BACKEND_PATH {args.backend_path}")
    # primus_config = PrimusParser().parse(args)
    primus_config, _ = load_primus_config(args, unknown)

    primus_path = Path(args.primus_path).resolve()
    log_info(f"PRIMUS_PATH is set to: {primus_path}")

    data_path = Path(args.data_path).resolve()
    log_info(f"DATA_PATH is set to: {data_path}")

    exp_path = Path(args.config).resolve()
    if not exp_path.is_file():
        log_error_and_exit(f"The specified EXP file does not exist: {exp_path}")
    log_info(f"EXP is set to: {exp_path}")

    log_info(f"PATCH-ARGS is set to: {Path(args.patch_args).resolve()} (unused in megatron prepare hook)")

    build_backend_path = resolve_megatron_path_for_helper(primus_path, args.backend_path)
    used_backend_path = setup_backend_path(framework="megatron", backend_path=args.backend_path, verbose=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = f"{used_backend_path}:{env.get('PYTHONPATH', '')}"

    mock_data = primus_config.get_module_config("pre_trainer").mock_data
    if mock_data:
        log_info(f"'mock_data: true', Skipping dataset preparation.")
    else:
        prepare_dataset_if_needed(
            primus_config=primus_config,
            data_path=data_path,
            test_size=args.test_size,
            seed=args.split_seed,
            env=env,
        )

    build_megatron_helper(megatron_path=build_backend_path)


if __name__ == "__main__":
    log_info("========== Prepare Megatron dataset ==========")
    main()
