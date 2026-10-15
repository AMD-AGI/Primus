#!/usr/bin/env python3
###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Load, validate and resolve the benchmark config (YAML file + command-line overrides).

Used by run_benchmarks.sh:
  config.py [options] --detected-gpu mi300 --emit shell   -> bash variable assignments (eval'd)
  config.py [options] --detected-gpu mi300 --emit yaml    -> resolved plan (saved with the results)
"""
import argparse
import difflib
import os
import shlex
import sys

try:
    import yaml
except ImportError:
    sys.exit(
        "ERROR: PyYAML is required on the host: pip install --user pyyaml  (or apt install python3-yaml)"
    )

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
GPUS = ("mi300", "mi355")
MODELS = ("flux-dev", "stable-diffusion-xl", "mochi-1", "hunyuan-video", "wan2_1-i2v")
ALIASES = {
    "flux": "flux-dev",
    "flux.1-dev": "flux-dev",
    "sdxl": "stable-diffusion-xl",
    "mochi": "mochi-1",
    "mochi1": "mochi-1",
    "hunyuan": "hunyuan-video",
    "hunyuanvideo": "hunyuan-video",
    "wan": "wan2_1-i2v",
    "wan2.1": "wan2_1-i2v",
    "wan2_1": "wan2_1-i2v",
    "wan2.1-i2v": "wan2_1-i2v",
}
DEFAULT_BS = {
    "flux-dev": {"mi300": 10, "mi355": 16},
    "stable-diffusion-xl": {"mi300": 20, "mi355": 12},
    "mochi-1": {"mi300": 1, "mi355": 4},
    "hunyuan-video": {"mi300": 1, "mi355": 3},
    "wan2_1-i2v": {"mi300": 1, "mi355": 1},
}
DLRM_PRECISIONS = ("TF32", "FP32")

TOP_KEYS = {
    "gpu",
    "image",
    "hf_cache_dir",
    "hf_cache_readonly",
    "download",
    "output_dir",
    "step_timeout",
    "download_timeout",
    "keep_container",
    "diffusion",
    "dlrm",
}
DIFFUSION_KEYS = {"enabled", "train_args", "hydra_overrides", "models"}
MODEL_KEYS = {"enabled", "batch_size", "train_args", "hydra_overrides"}
DLRM_KEYS = {"enabled", "image", "precisions", "batch_size"}

HF_HOME_IN_CONTAINER = "/root/.cache/huggingface"


class ConfigError(Exception):
    pass


def canon_model(name):
    name = str(name).strip()
    return ALIASES.get(name.lower(), name)


def check_keys(section, allowed, where):
    if section is None:
        return
    if not isinstance(section, dict):
        raise ConfigError(f"'{where}' must be a mapping")
    for key in section:
        if key not in allowed:
            hint = difflib.get_close_matches(str(key), allowed, n=1)
            raise ConfigError(
                f"Unknown key '{key}' in {where}"
                + (f" (did you mean '{hint[0]}'?)" if hint else "")
                + f". Allowed: {', '.join(sorted(allowed))}"
            )


def per_gpu(value, gpu, where):
    """Resolve {mi300: x, mi355: y} mappings to the value for `gpu`."""
    if isinstance(value, dict) and value and set(value) <= set(GPUS):
        if gpu not in value:
            raise ConfigError(f"{where}: no value for gpu '{gpu}'")
        return value[gpu]
    return value


def as_int_list(value, where):
    values = value if isinstance(value, list) else [value]
    out = []
    for v in values:
        try:
            iv = int(v)
        except (TypeError, ValueError):
            raise ConfigError(f"{where}: expected an integer or a list of integers, got {v!r}")
        if iv <= 0:
            raise ConfigError(f"{where}: batch size must be positive, got {iv}")
        out.append(iv)
    return out


def as_str_list(value, where):
    if value is None:
        return []
    if isinstance(value, str):
        return shlex.split(value)
    if isinstance(value, list):
        return [str(v) for v in value]
    raise ConfigError(f"{where}: expected a list of strings")


def train_args_overrides(train_args, gpu, where):
    if not train_args:
        return {}
    if not isinstance(train_args, dict):
        raise ConfigError(f"{where}: expected a mapping of train_args")
    return {str(k): per_gpu(v, gpu, f"{where}.{k}") for k, v in train_args.items()}


def resolve_path(path, base_dir):
    path = os.path.expanduser(os.path.expandvars(str(path)))
    if not os.path.isabs(path):
        path = os.path.join(base_dir, path)
    return os.path.normpath(path)


def hf_mount_target(hf_dir):
    """Decide whether hf_dir is an HF home (contains hub/) or a hub directory itself."""
    if os.path.basename(hf_dir.rstrip("/")) == "hub":
        return HF_HOME_IN_CONTAINER + "/hub"
    if os.path.isdir(hf_dir) and not os.path.isdir(os.path.join(hf_dir, "hub")):
        entries = os.listdir(hf_dir)
        if any(e.startswith(("models--", "datasets--")) for e in entries):
            return HF_HOME_IN_CONTAINER + "/hub"
    return HF_HOME_IN_CONTAINER


def parse_args(argv):
    p = argparse.ArgumentParser(
        prog="run_benchmarks.sh",
        description="Run AMDiffusionBenchmark + DLRMBenchmark end to end. Settings come from the YAML "
        "config; command-line options override it for a single run.",
    )
    p.add_argument(
        "-c",
        "--config",
        default=os.path.join(SCRIPT_DIR, "config.yaml"),
        help="YAML config file (default: config.yaml next to the script)",
    )
    p.add_argument("--gpu", choices=("auto",) + GPUS, help="Override target GPU")
    p.add_argument("--image", help="Override container image")
    p.add_argument("--dlrm-image", help="Override DLRM container image")
    p.add_argument(
        "--models", help="Comma separated diffusion models to run (overrides 'enabled'), or 'none'"
    )
    p.add_argument(
        "--bs",
        action="append",
        default=[],
        metavar="MODEL=N[,N...]",
        help="Override batch size(s) for a model; repeatable (e.g. --bs flux=12 --bs sdxl=16,20)",
    )
    p.add_argument("--num-iterations", type=int, help="Override train_args.num_iterations for all models")
    p.add_argument(
        "--extra-train-args", default="", help="Extra raw hydra overrides for every launcher.py call"
    )
    p.add_argument("--dlrm-precisions", help="Comma separated DLRM precisions, or 'none'")
    p.add_argument("--dlrm-batch-size", type=int, help="Override DLRM BATCH_SIZE")
    p.add_argument("--hf-cache", help="Override hf_cache_dir")
    p.add_argument("--hf-cache-readonly", action="store_true", help="Mount the HF cache read-only")
    p.add_argument("--skip-download", action="store_true", help="Skip the Hugging Face download step")
    p.add_argument("--output-dir", help="Override output_dir")
    p.add_argument("--timeout", type=int, help="Override step_timeout (seconds, 0 = none)")
    p.add_argument("--keep-container", action="store_true", help="Keep the container after the run")
    p.add_argument(
        "--label",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Docker label for the benchmark containers; repeatable (the perf batch runner "
        "finds and stops its containers by label)",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Start the container and run launcher.py with launcher.dry_run=true; "
        "skip downloads and DLRM (no GPU work)",
    )
    p.add_argument("--parse-only", metavar="RUN_DIR", help="Only (re)parse an existing results directory")
    p.add_argument("--detected-gpu", default="unknown", help=argparse.SUPPRESS)
    p.add_argument("--emit", choices=("shell", "yaml"), default="shell", help=argparse.SUPPRESS)
    return p.parse_args(argv)


def resolve(args):
    config_file = os.path.abspath(os.path.expanduser(args.config))
    if not os.path.isfile(config_file):
        raise ConfigError(f"Config file not found: {config_file}")
    with open(config_file) as f:
        cfg = yaml.safe_load(f) or {}
    base_dir = os.path.dirname(config_file)

    check_keys(cfg, TOP_KEYS, "config")
    diff_cfg = cfg.get("diffusion") or {}
    dlrm_cfg = cfg.get("dlrm") or {}
    check_keys(diff_cfg, DIFFUSION_KEYS, "diffusion")
    check_keys(dlrm_cfg, DLRM_KEYS, "dlrm")
    models_cfg = diff_cfg.get("models") or {}
    check_keys(models_cfg, set(MODELS), "diffusion.models")
    for name, mcfg in models_cfg.items():
        check_keys(mcfg or {}, MODEL_KEYS, f"diffusion.models.{name}")

    # GPU
    gpu = args.gpu or str(cfg.get("gpu", "auto")).lower()
    if gpu == "auto":
        gpu = args.detected_gpu
        if gpu not in GPUS:
            raise ConfigError("Could not auto-detect the GPU type; set 'gpu: mi300|mi355' or pass --gpu")
    if gpu not in GPUS:
        raise ConfigError(f"gpu must be auto, mi300 or mi355 (got '{gpu}')")

    image = args.image or cfg.get("image")
    if not image:
        raise ConfigError("'image' is required")

    # HF cache
    hf_cache = args.hf_cache or cfg.get("hf_cache_dir") or os.environ.get("HF_HOME") or "~/.cache/huggingface"
    hf_cache = resolve_path(hf_cache, os.getcwd() if args.hf_cache else base_dir)
    hf_readonly = bool(args.hf_cache_readonly or cfg.get("hf_cache_readonly", False))

    # Diffusion selection
    if args.models:
        if args.models.strip().lower() == "none":
            selected = []
        else:
            selected = [canon_model(m) for m in args.models.split(",") if m.strip()]
    elif diff_cfg.get("enabled", True):
        selected = [m for m in MODELS if (models_cfg.get(m) or {}).get("enabled", m in models_cfg)]
    else:
        selected = []
    for m in selected:
        if m not in MODELS:
            raise ConfigError(f"Unknown model '{m}'. Valid: {', '.join(MODELS)}")

    bs_override = {}
    for item in args.bs:
        if "=" not in item:
            raise ConfigError(f"--bs expects MODEL=N[,N...], got '{item}'")
        name, values = item.split("=", 1)
        bs_override[canon_model(name)] = as_int_list(values.split(","), f"--bs {item}")

    global_train_args = train_args_overrides(diff_cfg.get("train_args"), gpu, "diffusion.train_args")
    if args.num_iterations:
        global_train_args["num_iterations"] = args.num_iterations
    global_hydra = as_str_list(diff_cfg.get("hydra_overrides"), "diffusion.hydra_overrides")

    runs = []
    for m in selected:
        mcfg = models_cfg.get(m) or {}
        if m in bs_override:
            batch_sizes = bs_override[m]
        else:
            bs_value = per_gpu(mcfg.get("batch_size", DEFAULT_BS[m]), gpu, f"diffusion.models.{m}.batch_size")
            batch_sizes = as_int_list(bs_value, f"diffusion.models.{m}.batch_size")
        train_args = dict(global_train_args)
        train_args.update(
            train_args_overrides(mcfg.get("train_args"), gpu, f"diffusion.models.{m}.train_args")
        )
        overrides = [f"train_args.{k}={v}" for k, v in train_args.items()]
        overrides += global_hydra
        overrides += as_str_list(mcfg.get("hydra_overrides"), f"diffusion.models.{m}.hydra_overrides")
        overrides += shlex.split(args.extra_train_args)
        for bs in batch_sizes:
            runs.append({"model": m, "batch_size": bs, "overrides": overrides})

    # DLRM
    if args.dlrm_precisions:
        precs = (
            []
            if args.dlrm_precisions.strip().lower() == "none"
            else [p.strip().upper() for p in args.dlrm_precisions.split(",") if p.strip()]
        )
    elif dlrm_cfg.get("enabled", True):
        precs = [str(p).upper() for p in (dlrm_cfg.get("precisions") or list(DLRM_PRECISIONS))]
    else:
        precs = []
    for p in precs:
        if p not in DLRM_PRECISIONS:
            raise ConfigError(f"Unknown DLRM precision '{p}'. Valid: {', '.join(DLRM_PRECISIONS)}")
    dlrm_bs = args.dlrm_batch_size or per_gpu(dlrm_cfg.get("batch_size"), gpu, "dlrm.batch_size")
    if dlrm_bs is not None:
        dlrm_bs = as_int_list(dlrm_bs, "dlrm.batch_size")[0]
    dlrm_image = args.dlrm_image or dlrm_cfg.get("image") or image

    if not runs and not precs:
        raise ConfigError("Nothing to run: all diffusion models and DLRM are disabled")

    download = bool(cfg.get("download", True)) and not args.skip_download and not args.dry_run
    if hf_readonly:
        download = False

    output_dir = (
        resolve_path(args.output_dir, os.getcwd())
        if args.output_dir
        else resolve_path(cfg.get("output_dir", "./results"), base_dir)
    )

    step_timeout = args.timeout if args.timeout is not None else int(cfg.get("step_timeout", 14400))

    for label in args.label:
        if "=" not in label or not label.split("=", 1)[0]:
            raise ConfigError(f"--label expects KEY=VALUE, got '{label}'")

    return {
        "config_file": config_file,
        "gpu": gpu,
        "image": image,
        "dlrm_image": dlrm_image,
        "hf_cache_dir": hf_cache,
        "hf_mount_target": hf_mount_target(hf_cache),
        "hf_cache_readonly": hf_readonly,
        "download": download,
        "output_dir": output_dir,
        "step_timeout": step_timeout,
        "download_timeout": int(cfg.get("download_timeout", 0)),
        "keep_container": bool(args.keep_container or cfg.get("keep_container", False)),
        "dry_run": bool(args.dry_run),
        "diffusion_models": list(dict.fromkeys(r["model"] for r in runs)),
        "diffusion_runs": runs,
        "dlrm_precisions": precs,
        "dlrm_batch_size": dlrm_bs,
        "labels": list(args.label),
    }


def emit_shell(plan):
    def q(v):
        if isinstance(v, bool):
            v = int(v)
        return shlex.quote("" if v is None else str(v))

    def arr(values):
        return "(" + " ".join(shlex.quote(v) for v in values) + ")"

    lines = [
        f"CONFIG_FILE={q(plan['config_file'])}",
        f"GPU={q(plan['gpu'])}",
        f"IMAGE={q(plan['image'])}",
        f"DLRM_IMAGE={q(plan['dlrm_image'])}",
        f"HF_CACHE_DIR={q(plan['hf_cache_dir'])}",
        f"HF_MOUNT_TARGET={q(plan['hf_mount_target'])}",
        f"HF_READONLY={q(plan['hf_cache_readonly'])}",
        f"DOWNLOAD={q(plan['download'])}",
        f"OUTPUT_ROOT={q(plan['output_dir'])}",
        f"STEP_TIMEOUT={q(plan['step_timeout'])}",
        f"DOWNLOAD_TIMEOUT={q(plan['download_timeout'])}",
        f"KEEP_CONTAINER={q(plan['keep_container'])}",
        f"DRY_RUN={q(plan['dry_run'])}",
        f"DIFF_MODELS={arr(plan['diffusion_models'])}",
        # one element per run: model<TAB>batch_size<TAB>override...
        "DIFF_RUNS="
        + arr("\t".join([r["model"], str(r["batch_size"])] + r["overrides"]) for r in plan["diffusion_runs"]),
        f"DLRM_PRECS={arr(plan['dlrm_precisions'])}",
        f"DLRM_BATCH_SIZE={q(plan['dlrm_batch_size'])}",
        f"DOWNLOAD_ALL={q(sorted(plan['diffusion_models']) == sorted(MODELS))}",
        f"LABELS={arr(plan['labels'])}",
    ]
    return "\n".join(lines)


def main():
    args = parse_args(sys.argv[1:])
    if args.parse_only:
        print(f"PARSE_ONLY={shlex.quote(os.path.abspath(args.parse_only))}")
        return
    try:
        plan = resolve(args)
    except ConfigError as e:
        sys.exit(f"CONFIG ERROR: {e}")
    if args.emit == "yaml":
        print(yaml.safe_dump(plan, sort_keys=False), end="")
    else:
        print(emit_shell(plan))


if __name__ == "__main__":
    main()
