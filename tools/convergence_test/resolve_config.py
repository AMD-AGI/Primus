###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Resolve a convergence config the way the trainer will see it.

Shared by the driver, the linter and the dataset builder so that all three agree
on the backend, the effective hyper-parameters, and where the dataset lives.

For Megatron the effective config is PrimusParser's merged module config. MaxText
needs one more step: Primus only carries a thin overlay, and MaxText merges it
with its own ``configs/base.yml`` and ``configs/models/<model_name>.yml`` at
start-up. The precedence is not what the YAML suggests. Primus hands the overlay
to MaxText as the config *file*, so the model file is layered on top of it, and
only ``override_model`` entries (forwarded as kwargs) beat the model file. A
plain override of an architecture key such as ``vocab_size`` is silently
discarded. This module replays that merge with plain YAML (no JAX needed), so
the host sees the values MaxText will actually train with.

Printing the plan the driver evaluates:
    python3 tools/convergence_test/resolve_config.py --config <exp.yaml> [--key value ...]
"""

import argparse
import contextlib
import glob
import hashlib
import os
import re
import shlex
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

PRIMUS_PATH = Path(__file__).resolve().parents[2]

# A couple of model presets name tokenizer repos that do not exist: there is no
# meta-llama/Meta-Llama-3.2-{1B,3B}, the real repos drop the "Meta-" prefix, and
# meta-llama/Llama-2-7b holds Meta's original checkpoint format, which
# AutoTokenizer cannot load. Nobody notices because mock/synthetic data never
# loads a tokenizer. These are corrections, not a way around gating -- the
# corrected repos are still gated and still need HF_TOKEN.
TOKENIZER_NAME_FIXES = {
    "meta-llama/Meta-Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "meta-llama/Meta-Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
    "meta-llama/Llama-2-7b": "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-70b": "meta-llama/Llama-2-70b-hf",
}

# Params MaxTextPretrainTrainer strips before calling pyconfig.initialize, plus
# override_model, which it forwards as kwargs instead of through the file.
MAXTEXT_NOT_IN_FILE = (
    "file_sink_level",
    "stderr_sink_level",
    "sink_level",
    "trainable",
    "model",
    "override_model",
)

FRAMEWORKS = ("megatron", "maxtext")

# Corpora known by name. `valid_split` names an official held-out split to
# validate on; without one the validation documents are taken from the head of
# the training stream.
SOURCES = {
    "fineweb-edu": {"repo": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "c4": {"repo": "allenai/c4", "name": "en", "split": "train", "valid_split": "validation"},
    "wikitext103": {"repo": "Salesforce/wikitext", "name": "wikitext-103-raw-v1", "split": "train"},
}

# `datasets` builder for each local file type.
LOCAL_FORMATS = {".jsonl": "json", ".json": "json", ".parquet": "parquet", ".txt": "text"}

SOURCE_HELP = (
    "a corpus name ({names}), a Hugging Face dataset <owner>/<name>[:<subset>], "
    "or local {exts} files given as a path, directory or glob starting with /, ./, ../ or ~"
).format(names=", ".join(SOURCES), exts="/".join(LOCAL_FORMATS))


def _slug(text):
    return re.sub(r"[^a-z0-9.]+", "-", text.lower()).strip("-")


def parse_source(source):
    """Describe a ``--source``: where its documents come from and a directory-safe tag.

    Hub and local sources have no held-out split, so, like fineweb-edu, they are
    validated on the head of their training stream.
    """
    if source in SOURCES:
        return {**SOURCES[source], "kind": "hub", "tag": source}

    if source.startswith(("/", ".", "~")):
        pattern = os.path.abspath(os.path.expanduser(source))
        if os.path.isdir(pattern):
            pattern = os.path.join(pattern, "*")
        files = sorted(f for f in glob.glob(pattern) if os.path.splitext(f)[1] in LOCAL_FORMATS)
        if not files:
            raise ValueError(f"--source {source}: no {'/'.join(LOCAL_FORMATS)} files match {pattern}")
        builders = {LOCAL_FORMATS[os.path.splitext(f)[1]] for f in files}
        if len(builders) > 1:
            raise ValueError(f"--source {source}: mixes file types ({', '.join(sorted(builders))})")
        base = os.path.basename(pattern)
        if any(c in base for c in "*?["):
            base = os.path.basename(os.path.dirname(pattern))
        stem = _slug(os.path.splitext(base)[0]) or "files"
        digest = hashlib.sha1(pattern.encode()).hexdigest()[:8]
        return {
            "kind": "local",
            "repo": pattern,
            "name": None,
            "builder": builders.pop(),
            "files": files,
            "split": "train",
            "tag": f"local-{stem}-{digest}",
        }

    repo, _, subset = source.partition(":")
    if re.fullmatch(r"[\w.-]+/[\w.-]+", repo):
        tag = _slug(repo.replace("/", "-") + (f"-{subset}" if subset else ""))
        return {"kind": "hub", "repo": repo, "name": subset or None, "split": "train", "tag": tag}
    raise ValueError(f"--source {source!r} is not {SOURCE_HELP}")


def source_splits(spec):
    """The corpus split each dataset split is drawn from."""
    return {"valid": spec.get("valid_split", spec["split"]), "train": spec["split"]}


def _ensure_import_path():
    if str(PRIMUS_PATH) not in sys.path:
        sys.path.insert(0, str(PRIMUS_PATH))


def _to_dict(value):
    if isinstance(value, SimpleNamespace):
        return {k: _to_dict(v) for k, v in vars(value).items()}
    if isinstance(value, dict):
        return {k: _to_dict(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_dict(v) for v in value]
    return value


def _deep_merge(base, extra):
    merged = dict(base)
    for key, value in extra.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def as_number(value, default=None):
    """YAML 1.1 reads ``1e-5`` (no dot) as a string; accept it as a number."""
    if isinstance(value, bool):
        return default
    if isinstance(value, (int, float)):
        return value
    try:
        return float(str(value).replace("_", ""))
    except (TypeError, ValueError):
        return default


@dataclass
class ResolvedConfig:
    path: Path
    framework: str
    exp_root_path: str
    exp_name: str
    params: dict
    effective: dict
    # MaxText only: where base.yml was found, which model file was merged, and
    # overlay keys that the model file silently replaced ({key: (ours, theirs)}).
    maxtext_configs: Optional[Path] = None
    model_file: Optional[Path] = None
    shadowed: dict = field(default_factory=dict)

    def get(self, key, default=None):
        value = self.effective.get(key, default)
        return default if value is None else value

    @property
    def model_preset(self):
        return self.params.get("model") or ""

    @property
    def num_devices(self):
        return int(os.environ.get("GPUS_PER_NODE", "8")) * int(os.environ.get("NNODES", "1"))

    @property
    def global_batch_size(self):
        if self.framework == "maxtext":
            pdbs = as_number(self.get("per_device_batch_size"), 0)
            accum = int(as_number(self.get("gradient_accumulation_steps"), 1) or 1)
            return int(self.num_devices * pdbs) * accum
        return int(as_number(self.get("global_batch_size"), 0))

    @property
    def seq_length(self):
        key = "max_target_length" if self.framework == "maxtext" else "seq_length"
        return int(as_number(self.get(key), 0))

    @property
    def train_iters(self):
        key = "steps" if self.framework == "maxtext" else "train_iters"
        return int(as_number(self.get(key), 0))


def maxtext_configs_dir():
    """MaxText's configs directory, resolved like the prepare hook resolves the backend."""
    candidates = []
    if os.environ.get("MAXTEXT_PATH"):
        candidates.append(Path(os.environ["MAXTEXT_PATH"]))
    candidates.append(PRIMUS_PATH / "third_party" / "maxtext")
    for root in candidates:
        configs = root / "src" / "maxtext" / "configs"
        if (configs / "base.yml").is_file():
            return configs
    return None


def _merge_maxtext(params):
    """Replay pyconfig's merge: base.yml <- Primus overlay <- model file <- override_model."""
    import yaml

    overlay = {k: v for k, v in params.items() if k not in MAXTEXT_NOT_IN_FILE}
    kwargs = params.get("override_model") or {}
    configs = maxtext_configs_dir()
    if configs is None:
        return {**overlay, **kwargs}, None, None, {}

    base = yaml.safe_load((configs / overlay.get("base_config", "base.yml")).read_text()) or {}
    file_layer = {**base, **overlay}

    model_name = str(file_layer.get("model_name", "default")).replace("-Instruct", "")
    model_file, model_cfg = None, {}
    if model_name != "default":
        model_file = configs / "models" / f"{model_name}.yml"
        if model_file.is_file():
            loaded = yaml.safe_load(model_file.read_text()) or {}
            if file_layer.get("override_model_config"):
                model_cfg = {k: v for k, v in loaded.items() if k not in kwargs}
            else:
                model_cfg = loaded
        else:
            model_file = None

    shadowed = {
        k: (overlay[k], model_cfg[k]) for k in overlay if k in model_cfg and overlay[k] != model_cfg[k]
    }
    return {**file_layer, **model_cfg, **kwargs}, configs, model_file, shadowed


def load(config_path, overrides=None):
    """Resolve an experiment YAML, optionally with primus-cli style overrides.

    Relative paths in a config are relative to the Primus root, which is the
    trainer's working directory, so this changes the working directory to it.
    """
    _ensure_import_path()
    from primus.core.launcher.parser import PrimusParser
    from primus.core.utils.arg_utils import parse_cli_overrides

    config_path = Path(config_path).resolve()
    os.chdir(PRIMUS_PATH)
    # PrimusParser announces directories it creates on stdout; keep stdout for
    # callers that print machine-readable output.
    with contextlib.redirect_stdout(sys.stderr):
        exp = PrimusParser().parse(SimpleNamespace(config=str(config_path)))
    module = exp.get_module_config("pre_trainer")
    params = _to_dict(module)
    if overrides:
        params = _deep_merge(params, parse_cli_overrides(list(overrides)))

    framework = params.get("framework") or getattr(module, "framework", "")
    resolved = ResolvedConfig(
        path=config_path,
        framework=framework,
        exp_root_path=exp.exp_root_path,
        exp_name=os.path.basename(exp.exp_root_path.rstrip("/")),
        params=params,
        effective=params,
    )
    if framework == "maxtext":
        effective, configs, model_file, shadowed = _merge_maxtext(params)
        resolved.effective = effective
        resolved.maxtext_configs = configs
        resolved.model_file = model_file
        resolved.shadowed = shadowed
    return resolved


def tokenizer_from_preset(model_preset, framework):
    """(tokenizer_type, hub name) declared by a Primus model preset."""
    _ensure_import_path()
    from primus.core.config.preset_loader import PresetLoader

    if framework not in FRAMEWORKS:
        raise SystemExit(f"unsupported framework {framework!r}; expected one of {', '.join(FRAMEWORKS)}")
    if not model_preset.endswith(".yaml"):
        model_preset += ".yaml"
    preset = PresetLoader.load(model_preset, framework, config_type="models")
    if framework == "maxtext":
        tokenizer_type, name = preset.get("tokenizer_type") or "huggingface", preset.get("tokenizer_path")
    else:
        tokenizer_type, name = preset.get("tokenizer_type"), preset.get("tokenizer_model")
    if not name:
        raise SystemExit(f"{framework} model preset {model_preset} declares no tokenizer")

    fixed = TOKENIZER_NAME_FIXES.get(name)
    if fixed:
        print(
            f"[resolve-config] model preset names {name}, which cannot be loaded; using {fixed}",
            file=sys.stderr,
        )
        name = fixed
    return tokenizer_type, name


def default_dataset_dir(tokenizer_name, framework="megatron", seq_length=None, source="fineweb-edu"):
    """Where prepare_dataset.py puts a dataset when no --out-dir is given.

    MaxText rows are pre-chunked to the sequence length (grain rejects rows
    longer than max_target_length), so its datasets are keyed by it too.
    """
    root = Path(os.environ.get("DATA_PATH") or PRIMUS_PATH / "data")
    tag = f"{parse_source(source)['tag']}-{tokenizer_name.split('/')[-1].lower()}"
    if framework == "maxtext":
        tag += f"-maxtext-{seq_length}"
    return root / "convergence" / tag


def main():
    parser = argparse.ArgumentParser(
        description="Print the driver's plan for a convergence config.", allow_abbrev=False
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--tokenizer", help="Tokenizer substituted for the model preset's")
    parser.add_argument(
        "--source", default="fineweb-edu", help=f"Corpus the dataset is built from: {SOURCE_HELP}"
    )
    args, overrides = parser.parse_known_args()

    # Before load() moves to the Primus root: a relative local source is relative to the caller.
    try:
        source = parse_source(args.source)
    except ValueError as exc:
        raise SystemExit(f"error: {exc}") from exc
    if source["kind"] == "local":
        args.source = source["repo"]

    cfg = load(args.config, overrides)
    if cfg.framework not in FRAMEWORKS:
        raise SystemExit(
            f"framework {cfg.framework!r} is not supported by the convergence test "
            f"(supported: {', '.join(FRAMEWORKS)})"
        )
    tokenizer = args.tokenizer or tokenizer_from_preset(cfg.model_preset, cfg.framework)[1]
    plan = {
        "FRAMEWORK": cfg.framework,
        "EXP_DIR": cfg.exp_root_path,
        "EXP_NAME": cfg.exp_name,
        "MODEL_PRESET": cfg.model_preset,
        "GBS": cfg.global_batch_size,
        "SEQ": cfg.seq_length,
        "ITERS": cfg.train_iters,
        "DEFAULT_DATA_DIR": default_dataset_dir(tokenizer, cfg.framework, cfg.seq_length, args.source),
        # Runs on a non-default corpus are not comparable with the default ones,
        # so their output names say which corpus they trained on.
        "SOURCE_TAG": "" if args.source == "fineweb-edu" else f"-{source['tag']}",
        # Primus prints Megatron's per-iteration line at DEBUG on the last rank and
        # forwards it to the console only on a single node.
        "LOSS_ON_CONSOLE": int(
            cfg.framework == "maxtext"
            or (
                str(cfg.get("stderr_sink_level", "")).upper() == "DEBUG"
                and os.environ.get("NNODES", "1") == "1"
            )
        ),
    }
    if cfg.framework == "maxtext":
        plan["VOCAB"] = int(as_number(cfg.get("vocab_size"), 0))
        plan["RUN_NAME"] = cfg.get("run_name", "")
        plan["BASE_OUTPUT_DIR"] = cfg.get("base_output_directory", "")
    for key, value in plan.items():
        print(f"{key}={shlex.quote(str(value))}")


if __name__ == "__main__":
    main()
