###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Turn a convergence-test request into a runnable plan.

Takes what a user asked for in their own words -- a model name in any spelling,
a precision, a corpus, a docker image -- and prints JSON: the backend, the config
to run (a bundled recipe, or one generated from the model's example config), the
tokenizer and whether it needs a Hugging Face token, the dataset, and the exact
arguments for run_convergence_test.sh. Whatever it cannot decide comes back as a
question with options rather than as a guess.

    python3 tools/convergence_test/plan_request.py --model "llama 2 7b" --precision fp8 \
        --image rocm/primus:v26.7 --source c4

Status "ready" means the plan is saved to `plan_file` and can be launched with
`jobs.py start --plan <plan_file>`; "needs_input" means ask the user `questions`
and plan again with the answers.
"""

import argparse
import contextlib
import difflib
import io
import json
import os
import re
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import resolve_config as rc  # noqa: E402
from resolve_config import PRIMUS_PATH  # noqa: E402

TOOL_DIR = Path(__file__).resolve().parent
DRIVER = TOOL_DIR / "run_convergence_test.sh"
GPUS = ("MI355X", "MI325X", "MI300X")

PRECISION_WORDS = {
    "bf16": "bf16",
    "bfloat16": "bf16",
    "fp8": "fp8",
    "float8": "fp8",
    "nanoo_fp8": "fp8",
    "nanoo-fp8": "fp8",
    "nanoofp8": "fp8",
    "mxfp8": "mxfp8",
    "mxfp4": "mxfp4",
}
NAME_PRECISION = {"bf16": "bf16", "fp8": "fp8", "nanoo_fp8": "fp8", "mxfp8": "mxfp8", "mxfp4": "mxfp4"}
NOISE_WORDS = ("meta", "hf", "base", "instruct", "model", "pretrain", "pretraining")


# ---------------------------------------------------------------------------
# Names
# ---------------------------------------------------------------------------


def split_precision(text):
    """('llama2 7b', 'fp8') from 'Llama-2-7B FP8'; the precision is None when absent."""
    precision = None
    for word in sorted(PRECISION_WORDS, key=len, reverse=True):
        pattern = rf"(?<![a-z0-9]){re.escape(word)}(?![a-z0-9])"
        if re.search(pattern, text, re.IGNORECASE):
            precision = precision or PRECISION_WORDS[word]
            text = re.sub(pattern, " ", text, flags=re.IGNORECASE)
    return text, precision


def model_key(text):
    """A spelling-independent key: 'Meta-Llama-3.1-8B', 'llama3.1_8B' -> 'llama318b'."""
    text = split_precision(text)[0].lower()
    text = re.sub(r"[-_ ]v\d+\.\d+\b", " ", text)  # mixtral_8x7B_v0.1
    words = [w for w in re.split(r"[^a-z0-9.]+", text) if w and w not in NOISE_WORDS]
    return re.sub(r"[^a-z0-9]", "", "".join(words))


@dataclass
class Entry:
    backend: str
    kind: str  # "bundled" or "example"
    name: str
    precision: str
    key: str
    path: Path
    gpu: str = ""


def catalog():
    entries = []
    for path in sorted((TOOL_DIR / "configs").glob("*/*-convergence.yaml")):
        backend, stem = path.parent.name, path.name[: -len("-convergence.yaml")]
        match = re.match(r"^(.*?)(?:-(FP8|fp8|nanoo_fp8|BF16|bf16|MXFP8|MXFP4))?$", stem)
        precision = NAME_PRECISION[(match.group(2) or "bf16").lower()]
        entries.append(
            Entry(backend, "bundled", f"{backend}/{stem}", precision, model_key(match.group(1)), path)
        )
    example_re = re.compile(
        r"^(?P<model>.+?)[-_](?P<prec>BF16|FP8|MXFP8|MXFP4|bf16|fp8|nanoo_fp8)-pretrain\.yaml$"
    )
    for backend in rc.FRAMEWORKS:
        for path in sorted((PRIMUS_PATH / "examples" / backend / "configs").glob("*/*-pretrain.yaml")):
            match = example_re.match(path.name)
            if match:
                model, precision = match.group("model"), NAME_PRECISION[match.group("prec").lower()]
            elif split_precision(path.name)[1] is None:
                model, precision = path.name[: -len("-pretrain.yaml")], "bf16"
            else:
                continue  # a variant such as -FP8-mlperf-pretrain
            entries.append(
                Entry(backend, "example", path.name, precision, model_key(model), path, path.parent.name)
            )
    return entries


# ---------------------------------------------------------------------------
# Environment
# ---------------------------------------------------------------------------


def _run(cmd, timeout=30):
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.TimeoutExpired):
        return None


def detect_gpu():
    """(model, count) of this node's GPUs, e.g. ("MI325X", 8); (None, None) if unknown."""
    for cmd in (["amd-smi", "static", "--asic"], ["rocm-smi", "--showproductname"]):
        result = _run(cmd)
        models = re.findall(r"MI\d{3}[A-Z]?", result.stdout) if result else []
        if models:
            return models[0], len(models)
    return None, None


def default_image(backend):
    if backend == "maxtext":
        match = re.search(r'^MAXTEXT_DEFAULT_IMAGE="([^"]+)"', DRIVER.read_text(), re.MULTILINE)
        return match.group(1) if match else None
    import yaml

    config = yaml.safe_load((PRIMUS_PATH / "runner" / ".primus.yaml").read_text()) or {}
    return ((config.get("container") or {}).get("options") or {}).get("image")


def image_present(image):
    result = _run(["docker", "image", "inspect", image])
    return bool(result and result.returncode == 0)


def backend_from_image(image):
    """(backend, reason) from the image name, or by looking inside a local image."""
    name = image.lower()
    if "jax" in name or "maxtext" in name:
        return "maxtext", "the image name says JAX/MaxText"
    if any(word in name for word in ("primus", "torch", "megatron")):
        return "megatron", "the image name says PyTorch/Primus"
    if image_present(image):
        probe = "import importlib.util as u; print(int(bool(u.find_spec('jax'))), int(bool(u.find_spec('torch'))))"
        result = _run(["docker", "run", "--rm", "--entrypoint", "python3", image, "-c", probe], timeout=180)
        if result and result.returncode == 0:
            has_jax, has_torch = (result.stdout.split() + ["0", "0"])[:2]
            if has_jax == "1" and has_torch != "1":
                return "maxtext", "the image has JAX and no PyTorch"
            if has_torch == "1" and has_jax != "1":
                return "megatron", "the image has PyTorch and no JAX"
    return None, None


def hub_status(repo):
    """'open', 'gated', 'missing', or 'unknown' (offline, or a local path)."""
    if Path(repo).exists() or "/" not in repo:
        return "unknown"
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.utils import RepositoryNotFoundError

        try:
            info = HfApi().model_info(repo, timeout=15)
        except RepositoryNotFoundError:
            return "missing"
        return "gated" if info.gated else "open"
    except Exception:  # noqa: BLE001 - offline or an old huggingface_hub
        return "unknown"


def token_available():
    if os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN"):
        return True
    try:
        from huggingface_hub import get_token

        return bool(get_token())
    except Exception:  # noqa: BLE001
        return False


def ungated_mirror(repo):
    """An ungated copy of a gated meta-llama tokenizer with the same vocabulary."""
    if not repo.startswith("meta-llama/"):
        return None
    name = repo.split("/", 1)[1]
    for candidate in (f"NousResearch/{name}", f"NousResearch/Meta-{name}"):
        if hub_status(candidate) == "open":
            return candidate
    return None


# ---------------------------------------------------------------------------
# Planning
# ---------------------------------------------------------------------------


def estimate_duration(results_dir, exp_name, iters, build_minutes):
    """Minutes for the run, from the step time of the latest earlier run of this config here."""
    import csv
    import statistics

    for path in sorted(results_dir.glob(f"{exp_name}*_*.csv"), key=lambda p: p.stat().st_mtime, reverse=True):
        try:
            with open(path, newline="") as handle:
                steps = [
                    float(row["elapsed_ms"])
                    for row in csv.DictReader(handle)
                    if row.get("elapsed_ms") and int(float(row["iteration"])) > 2
                ]
        except (OSError, ValueError, KeyError):
            continue
        if len(steps) >= 5:
            seconds = statistics.median(steps) / 1000
            # start-up and compilation ~5 min, evaluation ~5% of training
            minutes = build_minutes + 5 + iters * seconds * 1.05 / 60
            return {"minutes": round(minutes), "seconds_per_iteration": round(seconds, 2), "from": str(path)}
    return {
        "minutes": None,
        "note": "no earlier run of this config here; the probe or the first iterations will tell",
    }


def question(qid, prompt, options):
    return {"id": qid, "prompt": prompt, "options": [{"id": o, "label": label} for o, label in options]}


def needs_input(plan, *questions):
    plan["status"] = "needs_input"
    plan["questions"] = list(questions)
    return plan


def plan_request(args):
    plan = {"status": "ready", "request": {k: v for k, v in vars(args).items() if v not in (None, [], False)}}
    warnings = plan.setdefault("warnings", [])
    model_text, named_precision = split_precision(args.model)
    precision = (args.precision or named_precision or "bf16").lower()
    precision = PRECISION_WORDS.get(precision, precision)
    key = model_key(model_text)
    plan["precision"] = precision

    entries = catalog()
    matches = [e for e in entries if e.key == key]
    if not matches:
        known = sorted({e.key: e.name for e in entries}.items())
        close = difflib.get_close_matches(key, [k for k, _ in known], n=6, cutoff=0.5)
        names = {}
        for e in entries:
            if e.key in close and e.key not in names:
                names[e.key] = re.sub(r"(-convergence|-pretrain)?\.yaml$", "", e.name.split("/")[-1])
        options = [(k, f"{names[k]}") for k in close]
        return needs_input(
            plan,
            question(
                "model",
                f"No config or example matches model {args.model!r}. Which model did you mean?",
                options or [("other", "None of these; I will name another")],
            ),
        )

    # Backend: asked for, implied by the image, or the only one with this model.
    backend, reason = args.backend, "requested"
    if not backend and args.image:
        backend, reason = backend_from_image(args.image)
    available = sorted({e.backend for e in matches})
    if not backend:
        if len(available) == 1:
            backend, reason = available[0], f"only {available[0]} has this model"
        else:
            return needs_input(
                plan,
                question(
                    "backend",
                    "Run it on which backend?",
                    [
                        ("megatron", f"Megatron (PyTorch image, default {default_image('megatron')})"),
                        ("maxtext", f"MaxText (JAX image, default {default_image('maxtext')})"),
                    ],
                ),
            )
    if backend not in available:
        return needs_input(
            plan,
            question(
                "backend",
                f"{args.model} has no {backend} config or example; it exists for {', '.join(available)}. "
                "Use that backend (and its image)?",
                [(b, b) for b in available] + [("cancel", "Cancel")],
            ),
        )
    plan["backend"], plan["backend_reason"] = backend, reason

    detected, count = detect_gpu()
    gpu = args.gpu or detected
    plan["gpu"] = gpu
    plan["gpu_count"] = count
    if not gpu:
        warnings.append(
            "could not detect the GPU model (no amd-smi or rocm-smi here); configs are sized for "
            f"{GPUS[0]}; pass --gpu MI300X|MI325X|MI355X if this node has another"
        )
    per_node = int(os.environ.get("GPUS_PER_NODE", "8"))
    if count and count != per_node:
        warnings.append(
            f"this node has {count} GPUs but runs are launched on {per_node}; "
            f"export GPUS_PER_NODE={count} (the global batch then changes for MaxText)"
        )
    if backend == "maxtext" and precision == "fp8" and gpu == "MI355X":
        warnings.append("MI355X takes OCP fp8; the bundled nanoo_fp8 recipes are for MI300X/MI325X")

    candidates = [e for e in matches if e.backend == backend]
    bundled = [e for e in candidates if e.kind == "bundled" and e.precision == precision]
    examples = [e for e in candidates if e.kind == "example" and e.precision == precision]
    if args.from_example:
        bundled = []
    if not bundled and not examples:
        offered = sorted({e.precision for e in candidates})
        return needs_input(
            plan,
            question(
                "precision",
                f"{args.model} on {backend} has no {precision} config; available: {', '.join(offered)}. Which one?",
                [(p, p) for p in offered],
            ),
        )

    driver_args = []
    if bundled:
        entry = bundled[0]
        plan["config"] = {"kind": "bundled", "name": entry.name, "path": str(entry.path)}
        driver_args += ["--model", entry.name]
        config_path = entry.path
        if backend == "maxtext" and precision == "fp8" and gpu == "MI355X":
            args.overrides = ["--quantization", "fp8"] + list(args.overrides)
    else:
        preferred = [g for g in ([gpu] if gpu else []) + list(GPUS) if g]
        ranked = sorted(
            examples, key=lambda e: preferred.index(e.gpu) if e.gpu in preferred else len(preferred)
        )
        entry = ranked[0]
        if gpu and entry.gpu != gpu:
            warnings.append(
                f"no {gpu} example for this model; generated from the {entry.gpu} one, whose batch "
                "and memory settings were tuned for that GPU"
            )
        import make_config

        with contextlib.redirect_stderr(io.StringIO()):
            generated = make_config.generate(
                entry.path,
                args.config_out,
                args.iters or make_config.DEFAULT_ITERS,
                args.source,
                args.tokenizer,
            )
        plan["config"] = {
            "kind": "generated",
            "name": Path(generated["path"]).name,
            "path": generated["path"],
            "example": str(entry.path),
            "notes": generated["notes"],
        }
        driver_args += ["--config", generated["path"]]
        config_path = Path(generated["path"])

    with contextlib.redirect_stderr(io.StringIO()):
        cfg = rc.load(config_path)
        preset_tokenizer = rc.tokenizer_from_preset(cfg.model_preset, backend)[1]
    iters = args.iters or cfg.train_iters
    plan["iterations"] = iters
    plan["global_batch_size"] = cfg.global_batch_size
    plan["seq_length"] = cfg.seq_length
    plan["tokens"] = iters * cfg.global_batch_size * cfg.seq_length
    if args.iters and plan["config"]["kind"] == "bundled":
        driver_args += ["--train-iters", str(args.iters)]

    # Corpus, then the tokenizer the dataset is built with.
    try:
        rc.parse_source(args.source)
    except ValueError as exc:
        return needs_input(
            plan,
            question(
                "source",
                f"{exc}. Which corpus?",
                [
                    ("fineweb-edu", "FineWeb-Edu (default)"),
                    ("c4", "C4 (allenai/c4 en)"),
                    ("wikitext103", "WikiText-103"),
                ],
            ),
        )
    plan["source"] = args.source
    if args.source != "fineweb-edu":
        driver_args += ["--source", args.source]
    if args.text_field:
        driver_args += ["--text-field", args.text_field]

    tokenizer = args.tokenizer or preset_tokenizer
    data_dir = (
        Path(args.data_dir)
        if args.data_dir
        else rc.default_dataset_dir(tokenizer, backend, cfg.seq_length, args.source)
    )
    manifest = data_dir / "dataset_info.json"
    built = (
        json.loads(manifest.read_text()).get("splits", {}).get("train", {}).get("tokens", 0)
        if manifest.exists()
        else 0
    )
    # 10% more than the run consumes, so no document repeats; later runs reuse it.
    needed = int(plan["tokens"] * 1.1)
    plan["dataset"] = {
        "dir": str(data_dir),
        "train_tokens": built,
        "needed_tokens": needed,
        "ready": built >= 0.95 * needed,
    }
    if not plan["dataset"]["ready"]:
        # Tokenising runs at 3-7M tokens/s on 48 cores, streaming included.
        plan["dataset"]["build_minutes"] = max(1, round(needed / 4e6 / 60))
        plan["dataset"]["disk_gb_at_most"] = round(needed * (2 if backend == "maxtext" else 4) / 1e9, 1)

    tok = {"name": tokenizer, "hub": hub_status(tokenizer), "token_available": token_available()}
    if tok["hub"] == "gated" and not tok["token_available"] and not args.tokenizer:
        if (data_dir / "tokenizer").is_dir():
            tok["note"] = "gated, but the dataset directory already holds a copy, so no token is needed"
        else:
            mirror = ungated_mirror(tokenizer)
            if mirror:
                tok["substitute"] = mirror
                tok["note"] = f"{tokenizer} is gated and no HF token is set; using the ungated copy {mirror}"
                driver_args += ["--tokenizer", mirror]
            else:
                return needs_input(
                    plan,
                    question(
                        "hf_token",
                        f"The tokenizer {tokenizer} is gated and no Hugging Face token is set. "
                        "Accept its licence on huggingface.co, then export HF_TOKEN, or name a substitute.",
                        [
                            ("token", "I exported HF_TOKEN"),
                            ("substitute", "Use another tokenizer (I will name it)"),
                        ],
                    ),
                )
    elif tok["hub"] == "missing":
        warnings.append(
            f"tokenizer {tokenizer} is not on the Hugging Face Hub; dataset preparation will fail"
        )
    elif args.tokenizer:
        driver_args += ["--tokenizer", args.tokenizer]
    plan["tokenizer"] = tok
    if args.data_dir:
        driver_args += ["--data-dir", args.data_dir]

    image = args.image or default_image(backend)
    plan["image"] = {
        "name": image,
        "default": not args.image,
        "present_locally": bool(image) and image_present(image),
    }
    if args.image:
        driver_args += ["--image", args.image]
    if not plan["image"]["present_locally"]:
        warnings.append(f"{image} is not on this node; docker will pull it at launch")

    if args.output_dir:
        driver_args += ["--output-dir", args.output_dir]
    results_dir = Path(args.output_dir) if args.output_dir else PRIMUS_PATH / "output" / "convergence"
    plan["results_dir"] = str(results_dir)
    plan["duration"] = estimate_duration(
        results_dir, cfg.exp_name, iters, plan["dataset"].get("build_minutes", 0)
    )
    if args.baseline:
        driver_args += ["--baseline", args.baseline]
    if args.overrides:
        driver_args += ["--"] + list(args.overrides)

    probe_args = ["--probe", "20"] + (["--budget-hours", str(args.hours)] if args.hours else [])
    if "--" in driver_args:
        cut = driver_args.index("--")
        probe_driver_args = driver_args[:cut] + probe_args + driver_args[cut:]
    else:
        probe_driver_args = driver_args + probe_args
    small_memory = gpu == "MI300X" and plan["config"]["kind"] == "bundled"
    if small_memory:
        warnings.append(
            "the bundled recipes were sized on 256 GB GPUs; MI300X has 192 GB, so the run probes first "
            "(the Mixtral recipes need about 225 GB)"
        )
    plan["probe_first"] = plan["config"]["kind"] == "generated" or bool(args.hours) or small_memory
    plan["driver_args"] = driver_args
    plan["probe_driver_args"] = probe_driver_args
    plan["command"] = shlex.join([str(DRIVER.relative_to(PRIMUS_PATH))] + driver_args)
    return plan


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter, allow_abbrev=False
    )
    parser.add_argument("--model", required=True, help="Model name in any spelling, e.g. 'Llama-2 7B FP8'")
    parser.add_argument("--precision", help="bf16 (default), fp8, mxfp8 or mxfp4")
    parser.add_argument("--backend", choices=rc.FRAMEWORKS, help="Default: from the image, else asked")
    parser.add_argument("--image", help="Docker image to train in")
    parser.add_argument("--source", default="fineweb-edu", help=f"Corpus: {rc.SOURCE_HELP}")
    parser.add_argument("--text-field", help="Text column of a custom corpus")
    parser.add_argument("--tokenizer", help="Tokenizer substituted for the model preset's")
    parser.add_argument("--gpu", choices=GPUS, help="GPU model (default: detected)")
    parser.add_argument("--iters", type=int, help="Run length in iterations")
    parser.add_argument("--hours", type=float, help="Wall-clock budget; probe first to size the run")
    parser.add_argument("--data-dir", help="Dataset directory")
    parser.add_argument("--output-dir", help="Where results go")
    parser.add_argument("--baseline", help="Reference run to gate against")
    parser.add_argument(
        "--from-example", action="store_true", help="Generate from the example config even if a recipe exists"
    )
    parser.add_argument("--config-out", help="Where a generated config is written")
    args, overrides = parser.parse_known_args()
    if overrides and overrides[0] == "--":
        overrides = overrides[1:]
    args.overrides = overrides
    for name in ("data_dir", "output_dir", "baseline", "config_out"):
        value = getattr(args, name)
        if value:
            setattr(args, name, os.path.abspath(value))
    if args.source.startswith((".", "~")):
        args.source = os.path.abspath(os.path.expanduser(args.source))

    plan = plan_request(args)
    if plan["status"] == "ready":
        # Saved so jobs.py can launch exactly this plan without re-quoting it.
        plans = PRIMUS_PATH / "output" / "convergence" / "plans"
        plans.mkdir(parents=True, exist_ok=True)
        stem = Path(plan["config"]["name"]).name.replace("-convergence.yaml", "").replace("/", "-")
        plan_file = plans / f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}.json"
        suffix = 1
        while plan_file.exists():
            suffix += 1
            plan_file = plans / f"{time.strftime('%Y%m%d-%H%M%S')}-{stem}-{suffix}.json"
        plan["plan_file"] = str(plan_file)
        plan_file.write_text(json.dumps(plan, indent=2, default=str) + "\n")
    print(json.dumps(plan, indent=2, default=str))


if __name__ == "__main__":
    main()
