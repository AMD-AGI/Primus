###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build a Megatron indexed dataset (.bin/.idx) from a real Hugging Face corpus.

The dataset is tokenised with the vocabulary of the model you intend to train,
which is the single most common thing to get wrong: a corpus tokenised with a
128k Llama-3 vocab is unusable for a 32k Mixtral model. Pass ``--model`` and the
tokenizer is resolved from the Primus model preset for you.

The stock ``examples/megatron/prepare.py`` flow cannot be used for this: it
downloads BookCorpus through a dataset loading script, which ``datasets>=3.0``
refuses to execute. ``examples/megatron/preprocess_data.py`` cannot be used
either: it imports ``megatron.training``, which needs a GPU at import time and
deadlocks its own ``multiprocessing.Pool`` (huggingface_hub's fork handler logs
a debug record through Primus' loguru bridge while the loguru lock is held).

This script imports only ``megatron.core.datasets.indexed_dataset``, tokenises
through ``transformers``, and uses the ``spawn`` start method.

Examples:
    # Tokenizer taken from the model preset
    python3 tools/convergence_test/prepare_dataset.py \
        --model mixtral_8x7B_v0.1.yaml --target-tokens 600e6

    # Explicit tokenizer, different corpus
    python3 tools/convergence_test/prepare_dataset.py \
        --tokenizer Qwen/Qwen2.5-7B --source c4 --target-tokens 1e9
"""

import argparse
import json
import os
import sys
import threading
import time
from pathlib import Path

PRIMUS_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PRIMUS_PATH))
sys.path.insert(0, os.environ.get("MEGATRON_PATH", str(PRIMUS_PATH / "third_party" / "Megatron-LM")))

SOURCES = {
    "fineweb-edu": {"repo": "HuggingFaceFW/fineweb-edu", "name": "sample-10BT", "split": "train"},
    "c4": {"repo": "allenai/c4", "name": "en", "split": "train"},
    "wikitext103": {"repo": "Salesforce/wikitext", "name": "wikitext-103-raw-v1", "split": "train"},
}

# A couple of model presets name tokenizer repos that do not exist: there is no
# meta-llama/Meta-Llama-3.2-{1B,3B}, the real repos drop the "Meta-" prefix.
# Nobody notices because mock_data swaps in NullTokenizer. These are typo
# corrections, not a way around gating -- the corrected repos are still gated
# and still need HF_TOKEN. Use --tokenizer to substitute any repo you prefer.
TOKENIZER_NAME_FIXES = {
    "meta-llama/Meta-Llama-3.2-1B": "meta-llama/Llama-3.2-1B",
    "meta-llama/Meta-Llama-3.2-3B": "meta-llama/Llama-3.2-3B",
}

_WORKER = {}


def log(msg):
    # stderr, so callers can capture a path off stdout without picking up chatter.
    print(f"[prepare-dataset] {msg}", file=sys.stderr, flush=True)


def resolve_tokenizer_from_model(model_preset):
    """Read tokenizer_type/tokenizer_model out of a Primus model preset."""
    from primus.core.config.preset_loader import PresetLoader

    if not model_preset.endswith(".yaml"):
        model_preset += ".yaml"
    preset = PresetLoader.load(model_preset, "megatron", config_type="models")
    tokenizer_type = preset.get("tokenizer_type")
    tokenizer_model = preset.get("tokenizer_model")
    if not tokenizer_model:
        raise SystemExit(f"model preset {model_preset} declares no tokenizer_model")

    fixed = TOKENIZER_NAME_FIXES.get(tokenizer_model)
    if fixed:
        log(f"model preset names {tokenizer_model}, which does not exist; using {fixed}")
        tokenizer_model = fixed
    return tokenizer_type, tokenizer_model


def load_tokenizer(name):
    """Load a tokenizer, turning HF's access errors into actionable advice."""
    from transformers import AutoTokenizer

    try:
        return AutoTokenizer.from_pretrained(name)
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance
        hint = (
            f"could not load tokenizer {name!r}: {type(exc).__name__}: {str(exc)[:200]}\n"
            "If the repo is gated, accept its licence on huggingface.co and export a\n"
            "token with access:  export HF_TOKEN=hf_...\n"
            "Otherwise pass --tokenizer <repo-or-path> to use an equivalent vocabulary."
        )
        raise SystemExit(hint) from exc


def _worker_init(tokenizer_dir):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    _WORKER["tokenizer"] = tokenizer
    _WORKER["eod"] = tokenizer.eos_token_id


def _worker_encode(texts):
    tokenizer = _WORKER["tokenizer"]
    eod = _WORKER["eod"]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    for ids in encoded:
        ids.append(eod)
    return encoded


def stream_batches(source, batch_size, min_chars):
    """Yield lists of raw document strings from a streaming HF dataset."""
    from datasets import load_dataset

    spec = SOURCES[source]
    stream = load_dataset(spec["repo"], spec["name"], split=spec["split"], streaming=True)

    batch = []
    for record in stream:
        text = record.get("text") or ""
        if len(text) < min_chars:
            continue
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def build_splits(source, splits, tokenizer_dir, num_proc, batch_size, min_chars, dtype):
    """Fill each split in turn from a single pass over the corpus stream.

    Sharing one stream keeps the splits disjoint without having to download and
    throw away the documents an earlier split already consumed.
    """
    import multiprocessing

    import numpy
    from megatron.core.datasets import indexed_dataset

    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(num_proc, initializer=_worker_init, initargs=(tokenizer_dir,))

    # imap's feeder thread would otherwise drain the whole corpus into memory;
    # keep at most a few batches per worker in flight.
    inflight = threading.Semaphore(num_proc * 4)

    def throttled(batches):
        for batch in batches:
            inflight.acquire()
            yield batch

    summary = {}
    try:
        results = pool.imap(_worker_encode, throttled(stream_batches(source, batch_size, min_chars)))
        for name, prefix, target_tokens in splits:
            builder = indexed_dataset.IndexedDatasetBuilder(f"{prefix}.bin", dtype=dtype)
            tokens = docs = batch_count = 0
            start = time.time()
            log(f"{name}: building {target_tokens/1e6:.0f}M tokens -> {prefix}.bin")

            for encoded in results:
                inflight.release()
                for ids in encoded:
                    builder.add_document(numpy.array(ids, dtype=dtype), [len(ids)])
                    tokens += len(ids)
                    docs += 1
                batch_count += 1
                if tokens >= target_tokens:
                    break
                if batch_count % 200 == 0:
                    elapsed = max(time.time() - start, 1e-6)
                    log(
                        f"  {name}: {tokens/1e6:.1f}M / {target_tokens/1e6:.0f}M tokens "
                        f"({docs} docs, {tokens/elapsed/1e6:.2f}M tok/s)"
                    )

            builder.finalize(f"{prefix}.idx")
            log(
                f"  {name}: done -- {docs} docs, {tokens/1e6:.1f}M tokens, "
                f"{Path(f'{prefix}.bin').stat().st_size/1e9:.2f} GB in {int(time.time()-start)}s"
            )
            summary[name] = {"documents": docs, "tokens": tokens, "prefix": str(prefix)}
    finally:
        pool.terminate()
        pool.join()

    return summary


def verify(prefix, tokenizer_dir, vocab_size):
    """Re-read a built split and check the things that actually break training.

    Counts are recomputed from the index rather than carried over from the build
    so the manifest stays correct even when the build step was skipped.
    """
    from megatron.core.datasets.indexed_dataset import IndexedDataset
    from transformers import AutoTokenizer

    dataset = IndexedDataset(prefix)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    count = len(dataset)
    tokens = int(dataset.index.sequence_lengths.sum())
    step = max(1, count // 200)
    max_id = max(int(dataset[i].max()) for i in range(0, count, step))
    eod_ok = all(int(dataset[i][-1]) == tokenizer.eos_token_id for i in range(0, count, step))

    log(
        f"  verify {Path(prefix).name}: {count} docs, {tokens/1e6:.1f}M tokens, "
        f"max token id {max_id}, eod-terminated {eod_ok}"
    )
    if max_id >= vocab_size:
        raise SystemExit(f"token id {max_id} exceeds vocab size {vocab_size} -- wrong tokenizer?")
    if not eod_ok:
        raise SystemExit("documents are not EOD-terminated")
    return {
        "documents": count,
        "tokens": tokens,
        "prefix": str(prefix),
        "preview": tokenizer.decode(dataset[0][:32]),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", help="Primus model preset, e.g. mixtral_8x7B_v0.1.yaml")
    parser.add_argument("--tokenizer", help="HF tokenizer repo or path (overrides --model)")
    parser.add_argument("--source", choices=sorted(SOURCES), default="fineweb-edu")
    parser.add_argument("--out-dir", help="Output directory (default: $DATA_PATH/convergence/<tag>)")
    parser.add_argument("--target-tokens", type=float, default=600e6, help="Training tokens to build")
    parser.add_argument("--valid-tokens", type=float, default=10e6, help="Validation tokens to build")
    parser.add_argument("--min-chars", type=int, default=200, help="Drop documents shorter than this")
    parser.add_argument("--num-proc", type=int, default=min(48, (os.cpu_count() or 8)))
    parser.add_argument("--batch-size", type=int, default=256, help="Documents per worker batch")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    args = parser.parse_args()

    if not args.tokenizer and not args.model:
        raise SystemExit("one of --model or --tokenizer is required")

    tokenizer_type = "HuggingFaceTokenizer"
    tokenizer_name = args.tokenizer
    if args.model:
        preset_type, preset_model = resolve_tokenizer_from_model(args.model)
        tokenizer_type = preset_type or tokenizer_type
        tokenizer_name = tokenizer_name or preset_model

    import numpy
    from megatron.core.datasets.indexed_dataset import DType

    # Resolve the output directory before touching the Hub: re-running against
    # an existing dataset should work offline, and for a gated tokenizer it is
    # the only way to re-verify without an access token.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        data_path = Path(os.environ.get("DATA_PATH", PRIMUS_PATH / "data"))
        tag = f"{args.source}-{tokenizer_name.split('/')[-1].lower()}"
        out_dir = data_path / "convergence" / tag
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = out_dir / "tokenizer"

    manifest_path = out_dir / "dataset_info.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    reuse_local = bool(previous) and not args.force and tokenizer_dir.is_dir()

    tokenizer = load_tokenizer(str(tokenizer_dir) if reuse_local else tokenizer_name)
    if reuse_local:
        log(f"reusing the tokenizer saved in {tokenizer_dir} (no Hub access needed)")
        recorded_name = previous.get("tokenizer")
        if recorded_name not in (None, tokenizer_name):
            # Not fatal: the saved copy is what both the data and the training
            # config use, so the pair stays self-consistent. But say so, because
            # it will not match the tokenizer the caller asked for.
            log(
                f"WARNING: this dataset was built with {recorded_name}, not the requested "
                f"{tokenizer_name}; the saved copy is being used. Pass --force to rebuild."
            )
    if tokenizer.eos_token_id is None:
        raise SystemExit(f"tokenizer {tokenizer_name} has no eos token, cannot append <eod>")
    vocab_size = len(tokenizer)

    log(f"output directory : {out_dir}")
    log(f"source corpus    : {SOURCES[args.source]['repo']} ({SOURCES[args.source]['name']})")
    log(f"tokenizer        : {tokenizer_name} ({tokenizer_type})")

    # Overwriting the saved tokenizer while keeping existing .bin files would
    # leave a corpus and a vocabulary that disagree, and a larger new vocab
    # would not even trip the max-token-id check in verify().
    if previous and not args.force:
        recorded = previous.get("vocab_size")
        if recorded is not None and recorded != vocab_size:
            raise SystemExit(
                f"{out_dir} holds a dataset tokenised with a {recorded}-token vocabulary, "
                f"but {tokenizer_name} has {vocab_size}.\nUse a different --out-dir, "
                "or --force to rebuild from scratch."
            )
    if not reuse_local:
        tokenizer.save_pretrained(tokenizer_dir)
    dtype = DType.optimal_dtype(vocab_size)
    log(f"vocab {vocab_size}, eod id {tokenizer.eos_token_id}, bin dtype {numpy.dtype(dtype)}")

    # Workers load the tokenizer from disk; keep them off the network entirely.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    train_prefix = out_dir / "train_text_document"
    valid_prefix = out_dir / "valid_text_document"
    splits = [("valid", valid_prefix, args.valid_tokens), ("train", train_prefix, args.target_tokens)]

    have_all = all(Path(f"{p}.idx").exists() for _, p, _ in splits)
    if have_all and not args.force:
        log("indexed datasets already exist, skipping build (use --force to rebuild)")
    else:
        build_splits(
            args.source,
            [(n, str(p), t) for n, p, t in splits],
            str(tokenizer_dir),
            args.num_proc,
            args.batch_size,
            args.min_chars,
            dtype,
        )

    summary = {name: verify(str(prefix), str(tokenizer_dir), vocab_size) for name, prefix, _ in splits}

    # A short run against a big existing dataset is fine, but silently training
    # on less data than asked for is not.
    train_tokens = summary["train"]["tokens"]
    if train_tokens < args.target_tokens * 0.95:
        log(
            f"WARNING: train split holds {train_tokens/1e6:.0f}M tokens but "
            f"{args.target_tokens/1e6:.0f}M were requested; re-run with --force to extend"
        )

    # The manifest is what check_config.py uses to prove the dataset and the
    # training config agree on a vocabulary.
    manifest_path.write_text(
        json.dumps(
            {
                "source": SOURCES[args.source]["repo"],
                "source_subset": SOURCES[args.source]["name"],
                # Record what the data was actually tokenised with, which is not
                # necessarily what this invocation asked for.
                "tokenizer": (
                    (previous or {}).get("tokenizer", tokenizer_name) if reuse_local else tokenizer_name
                ),
                "tokenizer_type": tokenizer_type,
                "tokenizer_dir": str(tokenizer_dir),
                "vocab_size": vocab_size,
                "eod_id": tokenizer.eos_token_id,
                "dtype": numpy.dtype(dtype).name,
                "splits": summary,
            },
            indent=2,
        )
        + "\n"
    )
    log(f"manifest written to {manifest_path}")

    print()
    print("Point your convergence config at this dataset:")
    print(f"      tokenizer_type: {tokenizer_type}")
    print(f"      tokenizer_model: {tokenizer_dir}")
    print(f"      train_data_path: [{train_prefix}]")
    print(f"      valid_data_path: [{valid_prefix}]")
    print(f"      test_data_path: [{valid_prefix}]")
    print()
    print(f"Or export PRIMUS_CONVERGENCE_DATA={out_dir} and use the bundled configs.")


if __name__ == "__main__":
    main()
    # The HF streaming reader leaves background threads that trip up CPython's
    # finaliser; everything is written and flushed by here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
