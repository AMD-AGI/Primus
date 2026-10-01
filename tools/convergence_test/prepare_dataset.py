###############################################################################
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Build a convergence dataset from a real Hugging Face corpus.

Two output formats, one per backend:

  megatron  indexed dataset (.bin/.idx); every document ends in EOD.
  maxtext   pre-tokenized parquet shards for MaxText's grain pipeline
            (tokenize_train_data: false). Every document is BOS + text + EOS,
            the same as MaxText's own tokenizer produces, split into rows of at
            most --seq-length tokens because grain refuses longer rows.

The dataset is tokenised with the vocabulary of the model you intend to train,
which is the single most common thing to get wrong: a corpus tokenised with a
128k Llama-3 vocab is unusable for a 32k Mixtral model. Pass ``--model`` and the
tokenizer is resolved from the Primus model preset for you.

Both formats are built from the same document stream, so a Megatron and a
MaxText dataset for the same tokenizer hold the same documents.

The stock ``examples/megatron/prepare.py`` flow cannot be used for this: it
downloads BookCorpus through a dataset loading script, which ``datasets>=3.0``
refuses to execute. ``examples/megatron/preprocess_data.py`` cannot be used
either: it imports ``megatron.training``, which needs a GPU at import time and
deadlocks its own ``multiprocessing.Pool`` (huggingface_hub's fork handler logs
a debug record through Primus' loguru bridge while the loguru lock is held).

This script imports only ``megatron.core.datasets.indexed_dataset`` (and only
for the megatron format), tokenises through ``transformers``, and uses the
``spawn`` start method.

Examples:
    # Tokenizer taken from the model preset
    python3 tools/convergence_test/prepare_dataset.py \
        --model mixtral_8x7B_v0.1.yaml --target-tokens 600e6

    # The same corpus for MaxText, in 4096-token rows
    python3 tools/convergence_test/prepare_dataset.py --format maxtext \
        --model mixtral_8x7B.yaml --seq-length 4096 --target-tokens 600e6

    # Explicit tokenizer, different corpus
    python3 tools/convergence_test/prepare_dataset.py \
        --tokenizer Qwen/Qwen2.5-7B --source c4 --target-tokens 1e9

    # Any Hub dataset with a text column, or your own files
    python3 tools/convergence_test/prepare_dataset.py \
        --model llama2_7B.yaml --source HuggingFaceFW/fineweb:sample-10BT --target-tokens 600e6
    python3 tools/convergence_test/prepare_dataset.py \
        --model llama2_7B.yaml --source ./corpus/ --text-field content --target-tokens 600e6
"""

import argparse
import glob
import itertools
import json
import os
import sys
import threading
import time
from pathlib import Path

PRIMUS_PATH = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PRIMUS_PATH))
sys.path.insert(0, os.environ.get("MEGATRON_PATH", str(PRIMUS_PATH / "third_party" / "Megatron-LM")))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from resolve_config import (  # noqa: E402
    SOURCE_HELP,
    default_dataset_dir,
    parse_source,
    source_splits,
    tokenizer_from_preset,
)

# MaxText's grain pipeline pads with, and masks from the loss, the tokenizer's
# pad id, falling back to unk and then to 0. Llama-3 tokenizers have neither, so
# every target equal to token 0 ("!") would silently drop out of the loss. Point
# pad at a reserved token instead; the vocabulary is unchanged.
PAD_CANDIDATES = ("<|finetune_right_pad_id|>", "<|reserved_special_token_0|>", "<pad>", "[PAD]")

# Rows per parquet row group; large enough to compress well, small enough that
# grain's sequential reader does not buffer much.
PARQUET_ROWS_PER_GROUP = 1024

_WORKER = {}


def log(msg):
    # stderr, so callers can capture a path off stdout without picking up chatter.
    print(f"[prepare-dataset] {msg}", file=sys.stderr, flush=True)


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


def maxtext_pad_id(tokenizer):
    """The id MaxText's grain pipeline will pad with and mask from the loss."""
    for token_id in (tokenizer.pad_token_id, tokenizer.unk_token_id):
        if token_id is not None:
            return token_id
    return 0


def ensure_pad_token(tokenizer):
    """Give a tokenizer with no pad and no unk token a reserved pad token."""
    if tokenizer.pad_token_id is not None or tokenizer.unk_token_id is not None:
        return
    for candidate in PAD_CANDIDATES:
        token_id = tokenizer.convert_tokens_to_ids(candidate)
        if isinstance(token_id, int) and token_id not in (tokenizer.bos_token_id, tokenizer.eos_token_id):
            tokenizer.pad_token = candidate
            log(f"tokenizer has no pad/unk token; using {candidate} ({token_id}) as pad for MaxText")
            return
    log(
        "WARNING: tokenizer has no pad, unk or reserved pad token; MaxText will pad with "
        "token 0 and drop every target equal to it from the loss"
    )


def _worker_init(tokenizer_dir, add_bos):
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    _WORKER["tokenizer"] = tokenizer
    _WORKER["bos"] = [tokenizer.bos_token_id] if add_bos else []
    _WORKER["eod"] = tokenizer.eos_token_id


def _worker_encode(texts):
    tokenizer = _WORKER["tokenizer"]
    bos, eod = _WORKER["bos"], _WORKER["eod"]
    encoded = tokenizer(texts, add_special_tokens=False)["input_ids"]
    return [bos + ids + [eod] for ids in encoded]


def describe_source(spec):
    if spec["kind"] == "local":
        return f"{len(spec['files'])} local {spec['builder']} file(s) matching {spec['repo']}"
    return spec["repo"] + (f" ({spec['name']})" if spec["name"] else "")


def open_stream(spec, split, text_field):
    """A streaming reader over one corpus split, opened and checked up front.

    Opening it here rather than inside the worker pool's feeder thread turns a
    wrong repo name, a gated dataset, or a missing column into a clear error
    instead of a traceback from a background thread.
    """
    from datasets import load_dataset

    try:
        if spec["kind"] == "local":
            # Plain text: one document per blank-line-separated paragraph.
            extra = {"sample_by": "paragraph"} if spec["builder"] == "text" else {}
            stream = load_dataset(
                spec["builder"], data_files=spec["files"], split=split, streaming=True, **extra
            )
        else:
            stream = load_dataset(spec["repo"], spec["name"], split=split, streaming=True)
        records = iter(stream)
        first = next(records)
    except StopIteration:
        raise SystemExit(f"{describe_source(spec)}: split {split!r} is empty") from None
    except Exception as exc:  # noqa: BLE001 - re-raised with guidance
        raise SystemExit(
            f"could not read {describe_source(spec)}, split {split!r}: {type(exc).__name__}: {str(exc)[:300]}\n"
            "If the dataset is gated, accept its terms on huggingface.co and export HF_TOKEN."
        ) from exc
    if text_field not in first:
        raise SystemExit(
            f"{describe_source(spec)} has no {text_field!r} column (it has: {', '.join(first)}); "
            "pass --text-field"
        )
    return itertools.chain([first], records)


def stream_batches(records, text_field, batch_size, min_chars):
    """Yield lists of raw document strings."""
    batch = []
    for record in records:
        text = record.get(text_field) or ""
        if len(text) < min_chars:
            continue
        batch.append(text)
        if len(batch) == batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


class MegatronWriter:
    """One indexed dataset (<prefix>.bin/.idx) per split."""

    def __init__(self, prefix, dtype):
        from megatron.core.datasets import indexed_dataset

        self.prefix = prefix
        self.dtype = dtype
        self.builder = indexed_dataset.IndexedDatasetBuilder(f"{prefix}.bin", dtype=dtype)

    def add(self, ids):
        import numpy

        self.builder.add_document(numpy.array(ids, dtype=self.dtype), [len(ids)])

    def finalize(self):
        self.builder.finalize(f"{self.prefix}.idx")

    def size_bytes(self):
        return Path(f"{self.prefix}.bin").stat().st_size


def parquet_files(prefix):
    return sorted(glob.glob(f"{glob.escape(str(prefix))}-*-of-*.parquet"))


class ParquetShardWriter:
    """Pre-chunked token rows spread over ``num_shards`` parquet files.

    Documents are dealt round-robin so every shard is a sample of the whole
    stream, and a document's rows stay together and in order, which is what
    MaxText's own TokenizeAndChunk would produce. Files are written under a
    temporary name and renamed on finalize, so an interrupted build never leaves
    a partial shard where the training glob would pick it up.
    """

    def __init__(self, prefix, num_shards, seq_length):
        self.prefix = prefix
        self.seq_length = seq_length
        self.paths = [f"{prefix}-{i:05d}-of-{num_shards:05d}.parquet" for i in range(num_shards)]
        self.buffers = [[] for _ in range(num_shards)]
        self.writers = [None] * num_shards
        self.documents = 0
        # A rebuild with a different shard count would otherwise leave the old
        # files matching the same glob, silently duplicating data.
        for stale in parquet_files(prefix) + glob.glob(f"{glob.escape(str(prefix))}-*.parquet.partial"):
            os.remove(stale)

    def add(self, ids):
        shard = self.documents % len(self.paths)
        self.documents += 1
        rows = self.buffers[shard]
        for start in range(0, len(ids), self.seq_length):
            rows.append(ids[start : start + self.seq_length])
        if len(rows) >= PARQUET_ROWS_PER_GROUP:
            self._flush(shard)

    def _flush(self, shard):
        import pyarrow as pa
        import pyarrow.parquet as pq

        rows = self.buffers[shard]
        if not rows:
            return
        table = pa.table({"tokens": pa.array(rows, type=pa.list_(pa.int32()))})
        if self.writers[shard] is None:
            self.writers[shard] = pq.ParquetWriter(
                f"{self.paths[shard]}.partial", table.schema, compression="zstd"
            )
        self.writers[shard].write_table(table)
        rows.clear()

    def finalize(self):
        for shard, path in enumerate(self.paths):
            self._flush(shard)
            if self.writers[shard] is not None:
                self.writers[shard].close()
                os.replace(f"{path}.partial", path)

    def size_bytes(self):
        return sum(Path(p).stat().st_size for p in self.paths if Path(p).exists())


def build_splits(spec, splits, tokenizer_dir, add_bos, num_proc, batch_size, min_chars, text_field="text"):
    """Fill each split from its corpus stream.

    ``splits`` is a list of (name, writer factory, target tokens, corpus split).
    Consecutive splits drawn from the same corpus split are filled in turn from a
    single pass over it, which keeps them disjoint without having to download and
    throw away the documents an earlier split already consumed.
    """
    for split, group in itertools.groupby(splits, key=lambda s: s[3]):
        _fill_from_stream(
            spec,
            split,
            [s[:3] for s in group],
            tokenizer_dir,
            add_bos,
            num_proc,
            batch_size,
            min_chars,
            text_field,
        )


def _fill_from_stream(
    spec, split, splits, tokenizer_dir, add_bos, num_proc, batch_size, min_chars, text_field
):
    import multiprocessing

    records = open_stream(spec, split, text_field)
    # One pool per stream: a pool dispatches tasks from a single thread, which
    # would stay blocked on the previous stream's throttled generator.
    ctx = multiprocessing.get_context("spawn")
    pool = ctx.Pool(num_proc, initializer=_worker_init, initargs=(tokenizer_dir, add_bos))

    # imap's feeder thread would otherwise drain the whole corpus into memory;
    # keep at most a few batches per worker in flight.
    inflight = threading.Semaphore(num_proc * 4)

    def throttled(batches):
        for batch in batches:
            inflight.acquire()
            yield batch

    try:
        results = pool.imap(
            _worker_encode, throttled(stream_batches(records, text_field, batch_size, min_chars))
        )
        for name, make_writer, target_tokens in splits:
            writer = make_writer()
            tokens = docs = batch_count = 0
            start = time.time()
            log(f"{name}: building {target_tokens/1e6:.0f}M tokens")

            for encoded in results:
                inflight.release()
                for ids in encoded:
                    writer.add(ids)
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

            if not docs:
                raise SystemExit(
                    f"{name}: the corpus ran out before any document of at least --min-chars "
                    "characters reached this split. Use a larger corpus, or run prepare_dataset.py "
                    "with a smaller --valid-tokens (the driver then takes --data-dir and --skip-prepare)"
                )
            writer.finalize()
            log(
                f"  {name}: done -- {docs} docs, {tokens/1e6:.1f}M tokens, "
                f"{writer.size_bytes()/1e9:.2f} GB in {int(time.time()-start)}s"
            )
    finally:
        pool.terminate()
        pool.join()


def verify_megatron(prefix, tokenizer_dir, vocab_size):
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


def verify_maxtext(prefix, tokenizer_dir, vocab_size, seq_length, pad_id):
    """Scan every row of a built split; parquet is cheap enough to read in full."""
    import pyarrow.compute as pc
    import pyarrow.parquet as pq
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)
    files = parquet_files(prefix)
    if not files:
        raise SystemExit(f"no parquet files match {prefix}-*-of-*.parquet")

    rows = tokens = eos = pad = 0
    max_id = max_len = -1
    first_row = None
    for path in files:
        column = pq.read_table(path, columns=["tokens"]).column("tokens").combine_chunks()
        flat = pc.list_flatten(column)
        rows += len(column)
        tokens += len(flat)
        max_len = max(max_len, pc.max(pc.list_value_length(column)).as_py() or 0)
        max_id = max(max_id, pc.max(flat).as_py() or 0)
        eos += pc.sum(pc.equal(flat, tokenizer.eos_token_id)).as_py() or 0
        if pad_id != tokenizer.eos_token_id:
            pad += pc.sum(pc.equal(flat, pad_id)).as_py() or 0
        if first_row is None and len(column):
            first_row = column[0].as_py()

    log(
        f"  verify {Path(prefix).name}: {len(files)} files, {eos} docs in {rows} rows, "
        f"{tokens/1e6:.1f}M tokens, longest row {max_len}, max token id {max_id}"
    )
    if max_id >= vocab_size:
        raise SystemExit(f"token id {max_id} exceeds vocab size {vocab_size} -- wrong tokenizer?")
    if max_len > seq_length:
        raise SystemExit(f"a row holds {max_len} tokens, more than --seq-length {seq_length}")
    if pad:
        log(
            f"  WARNING: pad id {pad_id} occurs {pad} times in the data; MaxText drops those targets from the loss"
        )
    return {
        "documents": eos,
        "rows": rows,
        "tokens": tokens,
        "files": len(files),
        "pattern": f"{prefix}-*.parquet",
        "preview": tokenizer.decode(first_row[:32]),
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--format", choices=["megatron", "maxtext"], default="megatron")
    parser.add_argument("--model", help="Primus model preset for --format, e.g. mixtral_8x7B_v0.1.yaml")
    parser.add_argument("--tokenizer", help="HF tokenizer repo or path (overrides --model)")
    parser.add_argument("--source", default="fineweb-edu", help=f"Corpus: {SOURCE_HELP}")
    parser.add_argument("--text-field", default="text", help="Column holding each document's text")
    parser.add_argument("--out-dir", help="Output directory (default: $DATA_PATH/convergence/<tag>)")
    parser.add_argument("--target-tokens", type=float, default=600e6, help="Training tokens to build")
    parser.add_argument("--valid-tokens", type=float, default=10e6, help="Validation tokens to build")
    parser.add_argument(
        "--seq-length", type=int, help="maxtext: longest row; use the model's max_target_length"
    )
    parser.add_argument(
        "--num-shards", type=int, default=64, help="maxtext: parquet files for the train split"
    )
    parser.add_argument("--min-chars", type=int, default=200, help="Drop documents shorter than this")
    parser.add_argument("--num-proc", type=int, default=min(48, (os.cpu_count() or 8)))
    parser.add_argument("--batch-size", type=int, default=256, help="Documents per worker batch")
    parser.add_argument("--force", action="store_true", help="Rebuild even if outputs exist")
    args = parser.parse_args()

    maxtext = args.format == "maxtext"
    if not args.tokenizer and not args.model:
        raise SystemExit("one of --model or --tokenizer is required")
    if maxtext and not args.seq_length:
        raise SystemExit("--format maxtext needs --seq-length (the config's max_target_length)")
    try:
        spec = parse_source(args.source)
    except ValueError as exc:
        raise SystemExit(str(exc)) from exc

    tokenizer_type = "huggingface" if maxtext else "HuggingFaceTokenizer"
    tokenizer_name = args.tokenizer
    if args.model:
        preset_type, preset_model = tokenizer_from_preset(args.model, args.format)
        tokenizer_type = preset_type or tokenizer_type
        tokenizer_name = tokenizer_name or preset_model

    # Resolve the output directory before touching the Hub: re-running against
    # an existing dataset should work offline, and for a gated tokenizer it is
    # the only way to re-verify without an access token.
    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:
        out_dir = default_dataset_dir(tokenizer_name, args.format, args.seq_length, args.source)
    out_dir = out_dir.resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    tokenizer_dir = out_dir / "tokenizer"

    manifest_path = out_dir / "dataset_info.json"
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else None
    if previous and not args.force and previous.get("format", "megatron") != args.format:
        raise SystemExit(
            f"{out_dir} holds a {previous.get('format', 'megatron')} dataset, not {args.format}.\n"
            "Use a different --out-dir, or --force to rebuild from scratch."
        )
    if previous and not args.force and "source" in previous:
        built_from = (previous["source"], previous.get("source_subset"), previous.get("text_field", "text"))
        if built_from != (spec["repo"], spec["name"], args.text_field):
            raise SystemExit(
                f"{out_dir} was built from {built_from[0]}"
                + (f" ({built_from[1]})" if built_from[1] else "")
                + f", column {built_from[2]!r}, not from {describe_source(spec)}, column {args.text_field!r}.\n"
                "Use a different --out-dir, or --force to rebuild from scratch."
            )
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
    if maxtext:
        ensure_pad_token(tokenizer)
    vocab_size = len(tokenizer)
    add_bos = maxtext and tokenizer.bos_token_id is not None

    log(f"output directory : {out_dir}")
    log(f"format           : {args.format}")
    streams = source_splits(spec)
    log(f"source corpus    : {describe_source(spec)}, validation from its {streams['valid']} split")
    log(f"tokenizer        : {tokenizer_name} ({tokenizer_type})")

    # Overwriting the saved tokenizer while keeping existing data files would
    # leave a corpus and a vocabulary that disagree, and a larger new vocab
    # would not even trip the max-token-id check in verification.
    if previous and not args.force:
        recorded = previous.get("vocab_size")
        if recorded is not None and recorded != vocab_size:
            raise SystemExit(
                f"{out_dir} holds a dataset tokenised with a {recorded}-token vocabulary, "
                f"but {tokenizer_name} has {vocab_size}.\nUse a different --out-dir, "
                "or --force to rebuild from scratch."
            )
        if maxtext and previous.get("seq_length") not in (None, args.seq_length):
            raise SystemExit(
                f"{out_dir} holds rows of up to {previous['seq_length']} tokens, but "
                f"--seq-length is {args.seq_length}.\nUse a different --out-dir, or --force to rebuild."
            )
    if not reuse_local:
        tokenizer.save_pretrained(tokenizer_dir)

    # Workers load the tokenizer from disk; keep them off the network entirely.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    if maxtext:
        dtype_name = "int32"
        pad_id = maxtext_pad_id(tokenizer)
        log(
            f"vocab {vocab_size}, bos id {tokenizer.bos_token_id if add_bos else None}, "
            f"eos id {tokenizer.eos_token_id}, pad id {pad_id}, rows of <= {args.seq_length} tokens"
        )
        prefixes = {"valid": out_dir / "valid", "train": out_dir / "train"}
        shards = {"valid": max(1, args.num_shards // 16), "train": args.num_shards}
        have_all = previous is not None and all(parquet_files(p) for p in prefixes.values())

        def writer_for(name):
            return lambda: ParquetShardWriter(str(prefixes[name]), shards[name], args.seq_length)

        def verify(name):
            return verify_maxtext(
                str(prefixes[name]), str(tokenizer_dir), vocab_size, args.seq_length, pad_id
            )

    else:
        import numpy
        from megatron.core.datasets.indexed_dataset import DType

        dtype = DType.optimal_dtype(vocab_size)
        dtype_name = numpy.dtype(dtype).name
        log(f"vocab {vocab_size}, eod id {tokenizer.eos_token_id}, bin dtype {dtype_name}")
        prefixes = {"valid": out_dir / "valid_text_document", "train": out_dir / "train_text_document"}
        have_all = previous is not None and all(Path(f"{p}.idx").exists() for p in prefixes.values())

        def writer_for(name):
            return lambda: MegatronWriter(str(prefixes[name]), dtype)

        def verify(name):
            return verify_megatron(str(prefixes[name]), str(tokenizer_dir), vocab_size)

    targets = {"valid": args.valid_tokens, "train": args.target_tokens}
    # The stream is deterministic, so a rebuild is a superset of what is there:
    # extend a dataset that is too small (say, built for a probe) rather than
    # hand a longer run less data than it asked for.
    built = ((previous or {}).get("splits", {}).get("train") or {}).get("tokens", 0)
    if have_all and not args.force and built < args.target_tokens * 0.95:
        log(
            f"the train split holds {built/1e6:.0f}M tokens, {args.target_tokens/1e6:.0f}M requested; rebuilding"
        )
        have_all = False
    if have_all and not args.force:
        log("datasets already exist, skipping build (use --force to rebuild)")
    else:
        # The manifest is what marks a dataset complete; without it an interrupted
        # build (a new .bin next to the old .idx) would be reused as finished.
        manifest_path.unlink(missing_ok=True)
        build_splits(
            spec,
            [(name, writer_for(name), targets[name], streams[name]) for name in ("valid", "train")],
            str(tokenizer_dir),
            add_bos,
            args.num_proc,
            args.batch_size,
            args.min_chars,
            args.text_field,
        )

    summary = {name: verify(name) for name in ("valid", "train")}

    # A short run against a big existing dataset is fine, but silently training
    # on less data than asked for is not. A too-small dataset was rebuilt above,
    # so only a corpus that ran out gets here.
    train_tokens = summary["train"]["tokens"]
    if train_tokens < args.target_tokens * 0.95:
        log(
            f"WARNING: train split holds {train_tokens/1e6:.0f}M tokens but "
            f"{args.target_tokens/1e6:.0f}M were requested; the {args.source} corpus ran out. "
            "Use a larger --source or a shorter run."
        )

    # The manifest is what check_config.py uses to prove the dataset and the
    # training config agree on a vocabulary.
    manifest = {
        "format": args.format,
        "source": spec["repo"],
        "source_subset": spec["name"],
        "source_splits": streams,
        "text_field": args.text_field,
        # Record what the data was actually tokenised with, which is not
        # necessarily what this invocation asked for.
        "tokenizer": (previous or {}).get("tokenizer", tokenizer_name) if reuse_local else tokenizer_name,
        "tokenizer_type": tokenizer_type,
        "tokenizer_dir": str(tokenizer_dir),
        "vocab_size": vocab_size,
        "eod_id": tokenizer.eos_token_id,
        "dtype": dtype_name,
    }
    if maxtext:
        manifest.update(
            {
                "bos_id": tokenizer.bos_token_id if add_bos else None,
                "pad_id": pad_id,
                "seq_length": args.seq_length,
                "column": "tokens",
            }
        )
    manifest["splits"] = summary
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"manifest written to {manifest_path}")

    print()
    print("Point your convergence config at this dataset:")
    if maxtext:
        print("      dataset_type: grain")
        print("      grain_file_type: parquet")
        print(f"      grain_train_files: {summary['train']['pattern']}")
        print(f"      grain_eval_files: {summary['valid']['pattern']}")
        print('      train_data_columns: ["tokens"]')
        print('      eval_data_columns: ["tokens"]')
        print("      tokenize_train_data: false")
        print("      tokenize_eval_data: false")
        print("      tokenizer_type: huggingface")
        print(f"      tokenizer_path: {tokenizer_dir}")
        print(f"      max_target_length: {args.seq_length}  # or longer")
    else:
        print(f"      tokenizer_type: {tokenizer_type}")
        print(f"      tokenizer_model: {tokenizer_dir}")
        print(f"      train_data_path: [{prefixes['train']}]")
        print(f"      valid_data_path: [{prefixes['valid']}]")
        print(f"      test_data_path: [{prefixes['valid']}]")
    print()
    print(f"Or export PRIMUS_CONVERGENCE_DATA={out_dir} and use the bundled configs.")


if __name__ == "__main__":
    main()
    # The HF streaming reader leaves background threads that trip up CPython's
    # finaliser; everything is written and flushed by here.
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)
