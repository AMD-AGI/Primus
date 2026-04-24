"""
Convert FLA's preprocessed Arrow dataset to Megatron binary format (.bin + .idx).

FAST: reads Arrow shard files directly with PyArrow, avoids HuggingFace datasets overhead.
Each FLA 2048-token sequence becomes one "document" in Megatron.
"""
import struct, time, glob, os
import numpy as np
from pathlib import Path

FLA_DATA = "/home/vanbhati@amd.com/flash-linear-attention/legacy/training/data/HuggingFaceFW/fineweb-edu/sample-10BT/train"
OUT_PREFIX = "/home/vanbhati@amd.com/Primus/data/fla_aligned/fla_fineweb_edu_10BT_text_sentence"

_INDEX_HEADER = b"MMIDIDX\x00\x00"
DTYPE_CODE_INT32 = 4
SEQ_LEN = 2048


def main():
    import pyarrow.ipc as ipc

    t0 = time.time()
    out_dir = Path(OUT_PREFIX).parent
    out_dir.mkdir(parents=True, exist_ok=True)
    bin_path = f"{OUT_PREFIX}.bin"
    idx_path = f"{OUT_PREFIX}.idx"

    shard_files = sorted(glob.glob(os.path.join(FLA_DATA, "data-*.arrow")))
    print(f"Found {len(shard_files)} Arrow shard files")

    total_tokens = 0
    num_samples = 0

    print(f"Writing {bin_path}...")
    with open(bin_path, "wb") as f_bin:
        for i, shard_path in enumerate(shard_files):
            t1 = time.time()
            # Arrow IPC stream format
            reader = ipc.open_stream(shard_path)
            while True:
                try:
                    batch = reader.read_next_batch()
                except StopIteration:
                    break
                col = batch.column("input_ids")
                # col is a ListArray of int32; .values is the flat child
                flat = col.values.to_numpy(zero_copy_only=False)
                if flat.dtype != np.int32:
                    flat = flat.astype(np.int32)
                f_bin.write(flat.tobytes())
                total_tokens += len(flat)
                num_samples += len(col)

            elapsed = time.time() - t0
            shard_elapsed = time.time() - t1
            pct = (i + 1) / len(shard_files) * 100
            print(f"  Shard {i+1:>3}/{len(shard_files)} ({pct:5.1f}%) | "
                  f"{num_samples:>10,} samples | {total_tokens:>13,} tokens | "
                  f"shard: {shard_elapsed:.1f}s | total: {elapsed:.0f}s")

    expected_tokens = num_samples * SEQ_LEN
    print(f"\n  Total: {num_samples:,} samples, {total_tokens:,} tokens")
    if total_tokens != expected_tokens:
        print(f"  WARNING: expected {expected_tokens:,} tokens (samples * {SEQ_LEN})")

    # ── Write .idx ──
    print(f"Writing {idx_path}...")
    seq_lengths = np.full(num_samples, SEQ_LEN, dtype=np.int32)
    seq_pointers = np.arange(num_samples, dtype=np.int64) * np.int64(SEQ_LEN * 4)
    doc_indices = np.arange(1, num_samples + 1, dtype=np.int64)

    with open(idx_path, "wb") as f:
        f.write(_INDEX_HEADER)
        f.write(struct.pack("<Q", 1))
        f.write(struct.pack("<B", DTYPE_CODE_INT32))
        f.write(struct.pack("<Q", num_samples))
        f.write(struct.pack("<Q", num_samples))
        f.write(seq_lengths.tobytes(order="C"))
        f.write(seq_pointers.tobytes(order="C"))
        f.write(doc_indices.tobytes(order="C"))

    # ── Verify ──
    print("\nVerifying...")
    data = np.memmap(bin_path, dtype=np.int32, mode='r')
    print(f"  Bin file tokens: {len(data):,}")
    print(f"  First 10: {data[:10].tolist()}")
    print(f"  Last 10:  {data[-10:].tolist()}")

    # Cross-check with first shard
    reader = ipc.open_stream(shard_files[0])
    batch = reader.read_next_batch()
    first_sample = batch.column("input_ids")[0].as_py()
    assert data[:10].tolist() == first_sample[:10], "MISMATCH at start!"
    print("  OK — tokens match!")

    bin_sz = Path(bin_path).stat().st_size
    idx_sz = Path(idx_path).stat().st_size
    print(f"\n  {bin_path}: {bin_sz / 1e9:.2f} GB")
    print(f"  {idx_path}: {idx_sz / 1e6:.2f} MB")
    print(f"\nDone! Total: {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
