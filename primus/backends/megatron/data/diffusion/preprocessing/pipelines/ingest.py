# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# Licensed under the Apache License, Version 2.0.

"""
Streaming Arrow-to-WebDataset pipeline for MLPerf Flux datasets.

Downloads Apache Arrow IPC files from MLCommons R2 storage using a
concurrent prefetch pool, converts numpy-serialized bfloat16 samples
directly into WebDataset tar shards, then deletes each Arrow file to
minimize peak disk usage.

Reduces the 6 TB disk requirement to roughly the final dataset size
(~1.2 TB for cc12m) plus a small prefetch buffer of Arrow files.
"""

import io
import json
import logging
import queue
import tarfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import pyarrow.ipc

from ..download import download_with_backoff, fetch_manifest
from .base import DatasetPipeline

logger = logging.getLogger(__name__)

ARROW_COLUMNS = ("t5_encodings", "clip_encodings", "mean", "logvar")
WDS_KEYS = ("t5.bytes", "clip.bytes", "mean.bytes", "logvar.bytes")

_SENTINEL = None


def _arrow_to_tar(
    arrow_path: Path,
    tar_path: Path,
    global_sample_offset: int,
) -> int:
    """Convert one Arrow IPC stream file into a WebDataset tar shard.

    Reads the Arrow stream, writes each row as a set of compound-key
    entries (e.g. ``00000000.t5.bytes``) into a tar archive. Uses the
    ``__key__`` column from the Arrow file when available, otherwise
    generates sequential keys.

    Returns the number of samples written.
    """
    reader = pyarrow.ipc.open_stream(str(arrow_path))
    table = reader.read_all()
    num_rows = table.num_rows

    has_key_col = "__key__" in table.schema.names

    tar_path.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(str(tar_path), "w") as tar:
        for row_idx in range(num_rows):
            if has_key_col:
                base_name = table.column("__key__")[row_idx].as_py()
            else:
                base_name = f"{global_sample_offset + row_idx:08d}"

            for col_name, wds_key in zip(ARROW_COLUMNS, WDS_KEYS):
                col = table.column(col_name)
                raw_bytes = col[row_idx].as_py()
                if raw_bytes is None:
                    continue
                data = bytes(raw_bytes)

                entry_name = f"{base_name}.{wds_key}"
                info = tarfile.TarInfo(name=entry_name)
                info.size = len(data)
                tar.addfile(info, io.BytesIO(data))

            meta = json.dumps({"key": base_name}).encode("utf-8")
            meta_info = tarfile.TarInfo(name=f"{base_name}.json")
            meta_info.size = len(meta)
            tar.addfile(meta_info, io.BytesIO(meta))

    return num_rows


class StreamingIngestPipeline(DatasetPipeline):
    """Streaming pipeline: download Arrow files from R2 -> WebDataset tars.

    Downloads are parallelized with a prefetch pool (``max_workers`` threads)
    while conversion runs sequentially on the main thread, preserving
    deterministic shard ordering.  A semaphore (``prefetch_depth``) limits
    how many Arrow files are kept on disk at once.

    Args:
        manifest_url: URL to the .uri manifest on MLCommons R2.
        input_dir: Local directory for temporary Arrow file storage.
        output_dir: Output directory for WebDataset tar shards.
        split_name: Subdirectory name for the split (e.g. 'train', 'val').
        max_files: Maximum number of Arrow files to process (None = all).
        max_workers: Number of concurrent download threads (default 4).
        prefetch_depth: Max Arrow files buffered on disk ahead of
            conversion (default 6).  Disk overhead ~ prefetch_depth x 200 MB.
    """

    def __init__(
        self,
        manifest_url: str,
        input_dir: str,
        output_dir: str,
        split_name: str = "train",
        max_files: Optional[int] = None,
        max_workers: int = 4,
        prefetch_depth: int = 6,
    ):
        self.manifest_url = manifest_url
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir) / split_name
        self.split_name = split_name
        self.max_files = max_files
        self.max_workers = max_workers
        self.prefetch_depth = prefetch_depth

    def run(self, **kwargs) -> Dict[str, int]:
        """Execute the streaming conversion pipeline.

        Returns:
            Dict with 'files_processed', 'samples_written', 'shards_created',
            'shards_skipped', 'files_failed'.
        """
        base_url, entries = fetch_manifest(self.manifest_url)

        if self.max_files is not None:
            entries = entries[: self.max_files]

        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)

        total_files = len(entries)

        logger.info("=" * 80)
        logger.info(f"Streaming Arrow->WebDataset conversion: {self.split_name}")
        logger.info(f"  Arrow files to process: {total_files}")
        logger.info(f"  Output: {self.output_dir}")
        logger.info(f"  Download workers: {self.max_workers}, " f"prefetch depth: {self.prefetch_depth}")
        logger.info("=" * 80)

        prefetch_q: queue.Queue[Union[Tuple[int, str, Path], None]] = queue.Queue()
        download_semaphore = threading.Semaphore(self.prefetch_depth)
        producer_error: List[BaseException] = []
        cancel_event = threading.Event()

        failed_files: List[Dict] = []
        failed_lock = threading.Lock()

        def _download_one(file_idx: int, md5: str, fname: str) -> Tuple[int, str, Path]:
            file_url = f"{base_url}/{fname}"
            # fname comes from a remote manifest; never let it traverse outside
            # input_dir (e.g. "../../etc/passwd"). Use the basename and verify
            # the resolved path stays under input_dir before writing.
            safe_name = Path(fname).name
            base_dir = self.input_dir.resolve()
            arrow_path = (base_dir / safe_name).resolve()
            if not safe_name or arrow_path.parent != base_dir:
                raise ValueError(f"Unsafe manifest filename rejected: {fname!r}")
            logger.info(f"  [download {file_idx + 1}/{total_files}] {fname}")
            download_with_backoff(file_url, arrow_path, expected_md5=md5)
            return file_idx, fname, arrow_path

        skipped_count = [0]

        def _producer() -> None:
            """Submit downloads and drain results concurrently.

            A separate drain thread processes completed futures in submission
            order and feeds the prefetch queue.  The semaphore acquire in the
            submit loop blocks when ``prefetch_depth`` files are already
            in-flight or queued, bounding temporary disk usage.
            """
            try:
                with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
                    futures_q: queue.Queue = queue.Queue()

                    def _drain() -> None:
                        while True:
                            item = futures_q.get()
                            if item is _SENTINEL:
                                break
                            fut, file_idx, fname = item
                            if cancel_event.is_set():
                                download_semaphore.release()
                                continue
                            try:
                                result = fut.result()
                                prefetch_q.put(result)
                            except Exception as exc:
                                download_semaphore.release()
                                logger.warning(
                                    f"  [FAILED download " f"{file_idx + 1}/{total_files}] " f"{fname}: {exc}"
                                )
                                with failed_lock:
                                    failed_files.append(
                                        {
                                            "file_idx": file_idx,
                                            "filename": fname,
                                            "error": str(exc),
                                            "stage": "download",
                                        }
                                    )

                    drain_thread = threading.Thread(target=_drain, daemon=True)
                    drain_thread.start()

                    for file_idx, (md5, fname) in enumerate(entries):
                        if cancel_event.is_set():
                            break
                        tar_path = self.output_dir / f"shard_{file_idx:06d}.tar"
                        if tar_path.exists():
                            skipped_count[0] += 1
                            logger.info(f"  [skip {file_idx + 1}/{total_files}] " f"{fname} (shard exists)")
                            continue
                        download_semaphore.acquire()
                        fut = pool.submit(_download_one, file_idx, md5, fname)
                        futures_q.put((fut, file_idx, fname))

                    futures_q.put(_SENTINEL)
                    drain_thread.join()
            except BaseException as exc:
                producer_error.append(exc)
            finally:
                prefetch_q.put(_SENTINEL)

        producer_thread = threading.Thread(target=_producer, daemon=True)
        producer_thread.start()

        global_sample_offset = 0
        total_samples = 0
        files_processed = 0

        try:
            while True:
                item = prefetch_q.get()
                if item is _SENTINEL:
                    break

                file_idx, fname, arrow_path = item

                shard_name = f"shard_{file_idx:06d}.tar"
                tar_path = self.output_dir / shard_name

                try:
                    num_samples = _arrow_to_tar(arrow_path, tar_path, global_sample_offset)
                except Exception as exc:
                    logger.warning(f"  [FAILED convert {file_idx + 1}/{total_files}] " f"{fname}: {exc}")
                    tar_path.unlink(missing_ok=True)
                    arrow_path.unlink(missing_ok=True)
                    download_semaphore.release()
                    with failed_lock:
                        failed_files.append(
                            {
                                "file_idx": file_idx,
                                "filename": fname,
                                "error": str(exc),
                                "stage": "conversion",
                            }
                        )
                    continue

                global_sample_offset += num_samples
                total_samples += num_samples
                files_processed += 1

                arrow_path.unlink(missing_ok=True)
                download_semaphore.release()

                logger.info(
                    f"[{files_processed}/{total_files}] {fname} -> {shard_name}: "
                    f"{num_samples} samples (cumulative: {total_samples})"
                )
        except BaseException:
            cancel_event.set()
            raise
        finally:
            producer_thread.join(timeout=10)

        if producer_error:
            raise RuntimeError(f"Download failed: {producer_error[0]}") from producer_error[0]

        if failed_files:
            manifest_path = self.output_dir / "failed_files.json"
            manifest_path.write_text(json.dumps(failed_files, indent=2))
            logger.warning(f"{len(failed_files)} file(s) failed. " f"See {manifest_path}")

        skipped = skipped_count[0]
        failed = len(failed_files)
        logger.info("=" * 80)
        logger.info(
            f"Done {self.split_name}: {files_processed} new + {skipped} skipped"
            f"{f' + {failed} failed' if failed else ''}"
            f" -> {files_processed + skipped} shards, "
            f"{total_samples} new samples"
        )
        logger.info("=" * 80)

        return {
            "files_processed": files_processed,
            "samples_written": total_samples,
            "shards_created": files_processed + skipped,
            "shards_skipped": skipped,
            "files_failed": failed,
        }
