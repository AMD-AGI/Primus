#!/usr/bin/env python3
# Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
# See LICENSE for license information.
"""Prepare the smoke fixture from a separately downloaded, pinned ShareGPT file."""

import argparse
import hashlib
import json
from pathlib import Path


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare(source, output, manifest):
    spec = manifest["dataset"]
    if sha256(source) != spec["source_sha256"]:
        raise ValueError("Source checksum differs from the pinned dataset")
    rows = json.loads(source.read_text(encoding="utf-8"))
    selected, seen = [], set()
    for row in rows:
        row_id = str(row.get("id", ""))
        messages = row.get("conversations", [])
        if not row_id or row_id in seen or not messages or len(messages) % 2:
            continue
        if any(
            message.get("from") != ("human" if i % 2 == 0 else "gpt")
            or not isinstance(message.get("value"), str)
            or not message["value"].strip()
            for i, message in enumerate(messages)
        ):
            continue
        selected.append(
            {
                "id": row_id,
                "conversations": [
                    {"role": "user" if i % 2 == 0 else "assistant", "content": message["value"]}
                    for i, message in enumerate(messages)
                ],
            }
        )
        seen.add(row_id)
        if len(selected) == spec["rows"]:
            break
    if len(selected) != spec["rows"]:
        raise ValueError("Not enough valid conversations")
    payload = "".join(
        json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n" for row in selected
    ).encode("utf-8")
    checksum = hashlib.sha256(payload).hexdigest()
    if spec["fixture_sha256"] and checksum != spec["fixture_sha256"]:
        raise ValueError("Prepared fixture differs from the frozen checksum")
    # Never silently replace an existing fixture.
    with output.open("xb") as stream:
        stream.write(payload)
    return {"rows": len(selected), "bytes": len(payload), "sha256": checksum}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument(
        "--manifest", type=Path, default=Path(__file__).with_name("specforge_smoke_fixture.json")
    )
    args = parser.parse_args()
    print(
        json.dumps(
            prepare(args.source, args.output, json.loads(args.manifest.read_text(encoding="utf-8"))), indent=2
        )
    )


if __name__ == "__main__":
    main()
