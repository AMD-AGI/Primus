# Modifications Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import hashlib
from concurrent.futures import ThreadPoolExecutor


def hash_file_md5(file_path, chunk_size=4194304):  # Default chunk size 4MB.
    """Hashes a single file using MD5 and returns the hex digest."""
    md5_hash = hashlib.md5()
    try:
        with open(file_path, "rb") as f:
            while chunk := f.read(chunk_size):
                md5_hash.update(chunk)
    except FileNotFoundError:
        return None
    return md5_hash.hexdigest()


def hash_files(file_paths):
    """Hashes an explicit list of files in parallel using MD5.

    Only the given files are considered, so unrelated content in the same
    directory (e.g. checkpoints) does not affect the result.
    """
    hashes = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(hash_file_md5, p) for p in file_paths]
        for future in futures:
            file_hash = future.result()
            if file_hash:
                hashes.append(file_hash)

    # Combine the individual file hashes into a single hash
    combined_hash = hashlib.md5()
    for file_hash in sorted(hashes):  # Sort to maintain consistency
        combined_hash.update(file_hash.encode())

    return combined_hash.hexdigest()
