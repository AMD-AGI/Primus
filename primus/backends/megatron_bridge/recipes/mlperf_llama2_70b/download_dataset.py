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

import argparse
import glob
import subprocess
import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

from dataset_hash import hash_files
from huggingface_hub import snapshot_download

parser = argparse.ArgumentParser()
parser.add_argument("--data_dir", default="/data", type=str, help="Path to the dataset location")
args = parser.parse_args()

snapshot_download(
    "regisss/scrolls_gov_report_preprocessed_mlperf_2",
    revision="21ff1233ee3e87bc780ab719c755170148aba1cb",
    allow_patterns="*.parquet",
    local_dir=args.data_dir,
    local_dir_use_symlinks=False,
    max_workers=16,
    repo_type="dataset",
)

# Move the dataset parquets up and remove only HF's own artifacts. Do NOT
# blanket-delete data_dir: it may hold a sibling checkpoint (the old
# `find <data_dir> ! -name '*.parquet' -exec rm -rf` wiped it).
subprocess.run(
    f"mv {args.data_dir}/data/*.parquet {args.data_dir}/ && rm -rf {args.data_dir}/data {args.data_dir}/.cache",
    shell=True,
    executable="/bin/bash",
    check=True,
)

# Verify only the downloaded parquets, so other content in data_dir does not
# affect the hash.
directory_hash = hash_files(sorted(glob.glob(f"{args.data_dir}/*.parquet")))
assert (
    directory_hash == "682a5f40b790a56751bf8303554efc08"
), f"Expected hash 682a5f40b790a56751bf8303554efc08, but got {directory_hash}"
print(f"Succesfully downloaded and verified dataset with hash {directory_hash}")
