# Modifications Copyright (c) 2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
import subprocess
import sys
from pathlib import Path

_RECIPE_DIR = Path(__file__).resolve().parent
if str(_RECIPE_DIR) not in sys.path:
    sys.path.insert(0, str(_RECIPE_DIR))

import numpy as np
import pandas as pd
from dataset_hash import hash_files


def convert(data_dir, split):
    df = pd.read_parquet(f"{data_dir}/{split}-00000-of-00001.parquet")
    transformed_data = df.apply(lambda row: transform_row(row), axis=1).tolist()
    np.save(f"{data_dir}/{split}", transformed_data)


def transform_row(row):
    return {
        "input_ids": row["input_ids"],
        "loss_mask": [int(x != -100) for x in row["labels"]],
        "seq_start_id": [0],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert dataset script")
    parser.add_argument("--data_dir", type=str, default="/data", help="The directory of the data files")
    args = parser.parse_args()

    convert(args.data_dir, "train")
    convert(args.data_dir, "validation")

    subprocess.run(
        f"rm {args.data_dir}/train-00000-of-00001.parquet {args.data_dir}/validation-00000-of-00001.parquet",
        shell=True,
        executable="/bin/bash",
        check=True,
    )
    # Verify only the produced files, so other content in data_dir does not
    # affect the hash.
    directory_hash = hash_files([f"{args.data_dir}/train.npy", f"{args.data_dir}/validation.npy"])
    print(f"Succesfully converted dataset with hash {directory_hash}")
