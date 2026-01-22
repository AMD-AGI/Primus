#!/usr/bin/env python3
"""
SaFE Protocol Wrapper

Features:
1. Write task configuration to SAFE_NFS_INPUT
2. Monitor SAFE_NFS_OUTPUT and wait for task completion
3. Collect results and output to GitHub Summary
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path


class SafeWrapper:
    def __init__(self):
        # Get SaFE protocol paths from environment variables
        self.safe_nfs_path = os.getenv("SAFE_NFS_PATH")
        self.safe_nfs_input = os.getenv("SAFE_NFS_INPUT")  # File name, e.g., "SAFE_INPUT"
        self.safe_nfs_output = os.getenv("SAFE_NFS_OUTPUT")  # File name, e.g., "SAFE_OUTPUT"

        # Validate environment variables
        if not all([self.safe_nfs_path, self.safe_nfs_input, self.safe_nfs_output]):
            raise RuntimeError(
                "❌ Missing required environment variables!\n"
                "Please ensure running on a Primus-SaFE platform runner.\n"
                f"SAFE_NFS_PATH: {self.safe_nfs_path}\n"
                f"SAFE_NFS_INPUT: {self.safe_nfs_input}\n"
                f"SAFE_NFS_OUTPUT: {self.safe_nfs_output}"
            )

        # Build complete paths
        self.input_file = Path(self.safe_nfs_path) / self.safe_nfs_input
        self.output_file = Path(self.safe_nfs_path) / self.safe_nfs_output

        print(f"✅ SaFE protocol initialized")
        print(f"   NFS root path: {self.safe_nfs_path}")
        print(f"   Input file: {self.input_file}")
        print(f"   Output file: {self.output_file}")

    def create_input_config(self, args):
        """Create SAFE_NFS_INPUT configuration file"""

        # Build training command
        train_command = (
            f"python3 /workspace/train.py "
            f"--model {args.model} "
            f"--gpus {args.gpus} "
            f"--batch-size {args.batch_size} "
            f"--output-dir {self.safe_nfs_path}/output"
        )

        # SaFE Input configuration (JSON format)
        config = {
            "model": f"safe_training_{args.model}",
            "command": train_command,
            "image": "harbor.tw325.primus-safe.amd.com/sync/rocm/training:latest",
            "resources": {
                "replica": args.num_nodes,  # Number of nodes
                "gpu": str(args.gpus),  # GPUs per node
                "cpu": "96",
                "memory": "1024Gi",
                "ephemeralStorage": "512Gi",
                "sharedMemory": "512Gi",
            },
            "env": {
                "SAFE_NFS_PATH": self.safe_nfs_path,
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                "NCCL_DEBUG": "INFO",
                "NUM_GPUS": str(args.gpus),
                "NNODES": str(args.num_nodes),
            },
            "timeout": 18000,  # 5 hours timeout
        }

        # Write to SAFE_NFS_INPUT file
        self.input_file.parent.mkdir(parents=True, exist_ok=True)

        with open(self.input_file, "w") as f:
            json.dump(config, f, indent=2)

        print(f"✅ Written to SAFE_NFS_INPUT: {self.input_file}")
        print(f"   Configuration:")
        print(json.dumps(config, indent=2, ensure_ascii=False))

        return config

    def wait_for_completion(self, timeout=18000, poll_interval=10):
        """Monitor SAFE_NFS_OUTPUT and wait for task completion"""

        start_time = time.time()

        print(f"\n⏳ Waiting for training task to complete...")
        print(f"   Monitoring file: {self.output_file}")
        print(f"   Timeout: {timeout} seconds")
        print(f"   Poll interval: {poll_interval} seconds")

        while True:
            elapsed = time.time() - start_time

            # Check timeout
            if elapsed > timeout:
                raise TimeoutError(f"❌ Training task timed out ({timeout} seconds)")

            # Check if output file exists
            if self.output_file.exists():
                try:
                    with open(self.output_file, "r") as f:
                        result = json.load(f)

                    phase = result.get("phase", "Unknown")
                    print(f"\n✅ Completion signal detected: {phase}")

                    if phase == "Succeeded":
                        print("🎉 Training task completed successfully!")
                        return result
                    elif phase == "Failed":
                        raise RuntimeError(f"❌ Training task failed!")
                    elif phase == "Stopped":
                        raise RuntimeError(f"⚠️ Training task was stopped!")
                    else:
                        raise RuntimeError(f"❌ Unknown status: {phase}")

                except json.JSONDecodeError:
                    print(f"⚠️ Output file format error, continuing to wait...")

            # Display progress
            print(f"   Elapsed: {int(elapsed)}s / {timeout}s", end="\r")
            time.sleep(poll_interval)

    def collect_results(self, output_csv):
        """Collect training results from SAFE_NFS_PATH

        Note: SAFE_NFS_PATH is a temporary directory that will be automatically
        deleted after task completion. All results must be collected at this time.
        """

        print(f"\n📊 Collecting training results...")
        print(
            f"   ⚠️ SAFE_NFS_PATH will be automatically deleted after task completion, collect results promptly"
        )

        # Assume user has written results to the specified file
        results_file = Path(self.safe_nfs_path) / output_csv

        if results_file.exists():
            print(f"   ✅ Results file found: {results_file.name}")
            return {"csv_file": str(results_file)}
        else:
            print(f"   ⚠️ Results file not found: {output_csv}")
            return {"csv_file": None}

    def write_github_summary(self, results, elapsed_time):
        """Write to GitHub Actions Job Summary"""

        summary_file = os.getenv("GITHUB_STEP_SUMMARY")
        if not summary_file:
            print("⚠️ GITHUB_STEP_SUMMARY environment variable not found, skipping summary generation")
            return

        print(f"\n📝 Generating GitHub Summary...")

        with open(summary_file, "a") as f:
            f.write("## 🎯 Training Task Completed\n\n")
            f.write(f"**Execution time**: {elapsed_time:.2f} seconds\n\n")

            if results.get("csv_file"):
                f.write("### 📊 Performance Data\n\n")
                csv_name = Path(results["csv_file"]).name
                f.write(f"- `{csv_name}`\n\n")
            else:
                f.write("⚠️ Results file not found\n\n")

        print("✅ GitHub Summary generated")


def main():
    parser = argparse.ArgumentParser(description="SaFE Protocol Wrapper")
    parser.add_argument("--model", required=True, help="Model name")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs per node")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--num-nodes", type=int, default=2, help="Number of nodes")
    parser.add_argument("--output-csv", default="results.csv", help="Output CSV filename")
    parser.add_argument("--timeout", type=int, default=18000, help="Timeout in seconds")
    args = parser.parse_args()

    try:
        # Initialize SaFE wrapper
        wrapper = SafeWrapper()

        # 1. Create and write to SAFE_NFS_INPUT
        wrapper.create_input_config(args)

        # 2. Wait for task completion
        start_time = time.time()
        result = wrapper.wait_for_completion(timeout=args.timeout)
        elapsed_time = time.time() - start_time

        # 3. Collect results
        results = wrapper.collect_results(args.output_csv)

        # 4. Write to GitHub Summary
        wrapper.write_github_summary(results, elapsed_time)

        print(f"\n✅ All completed! Total time: {elapsed_time:.2f} seconds")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
