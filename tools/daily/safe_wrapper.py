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

        self.PRIMUS_REPO = "https://github.com/AMD-AGI/Primus.git"
        self.PRIMUS_TURBO_REPO = "https://github.com/AMD-AGI/Primus-Turbo.git"
        self.PRIMUS_TURBO_COMMIT = "5233748e9c5c5795a6484ab31ece47c442d29ec2"
        self.PRIMUS_WORKDIR = os.getenv("PRIMUS_WORKDIR", "")
        self.BENCHMARK_LOG_DIR = os.getenv("BENCHMARK_LOG_DIR", "")

        print(f"✅ SaFE protocol initialized")
        print(f"   NFS root path: {self.safe_nfs_path}")
        print(f"   Input file: {self.input_file}")
        print(f"   Output file: {self.output_file}")

    def create_training_command(self, args):
        """Create training command"""
        lines = [
            f"git clone --recursive {self.PRIMUS_TURBO_REPO} /tmp/Primus-Turbo",
            f"cd /tmp/Primus-Turbo &&  git checkout {self.PRIMUS_TURBO_COMMIT}",
            f"MAX_JOBS=128 pip install --cache-dir={self.PRIMUS_WORKDIR}/primus-cache --no-build-isolation --no-clean -r requirements.txt",
            f"pip3 install --no-build-isolation -e . -v",
            f"cd && git clone --recurse-submodules https://github.com/AMD-AIG-AIMA/Primus.git && cd Primus && pip install -r requirements.txt",
        ]
        for config in args.config.split(","):
            path = Path(config)
            framework = path.parts[-4]
            log_file = f"{self.BENCHMARK_LOG_DIR}/{framework}/{path.stem}.log"
            lines.append(
                f"./runner/primus-cli direct --log_file {log_file} -- train pretrain --config {config}"
            )
        lines.append("echo Finished")
        return "\n".join(lines)

    def create_input_config(self, args):
        """Create SAFE_NFS_INPUT configuration file"""

        # Build training command
        train_command = self.create_training_command(args)

        # SaFE Input configuration (JSON format)
        config = {
            "model": f"safe_training_{args.num_nodes}nodes",
            "command": train_command,
            "image": "harbor.project1.tw325.primus-safe.amd.com/proxy/rocm/primus:v25.10",
            "resources": {
                "replica": args.num_nodes,  # Number of nodes
                "gpu": str(args.gpus),  # GPUs per node
                "cpu": "64",
                "memory": "512Gi",
                "ephemeralStorage": "512Gi",
                "sharedMemory": "512Gi",
            },
            "env": {
                "SAFE_NFS_PATH": self.safe_nfs_path,
                "HF_TOKEN": os.getenv("HF_TOKEN", ""),
                "DATA_PATH": os.getenv("DATA_PATH", ""),
                "BENCHMARK_LOG_DIR": os.getenv("BENCHMARK_LOG_DIR", ""),
                "PRIMUS_WORKDIR": os.getenv("PRIMUS_WORKDIR", ""),
                "DATA_PATH": os.getenv("DATA_PATH", ""),
                "HSA_NO_SCRATCH_RECLAIM": os.getenv("HSA_NO_SCRATCH_RECLAIM", ""),
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


def main():
    parser = argparse.ArgumentParser(description="SaFE Protocol Wrapper")
    parser.add_argument("--config", required=True, help="config path")
    parser.add_argument("--gpus", type=int, default=8, help="Number of GPUs per node")
    parser.add_argument("--log_file", required=False, help="log file path")
    parser.add_argument("--num-nodes", type=int, default=2, help="Number of nodes")
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

        print(f"\n✅ All completed! Total time: {elapsed_time:.2f} seconds")
        sys.exit(0)

    except Exception as e:
        print(f"\n❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
