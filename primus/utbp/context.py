import os
import socket
import uuid
from dataclasses import dataclass, field
from typing import Optional, Dict


@dataclass
class ValidationContext:
    run_id: str
    hostname: str
    runtime: str                 # slurm | container | direct
    node_rank: Optional[int]
    world_size: Optional[int]
    env: Dict[str, str]

    output_dir: str
    artifact_dir: str

    @staticmethod
    def detect_runtime(env: Dict[str, str]) -> str:
        if "SLURM_JOB_ID" in env:
            return "slurm"
        if "CONTAINER_RUNTIME" in env:
            return "container"
        return "direct"

    @classmethod
    def from_env(cls, output_dir: str) -> "ValidationContext":
        env = dict(os.environ)

        runtime = cls.detect_runtime(env)

        node_rank = None
        world_size = None

        if runtime == "slurm":
            node_rank = int(env.get("SLURM_NODEID", "0"))
            world_size = int(env.get("SLURM_NNODES", "1"))

        run_id = f"utbp-{uuid.uuid4().hex[:8]}"
        hostname = socket.gethostname()

        artifact_dir = os.path.join(output_dir, run_id)

        os.makedirs(artifact_dir, exist_ok=True)

        return cls(
            run_id=run_id,
            hostname=hostname,
            runtime=runtime,
            node_rank=node_rank,
            world_size=world_size,
            env=env,
            output_dir=output_dir,
            artifact_dir=artifact_dir,
        )
