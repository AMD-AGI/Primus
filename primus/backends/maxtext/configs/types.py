from pydantic.fields import Field

from MaxText.configs.types import MoEGeneral, DevelopmentAndDebugging

class PrimusMoEGeneral(MoEGeneral):
    expert_balance: bool = Field(False, description="Whether to use expert balancing.")


class PrimusDevelopmentAndDebugging(DevelopmentAndDebugging):
    jax_distributed_heartbeat_timeout_seconds: int = Field(
        100, description="How long before a missing heartbeat marks a task as dead. Increase for slow NFS checkpoint restores."
    )