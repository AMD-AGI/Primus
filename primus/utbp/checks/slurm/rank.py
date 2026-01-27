from primus.utbp.checks.registry import register
from primus.utbp.result import CheckResult

@register
class SlurmRankEnvCheck:
    name = "slurm.rank.env"
    scope = "node"
    supported_runtimes = {"slurm"}
    REQUIRED = ["SLURM_PROCID", "SLURM_LOCALID", "SLURM_NODEID", "SLURM_NTASKS"]

    def run(self, ctx):
        missing = [k for k in self.REQUIRED if ctx.env.get(k) is None]
        if missing:
            return CheckResult.fail(
                check=self.name,
                summary="Missing Slurm rank environment variables",
                details={"missing_env": missing},
            )
        procid = int(ctx.env.get("SLURM_PROCID", "0"))
        ntasks = int(ctx.env.get("SLURM_NTASKS", "0"))
        if procid < 0 or procid >= ntasks:
            return CheckResult.fail(
                check=self.name,
                summary="SLURM_PROCID out of range",
                details={"SLURM_PROCID": procid, "SLURM_NTASKS": ntasks},
            )
        return CheckResult.pass_(
            check=self.name,
            summary="Slurm rank env looks valid",
            details={k: ctx.env.get(k) for k in self.REQUIRED},
        )
