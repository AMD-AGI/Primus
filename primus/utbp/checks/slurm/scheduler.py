from primus.utbp.checks.registry import register
from primus.utbp.result import CheckResult
from primus.utbp.checks.slurm.utils import run_cmd, write_text

@register
class SlurmSchedulerPingCheck:
    name = "slurm.scheduler.ping"
    scope = "cluster"
    supported_runtimes = {"slurm"}

    def run(self, ctx):
        r = run_cmd(["scontrol", "ping"], timeout_s=10)
        log_path = f"{ctx.artifact_dir}/logs/slurm_scontrol_ping.log"
        write_text(log_path, (r.out or "") + ("\n" + r.err if r.err else ""))
        if r.rc != 0:
            return CheckResult.fail(
                check=self.name,
                summary="Slurm scheduler unreachable (scontrol ping failed)",
                details={"rc": r.rc, "stderr": r.err.strip()},
                evidence={"scontrol_ping_log": log_path},
            )
        return CheckResult.pass_(
            check=self.name,
            summary="Slurm scheduler reachable",
            details={"output": r.out.strip()},
            evidence={"scontrol_ping_log": log_path},
        )
