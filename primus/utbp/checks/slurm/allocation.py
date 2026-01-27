import socket
from primus.utbp.checks.registry import register
from primus.utbp.result import CheckResult
from primus.utbp.checks.slurm.utils import expand_nodelist, try_scontrol_hostnames

@register
class SlurmInsideAllocationCheck:
    name = "slurm.allocation.inside"
    scope = "cluster"
    supported_runtimes = {"slurm"}
    REQUIRED = ["SLURM_JOB_ID", "SLURM_NTASKS", "SLURM_NODELIST", "SLURM_NNODES"]

    def run(self, ctx):
        missing = [k for k in self.REQUIRED if not ctx.env.get(k)]
        if missing:
            return CheckResult.fail(
                check=self.name,
                summary="Not running inside a valid Slurm allocation",
                details={"missing_env": missing},
            )
        return CheckResult.pass_(
            check=self.name,
            summary="Running inside a Slurm allocation",
            details={k: ctx.env.get(k) for k in self.REQUIRED},
        )

@register
class SlurmNodeListConsistencyCheck:
    name = "slurm.allocation.nodelist_consistency"
    scope = "cluster"
    supported_runtimes = {"slurm"}

    def run(self, ctx):
        nodelist = ctx.env.get("SLURM_NODELIST", "")
        nnodes = int(ctx.env.get("SLURM_NNODES", "0") or "0")
        if not nodelist or nnodes <= 0:
            return CheckResult.fail(
                check=self.name,
                summary="Missing SLURM_NODELIST or invalid SLURM_NNODES",
                details={"SLURM_NODELIST": nodelist, "SLURM_NNODES": nnodes},
            )
        ok, names, err = try_scontrol_hostnames(nodelist)
        if not ok:
            names = expand_nodelist(nodelist)
        if len(names) != nnodes:
            return CheckResult.fail(
                check=self.name,
                summary="Expanded nodelist size mismatch with SLURM_NNODES",
                details={
                    "SLURM_NODELIST": nodelist,
                    "SLURM_NNODES": nnodes,
                    "expanded_count": len(names),
                    "expanded_sample": names[:20],
                    "scontrol_error": err,
                },
            )
        host = socket.gethostname()
        in_list = (host in names) or any(host.startswith(x) for x in names) or any(x.startswith(host) for x in names)
        if not in_list:
            return CheckResult.warn(
                check=self.name,
                summary="Current hostname not found in expanded SLURM_NODELIST (may be alias/fqdn mismatch)",
                details={"hostname": host, "expanded_sample": names[:20]},
            )
        return CheckResult.pass_(
            check=self.name,
            summary="Nodelist expansion consistent",
            details={"expanded_count": len(names), "expanded_sample": names[:20]},
        )
