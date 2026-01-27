from primus.utbp.checks.registry import register
from primus.utbp.result import CheckResult

def _parse_visible_devices(s: str) -> list[str]:
    s = (s or "").strip()
    if not s:
        return []
    return [x.strip() for x in s.split(",") if x.strip()]

@register
class SlurmGpuVisibilityCheck:
    name = "slurm.binding.gpu_visibility"
    scope = "node"
    supported_runtimes = {"slurm"}

    def run(self, ctx):
        rocr = ctx.env.get("ROCR_VISIBLE_DEVICES", "")
        hip = ctx.env.get("HIP_VISIBLE_DEVICES", "")
        cuda = ctx.env.get("CUDA_VISIBLE_DEVICES", "")
        details = {
            "ROCR_VISIBLE_DEVICES": rocr,
            "HIP_VISIBLE_DEVICES": hip,
            "CUDA_VISIBLE_DEVICES": cuda,
        }
        if not rocr and not hip and not cuda:
            return CheckResult.warn(
                check=self.name,
                summary="No *VISIBLE_DEVICES env is set; risk of GPU contention",
                details=details,
            )
        vis = _parse_visible_devices(rocr or hip or cuda)
        if len(vis) == 0:
            return CheckResult.warn(
                check=self.name,
                summary="*VISIBLE_DEVICES set but empty",
                details=details,
            )
        return CheckResult.pass_(
            check=self.name,
            summary="GPU visibility env present",
            details={**details, "parsed": vis},
        )
