from __future__ import annotations

import logging
from typing import Any

from primus.moe_umco.config import load_umco_config
from primus.moe_umco.megatron.adapter import build_dispatch_impl, build_gather_impl
from primus.moe_umco.orchestrator import UnifiedMoECommOrchestrator
from primus.moe_umco.types import MoEWorldInfo

logger = logging.getLogger("primus.moe_umco")


def _safe_dist_world() -> tuple[int, int, int]:
    try:
        import torch.distributed as dist
    except Exception:
        return 1, 0, 0
    if dist.is_available() and dist.is_initialized():
        rank = int(dist.get_rank())
        world = int(dist.get_world_size())
        return world, rank, rank
    return 1, 0, 0


def maybe_wrap_megatron_moe_dispatcher(
    original_dispatcher: type[Any], exp_config: dict[str, Any] | None
) -> type[Any]:
    cfg = load_umco_config(exp_config)
    if not cfg.enable:
        return original_dispatcher

    original_token_permutation = getattr(original_dispatcher, "token_permutation", None)
    original_token_unpermutation = getattr(original_dispatcher, "token_unpermutation", None)
    if original_token_permutation is None or original_token_unpermutation is None:
        logger.warning(
            "UMCO enabled, but dispatcher %s lacks token_permutation/token_unpermutation", original_dispatcher
        )
        return original_dispatcher

    class UmcoMegatronDispatcher(original_dispatcher):  # type: ignore[misc, valid-type]
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            super().__init__(*args, **kwargs)
            world_size, rank, local_rank = _safe_dist_world()
            config = getattr(self, "config", None)
            ep_size = int(getattr(config, "expert_model_parallel_size", 1))
            tp_size = int(getattr(config, "expert_tensor_parallel_size", 1))
            pp_size = int(getattr(config, "pipeline_model_parallel_size", 1))

            self._umco_orchestrator = UnifiedMoECommOrchestrator(
                cfg=cfg,
                world_info=MoEWorldInfo(
                    world_size=world_size,
                    rank=rank,
                    local_rank=local_rank,
                    ep_size=ep_size,
                    tp_size=tp_size,
                    pp_size=pp_size,
                ),
            )
            self._umco_dispatcher = None

        def token_permutation(self, hidden_states: Any, probs: Any, routing_map: Any):  # type: ignore[override]
            max_tokens = (
                int(hidden_states.shape[0] * hidden_states.shape[1])
                if hidden_states.dim() >= 2
                else int(hidden_states.shape[0])
            )
            dtype_bytes = int(hidden_states.element_size()) if hasattr(hidden_states, "element_size") else 2
            self._umco_dispatcher = self._umco_orchestrator.get_dispatcher(
                baseline_dispatch_impl=build_dispatch_impl(
                    lambda hs, pr, rm, **kw: original_token_permutation(self, hs, pr, rm, **kw)
                ),
                baseline_gather_impl=build_gather_impl(
                    lambda out, **kw: original_token_unpermutation(self, out, **kw)
                ),
                baseline_dispatch_fn_ref=original_token_permutation,
                baseline_gather_fn_ref=original_token_unpermutation,
                max_tokens=max_tokens,
                dtype_bytes=dtype_bytes,
            )
            result = self._umco_dispatcher.dispatch(
                hidden_states=hidden_states, routing=routing_map, probs=probs
            )
            return result.hidden_states, result.tokens_per_expert

        def token_unpermutation(self, hidden_states: Any, bias: Any = None):  # type: ignore[override]
            if self._umco_dispatcher is None:
                return original_token_unpermutation(self, hidden_states, bias)
            output = self._umco_dispatcher.gather(expert_out=hidden_states, routing=None, bias=bias)
            return output, None

    UmcoMegatronDispatcher.__name__ = f"UmcoWrapped{original_dispatcher.__name__}"
    return UmcoMegatronDispatcher
