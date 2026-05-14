"""Tests for `pilot.tools.constraint`.

Coverage:

1. The classic parallelism arithmetic gate (`tp*pp*ep*cp <= world`,
   `gbs % (mbs*dp) == 0`).
2. The Megatron-side hard-mutex / required-companion table introduced after
   session 20260513T024603Z (`axis_taxonomy.md §2.14`). These are the rules
   that previously let INVALID_CONFIG candidates through to the trial path
   — see `IMPL_VS_DESIGN.md §3`.
3. `check_env` warnings for profile×HIPBLASLT and the
   `HSA_ENABLE_INTERRUPT=0` measured-regression block.
"""

from __future__ import annotations

from pilot.tools import constraint as c

_CLUSTER_8GPU = {
    "cluster_id": "test",
    "mode": "single",
    "single": {"max_local_gpus": 8},
}


def _plan(**overrides) -> dict:
    return {"modules": {"pre_trainer": {"overrides": overrides}}}


# ---------------------------------------------------------------------------
# Pre-existing parallelism gate — sanity check (must still pass)
# ---------------------------------------------------------------------------


def test_parallelism_arithmetic_pass() -> None:
    out = c.check(
        _plan(
            tensor_model_parallel_size=2,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=4,
            context_parallel_size=1,
            micro_batch_size=1,
            global_batch_size=8,
        ),
        _CLUSTER_8GPU,
    )
    assert out["valid"], out["violations"]


def test_parallelism_exceeds_world_violation() -> None:
    out = c.check(
        _plan(
            tensor_model_parallel_size=4,
            pipeline_model_parallel_size=4,
            expert_model_parallel_size=2,
            context_parallel_size=1,
        ),
        _CLUSTER_8GPU,
    )
    assert not out["valid"]
    assert any("exceeds world_size" in v for v in out["violations"])


# ---------------------------------------------------------------------------
# Megatron mutex / required-companion table (axis_taxonomy.md §2.14)
# Each rule below corresponds to a real failure we hit in session R3.
# ---------------------------------------------------------------------------


def test_mutex_cuda_graph_impl_and_enable_cuda_graph() -> None:
    """MUTEX-CG-IMPL: setting both --enable-cuda-graph and --cuda-graph-impl
    is rejected by Megatron's arguments.py."""
    out = c.check(
        _plan(cuda_graph_impl="local", enable_cuda_graph=True),
        _CLUSTER_8GPU,
    )
    assert not out["valid"]
    assert any("MUTEX-CG-IMPL" in v for v in out["violations"])


def test_req_pp_defer_embedding_wgrad_blocks_pp1() -> None:
    """REQ-PP-DEFER-EMB: defer_embedding_wgrad_compute requires pp>=2."""
    out = c.check(
        _plan(defer_embedding_wgrad_compute=True, pipeline_model_parallel_size=1),
        _CLUSTER_8GPU,
    )
    assert not out["valid"]
    assert any("REQ-PP-DEFER-EMB" in v for v in out["violations"])


def test_req_pp_defer_embedding_wgrad_passes_with_pp2() -> None:
    out = c.check(
        _plan(
            defer_embedding_wgrad_compute=True,
            pipeline_model_parallel_size=2,
            tensor_model_parallel_size=1,
            expert_model_parallel_size=1,
        ),
        _CLUSTER_8GPU,
    )
    # No DEFER-EMB violation; arithmetic still must hold.
    assert not any("REQ-PP-DEFER-EMB" in v for v in out["violations"])


def test_req_pp_overlap_p2p_requires_pp_and_vpp() -> None:
    out = c.check(
        _plan(overlap_p2p_communication=True, pipeline_model_parallel_size=1),
        _CLUSTER_8GPU,
    )
    assert any(
        "REQ-PP-OVRLP-P2P" in v and "pipeline_model_parallel_size >= 2" in v for v in out["violations"]
    )

    out2 = c.check(
        _plan(
            overlap_p2p_communication=True,
            pipeline_model_parallel_size=2,
            virtual_pipeline_model_parallel_size=1,
        ),
        _CLUSTER_8GPU,
    )
    assert any(
        "REQ-PP-OVRLP-P2P" in v and "virtual_pipeline_model_parallel_size > 1" in v
        for v in out2["violations"]
    )


def test_mutex_deepep_router_dtype_must_be_fp32() -> None:
    """MUTEX-DEEPEP-ROUTER: use_turbo_deepep + non-fp32 router → DeepEP runtime error."""
    out = c.check(
        _plan(use_turbo_deepep=True, moe_router_dtype="bf16"),
        _CLUSTER_8GPU,
    )
    assert any("MUTEX-DEEPEP-ROUTER" in v for v in out["violations"])

    out_ok = c.check(
        _plan(use_turbo_deepep=True, moe_router_dtype="fp32"),
        _CLUSTER_8GPU,
    )
    assert not any("MUTEX-DEEPEP-ROUTER" in v for v in out_ok["violations"])


def test_mutex_deepep_shared_expert_overlap() -> None:
    """MUTEX-DEEPEP-SHAREDOVRLP: turbo_deepep + moe_shared_expert_overlap=true → hang."""
    out = c.check(
        _plan(use_turbo_deepep=True, moe_shared_expert_overlap=True),
        _CLUSTER_8GPU,
    )
    assert any("MUTEX-DEEPEP-SHAREDOVRLP" in v for v in out["violations"])


def test_req_deepep_load_balancing_must_be_true() -> None:
    """REQ-DEEPEP-LBAL: turbo_deepep + force_load_balancing=false → real router hang."""
    out = c.check(
        _plan(use_turbo_deepep=True, moe_router_force_load_balancing=False),
        _CLUSTER_8GPU,
    )
    assert any("REQ-DEEPEP-LBAL" in v for v in out["violations"])


def test_clean_plan_passes_all_mutexes() -> None:
    """The session's final champion config (R9) must pass without warnings."""
    out = c.check(
        _plan(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            expert_model_parallel_size=8,
            context_parallel_size=1,
            micro_batch_size=1,
            global_batch_size=8,
            turbo_deepep_num_cu=80,
            fp8_recipe="delayed",
            apply_rope_fusion=True,
            use_turbo_deepep=True,
            moe_router_force_load_balancing=True,
            moe_shared_expert_overlap=False,
        ),
        _CLUSTER_8GPU,
    )
    assert out["valid"], out["violations"]


# ---------------------------------------------------------------------------
# check_env additions
# ---------------------------------------------------------------------------


def test_check_env_profile_hipblaslt_warning() -> None:
    out = c.check_env({"PRIMUS_HIPBLASLT_TUNING": "1"}, {})
    assert out["valid"]
    assert any("MUTEX-PROFILE-HIPBLASLT" in w for w in out["warnings"])


def test_check_env_hsa_interrupt_off_blocks_without_ack() -> None:
    """HSA_ENABLE_INTERRUPT=0 was -13.28% TFLOPS; require explicit ack."""
    out = c.check_env({"HSA_ENABLE_INTERRUPT": "0"}, {})
    assert not out["valid"]
    assert any("WARN-HSA-INTERRUPT-OFF" in v for v in out["violations"])


def test_check_env_hsa_interrupt_off_allowed_with_ack() -> None:
    out = c.check_env(
        {"HSA_ENABLE_INTERRUPT": "0", "__axis_meta__": {"acknowledge_regression": True}},
        {},
    )
    assert out["valid"], out["violations"]
