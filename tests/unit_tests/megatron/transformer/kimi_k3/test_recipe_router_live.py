###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Phase-2 recipe gate: the router must actually route, and the published
LR / weight-decay recipe must reach Megatron's args.

Why this file exists
--------------------
The phase-1 validation (``validate/VALIDATION.md`` §7.2) found that
``moe_router_force_load_balancing: true`` was set for the whole run.
``TopKRouter.forward`` (``router.py:696-698``) then calls
``apply_random_logits``, and ``RandomSTE.forward`` (``moe_utils.py:1177-1198``)
**discards the gating output** and returns ``logits.clone().normal_()``:

.. code-block:: python

    class RandomSTE(torch.autograd.Function):
        @staticmethod
        def forward(ctx, logits):
            with get_cuda_rng_tracker().fork(get_expert_parallel_rng_tracker_name()):
                random_logits = logits.clone().normal_()
            return random_logits          # the router's own logits are DISCARDED
        @staticmethod
        def backward(ctx, grad_output):
            return grad_output            # straight-through

So routing was uniformly random and the router weight only ever received a
straight-through gradient computed against a decision it did not make. The
phase-1 "router gradient is non-zero" check passed anyway, which is exactly
why a shape-only check is not enough here: :func:`test_recipe_router_uses_its_own_logits`
asserts the routing map *equals* the top-k of the router's own biased scores,
and its negative control asserts that the same check fails when the flag is on.

Upstream's own default is ``False`` (``transformer_config.py:714``); the
flag's docstring says it is "only for benchmark".

The recipe half
---------------
Tech report §3.3 (``research/raw/extras.txt:16``): "We use a cosine learning
rate schedule with a 1% linear warmup. Weight decay is set to 0.1 throughout."

Where those values have to live is **not** obvious, and getting it wrong is
silent. ``PrimusParser`` merges the model preset into the module config with
``merge_namespace(module_config, model_config, allow_override=False)``
(``parser.py:320``), and that helper *skips* any key already present in the
destination (``yaml_utils.py:121-122``: ``continue  # Skip duplicate keys,
keep dst value``). Every LR/WD key is already declared by
``trainer_base.yaml:73-96`` and re-set by ``pre_trainer.yaml:22-28``, so a
value written into ``kimi_k3_base.yaml`` would be **silently dropped**. Only
the experiment yaml's ``overrides:`` block wins, because that goes through
``override_namespace`` -> ``deep_merge_namespace`` (``parser.py:323-324``).
:func:`test_recipe_model_yaml_cannot_carry_lr_or_wd` pins that mechanism so
the next person does not have to rediscover it.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path
from typing import Any, Dict

import pytest
import torch  # noqa: F401  # must precede any transformer_engine import; see DECISIONS.md

_REPO_ROOT = Path(__file__).resolve().parents[5]
_YAML_DIR = _REPO_ROOT / "primus" / "configs" / "models" / "megatron"
_EXP_YAML = _REPO_ROOT / "examples" / "megatron" / "configs" / "MI355X" / "kimi_k3-BF16-pretrain.yaml"
_MODEL_YAMLS = ["kimi_k3_base.yaml", "kimi_k3.yaml", "kimi_k3_debug.yaml"]

# Tech report §3.3.
_RECIPE_LR_DECAY_STYLE = "cosine"
_RECIPE_LR_WARMUP_FRACTION = 0.01
_RECIPE_WEIGHT_DECAY = 0.1

# Keys whose value can only be set from the experiment yaml's ``overrides:``
# block, because ``trainer_base.yaml`` already declares them.
_TRAINER_OWNED_RECIPE_KEYS = (
    "lr_decay_style",
    "lr_warmup_fraction",
    "lr_warmup_iters",
    "weight_decay",
)


@pytest.fixture(scope="module")
def parse_yaml_fn():
    from primus.core.config.yaml_loader import parse_yaml

    return parse_yaml


def _exp_overrides(parse_yaml_fn) -> Dict[str, Any]:
    return parse_yaml_fn(str(_EXP_YAML))["modules"]["pre_trainer"]["overrides"]


# ---------------------------------------------------------------------------
# Task 1 -- the router caveat, at config level
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("yaml_name", _MODEL_YAMLS)
def test_recipe_model_yaml_never_forces_load_balancing(parse_yaml_fn, yaml_name: str) -> None:
    """No Kimi K3 model preset may turn the benchmark switch on."""
    parsed = parse_yaml_fn(str(_YAML_DIR / yaml_name))
    value = parsed.get("moe_router_force_load_balancing", False)
    assert value is False, (
        f"{yaml_name}: moe_router_force_load_balancing={value!r}. "
        "This does not balance anything -- TopKRouter.forward replaces the gating "
        "output with normal_() draws (router.py:696-698 -> moe_utils.py:1177-1198), "
        "so routing becomes uniformly random and neither noaux_tc nor Quantile "
        "Balancing is exercised. See validate/VALIDATION.md §7.2."
    )


def test_recipe_experiment_yaml_router_is_live(parse_yaml_fn) -> None:
    """The MI355X experiment yaml must leave the router in charge.

    The value is written as ``${PRIMUS_MOE_FORCE_LOAD_BALANCING:false}``; the
    Primus loader resolves the ``:default`` form and maps the string ``false``
    onto :data:`False` (``yaml_loader.py:15``), so what is asserted here is the
    value a run with no env override actually gets.
    """
    overrides = _exp_overrides(parse_yaml_fn)
    assert "moe_router_force_load_balancing" in overrides, (
        "kimi_k3-BF16-pretrain.yaml must state moe_router_force_load_balancing "
        "explicitly rather than relying on the upstream default, so that flipping "
        "it back on is a visible diff."
    )
    assert overrides["moe_router_force_load_balancing"] is False, (
        "kimi_k3-BF16-pretrain.yaml re-enables forced random routing. Phase 1 ran "
        "this way for its entire life and the MoE router never learned anything "
        "(validate/VALIDATION.md §7.2)."
    )


def test_recipe_expert_bias_recipe_is_still_intact(parse_yaml_fn) -> None:
    """Turning the benchmark switch off is only useful if noaux_tc is on.

    ``DECISIONS.md`` §2 (and its 2026-07-28 refinement) require the sigmoid
    score function together with the expert bias; ``transformer_config.py:1769``
    raises for any other score function once the bias is enabled.
    """
    parsed = parse_yaml_fn(str(_YAML_DIR / "kimi_k3_base.yaml"))
    assert parsed["moe_router_enable_expert_bias"] is True
    assert parsed["moe_router_score_function"] == "sigmoid"
    assert parsed["moe_router_pre_softmax"] is False


# ---------------------------------------------------------------------------
# Task 4 -- the published LR / weight-decay recipe
# ---------------------------------------------------------------------------


def test_recipe_lr_schedule_and_weight_decay(parse_yaml_fn) -> None:
    """Tech report §3.3: cosine decay, 1% linear warmup, weight decay 0.1."""
    overrides = _exp_overrides(parse_yaml_fn)

    assert overrides.get("lr_decay_style") == _RECIPE_LR_DECAY_STYLE
    assert float(overrides.get("lr_warmup_fraction")) == pytest.approx(_RECIPE_LR_WARMUP_FRACTION), (
        "Report §3.3 specifies a 1% *linear* warmup. Megatron expresses that as "
        "lr_warmup_fraction, which OptimizerParamScheduler multiplies by the decay "
        "horizon; lr_warmup_iters is an absolute count and cannot track train_iters."
    )
    assert float(overrides.get("weight_decay")) == pytest.approx(_RECIPE_WEIGHT_DECAY)

    # arguments.py:1005-1007 asserts the two warmup knobs are mutually exclusive:
    # 'can only specify one of lr-warmup-fraction and lr-warmup-iters'.
    assert int(overrides.get("lr_warmup_iters", 0)) == 0, (
        "lr_warmup_fraction and lr_warmup_iters cannot both be set "
        "(arguments.py:1005-1007). pre_trainer.yaml:25 defaults lr_warmup_iters to "
        "40, so the experiment yaml has to pin it back to 0."
    )


@pytest.mark.parametrize("yaml_name", _MODEL_YAMLS)
def test_recipe_model_yaml_cannot_carry_lr_or_wd(parse_yaml_fn, yaml_name: str) -> None:
    """A model preset must not *look* like it sets the LR/WD recipe.

    It cannot: ``merge_namespace(module_config, model_config,
    allow_override=False)`` (``parser.py:320``) skips every key the module
    config already declares (``yaml_utils.py:121-122``), and
    ``trainer_base.yaml:73-96`` declares all of them. A value here would be
    dead config that reads as if it were live -- the same class of trap as the
    DeepSeek-V4 yamls that only work because their launcher overrides them.
    """
    parsed = parse_yaml_fn(str(_YAML_DIR / yaml_name))
    present = [k for k in _TRAINER_OWNED_RECIPE_KEYS if k in parsed]
    assert not present, (
        f"{yaml_name} sets {present}, which PrimusParser silently discards "
        "(parser.py:320 -> yaml_utils.py:121-122). Put the recipe in the "
        "experiment yaml's overrides: block instead."
    )


def test_recipe_merge_precedence_is_what_this_file_assumes() -> None:
    """Pin ``merge_namespace``'s skip-duplicates behaviour directly.

    The two tests above are only meaningful if this is true, and it is a
    one-word change upstream (``continue`` -> ``dst[key] = value``) that no
    other test would catch.
    """
    from types import SimpleNamespace

    from primus.core.utils import yaml_utils

    dst = SimpleNamespace(weight_decay=0.0, only_in_dst=1)
    src = SimpleNamespace(weight_decay=0.1, only_in_src=2)
    yaml_utils.merge_namespace(dst, src, allow_override=False)

    assert dst.weight_decay == 0.0, (
        "merge_namespace no longer skips duplicate keys. If the model preset now "
        "wins over pre_trainer.yaml, move the Kimi K3 LR/WD recipe into "
        "kimi_k3_base.yaml and delete test_recipe_model_yaml_cannot_carry_lr_or_wd."
    )
    assert dst.only_in_src == 2
    assert dst.only_in_dst == 1


# ---------------------------------------------------------------------------
# Task 1 -- the router caveat, functionally, on real hardware
# ---------------------------------------------------------------------------

_GPU_ONLY = pytest.mark.skipif(
    not torch.cuda.is_available(),
    reason="TopKRouter allocates its expert_bias / local_tokens_per_expert buffers "
    "on torch.cuda.current_device() in __init__ (router.py:172-189).",
)

HIDDEN = 64
NUM_EXPERTS = 16
TOPK = 4
NUM_TOKENS = 256


@pytest.fixture()
def router_parallel_state():
    """TP=PP=EP=CP=1 parallel state; same shape as ``test_stable_latent_moe.py``."""
    from megatron.core import parallel_state as ps

    if not torch.distributed.is_initialized():
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
            sock.bind(("127.0.0.1", 0))
            port = sock.getsockname()[1]
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(port))
        try:
            torch.distributed.init_process_group(
                backend="nccl", init_method=f"tcp://127.0.0.1:{port}", world_size=1, rank=0
            )
        except Exception as exc:  # pragma: no cover - environment guard
            pytest.skip(f"could not initialize torch.distributed: {exc}")

    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()
    ps.initialize_model_parallel(
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        expert_model_parallel_size=1,
        context_parallel_size=1,
    )

    from megatron.core.tensor_parallel import random as tp_random

    try:
        tp_random.initialize_rng_tracker(use_te_rng_tracker=True, force_reset=True)
    except (ImportError, AssertionError):  # pragma: no cover - environment guard
        tp_random.initialize_rng_tracker(use_cudagraphable_rng=True, force_reset=True)
    tp_random.model_parallel_cuda_manual_seed(42)

    yield

    if ps.model_parallel_is_initialized():
        ps.destroy_model_parallel()


def _router_config(force_load_balancing: bool):
    """A Kimi K3 router config: sigmoid scores + the noaux_tc selection bias."""
    import torch.nn.functional as F

    from primus.backends.megatron.core.models.kimi_k3.kimi_k3_transformer_config import (
        KimiK3TransformerConfig,
    )

    return KimiK3TransformerConfig(
        num_layers=1,
        hidden_size=HIDDEN,
        num_attention_heads=8,
        kv_channels=8,
        ffn_hidden_size=HIDDEN,
        moe_ffn_hidden_size=HIDDEN,
        num_moe_experts=NUM_EXPERTS,
        moe_router_topk=TOPK,
        moe_router_score_function="sigmoid",
        moe_router_enable_expert_bias=True,
        moe_router_pre_softmax=False,
        moe_router_topk_scaling_factor=1.0,
        moe_router_num_groups=None,
        moe_router_group_topk=None,
        moe_router_dtype="fp32",
        moe_router_load_balancing_type="none",
        moe_aux_loss_coeff=0.0,
        moe_router_force_load_balancing=force_load_balancing,
        activation_func=F.silu,
        gated_linear_unit=True,
        add_bias_linear=False,
        normalization="RMSNorm",
        params_dtype=torch.float32,
        sequence_parallel=False,
    )


def _build_router(config):
    # get_default_pg_collection lives in moe_utils (``moe_utils.py:1348``), not
    # in process_groups_config -- it is the same helper MoELayer.__init__ falls
    # back to at ``moe_layer.py:170-171``.
    from megatron.core.transformer.moe.moe_utils import get_default_pg_collection
    from megatron.core.transformer.moe.router import TopKRouter

    router = TopKRouter(config=config, pg_collection=get_default_pg_collection()).cuda()
    # A non-trivial bias, so "selection follows scores + bias" is a stronger
    # statement than "selection follows scores".
    with torch.no_grad():
        router.expert_bias.copy_(torch.linspace(-0.05, 0.05, NUM_EXPERTS, device=router.expert_bias.device))
    return router


def _expected_routing_map(router, hidden_states):
    """Top-k of ``sigmoid(W_r x) + expert_bias``, recomputed independently.

    Mirrors ``topk_routing_with_score_function``'s sigmoid branch
    (``moe_utils.py:772-780``) and ``KimiMoEGate.forward``
    (``modeling_kimi_linear.py:711-723``): the bias shifts *selection* only.
    """
    logits = router.gating(hidden_states).view(-1, NUM_EXPERTS)
    scores = torch.sigmoid(logits.float())
    biased = scores + router.expert_bias
    top_indices = torch.topk(biased, k=TOPK, dim=1).indices
    expected = torch.zeros_like(scores, dtype=torch.bool)
    expected.scatter_(1, top_indices, True)
    return expected


@_GPU_ONLY
def test_recipe_router_uses_its_own_logits(router_parallel_state) -> None:
    """The routing map equals the top-k of the router's own biased scores.

    This is the check phase 1 did not have. "The router weight has a gradient"
    is satisfied by the straight-through estimator even when the logits are
    thrown away; equality with an independently recomputed ``argtopk`` is not.
    """
    config = _router_config(force_load_balancing=False)
    router = _build_router(config)
    hidden = torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda", dtype=torch.float32)

    _, routing_map = router(hidden)
    expected = _expected_routing_map(router, hidden)

    assert routing_map.shape == expected.shape
    assert torch.equal(routing_map, expected), (
        "routing_map does not match argtopk(sigmoid(W_r x) + expert_bias). Either "
        "the router's logits are being replaced (moe_router_force_load_balancing) "
        "or the score/bias convention changed."
    )
    # Exactly topk experts per token, and not the degenerate all-experts case.
    assert torch.all(routing_map.sum(dim=1) == TOPK)


@_GPU_ONLY
def test_recipe_forced_load_balancing_would_break_that_check(router_parallel_state) -> None:
    """Negative control: the assertion above can actually fail.

    Without this, ``test_recipe_router_uses_its_own_logits`` proves nothing --
    the phase-1 validation was caught out once already by a positive control
    that could not fail (``validate/VALIDATION.md`` §2.3).
    """
    config = _router_config(force_load_balancing=True)
    router = _build_router(config)
    hidden = torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda", dtype=torch.float32)

    _, routing_map = router(hidden)
    expected = _expected_routing_map(router, hidden)

    overlap = (routing_map & expected).sum().item()
    total = int(routing_map.sum().item())
    assert not torch.equal(routing_map, expected), (
        "moe_router_force_load_balancing=True produced exactly the router's own "
        "top-k, which cannot happen if apply_random_logits is still substituting "
        "normal_() draws (moe_utils.py:1177-1198)."
    )
    # Random routing overlaps the real top-k at roughly the chance rate
    # topk/num_experts; anything close to 1.0 would mean the flag is inert.
    assert overlap / total < 0.5


@_GPU_ONLY
def test_recipe_live_router_is_input_determined_not_rng_determined(router_parallel_state) -> None:
    """Two forwards on the same input must route identically.

    Under forced load balancing they do not, because ``RandomSTE`` draws fresh
    ``normal_()`` values inside the expert-parallel RNG fork on every call.
    Phase 1's causality tests had to reset ``model_parallel_cuda_manual_seed``
    before every forward to get bit-identical logits
    (``validate/VALIDATION.md`` §2.1); with the router live, that crutch is
    unnecessary, and this test is what says so.
    """
    hidden = torch.randn(NUM_TOKENS, 1, HIDDEN, device="cuda", dtype=torch.float32)

    live = _build_router(_router_config(force_load_balancing=False))
    _, map_a = live(hidden)
    _, map_b = live(hidden)
    assert torch.equal(map_a, map_b), "live routing must be a function of the input alone"

    forced = _build_router(_router_config(force_load_balancing=True))
    _, forced_a = forced(hidden)
    _, forced_b = forced(hidden)
    assert not torch.equal(forced_a, forced_b), (
        "forced load balancing produced identical routing across two calls; the "
        "RNG fork in RandomSTE (moe_utils.py:1180-1183) is not doing anything."
    )


@_GPU_ONLY
def test_recipe_live_router_load_responds_to_the_gate_weight(router_parallel_state) -> None:
    """Expert load must follow the router weight, not a uniform prior.

    Skewing the gate weight towards one expert must skew the load. Under
    ``apply_random_logits`` the load stays uniform no matter what the weight
    says, which is the concrete reason phase 1's run tells us nothing about
    the ``noaux_tc`` recipe.
    """
    config = _router_config(force_load_balancing=False)
    router = _build_router(config)
    # Strictly positive input, so the sign of the boosted experts' logit is
    # decided by the weight alone and the outcome is deterministic rather than
    # data-dependent.
    hidden = torch.rand(NUM_TOKENS, 1, HIDDEN, device="cuda", dtype=torch.float32) + 0.1

    with torch.no_grad():
        router.weight.zero_()
        router.weight[:TOPK] = 1.0  # experts 0..TOPK-1 respond to every token
        router.expert_bias.zero_()

    _, routing_map = router(hidden)
    load = routing_map.sum(dim=0)

    # Boosted experts score sigmoid(positive) > 0.5; every other expert scores
    # sigmoid(0) = 0.5 exactly. So the top-k is the boosted block, for every
    # token, and the load is a step function.
    assert (
        load[:TOPK].tolist() == [NUM_TOKENS] * TOPK
    ), f"expert load {load.tolist()} ignores the gate weight; the router is not driving expert selection"
    assert load[TOPK:].sum().item() == 0
