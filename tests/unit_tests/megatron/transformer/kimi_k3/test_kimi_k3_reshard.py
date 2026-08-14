###############################################################################
# Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################

"""Distributed-checkpoint sharding for Kimi K3's new parameter slots (WP7).

Resharding -- saving a checkpoint at one parallel layout and loading it at
another -- is decided entirely by what each module's ``sharded_state_dict``
declares about its parameters. For every parameter that declaration answers one
question: is the tensor this rank holds a **slice** of a larger global tensor, or
a **replica** of the whole thing?

Getting that answer wrong does not raise. A genuinely TP-sharded parameter
declared replicated makes every rank claim to be a replica of the same global
tensor, so the writer keeps one rank's copy and the reader hands it to all of
them -- rank 1's weights are silently replaced by rank 0's and training carries on
with a quietly wrong model. The reverse mistake multiplies the declared global
width by the TP size, which usually *does* raise, but only on the first load at a
different layout, which may be weeks later.

Kimi K3 introduces parameter slots upstream resharding has never seen, and they
land on both sides of that line:

* full-width, replicated -- the attention-residual mixers' ``norm_weight`` /
  ``proj_weight`` (``attention_residual.py:122-129``) and the post-stack head's,
  ``StableLatentMoE``'s ``routed_expert_norm`` and its two ``*_latent_proj``
  (built ``parallel_mode="duplicated"``, ``moe_layer.py:200-221``), and KDA's
  ``out_norm.weight``, which is per-channel *within* a head and therefore shared
  by every head;
* TP-sharded -- KDA's ``A_log`` and ``dt_bias`` (one entry per local head and per
  local head-channel), its three depthwise ``*_conv1d.weight``, and MLA's
  ``linear_o_gate``.

Two of those are easy to get backwards. The mixers carry
``param.sequence_parallel = True``, which looks like a sharding flag but is not:
it asks ``finalize_model_grads`` to *sum a replicated parameter's partial
gradient* over the TP group (``finalize_model_grads.py:357-370``), and KDA's
``out_norm.weight`` carries it for the same reason. A future reader who "fixed"
that into a sharded checkpoint declaration would corrupt every reshard while
leaving the gradient reduction working.

**Why the fake TP size.** At TP=1 a replicated declaration and a TP-sharded one
are indistinguishable: ``global_shape == local_shape``, every
``axis_fragmentations`` entry is 1, and ``replica_id``'s tp component is 0 either
way -- so WP7's TP=1 checkpoint audit could not have caught a wrong axis. A
single-process test can only tell them apart by making the *metadata* believe the
TP group is wider than it is. ``make_tp_sharded_tensor_for_checkpoint`` and
``make_sharded_tensor_for_checkpoint`` read the group only through
``megatron.core.utils.get_pg_size`` / ``get_pg_rank`` (``utils.py:549-574``), so
overriding those two for the tensor-parallel group -- and only for it -- yields
exactly the declaration a real TP rank would emit, without needing a second GPU.
The local tensors keep their TP=1 shapes, which is fine: what is under test is
which axis was declared split, not how wide the result is.

A real multi-rank reshard is the integration counterpart and lives in ``tpfix/``.
"""

from __future__ import annotations

import contextlib
import os
import re
from typing import Dict, List, Optional

import pytest
import torch

# The geometry and the block builder are shared with the pipeline-seam tests on
# purpose: both files must describe the *same* model, or a sharding claim proved
# here would not be a claim about the model that file exercises.
from tests.unit_tests.megatron.transformer.kimi_k3.test_kimi_k3_pp_shapes import (
    _build_stage,
    _make_config,
    _stage_specs,
)

# Deliberately not a divisor of the head count (8), the expert count (8) or the
# layer count (8), so a fragmentation of exactly this value can only have come
# from the tensor-parallel axis and never from a layer or expert axis.
FAKE_TP_SIZE = 5
FAKE_TP_RANK = 3
# Distinct primes per axis, none of them a divisor of the head count (8), the expert
# count (8) or the layer count (8), so a fragmentation can be traced back to exactly
# one faked group even when several appear on the same tensor.
FAKE_EP_SIZE = 5
FAKE_EP_RANK = 3
FAKE_DP_SIZE = 7
FAKE_DP_RANK = 4

# Kimi K3's new parameter slots and the axis each must be declared on; ``None``
# means replicated. Matched as a suffix of the parameter name.
K3_SLOT_TP_AXIS: Dict[str, Optional[int]] = {
    # Attention residuals: the residual stream is full width on every TP rank and
    # the score reduces over the whole hidden axis, so no shard is needed.
    "attn_res_mixer.norm_weight": None,
    "attn_res_mixer.proj_weight": None,
    "mlp_res_mixer.norm_weight": None,
    "mlp_res_mixer.proj_weight": None,
    "attn_res_head.norm_weight": None,
    "attn_res_head.proj_weight": None,
    # MLA's sigmoid output gate is a column-parallel projection over heads.
    "self_attention.linear_o_gate.weight": 0,
    # Stable Latent MoE: the norm and both latent projections are replicated.
    "mlp.routed_expert_norm.weight": None,
    "mlp.fc1_latent_proj.weight": None,
    "mlp.fc2_latent_proj.weight": None,
    # KDA: one entry per local head, and per local head-channel.
    "self_attention.A_log": 0,
    "self_attention.dt_bias": 0,
    "self_attention.q_conv1d.weight": 0,
    "self_attention.k_conv1d.weight": 0,
    "self_attention.v_conv1d.weight": 0,
    # KDA's gain is per-channel within a head and shared by every head.
    "self_attention.out_norm.weight": None,
}

# The grouped routed experts describe their sharding only inside
# ``sharded_state_dict``: ``TEGroupedMLP`` allocates ``weight{i}`` per expert
# without ``set_tensor_model_parallel_attributes`` (``experts.py:406-440``), so the
# attribute-derived ground truth cannot see it. They are excluded from the strict
# comparison and checked only for presence.
_EXPERT_MARKERS = ("mlp.experts.linear_fc1.weight", "mlp.experts.linear_fc2.weight")

# Every fused ``[gate | up]`` weight is a ``ShardedTensorFactory``: rank ``r``
# holds ``[gate[r], up[r]]``, so the global tensor is *not* the concatenation of
# the local ones and the halves have to be de-interleaved before they are
# reassembled. ``apply_swiglu_sharded_factory`` encodes that (``mlp.py:371``), and
# it covers the dense MLP on layer 0, the shared experts, and each grouped routed
# expert. A factory is opaque to the axis cross-check, so the set of them is
# pinned rather than left open.
_FUSED_GLU_WEIGHT = re.compile(r"linear_fc1\.weight\d*$")


# ---------------------------------------------------------------------------
# fixtures and helpers
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mpu_tp1():
    """A 1-rank process group plus Megatron model-parallel state."""
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel import model_parallel_cuda_manual_seed

    created = False
    if not dist.is_initialized():
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", "29587")
        os.environ.setdefault("WORLD_SIZE", "1")
        os.environ.setdefault("RANK", "0")
        os.environ.setdefault("LOCAL_RANK", "0")
        dist.init_process_group(
            backend="nccl" if torch.cuda.is_available() else "gloo", world_size=1, rank=0
        )
        created = True
    try:
        if not parallel_state.model_parallel_is_initialized():
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=1, pipeline_model_parallel_size=1
            )
        if torch.cuda.is_available():
            model_parallel_cuda_manual_seed(1234)
        yield
    finally:
        if created:
            parallel_state.destroy_model_parallel()
            dist.destroy_process_group()


@contextlib.contextmanager
def pretend_group(group, size: int, rank: int):
    """Make checkpoint metadata believe one process group is wider than it is.

    ``make_{tp_,}sharded_tensor_for_checkpoint`` and
    ``TEGroupedLinear._sharded_state_dict_grouped`` both learn their group's size and
    rank only through ``megatron.core.utils.get_pg_size`` / ``get_pg_rank``
    (``utils.py:549-574``, ``transformer_engine.py:1766-1767``), so overriding those
    two for a single group -- and only for that group -- yields exactly the
    declaration a real rank of that group would emit. Every other group keeps its
    real size, so faking the expert axis cannot disturb the tensor-parallel one or
    the data-parallel component of ``replica_id``.
    """
    import sys

    import megatron.core.utils as mcu

    real_size, real_rank = mcu.get_pg_size, mcu.get_pg_rank

    def _size(g=None):
        return size if g is group else real_size(g)

    def _rank(g=None):
        return rank if g is group else real_rank(g)

    # Patching `megatron.core.utils` alone is not enough, and finding that out is what
    # the per-axis canary tests are for. `make_{tp_,}sharded_tensor_for_checkpoint` is
    # *defined in* that module, so it resolves `get_pg_size` as one of its own globals
    # and sees the override. `TEGroupedLinear._sharded_state_dict_grouped` -- the only
    # place the expert axis is decided -- lives in
    # `megatron.core.extensions.transformer_engine`, which did
    # `from megatron.core.utils import get_pg_size` and therefore holds its own
    # binding that a patch on the source module never touches. So rebind the name
    # wherever it has been imported, identified by object identity rather than by a
    # hardcoded list of module names, so a future consumer is covered automatically.
    patched = []
    for mod in list(sys.modules.values()):
        if mod is None:
            continue
        try:
            if getattr(mod, "get_pg_size", None) is real_size:
                mod.get_pg_size = _size
                patched.append((mod, "get_pg_size", real_size))
            if getattr(mod, "get_pg_rank", None) is real_rank:
                mod.get_pg_rank = _rank
                patched.append((mod, "get_pg_rank", real_rank))
        except Exception:  # a module that objects to attribute access is not a consumer
            continue
    try:
        yield
    finally:
        for mod, name, original in patched:
            setattr(mod, name, original)


def pretend_tp(tp_size: int = FAKE_TP_SIZE, tp_rank: int = FAKE_TP_RANK):
    """:func:`pretend_group` on the tensor-parallel group."""
    from megatron.core import parallel_state

    return pretend_group(parallel_state.get_tensor_model_parallel_group(), tp_size, tp_rank)


def module_ep_group(module):
    """The expert group object the grouped-expert linears actually read.

    ``TEGroupedLinear._sharded_state_dict_grouped`` sizes the expert axis from
    ``get_pg_size(self._pg_collection.ep)`` (``transformer_engine.py:1766-1767``), and
    that is the object identity :func:`pretend_group` has to match. Taking it from
    ``parallel_state`` instead is not equivalent -- the module may hold a different
    object, or ``None``, in which case ``get_pg_size`` short-circuits to 1 and no
    override on any real group can reach it.
    """
    for sub in module.modules():
        pgc = getattr(sub, "_pg_collection", None)
        if pgc is not None and getattr(pgc, "ep", None) is not None:
            return pgc.ep
    return None


def pretend_ep(group, ep_size: int = FAKE_EP_SIZE, ep_rank: int = FAKE_EP_RANK):
    """:func:`pretend_group` on the expert-model-parallel group."""
    return pretend_group(group, ep_size, ep_rank)


def pretend_dp(dp_size: int = FAKE_DP_SIZE, dp_rank: int = FAKE_DP_RANK):
    """:func:`pretend_group` on the data-parallel (with context-parallel) group."""
    from megatron.core import parallel_state

    return pretend_group(
        parallel_state.get_data_parallel_group(with_context_parallel=True), dp_size, dp_rank
    )


def truth_axis(param) -> Optional[int]:
    """The axis this parameter is really TP-sharded on, from its attributes.

    ``set_tensor_model_parallel_attributes`` writes ``tensor_model_parallel`` and
    ``partition_dim`` together, but a module that allocates a sharded parameter by
    hand may set only the flag and leave ``partition_dim`` at its ``-1`` default --
    KDA does exactly that for ``A_log`` / ``dt_bias`` and the conv weights
    (``kimi_delta_attention.py:345,353,513``). Axis 0 is the documented intent
    there, and this mirrors ``wp7/tp_parity.py``'s rule so the two audits agree.
    """
    if not getattr(param, "tensor_model_parallel", False):
        return None
    dim = int(getattr(param, "partition_dim", -1))
    return 0 if dim < 0 else dim


def declared_axes(entry) -> List[int]:
    """The local axes a ``ShardedTensor`` declares as split by the TP group.

    Axes below ``prepend_axis_num`` belong to the pipeline / layer index rather
    than to the tensor itself, and a fragmentation that is not exactly
    ``FAKE_TP_SIZE`` cannot have come from the faked tensor-parallel group.
    """
    prepend = int(entry.prepend_axis_num)
    frag = entry.axis_fragmentations or ()
    return [i - prepend for i, f in enumerate(frag) if i >= prepend and f == FAKE_TP_SIZE]


def is_expert_param(name: str) -> bool:
    return any(marker in name for marker in _EXPERT_MARKERS)


def audit(module, ctx=None) -> Dict[str, dict]:
    """One row per parameter: the attribute truth against the declaration.

    ``ctx`` is the faked-group context to build the declaration under; it defaults to
    the tensor-parallel one.
    """
    from megatron.core.dist_checkpointing.mapping import (
        ShardedObject,
        ShardedTensor,
        ShardedTensorFactory,
    )

    with (ctx if ctx is not None else pretend_tp()):
        declared = module.sharded_state_dict(prefix="")

    rows: Dict[str, dict] = {}
    for name, param in module.named_parameters():
        entry = declared.get(name)
        row = {
            "truth": truth_axis(param),
            "shape": tuple(param.shape),
            "sequence_parallel": bool(getattr(param, "sequence_parallel", False)),
            "entry": entry,
        }
        if entry is None:
            row["kind"] = "missing"
        elif isinstance(entry, ShardedTensorFactory):
            row["kind"] = "factory"
            built = entry.build()
            inner = built if isinstance(built, ShardedTensor) else None
            if isinstance(built, dict):
                inner = next((v for v in built.values() if isinstance(v, ShardedTensor)), None)
            row["built"] = inner
        elif isinstance(entry, ShardedObject):
            row["kind"] = "object"
        elif isinstance(entry, ShardedTensor):
            row["kind"] = "tensor"
            row["declared"] = declared_axes(entry)
            row["replica_tp"] = entry.replica_id[1] if len(entry.replica_id) > 1 else None
        else:
            row["kind"] = type(entry).__name__
        rows[name] = row
    return rows


def describe(axis: Optional[int]) -> str:
    return "replicated" if axis is None else f"split on axis {axis}"


@pytest.fixture(scope="module")
def k3_block(mpu_tp1):
    """The 8-layer debug stack, built once for the whole module."""
    config = _make_config()
    return _build_stage(config, _stage_specs(config), pre_process=True, post_process=True)


@pytest.fixture(scope="module")
def block_rows(k3_block):
    return audit(k3_block)


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------


def test_the_fake_tp_size_actually_reaches_the_checkpoint_metadata(block_rows):
    """Guard the guard.

    Most assertions below are of the form "this parameter is *not* declared
    sharded". If :func:`pretend_tp` silently failed to take effect, nothing would
    ever be declared sharded and all of them would pass vacuously. So first demand
    that something known-sharded really does come out sharded, with the offsets a
    real rank would have.
    """
    sharded = {n: r for n, r in block_rows.items() if r.get("declared")}
    assert sharded, (
        "no parameter was declared TP-sharded, so pretend_tp() did not reach "
        "make_tp_sharded_tensor_for_checkpoint and every check below would be vacuous"
    )

    a_logs = {n: r for n, r in sharded.items() if n.endswith("self_attention.A_log")}
    assert a_logs, f"A_log is not declared sharded; sharded set was {sorted(sharded)[:8]}"
    for name, row in a_logs.items():
        entry = row["entry"]
        assert row["declared"] == [0], name
        assert entry.global_shape[0] == entry.local_shape[0] * FAKE_TP_SIZE, name
        assert entry.global_offset[0] == FAKE_TP_RANK * entry.local_shape[0], name


def test_every_parameter_declaration_matches_its_tensor_parallel_attributes(block_rows):
    """The whole model, cross-checked against ``param.tensor_model_parallel``.

    This is the test that catches a parameter which is replicated but declared
    sharded, or sharded but declared replicated, anywhere in the stack -- the
    failure mode that silently corrupts a reshard instead of raising.
    """
    missing = [n for n, r in block_rows.items() if r["kind"] == "missing"]
    assert not missing, f"{len(missing)} parameters absent from the checkpoint: {missing[:8]}"

    problems = []
    for name, row in block_rows.items():
        if is_expert_param(name) or row["kind"] != "tensor":
            continue
        expected = [] if row["truth"] is None else [row["truth"]]
        if row["declared"] != expected:
            problems.append(
                f"{name} {row['shape']}: the tensor_model_parallel attributes say "
                f"{describe(row['truth'])} but the checkpoint declares split axes "
                f"{row['declared']}"
            )
            continue
        # A replicated tensor must carry the tp rank in replica_id so exactly one
        # rank is the primary writer; a sharded one must not, because its shards
        # are not replicas of each other.
        expect_replica_tp = FAKE_TP_RANK if row["truth"] is None else 0
        if row["replica_tp"] != expect_replica_tp:
            problems.append(
                f"{name}: replica_id tp component {row['replica_tp']} != "
                f"{expect_replica_tp} for a {describe(row['truth'])} parameter"
            )
    assert not problems, "\n".join(problems)


def test_only_fused_gate_up_weights_are_hidden_behind_a_factory(block_rows):
    """Nothing new-in-K3 may arrive as a ``ShardedTensorFactory``.

    A factory is opaque to the attribute cross-check above, so a K3 parameter
    arriving as one would silently drop out of this file's coverage. The only
    legitimate factories are the fused ``[gate | up]`` weights, whose global
    tensor is not the concatenation of the local ones.
    """
    factories = sorted(n for n, r in block_rows.items() if r["kind"] == "factory")
    assert factories, "no factory entries at all; apply_swiglu_sharded_factory stopped firing"
    unexpected = [n for n in factories if not _FUSED_GLU_WEIGHT.search(n)]
    assert not unexpected, (
        f"{len(unexpected)} parameters are checkpointed through a ShardedTensorFactory "
        f"and are therefore outside the axis cross-check: {unexpected[:8]}"
    )


@pytest.mark.parametrize("slot, axis", sorted(K3_SLOT_TP_AXIS.items()))
def test_kimi_k3_new_slot_is_declared_on_the_expected_axis(block_rows, slot, axis):
    """Each new-in-K3 slot, pinned individually.

    The blanket test above compares the declaration with the parameter's own
    attributes, so it cannot catch the two being wrong *together*. This one states
    the intended answer independently, from the architecture.
    """
    hits = {n: r for n, r in block_rows.items() if n.endswith(slot)}
    assert hits, f"no parameter ends with {slot!r}; the slot vanished or was renamed"
    for name, row in hits.items():
        assert row["kind"] == "tensor", f"{name}: unexpected checkpoint entry {row['kind']}"
        expected = [] if axis is None else [axis]
        assert row["declared"] == expected, (
            f"{name} {row['shape']}: expected {describe(axis)}, got split axes {row['declared']}"
        )


def test_sequence_parallel_on_a_mixer_parameter_does_not_shard_it(mpu_tp1):
    """``param.sequence_parallel`` must not become a sharded declaration.

    The attention-residual mixers set it whenever the config does
    (``attention_residual.py:127-129``), and it means "replicated parameter,
    partial gradient, sum it over the TP group" -- not "this tensor is sharded".
    Both facts have to hold at once, which is what makes this worth its own test:
    conflating them would keep the gradient reduction working and break only
    resharding, silently.

    Built standalone rather than through the block because the flag is only set
    when ``config.sequence_parallel`` is on, and a whole K3 stack with sequence
    parallelism at TP=1 is not a configuration Megatron accepts. The mixer needs
    no TransformerEngine module, so it can be built against a config that has it.
    """
    from primus.backends.megatron.core.transformer.kimi_k3.attention_residual import (
        AttentionResidualHead,
        AttentionResidualMixer,
    )

    config = _make_config()
    config.sequence_parallel = True

    for cls in (AttentionResidualMixer, AttentionResidualHead):
        module = cls(config)
        rows = audit(module)
        assert set(rows) == {"norm_weight", "proj_weight"}, (cls.__name__, sorted(rows))
        for name, row in rows.items():
            assert row["sequence_parallel"], f"{cls.__name__}.{name} lost sequence_parallel"
            assert row["truth"] is None, (
                f"{cls.__name__}.{name} is both sequence_parallel and tensor_model_parallel"
            )
            assert row["kind"] == "tensor", f"{cls.__name__}.{name}: {row['kind']}"
            assert row["declared"] == [], (
                f"{cls.__name__}.{name}: sequence_parallel=True must not make the checkpoint "
                f"declare a TP split, got axes {row['declared']}"
            )
            assert row["replica_tp"] == FAKE_TP_RANK, (
                f"{cls.__name__}.{name}: a replicated parameter's replica_id must carry the "
                f"tp rank, got {row['replica_tp']}"
            )


def test_block_round_trips_through_a_distributed_checkpoint(mpu_tp1, tmp_path):
    """Save the block, load it into a differently-initialised one, compare.

    ``sharded_state_dict`` is only half the contract: a parameter can be declared
    correctly and still be dropped if the key it is stored under is not the key the
    loader asks for. Loading into a *fresh* block -- whose weights differ
    everywhere, because ``proj_weight`` and every projection are random -- makes a
    silently skipped tensor visible as a surviving difference rather than as a
    coincidence.
    """
    from megatron.core import dist_checkpointing

    config = _make_config()
    src = _build_stage(config, _stage_specs(config), pre_process=True, post_process=True)
    dst = _build_stage(config, _stage_specs(config), pre_process=True, post_process=True)

    src_state = {n: p.detach().clone() for n, p in src.named_parameters()}
    differing_before = [
        n for n, p in dst.named_parameters() if not torch.equal(p, src_state[n].to(p.device))
    ]
    assert differing_before, "the two freshly built blocks are identical; the test has no power"

    ckpt = str(tmp_path / "block")
    os.makedirs(ckpt, exist_ok=True)
    dist_checkpointing.save(src.sharded_state_dict(prefix=""), ckpt)

    loaded = dist_checkpointing.load(dst.sharded_state_dict(prefix=""), ckpt)
    report = dst.load_state_dict(loaded, strict=False)
    absent = [k for k in getattr(report, "missing_keys", []) if not k.endswith("_extra_state")]
    assert not absent, f"load_state_dict could not fill: {absent[:8]}"

    still_differing = [
        n for n, p in dst.named_parameters() if not torch.equal(p, src_state[n].to(p.device))
    ]
    assert not still_differing, (
        f"{len(still_differing)} parameters did not survive the checkpoint round trip: "
        f"{still_differing[:8]}"
    )


# ---------------------------------------------------------------------------
# The expert-parallel axis
#
# Expert parallelism is the configuration this project actually trains at (EP=8 in
# every phase-2 run), and it shards parameters along an axis nothing above touches.
# `TEGroupedLinear._sharded_state_dict_grouped` *prepends* an axis for the expert
# index, sized `get_pg_size(pg_collection.ep) * num_gemms` and offset
# `get_pg_rank(...) * num_gemms + local_index` (`transformer_engine.py:1757-1772`).
# A wrong offset there places one rank's experts at another rank's indices, and --
# exactly as with a wrong TP axis -- nothing raises: the shapes still fit, the save
# still succeeds, and the model silently comes back with experts permuted.
#
# At EP=1, which is what the unit suite runs, the offset is 0 whatever the rule, so
# these checks need the same faked-group trick as the TP ones.
# ---------------------------------------------------------------------------

_EXPERT_WEIGHT = re.compile(r"mlp\.experts\.linear_fc[12]\.weight(\d+)$")


def _expert_entries(rows: Dict[str, dict]):
    """``{name: (ShardedTensor, local_expert_index)}`` for the grouped routed experts."""
    out = {}
    for name, row in rows.items():
        m = _EXPERT_WEIGHT.search(name)
        if not m:
            continue
        entry = row.get("built") if row["kind"] == "factory" else row.get("entry")
        if entry is not None:
            out[name] = (entry, int(m.group(1)))
    return out


@pytest.fixture(scope="module")
def block_rows_ep(k3_block):
    group = module_ep_group(k3_block)
    if group is None:
        pytest.skip(
            "the grouped-expert linears hold no expert process group at EP=1, so "
            "get_pg_size() short-circuits to 1 and the expert axis cannot be widened "
            "in-process; the multi-rank audit in tpfix/rs/shard_meta.py covers it at "
            "EP=8/4/2 instead"
        )
    if group is parallel_state_tp_group():
        pytest.skip("the expert and tensor-parallel groups are the same object at one rank")
    return audit(k3_block, pretend_ep(group))


def parallel_state_tp_group():
    from megatron.core import parallel_state

    return parallel_state.get_tensor_model_parallel_group()


def test_the_fake_ep_size_actually_reaches_the_expert_axis(block_rows_ep):
    """Guard the guard, for the expert axis.

    Every expert assertion below is of the form "this offset is what the rule says".
    If :func:`pretend_ep` failed to take effect the expert axis would come out
    length 1 and the offsets would all be 0, which several wrong rules also produce.
    So first demand that the faked width is visible.
    """
    experts = _expert_entries(block_rows_ep)
    assert experts, "no grouped routed-expert tensors found; the MoE spec changed"
    widths = {e.axis_fragmentations[0] for e, _ in experts.values()}
    assert widths == {FAKE_EP_SIZE * 8}, (
        f"expert axis widths {widths} do not reflect a faked expert group of "
        f"{FAKE_EP_SIZE} ranks holding 8 local experts each; pretend_ep() did not "
        f"reach TEGroupedLinear._sharded_state_dict_grouped and the checks below "
        f"would be vacuous"
    )


def test_expert_weights_are_declared_at_their_global_expert_index(block_rows_ep):
    """The prepended expert axis must place local expert *i* at the global index.

    ``ep_rank * num_local_experts + i``. Off-by-one or a missing ``ep_rank`` term
    would overwrite another rank's experts on save and load somebody else's on
    resume, with no error either way.
    """
    experts = _expert_entries(block_rows_ep)
    n_local = 8  # the debug shape has 8 experts and the suite runs at EP=1
    problems = []
    for name, (entry, local_idx) in sorted(experts.items()):
        if entry.prepend_axis_num < 1:
            problems.append(f"{name}: no prepended expert axis "
                            f"(prepend_axis_num={entry.prepend_axis_num})")
            continue
        expect_offset = FAKE_EP_RANK * n_local + local_idx
        expect_width = FAKE_EP_SIZE * n_local
        if entry.global_shape[0] != expect_width:
            problems.append(f"{name}: expert axis global_shape {entry.global_shape[0]} "
                            f"!= {expect_width}")
        if entry.global_offset[0] != expect_offset:
            problems.append(f"{name}: expert axis global_offset {entry.global_offset[0]} "
                            f"!= ep_rank {FAKE_EP_RANK} * {n_local} + {local_idx} "
                            f"= {expect_offset}")
    assert not problems, "\n".join(problems)


@pytest.mark.parametrize(
    "slot",
    [
        "mlp.router.weight",
        "mlp.routed_expert_norm.weight",
        "mlp.fc1_latent_proj.weight",
        "mlp.fc2_latent_proj.weight",
        "mlp.shared_experts.linear_fc2.weight",
    ],
)
def test_kimi_k3_moe_slot_outside_the_experts_is_not_expert_sharded(block_rows_ep, slot):
    """The Stable Latent MoE's own parameters live outside the expert group.

    The router scores all experts, the latent bottleneck is entered and left once per
    layer, and ``routed_expert_norm`` normalises the *combined* routed output — none
    of them is per-expert, so none may carry the expert axis. Declaring one of them
    sharded over EP would make each expert rank claim a slice of a tensor it holds in
    full, and the reconstructed tensor would be `ep_size` times too wide.
    """
    hits = {n: r for n, r in block_rows_ep.items() if n.endswith(slot)}
    assert hits, f"no parameter ends with {slot!r}; the slot vanished or was renamed"
    for name, row in hits.items():
        entry = row.get("built") if row["kind"] == "factory" else row.get("entry")
        assert entry is not None, f"{name}: no ShardedTensor ({row['kind']})"
        assert entry.prepend_axis_num == 0, (
            f"{name} carries {entry.prepend_axis_num} prepended axis/axes; a "
            f"non-per-expert parameter must not be sharded over the expert group"
        )
        assert FAKE_EP_SIZE not in (entry.axis_fragmentations or ()), (
            f"{name}: axis_fragmentations {entry.axis_fragmentations} shows the faked "
            f"expert width {FAKE_EP_SIZE}, so it is declared expert-sharded"
        )


# ---------------------------------------------------------------------------
# The data-parallel axis
#
# Data parallelism does not shard model parameters, but it does decide *which* rank
# is the primary writer of each replicated tensor, through the third component of
# `replica_id`. If that were 0 on every data-parallel rank, every rank would claim to
# be the primary and the writer would have to pick arbitrarily between copies that
# are supposed to be identical -- the same class of silent failure as a wrong shard
# axis, one level up.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def block_rows_dp(k3_block):
    from megatron.core import parallel_state

    if parallel_state.get_data_parallel_group(with_context_parallel=True) is (
        parallel_state.get_tensor_model_parallel_group()
    ):
        pytest.skip("the data-parallel and tensor-parallel groups are the same object")
    return audit(k3_block, pretend_dp())


def test_the_fake_dp_size_actually_reaches_replica_id(block_rows_dp):
    """Guard the guard, for the data-parallel axis."""
    seen = set()
    for row in block_rows_dp.values():
        entry = row["entry"] if row["kind"] == "tensor" else row.get("built")
        if entry is not None and len(entry.replica_id) > 2:
            seen.add(entry.replica_id[2])
    assert FAKE_DP_RANK in seen, (
        f"no tensor carries the faked data-parallel rank {FAKE_DP_RANK} in "
        f"replica_id; saw {sorted(v for v in seen if v is not None)}. pretend_dp() "
        f"did not reach the checkpoint metadata and the check below would be vacuous"
    )


def test_every_tensor_carries_the_data_parallel_rank_in_replica_id(block_rows_dp):
    """Data-parallel replicas must be distinguishable, for every tensor.

    Both helpers put the data-parallel rank last in ``replica_id``
    (``utils.py:1051,1123``), which is what lets the writer keep one copy of a
    replicated tensor and lets the reader broadcast it. A tensor that hardcoded 0
    there would be written by all ranks at once.
    """
    problems = []
    for name, row in block_rows_dp.items():
        if row["kind"] == "tensor":
            entry = row["entry"]
        elif row["kind"] == "factory":
            entry = row.get("built")
        else:
            continue
        if entry is None:
            continue
        if is_expert_param(name) or _EXPERT_WEIGHT.search(name):
            # Expert tensors are replicated over the *expert*-data-parallel group
            # (`DP / EP`), not the data-parallel one, so they legitimately do not
            # carry this group's rank. Their replica ids are checked at real EP by
            # tpfix/rs/shard_meta.py.
            continue
        rid = entry.replica_id
        if len(rid) < 3:
            problems.append(f"{name}: replica_id {rid} has no data-parallel component")
        elif rid[2] != FAKE_DP_RANK:
            problems.append(f"{name}: replica_id {rid} data-parallel component "
                            f"{rid[2]} != {FAKE_DP_RANK}")
    assert not problems, "\n".join(problems[:12])
