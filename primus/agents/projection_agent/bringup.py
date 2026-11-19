"""Op bring-up: turn one op the projection does not model into a validated op model.

New models keep shipping ops the projection's built-in profilers do not know
(a linear-attention layer, a new gate, a multi-token-prediction head).  The
projection reports each one as a ``PRIMUS_OP_GAP`` marker; this module turns a
gap into a validated op model under
``primus/core/projection/op_models/generated/``, following the perf-model step
of the TraceLens model bring-up flow:

1. **Context** -- the gap and its config keys, the model YAML chain (comments
   included), the HF ``config.json``, any paper text given, the op-model API,
   and the existing op models / profilers closest to the gap's slot.  Written
   to ``<out>/op_bringup/<op_id>/context.md``.
2. **Create** -- a coder writes one op-model module from that context: an LLM
   through DSPy / LiteLLM, an external coding agent (``--op-coder-cmd``), or
   none (the context pack is the deliverable, for a human or an IDE agent).
3. **Validate** -- :mod:`primus.core.projection.op_models.validate` in a
   subprocess, on the workload's shapes: roofline floor, closed-form FLOPs,
   backward ratio, monotonicity, key coverage.  Failures go back to the coder
   as feedback, up to ``max_attempts`` times.
4. **Register** -- the passing module is copied into ``generated/`` (the
   projection discovers it on its next run) with a regression test.  A coder
   may instead judge the op negligible and *acknowledge* its keys, with the
   estimate, in ``generated/acknowledged.yaml``.

:class:`~primus.agents.projection_agent.ProjectionAgent` drives this for every
gap of a workload.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import textwrap
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence

# Section caps keep the context inside a coder's window.
_MAX_SECTION = 40_000
_MAX_PAPER = 60_000
_HF_URL = "https://huggingface.co/{repo}/raw/main/config.json"
_ACK_PREFIX = "ACKNOWLEDGE"

_GUIDE = """\
You are adding an analytical performance model for one op to the Primus
training-performance projection.  The projection prices a model layer by layer
on a simulated GPU; this op is currently priced as whatever built-in block it
resembles, which is wrong.

Write ONE Python module that defines and registers ONE `OpModel` subclass.

How to price things -- use the same primitives the built-in profilers use, via
the `OpContext` you are given (never hard-code hardware numbers):
  * every dense matmul / projection: `ctx.linear(m, n, k)` (fwd, bwd) or
    `ctx.gemm(...)`; M is `ctx.tokens` (TP and CP are already folded into it,
    so pass FULL output widths, as the attention profiler does);
  * a softmax-attention core (any new attention that is still softmax
    attention -- sliding window, sparse, latent, sink, ...): `ctx.sdpa(...)`,
    the FAv3 tile-level simulator;
  * any other fused kernel (chunked linear-attention recurrences, scans,
    blockwise scores): describe the matmuls ONE workgroup runs as `(m, n, k)`
    tiles and the number of workgroups, and call `ctx.fused_tiles(tiles, n_wg)`
    -- it prices the tile program on one CU and multiplies by the waves, the
    same construction FAv3 uses.  Fold a loop a workgroup runs over the
    sequence into one tile dimension, e.g. `(n_chunks * C, n, k)`;
  * elementwise ops, norms, gathers, convolutions with tiny kernels, state
    reads/writes: count the HBM bytes and use `ctx.stream_ms(bytes)`;
  * a memory-bound fused kernel takes `max(compute_ms, ctx.stream_ms(bytes))`.

Rules the validator enforces:
  * `name` snake_case; `slot` one of attention | attention_extra | mlp |
    layer_extra (attention / mlp REPLACE the built-in block on
    `layer_fraction` of layers; *_extra ADD to it);
  * `claims` maps the selector key(s) to the value(s) that turn the op on;
    every gap key is in `claims`, `consumes`, or `ignores` (key -> reason it
    does not change cost);
  * `references` cites the paper / model card / kernel source;
  * cost() returns an `OpCost` with finite, positive fwd/bwd ms, FLOPs and
    HBM bytes per rank per layer, and a `breakdown` whose entries sum to the
    totals; time never beats the roofline max(FLOPs/peak, bytes/BW);
  * bwd/fwd between 0.8 and 4 (dgrad + wgrad for matmuls, recompute for
    fused kernels);
  * `reference_flops()` returns (fwd, bwd) FLOPs counted straight from the
    paper's equations, written independently of cost(); it must agree with
    cost()'s FLOPs within 25 %;
  * cost grows with batch and sequence;
  * set `num_params` (full, unsharded, per layer) and `activation_bytes` (per
    rank) when the op changes them versus the block it replaces.

If the op is genuinely negligible (< ~1 % of its layer -- a scalar gate, a
routing constraint, a numerics-only flag), do not write a model.  Reply with
lines of the form

    ACKNOWLEDGE <config_key>: <why it does not matter, with your estimate>

for each gap key, and nothing else.

Otherwise reply with the complete module in a single ```python fenced block.
Model it the way the existing implementations below are modeled.
"""

_TEMPLATE = '''\
"""<Op name> (<model family>), training.

<The op's equations, from the paper, and how each kernel is priced.>
"""

from __future__ import annotations

from primus.core.projection.op_models.base import (
    SLOT_ATTENTION,
    OpContext,
    OpCost,
    OpModel,
    cfg_value,
    interleave_fraction,
)
from primus.core.projection.op_models.registry import register_op_model


@register_op_model
class MyOp(OpModel):
    name = "my_op"
    slot = SLOT_ATTENTION
    claims = {"selector_key": ("value",)}
    consumes = ("param_key_a", "param_key_b")
    ignores = {}
    references = ("Paper title. arXiv:XXXX.XXXXX",)

    def layer_fraction(self, model_config) -> float:
        return 1.0

    def cost(self, ctx: OpContext) -> OpCost:
        ...
        return OpCost(fwd_ms=..., bwd_ms=..., fwd_flops=..., bwd_flops=...,
                      fwd_bytes=..., bwd_bytes=..., breakdown={...})

    def reference_flops(self, ctx: OpContext):
        return fwd_flops, bwd_flops
'''


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------


@dataclass
class BringupResult:
    op_id: str
    status: str  # registered | acknowledged | failed | context_only | skipped
    message: str = ""
    op_name: Optional[str] = None
    module_path: Optional[str] = None
    test_path: Optional[str] = None
    context_path: Optional[str] = None
    attempts: int = 0
    validation: dict = field(default_factory=dict)
    acknowledged: dict = field(default_factory=dict)

    @property
    def changed_projection(self) -> bool:
        return self.status in ("registered", "acknowledged")

    def as_dict(self) -> dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# Context
# ---------------------------------------------------------------------------


def _op_models_dir(primus_root: Path) -> Path:
    return primus_root / "primus" / "core" / "projection" / "op_models"


def _clip(text: str, cap: int = _MAX_SECTION) -> str:
    if len(text) <= cap:
        return text
    return text[:cap] + f"\n... [{len(text) - cap} more characters omitted]\n"


def _yaml_chain_text(model_path: Path, seen: Optional[set] = None) -> List[tuple]:
    """The model YAML and every file it ``extends`` / ``bases``, comments kept."""
    seen = seen if seen is not None else set()
    model_path = model_path.resolve()
    if model_path in seen or not model_path.is_file():
        return []
    seen.add(model_path)
    text = model_path.read_text()
    out = [(model_path, text)]
    try:
        import yaml

        data = yaml.safe_load(text) or {}
    except Exception:
        data = {}
    parents = data.get("extends") or data.get("bases") or []
    if isinstance(parents, str):
        parents = [parents]
    for p in parents:
        out.extend(_yaml_chain_text(model_path.parent / str(p), seen))
    return out


def fetch_hf_config(
    spec: Optional[str], tokenizer_model: Optional[str] = None, timeout: float = 10.0
) -> Optional[str]:
    """``config.json`` text from a path, a URL, or a HF repo id; ``None`` on failure."""
    for cand in (spec, tokenizer_model):
        if not cand:
            continue
        cand = str(cand)
        p = Path(cand).expanduser()
        if p.is_file():
            return p.read_text()
        url = cand if cand.startswith(("http://", "https://")) else None
        if url is None and re.match(r"^[\w.\-]+/[\w.\-]+$", cand):
            url = _HF_URL.format(repo=cand)
        if url is None:
            continue
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "primus-op-bringup"})
            token = os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN")
            if token:
                req.add_header("Authorization", f"Bearer {token}")
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read().decode("utf-8", "replace")
        except Exception:
            continue
    return None


def _read_paper(spec: str) -> str:
    p = Path(spec).expanduser()
    if p.is_file():
        if p.suffix.lower() == ".pdf":
            try:
                from pypdf import PdfReader  # optional

                return "\n".join(page.extract_text() or "" for page in PdfReader(str(p)).pages)
            except Exception:
                return f"(PDF {p} could not be converted to text; pass a .txt / .md export)"
        return p.read_text(errors="replace")
    if spec.startswith(("http://", "https://")):
        try:
            with urllib.request.urlopen(spec, timeout=15) as resp:
                body = resp.read()
            if body[:4] == b"%PDF":
                return f"(URL {spec} is a PDF; pass a text export with --paper)"
            return body.decode("utf-8", "replace")
        except Exception as exc:
            return f"(could not fetch {spec}: {exc})"
    return spec  # inline text


def _source_of(obj_path: str) -> str:
    """Source of ``module:attr`` (or the whole module), best effort."""
    import importlib

    mod_name, _, attr = obj_path.partition(":")
    try:
        mod = importlib.import_module(mod_name)
        obj = mod
        for part in filter(None, attr.split(".")):
            obj = getattr(obj, part)
        return inspect.getsource(obj)
    except Exception as exc:
        return f"# (source of {obj_path} unavailable: {exc})"


# Built-in code each slot's new op should be modeled after.
_REFERENCES_BY_SLOT = {
    "attention": (
        "primus.core.projection.module_profilers.attention:AttentionProfiler._get_built_in_simulated_results",
        "primus.core.projection.simulation_backends.sdpa_simulator:SDPASimulator._simulate_tile_level",
    ),
    "attention_extra": (
        "primus.core.projection.module_profilers.attention:AttentionProfiler._get_built_in_simulated_results",
    ),
    "mlp": (
        "primus.core.projection.module_profilers.moe_mlp",
        "primus.core.projection.module_profilers.dense_mlp",
    ),
    "layer_extra": ("primus.core.projection.module_profilers.transformer_layer",),
}


def _existing_op_models(primus_root: Path, slot: str) -> List[Path]:
    gen = _op_models_dir(primus_root) / "generated"
    files = [p for p in sorted(gen.glob("*.py")) if not p.name.startswith("_")]
    same = [p for p in files if re.search(rf"slot\s*=\s*SLOT_{slot.upper()}\b", p.read_text())]
    return same + [p for p in files if p not in same]


def build_context(
    gap,
    *,
    primus_root: Path,
    model_path: Optional[Path] = None,
    raw_model: Optional[dict] = None,
    hf_config: Optional[str] = None,
    papers: Sequence[str] = (),
) -> str:
    """The context pack for one gap, as markdown."""
    sec: List[str] = [f"# Op bring-up: `{gap.op_id}`\n", _GUIDE]
    sec.append(
        "## The gap\n\n"
        f"- op id: `{gap.op_id}`\n- slot (best guess): `{gap.slot}`\n- severity: `{gap.severity}`\n"
        f"- why: {gap.reason}\n- config keys:\n\n```json\n{json.dumps(gap.keys, indent=2, default=str)}\n```\n"
    )
    if model_path is not None:
        chain = _yaml_chain_text(Path(model_path))
        body = "\n".join(f"### `{p.name}`\n\n```yaml\n{t}\n```\n" for p, t in chain)
        sec.append("## Model config (YAML chain, comments included)\n\n" + _clip(body))
    elif raw_model:
        sec.append(
            "## Model config (resolved)\n\n```json\n"
            + _clip(json.dumps(raw_model, indent=2, default=str))
            + "\n```\n"
        )
    tok = (raw_model or {}).get("tokenizer_model")
    hf = fetch_hf_config(hf_config, tok)
    if hf:
        sec.append(f"## HF config.json ({hf_config or tok})\n\n```json\n{_clip(hf, 20_000)}\n```\n")
    else:
        sec.append("## HF config.json\n\n(not available; rely on the YAML above)\n")
    for i, spec in enumerate(papers, 1):
        sec.append(
            f"## Paper {i}: {spec if len(spec) < 200 else spec[:200] + '...'}\n\n"
            + _clip(_read_paper(spec), _MAX_PAPER)
        )

    od = _op_models_dir(primus_root)
    sec.append(
        "## Op-model API (`op_models/base.py`)\n\n```python\n" + (od / "base.py").read_text() + "\n```\n"
    )
    for p in _existing_op_models(primus_root, gap.slot)[:3]:
        sec.append(f"## Existing op model: `generated/{p.name}`\n\n```python\n{_clip(p.read_text())}\n```\n")
    for ref in _REFERENCES_BY_SLOT.get(gap.slot, ()):
        sec.append(f"## Built-in reference: `{ref}`\n\n```python\n{_clip(_source_of(ref))}\n```\n")
    sec.append("## Module template\n\n```python\n" + _TEMPLATE + "```\n")
    return "\n".join(sec)


# ---------------------------------------------------------------------------
# Coders
# ---------------------------------------------------------------------------

Coder = Callable[[str, Path], str]
"""``coder(prompt, target_file) -> reply``: the module source or ACKNOWLEDGE lines."""


def extract_module(reply: str) -> Optional[str]:
    """Python source from a coder reply (fenced block, or the bare module)."""
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", reply or "", flags=re.S)
    blocks = [b for b in blocks if "OpModel" in b]
    if blocks:
        return max(blocks, key=len).strip() + "\n"
    if "class " in (reply or "") and "OpModel" in reply:
        return reply.strip() + "\n"
    return None


def parse_acknowledgements(reply: str) -> dict:
    out = {}
    for line in (reply or "").splitlines():
        m = re.match(rf"^\s*{_ACK_PREFIX}\s+([A-Za-z_][\w.]*)\s*:\s*(.+)$", line)
        if m:
            out[m.group(1)] = m.group(2).strip()
    return out


def llm_model_from_env() -> Optional[str]:
    return os.getenv("PRIMUS_OP_LLM_MODEL") or os.getenv("LLM_MODEL")


def dspy_coder(model: Optional[str] = None, *, timeout: int = 600, max_tokens: int = 16000) -> Coder:
    """Coder backed by an LLM through DSPy / LiteLLM.

    Uses the LM already configured in ``dspy.settings`` (the tuning agent's),
    else builds one from *model*, ``$PRIMUS_OP_LLM_MODEL`` or ``$LLM_MODEL`` -- a
    LiteLLM model string such as ``anthropic/claude-opus-4-5``, with the
    provider's usual key variable (or ``$LLM_API_KEY`` / ``$OPENAI_API_BASE``).
    """
    import dspy

    lm = getattr(dspy.settings, "lm", None)
    if lm is None:
        model = model or llm_model_from_env()
        if not model:
            raise RuntimeError("no LLM configured: pass --op-llm-model or set LLM_MODEL")
        kw: dict = {"cache": False, "timeout": timeout, "max_tokens": max_tokens}
        if os.getenv("LLM_API_KEY"):
            kw["api_key"] = os.environ["LLM_API_KEY"]
        if os.getenv("OPENAI_API_BASE"):
            kw["api_base"] = os.environ["OPENAI_API_BASE"]
        lm = dspy.LM(model, **kw)

    def _code(prompt: str, target: Path) -> str:
        out = lm(messages=[{"role": "user", "content": prompt}])
        first = out[0] if isinstance(out, list) and out else out
        return first.get("text", "") if isinstance(first, dict) else str(first)

    return _code


def command_coder(cmd_template: str, timeout: int = 1800) -> Coder:
    """Coder that shells out to a coding agent.

    ``cmd_template`` may use ``{prompt_file}``, ``{target_file}`` and
    ``{op_models_dir}``.  The agent either writes ``{target_file}`` or prints
    the module (or ACKNOWLEDGE lines) on stdout.
    """

    def _code(prompt: str, target: Path) -> str:
        prompt_file = target.with_suffix(".prompt.md")
        prompt_file.write_text(prompt)
        if target.exists():
            target.unlink()
        cmd = cmd_template.format(
            prompt_file=shlex.quote(str(prompt_file)),
            target_file=shlex.quote(str(target)),
            op_models_dir=shlex.quote(str(target.parent)),
        )
        proc = subprocess.run(
            cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout
        )
        if target.is_file() and target.stat().st_size > 0:
            return "```python\n" + target.read_text() + "\n```"
        return proc.stdout or ""

    return _code


def make_coder(kind: str, *, cmd: Optional[str] = None, model: Optional[str] = None) -> Optional[Coder]:
    if kind == "none":
        return None
    if kind == "command":
        if not cmd:
            raise ValueError("--op-coder command needs --op-coder-cmd")
        return command_coder(cmd)
    if kind == "dspy":
        return dspy_coder(model)
    raise ValueError(f"unknown coder {kind!r}")


# ---------------------------------------------------------------------------
# Validate / register
# ---------------------------------------------------------------------------


def _env(primus_root: Path, gpu_arch: Optional[str], env: Optional[dict]) -> dict:
    e = dict(env or os.environ)
    parts = [p for p in e.get("PYTHONPATH", "").split(os.pathsep) if p and Path(p) != primus_root]
    e["PYTHONPATH"] = os.pathsep.join([str(primus_root), *parts])
    e.setdefault("NNODES", "1")
    e.setdefault("GPUS_PER_NODE", "8")
    if gpu_arch:
        e["PRIMUS_GPU_ARCH"] = gpu_arch
    e["PRIMUS_OP_COVERAGE"] = "0"
    return e


def run_validator(
    module: Path,
    *,
    workload: Path,
    primus_root: Path,
    gap_keys: Iterable[str] = (),
    gpu_arch: Optional[str] = None,
    env: Optional[dict] = None,
    python: str = sys.executable,
    timeout: int = 600,
) -> dict:
    """Validator report as a dict; runs out of process so a broken module cannot hurt the caller."""
    cmd = [
        python,
        "-m",
        "primus.core.projection.op_models.validate",
        "--config",
        str(workload),
        "--op-file",
        str(module),
        "--gap-keys",
        ",".join(gap_keys),
        "--json",
    ]
    if gpu_arch:
        cmd += ["--gpu-arch", gpu_arch]
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(primus_root),
            env=_env(primus_root, gpu_arch, env),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=timeout,
        )
        out = proc.stdout or ""
    except subprocess.TimeoutExpired:
        return {
            "ok": False,
            "checks": [{"name": "load", "status": "fail", "detail": f"validator timed out after {timeout}s"}],
            "cases": [],
        }
    start = out.find("{\n")
    try:
        return json.loads(out[start:]) if start >= 0 else json.loads(out)
    except ValueError:
        return {
            "ok": False,
            "checks": [{"name": "load", "status": "fail", "detail": out[-3000:]}],
            "cases": [],
        }


def validation_feedback(report: dict) -> str:
    lines = [
        f"Your module FAILED validation (op={report.get('op')}). Fix these and return the whole module again:"
    ]
    for c in report.get("checks", []):
        if c.get("status") != "pass":
            lines.append(f"- [{c['status'].upper()}] {c['name']}: {c.get('detail', '')}")
    for case in report.get("cases", []):
        lines.append(
            f"  case {case['label']} (B={case['batch_size']}, S={case['seq_len']}): fwd={case['fwd_ms']:.4f} ms "
            f"bwd={case['bwd_ms']:.4f} ms roofline_fwd={case['roofline_fwd_ms']:.4f} ms "
            f"fwd_flops={case['fwd_flops']:.3e} fwd_bytes={case['fwd_bytes']:.3e}"
        )
    return "\n".join(lines)


_TEST_TEMPLATE = '''\
"""Regression test for the generated op model ``{name}`` (written by op bring-up)."""

from pathlib import Path

import pytest

pytest.importorskip("yaml")

from primus.core.projection.op_models.registry import registered_op_models
from primus.core.projection.op_models.validate import load_workload, validate_op_model

_ROOT = Path(__file__).resolve().parents[{depth}]
_WORKLOAD = _ROOT / {workload}
_GAP_KEYS = {gap_keys}


@pytest.mark.skipif(not _WORKLOAD.is_file(), reason="workload YAML not in this checkout")
def test_{name}_validates(monkeypatch):
    monkeypatch.setenv("NNODES", "1")
    monkeypatch.setenv("GPUS_PER_NODE", "8")
    monkeypatch.setenv("PRIMUS_OP_COVERAGE", "0")
    ops = {{c.name: c for c in registered_op_models()}}
    assert "{name}" in ops
    wl = load_workload(str(_WORKLOAD), simulate_backends=False)
    rep = validate_op_model(ops["{name}"], wl, gap_keys=_GAP_KEYS)
    assert rep.ok, rep.feedback()
'''


def _write_test(primus_root: Path, name: str, workload: Path, gap_keys: Sequence[str]) -> Path:
    tdir = primus_root / "tests" / "unit_tests" / "core" / "projection" / "op_models"
    tdir.mkdir(parents=True, exist_ok=True)
    init = tdir / "__init__.py"
    if not init.exists():
        init.write_text("")
    try:
        rel = str(workload.resolve().relative_to(primus_root.resolve()))
    except ValueError:
        rel = str(workload.resolve())
    depth = len(tdir.relative_to(primus_root).parts)
    path = tdir / f"test_generated_{name}.py"
    path.write_text(
        _TEST_TEMPLATE.format(
            name=name, workload=json.dumps(rel), gap_keys=json.dumps(list(gap_keys)), depth=depth
        )
    )
    return path


def acknowledge_keys(primus_root: Path, entries: dict) -> Path:
    path = _op_models_dir(primus_root) / "generated" / "acknowledged.yaml"
    lines = [] if not path.exists() else [path.read_text().rstrip("\n"), ""]
    for k, why in entries.items():
        why = " ".join(str(why).split())
        lines.append(f"{k}: >-\n" + textwrap.indent(textwrap.fill(why, 86), "  "))
    path.write_text("\n".join(lines) + "\n")
    return path


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def bring_up_op(
    gap,
    *,
    workload: Path,
    primus_root: Path,
    out_dir: Path,
    coder: Optional[Coder],
    gpu_arch: Optional[str] = None,
    model_path: Optional[Path] = None,
    raw_model: Optional[dict] = None,
    hf_config: Optional[str] = None,
    papers: Sequence[str] = (),
    max_attempts: int = 3,
    register: bool = True,
    env: Optional[dict] = None,
    log: Callable[[str], None] = print,
) -> BringupResult:
    """Context -> Create -> Validate -> Register for one gap."""
    work = Path(out_dir) / "op_bringup" / gap.op_id
    work.mkdir(parents=True, exist_ok=True)
    context = build_context(
        gap,
        primus_root=primus_root,
        model_path=model_path,
        raw_model=raw_model,
        hf_config=hf_config,
        papers=papers,
    )
    ctx_path = work / "context.md"
    ctx_path.write_text(context)
    res = BringupResult(gap.op_id, "context_only", context_path=str(ctx_path))
    log(f"[op-bringup] {gap.op_id}: context pack {ctx_path} ({len(context) // 1000} kB)")
    if coder is None:
        res.message = "no coder configured; hand context.md to a coding agent, then re-run the projection"
        return _finish(res, work)

    feedback = ""
    gap_keys = list(gap.keys)
    for attempt in range(1, max_attempts + 1):
        res.attempts = attempt
        target = work / f"attempt_{attempt}.py"
        prompt = context + ("\n\n## Feedback on your previous attempt\n\n" + feedback if feedback else "")
        t0 = time.time()
        try:
            reply = coder(prompt, target)
        except Exception as exc:
            res.status, res.message = "failed", f"coder failed: {type(exc).__name__}: {exc}"
            break
        (work / f"attempt_{attempt}.reply.md").write_text(reply or "")
        log(f"[op-bringup] {gap.op_id}: attempt {attempt} reply in {time.time() - t0:.0f}s")

        acks = parse_acknowledgements(reply)
        source = extract_module(reply)
        if acks and source is None:
            unknown = sorted(set(acks) - set(gap_keys))
            missing = sorted(set(gap_keys) - set(acks))
            if unknown or missing:
                feedback = (
                    f"ACKNOWLEDGE lines must cover exactly the gap keys {gap_keys}; "
                    f"unknown: {unknown}, missing: {missing}."
                )
                continue
            res.acknowledged = acks
            if register:
                acknowledge_keys(primus_root, acks)
            res.status, res.message = (
                "acknowledged",
                "judged negligible; keys added to generated/acknowledged.yaml",
            )
            break
        if source is None:
            feedback = (
                "Your reply contained no ```python module with an OpModel subclass and no ACKNOWLEDGE lines."
            )
            continue
        target.write_text(source)
        report = run_validator(
            target, workload=workload, primus_root=primus_root, gap_keys=gap_keys, gpu_arch=gpu_arch, env=env
        )
        res.validation = report
        (work / f"attempt_{attempt}.validation.json").write_text(json.dumps(report, indent=2))
        if not report.get("ok"):
            feedback = validation_feedback(report)
            log(f"[op-bringup] {gap.op_id}: attempt {attempt} failed validation")
            continue
        name = str(report.get("op") or gap.op_id)
        res.op_name = name
        if register:
            dest = _op_models_dir(primus_root) / "generated" / f"{name}.py"
            if dest.exists():
                shutil.copy2(dest, work / f"previous_{name}.py")
            shutil.copy2(target, dest)
            res.module_path = str(dest)
            res.test_path = str(_write_test(primus_root, name, workload, gap_keys))
        else:
            res.module_path = str(target)
        res.status, res.message = "registered", f"validated on attempt {attempt}"
        break
    else:
        res.status, res.message = "failed", f"no valid module after {max_attempts} attempts"
    log(f"[op-bringup] {gap.op_id}: {res.status} -- {res.message}")
    return _finish(res, work)


def _finish(res: BringupResult, work: Path) -> BringupResult:
    (work / "summary.json").write_text(json.dumps(res.as_dict(), indent=2, default=str))
    md = [f"# Op bring-up `{res.op_id}`: {res.status}", "", res.message, ""]
    if res.module_path:
        md.append(f"- module: `{res.module_path}`")
    if res.test_path:
        md.append(f"- test: `{res.test_path}`")
    if res.acknowledged:
        md += [f"- acknowledged `{k}`: {v}" for k, v in res.acknowledged.items()]
    for case in (res.validation or {}).get("cases", []):
        md.append(
            f"- {case['label']}: fwd {case['fwd_ms']:.4f} ms, bwd {case['bwd_ms']:.4f} ms "
            f"(roofline fwd {case['roofline_fwd_ms']:.4f} ms)"
        )
    (work / "report.md").write_text("\n".join(md) + "\n")
    return res


def bring_up_gaps(gaps: Sequence, **kwargs) -> List[BringupResult]:
    """Bring up every blocking gap, de-duplicated by op id."""
    out, seen = [], set()
    for g in gaps:
        if g.severity != "gap" or g.op_id in seen:
            continue
        seen.add(g.op_id)
        out.append(bring_up_op(g, **kwargs))
    return out
