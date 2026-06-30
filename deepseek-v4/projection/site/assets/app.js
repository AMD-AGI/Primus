"use strict";

// DeepSeek-V4 performance projection — static, no-build. Implements the math in
// design/04-projection-math.md. All breakdown times are microseconds (us) for
// one microbatch (seq from capture); the projection scales to a full model run.

const STATE = {
  data: null,
  gpu: "MI355X",
  controls: null,
};

const $ = (sel) => document.querySelector(sel);
const el = (tag, attrs = {}, ...kids) => {
  const n = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") n.className = v;
    else if (k === "html") n.innerHTML = v;
    else n.setAttribute(k, v);
  }
  for (const kid of kids) n.append(kid?.nodeType ? kid : document.createTextNode(kid ?? ""));
  return n;
};
const fmt = (x, d = 1) =>
  x == null || !isFinite(x) ? "—" : Number(x).toLocaleString(undefined, { maximumFractionDigits: d, minimumFractionDigits: d });
const fmtInt = (x) => (x == null || !isFinite(x) ? "—" : Math.round(x).toLocaleString());

// ---------------------------------------------------------------------------
// Load
// ---------------------------------------------------------------------------
function modelFromQuery() {
  const m = new URLSearchParams(location.search).get("model");
  return m === "flash" || m === "pro" ? m : "pro";
}

async function loadModel(model) {
  const res = await fetch(`./data/${model}.json`, { cache: "no-store" });
  if (!res.ok) throw new Error(`failed to load data/${model}.json (${res.status})`);
  return res.json();
}

// ---------------------------------------------------------------------------
// Controls
// ---------------------------------------------------------------------------
function defaultControls(data) {
  const hw = data.hardware || {};
  const m355 = hw.MI355X || { peak_tflops_bf16: 2500, hbm_bandwidth_gbps: 8000 };
  const m455 = hw.MI455X || { peak_tflops_bf16: 5000, hbm_bandwidth_gbps: 16000 };
  const isPro = data.model === "pro";
  const optBytes = data.optimizer?.bytes_per_param && data.optimizer.bytes_per_param !== 18
    ? data.optimizer.bytes_per_param : 30;
  return {
    world: isPro ? 256 : 32,
    pp: isPro ? 16 : 4, vpp: 1, ep: 8, tp: 1, cp: 1,
    mbs: 1, gbs: isPro ? 1024 : 256,
    recompute: isPro ? "full" : "first-n",
    recomputeLayers: isPro ? 0 : 3,
    ppLayout: data.model_config?.pipeline_layout || "",
    optEff: 0.7, computeEff: 1.0, calibFactor: 0.91, bytesPerParam: optBytes,
    peak355: m355.peak_tflops_bf16, bw355: m355.hbm_bandwidth_gbps,
    peak455: m455.peak_tflops_bf16, bw455: m455.hbm_bandwidth_gbps,
    // Modeling mode: "trace" (derive per-layer fwd/bwd from the breakdown JSON)
    // or "manual" (user types per-cr fwd/bwd directly). Manual values are stored
    // per GPU because a hand-entered time already targets a specific GPU, so the
    // MI355->MI455 scaling is bypassed in manual mode.
    modelMode: "trace",
    man: {
      MI355X: emptyManual(),
      MI455X: emptyManual(),
    },
  };
}

// Per-cr manual fwd/bwd holders (µs). null = "not set yet" -> falls back to the
// trace-derived value, so toggling into manual mode never changes the result
// until the user actually edits a field.
function emptyManual() {
  return { f0: null, b0: null, f4: null, b4: null, f128: null, b128: null };
}

const MANUAL_CR_KEYS = { "0": ["f0", "b0"], "4": ["f4", "b4"], "128": ["f128", "b128"] };

const CONTROL_DEFS = [
  { key: "world", label: "World size (GPUs)", kind: "int" },
  { key: "pp", label: "PP (pipeline)", kind: "int" },
  { key: "vpp", label: "VPP (interleave)", kind: "int" },
  { key: "ep", label: "EP (expert)", kind: "int" },
  { key: "dp", label: "DP (derived)", kind: "ro" },
  { key: "tp", label: "TP (tensor)", kind: "int" },
  { key: "cp", label: "CP (context)", kind: "int" },
  { key: "mbs", label: "Micro batch size", kind: "int" },
  { key: "gbs", label: "Global batch size", kind: "int" },
  { key: "recompute", label: "Recompute", kind: "sel" },
  { key: "recomputeLayers", label: "Recompute layers", kind: "int" },
  { key: "ppLayout", label: "PP layout", kind: "txt", full: true },
  { key: "bytesPerParam", label: "Optim bytes/param", kind: "int" },
  { key: "calibFactor", label: "Calibration factor", kind: "f" },
  { key: "optEff", label: "Optim efficiency", kind: "f" },
  { key: "computeEff", label: "MI455 compute eff", kind: "f" },
  { key: "peak355", label: "MI355 peak TFLOPs", kind: "int" },
  { key: "bw355", label: "MI355 HBM GB/s", kind: "int" },
  { key: "peak455", label: "MI455 peak TFLOPs", kind: "int" },
  { key: "bw455", label: "MI455 HBM GB/s", kind: "int" },
];

// DP is derived from the user-set world size: DP = world / (PP*TP*CP). EP is a
// sub-grouping of DP (EP <= DP) and does not multiply world size.
const derivedDP = (c) => {
  const denom = c.pp * c.tp * c.cp;
  if (!Number.isFinite(denom) || denom <= 0 || !Number.isFinite(c.world)) return NaN;
  return c.world / denom;
};

function parseControlValue(input, kind) {
  if (kind === "sel" || kind === "txt") return input.value;
  if (kind === "int") return input.value.trim() === "" ? NaN : Number(input.value);
  if (kind === "f") return input.value.trim() === "" ? NaN : Number(input.value);
  return input.value;
}

const isPositiveInt = (x) => Number.isInteger(x) && x > 0;
const isNonNegativeInt = (x) => Number.isInteger(x) && x >= 0;
const isPositiveNumber = (x) => Number.isFinite(x) && x > 0;

function renderControls() {
  const grid = $("#controls-grid");
  grid.innerHTML = "";
  const c = STATE.controls;
  for (const def of CONTROL_DEFS) {
    const { key, label, kind } = def;
    const field = el("div", { class: "field" });
    if (def.full) field.classList.add("field--full");
    field.append(el("span", {}, label));
    let input;
    if (kind === "sel") {
      input = el("select", { id: `ctl-${key}` });
      for (const opt of ["none", "full", "first-n"]) {
        const o = el("option", { value: opt }, opt);
        if (c[key] === opt) o.selected = true;
        input.append(o);
      }
    } else if (kind === "ro") {
      const dp = derivedDP(c);
      input = el("input", { id: `ctl-${key}`, value: Number.isFinite(dp) ? dp : "—", disabled: "true" });
    } else if (kind === "txt") {
      input = el("input", { id: `ctl-${key}`, type: "text", value: c[key] || "" });
    } else {
      input = el("input", { id: `ctl-${key}`, type: "number", value: c[key], step: kind === "f" ? "0.05" : "1" });
    }
    if (kind !== "ro") {
      input.addEventListener("change", () => {
        c[key] = parseControlValue(input, kind);
        renderAll();
      });
    }
    field.append(input);
    grid.append(field);
  }
}

// Prefill the active GPU's manual fields from the trace-derived times so that
// switching into manual mode (or switching GPU while in manual mode) starts from
// the current baseline instead of empty boxes. Already-set fields are kept.
function prefillManual(gpu) {
  const c = STATE.controls;
  const m = c.man[gpu];
  for (const cr of ["0", "4", "128"]) {
    const [fk, bk] = MANUAL_CR_KEYS[cr];
    const t = layerTimes(STATE.data, cr, gpu, c);
    if (!isPositiveNumber(m[fk])) m[fk] = Math.round(t.fwd);
    if (!isPositiveNumber(m[bk])) m[bk] = Math.round(t.bwd);
  }
}

function renderModeSwitch() {
  document.querySelectorAll(".mode-tab").forEach((t) => {
    t.classList.toggle("is-active", t.dataset.mode === STATE.controls.modelMode);
  });
}

function renderManualGrid() {
  const host = $("#manual-grid");
  if (!host) return;
  host.hidden = STATE.controls.modelMode !== "manual";
  host.innerHTML = "";
  if (host.hidden) return;
  const c = STATE.controls, gpu = STATE.gpu;
  const counts = STATE.data.model_config.cr_layer_counts || {};
  host.append(el("p", { class: "muted manual-grid__hint" },
    `Per-layer fwd/bwd (µs) for ${gpu}. Set values override the trace; placeholders show the current trace baseline. Non-layer parts (embedding / output / loss) and MTP still come from the trace breakdown.`));
  const rows = el("div", { class: "manual-rows" });
  for (const cr of ["0", "4", "128"]) {
    const [fk, bk] = MANUAL_CR_KEYS[cr];
    const m = c.man[gpu];
    const t = layerTimes(STATE.data, cr, gpu, c);
    const row = el("div", { class: "manual-row" });
    row.append(el("span", { class: `manual-row__lab cr-tag cr-${cr}` }, `cr=${cr} ×${counts[cr] || 0}`));
    for (const [label, key, traceVal] of [["fwd", fk, t.fwd], ["bwd", bk, t.bwd]]) {
      const field = el("div", { class: "field" });
      field.append(el("span", {}, `${label} µs`));
      const input = el("input", {
        id: `man-${gpu}-${key}`, type: "number", step: "1", min: "0",
        placeholder: String(Math.round(traceVal)),
      });
      if (isPositiveNumber(m[key])) input.value = m[key];
      input.addEventListener("change", () => {
        const v = input.value.trim();
        m[key] = v === "" ? null : Number(v);
        renderAll();
      });
      field.append(input);
      row.append(field);
    }
    rows.append(row);
  }
  host.append(rows);
}

// ---------------------------------------------------------------------------
// Hardware scaling (Step 6)
// ---------------------------------------------------------------------------
function rowScaledTime(row, gpu, c) {
  if (gpu === "MI355X") return row.time_us;
  const compute = row.class === "compute_bound";
  if (compute) return row.time_us * (c.peak355 / c.peak455) / c.computeEff;
  return row.time_us * (c.bw355 / c.bw455);
}
function rowTflops(row, scaledTimeUs) {
  if (!row.flops || !scaledTimeUs) return null;
  return row.flops / (scaledTimeUs * 1e-6) / 1e12;
}

const sumTime = (list, gpu, c) => list.reduce((a, r) => a + rowScaledTime(r, gpu, c), 0);
const sumFlops = (list) => list.reduce((a, r) => a + (r.flops || 0), 0);

// ---------------------------------------------------------------------------
// Per-layer base (Step 0/1)
// ---------------------------------------------------------------------------
function layerTimes(data, cr, gpu, c) {
  const L = data.layers[cr];
  if (!L) return { fwd: 0, bwd: 0, fFlops: 0, bFlops: 0 };
  const aF = L.attention.forward, aB = L.attention.backward;
  const mF = L.moe.forward, mB = L.moe.backward;
  let fwd = sumTime(aF, gpu, c) + sumTime(mF, gpu, c);
  let bwd = sumTime(aB, gpu, c) + sumTime(mB, gpu, c);
  let fFlops = sumFlops(aF) + sumFlops(mF);
  let bFlops = sumFlops(aB) + sumFlops(mB);
  return { fwd, bwd, fFlops, bFlops };
}

// Effective per-layer time used by the projection. In "manual" mode a set field
// overrides the trace-derived time for the active GPU; unset fields fall back to
// trace. FLOPs always stay analytic/trace-derived (manual only overrides time),
// so TFLOP/s/GPU remains meaningful.
function effectiveLayerTimes(data, cr, gpu, c) {
  const trace = layerTimes(data, cr, gpu, c);
  if (c.modelMode !== "manual") return trace;
  const m = (c.man && c.man[gpu]) || {};
  const [fk, bk] = MANUAL_CR_KEYS[cr] || [];
  return {
    fwd: isPositiveNumber(m[fk]) ? m[fk] : trace.fwd,
    bwd: isPositiveNumber(m[bk]) ? m[bk] : trace.bwd,
    fFlops: trace.fFlops,
    bFlops: trace.bFlops,
  };
}

function expandLayoutRepeats(layout) {
  let out = layout;
  let prev;
  do {
    prev = out;
    out = out.replace(/\(([^()]+)\)\*(\d+)/g, (_m, body, count) => body.repeat(Number(count)));
  } while (out !== prev);
  return out;
}

function parsePipelineLayout(layout, numLayers, chunks) {
  const raw = String(layout ?? "").trim();
  if (!raw) return { ok: true, stages: null, counts: [], normalized: "", message: "empty layout; using balanced fallback" };
  const normalized = expandLayoutRepeats(raw.replace(/^['"]|['"]$/g, ""));
  if (/[()]/.test(normalized)) {
    return { ok: false, stages: null, counts: [], normalized, message: "unsupported nested or malformed repeat expression" };
  }
  const specs = normalized.split("|").map((x) => x.trim()).filter(Boolean);
  if (specs.length !== chunks) {
    return {
      ok: false,
      stages: null,
      counts: specs.map((spec) => [...spec.matchAll(/[tT](?:\*(\d+))?/g)].reduce((a, m) => a + Number(m[1] || 1), 0)),
      normalized,
      message: `layout has ${specs.length} stages, expected PP*VPP=${chunks}`,
    };
  }
  let nextLayer = 0;
  const out = [];
  const counts = [];
  for (const spec of specs) {
    const layers = [];
    for (const m of spec.matchAll(/[tT](?:\*(\d+))?/g)) {
      const n = m[1] ? Number(m[1]) : 1;
      for (let i = 0; i < n && nextLayer < numLayers; i++) layers.push(nextLayer++);
    }
    counts.push(layers.length);
    out.push(layers);
  }
  if (nextLayer !== numLayers) {
    return {
      ok: false,
      stages: null,
      counts,
      normalized,
      message: `layout maps ${nextLayer} decoder layers, expected ${numLayers}`,
    };
  }
  return { ok: true, stages: out, counts, normalized, message: "layout applied" };
}

function recomputeLayer(c, stageOrdinal) {
  if (c.recompute === "full") return true;
  if (c.recompute === "first-n") return stageOrdinal < Math.max(0, c.recomputeLayers || 0);
  return false;
}

function validateControls(data, c, gpu = STATE.gpu) {
  const errors = [];
  const warnings = [];
  const ints = ["world", "pp", "vpp", "ep", "tp", "cp", "mbs", "gbs", "bytesPerParam", "peak355", "bw355", "peak455", "bw455"];
  for (const key of ints) {
    if (!isPositiveInt(c[key])) errors.push(`${key} must be a positive integer.`);
  }
  if (!isNonNegativeInt(c.recomputeLayers)) errors.push("recomputeLayers must be a non-negative integer.");
  for (const key of ["calibFactor", "optEff", "computeEff"]) {
    if (!isPositiveNumber(c[key])) errors.push(`${key} must be a positive number.`);
  }
  if (!["none", "full", "first-n"].includes(c.recompute)) errors.push(`unsupported recompute mode: ${c.recompute}`);

  const denom = c.pp * c.tp * c.cp;
  const dp = derivedDP(c);
  if (isPositiveInt(c.world) && isPositiveInt(denom) && c.world % denom !== 0) {
    errors.push(`world must be divisible by PP*TP*CP (${denom}); got world=${c.world}.`);
  }
  if (Number.isFinite(dp) && isPositiveInt(c.gbs) && isPositiveInt(c.mbs) && !Number.isInteger(c.gbs / (dp * c.mbs))) {
    errors.push(`GA must be an integer: GBS / (DP*MBS) = ${c.gbs} / (${dp}*${c.mbs}).`);
  }
  if (Number.isFinite(dp) && isPositiveInt(c.ep) && c.ep > dp) {
    errors.push(`EP must be <= DP; got EP=${c.ep}, DP=${dp}.`);
  }

  const chunks = c.pp * c.vpp;
  const layout = parsePipelineLayout(c.ppLayout, data.model_config.compress_ratios.length, chunks);
  if (!layout.ok) errors.push(`PP layout invalid: ${layout.message}.`);

  if (c.modelMode === "manual") {
    const man = (c.man && c.man[gpu]) || {};
    for (const cr of ["0", "4", "128"]) {
      const count = data.model_config.cr_layer_counts?.[cr] || 0;
      if (!count) continue; // cr not used by this model; ignore its inputs
      for (const k of MANUAL_CR_KEYS[cr]) {
        const v = man[k];
        if (v != null && v !== "" && !isPositiveNumber(Number(v))) {
          errors.push(`manual ${k} (cr=${cr}) must be a positive number (µs).`);
        }
      }
    }
    warnings.push(`Manual mode for ${gpu}: per-cr fwd/bwd you set override the trace; unset fields fall back to trace. Non-layer (embedding/output/loss) and MTP still use the trace breakdown.`);
  }

  const captureEp = data.capture?.ep;
  if (captureEp && c.ep !== captureEp) {
    warnings.push(`EP=${c.ep} is only partially modeled; traces captured EP=${captureEp}, so MoE dispatch/combine are not re-estimated.`);
  } else {
    warnings.push(`EP is a consistency control only; captured MoE dispatch/combine are reused from EP=${captureEp || 8}.`);
  }
  if (c.tp !== 1 || c.cp !== 1) {
    warnings.push("TP/CP values affect derived DP/GA/optimizer only; TP/CP layer compute and communication are not re-modeled.");
  }
  if (gpu === "MI355X") warnings.push("MI455 peak/bandwidth/compute-eff controls affect only the MI455X tab.");
  warnings.push(`Sequence length is fixed at captured seq=${data.capture.seq_length}; changing seq is not currently exposed.`);

  return { errors, warnings, layout, dp };
}

function nonLayer(data, which, phase, gpu, c) {
  const bd = data.non_layer[which];
  return bd ? sumTime(bd[phase], gpu, c) : 0;
}
function nonLayerFlops(data, which, phase) {
  const bd = data.non_layer[which];
  return bd ? sumFlops(bd[phase]) : 0;
}

function mtpTimes(data, gpu, c, lt) {
  const cfg = data.model_config;
  const depth = cfg.mtp_num_layers || 0;
  if (!depth) return { fwd: 0, bwd: 0, ehUs: 0 };

  const cr = String((cfg.mtp_compress_ratios && cfg.mtp_compress_ratios[0]) || 0);
  const inner = lt[cr] || lt["0"];
  const innerBwd = inner.bwd + (c.recompute === "full" ? inner.fwd : 0);
  const outF = nonLayer(data, "output", "forward", gpu, c) + nonLayer(data, "loss", "forward", gpu, c);
  const outB = nonLayer(data, "output", "backward", gpu, c) + nonLayer(data, "loss", "backward", gpu, c);

  // No MTP trace row exists yet. Approximate eh_proj as GEMM-like work scaled
  // from the measured output projection time, then split fwd/bwd as 1:2.
  const af = data.analytic_flops || {};
  const mtp = af.mtp || {};
  const outFlops = af.output_flops || 0;
  const outUs = outF + outB;
  const ehUs = outFlops > 0 ? outUs * ((mtp.eh_proj_flops || 0) / outFlops) : 0;

  return {
    fwd: depth * (inner.fwd + outF) + ehUs / 3,
    bwd: depth * (innerBwd + outB) + (2 * ehUs) / 3,
    ehUs,
  };
}

// ---------------------------------------------------------------------------
// Param estimate (Step 4)
// ---------------------------------------------------------------------------
function estimateParams(cfg) {
  // Prefer the exact V4 count emitted by parse_trace (MLA low-rank attention +
  // MoE + tied-free embedding/output). Fall back to a crude estimate.
  if (cfg.total_params) return cfg.total_params;
  const h = cfg.hidden_size, exp = cfg.num_experts, mff = cfg.moe_ffn_hidden_size;
  const sff = cfg.moe_shared_expert_intermediate_size || mff, V = cfg.vocab_size, L = cfg.num_layers;
  const perLayer = 4 * h * h + exp * 3 * h * mff + 3 * h * sff;
  return L * perLayer + 2 * V * h;
}

function estimateOptimizerParams(data, c, dp) {
  const cfg = data.model_config;
  const totalParams = estimateParams(cfg);
  const pp = Math.max(1, c.pp);
  const tp = Math.max(1, c.tp);
  const dpSize = Math.max(1, dp);

  // total_params is a full-model count. CP does not shard weights; EP is already
  // represented in the full expert count and cancels out with the DP replica
  // count for ZeRO-1 optimizer sharding, so the average rank owns total/(PP*TP)
  // model params and updates total/(PP*TP*DP) params per optimizer step.
  const localModelParams = totalParams / (pp * tp);
  const perRankParams = localModelParams / dpSize;
  const measuredUs = data.optimizer?.time_us ?? null;
  return { totalParams, localModelParams, perRankParams, measuredUs };
}

// ---------------------------------------------------------------------------
// Pipeline mapping + projection (Steps 2-5)
// ---------------------------------------------------------------------------
function project(data, gpu, c, validation = validateControls(data, c, gpu)) {
  if (validation.errors.length) return null;
  const cfg = data.model_config;
  const crs = cfg.compress_ratios;
  const L = crs.length;
  const dp = validation.dp;
  const world = c.world;
  const lt = {};
  for (const cr of ["0", "4", "128"]) lt[cr] = effectiveLayerTimes(data, cr, gpu, c);

  // Step 2: assign layers to PP*VPP chunks -> devices
  const C = c.pp * c.vpp;
  const perChunk = Math.ceil(L / C);
  const Df = new Array(c.pp).fill(0), Db = new Array(c.pp).fill(0);
  const layoutInfo = validation.layout;
  const layoutStages = layoutInfo.ok ? layoutInfo.stages : null;
  const stageOrdinals = new Array(c.pp).fill(0);
  if (layoutStages) {
    layoutStages.forEach((layers, chunk) => {
      const dev = chunk % c.pp;
      for (const i of layers) {
        const t = lt[String(crs[i])] || { fwd: 0, bwd: 0 };
        const recompute = recomputeLayer(c, stageOrdinals[dev]++);
        Df[dev] += t.fwd;
        Db[dev] += t.bwd + (recompute ? t.fwd : 0);
      }
    });
  } else {
    for (let i = 0; i < L; i++) {
      const chunk = Math.floor(i / perChunk);
      const dev = chunk % c.pp;
      const t = lt[String(crs[i])] || { fwd: 0, bwd: 0 };
      const recompute = recomputeLayer(c, stageOrdinals[dev]++);
      Df[dev] += t.fwd;
      Db[dev] += t.bwd + (recompute ? t.fwd : 0);
    }
  }
  // non-layer parts on first / last device
  Df[0] += nonLayer(data, "embedding", "forward", gpu, c);
  Db[0] += nonLayer(data, "embedding", "backward", gpu, c);
  const last = c.pp - 1;
  Df[last] += nonLayer(data, "output", "forward", gpu, c) + nonLayer(data, "loss", "forward", gpu, c);
  Db[last] += nonLayer(data, "output", "backward", gpu, c) + nonLayer(data, "loss", "backward", gpu, c);
  const mtp = mtpTimes(data, gpu, c, lt);
  Df[last] += mtp.fwd;
  Db[last] += mtp.bwd;

  const critF = Math.max(...Df), critB = Math.max(...Db);

  // Step 3: pipeline compute time (us) ; GA = gbs/(dp*mbs)
  // calibFactor corrects the single-layer-capture -> full-model bias (~0.93,
  // from the flash 16-layer single-node calibration; see design/06).
  const ga = c.gbs / (dp * c.mbs);
  const pipeUs = (ga + (c.pp - 1) / c.vpp) * (critF + critB) * c.calibFactor;
  const bubbleFrac = (c.pp - 1) / c.vpp / (ga + (c.pp - 1) / c.vpp);

  // Step 4: optimizer (memory-bound, zero1-sharded over DP; CP does not shard params)
  const optParams = estimateOptimizerParams(data, c, dp);
  const totalParams = optParams.totalParams;
  const perRankParams = optParams.perRankParams;
  const bw = (gpu === "MI355X" ? c.bw355 : c.bw455) * 1e9; // bytes/s
  const optTimeS = (perRankParams * c.bytesPerParam) / bw / c.optEff;
  const optUs = optTimeS * 1e6;

  // Step 5: totals
  const iterUs = pipeUs + optUs;
  const iterS = iterUs * 1e-6;
  const seq = data.capture.seq_length;
  const tokIter = c.gbs * seq;
  const tokS = tokIter / iterS;
  const tokSgpu = tokS / world;

  // FLOPs/iter — Megatron-convention V4 analytic model FLOPs (independent of
  // recompute; recompute adds time, not model flops). Falls back to breakdown
  // gemm flops if analytic_flops is absent.
  const counts = cfg.cr_layer_counts;
  let fMb = 0;
  const af = data.analytic_flops;
  if (af && af.per_cr_layer_flops) {
    for (const cr of ["0", "4", "128"]) fMb += (counts[cr] || 0) * (af.per_cr_layer_flops[cr] || 0);
    fMb += af.output_flops || 0;
    if (af.mtp) {
      fMb += (af.mtp.inner_layer_flops || 0)
        + (af.mtp.eh_proj_flops || 0)
        + (af.mtp.extra_logits_flops || 0)
        + (af.mtp.hc_head_flops || 0);
    }
  } else {
    for (const cr of ["0", "4", "128"]) fMb += (counts[cr] || 0) * (lt[cr].fFlops + lt[cr].bFlops);
    fMb += nonLayerFlops(data, "output", "forward") + nonLayerFlops(data, "output", "backward");
  }
  const flopsIter = fMb * ga * dp;
  const tflopsGpu = flopsIter / iterS / world / 1e12;

  return {
    lt, Df, Db, critF, critB, ga, pipeUs, bubbleFrac, world, dp, totalParams,
    layoutApplied: Boolean(layoutStages),
    layoutCounts: layoutInfo.counts,
    layoutMessage: layoutInfo.message,
    localModelParams: optParams.localModelParams, perRankParams, measuredOptUs: optParams.measuredUs,
    mtp, optUs, iterUs, tokIter, tokS, tokSgpu, flopsIter, tflopsGpu, seq,
  };
}

// ---------------------------------------------------------------------------
// Rendering
// ---------------------------------------------------------------------------
function renderConfig() {
  const cfg = STATE.data.model_config;
  const grid = $("#config-grid");
  grid.innerHTML = "";
  const items = [
    ["Model", STATE.data.model.toUpperCase()],
    ["Layers", cfg.num_layers],
    ["Hidden", cfg.hidden_size],
    ["Attn heads", cfg.num_attention_heads],
    ["Experts", cfg.num_experts],
    ["Router top-k", cfg.moe_router_topk],
    ["MoE FFN", cfg.moe_ffn_hidden_size],
    ["Index top-k", cfg.index_topk],
    ["Vocab", cfg.vocab_size],
    ["MTP depths", cfg.mtp_num_layers || 0],
    ["Capture seq", STATE.data.capture.seq_length],
    ["cr=0 layers", cfg.cr_layer_counts["0"]],
    ["cr=4 layers", cfg.cr_layer_counts["4"]],
    ["cr=128 layers", cfg.cr_layer_counts["128"]],
  ];
  for (const [k, v] of items) {
    const kv = el("div", { class: "kv" });
    kv.append(el("b", {}, k), el("span", {}, String(v)));
    grid.append(kv);
  }
  // cr schedule strip
  const strip = $("#cr-schedule");
  strip.innerHTML = "";
  const row = el("div", { class: "cr-schedule" });
  for (const cr of cfg.compress_ratios) row.append(el("span", { class: `cr-cell cr-${cr}`, title: `cr=${cr}` }, "x"));
  strip.append(row);
  strip.append(el("div", { class: "cr-legend", html:
    '<span><i class="cr-0"></i>cr=0 dense+SWA</span><span><i class="cr-4"></i>cr=4 CSA</span><span><i class="cr-128"></i>cr=128 HCA</span>' }));
}

function breakdownBlock(title, fwdList, bwdList) {
  const c = STATE.controls, gpu = STATE.gpu;
  const block = el("div", { class: "bd-block" });
  block.append(el("h3", {}, title));
  const scroll = el("div", { class: "bd-scroll" });
  const table = el("table", { class: "bd" });

  const fwd = fwdList.map((r) => ({ r, t: rowScaledTime(r, gpu, c), phase: "F" }));
  const bwd = bwdList.map((r) => ({ r, t: rowScaledTime(r, gpu, c), phase: "B" })).reverse();
  const cols = [...fwd, ...bwd];
  const dividerIdx = fwd.length;

  const head = el("tr");
  head.append(el("th", { class: "rowlab" }, "Module"));
  cols.forEach((col, i) => {
    const th = el("th", { class: i === dividerIdx ? "divider" : "" });
    th.append(el("div", {}, col.r.module.replace(/^(attn|moe)\./, "")));
    th.append(el("div", { class: "phase-tag" }, col.phase));
    head.append(th);
  });
  table.append(head);

  const timeRow = el("tr");
  timeRow.append(el("td", { class: "rowlab" }, "Time µs"));
  cols.forEach((col, i) => {
    const cls = (col.r.class === "compute_bound" ? "cell-compute" : "cell-memory") + (i === dividerIdx ? " divider" : "");
    timeRow.append(el("td", { class: cls }, fmt(col.t, 0)));
  });
  table.append(timeRow);

  const tfRow = el("tr");
  tfRow.append(el("td", { class: "rowlab" }, "TFLOP/s (kernel)"));
  cols.forEach((col, i) => {
    const tf = rowTflops(col.r, col.t);
    tfRow.append(el("td", { class: i === dividerIdx ? "divider" : "" }, tf ? fmt(tf, 0) : "—"));
  });
  table.append(tfRow);

  scroll.append(table);
  block.append(scroll);
  return block;
}

function renderValidation(validation) {
  const host = $("#validation-panel");
  if (!host) return;
  host.innerHTML = "";
  if (!validation.errors.length && !validation.warnings.length) {
    host.hidden = true;
    return;
  }
  host.hidden = false;
  if (validation.errors.length) {
    const box = el("div", { class: "validation validation--error" });
    box.append(el("b", {}, "Errors"));
    const ul = el("ul");
    for (const msg of validation.errors) ul.append(el("li", {}, msg));
    box.append(ul);
    host.append(box);
  }
  if (validation.warnings.length) {
    const box = el("div", { class: "validation validation--warn" });
    box.append(el("b", {}, "Warnings"));
    const ul = el("ul");
    for (const msg of validation.warnings) ul.append(el("li", {}, msg));
    box.append(ul);
    host.append(box);
  }
}

function renderBreakdown() {
  $("#breakdown-gpu").textContent = `· ${STATE.gpu}`;
  const panel = $("#breakdown-panel");
  if (panel) panel.classList.toggle("is-muted", STATE.controls?.modelMode === "manual");
  const note = $("#breakdown-manual-note");
  if (note) note.hidden = STATE.controls?.modelMode !== "manual";
  const host = $("#breakdown-blocks");
  host.innerHTML = "";
  const d = STATE.data;
  host.append(breakdownBlock("Embedding", d.non_layer.embedding.forward, d.non_layer.embedding.backward));
  for (const cr of ["0", "4", "128"]) {
    const L = d.layers[cr];
    host.append(breakdownBlock(
      `Layer cr=${cr} — attention`, L.attention.forward, L.attention.backward));
    host.append(breakdownBlock(
      `Layer cr=${cr} — MoE`, L.moe.forward, L.moe.backward));
  }
  host.append(breakdownBlock("Output (norm + lm_head)", d.non_layer.output.forward, d.non_layer.output.backward));
  host.append(breakdownBlock("Loss", d.non_layer.loss.forward, d.non_layer.loss.backward));
}

function step(label, value, sub) {
  const s = el("div", { class: "step" });
  s.append(el("span", { class: "num" }, value));
  s.append(el("b", {}, label));
  if (sub) { s.append(document.createElement("br")); s.append(el("small", {}, sub)); }
  return s;
}

function renderResults(validation) {
  const c = STATE.controls, gpu = STATE.gpu, d = STATE.data;
  $("#results-gpu").textContent = `· ${gpu}`;
  const p = project(d, gpu, c, validation);

  const head = $("#results-headline");
  head.innerHTML = "";
  const steps = $("#results-steps");
  steps.innerHTML = "";
  if (!p) {
    head.append(el("div", { class: "metric metric--error" }, el("b", {}, "Projection blocked"), el("span", {}, "Invalid controls")));
    steps.append(step("Fix validation errors", "", "Projection is not recomputed while errors are present."));
    return;
  }
  const mk = (label, val, primary) => {
    const m = el("div", { class: "metric" + (primary ? " metric--primary" : "") });
    m.append(el("b", {}, label), el("span", {}, val));
    return m;
  };
  head.append(mk("tokens/s/GPU", fmtInt(p.tokSgpu), true));
  head.append(mk("TFLOP/s/GPU", fmt(p.tflopsGpu, 0)));
  head.append(mk("Iteration time", `${fmt(p.iterUs / 1000, 1)} ms`));
  head.append(mk("Pipeline bubble", `${fmt(p.bubbleFrac * 100, 1)} %`));

  const ltDesc = ["0", "4", "128"].map((cr) =>
    `cr${cr}: F ${fmt(p.lt[cr].fwd, 0)} / B ${fmt(p.lt[cr].bwd, 0)} µs`).join("  ·  ");
  const recomputeDesc = c.recompute === "first-n" ? `first ${c.recomputeLayers} layers/stage` : (c.recompute === "full" ? "recompute on" : "no recompute");
  const sourceTag = c.modelMode === "manual" ? "manual" : "trace";
  steps.append(step(`Per-layer fwd/bwd (µs, ${sourceTag}, ${recomputeDesc})`, "", ltDesc));
  steps.append(step("Critical PP stage (µs)", `F ${fmt(p.critF, 0)} + B ${fmt(p.critB, 0)}`,
    `max over ${c.pp} stages; layout ${p.layoutApplied ? "applied" : "balanced fallback"} (${p.layoutMessage}); counts=[${p.layoutCounts.join(", ")}]; per-device fwd=[${p.Df.map((x) => fmt(x / 1000, 1)).join(", ")}] ms`));
  if ((d.model_config.mtp_num_layers || 0) > 0) {
    steps.append(step("MTP on last stage", `F ${fmt(p.mtp.fwd, 0)} + B ${fmt(p.mtp.bwd, 0)} µs`,
      `${d.model_config.mtp_num_layers} depth(s), inner cr=${(d.model_config.mtp_compress_ratios || [0])[0]}, eh_proj estimated from output GEMM throughput`));
  }
  steps.append(step("GA (microbatches)", fmt(p.ga, 0), `GA = GBS ${c.gbs} / (DP ${p.dp} × MBS ${c.mbs})`));
  steps.append(step("Pipeline compute / iter", `${fmt(p.pipeUs / 1000, 2)} ms`,
    `(GA + (PP−1)/VPP) × (F+B)_crit ; bubble ${fmt(p.bubbleFrac * 100, 1)}%`));
  const optHint = p.measuredOptUs
    ? `; one-layer trace optimizer ref ${fmt(p.measuredOptUs / 1000, 2)} ms`
    : "";
  steps.append(step("Optimizer step / iter", `${fmt(p.optUs / 1000, 2)} ms`,
    `zero1: ${fmtInt(p.perRankParams / 1e6)}M optim params/rank (${fmtInt(p.localModelParams / 1e6)}M local model params) × ${c.bytesPerParam}B / HBM-BW / eff ${c.optEff}${optHint}`));
  steps.append(step("Iteration time", `${fmt(p.iterUs / 1000, 2)} ms`, "pipeline compute + optimizer (DP/PP comm assumed hidden)"));
  steps.append(step("World size", fmtInt(p.world), `PP ${c.pp} × TP ${c.tp} × CP ${c.cp} × DP ${p.dp} (derived); EP ${c.ep} ≤ DP`));
  steps.append(step("Tokens / iter", fmtInt(p.tokIter), `GBS ${c.gbs} × seq ${p.seq}`));
  steps.append(step("tokens/s/GPU", fmtInt(p.tokSgpu), `${fmtInt(p.tokS)} tok/s ÷ ${p.world} GPUs`));
  steps.append(step("TFLOP/s/GPU", fmt(p.tflopsGpu, 0), "V4 analytic model FLOPs (Megatron convention)"));

  // self-consistency hint
  const cap = d.capture;
  if (cap && cap.measured_iter_time_ms) {
    steps.append(step("Self-consistency (measured)", `${fmt(cap.measured_iter_time_ms, 1)} ms`,
      "set PP=1,VPP=1,DP=1,EP=8,MBS=1,GBS=2 to compare against capture"));
  }
}

function renderAll() {
  const validation = validateControls(STATE.data, STATE.controls, STATE.gpu);
  // keep the derived-DP readonly box in sync
  const w = STATE.controls;
  const roDp = document.getElementById("ctl-dp");
  if (roDp) roDp.value = Number.isFinite(validation.dp) ? validation.dp : "—";
  renderValidation(validation);
  renderModeSwitch();
  renderManualGrid();
  renderConfig();
  renderBreakdown();
  renderResults(validation);
}

// ---------------------------------------------------------------------------
// Bootstrap
// ---------------------------------------------------------------------------
async function init(model) {
  try {
    STATE.data = await loadModel(model);
    STATE.controls = defaultControls(STATE.data);
    $("#mock-badge").hidden = !(STATE.data.provenance && STATE.data.provenance.mock);
    $("#model-select").value = STATE.data.model;
    renderControls();
    renderAll();
  } catch (e) {
    const err = $("#error-state");
    err.hidden = false;
    err.textContent = String(e);
  }
}

globalThis.DSV4Projection = {
  defaultControls,
  derivedDP,
  expandLayoutRepeats,
  parsePipelineLayout,
  validateControls,
  effectiveLayerTimes,
  project,
};

if (typeof document !== "undefined") {
  $("#model-select").addEventListener("change", (e) => {
    const m = e.target.value;
    const u = new URL(location);
    u.searchParams.set("model", m);
    history.replaceState(null, "", u);
    init(m);
  });

  document.querySelectorAll(".tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      document.querySelectorAll(".tab").forEach((t) => t.classList.remove("is-active"));
      tab.classList.add("is-active");
      STATE.gpu = tab.dataset.gpu;
      if (STATE.controls?.modelMode === "manual") prefillManual(STATE.gpu);
      renderAll();
    });
  });

  document.querySelectorAll(".mode-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      const mode = tab.dataset.mode;
      if (!STATE.controls || STATE.controls.modelMode === mode) return;
      STATE.controls.modelMode = mode;
      if (mode === "manual") prefillManual(STATE.gpu);
      renderAll();
    });
  });

  init(modelFromQuery());
}
