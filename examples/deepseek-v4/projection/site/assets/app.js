"use strict";

// DeepSeek-V4 performance projection — static, no-build. Implements the math in
// design/04-projection-math.md. All breakdown times are microseconds (us) for
// one microbatch (seq from capture); the projection scales to a full model run.

const STATE = {
  data: null,
  gpu: "MI355X",
  controls: null,
  // iteration-timeline view (design/07): active level (1|2|3). Level 3
  // interleaving follows the VPP control directly (no separate toggle).
  tlLevel: 1,
  // layout: "tabs" (one level via buttons) or "stacked" (all three + linkage).
  tlView: "stacked",
  // Level 3 Gantt horizontal zoom (1 = fit-to-view, no scrollbar; >1 widens the
  // chart and reveals per-cell microbatch numbers, with horizontal scroll).
  tlZoom: 1,
  // cross-level selection for drill-down highlight. cr: focus a compression
  // ratio; devices: focus one or more PP ranks. Mutually exclusive.
  tlSel: { cr: null, devices: null },
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
  return {
    f0: null, b0: null, f4: null, b4: null, f128: null, b128: null,
    // non-layer + MTP overrides (per iteration, one device)
    emb_f: null, emb_b: null, out_f: null, out_b: null, loss_f: null, loss_b: null, mtp_f: null, mtp_b: null,
  };
}

const MANUAL_CR_KEYS = { "0": ["f0", "b0"], "4": ["f4", "b4"], "128": ["f128", "b128"] };
// Manually-overridable non-layer parts: key prefix, label, and the JSON section
// (mtp is synthesized, not a non_layer entry).
const MANUAL_NONLAYER = [
  { key: "emb", label: "embedding", which: "embedding" },
  { key: "out", label: "output", which: "output" },
  { key: "loss", label: "loss", which: "loss" },
  { key: "mtp", label: "MTP", which: "mtp" },
];
const MANUAL_NONLAYER_KEYS = MANUAL_NONLAYER.flatMap((n) => [`${n.key}_f`, `${n.key}_b`]);

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
const isNonNegativeNumber = (x) => Number.isFinite(x) && x >= 0;

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
  const lt = {};
  for (const cr of ["0", "4", "128"]) {
    const [fk, bk] = MANUAL_CR_KEYS[cr];
    const t = layerTimes(STATE.data, cr, gpu, c);
    lt[cr] = t;
    if (!isPositiveNumber(m[fk])) m[fk] = Math.round(t.fwd);
    if (!isPositiveNumber(m[bk])) m[bk] = Math.round(t.bwd);
  }
  // non-layer parts (embedding / output / loss). Skip seeding fields whose trace
  // value is 0 (e.g. output/loss backward) — leaving them unset shows the "0"
  // placeholder and falls back to trace, instead of pinning an explicit 0.
  const seed = (key, val) => { if (!isPositiveNumber(m[key]) && Math.round(val) > 0) m[key] = Math.round(val); };
  for (const which of ["embedding", "output", "loss"]) {
    const key = NONLAYER_KEY[which];
    seed(`${key}_f`, nonLayer(STATE.data, which, "forward", gpu, c));
    seed(`${key}_b`, nonLayer(STATE.data, which, "backward", gpu, c));
  }
  // MTP (only when the model uses it)
  if ((STATE.data.model_config.mtp_num_layers || 0) > 0) {
    const base = mtpTimes(STATE.data, gpu, c, lt);
    seed("mtp_f", base.fwd);
    seed("mtp_b", base.bwd);
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
  const m = c.man[gpu];
  const header = el("div", { class: "manual-grid__header" });
  header.append(el("p", { class: "muted manual-grid__hint" },
    `Per-layer and non-layer fwd/bwd (µs) for ${gpu}. Set values override the trace; placeholders show the current trace baseline. Embedding is on PP stage 0; output / loss / MTP are on the last PP stage.`));
  const resetBtn = el("button", { class: "manual-reset" }, "Restore defaults");
  resetBtn.addEventListener("click", () => {
    c.man[gpu] = emptyManual();
    prefillManual(gpu);
    renderAll();
  });
  header.append(resetBtn);
  host.append(header);

  // one fwd/bwd input row
  const makeRow = (labelNode, fk, bk, traceF, traceB) => {
    const row = el("div", { class: "manual-row" });
    row.append(labelNode);
    for (const [label, key, traceVal] of [["fwd", fk, traceF], ["bwd", bk, traceB]]) {
      const field = el("div", { class: "field" });
      field.append(el("span", {}, `${label} µs`));
      const input = el("input", {
        id: `man-${gpu}-${key}`, type: "number", step: "1", min: "0",
        placeholder: String(Math.round(traceVal || 0)),
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
    return row;
  };

  const rows = el("div", { class: "manual-rows" });
  for (const cr of ["0", "4", "128"]) {
    if (!(counts[cr] > 0)) continue;
    const [fk, bk] = MANUAL_CR_KEYS[cr];
    const t = layerTimes(STATE.data, cr, gpu, c);
    const lab = el("span", { class: `manual-row__lab cr-tag cr-${cr}` }, `cr=${cr} ×${counts[cr] || 0}`);
    rows.append(makeRow(lab, fk, bk, t.fwd, t.bwd));
  }
  host.append(rows);

  // non-layer + MTP rows
  host.append(el("p", { class: "muted manual-grid__subhead" }, "Non-layer (once per iteration)"));
  const nlRows = el("div", { class: "manual-rows" });
  const lt = {};
  for (const cr of ["0", "4", "128"]) lt[cr] = layerTimes(STATE.data, cr, gpu, c);
  for (const nl of MANUAL_NONLAYER) {
    if (nl.which === "mtp" && !((STATE.data.model_config.mtp_num_layers || 0) > 0)) continue;
    let tF, tB;
    if (nl.which === "mtp") {
      const base = mtpTimes(STATE.data, gpu, c, lt);
      tF = base.fwd; tB = base.bwd;
    } else {
      tF = nonLayer(STATE.data, nl.which, "forward", gpu, c);
      tB = nonLayer(STATE.data, nl.which, "backward", gpu, c);
    }
    const lab = el("span", { class: "manual-row__lab manual-row__lab--nl" }, nl.label);
    nlRows.append(makeRow(lab, `${nl.key}_f`, `${nl.key}_b`, tF, tB));
  }
  host.append(nlRows);
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
        if (v != null && v !== "" && !isNonNegativeNumber(Number(v))) {
          errors.push(`manual ${k} (cr=${cr}) must be a non-negative number (µs).`);
        }
      }
    }
    for (const k of MANUAL_NONLAYER_KEYS) {
      const v = man[k];
      if (v != null && v !== "" && !isNonNegativeNumber(Number(v))) {
        errors.push(`manual ${k} must be a non-negative number (µs).`);
      }
    }
    warnings.push(`Manual mode for ${gpu}: per-cr layer, non-layer (embedding/output/loss) and MTP fwd/bwd you set override the trace; unset fields fall back to trace.`);
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

// Non-layer time with manual override (embedding / output / loss). Falls back to
// the trace time when the field is unset or not in manual mode.
const NONLAYER_KEY = { embedding: "emb", output: "out", loss: "loss" };
function effectiveNonLayer(data, which, phase, gpu, c) {
  const trace = nonLayer(data, which, phase, gpu, c);
  if (c.modelMode !== "manual") return trace;
  const m = (c.man && c.man[gpu]) || {};
  const v = m[`${NONLAYER_KEY[which]}_${phase === "forward" ? "f" : "b"}`];
  return isPositiveNumber(v) ? v : trace;
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

// MTP time with manual override. Falls back to the analytic estimate.
function effectiveMtp(data, gpu, c, lt) {
  const base = mtpTimes(data, gpu, c, lt);
  if (c.modelMode !== "manual" || !((data.model_config.mtp_num_layers || 0) > 0)) return base;
  const m = (c.man && c.man[gpu]) || {};
  return {
    fwd: isPositiveNumber(m.mtp_f) ? m.mtp_f : base.fwd,
    bwd: isPositiveNumber(m.mtp_b) ? m.mtp_b : base.bwd,
    ehUs: base.ehUs,
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
  // Per virtual-chunk detail (k = 0..C-1): device = k % PP, vpp index = floor(k/PP).
  // Kept for the iteration-timeline view (design/07); does not affect Df/Db math.
  const chunks = [];
  for (let k = 0; k < C; k++) {
    chunks.push({ chunk: k, device: k % c.pp, vpp: Math.floor(k / c.pp), layers: [], fwd: 0, bwd: 0 });
  }
  const addLayer = (chunkIdx, i) => {
    const dev = chunkIdx % c.pp;
    const t = lt[String(crs[i])] || { fwd: 0, bwd: 0 };
    const recompute = recomputeLayer(c, stageOrdinals[dev]++);
    const effBwd = t.bwd + (recompute ? t.fwd : 0);
    Df[dev] += t.fwd;
    Db[dev] += effBwd;
    const ch = chunks[chunkIdx];
    ch.layers.push({ globalIdx: i, cr: crs[i], recompute, fwd: t.fwd, bwd: effBwd });
    ch.fwd += t.fwd;
    ch.bwd += effBwd;
  };
  if (layoutStages) {
    layoutStages.forEach((layers, chunk) => {
      for (const i of layers) addLayer(chunk, i);
    });
  } else {
    for (let i = 0; i < L; i++) addLayer(Math.floor(i / perChunk), i);
  }
  // non-layer parts on first / last device (manual-overridable)
  const embFwd = effectiveNonLayer(data, "embedding", "forward", gpu, c);
  const embBwd = effectiveNonLayer(data, "embedding", "backward", gpu, c);
  Df[0] += embFwd;
  Db[0] += embBwd;
  const last = c.pp - 1;
  const outFwd = effectiveNonLayer(data, "output", "forward", gpu, c) + effectiveNonLayer(data, "loss", "forward", gpu, c);
  const outBwd = effectiveNonLayer(data, "output", "backward", gpu, c) + effectiveNonLayer(data, "loss", "backward", gpu, c);
  Df[last] += outFwd;
  Db[last] += outBwd;
  const mtp = effectiveMtp(data, gpu, c, lt);
  Df[last] += mtp.fwd;
  Db[last] += mtp.bwd;

  const critF = Math.max(...Df), critB = Math.max(...Db);

  // Assemble per-device schedule detail (design/07). Non-layer parts are attached
  // to their owning device so the timeline can render them; Df/Db already include
  // them above.
  const critFdev = Df.indexOf(critF), critBdev = Db.indexOf(critB);
  const devices = [];
  for (let d = 0; d < c.pp; d++) {
    devices.push({
      device: d,
      chunks: chunks.filter((ch) => ch.device === d).sort((a, b) => a.vpp - b.vpp),
      Df: Df[d], Db: Db[d],
      hasEmb: d === 0, hasOut: d === last, hasMtp: d === last && mtp.fwd > 0,
      embFwd: d === 0 ? embFwd : 0, embBwd: d === 0 ? embBwd : 0,
      outFwd: d === last ? outFwd : 0, outBwd: d === last ? outBwd : 0,
      mtpFwd: d === last ? mtp.fwd : 0, mtpBwd: d === last ? mtp.bwd : 0,
      isCritF: d === critFdev, isCritB: d === critBdev,
    });
  }
  const schedule = { chunks, devices, C, critFdev, critBdev };

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
    schedule,
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
    const errs = (validation && validation.errors) || [];
    const summary = errs.length === 0 ? "Invalid controls" : `${errs.length} error${errs.length > 1 ? "s" : ""}`;
    head.append(el("div", { class: "metric metric--error" }, el("b", {}, "Projection blocked"), el("span", {}, summary)));
    if (errs.length) {
      for (const msg of errs) {
        const s = el("div", { class: "step step--error" });
        s.append(el("b", {}, "⚠ Validation error"));
        s.append(document.createElement("br"));
        s.append(el("small", {}, msg));
        steps.append(s);
      }
    } else {
      steps.append(step("Fix validation errors", "", "Projection is not recomputed while errors are present."));
    }
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

// ---------------------------------------------------------------------------
// Iteration timeline (design/07): 3-level composition view
// ---------------------------------------------------------------------------
const TL_CATS = ["attn", "mlp", "a2a", "misc"];
const TL_CAT_LABEL = { attn: "attn", mlp: "mlp", a2a: "a2a (dispatch+combine)", misc: "misc/unattrib" };
const CR_HEX = { "0": "#6b4a2d", "4": "#2d6b4a", "128": "#2d4a6b" };

// Map a module name to one of the Level-1 categories (design/07).
function moduleCategory(module) {
  const m = String(module || "");
  if (m.startsWith("attn.")) return "attn";
  if (m === "moe.dispatch" || m === "moe.combine") return "a2a";
  if (/(^|\.)misc$|unattrib/.test(m)) return "misc";
  return "mlp"; // moe.grouped_gemm / shared_expert / router and any other moe.*
}

// Per-cr forward/backward time split into categories, scaled to the active GPU.
// In manual mode the trace composition is rescaled so the bar total matches the
// effective (manual-overridden) per-layer time used by the projection.
function categoryBreakdown(data, cr, gpu, c) {
  const out = { forward: { attn: 0, mlp: 0, a2a: 0, misc: 0 }, backward: { attn: 0, mlp: 0, a2a: 0, misc: 0 } };
  const L = data.layers[cr];
  if (!L) return out;
  for (const bucket of [L.attention, L.moe]) {
    for (const phase of ["forward", "backward"]) {
      for (const r of bucket[phase]) out[phase][moduleCategory(r.module)] += rowScaledTime(r, gpu, c);
    }
  }
  if (c.modelMode === "manual") {
    const eff = effectiveLayerTimes(data, cr, gpu, c);
    for (const phase of ["forward", "backward"]) {
      const sum = TL_CATS.reduce((a, k) => a + out[phase][k], 0);
      const target = phase === "forward" ? eff.fwd : eff.bwd;
      if (sum > 0 && target > 0) for (const k of TL_CATS) out[phase][k] *= target / sum;
    }
  }
  return out;
}

// True when, in manual mode, the user has actually changed this cr's fwd/bwd away
// from the trace-derived baseline (comparison is rounded to µs so the prefilled
// baseline itself does not count as an edit).
function layerManualEdited(data, cr, gpu, c) {
  if (c.modelMode !== "manual") return false;
  const trace = layerTimes(data, cr, gpu, c);
  const eff = effectiveLayerTimes(data, cr, gpu, c);
  return Math.round(eff.fwd) !== Math.round(trace.fwd) || Math.round(eff.bwd) !== Math.round(trace.bwd);
}

// Small hex lighten/darken for VPP chunk shading (Level 3).
function shadeHex(hex, amt) {
  const n = parseInt(hex.slice(1), 16);
  const clamp = (x) => Math.max(0, Math.min(255, Math.round(x)));
  const r = clamp(((n >> 16) & 255) + amt), g = clamp(((n >> 8) & 255) + amt), b = clamp((n & 255) + amt);
  return `#${((r << 16) | (g << 8) | b).toString(16).padStart(6, "0")}`;
}

// Group PP devices with an identical ordered layer/recompute/non-layer signature
// so identical middle stages are drawn once (design/07 Level 2 dedup).
function dedupDevices(devices) {
  const groups = [];
  const byKey = new Map();
  for (const dev of devices) {
    const sig = JSON.stringify({
      layers: dev.chunks.map((ch) => ch.layers.map((l) => `${l.cr}${l.recompute ? "r" : ""}`)),
      emb: dev.hasEmb, out: dev.hasOut, mtp: dev.hasMtp,
    });
    if (byKey.has(sig)) byKey.get(sig).members.push(dev.device);
    else {
      const g = { rep: dev, members: [dev.device] };
      byKey.set(sig, g);
      groups.push(g);
    }
  }
  return groups;
}

// 1F1B (optionally interleaved) schedule simulator (design/07 Level 3). Returns
// per-device event lists with start/dur (µs) plus the drawn iteration length.
const TL_VIS_GA_CAP = 48;
function simulateSchedule(p, c, { interleaved }) {
  const PP = c.pp;
  const VPP = interleaved ? Math.max(1, c.vpp) : 1;
  const C = PP * VPP;
  const gaFull = Math.round(p.ga);
  const GA = Math.min(gaFull, TL_VIS_GA_CAP);
  const capped = gaFull > GA;

  // per-virtual-chunk fwd/bwd durations (k = 0..C-1, device = k % PP)
  const fdur = new Array(C).fill(0), bdur = new Array(C).fill(0);
  if (VPP === 1) {
    for (let d = 0; d < PP; d++) { fdur[d] = p.Df[d]; bdur[d] = p.Db[d]; }
  } else {
    for (const ch of p.schedule.chunks) { fdur[ch.chunk] = ch.fwd; bdur[ch.chunk] = ch.bwd; }
    // fold non-layer parts into first/last virtual chunk so the drawn length is
    // consistent with the per-device Df/Db (which include them).
    const dev0 = p.schedule.devices[0], devL = p.schedule.devices[PP - 1];
    fdur[0] += dev0.embFwd; bdur[0] += dev0.embBwd;
    fdur[C - 1] += devL.outFwd + devL.mtpFwd; bdur[C - 1] += devL.outBwd + devL.mtpBwd;
  }

  const total = GA * VPP;
  const group = PP * VPP;
  const fwdChunkMb = (i) => {
    const inGroup = i % group;
    const v = Math.floor(inGroup / PP);
    const m = Math.floor(i / group) * PP + (inGroup % PP);
    return { v, m };
  };

  const ops = []; // per device ordered op list
  for (let d = 0; d < PP; d++) {
    const warmup = Math.min(
      VPP === 1 ? PP - 1 - d : (PP - d - 1) * 2 + (VPP - 1) * PP,
      total,
    );
    const list = [];
    for (let i = 0; i < warmup; i++) {
      const { v, m } = fwdChunkMb(i);
      list.push({ kind: "F", m, k: v * PP + d });
    }
    let fptr = warmup, bptr = 0;
    const n1f1b = total - warmup;
    for (let i = 0; i < n1f1b; i++) {
      const f = fwdChunkMb(fptr++);
      list.push({ kind: "F", m: f.m, k: f.v * PP + d });
      const b = fwdChunkMb(bptr++);
      list.push({ kind: "B", m: b.m, k: (VPP - 1 - b.v) * PP + d });
    }
    for (let i = 0; i < warmup; i++) {
      const b = fwdChunkMb(bptr++);
      list.push({ kind: "B", m: b.m, k: (VPP - 1 - b.v) * PP + d });
    }
    ops.push(list);
  }

  // ASAP scheduling: forward flows k-1 -> k, backward flows k+1 -> k (p2p hidden).
  const endF = new Map(), endB = new Map();
  const kf = (m, k) => `${m}:${k}`;
  const ptr = new Array(PP).fill(0);
  const free = new Array(PP).fill(0);
  const events = Array.from({ length: PP }, () => []);
  let remaining = ops.reduce((a, l) => a + l.length, 0);
  let guard = remaining * 4 + 16;
  while (remaining > 0 && guard-- > 0) {
    let progressed = false;
    for (let d = 0; d < PP; d++) {
      if (ptr[d] >= ops[d].length) continue;
      const op = ops[d][ptr[d]];
      let dep = 0, ready = true;
      if (op.kind === "F") {
        if (op.k > 0) {
          const e = endF.get(kf(op.m, op.k - 1));
          if (e == null) ready = false; else dep = e;
        }
      } else {
        if (op.k < C - 1) {
          const e = endB.get(kf(op.m, op.k + 1));
          if (e == null) ready = false; else dep = e;
        } else {
          const e = endF.get(kf(op.m, op.k));
          if (e == null) ready = false; else dep = e;
        }
      }
      if (!ready) continue;
      const dur = op.kind === "F" ? fdur[op.k] : bdur[op.k];
      const start = Math.max(free[d], dep);
      const end = start + dur;
      events[d].push({ kind: op.kind, m: op.m, k: op.k, vpp: Math.floor(op.k / PP), start, dur });
      free[d] = end;
      (op.kind === "F" ? endF : endB).set(kf(op.m, op.k), end);
      ptr[d]++; remaining--; progressed = true;
    }
    if (!progressed) break; // dependency stall guard
  }
  const drawnUs = Math.max(0, ...free);
  return { events, drawnUs, GA, capped, PP, VPP, C };
}

// --- cross-level selection (drill-down linkage) ---
function tlSelActive() {
  return STATE.tlSel.cr != null || (STATE.tlSel.devices && STATE.tlSel.devices.length);
}
function tlSelectCr(cr) {
  STATE.tlSel = STATE.tlSel.cr === cr ? { cr: null, devices: null } : { cr, devices: null };
  renderTimeline();
}
function tlSelectDevices(devices) {
  const same = STATE.tlSel.devices && STATE.tlSel.devices.length === devices.length && STATE.tlSel.devices.every((d, i) => d === devices[i]);
  STATE.tlSel = same ? { cr: null, devices: null } : { cr: null, devices };
  renderTimeline();
}
function tlClearSel() {
  STATE.tlSel = { cr: null, devices: null };
  renderTimeline();
}
// Compression ratios "active" under the current selection (drives L1 highlight).
function tlActiveCrSet(p) {
  if (STATE.tlSel.cr != null) return new Set([String(STATE.tlSel.cr)]);
  if (STATE.tlSel.devices && STATE.tlSel.devices.length) {
    const s = new Set();
    for (const d of STATE.tlSel.devices) {
      for (const ch of p.schedule.devices[d].chunks) for (const l of ch.layers) s.add(String(l.cr));
    }
    return s;
  }
  return null;
}
const tlDevSet = () => (STATE.tlSel.devices && STATE.tlSel.devices.length ? new Set(STATE.tlSel.devices) : null);

function renderTimeline() {
  const host = $("#tl-body");
  if (!host) return;
  $("#timeline-gpu").textContent = `· ${STATE.gpu}`;
  const stacked = STATE.tlView === "stacked";
  document.querySelectorAll(".tl-view").forEach((t) => t.classList.toggle("is-active", (t.dataset.view === "stacked") === stacked));
  document.querySelectorAll(".tl-tab").forEach((t) => t.classList.toggle("is-active", Number(t.dataset.level) === STATE.tlLevel));
  const tabs = $("#tl-tabs");
  if (tabs) tabs.style.display = stacked ? "none" : "";
  host.innerHTML = "";
  const validation = validateControls(STATE.data, STATE.controls, STATE.gpu);
  const p = project(STATE.data, STATE.gpu, STATE.controls, validation);
  if (!p) {
    host.append(el("p", { class: "tl-warn" }, "Timeline unavailable: fix the validation errors above."));
    return;
  }

  // selection / linkage indicator: a fixed-position floating card appended to
  // <body> (NOT into #tl-body) so it is fully outside the timeline's flow and
  // toggling a selection never shifts the page layout at all.
  document.getElementById("tl-selbar-float")?.remove();
  if (tlSelActive()) {
    const bar = el("div", { id: "tl-selbar-float", class: "tl-selbar" });
    const txt = el("div", { class: "tl-selbar__txt" });
    const desc = STATE.tlSel.cr != null
      ? `Focused on cr=${STATE.tlSel.cr}`
      : `Focused on PP rank(s) ${STATE.tlSel.devices.join(", ")}`;
    txt.append(el("b", {}, desc));
    txt.append(el("span", { class: "muted" }, stacked ? " — highlighted across all levels" : " — highlighted; switch to Stacked to see all levels"));
    bar.append(txt);
    const btn = el("button", { class: "tl-clear" }, "Clear");
    btn.addEventListener("click", tlClearSel);
    bar.append(btn);
    document.body.append(bar);
  }

  if (stacked) {
    const section = (title, sub, fn) => {
      const sec = el("div", { class: "tl-section" });
      const h = el("h3", { class: "tl-section__title" }, title);
      if (sub) h.append(el("span", { class: "tl-section__sub" }, sub));
      sec.append(h);
      const body = el("div", {});
      fn(body);
      sec.append(body);
      host.append(sec);
    };
    section("Level 1 · single layer", "attn / mlp / a2a — click a cr to link", (b) => renderTimelineL1(b, p));
    section("Level 2 · pipeline ranks", "layer granularity — click a rank or a layer to link", (b) => renderTimelineL2(b, p));
    section("Level 3 · pipeline schedule", "1F1B / interleaved — click a device to link", (b) => renderTimelineL3(b, p));
  } else if (STATE.tlLevel === 1) renderTimelineL1(host, p);
  else if (STATE.tlLevel === 2) renderTimelineL2(host, p);
  else renderTimelineL3(host, p);
}

// --- gantt export (SVG / PNG) ---
function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const a = el("a", { href: url, download: filename });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 2000);
}
function serializeGantt(svg) {
  const clone = svg.cloneNode(true);
  const cs = getComputedStyle(document.documentElement);
  const bg = (cs.getPropertyValue("--bg").trim() || "#0f1218");
  let markup = new XMLSerializer().serializeToString(clone);
  for (const v of ["--panel-2", "--border", "--muted", "--text", "--bg"]) {
    markup = markup.split(`var(${v})`).join(cs.getPropertyValue(v).trim() || "#888");
  }
  if (!markup.includes("xmlns=")) markup = markup.replace("<svg", '<svg xmlns="http://www.w3.org/2000/svg"');
  const vb = svg.viewBox.baseVal;
  markup = markup.replace(/(<svg[^>]*>)/, `$1<rect x="0" y="0" width="${vb.width}" height="${vb.height}" fill="${bg}"/>`);
  return markup;
}
function exportGanttSvg(svg) {
  downloadBlob(new Blob([serializeGantt(svg)], { type: "image/svg+xml" }), `dsv4-pp-schedule-${STATE.gpu}.svg`);
}
function exportGanttPng(svg) {
  const markup = serializeGantt(svg);
  const vb = svg.viewBox.baseVal, scale = 2;
  const img = new Image();
  img.onload = () => {
    const canvas = el("canvas");
    canvas.width = vb.width * scale;
    canvas.height = vb.height * scale;
    const ctx = canvas.getContext("2d");
    ctx.scale(scale, scale);
    ctx.drawImage(img, 0, 0);
    canvas.toBlob((b) => downloadBlob(b, `dsv4-pp-schedule-${STATE.gpu}.png`));
  };
  img.src = "data:image/svg+xml;base64," + btoa(unescape(encodeURIComponent(markup)));
}

function catLegend() {
  return el("div", { class: "tl-legend", html:
    '<span><i class="tl-c-attn"></i>attn</span>' +
    '<span><i class="tl-c-mlp"></i>mlp (experts)</span>' +
    '<span><i class="tl-c-a2a"></i>a2a (dispatch+combine)</span>' +
    '<span><i class="tl-c-misc"></i>misc/unattributed</span>' });
}

function stackedTrack(phaseObj, maxTotal) {
  const track = el("div", { class: "tl-bar__track" });
  const total = TL_CATS.reduce((a, k) => a + phaseObj[k], 0);
  for (const cat of TL_CATS) {
    const t = phaseObj[cat];
    if (t <= 0) continue;
    const w = maxTotal > 0 ? (t / maxTotal) * 100 : 0;
    const seg = el("div", {
      class: `tl-seg tl-c-${cat}`,
      style: `width:${w}%`,
      title: `${TL_CAT_LABEL[cat]}: ${fmt(t, 0)} µs (${fmt(total > 0 ? (t / total) * 100 : 0, 0)}%)`,
    }, w > 6 ? cat : "");
    track.append(seg);
  }
  return { track, total };
}

// Single-segment bar used when a cr's layer time is manually set: no module split
// is possible, so show only the total fwd/bwd.
function aggregateTrack(timeUs, maxTotal, label) {
  const track = el("div", { class: "tl-bar__track" });
  const w = maxTotal > 0 ? (timeUs / maxTotal) * 100 : 0;
  track.append(el("div", {
    class: "tl-seg tl-c-manual", style: `width:${w}%`,
    title: `${label}: ${fmt(timeUs, 0)} µs (manual per-layer)`,
  }, w > 6 ? "manual" : ""));
  return { track, total: timeUs };
}

function renderTimelineL1(host, p) {
  const d = STATE.data, gpu = STATE.gpu, c = STATE.controls;
  host.append(el("p", { class: "tl-note" },
    "One representative layer per compression ratio (MoE is cr-independent, so only attention differs). Forward and backward split into attn / mlp / a2a; bar length is comparable across cr."));
  const crs = ["0", "4", "128"].filter((cr) => (d.model_config.cr_layer_counts?.[cr] || 0) > 0);
  const crSet = tlActiveCrSet(p);
  // Per-cr: whether the layer time is manually overridden (then no module split),
  // the module breakdown, and the fwd/bwd totals used for the shared scale.
  const info = {};
  let maxTotal = 0;
  let anyManual = false;
  for (const cr of crs) {
    const edited = layerManualEdited(d, cr, gpu, c);
    const bd = edited ? null : categoryBreakdown(d, cr, gpu, c);
    const eff = effectiveLayerTimes(d, cr, gpu, c);
    const fT = edited ? eff.fwd : TL_CATS.reduce((a, k) => a + bd.forward[k], 0);
    const bT = edited ? eff.bwd : TL_CATS.reduce((a, k) => a + bd.backward[k], 0);
    info[cr] = { edited, bd, fT, bT };
    anyManual = anyManual || edited;
    maxTotal = Math.max(maxTotal, fT, bT);
  }
  for (const cr of crs) {
    const { edited, bd, fT, bT } = info[cr];
    const isSel = STATE.tlSel.cr === cr;
    const dim = crSet && !crSet.has(cr);
    const row = el("div", { class: "tl-l1-row tl-clickable" + (isSel ? " is-sel" : "") + (dim ? " tl-dim" : "") });
    row.addEventListener("click", () => tlSelectCr(cr));
    const lab = el("div", { class: "tl-l1-lab" });
    lab.append(el("b", {}, `cr=${cr}`), el("small", {}, `×${d.model_config.cr_layer_counts[cr]} layers${edited ? " · manual" : ""}`));
    row.append(lab);
    const bars = el("div", { class: "tl-bars" });
    for (const [tag, phase, tot] of [["fwd", "forward", fT], ["bwd", "backward", bT]]) {
      const bar = el("div", { class: "tl-bar" });
      bar.append(el("span", { class: "tl-bar__tag" }, tag));
      const { track, total } = edited ? aggregateTrack(tot, maxTotal, `${tag} layer`) : stackedTrack(bd[phase], maxTotal);
      bar.append(track);
      bar.append(el("span", { class: "tl-bar__total" }, `${fmt(total, 0)} µs`));
      bars.append(bar);
    }
    row.append(bars);
    host.append(row);
  }
  if (anyManual) {
    host.append(el("div", { class: "tl-legend", html:
      '<span><i class="tl-c-manual"></i>manual per-layer total (no module split)</span>' }));
    host.append(el("p", { class: "tl-note" },
      "A cr marked “manual” uses a hand-entered whole-layer fwd/bwd time, so it cannot be split into attn / mlp / a2a — only the total is shown. Switch layer timing back to “Trace-derived”, or click “Restore defaults”, to see the per-module breakdown again."));
  }
  host.append(catLegend());
}

function renderTimelineL2(host, p) {
  const c = STATE.controls;
  host.append(el("p", { class: "tl-note" },
    "Each pipeline rank's layers (coloured by cr; hatched = recomputed). Identical ranks are drawn once. The critical stage (max fwd/bwd, sets the pipeline critical path) is outlined."));
  const groups = dedupDevices(p.schedule.devices);
  const maxTotal = Math.max(1, ...p.schedule.devices.map((dv) => dv.Df + dv.Db));
  const selDevices = tlDevSet();
  const selCr = STATE.tlSel.cr != null ? String(STATE.tlSel.cr) : null;
  for (const g of groups) {
    const dev = g.rep;
    const isCrit = dev.isCritF || dev.isCritB;
    const isSelGroup = selDevices && g.members.some((m) => selDevices.has(m));
    const dimRow = (selDevices && !isSelGroup) || (selCr && !g.members.some((m) => p.schedule.devices[m].chunks.some((ch) => ch.layers.some((l) => String(l.cr) === selCr))));
    const row = el("div", { class: "tl-l2-row tl-clickable" + (isCrit ? " is-critical" : "") + (isSelGroup ? " is-sel" : "") + (dimRow ? " tl-dim" : "") });
    row.addEventListener("click", () => tlSelectDevices(g.members));
    const lab = el("div", { class: "tl-l2-lab" });
    const members = g.members;
    const rangeTxt = members.length > 1 ? `PP ranks ${members[0]}–${members[members.length - 1]} (×${members.length})` : `PP rank ${members[0]}`;
    lab.append(el("b", {}, rangeTxt));
    lab.append(el("small", {}, `${dev.chunks.reduce((a, ch) => a + ch.layers.length, 0)} layers · ${dev.chunks.length} vpp chunk(s)${isCrit ? " · critical" : ""}`));
    row.append(lab);

    const total = dev.Df + dev.Db;
    const strip = el("div", { class: "tl-strip", style: `width:${(total / maxTotal) * 100}%` });
    if (dev.hasEmb) {
      const t = dev.embFwd + dev.embBwd;
      strip.append(el("div", { class: "tl-cell tl-cell--emb", style: `width:${(t / total) * 100}%`, title: `embedding F ${fmt(dev.embFwd, 0)} / B ${fmt(dev.embBwd, 0)} µs` }));
    }
    dev.chunks.forEach((ch, ci) => {
      if (ci > 0 || dev.hasEmb) strip.append(el("div", { class: "tl-chunk-gap" }));
      for (const l of ch.layers) {
        const t = l.fwd + l.bwd;
        const cellSel = selCr && String(l.cr) === selCr;
        const cellDim = selCr && String(l.cr) !== selCr;
        const cell = el("div", {
          class: "tl-cell tl-clickable" + (l.recompute ? " tl-cell--recompute" : "") + (cellSel ? " tl-cell--sel" : "") + (cellDim ? " tl-dim" : ""),
          style: `width:${(t / total) * 100}%; background:${CR_HEX[String(l.cr)] || "#555"}`,
          title: `layer #${l.globalIdx} cr=${l.cr}${l.recompute ? " (recompute)" : ""} · F ${fmt(l.fwd, 0)} / B ${fmt(l.bwd, 0)} µs`,
        });
        cell.addEventListener("click", (e) => { e.stopPropagation(); tlSelectCr(String(l.cr)); });
        strip.append(cell);
      }
    });
    if (dev.hasOut) {
      const t = dev.outFwd + dev.outBwd + dev.mtpFwd + dev.mtpBwd;
      strip.append(el("div", { class: "tl-chunk-gap" }));
      strip.append(el("div", { class: "tl-cell tl-cell--out", style: `width:${(t / total) * 100}%`, title: `output/loss${dev.hasMtp ? "+MTP" : ""} F ${fmt(dev.outFwd + dev.mtpFwd, 0)} / B ${fmt(dev.outBwd + dev.mtpBwd, 0)} µs` }));
    }
    row.append(strip);
    row.append(el("div", { class: "tl-l2-meta" }, `Df ${fmt(dev.Df / 1000, 2)} / Db ${fmt(dev.Db / 1000, 2)} ms`));
    host.append(row);
  }
  host.append(el("div", { class: "cr-legend", html:
    '<span><i class="cr-0"></i>cr=0</span><span><i class="cr-4"></i>cr=4</span><span><i class="cr-128"></i>cr=128</span>' +
    '<span><i class="tl-c-emb"></i>embedding</span><span><i class="tl-c-out"></i>output/loss/MTP</span>' }));
}

function renderTimelineL3(host, p) {
  const c = STATE.controls;
  // Interleaving is driven directly by the VPP control (no separate toggle):
  // VPP=1 is plain 1F1B, VPP>1 is interleaved 1F1B.
  const interleaved = c.vpp > 1;

  const controls = el("div", { class: "tl-controls" });
  controls.append(el("span", { class: "mode-switch__label" }, `Schedule · 1F1B${interleaved ? ` (interleaved, VPP=${c.vpp})` : ""}`));

  // Zoom slider: 1x fits the whole schedule in view (no scrollbar); zooming in
  // widens the chart so the per-cell microbatch numbers become readable.
  const zoomWrap = el("span", { class: "tl-zoom-wrap" });
  zoomWrap.append(el("span", { class: "mode-switch__label" }, "Zoom"));
  const zoom = el("input", { type: "range", min: "1", max: "10", step: "0.5", value: String(STATE.tlZoom), class: "tl-zoom" });
  const zlab = el("span", { class: "tl-zoom-val" }, `${STATE.tlZoom}×`);
  zoom.addEventListener("input", () => { zlab.textContent = `${zoom.value}×`; });
  zoom.addEventListener("change", () => { STATE.tlZoom = Number(zoom.value); renderTimeline(); });
  zoomWrap.append(zoom, zlab);
  controls.append(zoomWrap);
  host.append(controls);
  if (!interleaved) {
    host.append(el("p", { class: "tl-note" }, "VPP=1 → plain 1F1B. Set VPP>1 in the controls to interleave the schedule and shrink the pipeline bubble."));
  }

  const sim = simulateSchedule(p, c, { interleaved });
  const PP = sim.PP;
  const rowH = 30, gap = 6, padL = 54, padT = 8, padB = 26;
  // Zoom widens the coordinate system itself (more px per µs; font size stays
  // fixed so cells become readable) instead of CSS-scaling the SVG. At 1x the
  // chart is sized to the actual right-content width so it fits with no
  // scrollbar; >1x overflows and scrolls horizontally, undistorted.
  const avail = Math.max(600, (($("#tl-body")?.clientWidth) || 1100) - 16);
  const plotW = Math.round((avail - padL) * STATE.tlZoom);
  const width = padL + plotW + 8;
  const scale = sim.drawnUs > 0 ? plotW / sim.drawnUs : 0;
  const height = padT + PP * (rowH + gap) + padB;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.setAttribute("class", "tl-gantt");
  svg.setAttribute("viewBox", `0 0 ${width} ${height}`);
  svg.setAttribute("width", String(width));
  svg.setAttribute("height", String(height));
  const mkEl = (tag, attrs, text) => {
    const n = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [k, v] of Object.entries(attrs)) n.setAttribute(k, v);
    if (text != null) n.textContent = text;
    return n;
  };

  const selDevices = tlDevSet();
  for (let d = 0; d < PP; d++) {
    const y = padT + d * (rowH + gap);
    const isSel = selDevices ? selDevices.has(d) : false;
    const g = mkEl("g", { class: "tl-devrow" });
    if (selDevices) g.setAttribute("opacity", isSel ? "1" : "0.3");
    g.append(mkEl("text", { x: padL - 8, y: y + rowH / 2 + 4, "text-anchor": "end", fill: "#e6eaf2", "font-size": "11" }, `dev ${d}`));
    const bgRect = mkEl("rect", {
      x: padL, y, width: plotW, height: rowH, rx: "3",
      fill: isSel ? "var(--panel)" : "var(--panel-2)",
      stroke: isSel ? "#36c08f" : "var(--border)", "stroke-width": isSel ? "2" : "1",
      style: "cursor:pointer",
    });
    bgRect.addEventListener("click", () => tlSelectDevices([d]));
    g.append(bgRect);
    for (const ev of sim.events[d]) {
      const x = padL + ev.start * scale;
      const w = Math.max(1, ev.dur * scale);
      const base = ev.kind === "F" ? "#4f8cff" : "#36c08f";
      const fill = shadeHex(base, ev.vpp * -26);
      const rect = mkEl("rect", {
        x, y: y + 2, width: w, height: rowH - 4, rx: "2",
        fill, class: ev.kind === "F" ? "tl-fwd" : "tl-bwd",
      });
      rect.append(mkEl("title", {}, `${ev.kind === "F" ? "Forward" : "Backward"} · microbatch ${ev.m}${sim.VPP > 1 ? ` · vpp chunk ${ev.vpp}` : ""} · compute ${fmt(ev.dur, 0)} µs · starts @ ${fmt(ev.start / 1000, 2)} ms (from iter start)`));
      g.append(rect);
      if (w >= 8) g.append(mkEl("text", { x: x + w / 2, y: y + rowH / 2 + 3, "text-anchor": "middle", fill: "#000000", "font-size": "9" }, String(ev.m)));
    }
    svg.append(g);
  }
  // time axis ticks
  const yb = padT + PP * (rowH + gap);
  for (let i = 0; i <= 4; i++) {
    const tx = padL + (plotW * i) / 4;
    svg.append(mkEl("text", { x: tx, y: yb + 16, "text-anchor": "middle", fill: "#8d97a8", "font-size": "10" }, `${fmt((sim.drawnUs * i) / 4 / 1000, 1)} ms`));
  }

  const toolbar = el("div", { class: "tl-export" });
  const svgBtn = el("button", { class: "mode-tab" }, "Export SVG");
  const pngBtn = el("button", { class: "mode-tab" }, "Export PNG");
  svgBtn.addEventListener("click", () => exportGanttSvg(svg));
  pngBtn.addEventListener("click", () => exportGanttPng(svg));
  toolbar.append(svgBtn, pngBtn);
  host.append(toolbar);

  const wrap = el("div", { class: "tl-gantt-wrap" });
  wrap.append(svg);
  host.append(wrap);

  // legend + axis/tooltip explanation + self-check vs analytic pipe time
  host.append(el("div", { class: "tl-legend", html:
    '<span><i class="tl-c-attn"></i>forward</span><span><i class="tl-c-mlp"></i>backward</span>' +
    (sim.VPP > 1 ? '<span>lighter→darker = VPP chunk 0→' + (sim.VPP - 1) + '</span>' : '') +
    '<span>gaps = pipeline bubble (idle)</span>' }));
  host.append(el("p", { class: "tl-note tl-axis-note" },
    "X-axis = wall-clock time measured from the start of the pipeline (ms). Each cell is one microbatch's forward or backward on that device; the number inside is the microbatch index. Hover a cell to read its compute duration in µs (the “… µs” before @) and its start time in ms from the iteration start (the “… ms” after @)."));

  const analyticPipeUs = (p.ga + (c.pp - 1) / sim.VPP) * (p.critF + p.critB); // pre-calibFactor, matches the drawn schedule's VPP
  const diff = analyticPipeUs > 0 ? Math.abs(sim.drawnUs - analyticPipeUs) / analyticPipeUs : 0;
  const summary = el("div", { class: "tl-summary" });
  const gaTxt = sim.capped ? `${sim.GA} of ${Math.round(p.ga)} microbatches (capped for display)` : `${sim.GA} microbatches`;
  summary.append(el("div", {}, `Drawn iteration (pipeline compute): ${fmt(sim.drawnUs / 1000, 2)} ms over ${gaTxt}; PP=${c.pp}, VPP=${interleaved ? c.vpp : 1}. Analytic bubble fraction ${fmt(p.bubbleFrac * 100, 1)}%.`));
  if (!sim.capped) {
    const cls = diff > 0.08 ? "tl-warn" : "";
    summary.append(el("div", { class: cls },
      `Self-check vs analytic (GA + (PP−1)/VPP)·(F+B)_crit = ${fmt(analyticPipeUs / 1000, 2)} ms → ${fmt(diff * 100, 1)}% ${diff > 0.08 ? "difference (imbalanced stages; analytic uses per-device max)" : "match"}. Official iteration time uses the analytic value × calibFactor.`));
  }
  host.append(summary);
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
  renderTimeline();
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
  moduleCategory,
  categoryBreakdown,
  dedupDevices,
  simulateSchedule,
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

  document.querySelectorAll(".tl-tab").forEach((tab) => {
    tab.addEventListener("click", () => {
      STATE.tlLevel = Number(tab.dataset.level);
      renderTimeline();
    });
  });

  document.querySelectorAll(".tl-view").forEach((tab) => {
    tab.addEventListener("click", () => {
      STATE.tlView = tab.dataset.view;
      renderTimeline();
    });
  });

  // Re-fit the schedule Gantt to the content width when the window is resized.
  let _tlResizeTimer;
  window.addEventListener("resize", () => {
    clearTimeout(_tlResizeTimer);
    _tlResizeTimer = setTimeout(() => { if (STATE.data) renderTimeline(); }, 150);
  });

  init(modelFromQuery());
}
