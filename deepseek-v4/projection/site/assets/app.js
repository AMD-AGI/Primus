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
  return {
    pp: isPro ? 8 : 4, vpp: 1, ep: 8, dp: isPro ? 32 : 16, tp: 1, cp: 1,
    mbs: 1, gbs: 512,
    recompute: "full",
    optEff: 0.7, computeEff: 1.0, bytesPerParam: data.optimizer?.bytes_per_param || 18,
    peak355: m355.peak_tflops_bf16, bw355: m355.hbm_bandwidth_gbps,
    peak455: m455.peak_tflops_bf16, bw455: m455.hbm_bandwidth_gbps,
  };
}

const CONTROL_DEFS = [
  ["world", "World size (derived)", "ro"],
  ["pp", "PP (pipeline)", "int"],
  ["vpp", "VPP (interleave)", "int"],
  ["ep", "EP (expert)", "int"],
  ["dp", "DP (data)", "int"],
  ["tp", "TP (tensor)", "int"],
  ["cp", "CP (context)", "int"],
  ["mbs", "Micro batch size", "int"],
  ["gbs", "Global batch size", "int"],
  ["recompute", "Recompute", "sel"],
  ["bytesPerParam", "Optim bytes/param", "int"],
  ["optEff", "Optim efficiency", "f"],
  ["computeEff", "MI455 compute eff", "f"],
  ["peak355", "MI355 peak TFLOPs", "int"],
  ["bw355", "MI355 HBM GB/s", "int"],
  ["peak455", "MI455 peak TFLOPs", "int"],
  ["bw455", "MI455 HBM GB/s", "int"],
];

function renderControls() {
  const grid = $("#controls-grid");
  grid.innerHTML = "";
  const c = STATE.controls;
  const world = c.pp * c.tp * c.cp * c.dp;
  for (const [key, label, kind] of CONTROL_DEFS) {
    const field = el("div", { class: "field" });
    field.append(el("span", {}, label));
    let input;
    if (kind === "sel") {
      input = el("select", { id: `ctl-${key}` });
      for (const opt of ["none", "full"]) {
        const o = el("option", { value: opt }, opt);
        if (c[key] === opt) o.selected = true;
        input.append(o);
      }
    } else if (kind === "ro") {
      input = el("input", { id: `ctl-${key}`, value: world, disabled: "true" });
    } else {
      input = el("input", { id: `ctl-${key}`, type: "number", value: c[key], step: kind === "f" ? "0.05" : "1" });
    }
    if (kind !== "ro") {
      input.addEventListener("change", () => {
        const v = kind === "sel" ? input.value : Number(input.value);
        c[key] = v;
        renderAll();
      });
    }
    field.append(input);
    grid.append(field);
  }
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
  if (c.recompute === "full") {
    bwd += fwd;        // replay one forward (Step 1)
    bFlops += fFlops;
  }
  return { fwd, bwd, fFlops, bFlops };
}

function nonLayer(data, which, phase, gpu, c) {
  const bd = data.non_layer[which];
  return bd ? sumTime(bd[phase], gpu, c) : 0;
}
function nonLayerFlops(data, which, phase) {
  const bd = data.non_layer[which];
  return bd ? sumFlops(bd[phase]) : 0;
}

// ---------------------------------------------------------------------------
// Param estimate (Step 4)
// ---------------------------------------------------------------------------
function estimateParams(cfg) {
  const h = cfg.hidden_size, exp = cfg.num_experts, mff = cfg.moe_ffn_hidden_size;
  const sff = cfg.moe_shared_expert_intermediate_size || mff, V = cfg.vocab_size, L = cfg.num_layers;
  const attn = 4 * h * h;
  const experts = exp * 3 * h * mff;
  const shared = 3 * h * sff;
  const perLayer = attn + experts + shared;
  return L * perLayer + 2 * V * h;
}

// ---------------------------------------------------------------------------
// Pipeline mapping + projection (Steps 2-5)
// ---------------------------------------------------------------------------
function project(data, gpu, c) {
  const cfg = data.model_config;
  const crs = cfg.compress_ratios;
  const L = crs.length;
  const lt = {};
  for (const cr of ["0", "4", "128"]) lt[cr] = layerTimes(data, cr, gpu, c);

  // Step 2: assign layers to PP*VPP chunks -> devices
  const C = c.pp * c.vpp;
  const perChunk = Math.ceil(L / C);
  const Df = new Array(c.pp).fill(0), Db = new Array(c.pp).fill(0);
  for (let i = 0; i < L; i++) {
    const chunk = Math.floor(i / perChunk);
    const dev = chunk % c.pp;
    const t = lt[String(crs[i])] || { fwd: 0, bwd: 0 };
    Df[dev] += t.fwd;
    Db[dev] += t.bwd;
  }
  // non-layer parts on first / last device
  Df[0] += nonLayer(data, "embedding", "forward", gpu, c);
  Db[0] += nonLayer(data, "embedding", "backward", gpu, c);
  const last = c.pp - 1;
  Df[last] += nonLayer(data, "output", "forward", gpu, c) + nonLayer(data, "loss", "forward", gpu, c);
  Db[last] += nonLayer(data, "output", "backward", gpu, c) + nonLayer(data, "loss", "backward", gpu, c);

  const critF = Math.max(...Df), critB = Math.max(...Db);

  // Step 3: pipeline compute time (us) ; GA = gbs/(dp*mbs)
  const ga = c.gbs / (c.dp * c.mbs);
  const pipeUs = (ga + (c.pp - 1) / c.vpp) * (critF + critB);
  const bubbleFrac = (c.pp - 1) / c.vpp / (ga + (c.pp - 1) / c.vpp);

  // Step 4: optimizer (memory-bound, zero1-sharded over world)
  const world = c.pp * c.tp * c.cp * c.dp;
  const totalParams = estimateParams(cfg);
  const perRankParams = totalParams / world;
  const bw = (gpu === "MI355X" ? c.bw355 : c.bw455) * 1e9; // bytes/s
  const optTimeS = (perRankParams * c.bytesPerParam * 2) / bw / c.optEff;
  const optUs = optTimeS * 1e6;

  // Step 5: totals
  const iterUs = pipeUs + optUs;
  const iterS = iterUs * 1e-6;
  const seq = data.capture.seq_length;
  const tokIter = c.gbs * seq;
  const tokS = tokIter / iterS;
  const tokSgpu = tokS / world;

  // FLOPs/iter (matmul convention)
  let fMb = 0;
  const counts = cfg.cr_layer_counts;
  for (const cr of ["0", "4", "128"]) fMb += (counts[cr] || 0) * (lt[cr].fFlops + lt[cr].bFlops);
  fMb += nonLayerFlops(data, "embedding", "forward") + nonLayerFlops(data, "embedding", "backward");
  fMb += nonLayerFlops(data, "output", "forward") + nonLayerFlops(data, "output", "backward");
  const flopsIter = fMb * ga * c.dp;
  const tflopsGpu = flopsIter / iterS / world / 1e12;

  return {
    lt, Df, Db, critF, critB, ga, pipeUs, bubbleFrac, world, totalParams,
    perRankParams, optUs, iterUs, tokIter, tokS, tokSgpu, flopsIter, tflopsGpu, seq,
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
  tfRow.append(el("td", { class: "rowlab" }, "TFLOP/s"));
  cols.forEach((col, i) => {
    const tf = rowTflops(col.r, col.t);
    tfRow.append(el("td", { class: i === dividerIdx ? "divider" : "" }, tf ? fmt(tf, 0) : "—"));
  });
  table.append(tfRow);

  scroll.append(table);
  block.append(scroll);
  return block;
}

function renderBreakdown() {
  $("#breakdown-gpu").textContent = `· ${STATE.gpu}`;
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

function renderResults() {
  const c = STATE.controls, gpu = STATE.gpu, d = STATE.data;
  $("#results-gpu").textContent = `· ${gpu}`;
  const p = project(d, gpu, c);

  const head = $("#results-headline");
  head.innerHTML = "";
  const mk = (label, val, primary) => {
    const m = el("div", { class: "metric" + (primary ? " metric--primary" : "") });
    m.append(el("b", {}, label), el("span", {}, val));
    return m;
  };
  head.append(mk("tokens/s/GPU", fmtInt(p.tokSgpu), true));
  head.append(mk("TFLOP/s/GPU", fmt(p.tflopsGpu, 0)));
  head.append(mk("Iteration time", `${fmt(p.iterUs / 1000, 1)} ms`));
  head.append(mk("Pipeline bubble", `${fmt(p.bubbleFrac * 100, 1)} %`));

  const steps = $("#results-steps");
  steps.innerHTML = "";
  const ltDesc = ["0", "4", "128"].map((cr) =>
    `cr${cr}: F ${fmt(p.lt[cr].fwd, 0)} / B ${fmt(p.lt[cr].bwd, 0)} µs`).join("  ·  ");
  steps.append(step("Per-layer fwd/bwd (µs, " + (c.recompute === "full" ? "recompute on" : "no recompute") + ")", "", ltDesc));
  steps.append(step("Critical PP stage (µs)", `F ${fmt(p.critF, 0)} + B ${fmt(p.critB, 0)}`,
    `max over ${c.pp} stages; per-device fwd=[${p.Df.map((x) => fmt(x / 1000, 1)).join(", ")}] ms`));
  steps.append(step("GA (microbatches)", fmtInt(p.ga), `GA = GBS ${c.gbs} / (DP ${c.dp} × MBS ${c.mbs})`));
  steps.append(step("Pipeline compute / iter", `${fmt(p.pipeUs / 1000, 2)} ms`,
    `(GA + (PP−1)/VPP) × (F+B)_crit ; bubble ${fmt(p.bubbleFrac * 100, 1)}%`));
  steps.append(step("Optimizer step / iter", `${fmt(p.optUs / 1000, 2)} ms`,
    `zero1: ${fmtInt(p.perRankParams / 1e6)}M params/rank × ${c.bytesPerParam}B ×2 / HBM-BW / eff ${c.optEff}`));
  steps.append(step("Iteration time", `${fmt(p.iterUs / 1000, 2)} ms`, "pipeline compute + optimizer (DP/PP comm assumed hidden)"));
  steps.append(step("World size", fmtInt(p.world), `PP ${c.pp} × TP ${c.tp} × CP ${c.cp} × DP ${c.dp}`));
  steps.append(step("Tokens / iter", fmtInt(p.tokIter), `GBS ${c.gbs} × seq ${p.seq}`));
  steps.append(step("tokens/s/GPU", fmtInt(p.tokSgpu), `${fmtInt(p.tokS)} tok/s ÷ ${p.world} GPUs`));
  steps.append(step("TFLOP/s/GPU", fmt(p.tflopsGpu, 0), "matmul FLOPs (gemm+grouped_gemm+attn) only"));

  // self-consistency hint
  const cap = d.capture;
  if (cap && cap.measured_iter_time_ms) {
    steps.append(step("Self-consistency (measured)", `${fmt(cap.measured_iter_time_ms, 1)} ms`,
      "set PP=1,VPP=1,DP=1,EP=8,MBS=1,GBS=2 to compare against capture"));
  }
}

function renderAll() {
  // keep world-size readonly box in sync
  const w = STATE.controls;
  const ro = document.getElementById("ctl-world");
  if (ro) ro.value = w.pp * w.tp * w.cp * w.dp;
  renderConfig();
  renderBreakdown();
  renderResults();
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
    renderBreakdown();
    renderResults();
  });
});

init(modelFromQuery());
