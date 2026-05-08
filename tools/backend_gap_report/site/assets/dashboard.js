"use strict";

const BACKEND_GAP_DATA_URL = "./dashboard-data/index.json";
const WEEKLY_DATA_URL = "./weekly-reports-data/index.json";

const state = {
  backendGap: {
    reports: [],
    filteredReports: [],
  },
  weekly: {
    reports: [],
    filteredReports: [],
  },
};

const el = (id) => document.getElementById(id);

const elements = {
  heroChips: el("hero-chips"),

  weekly: {
    statRow: el("weekly-stat-row"),
    featured: el("weekly-featured"),
    featuredTitle: el("weekly-featured-title"),
    featuredWindow: el("weekly-featured-window"),
    featuredPrs: el("weekly-featured-prs"),
    featuredCategories: el("weekly-featured-categories"),
    featuredRecommendations: el("weekly-featured-recommendations"),
    featuredFindings: el("weekly-featured-findings"),
    filterRow: el("weekly-filter-row"),
    recommendationFilter: el("weekly-recommendation-filter"),
    searchFilter: el("weekly-search-filter"),
    resultCount: el("weekly-result-count"),
    loadingState: el("weekly-loading-state"),
    errorState: el("weekly-error-state"),
    emptyState: el("weekly-empty-state"),
    list: el("weekly-list"),
  },

  backendGap: {
    statRow: el("gap-stat-row"),
    filterRow: el("gap-filter-row"),
    totalReports: null,
    backendFilter: el("backend-filter"),
    statusFilter: el("status-filter"),
    searchFilter: el("search-filter"),
    resultCount: el("gap-result-count"),
    loadingState: el("loading-state"),
    errorState: el("error-state"),
    emptyState: el("empty-state"),
    reportList: el("report-list"),
  },
};

// ---------- Utilities ----------

function formatNumber(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return new Intl.NumberFormat("en-US").format(value);
}

function formatDate(value) {
  if (!value) return "-";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toISOString().slice(0, 10);
}

function formatDateTime(value) {
  if (!value) return "-";
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return value;
  return d.toUTCString().replace("GMT", "UTC");
}

function recommendationClass(value) {
  if (!value || typeof value !== "string") return "pill--neutral";
  const lowered = value.toLowerCase();
  if (lowered.includes("urgent")) return "pill--urgent";
  if (lowered.includes("plan")) return "pill--plan";
  if (lowered.includes("monitor") || lowered.includes("ok") || lowered.includes("no drift"))
    return "pill--monitor";
  return "pill--neutral";
}

function createPill(text, className) {
  const span = document.createElement("span");
  span.className = `pill ${className}`;
  span.textContent = text;
  return span;
}

function statTile(label, value, sub) {
  const tile = document.createElement("div");
  tile.className = "stat-tile";

  const labelEl = document.createElement("span");
  labelEl.className = "stat-tile__label";
  labelEl.textContent = label;

  const valueEl = document.createElement("span");
  valueEl.className = "stat-tile__value";
  valueEl.textContent = value;

  tile.append(labelEl, valueEl);

  if (sub) {
    const subEl = document.createElement("span");
    subEl.className = "stat-tile__sub";
    subEl.textContent = sub;
    tile.append(subEl);
  }
  return tile;
}

function heroChip(label, value) {
  const chip = document.createElement("span");
  chip.className = "hero-chip";
  const labelEl = document.createElement("span");
  labelEl.textContent = label;
  const valueEl = document.createElement("strong");
  valueEl.textContent = value;
  chip.append(labelEl, valueEl);
  return chip;
}

function recommendationCounts(summary) {
  const counts = summary && summary.recommendation_counts ? summary.recommendation_counts : {};
  return {
    urgent: counts["urgent sync"] || 0,
    plan: counts["plan sync"] || 0,
    monitor: counts.monitor || 0,
  };
}

// ---------- Weekly Reports ----------

function renderHeroChips(weeklyPayload, gapPayload) {
  elements.heroChips.innerHTML = "";
  const weeklySummary = weeklyPayload && weeklyPayload.summary ? weeklyPayload.summary : {};
  const gapSummary = gapPayload && gapPayload.summary ? gapPayload.summary : {};
  const recCounts = recommendationCounts(weeklySummary);

  if (weeklySummary.latest_report_id) {
    elements.heroChips.append(heroChip("Latest week", weeklySummary.latest_report_id));
  }
  if (weeklySummary.latest_merged_pr_count !== undefined) {
    elements.heroChips.append(
      heroChip("Merged PRs in latest week", formatNumber(weeklySummary.latest_merged_pr_count))
    );
  }
  if (weeklySummary.total_reports !== undefined) {
    elements.heroChips.append(
      heroChip("Weekly reports", formatNumber(weeklySummary.total_reports))
    );
  }
  if (recCounts.urgent) {
    elements.heroChips.append(heroChip("Urgent sync", formatNumber(recCounts.urgent)));
  }
  if (recCounts.plan) {
    elements.heroChips.append(heroChip("Plan sync", formatNumber(recCounts.plan)));
  }
  if (gapSummary.total_reports !== undefined) {
    elements.heroChips.append(heroChip("Backend reports", formatNumber(gapSummary.total_reports)));
  }
}

function renderWeeklyStatRow(payload) {
  elements.weekly.statRow.innerHTML = "";
  const summary = payload && payload.summary ? payload.summary : {};
  const recCounts = recommendationCounts(summary);
  elements.weekly.statRow.append(
    statTile("Total weekly reports", formatNumber(summary.total_reports || 0))
  );
  elements.weekly.statRow.append(
    statTile("Latest week", summary.latest_report_id || "-", formatDate(summary.latest_generated_at))
  );
  elements.weekly.statRow.append(
    statTile("Merged PRs (latest)", formatNumber(summary.latest_merged_pr_count || 0))
  );
  elements.weekly.statRow.append(
    statTile("Urgent sync", formatNumber(recCounts.urgent))
  );
  elements.weekly.statRow.append(
    statTile("Plan sync", formatNumber(recCounts.plan))
  );
  elements.weekly.statRow.append(
    statTile("Monitor", formatNumber(recCounts.monitor))
  );
}

function renderWeeklyFeatured(report) {
  if (!report) {
    elements.weekly.featured.hidden = true;
    return;
  }
  elements.weekly.featured.hidden = false;
  elements.weekly.featuredTitle.textContent = report.title || report.report_id;

  const tw = report.time_window || {};
  const windowLabel =
    tw.start && tw.end
      ? `${formatDateTime(tw.start)} → ${formatDateTime(tw.end)} (${tw.timezone || "UTC"})`
      : "-";
  elements.weekly.featuredWindow.textContent = `${report.report_id} · ${windowLabel}`;

  elements.weekly.featuredPrs.textContent = formatNumber(report.merged_pr_count);

  elements.weekly.featuredCategories.innerHTML = "";
  const categories = report.category_breakdown || {};
  Object.entries(categories)
    .filter(([, count]) => count > 0)
    .sort((a, b) => b[1] - a[1])
    .forEach(([label, count]) => {
      const li = document.createElement("li");
      const strong = document.createElement("strong");
      strong.textContent = `${count}`;
      li.append(strong, document.createTextNode(`  ${label}`));
      elements.weekly.featuredCategories.append(li);
    });

  elements.weekly.featuredRecommendations.innerHTML = "";
  const targetLabels = {
    megatron: "Megatron-LM",
    torchtitan: "torchtitan",
    primus_turbo: "Primus-Turbo",
  };
  const recommendations = report.recommendations || {};
  Object.entries(recommendations).forEach(([key, rec]) => {
    const row = document.createElement("li");
    row.className = "recommendation-list__row";
    const label = document.createElement("span");
    label.className = "label";
    label.textContent = targetLabels[key] || key;
    row.append(label, createPill(rec || "-", recommendationClass(rec)));
    elements.weekly.featuredRecommendations.append(row);
  });

  elements.weekly.featuredFindings.innerHTML = "";
  (report.key_findings || []).slice(0, 6).forEach((finding) => {
    const li = document.createElement("li");
    li.textContent = finding;
    elements.weekly.featuredFindings.append(li);
  });

}

function buildWeeklyCard(report) {
  const card = document.createElement("article");
  card.className = "weekly-card";

  const header = document.createElement("div");
  header.className = "weekly-card__header";

  const idEl = document.createElement("h3");
  idEl.className = "weekly-card__id";
  idEl.textContent = report.report_id;

  const tw = report.time_window || {};
  const windowEl = document.createElement("span");
  windowEl.className = "weekly-card__window";
  if (tw.start && tw.end) {
    windowEl.textContent = `${formatDateTime(tw.start)} → ${formatDateTime(tw.end)}`;
  } else {
    windowEl.textContent = report.generated_at ? formatDate(report.generated_at) : "";
  }

  header.append(idEl, windowEl);

  const meta = document.createElement("div");
  meta.className = "weekly-card__meta";
  const merged = document.createElement("span");
  merged.innerHTML = `Merged PRs <strong>${formatNumber(report.merged_pr_count)}</strong>`;
  meta.append(merged);

  const categoryParts = Object.entries(report.category_breakdown || {})
    .filter(([, count]) => count > 0)
    .map(([label, count]) => `${label} ${count}`);
  if (categoryParts.length) {
    const categories = document.createElement("span");
    categories.textContent = categoryParts.join(" · ");
    meta.append(categories);
  }

  const pills = document.createElement("div");
  pills.className = "weekly-card__pills";
  const targetLabels = {
    megatron: "Megatron-LM",
    torchtitan: "torchtitan",
    primus_turbo: "Primus-Turbo",
  };
  Object.entries(report.recommendations || {}).forEach(([key, value]) => {
    const text = `${targetLabels[key] || key}: ${value}`;
    pills.append(createPill(text, recommendationClass(value)));
  });

  const findingsList = document.createElement("ul");
  findingsList.className = "findings-list";
  (report.key_findings || []).slice(0, 3).forEach((finding) => {
    const li = document.createElement("li");
    li.textContent = finding;
    findingsList.append(li);
  });

  const footer = document.createElement("div");
  footer.className = "weekly-card__footer";
  const generated = document.createElement("span");
  generated.className = "weekly-card__window";
  generated.textContent = report.generated_at
    ? `Generated ${formatDate(report.generated_at)}`
    : "";
  footer.append(generated);

  if (report.report_github_url) {
    const link = document.createElement("a");
    link.className = "button";
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    link.href = report.report_github_url;
    link.textContent = "Open Markdown report";
    footer.append(link);
  }

  card.append(header, meta, pills, findingsList, footer);
  return card;
}

function renderWeeklyList(reports) {
  elements.weekly.list.innerHTML = "";
  if (elements.weekly.filterRow) {
    elements.weekly.filterRow.hidden = state.weekly.reports.length <= 1;
  }
  elements.weekly.resultCount.textContent = `${reports.length} report${
    reports.length === 1 ? "" : "s"
  }`;

  if (!reports.length) {
    elements.weekly.list.hidden = true;
    elements.weekly.emptyState.hidden = false;
    return;
  }

  elements.weekly.emptyState.hidden = true;
  elements.weekly.list.hidden = false;
  reports.forEach((report) => {
    elements.weekly.list.append(buildWeeklyCard(report));
  });
}

function applyWeeklyFilters() {
  const rec = elements.weekly.recommendationFilter.value;
  const search = elements.weekly.searchFilter.value.trim().toLowerCase();

  state.weekly.filteredReports = state.weekly.reports.filter((report) => {
    const recs = Object.values(report.recommendations || {}).map((v) =>
      (v || "").toLowerCase()
    );
    const recMatch = rec === "all" || recs.includes(rec.toLowerCase());

    const haystackParts = [
      report.report_id,
      report.title,
      ...(report.key_findings || []),
      ...Object.values(report.recommendations || {}),
      ...Object.keys(report.category_breakdown || {}),
    ];
    const haystack = haystackParts.join(" ").toLowerCase();
    const searchMatch = !search || haystack.includes(search);
    return recMatch && searchMatch;
  });

  renderWeeklyList(state.weekly.filteredReports);
}

function attachWeeklyFilters() {
  elements.weekly.recommendationFilter.addEventListener("change", applyWeeklyFilters);
  elements.weekly.searchFilter.addEventListener("input", applyWeeklyFilters);
}

// ---------- Backend Gap ----------

function createArtifactLink(artifact) {
  const link = document.createElement("a");
  link.className = `artifact-link${artifact.primary ? " artifact-link--primary" : ""}`;
  link.href = artifact.path;
  link.textContent = artifact.label;
  if (artifact.format === "pdf") {
    link.target = "_blank";
    link.rel = "noopener noreferrer";
  }
  return link;
}

function compactFact(label, value, tone) {
  const item = document.createElement("div");
  item.className = `compact-fact${tone ? ` compact-fact--${tone}` : ""}`;

  const labelEl = document.createElement("span");
  labelEl.className = "compact-fact__label";
  labelEl.textContent = label;

  const valueEl = document.createElement("strong");
  valueEl.className = "compact-fact__value";
  valueEl.textContent = value || "-";

  item.append(labelEl, valueEl);
  return item;
}

function driftTone(commitGap) {
  if (commitGap >= 500) return "danger";
  if (commitGap >= 100) return "warning";
  return "success";
}

function buildBackendFocusItems(report) {
  const summary = report.dashboard_summary || {};
  if (Array.isArray(summary.why_it_matters) && summary.why_it_matters.length) {
    return summary.why_it_matters.slice(0, 5);
  }

  const items = [];
  const commitGap = report.stats && report.stats.commit_gap;
  const backendLabel = report.backend && report.backend.label ? report.backend.label : "Backend";

  if (commitGap !== undefined) {
    items.push(
      `${backendLabel} is ${formatNumber(commitGap)} upstream commits behind; review upgrade scope before syncing.`
    );
  }

  if (report.local && report.upstream) {
    items.push(
      `Version position: Primus ${report.local.version || "-"} vs upstream ${
        report.upstream.version || "-"
      }.`
    );
  }

  (report.highlights || []).forEach((highlight) => items.push(highlight));

  if (report.integration && report.integration.integration_model) {
    items.push(`Integration surface: ${report.integration.integration_model}.`);
  }

  return items.slice(0, 5);
}

function listSection(title, items) {
  if (!Array.isArray(items) || !items.length) {
    return null;
  }

  const section = document.createElement("section");
  section.className = "backend-card__insight";

  const heading = document.createElement("h4");
  heading.textContent = title;

  const list = document.createElement("ul");
  list.className = "compact-list";
  items.slice(0, 5).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    list.append(li);
  });

  section.append(heading, list);
  return section;
}

function renderBackendGapStats(payload) {
  elements.backendGap.statRow.innerHTML = "";
  const summary = (payload && payload.summary) || {};
  elements.backendGap.statRow.append(
    statTile("Total reports", formatNumber(summary.total_reports || 0))
  );
  elements.backendGap.statRow.append(
    statTile("Verified reports", formatNumber(summary.verified_reports || 0))
  );
  elements.backendGap.statRow.append(
    statTile("Tracked backends", formatNumber(summary.total_backends || 0))
  );
  elements.backendGap.statRow.append(
    statTile("Latest report", summary.latest_report_date || "-")
  );
}

function buildBackendDeepDiveCard(report) {
  const card = document.createElement("article");
  card.className = "backend-card";

  const backendLabel = report.backend && report.backend.label ? report.backend.label : "Backend";
  const stats = report.stats || {};
  const integration = report.integration || {};
  const summary = report.dashboard_summary || {};

  const header = document.createElement("header");
  header.className = "backend-card__header";

  const titleGroup = document.createElement("div");
  titleGroup.className = "backend-card__title-group";

  const eyebrow = document.createElement("p");
  eyebrow.className = "backend-card__eyebrow";
  eyebrow.textContent = backendLabel;

  const title = document.createElement("h3");
  title.textContent = report.title || backendLabel;

  const scope = document.createElement("p");
  scope.className = "backend-card__scope";
  scope.textContent = summary.headline || report.scope || "";

  titleGroup.append(eyebrow, title, scope);

  const gapBadge = document.createElement("div");
  gapBadge.className = `commit-gap-badge commit-gap-badge--${driftTone(stats.commit_gap || 0)}`;
  const gapValue = document.createElement("strong");
  gapValue.textContent = formatNumber(stats.commit_gap);
  const gapLabel = document.createElement("span");
  gapLabel.textContent = "commits behind";
  gapBadge.append(gapValue, gapLabel);

  header.append(titleGroup, gapBadge);

  const facts = document.createElement("div");
  facts.className = "backend-card__facts";
  facts.append(
    compactFact("Primus baseline", report.local && report.local.version),
    compactFact("Upstream version", report.upstream && report.upstream.version),
    compactFact("Recommendation", summary.recommendation || report.status),
    compactFact("Integration files", formatNumber(integration.backend_files))
  );

  const focus = document.createElement("section");
  focus.className = "backend-card__focus";
  const focusTitle = document.createElement("h4");
  focusTitle.textContent = "What matters";
  const focusList = document.createElement("ul");
  focusList.className = "findings-list";
  buildBackendFocusItems(report).forEach((item) => {
    const li = document.createElement("li");
    li.textContent = item;
    focusList.append(li);
  });
  focus.append(focusTitle, focusList);

  const insights = document.createElement("div");
  insights.className = "backend-card__insights";
  [
    listSection("New upstream capabilities", summary.feature_deltas),
    listSection("Dependency shift", summary.dependency_deltas),
    listSection("Primus integration risk", summary.integration_risks),
  ]
    .filter(Boolean)
    .forEach((section) => insights.append(section));

  const footer = document.createElement("footer");
  footer.className = "backend-card__footer";
  const meta = document.createElement("span");
  meta.className = "backend-card__meta";
  meta.textContent = `Updated ${report.generated_at || "-"} · ${report.status || "unknown"}`;
  footer.append(meta);

  const artifactBar = document.createElement("div");
  artifactBar.className = "artifacts";
  (report.artifacts || [])
    .filter((artifact) => artifact.format === "pdf")
    .forEach((artifact) => {
      artifactBar.append(createArtifactLink(artifact));
    });
  footer.append(artifactBar);

  card.append(header, facts, focus);
  if (insights.children.length) {
    card.append(insights);
  }
  card.append(footer);
  return card;
}

function renderBackendGapReports(reports) {
  const list = elements.backendGap.reportList;
  list.innerHTML = "";
  if (elements.backendGap.filterRow) {
    elements.backendGap.filterRow.hidden = state.backendGap.reports.length <= 1;
  }
  elements.backendGap.resultCount.textContent = `${reports.length} report${
    reports.length === 1 ? "" : "s"
  }`;

  if (!reports.length) {
    list.hidden = true;
    elements.backendGap.emptyState.hidden = false;
    return;
  }

  elements.backendGap.emptyState.hidden = true;
  list.hidden = false;

  reports.forEach((report) => {
    list.append(buildBackendDeepDiveCard(report));
  });
}

function applyBackendGapFilters() {
  const backendValue = elements.backendGap.backendFilter.value;
  const statusValue = elements.backendGap.statusFilter.value;
  const searchValue = elements.backendGap.searchFilter.value.trim().toLowerCase();

  state.backendGap.filteredReports = state.backendGap.reports.filter((report) => {
    const backendMatch = backendValue === "all" || report.backend.key === backendValue;
    const statusMatch = statusValue === "all" || report.status === statusValue;

    const haystack = [
      report.title,
      report.scope,
      report.backend && report.backend.label,
      report.backend && report.backend.key,
      report.local && report.local.version,
      report.upstream && report.upstream.version,
      report.integration && report.integration.integration_model,
      report.dashboard_summary && report.dashboard_summary.headline,
      report.dashboard_summary && report.dashboard_summary.recommendation,
      ...((report.dashboard_summary && report.dashboard_summary.why_it_matters) || []),
      ...((report.dashboard_summary && report.dashboard_summary.feature_deltas) || []),
      ...((report.dashboard_summary && report.dashboard_summary.dependency_deltas) || []),
      ...((report.dashboard_summary && report.dashboard_summary.integration_risks) || []),
      ...(report.highlights || []),
    ]
      .join(" ")
      .toLowerCase();

    const searchMatch = !searchValue || haystack.includes(searchValue);
    return backendMatch && statusMatch && searchMatch;
  });

  renderBackendGapReports(state.backendGap.filteredReports);
}

function populateBackendFilter(backends) {
  (backends || []).forEach((backend) => {
    const option = document.createElement("option");
    option.value = backend.key;
    option.textContent = `${backend.label} (${backend.count})`;
    elements.backendGap.backendFilter.append(option);
  });
}

function attachBackendGapFilters() {
  elements.backendGap.backendFilter.addEventListener("change", applyBackendGapFilters);
  elements.backendGap.statusFilter.addEventListener("change", applyBackendGapFilters);
  elements.backendGap.searchFilter.addEventListener("input", applyBackendGapFilters);
}

// ---------- Loader ----------

async function fetchJsonOptional(url) {
  try {
    const response = await fetch(url, { cache: "no-store" });
    if (!response.ok) {
      return { ok: false, error: `HTTP ${response.status}` };
    }
    const payload = await response.json();
    return { ok: true, payload };
  } catch (error) {
    return { ok: false, error: error.message };
  }
}

async function loadDashboard() {
  attachWeeklyFilters();
  attachBackendGapFilters();

  const weeklyResult = await fetchJsonOptional(WEEKLY_DATA_URL);
  const gapResult = await fetchJsonOptional(BACKEND_GAP_DATA_URL);

  // Hero chips (best effort — show whatever loaded)
  renderHeroChips(weeklyResult.payload, gapResult.payload);

  // --- Weekly reports section ---
  elements.weekly.loadingState.hidden = true;
  if (weeklyResult.ok && weeklyResult.payload) {
    state.weekly.reports = weeklyResult.payload.reports || [];
    renderWeeklyStatRow(weeklyResult.payload);
    renderWeeklyFeatured(state.weekly.reports[0]);
    applyWeeklyFilters();
  } else {
    elements.weekly.errorState.hidden = false;
    elements.weekly.errorState.textContent = `Unable to load weekly reports. ${weeklyResult.error || ""}`.trim();
    elements.weekly.list.hidden = true;
    elements.weekly.resultCount.textContent = "0 reports";
    elements.weekly.featured.hidden = true;
  }

  // --- Backend gap section ---
  elements.backendGap.loadingState.hidden = true;
  if (gapResult.ok && gapResult.payload) {
    state.backendGap.reports = gapResult.payload.reports || [];
    renderBackendGapStats(gapResult.payload);
    populateBackendFilter(gapResult.payload.backends || []);
    applyBackendGapFilters();
  } else {
    elements.backendGap.errorState.hidden = false;
    elements.backendGap.errorState.textContent = `Unable to load backend gap data. ${gapResult.error || ""}`.trim();
    elements.backendGap.reportList.hidden = true;
    elements.backendGap.resultCount.textContent = "0 reports";
  }
}

loadDashboard();
