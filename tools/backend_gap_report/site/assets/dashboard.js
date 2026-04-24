const DATA_URL = "./dashboard-data/index.json";

const state = {
  reports: [],
  filteredReports: [],
};

const elements = {
  totalReports: document.getElementById("total-reports"),
  totalBackends: document.getElementById("total-backends"),
  verifiedReports: document.getElementById("verified-reports"),
  latestReportDate: document.getElementById("latest-report-date"),
  backendFilter: document.getElementById("backend-filter"),
  statusFilter: document.getElementById("status-filter"),
  searchFilter: document.getElementById("search-filter"),
  resultCount: document.getElementById("result-count"),
  loadingState: document.getElementById("loading-state"),
  errorState: document.getElementById("error-state"),
  emptyState: document.getElementById("empty-state"),
  reportList: document.getElementById("report-list"),
};


function formatNumber(value) {
  if (value === undefined || value === null || value === "") {
    return "-";
  }
  return new Intl.NumberFormat("en-US").format(value);
}


function setSummary(summary) {
  elements.totalReports.textContent = formatNumber(summary.total_reports);
  elements.totalBackends.textContent = formatNumber(summary.total_backends);
  elements.verifiedReports.textContent = formatNumber(summary.verified_reports);
  elements.latestReportDate.textContent = summary.latest_report_date || "-";
}


function buildBadge(label, className) {
  const span = document.createElement("span");
  span.className = `badge ${className}`;
  span.textContent = label;
  return span;
}


function buildMetric(label, value) {
  const wrapper = document.createElement("div");
  wrapper.className = "metric";

  const labelNode = document.createElement("span");
  labelNode.className = "metric__label";
  labelNode.textContent = label;

  const valueNode = document.createElement("strong");
  valueNode.textContent = value;

  wrapper.append(labelNode, valueNode);
  return wrapper;
}


function createDetailBlock(title, rows) {
  const block = document.createElement("section");
  block.className = "detail-block";

  const heading = document.createElement("h3");
  heading.textContent = title;

  const list = document.createElement("dl");
  rows.forEach(([label, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value || "-";
    list.append(dt, dd);
  });

  block.append(heading, list);
  return block;
}


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


function renderReports(reports) {
  elements.reportList.innerHTML = "";
  elements.resultCount.textContent = `${reports.length} report${reports.length === 1 ? "" : "s"}`;

  if (!reports.length) {
    elements.reportList.hidden = true;
    elements.emptyState.hidden = false;
    return;
  }

  elements.emptyState.hidden = true;
  elements.reportList.hidden = false;

  reports.forEach((report) => {
    const card = document.createElement("article");
    card.className = "report-card";

    const header = document.createElement("div");
    header.className = "report-card__header";

    const titleGroup = document.createElement("div");
    titleGroup.className = "report-card__title-group";

    const title = document.createElement("h2");
    title.textContent = report.title;

    const scope = document.createElement("p");
    scope.className = "report-card__scope";
    scope.textContent = report.scope;

    titleGroup.append(title, scope);

    const badges = document.createElement("div");
    badges.className = "badges";
    badges.append(
      buildBadge(report.backend.label, "badge--backend"),
      buildBadge(report.status, `badge--${report.status}`)
    );

    header.append(titleGroup, badges);

    const metrics = document.createElement("div");
    metrics.className = "metrics";
    metrics.append(
      buildMetric("Generated", report.generated_at),
      buildMetric("Commit Gap", formatNumber(report.stats.commit_gap)),
      buildMetric("Diff Files", formatNumber(report.stats.diff_files)),
      buildMetric(
        "Diff Size",
        `+${formatNumber(report.stats.insertions)} / -${formatNumber(report.stats.deletions)}`
      )
    );

    const detailGrid = document.createElement("div");
    detailGrid.className = "detail-grid";
    detailGrid.append(
      createDetailBlock("Local", [
        ["Source", report.local.source_path],
        ["Version", report.local.version],
        ["Commit", report.local.commit],
        ["Date", report.local.commit_date],
      ]),
      createDetailBlock("Upstream", [
        ["Repository", report.upstream.repo],
        ["Ref", report.upstream.ref],
        ["Version", report.upstream.version],
        ["Commit", report.upstream.commit],
      ]),
      createDetailBlock("Integration", [
        ["Model", report.integration?.integration_model || "-"],
        ["Backend Files", formatNumber(report.integration?.backend_files)],
        ["Tracked Files", formatNumber(report.integration?.tracked_files)],
        ["Status", report.status],
      ])
    );

    const highlightList = document.createElement("ul");
    highlightList.className = "highlights";
    report.highlights.forEach((highlight) => {
      const item = document.createElement("li");
      item.textContent = highlight;
      highlightList.append(item);
    });

    const artifactBar = document.createElement("div");
    artifactBar.className = "artifacts";
    report.artifacts.forEach((artifact) => {
      artifactBar.append(createArtifactLink(artifact));
    });

    card.append(header, metrics, detailGrid, highlightList, artifactBar);
    elements.reportList.append(card);
  });
}


function applyFilters() {
  const backendValue = elements.backendFilter.value;
  const statusValue = elements.statusFilter.value;
  const searchValue = elements.searchFilter.value.trim().toLowerCase();

  state.filteredReports = state.reports.filter((report) => {
    const backendMatch =
      backendValue === "all" || report.backend.key === backendValue;
    const statusMatch =
      statusValue === "all" || report.status === statusValue;

    const haystack = [
      report.title,
      report.scope,
      report.backend.label,
      report.backend.key,
      ...(report.highlights || []),
    ]
      .join(" ")
      .toLowerCase();

    const searchMatch = !searchValue || haystack.includes(searchValue);
    return backendMatch && statusMatch && searchMatch;
  });

  renderReports(state.filteredReports);
}


function populateBackendFilter(backends) {
  backends.forEach((backend) => {
    const option = document.createElement("option");
    option.value = backend.key;
    option.textContent = `${backend.label} (${backend.count})`;
    elements.backendFilter.append(option);
  });
}


function attachFilters() {
  elements.backendFilter.addEventListener("change", applyFilters);
  elements.statusFilter.addEventListener("change", applyFilters);
  elements.searchFilter.addEventListener("input", applyFilters);
}


async function loadDashboard() {
  try {
    const response = await fetch(DATA_URL, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Failed to load ${DATA_URL}: ${response.status}`);
    }

    const payload = await response.json();
    state.reports = payload.reports || [];

    setSummary(payload.summary || {});
    populateBackendFilter(payload.backends || []);
    attachFilters();
    applyFilters();

    elements.loadingState.hidden = true;
    elements.errorState.hidden = true;
  } catch (error) {
    elements.loadingState.hidden = true;
    elements.reportList.hidden = true;
    elements.emptyState.hidden = true;
    elements.errorState.hidden = false;
    elements.errorState.textContent = `Unable to load dashboard data. ${error.message}`;
  }
}


loadDashboard();
