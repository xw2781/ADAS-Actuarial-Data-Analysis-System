import { fetchProjectDatasetTypes } from "/ui/dataset/dataset_types_source.js";
import { buildPathTreeFromPaths } from "/ui/shared/path_tree_picker.js";

const qs = new URLSearchParams(window.location.search);
const projectName = String(qs.get("project") || "").trim();
const projectFolder = String(qs.get("folder") || "").trim();
const projectTablePath = String(qs.get("tablePath") || "").trim();

const els = {
  root: document.getElementById("projectInstanceRoot"),
  title: document.getElementById("projectTitle"),
  subtitle: document.getElementById("projectSubtitle"),
  status: document.getElementById("projectInstanceStatus"),
  pathTree: document.getElementById("pathTree"),
  selectedPathText: document.getElementById("selectedPathText"),
  datasetTableBody: document.getElementById("datasetTableBody"),
  windowLayer: document.getElementById("datasetWindowLayer"),
};

const DATASET_COLUMNS = 9;
let selectedPath = "";
let datasetRows = [];
let nextWindowZ = 1;
let windowSeq = 1;

function toText(value) {
  return String(value ?? "").trim();
}

function normalizePath(value) {
  return toText(value)
    .split("\\")
    .map((part) => part.trim())
    .filter(Boolean)
    .join("\\");
}

async function fetchJson(url, errorPrefix) {
  const resp = await fetch(url);
  if (!resp.ok) {
    const detail = await resp.text().catch(() => "");
    throw new Error(detail || `${errorPrefix} (${resp.status}).`);
  }
  return await resp.json().catch(() => ({}));
}

function setStatus(text, isError = false) {
  if (!els.status) return;
  els.status.textContent = toText(text);
  els.status.classList.toggle("error", !!isError);
}

function setEmptyTable(message) {
  if (!els.datasetTableBody) return;
  els.datasetTableBody.innerHTML = "";
  const tr = document.createElement("tr");
  const td = document.createElement("td");
  td.className = "pi-table-empty";
  td.colSpan = DATASET_COLUMNS;
  td.textContent = message;
  tr.appendChild(td);
  els.datasetTableBody.appendChild(tr);
}

function getDatasetName(row) {
  return toText(row?.[0]);
}

function getMethodType(row) {
  const formula = toText(row?.[4]);
  const calculated = row?.[3] === true || String(row?.[3] || "").trim().toLowerCase() === "true";
  if (formula || calculated) return "Calculated";
  return "Source";
}

function renderDatasetTable() {
  if (!els.datasetTableBody) return;
  els.datasetTableBody.innerHTML = "";
  if (!datasetRows.length) {
    setEmptyTable("No dataset types are defined for this project.");
    return;
  }

  for (const row of datasetRows) {
    const datasetName = getDatasetName(row);
    const tr = document.createElement("tr");
    tr.title = selectedPath
      ? `Open ${datasetName} for ${selectedPath}`
      : "Select a reserving class path before opening a dataset.";

    const values = [
      datasetName,
      datasetName,
      toText(row?.[1]),
      toText(row?.[4]),
      toText(row?.[2]),
      getMethodType(row),
      "",
      "",
      "",
    ];

    for (const value of values) {
      const td = document.createElement("td");
      td.textContent = value;
      td.title = value;
      tr.appendChild(td);
    }

    tr.addEventListener("dblclick", () => {
      openDatasetWindow(datasetName);
    });

    els.datasetTableBody.appendChild(tr);
  }
}

function getNodeValueType(node, fallback = "") {
  return toText(node?.valueType ?? node?.value_type ?? node?.nodeType ?? node?.node_type ?? fallback).toLowerCase();
}

function createTypeIcon(rawType) {
  const type = toText(rawType).toLowerCase();
  if (!type) return null;

  const el = document.createElement("span");
  el.className = `ptree-type-icon ${type}`;

  if (type === "folder") {
    el.title = "Folder";
    el.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>';
    return el;
  }
  if (type === "calculated") {
    el.title = "Calculated class type";
    el.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h8v6H4zM12 11h8v6h-8zM12 3h8v6h-8z"/></svg>';
    return el;
  }
  if (type === "calculated-muted" || type === "calculated_muted") {
    el.className = "ptree-type-icon calculated-muted";
    el.title = "Calculated class context (imported node)";
    el.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M4 7h8v6H4zM12 11h8v6h-8zM12 3h8v6h-8z"/></svg>';
    return el;
  }
  if (type === "imported" || type === "source") {
    el.className = "ptree-type-icon imported";
    el.title = "Imported value type";
    el.innerHTML = '<svg viewBox="0 0 24 24" aria-hidden="true"><ellipse cx="12" cy="5.5" rx="7" ry="2.5"/><path d="M5 5.5v9c0 1.4 3.1 2.5 7 2.5s7-1.1 7-2.5v-9"/></svg>';
    return el;
  }
  return null;
}

function setFolderIconExpanded(icon, expanded) {
  if (!icon || !icon.classList?.contains("folder")) return;
  icon.classList.toggle("open", !!expanded);
  icon.innerHTML = expanded
    ? '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M20 6h-8l-2-2H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2zm0 12H4V8h16v10z"/></svg>'
    : '<svg viewBox="0 0 24 24" aria-hidden="true"><path d="M10 4H4c-1.1 0-1.99.9-1.99 2L2 18c0 1.1.9 2 2 2h16c1.1 0 2-.9 2-2V8c0-1.1-.9-2-2-2h-8l-2-2z"/></svg>';
}

function normalizeTreeNode(rawNode, parentPath = "", levelLabels = []) {
  const raw = rawNode && typeof rawNode === "object" ? rawNode : {};
  const name = toText(raw.name);
  const path = normalizePath(raw.path || (parentPath && name ? `${parentPath}\\${name}` : name));
  const rawChildren = Array.isArray(raw.children) ? raw.children : [];
  const levelIndexRaw = Number(raw.levelIndex ?? raw.level_index);
  const levelIndex = Number.isInteger(levelIndexRaw) && levelIndexRaw >= 0
    ? levelIndexRaw
    : (path ? path.split("\\").length : 0);
  const levelLabel = toText(raw.levelLabel ?? raw.level_label)
    || levelLabels[Math.max(0, levelIndex - 1)]
    || (levelIndex > 0 ? `Level ${levelIndex}` : "");
  const children = rawChildren.map((child) => normalizeTreeNode(child, path, levelLabels));
  return {
    name: name || (path ? path.split("\\").pop() : "All"),
    path,
    levelIndex,
    levelLabel,
    valueType: getNodeValueType(raw, children.length ? "folder" : "imported"),
    children,
  };
}

function getFirstSelectablePath(nodes) {
  for (const node of Array.isArray(nodes) ? nodes : []) {
    if (Array.isArray(node.children) && node.children.length) {
      const childPath = getFirstSelectablePath(node.children);
      if (childPath) return childPath;
    } else if (node.path) {
      return node.path;
    }
  }
  return "";
}

function getLevelLabelsFromPayload(data) {
  const levels = Array.isArray(data?.levels) ? data.levels : [];
  return levels.map((level, index) => (
    toText(level?.field_name ?? level?.name ?? level?.label ?? level?.level_label)
    || `Level ${index + 1}`
  ));
}

function setSelectedPath(path) {
  selectedPath = normalizePath(path);
  if (els.selectedPathText) {
    els.selectedPathText.textContent = selectedPath || "Select a reserving class path.";
    els.selectedPathText.title = selectedPath;
  }
  if (els.pathTree) {
    els.pathTree.querySelectorAll(".ptree-leaf").forEach((leaf) => {
      leaf.classList.toggle("active-path", normalizePath(leaf.dataset.path || "") === selectedPath);
    });
  }
  renderDatasetTable();
}

function renderTreeNode(node, depth = 0) {
  const container = document.createElement("div");
  container.className = "ptree-node";
  const children = Array.isArray(node?.children) ? node.children : [];
  const hasChildren = children.length > 0;
  const path = normalizePath(node?.path || "");
  const levelLabel = toText(node?.levelLabel ?? node?.level_label ?? node?.levelIndex ?? node?.level_index ?? "");

  if (hasChildren) {
    const row = document.createElement("div");
    row.className = "ptree-folder";
    row.style.paddingLeft = `${8 + depth * 16}px`;

    const arrow = document.createElement("span");
    arrow.className = "ptree-arrow expanded";
    arrow.innerHTML = '<svg viewBox="0 0 24 24" fill="currentColor" aria-hidden="true"><path d="M8.59 16.59 13.17 12 8.59 7.41 10 6l6 6-6 6z"/></svg>';

    const icon = createTypeIcon("folder");
    setFolderIconExpanded(icon, true);

    const label = document.createElement("span");
    label.className = "ptree-label";
    label.textContent = toText(node?.name) || path || "All";
    label.title = path || label.textContent;

    const level = document.createElement("span");
    level.className = "ptree-level";
    level.textContent = levelLabel;
    level.title = levelLabel;

    row.append(arrow, icon, label, level);

    const childWrap = document.createElement("div");
    childWrap.className = "ptree-children expanded";
    for (const child of children) childWrap.appendChild(renderTreeNode(child, depth + 1));

    row.addEventListener("click", () => {
      const expanded = childWrap.classList.toggle("expanded");
      arrow.classList.toggle("expanded", expanded);
      setFolderIconExpanded(icon, expanded);
    });

    container.append(row, childWrap);
    return container;
  }

  const leaf = document.createElement("div");
  leaf.className = "ptree-leaf";
  leaf.dataset.path = path;
  leaf.style.paddingLeft = `${24 + depth * 16}px`;

  const icon = createTypeIcon(getNodeValueType(node, "imported"));

  const label = document.createElement("span");
  label.className = "ptree-label";
  label.textContent = toText(node?.name) || path;
  label.title = path;

  leaf.append(icon, label);
  if (levelLabel) {
    const level = document.createElement("span");
    level.className = "ptree-level";
    level.textContent = levelLabel;
    level.title = levelLabel;
    leaf.appendChild(level);
  }
  leaf.addEventListener("click", () => setSelectedPath(path));
  container.appendChild(leaf);
  return container;
}

async function loadPathTree() {
  if (!els.pathTree) return;
  els.pathTree.innerHTML = '<div class="ptree-empty">Loading reserving class paths...</div>';
  if (!projectName) {
    els.pathTree.innerHTML = '<div class="ptree-empty">Project name is missing.</div>';
    return;
  }

  try {
    const payload = await fetchJson(
      `/reserving_class_path_tree?project_name=${encodeURIComponent(projectName)}`,
      "Failed to load reserving class path tree",
    );
    const data = payload?.data && typeof payload.data === "object" ? payload.data : {};
    const levelLabels = getLevelLabelsFromPayload(data);
    let root = data?.tree && typeof data.tree === "object"
      ? normalizeTreeNode(data.tree, "", levelLabels)
      : null;
    if ((!root || !Array.isArray(root.children) || !root.children.length) && Array.isArray(data.paths) && data.paths.length) {
      root = normalizeTreeNode(buildPathTreeFromPaths(data.paths, { delimiter: "\\", levelLabels }), "", levelLabels);
    }
    const rootChildren = Array.isArray(root?.children) ? root.children : [];
    if (!rootChildren.length) {
      els.pathTree.innerHTML = '<div class="ptree-empty">No reserving class paths found.</div>';
      return;
    }
    els.pathTree.innerHTML = "";
    for (const child of rootChildren) {
      els.pathTree.appendChild(renderTreeNode(child, 0));
    }
    setSelectedPath(getFirstSelectablePath(rootChildren));
  } catch (err) {
    console.error("Failed to load reserving class paths:", err);
    els.pathTree.innerHTML = '<div class="ptree-empty">Failed to load reserving class paths.</div>';
    setStatus(toText(err?.message) || "Failed to load reserving class paths.", true);
  }
}

async function loadDatasets() {
  setEmptyTable("Loading dataset types...");
  if (!projectName) {
    setEmptyTable("Project name is missing.");
    return;
  }
  try {
    const fetched = await fetchProjectDatasetTypes(projectName);
    datasetRows = Array.isArray(fetched?.data?.rows)
      ? fetched.data.rows.filter((row) => getDatasetName(row))
      : [];
    renderDatasetTable();
  } catch (err) {
    console.error("Failed to load dataset types:", err);
    setEmptyTable("Failed to load dataset types.");
    setStatus(toText(err?.message) || "Failed to load dataset types.", true);
  }
}

function getWindowBounds() {
  const rect = els.root?.getBoundingClientRect?.();
  return {
    width: Math.max(480, Number(rect?.width || window.innerWidth || 900)),
    height: Math.max(360, Number(rect?.height || window.innerHeight || 640)),
  };
}

function clampWindowRect(rect) {
  const bounds = getWindowBounds();
  const width = Math.max(420, Math.min(Number(rect.width) || 760, bounds.width));
  const height = Math.max(280, Math.min(Number(rect.height) || 500, bounds.height));
  const x = Math.max(0, Math.min(Number(rect.x) || 0, bounds.width - width));
  const y = Math.max(0, Math.min(Number(rect.y) || 0, bounds.height - height));
  return { x, y, width, height };
}

function applyWindowRect(frame, rect) {
  const next = clampWindowRect(rect);
  frame.style.left = `${Math.round(next.x)}px`;
  frame.style.top = `${Math.round(next.y)}px`;
  frame.style.width = `${Math.round(next.width)}px`;
  frame.style.height = `${Math.round(next.height)}px`;
  return next;
}

function raiseWindow(frame) {
  frame.style.zIndex = String(++nextWindowZ);
}

function lockDatasetViewerInputs(iframe, datasetName) {
  let doc = null;
  try {
    doc = iframe.contentDocument || iframe.contentWindow?.document || null;
  } catch {
    return;
  }
  if (!doc) return;

  const projectInput = doc.getElementById("projectSelect");
  const pathInput = doc.getElementById("pathInput");
  const triInput = doc.getElementById("triInput");
  if (projectInput) {
    projectInput.value = projectName;
    projectInput.readOnly = true;
    projectInput.title = "Project is set by the project instance tab.";
  }
  if (pathInput) {
    pathInput.value = selectedPath;
    pathInput.readOnly = true;
    pathInput.title = "Reserving class path is set by the project instance tab.";
  }
  if (triInput && datasetName) {
    triInput.value = datasetName;
  }
  for (const id of ["projectTreeBtn", "pathTreeBtn"]) {
    const button = doc.getElementById(id);
    if (button) {
      button.disabled = true;
      button.title = "Set by the project instance tab";
    }
  }
}

function buildDatasetViewerUrl(datasetName, inst) {
  const params = new URLSearchParams();
  params.set("project", projectName);
  params.set("path", selectedPath);
  params.set("tri", datasetName);
  params.set("inst", inst);
  params.set("project_instance", "1");
  params.set("v", String(Date.now()));
  return `/ui/dataset/dataset_viewer.html?${params.toString()}`;
}

function startMove(frame, event) {
  if (event.button !== 0) return;
  raiseWindow(frame);
  const startRect = frame.getBoundingClientRect();
  const rootRect = els.root.getBoundingClientRect();
  const start = {
    x: startRect.left - rootRect.left,
    y: startRect.top - rootRect.top,
    width: startRect.width,
    height: startRect.height,
    px: event.clientX,
    py: event.clientY,
  };

  const onMove = (e) => {
    applyWindowRect(frame, {
      x: start.x + e.clientX - start.px,
      y: start.y + e.clientY - start.py,
      width: start.width,
      height: start.height,
    });
  };
  const onUp = () => {
    document.removeEventListener("mousemove", onMove, true);
    document.removeEventListener("mouseup", onUp, true);
  };
  document.addEventListener("mousemove", onMove, true);
  document.addEventListener("mouseup", onUp, true);
  event.preventDefault();
}

function startResize(frame, event) {
  if (event.button !== 0) return;
  raiseWindow(frame);
  const startRect = frame.getBoundingClientRect();
  const rootRect = els.root.getBoundingClientRect();
  const start = {
    x: startRect.left - rootRect.left,
    y: startRect.top - rootRect.top,
    width: startRect.width,
    height: startRect.height,
    px: event.clientX,
    py: event.clientY,
  };

  const onMove = (e) => {
    applyWindowRect(frame, {
      x: start.x,
      y: start.y,
      width: start.width + e.clientX - start.px,
      height: start.height + e.clientY - start.py,
    });
  };
  const onUp = () => {
    document.removeEventListener("mousemove", onMove, true);
    document.removeEventListener("mouseup", onUp, true);
  };
  document.addEventListener("mousemove", onMove, true);
  document.addEventListener("mouseup", onUp, true);
  event.preventDefault();
}

function openDatasetWindow(datasetName) {
  const name = toText(datasetName);
  if (!name) return;
  if (!selectedPath) {
    setStatus("Select a reserving class path before opening a dataset.", true);
    return;
  }

  const title = `${selectedPath}\\${name}`;
  const inst = `pi_ds_${Date.now()}_${windowSeq++}`;
  const frame = document.createElement("section");
  frame.className = "pi-window";
  frame.setAttribute("aria-label", title);
  frame.innerHTML = `
    <header class="pi-window-titlebar">
      <span class="pi-window-title"></span>
      <button class="pi-window-close" type="button" title="Close" aria-label="Close">
        <svg viewBox="0 0 24 24" aria-hidden="true" fill="none" stroke-width="2" stroke-linecap="round">
          <line x1="6" y1="6" x2="18" y2="18"></line>
          <line x1="18" y1="6" x2="6" y2="18"></line>
        </svg>
      </button>
    </header>
    <div class="pi-window-body"></div>
    <div class="pi-window-resize" title="Resize"></div>
  `;

  const titleEl = frame.querySelector(".pi-window-title");
  titleEl.textContent = title;
  titleEl.title = title;

  const body = frame.querySelector(".pi-window-body");
  const iframe = document.createElement("iframe");
  iframe.src = buildDatasetViewerUrl(name, inst);
  iframe.addEventListener("load", () => {
    lockDatasetViewerInputs(iframe, name);
    window.setTimeout(() => lockDatasetViewerInputs(iframe, name), 250);
  });
  body.appendChild(iframe);

  frame.querySelector(".pi-window-titlebar")?.addEventListener("mousedown", (e) => {
    if (e.target.closest("button")) return;
    startMove(frame, e);
  });
  frame.querySelector(".pi-window-resize")?.addEventListener("mousedown", (e) => startResize(frame, e));
  frame.querySelector(".pi-window-close")?.addEventListener("click", () => frame.remove());
  frame.addEventListener("mousedown", () => raiseWindow(frame));

  const bounds = getWindowBounds();
  const offset = ((windowSeq - 1) % 5) * 26;
  els.windowLayer.appendChild(frame);
  applyWindowRect(frame, {
    x: Math.max(12, Math.round((bounds.width - Math.min(920, bounds.width * 0.82)) / 2) + offset),
    y: Math.max(48, Math.round((bounds.height - Math.min(600, bounds.height * 0.78)) / 2) + offset),
    width: Math.min(920, Math.round(bounds.width * 0.82)),
    height: Math.min(600, Math.round(bounds.height * 0.78)),
  });
  raiseWindow(frame);
  setStatus(`Opened ${title}`);
}

function initHeader() {
  if (els.title) els.title.textContent = projectName || "Project";
  const detail = [projectFolder, projectTablePath].filter(Boolean).join(" | ");
  if (els.subtitle) {
    els.subtitle.textContent = detail;
    els.subtitle.title = detail;
  }
}

window.addEventListener("message", (event) => {
  const msg = event.data;
  if (!msg || typeof msg !== "object") return;
  if (msg.type === "arcrho:status" || msg.type === "arcrho:tooltip") {
    try { window.parent.postMessage(msg, "*"); } catch {}
  }
});

async function boot() {
  initHeader();
  if (!projectName) {
    setStatus("Project name is missing.", true);
    setEmptyTable("Project name is missing.");
    if (els.pathTree) els.pathTree.innerHTML = '<div class="ptree-empty">Project name is missing.</div>';
    return;
  }
  await Promise.all([loadPathTree(), loadDatasets()]);
  if (!els.status?.classList.contains("error")) {
    setStatus("Ready");
  }
}

boot();
