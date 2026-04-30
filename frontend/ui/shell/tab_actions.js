import { shell } from "./shell_context.js?v=20260430d";
import { isFloatingTab } from "./floating_tabs.js?v=20260430d";
import {
  getLastViewedDatasetInputs,
  normalizeBrowsingHistoryEntry,
} from "/ui/shell/browsing_history.js";

export function setActive(id) {
  const tab = shell.state.tabs.find(t => t.id === id) || shell.state.tabs.find(t => t.id === "home");
  if (!tab) return;
  shell.state.activeId = tab.id;
  if (isFloatingTab(tab)) tab.floatZ = shell.state.nextFloatZ++;
  else shell.state.lastDockedActiveId = tab.id;
  if (tab.type !== "dfm") shell.setDfmEditEnabled?.(false);
  shell.render?.();
  shell.saveState?.();
}

export function setDockedActive(id) {
  shell.state.activeId = id;
  shell.state.lastDockedActiveId = id;
}

export function floatTab(id, rect) {
  const tab = shell.state.tabs.find(t => t.id === id);
  if (!tab || tab.id === "home") return;
  shell.ensureIframe?.(tab);
  tab.layout = "floating";
  tab.floatRect = shell.clampFloatRect?.(rect || shell.defaultFloatRectFromPointer?.(window.innerWidth / 2, window.innerHeight / 2));
  tab.floatZ = shell.state.nextFloatZ++;
  tab.floatMinimized = false;
  shell.state.activeId = tab.id;
  if (shell.state.lastDockedActiveId === tab.id) shell.state.lastDockedActiveId = shell.getFirstDockedTabId?.() || "home";
  shell.ensureActiveTabInvariant?.();
  shell.render?.();
  shell.saveState?.();
}

export function dockTab(id) {
  const tab = shell.state.tabs.find(t => t.id === id);
  if (!tab || tab.id === "home") return;
  tab.layout = "docked";
  tab.floatRect = null;
  tab.floatZ = 0;
  tab.floatMinimized = false;
  setDockedActive(tab.id);
  shell.render?.();
  shell.saveState?.();
}

export function closeTab(id, skipConfirm = false) {
  if (id === "home") return;
  const idx = shell.state.tabs.findIndex(t => t.id === id);
  if (idx < 0) return;
  const tab = shell.state.tabs[idx];
  if (!skipConfirm && tab.isDirty) {
    const confirmed = confirm("This tab has unsaved changes. Are you sure you want to close it?");
    if (!confirmed) return;
  }
  const wasActive = shell.state.activeId === id;
  if (tab.iframe && tab.iframe.parentNode) tab.iframe.parentNode.removeChild(tab.iframe);
  shell.state.tabs.splice(idx, 1);
  if (wasActive) {
    const fallback = shell.state.tabs[Math.max(0, idx - 1)];
    shell.state.activeId = fallback ? fallback.id : "home";
  }
  if (shell.state.lastDockedActiveId === id) shell.state.lastDockedActiveId = shell.getFirstDockedTabId?.() || "home";
  shell.ensureActiveTabInvariant?.();
  shell.render?.();
  shell.saveState?.();
}

export function openDatasetTab(options = {}) {
  const requestedInputs = normalizeBrowsingHistoryEntry(options?.datasetInputs || null);
  const lastViewedInputs = getLastViewedDatasetInputs();
  const datasetInputs = requestedInputs || lastViewedInputs || null;
  const id = `ds_${shell.state.nextId++}`;
  shell.state.tabs.push({
    id,
    title: "Dataset View",
    type: "dataset",
    iframe: null,
    layout: "docked",
    dsInst: `ds_${id}_${Date.now()}`,
    datasetInputs: datasetInputs || undefined,
  });
  setDockedActive(id);
  shell.render?.();
  shell.saveState?.();
}

export function openDFMTab() {
  const id = `dfm_${shell.state.nextId++}`;
  shell.state.tabs.push({
    id,
    title: "DFM",
    type: "dfm",
    iframe: null,
    layout: "docked",
    dsInst: `dfm_${id}_${Date.now()}`,
    dfmTab: "details",
    isDirty: false,
  });
  setDockedActive(id);
  shell.render?.();
  shell.saveState?.();
}

export function openWorkflowTab() {
  const id = `wf_${shell.state.nextId++}`;
  const wfInst = `wf_${shell.state.nextId - 1}_${Date.now()}`;
  const tab = {
    id,
    title: `Workflow ${shell.state.nextId - 1}`,
    type: "workflow",
    iframe: null,
    layout: "docked",
    wfInst,
    isDirty: false,
  };
  shell.state.tabs.push(tab);
  setDockedActive(id);
  shell.render?.();
  shell.saveState?.();
  return tab;
}

export function openProjectSettingsTab() {
  const existing = shell.state.tabs.find(t => t.type === "project_settings");
  if (existing) {
    if (!existing.projectSettingsRibbon) existing.projectSettingsRibbon = "summary";
    setActive(existing.id);
    return;
  }
  const id = `ps_${shell.state.nextId++}`;
  shell.state.tabs.push({
    id,
    title: "Project Explorer",
    type: "project_settings",
    projectSettingsRibbon: "summary",
    iframe: null,
    layout: "docked",
  });
  setDockedActive(id);
  shell.render?.();
  shell.saveState?.();
}

export function openBrowsingHistoryTab() {
  const existing = shell.state.tabs.find(t => t.type === "browsing_history");
  if (existing) {
    setActive(existing.id);
    return;
  }
  const id = `bh_${shell.state.nextId++}`;
  shell.state.tabs.push({ id, title: "Browsing History", type: "browsing_history", iframe: null, layout: "docked" });
  setDockedActive(id);
  shell.render?.();
  shell.saveState?.();
}

export function openScriptingTab(options = {}) {
  const forceNew = !!options?.forceNew;
  const existing = !forceNew ? shell.state.tabs.find(t => t.type === "scripting") : null;
  if (existing) {
    setActive(existing.id);
    return;
  }
  const id = `sc_${shell.state.nextId++}`;
  const scInst = `sc_${shell.state.nextId - 1}_${Date.now()}`;
  shell.state.tabs.push({ id, title: "Untitled Notebook", type: "scripting", scInst, scFresh: true, iframe: null, layout: "docked" });
  setDockedActive(id);
  shell.render?.();
  shell.saveState?.();
}

function removeTabById(id) {
  if (id === "home") return;
  const idx = shell.state.tabs.findIndex(t => t.id === id);
  if (idx < 0) return;
  const tab = shell.state.tabs[idx];
  if (tab.iframe && tab.iframe.parentNode) tab.iframe.parentNode.removeChild(tab.iframe);
  shell.state.tabs.splice(idx, 1);
}

export function closeTabsExcept(keepIds) {
  const keep = new Set(keepIds || []);
  keep.add("home");
  const toRemove = shell.state.tabs.filter(t => !keep.has(t.id));
  const dirtyTabs = toRemove.filter(t => t.isDirty);
  if (dirtyTabs.length > 0) {
    const confirmed = confirm(`${dirtyTabs.length} tab(s) have unsaved changes. Are you sure you want to close them?`);
    if (!confirmed) return;
  }
  toRemove.forEach(t => removeTabById(t.id));
  if (!keep.has(shell.state.activeId)) shell.state.activeId = keepIds && keepIds.length ? keepIds[0] : "home";
  shell.ensureActiveTabInvariant?.();
  shell.render?.();
  shell.saveState?.();
}
