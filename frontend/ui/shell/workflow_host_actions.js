import { shell } from "./shell_context.js?v=20260430k";

const LAST_WF_PATH_KEY = "arcrho_last_workflow_path_v1";
const WF_IMPORT_HANDLE_KEY = "workflow_import_handle";
const WF_IMPORT_PICKER_ID = "workflow_import_picker";

export function getLastWorkflowPath() {
  try { return localStorage.getItem(LAST_WF_PATH_KEY) || ""; } catch { return ""; }
}

export function setLastWorkflowPath(path) {
  try { localStorage.setItem(LAST_WF_PATH_KEY, path || ""); } catch {}
}

export function getLastWorkflowDir() {
  const p = getLastWorkflowPath();
  if (!p) return "";
  const lastSlash = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  if (lastSlash <= 0) return "";
  return p.slice(0, lastSlash);
}

function pathBasename(p) {
  if (!p) return "";
  const lastSlash = Math.max(p.lastIndexOf("/"), p.lastIndexOf("\\"));
  return lastSlash >= 0 ? p.slice(lastSlash + 1) : p;
}

async function loadWorkflowFromPath(path) {
  const res = await fetch("/workflow/load", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ path }),
  });
  if (!res.ok) throw new Error("Failed to load workflow");
  const out = await res.json();
  const name = pathBasename(out.path || path);
  return { text: JSON.stringify(out.data || {}), name };
}

export function getWorkflowTabState(tab) {
  if (!tab || tab.type !== "workflow") return null;
  try {
    const raw = localStorage.getItem(`arcrho_workflow_state_v1::${tab.wfInst}`);
    if (!raw) return null;
    const s = JSON.parse(raw);
    return s && typeof s === "object" ? s : null;
  } catch {
    return null;
  }
}

function isWorkflowTabEmpty(tab) {
  const s = getWorkflowTabState(tab);
  if (!s || !Array.isArray(s.steps)) return true;
  return s.steps.length === 0;
}

export function postToWorkflowTab(tab, msg) {
  if (!tab || tab.type !== "workflow") return;
  shell.ensureIframe?.(tab);
  const iframe = tab.iframe;
  if (!iframe) return;
  const send = () => {
    try { iframe.contentWindow?.postMessage(msg, "*"); } catch {}
  };
  if (iframe.dataset.ready === "1") {
    send();
    return;
  }
  const onLoad = () => {
    iframe.dataset.ready = "1";
    iframe.removeEventListener("load", onLoad);
    send();
  };
  iframe.addEventListener("load", onLoad);
  send();
}

function openHandleDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open("arcrho_handles", 1);
    req.onupgradeneeded = () => {
      const db = req.result;
      if (!db.objectStoreNames.contains("handles")) db.createObjectStore("handles");
    };
    req.onerror = () => reject(req.error);
    req.onsuccess = () => resolve(req.result);
  });
}

async function getStoredHandle() {
  try {
    const db = await openHandleDb();
    return await new Promise((resolve) => {
      const tx = db.transaction("handles", "readonly");
      const store = tx.objectStore("handles");
      const req = store.get(WF_IMPORT_HANDLE_KEY);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => resolve(null);
    });
  } catch {
    return null;
  }
}

async function getDefaultWorkflowDir() {
  try {
    const res = await fetch("/workflow/default_dir");
    if (!res.ok) return "";
    const out = await res.json();
    return out.path || "";
  } catch {
    return "";
  }
}

async function setStoredHandle(handle) {
  try {
    const db = await openHandleDb();
    await new Promise((resolve) => {
      const tx = db.transaction("handles", "readwrite");
      const store = tx.objectStore("handles");
      store.put(handle, WF_IMPORT_HANDLE_KEY);
      tx.oncomplete = () => resolve(null);
      tx.onerror = () => resolve(null);
    });
  } catch {}
}

async function pickWorkflowFile() {
  const hostApi = shell.getHostApi?.();
  if (hostApi?.pickOpenWorkflowFile) {
    const lastDir = getLastWorkflowDir();
    try {
      const path = await hostApi.pickOpenWorkflowFile(lastDir);
      if (path) return await loadWorkflowFromPath(path);
    } catch {}
  }

  if (window.showOpenFilePicker) {
    const lastDir = getLastWorkflowDir();
    if (lastDir) {
      try {
        const [fileHandle] = await window.showOpenFilePicker({
          types: [{ description: "Workflow", accept: { "application/json": [".arcwf", ".json"] } }],
          multiple: false,
          startIn: lastDir,
          id: WF_IMPORT_PICKER_ID,
        });
        if (!fileHandle) return null;
        await setStoredHandle(fileHandle);
        const file = await fileHandle.getFile();
        const text = await file.text();
        return { text, name: file.name };
      } catch {}
    }
    let startIn = "documents";
    const handle = await getStoredHandle();
    if (handle) {
      startIn = handle;
    } else {
      const defaultDir = await getDefaultWorkflowDir();
      if (defaultDir) {
        try {
          const [fileHandle] = await window.showOpenFilePicker({
            types: [{ description: "Workflow JSON", accept: { "application/json": [".json"] } }],
            multiple: false,
            startIn: defaultDir,
            id: WF_IMPORT_PICKER_ID,
          });
          await setStoredHandle(fileHandle);
          const file = await fileHandle.getFile();
          const text = await file.text();
          return { text, name: file.name };
        } catch {}
      }
    }
    const [fileHandle] = await window.showOpenFilePicker({
      types: [{ description: "Workflow", accept: { "application/json": [".arcwf", ".json"] } }],
      multiple: false,
      startIn,
      id: WF_IMPORT_PICKER_ID,
    });
    if (!fileHandle) return null;
    await setStoredHandle(fileHandle);
    const file = await fileHandle.getFile();
    const text = await file.text();
    return { text, name: file.name };
  }

  return await new Promise((resolve) => {
    const input = document.createElement("input");
    input.type = "file";
    input.accept = ".arcwf,.json,application/json";
    input.addEventListener("change", async () => {
      const file = input.files?.[0];
      if (!file) return resolve(null);
      const text = await file.text();
      resolve({ text, name: file.name });
    });
    input.click();
  });
}

export async function importWorkflow() {
  let picked = null;
  try { picked = await pickWorkflowFile(); } catch { return; }
  if (!picked || !picked.text) return;

  let data = null;
  try { data = JSON.parse(picked.text); } catch {
    alert("Import failed: invalid JSON.");
    return;
  }

  let targetTab = null;
  const active = shell.state?.tabs.find(t => t.id === shell.state.activeId);
  if (active && active.type === "workflow" && isWorkflowTabEmpty(active)) targetTab = active;
  else targetTab = shell.openWorkflowTab?.();
  if (!targetTab) return;
  postToWorkflowTab(targetTab, { type: "arcrho:workflow-load", data });
}

export function clearTestData() {
  const ok = window.confirm("Clear test data (import handle cache)?");
  if (!ok) return;
  try { indexedDB.deleteDatabase("arcrho_handles"); } catch {}
}
