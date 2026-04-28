// ---------------------------------------------------------------------------
// Notebook save / open
// ---------------------------------------------------------------------------

function getNotebookFilenameFromPath(pathLike) {
  const normalized = String(pathLike || "").replace(/\\/g, "/").trim();
  if (!normalized) return "";
  const parts = normalized.split("/").filter(Boolean);
  return parts.length ? parts[parts.length - 1] : "";
}

function getNotebookDisplayTitle() {
  return currentNotebookFilename || DEFAULT_NOTEBOOK_TITLE;
}

function updateNotebookTitleUI() {
  const title = getNotebookDisplayTitle();
  if (toolbarNotebookTitleEl) {
    toolbarNotebookTitleEl.textContent = title;
  }
  try {
    window.parent?.postMessage({ type: "arcrho:update-active-tab-title", title }, "*");
  } catch {}
}

function setCurrentNotebookFilename(pathLike) {
  currentNotebookFilename = getNotebookFilenameFromPath(pathLike);
  updateNotebookTitleUI();
}

function openSaveNbDialog(defaultName = "") {
  const overlay = document.getElementById("saveNbOverlay");
  const input = document.getElementById("saveNbName");
  const proposed = String(defaultName || currentNotebookFilename || DEFAULT_NOTEBOOK_FILENAME).trim();
  overlay.classList.add("open");
  input.value = proposed;
  input.focus();
  input.select();
}

function closeSaveNbDialog() {
  document.getElementById("saveNbOverlay").classList.remove("open");
}

function postShellStatus(text) {
  const msg = String(text || "").trim();
  if (!msg) return;
  try {
    window.parent?.postMessage({ type: "arcrho:status", text: msg }, "*");
  } catch {}
}

function getNotebookSavePayload() {
  return cells.map((c) => {
    const entry = {
      type: normalizeCellType(c.type),
      source: c.editor ? c.editor.getValue() : "",
    };
    if (c.executionTimeMs != null) entry.execution_time_ms = Math.round(c.executionTimeMs);
    if (c.execStartTime) entry.exec_start_time = c.execStartTime instanceof Date ? c.execStartTime.toISOString() : c.execStartTime;
    if (c.execEndTime) entry.exec_end_time = c.execEndTime instanceof Date ? c.execEndTime.toISOString() : c.execEndTime;
    return entry;
  });
}

async function saveNotebookByFilename(filename, { closeDialog = true } = {}) {
  const nextName = String(filename || "").trim();
  if (!nextName) {
    setStatus("Enter a filename");
    postShellStatus("Enter a filename");
    return false;
  }

  try {
    const resp = await scriptingFetch("/scripting/save-notebook", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: nextName, cells: getNotebookSavePayload() }),
    });
    const result = await resp.json();
    if (result && result.success === false) {
      const msg = result.message || "Save failed";
      setStatus(msg);
      postShellStatus(msg);
      return false;
    }

    const savedName = getNotebookFilenameFromPath(result?.path) || nextName;
    setCurrentNotebookFilename(savedName);
    const msg = result?.message || `Saved ${savedName}`;
    setStatus(msg);
    postShellStatus(msg);
    if (closeDialog) closeSaveNbDialog();
    return true;
  } catch {
    setStatus("Save failed");
    postShellStatus("Save failed");
    return false;
  }
}

async function confirmSaveNb() {
  const input = document.getElementById("saveNbName");
  const filename = input.value.trim();
  if (!filename) {
    setStatus("Enter a filename");
    postShellStatus("Enter a filename");
    return;
  }
  await saveNotebookByFilename(filename, { closeDialog: true });
}

async function requestNotebookSave(forcePrompt = false) {
  if (forcePrompt || !currentNotebookFilename) {
    postShellStatus("Save As...");
    openSaveNbDialog();
    return;
  }
  await saveNotebookByFilename(currentNotebookFilename, { closeDialog: false });
}

async function openOpenNbDialog() {
  const overlay = document.getElementById("openNbOverlay");
  const list = document.getElementById("openNbList");
  overlay.classList.add("open");
  list.innerHTML = `<li style="color:#bbb;text-align:center">Loading...</li>`;
  try {
    const resp = await scriptingFetch("/scripting/notebooks");
    const notebooks = await resp.json();
    if (notebooks.length === 0) {
      list.innerHTML = `<li style="color:#bbb;text-align:center">No saved notebooks.</li>`;
      return;
    }
    list.innerHTML = "";
    notebooks.forEach((nb) => {
      const li = document.createElement("li");
      li.innerHTML = `<span class="nb-name">${escapeHtml(nb.name)}</span><span class="nb-meta">${escapeHtml(nb.size)}</span>`;
      li.addEventListener("click", () => loadNotebook(nb.name));
      list.appendChild(li);
    });
  } catch {
    list.innerHTML = `<li style="color:#bbb;text-align:center">Failed to load.</li>`;
  }
}

function closeOpenNbDialog() {
  document.getElementById("openNbOverlay").classList.remove("open");
}

async function loadNotebook(filename) {
  try {
    const resp = await scriptingFetch("/scripting/load-notebook", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename }),
    });
    const result = await resp.json();
    if (!result.success) {
      setStatus(result.message || "Load failed");
      return;
    }

    commitPendingEditUndoSnapshot();
    recordNotebookUndoSnapshot();
    setCurrentNotebookFilename(result.path || filename);

    let unsupportedOutputCells = 0;
    const loadedCells = Array.isArray(result.cells) ? result.cells : [];
    withNotebookUndoSuspended(() => {
      discardPendingEditUndoSnapshot();
      editingCellId = null;
      focusedCellId = null;
      selectedCellIds.clear();
      rangeSelectionAnchorId = null;
      resetSectionCollapseState();

      cells.forEach((cell) => {
        if (cell.editor) cell.editor.dispose();
        cell.cellEl.remove();
      });
      cells = [];
      nextCellId = 1;

      loadedCells.forEach((c) => {
        const cell = addCell(c.source || "", null, "after", c.type || "code", { recordUndo: false, persist: false });
        const applied = applyImportedCellState(cell, c);
        if (applied.hasUnsupported) unsupportedOutputCells += 1;
      });

      if (cells.length === 0) {
        addCell("", null, "after", CELL_TYPES.CODE, { recordUndo: false, persist: false });
      }

      focusCell(cells[0]?.id);
      refreshToc();
      saveCellsToStorage();
    });
    renderAllMarkdownCells({ setStatusMessage: false });

    closeOpenNbDialog();
    if (unsupportedOutputCells > 0) {
      setStatus(`Opened ${filename} (${unsupportedOutputCells} cells include unsupported rich outputs)`);
    } else {
      setStatus(`Opened ${filename}`);
    }
  } catch {
    setStatus("Load failed");
  }
}


// ---------------------------------------------------------------------------
// API help
// ---------------------------------------------------------------------------

let apiLoaded = false;

async function ensureApiHelpLoaded() {
  if (!apiList || apiLoaded) return;
  try {
    const resp = await scriptingFetch("/scripting/api-help");
    const funcs = await resp.json();
    apiList.innerHTML = funcs
      .map(
        (f) =>
          `<div><span class="sc-api-fn">${escapeHtml(f.name)}</span><span class="sc-api-desc"> - ${escapeHtml(f.description)}</span></div>`
      )
      .join("");
    apiLoaded = true;
  } catch {
    apiList.textContent = "Failed to load.";
  }
}


// ---------------------------------------------------------------------------
// Persistence (localStorage)
// ---------------------------------------------------------------------------

function saveCellsToStorage() {
  try {
    const payload = cells.map((c) => ({
      type: normalizeCellType(c.type),
      source: c.editor ? c.editor.getValue() : "",
    }));
    localStorage.setItem(CELLS_STORAGE_KEY, JSON.stringify(payload));
  } catch {}
}

function parseStoredCells(raw) {
  if (!raw) return null;
  const parsed = JSON.parse(raw);
  if (!Array.isArray(parsed)) return null;

  // Backward compatibility: older saves used string[] only.
  return parsed.map((entry) => {
    if (typeof entry === "string") {
      return { type: CELL_TYPES.CODE, source: entry };
    }
    if (entry && typeof entry === "object") {
      const source = typeof entry.source === "string"
        ? entry.source
        : (typeof entry.code === "string" ? entry.code : "");
      return { type: normalizeCellType(entry.type), source };
    }
    return { type: CELL_TYPES.CODE, source: "" };
  });
}

function loadCellsFromStorage() {
  const keys = [CELLS_STORAGE_KEY];
  if (CELLS_STORAGE_KEY !== LEGACY_CELLS_STORAGE_KEY) {
    keys.push(LEGACY_CELLS_STORAGE_KEY);
  }

  for (const key of keys) {
    try {
      const normalized = parseStoredCells(localStorage.getItem(key));
      if (normalized && normalized.length > 0) return normalized;
    } catch {
      // ignore malformed storage and continue fallback keys
    }
  }
  return null;
}


// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

function escapeHtml(s) {
  const el = document.createElement("span");
  el.textContent = s;
  return el.innerHTML;
}

function setStatus(text) {
  statusText.textContent = text;
  setTimeout(() => {
    if (statusText.textContent === text) statusText.textContent = "";
  }, 4000);
}
