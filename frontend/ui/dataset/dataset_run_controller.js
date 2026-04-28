export function createDatasetRunController(deps) {
  const {
    config,
    state,
    $,
    logLine,
    getDataset,
    patchDataset,
    renderTable,
    renderChart,
    notifyDatasetUpdated,
    isForceRebuildEnabled,
    validateTriInputsBeforeRun,
    getTriInputs,
    buildTriRequestPayload,
    precheckAdasTriCsv,
    clearHeadersCacheForProject,
    ensureHeadersForProject,
    ensureDevHeadersForProject,
    saveLastDsId,
    recordDatasetBrowsingHistory,
    syncNotesForCurrentDataset,
    updateCurrentTabTitle,
    setStatus,
    applyGridSelectionFromState,
    stepId,
  } = deps;

  let autoRunTimer = null;
  let lastAutoKey = "";
  let runInFlight = false;
  let datasetLoadingPopupEl = null;
  let datasetLoadingPopupTimer = null;
  let datasetLoadingPopupStart = 0;

  function ensureDatasetLoadingPopupStyles(doc = document) {
    if (doc.getElementById("adas-load-popup-style")) return;
    const style = doc.createElement("style");
    style.id = "adas-load-popup-style";
    style.textContent = `
      .adas-load-popup-overlay {
        position: fixed;
        inset: 0;
        background: rgba(0, 0, 0, 0.18);
        display: flex;
        align-items: center;
        justify-content: center;
        z-index: 100000;
      }
      .adas-load-popup-card {
        min-width: 340px;
        max-width: min(92vw, 680px);
        border-radius: 10px;
        border: 1px solid #c9d1dc;
        background: #fff;
        box-shadow: 0 20px 44px rgba(15, 23, 42, 0.22);
        padding: 18px 20px 16px;
        color: #0f172a;
        font-family: "Segoe UI", Tahoma, Arial, sans-serif;
      }
      .adas-load-popup-title {
        font-size: 14px;
        font-weight: 600;
        margin-bottom: 8px;
      }
      .adas-load-popup-msg {
        font-size: 13px;
        line-height: 1.35;
        white-space: normal;
        word-break: break-word;
        color: #334155;
      }
      .adas-load-popup-spinner {
        width: 34px;
        height: 34px;
        margin: 11px auto 7px;
        border-radius: 50%;
        position: relative;
      }
      .adas-load-popup-spinner::before {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 50%;
        border: 2px solid rgba(120, 178, 224, 0.24);
        box-shadow:
          inset 0 0 10px rgba(116, 182, 235, 0.14),
          0 0 0 1px rgba(134, 188, 229, 0.1);
      }
      .adas-load-popup-spinner::after {
        content: "";
        position: absolute;
        inset: 0;
        border-radius: 50%;
        background:
          conic-gradient(
            from 220deg,
            rgba(86, 176, 236, 0) 0deg,
            rgba(86, 176, 236, 0) 238deg,
            rgba(134, 224, 255, 0.92) 308deg,
            rgba(74, 144, 217, 0.98) 338deg,
            rgba(74, 144, 217, 0) 360deg
          );
        -webkit-mask: radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 3px));
        mask: radial-gradient(farthest-side, transparent calc(100% - 4px), #000 calc(100% - 3px));
        filter:
          drop-shadow(0 0 6px rgba(95, 196, 255, 0.42))
          drop-shadow(0 0 13px rgba(84, 161, 228, 0.24));
        animation: adas-load-popup-sweep 1.05s linear infinite;
        pointer-events: none;
      }
      @keyframes adas-load-popup-sweep {
        to { transform: rotate(360deg); }
      }
      .adas-load-popup-elapsed {
        margin-top: 10px;
        font-size: 12px;
        color: #64748b;
      }
    `;
    (doc.head || doc.documentElement).appendChild(style);
  }

  function showDatasetLoadingPopup(message = "") {
    const doc = document;
    ensureDatasetLoadingPopupStyles(doc);
    if (!datasetLoadingPopupEl || !datasetLoadingPopupEl.isConnected) {
      const overlay = doc.createElement("div");
      overlay.className = "adas-load-popup-overlay";
      overlay.innerHTML = `
        <div class="adas-load-popup-card" role="alert" aria-live="polite">
          <div class="adas-load-popup-title">Loading Dataset</div>
          <div class="adas-load-popup-msg"></div>
          <div class="adas-load-popup-spinner" aria-hidden="true"></div>
          <div class="adas-load-popup-elapsed">Elapsed: 0.0s</div>
        </div>
      `;
      doc.body.appendChild(overlay);
      datasetLoadingPopupEl = overlay;
    }
    const msgEl = datasetLoadingPopupEl.querySelector(".adas-load-popup-msg");
    if (msgEl) msgEl.textContent = String(message || "Loading...");

    datasetLoadingPopupStart = performance.now();
    if (datasetLoadingPopupTimer) cancelAnimationFrame(datasetLoadingPopupTimer);
    const elapsedEl = datasetLoadingPopupEl.querySelector(".adas-load-popup-elapsed");
    const tick = () => {
      if (!datasetLoadingPopupEl) return;
      const sec = (performance.now() - datasetLoadingPopupStart) / 1000;
      if (elapsedEl) elapsedEl.textContent = `Elapsed: ${sec.toFixed(1)}s`;
      datasetLoadingPopupTimer = requestAnimationFrame(tick);
    };
    datasetLoadingPopupTimer = requestAnimationFrame(tick);
  }

  function hideDatasetLoadingPopup() {
    if (datasetLoadingPopupTimer) {
      cancelAnimationFrame(datasetLoadingPopupTimer);
      datasetLoadingPopupTimer = null;
    }
    if (!datasetLoadingPopupEl) return;
    if (datasetLoadingPopupEl.parentNode) {
      datasetLoadingPopupEl.parentNode.removeChild(datasetLoadingPopupEl);
    }
    datasetLoadingPopupEl = null;
  }

  function scheduleAutoRun(delayMs = 150) {
    if (autoRunTimer) clearTimeout(autoRunTimer);
    autoRunTimer = setTimeout(() => autoRun(), delayMs);
  }

  function bindAutoRunOnEnter(el) {
    if (!el) return;

    el.addEventListener("keydown", (e) => {
      if (e.key !== "Enter") return;

      e.preventDefault();
      el.blur();
      scheduleAutoRun(0);
    });
  }

  async function autoRun() {
    const { project, path, tri, cumulative, originLen, devLen } = getTriInputs();

    if (!project || !path || !tri) return;

    const key = `${project}||${path}||${tri}||${cumulative}||${originLen}||${devLen}`;

    if (key === lastAutoKey) return;

    if (runInFlight) {
      scheduleAutoRun(500);
      return;
    }

    lastAutoKey = key;
    await runAdasTri({ showValidationMessage: false });
  }

  async function runAdasTri(opts = {}) {
    const showValidationMessage = !!opts?.showValidationMessage;
    const clearCacheRequested = !!opts?.clearCache;
    const forceRebuild = isForceRebuildEnabled();
    let clearCache = clearCacheRequested || forceRebuild;
    if (runInFlight) return;
    runInFlight = true;

    const btn = document.getElementById("runAdasTriBtn");
    const clearBtn = document.getElementById("clearCacheReloadBtn");
    const status = document.getElementById("adasTriStatus");
    let validated = null;
    try {
      validated = await validateTriInputsBeforeRun({ showMessage: showValidationMessage });
    } catch (err) {
      console.error("Failed to validate ADASTri inputs:", err);
      runInFlight = false;
      if (showValidationMessage) {
        setStatus("Failed to validate inputs. Please check project/reserving class/dataset values.");
      }
      return;
    }
    if (!validated.ok) {
      runInFlight = false;
      return;
    }
    const forceLocalCsvOnly = !!validated?.dependencyBypassedByExistingCsv;
    if (forceLocalCsvOnly && clearCache) {
      clearCache = false;
      setStatus("Dependencies unresolved: clear-cache refresh disabled; trying local CSV only.");
    }
    const { cumulative, originLen, devLen } = getTriInputs();
    const { project, path, tri } = validated;
    const triRequestInputs = { project, path, tri, cumulative, originLen, devLen };
    const requestPayload = buildTriRequestPayload(triRequestInputs);
    const loadingTarget = String(tri || config.DS_ID || "").trim() || "dataset";
    let loadingPopupVisible = false;
    const showLoadingPopup = () => {
      if (loadingPopupVisible) return;
      showDatasetLoadingPopup(`Loading dataset "${loadingTarget}" ...`);
      loadingPopupVisible = true;
    };
    const hideLoadingPopup = () => {
      if (!loadingPopupVisible) return;
      hideDatasetLoadingPopup();
      loadingPopupVisible = false;
    };

    if (status) {
      status.textContent = clearCache
        ? "Clearing cache and sending request..."
        : "Sending request...";
    }
    if (btn) btn.disabled = true;
    if (clearBtn) clearBtn.disabled = true;
    if (clearCache) {
      showLoadingPopup();
    } else {
      const precheckResult = await precheckAdasTriCsv(triRequestInputs);
      if (precheckResult.ok && precheckResult?.data?.ok && precheckResult.data.need_request === true) {
        // Show ASAP after the app server decides a request must be sent.
        showLoadingPopup();
      } else if (!precheckResult.ok && !precheckResult.skipped) {
        console.warn("ADASTri precheck failed.");
      }
    }

    try {
      const endpoint = clearCache ? "/adas/tri/refresh" : "/adas/tri";
      const resp = await fetch(endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(requestPayload),
      });

      const data = await resp.json();

      if (!resp.ok) {
        logLine(`${clearCache ? "ADASTri refresh" : "ADASTri"} failed: ${resp.status}`);
        if (status) status.textContent = `Error: ${resp.status}`;
        lastAutoKey = null;
        setStatus(`Error: ${resp.status}`);
        return;
      }

      if (!data.ok) {
        logLine(`${clearCache ? "ADASTri refresh" : "ADASTri"} timeout. data_path=${data.data_path}`);
        if (status) status.textContent = "Timeout waiting for csv (try again).";
        lastAutoKey = null;
        setStatus("Timeout waiting for csv (try again).");
        return;
      }

      logLine(`${clearCache ? "ADASTri refresh" : "ADASTri"} OK. ds_id=${data.ds_id}`);
      if (status) status.textContent = `OK: ${data.ds_id}`;

      // switch dataset and load (and persist)
      config.DS_ID = data.ds_id;
      saveLastDsId(config.DS_ID);
      const needRequest = !clearCache && (
        data?.need_request === true
        || !!String(data?.request_file || "").trim()
      );
      if (needRequest) {
        showLoadingPopup();
      } else if (!clearCache) {
        // Cache hit: avoid showing loading popup just for quick local load.
        hideLoadingPopup();
        loadingPopupVisible = false;
      }
      if (clearCache && project) {
        try {
          await clearHeadersCacheForProject(project, { remote: true });
        } catch (err) {
          console.warn("Failed to clear ADASHeaders cache:", err);
        }
        try {
          await ensureHeadersForProject(project, { forceRefresh: true });
          await ensureDevHeadersForProject(project, { forceRefresh: true });
        } catch (err) {
          console.warn("Failed to refresh header labels after cache clear:", err);
        }
      }
      await loadDataset();
      recordDatasetBrowsingHistory({ project, path, tri });
    } finally {
      hideLoadingPopup();
      runInFlight = false;
      if (btn) btn.disabled = false;
      if (clearBtn) clearBtn.disabled = false;
    }
  }

  async function loadDataset() {
    state.dirty.clear();

    const { ok, status, data } = await getDataset(config.DS_ID, config.START_YEAR);

    if (!ok) {
      logLine(`ERROR loading dataset: ${status}`);
      $("tableWrap").innerHTML = `<div style="color:#b00;"><b>Load failed:</b> ${status}</div>`;
      setStatus("Ready");
      return;
    }

    // persist the last successfully loaded dataset
    saveLastDsId(config.DS_ID);

    state.model = data;
    state.fileMtime = data.mtime;

    // Apply cached header labels, if available
    if (Array.isArray(state.headerLabels) && state.headerLabels.length) {
      state.model.origin_labels = state.headerLabels.map(String);
    }
    if (Array.isArray(state.devHeaderLabels) && state.devHeaderLabels.length) {
      // Do not truncate dev labels by the UI selector.
      // The triangle CSV may contain more columns than the current selector value.
      state.model.dev_labels = state.devHeaderLabels.map(String);
    }

    renderTable();
    notifyDatasetUpdated();
    applyGridSelectionFromState();
    if (typeof syncNotesForCurrentDataset === "function") {
      await syncNotesForCurrentDataset();
    }

    $("dsMeta").textContent =
      `id=${data.id} | origins=${data.origin_labels.length} | dev=${data.dev_labels.length} | mtime=${data.mtime}`;

    logLine("Loaded dataset");
    {
      const path = (document.getElementById("pathInput")?.value || "").trim();
      const tri = (document.getElementById("triInput")?.value || "").trim();
      const meta = [path, tri].filter(Boolean).join(" | ");
      setStatus(meta || "Ready");
    }
    const title = updateCurrentTabTitle() || config.DS_ID || "Dataset";

    // In DFM context, step title is managed by DFM method naming logic.
    // Avoid overwriting it with transient dataset ids such as "adastri_*".
    if (stepId && !window.ADA_DFM_CONTEXT) {
      window.parent.postMessage(
        {
          type: "arcrho:update-workflow-step-title",
          stepId: stepId,
          title: title,
        },
        "*"
      );
    }
  }

  async function savePatch() {
    if (state.dirty.size === 0) {
      logLine("No changes to save.");
      return;
    }

    const items = [];
    for (const [key, value] of state.dirty.entries()) {
      const [r, c] = key.split(",").map((x) => parseInt(x, 10));
      items.push({ r, c, value });
    }

    const { status, data } = await patchDataset(items, state.fileMtime, config.DS_ID);

    if (status === 409) {
      logLine("Conflict: file changed on disk. Reload first.");
      return;
    }

    logLine(`Saved patch: applied=${data.applied}, rejected=${(data.rejected || []).length}, new_mtime=${data.mtime}`);
    await loadDataset();
  }

  function toggleBlanks() {
    state.showBlanks = !state.showBlanks;
    $("toggleBlankBtn").textContent = state.showBlanks ? "Hide blanks" : "Show blanks";
    renderTable(); // re-render only, no reload
    notifyDatasetUpdated();
    applyGridSelectionFromState();
  }

  return {
    bindAutoRunOnEnter,
    hideDatasetLoadingPopup,
    isRunInFlight: () => runInFlight,
    loadDataset,
    runAdasTri,
    savePatch,
    scheduleAutoRun,
    showDatasetLoadingPopup,
    toggleBlanks,
  };
}
