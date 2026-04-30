export const FLOAT_MIN_W = 360;
export const FLOAT_MIN_H = 240;
export const FLOAT_DEFAULT_RATIO = 0.8;
export const FLOAT_TITLEBAR_H = 31;
export const FLOAT_TITLEBAR_REACHABLE_H = 36;
export const FLOAT_VERTICAL_THRESHOLD_PX = 36;

export function isFloatingTab(tab) {
  return !!tab && tab.layout === "floating" && tab.id !== "home";
}

export function normalizeFloatRect(raw) {
  if (!raw || typeof raw !== "object") return null;
  const x = Number(raw.x);
  const y = Number(raw.y);
  const width = Number(raw.width);
  const height = Number(raw.height);
  if (![x, y, width, height].every(Number.isFinite)) return null;
  return {
    x,
    y,
    width: Math.max(FLOAT_MIN_W, width),
    height: Math.max(FLOAT_MIN_H, height),
  };
}

export function createFloatingTabsController(deps) {
  const {
    closeTab,
    dockTab,
    ensureIframe,
    getContentElement,
    getFloatingHost,
    getState,
    saveState,
    setActive,
  } = deps;

  let floatPreviewEl = null;

  function getContentRect() {
    const content = getContentElement();
    return content ? content.getBoundingClientRect() : { left: 0, top: 0, width: window.innerWidth, height: window.innerHeight };
  }

  function clampFloatRect(rect) {
    const hostRect = getContentRect();
    const hostW = Math.max(FLOAT_MIN_W, hostRect.width || window.innerWidth || FLOAT_MIN_W);
    const hostH = Math.max(FLOAT_MIN_H, hostRect.height || window.innerHeight || FLOAT_MIN_H);
    const width = Math.max(FLOAT_MIN_W, Math.min(Number(rect?.width) || FLOAT_MIN_W, hostW));
    const height = Math.max(FLOAT_MIN_H, Math.min(Number(rect?.height) || FLOAT_MIN_H, hostH));
    const maxX = Math.max(0, hostW - width);
    const maxY = Math.max(0, hostH - FLOAT_TITLEBAR_REACHABLE_H);
    const x = Math.max(0, Math.min(Number(rect?.x) || 0, maxX));
    const y = Math.max(0, Math.min(Number(rect?.y) || 0, maxY));
    return { x, y, width, height };
  }

  function defaultFloatRectFromPointer(clientX, clientY) {
    const hostRect = getContentRect();
    const width = Math.max(FLOAT_MIN_W, Math.round((hostRect.width || window.innerWidth) * FLOAT_DEFAULT_RATIO));
    const height = Math.max(FLOAT_MIN_H, Math.round((hostRect.height || window.innerHeight) * FLOAT_DEFAULT_RATIO));
    const x = (clientX - hostRect.left) - width / 2;
    const y = (clientY - hostRect.top) - 18;
    return clampFloatRect({ x, y, width, height });
  }

  function ensureFloatPreview() {
    if (floatPreviewEl && floatPreviewEl.isConnected) return floatPreviewEl;
    const content = getContentElement();
    if (!content) return null;
    const el = document.createElement("div");
    el.id = "floatingTabPreview";
    content.appendChild(el);
    floatPreviewEl = el;
    return el;
  }

  function updateFloatPreview(clientX, clientY) {
    const el = ensureFloatPreview();
    if (!el) return;
    const r = defaultFloatRectFromPointer(clientX, clientY);
    el.style.left = `${Math.round(r.x)}px`;
    el.style.top = `${Math.round(r.y)}px`;
    el.style.width = `${Math.round(r.width)}px`;
    el.style.height = `${Math.round(r.height)}px`;
  }

  function removeFloatPreview() {
    if (floatPreviewEl && floatPreviewEl.parentNode) {
      floatPreviewEl.parentNode.removeChild(floatPreviewEl);
    }
    floatPreviewEl = null;
  }

  function getFloatingLayerBase(tab) {
    return 100 + (Number(tab?.floatZ) || 0) * 10;
  }

  function applyFloatingFrameRect(tab, rect) {
    if (!tab) return;
    const minimized = !!tab.floatMinimized;
    const layerBase = getFloatingLayerBase(tab);
    const floatingHost = getFloatingHost();
    const frame = floatingHost?.querySelector?.(`.floatingTabWindow[data-tab-id="${CSS.escape(tab.id)}"]`);
    if (frame) {
      frame.style.left = `${Math.round(rect.x)}px`;
      frame.style.top = `${Math.round(rect.y)}px`;
      frame.style.width = `${Math.round(rect.width)}px`;
      frame.style.height = `${Math.round(minimized ? FLOAT_TITLEBAR_H + 2 : rect.height)}px`;
      frame.style.zIndex = String(layerBase + 1);
    }
    if (tab.iframe) {
      tab.iframe.classList.add("floatingTabIframe");
      tab.iframe.style.position = "absolute";
      tab.iframe.style.display = minimized ? "none" : "block";
      tab.iframe.style.left = `${Math.round(rect.x)}px`;
      tab.iframe.style.top = `${Math.round(rect.y + FLOAT_TITLEBAR_H)}px`;
      tab.iframe.style.width = `${Math.round(rect.width)}px`;
      tab.iframe.style.height = `${Math.max(0, Math.round((minimized ? FLOAT_TITLEBAR_H : rect.height) - FLOAT_TITLEBAR_H))}px`;
      tab.iframe.style.zIndex = String(layerBase);
      tab.iframe.style.background = "#fff";
      tab.iframe.style.pointerEvents = minimized ? "none" : "auto";
    }
  }

  function applyDockedIframeLayout(tab) {
    if (!tab?.iframe) return;
    tab.iframe.classList.remove("floatingTabIframe");
    tab.iframe.style.position = "absolute";
    tab.iframe.style.left = "0";
    tab.iframe.style.top = "0";
    tab.iframe.style.width = "100%";
    tab.iframe.style.height = "100%";
    tab.iframe.style.zIndex = "1";
    tab.iframe.style.pointerEvents = "auto";
  }

  function beginFloatingPointerInteraction(target, pointerId) {
    try { target?.setPointerCapture?.(pointerId); } catch {}
    try { document.body.classList.add("floatingTabDragActive"); } catch {}
  }

  function endFloatingPointerInteraction(target, pointerId) {
    try { target?.releasePointerCapture?.(pointerId); } catch {}
    try { document.body.classList.remove("floatingTabDragActive"); } catch {}
  }

  function startFloatingMove(tabId, e) {
    if (e.button !== 0) return;
    const state = getState();
    const tab = state.tabs.find(t => t.id === tabId);
    if (!tab) return;
    const pointerTarget = e.currentTarget;
    const pointerId = e.pointerId;
    const startX = e.clientX;
    const startY = e.clientY;
    const startRect = clampFloatRect(tab.floatRect || defaultFloatRectFromPointer(e.clientX, e.clientY));
    beginFloatingPointerInteraction(pointerTarget, pointerId);

    const onMove = (ev) => {
      const next = clampFloatRect({
        ...startRect,
        x: startRect.x + (ev.clientX - startX),
        y: startRect.y + (ev.clientY - startY),
      });
      tab.floatRect = next;
      applyFloatingFrameRect(tab, next);
    };

    const finish = () => {
      document.removeEventListener("pointermove", onMove, true);
      document.removeEventListener("pointerup", finish, true);
      document.removeEventListener("pointercancel", finish, true);
      endFloatingPointerInteraction(pointerTarget, pointerId);
      tab.floatRect = clampFloatRect(tab.floatRect || startRect);
      saveState();
    };

    document.addEventListener("pointermove", onMove, true);
    document.addEventListener("pointerup", finish, true);
    document.addEventListener("pointercancel", finish, true);
    e.preventDefault();
  }

  function startFloatingResize(tabId, e, corner = "se") {
    if (e.button !== 0) return;
    const state = getState();
    const tab = state.tabs.find(t => t.id === tabId);
    if (!tab || tab.floatMinimized) return;
    const pointerTarget = e.currentTarget;
    const pointerId = e.pointerId;
    const startX = e.clientX;
    const startY = e.clientY;
    const startRect = clampFloatRect(tab.floatRect || defaultFloatRectFromPointer(e.clientX, e.clientY));
    beginFloatingPointerInteraction(pointerTarget, pointerId);

    const onMove = (ev) => {
      const dx = ev.clientX - startX;
      const dy = ev.clientY - startY;
      let next = { ...startRect };
      if (corner.includes("e")) next.width = startRect.width + dx;
      if (corner.includes("s")) next.height = startRect.height + dy;
      if (corner.includes("w")) {
        next.x = startRect.x + dx;
        next.width = startRect.width - dx;
      }
      if (corner.includes("n")) {
        next.y = startRect.y + dy;
        next.height = startRect.height - dy;
      }
      if (next.width < FLOAT_MIN_W && corner.includes("w")) {
        next.x = startRect.x + startRect.width - FLOAT_MIN_W;
        next.width = FLOAT_MIN_W;
      }
      if (next.height < FLOAT_MIN_H && corner.includes("n")) {
        next.y = startRect.y + startRect.height - FLOAT_MIN_H;
        next.height = FLOAT_MIN_H;
      }
      next = clampFloatRect(next);
      tab.floatRect = next;
      applyFloatingFrameRect(tab, next);
    };

    const finish = () => {
      document.removeEventListener("pointermove", onMove, true);
      document.removeEventListener("pointerup", finish, true);
      document.removeEventListener("pointercancel", finish, true);
      endFloatingPointerInteraction(pointerTarget, pointerId);
      tab.floatRect = clampFloatRect(tab.floatRect || startRect);
      saveState();
    };

    document.addEventListener("pointermove", onMove, true);
    document.addEventListener("pointerup", finish, true);
    document.addEventListener("pointercancel", finish, true);
    e.preventDefault();
  }

  function renderFloatingWindows() {
    const floatingHost = getFloatingHost();
    if (!floatingHost) return;
    const state = getState();
    const floatingIds = new Set(state.tabs.filter(isFloatingTab).map(t => t.id));
    floatingHost.querySelectorAll(".floatingTabWindow").forEach((frame) => {
      const id = frame.getAttribute("data-tab-id");
      if (!floatingIds.has(id)) frame.remove();
    });

    for (const tab of state.tabs) {
      if (!isFloatingTab(tab)) continue;
      ensureIframe(tab);
      if (tab.iframe && tab.iframe.dataset.floatingFocusWired !== "1") {
        tab.iframe.dataset.floatingFocusWired = "1";
        tab.iframe.addEventListener("pointerdown", () => {
          if (isFloatingTab(tab)) setActive(tab.id);
        });
      }
      tab.floatRect = clampFloatRect(tab.floatRect || defaultFloatRectFromPointer(window.innerWidth / 2, window.innerHeight / 2));
      if (!tab.floatZ) tab.floatZ = state.nextFloatZ++;

      let frame = floatingHost.querySelector(`.floatingTabWindow[data-tab-id="${CSS.escape(tab.id)}"]`);
      if (!frame) {
        frame = document.createElement("section");
        frame.className = "floatingTabWindow";
        frame.setAttribute("data-tab-id", tab.id);
        frame.innerHTML = `
          <header class="floatingTabTitlebar">
            <span class="floatingTabTitle"></span>
            <span class="floatingTabDirty" aria-hidden="true"></span>
            <span class="floatingTabControls">
              <button class="titlebarBtn floatingTabButton" data-action="minimize" type="button" title="Minimize" aria-label="Minimize">
                <svg class="titlebarIcon" viewBox="0 0 10 10" aria-hidden="true">
                  <line x1="2" y1="7" x2="8" y2="7"></line>
                </svg>
              </button>
              <button class="titlebarBtn floatingTabButton" data-action="dock" type="button" title="Dock" aria-label="Dock">
                <svg class="titlebarIcon" viewBox="0 0 10 10" aria-hidden="true">
                  <rect x="2" y="2" width="6" height="6" rx="0.6"></rect>
                </svg>
              </button>
              <button class="titlebarBtn floatingTabButton" data-action="close" type="button" title="Close" aria-label="Close">
                <svg class="titlebarIcon" viewBox="0 0 10 10" aria-hidden="true">
                  <line x1="2" y1="2" x2="8" y2="8"></line>
                  <line x1="8" y1="2" x2="2" y2="8"></line>
                </svg>
              </button>
            </span>
          </header>
          <div class="floatingTabBody"></div>
          <div class="floatingTabResizeHandle floatingTabResizeNw" data-corner="nw" title="Resize"></div>
          <div class="floatingTabResizeHandle floatingTabResizeNe" data-corner="ne" title="Resize"></div>
          <div class="floatingTabResizeHandle floatingTabResizeSw" data-corner="sw" title="Resize"></div>
          <div class="floatingTabResizeHandle floatingTabResizeSe" data-corner="se" title="Resize"><span class="resizeIcon"></span></div>
        `;
        frame.addEventListener("pointerdown", () => setActive(tab.id));
        const titlebar = frame.querySelector(".floatingTabTitlebar");
        titlebar?.addEventListener("pointerdown", (e) => {
          if (e.target?.closest?.("button")) return;
          e.stopPropagation();
          setActive(tab.id);
          startFloatingMove(tab.id, e);
        });
        titlebar?.addEventListener("dblclick", (e) => {
          if (e.target?.closest?.("button")) return;
          dockTab(tab.id);
        });
        frame.querySelector('[data-action="minimize"]')?.addEventListener("click", (e) => {
          e.stopPropagation();
          tab.floatMinimized = !tab.floatMinimized;
          setActive(tab.id);
          renderFloatingWindows();
          saveState();
        });
        frame.querySelector('[data-action="dock"]')?.addEventListener("click", (e) => {
          e.stopPropagation();
          dockTab(tab.id);
        });
        frame.querySelector('[data-action="close"]')?.addEventListener("click", (e) => {
          e.stopPropagation();
          closeTab(tab.id);
        });
        frame.querySelectorAll(".floatingTabResizeHandle").forEach((handle) => handle.addEventListener("pointerdown", (e) => {
          e.stopPropagation();
          setActive(tab.id);
          startFloatingResize(tab.id, e, handle.getAttribute("data-corner") || "se");
        }));
        floatingHost.appendChild(frame);
      }

      frame.classList.toggle("active", tab.id === state.activeId);
      frame.classList.toggle("minimized", !!tab.floatMinimized);
      applyFloatingFrameRect(tab, tab.floatRect);
      const title = frame.querySelector(".floatingTabTitle");
      if (title) title.textContent = tab.title || "Untitled";
      const dirty = frame.querySelector(".floatingTabDirty");
      if (dirty) dirty.classList.toggle("show", !!tab.isDirty);
      const minimizeButton = frame.querySelector('[data-action="minimize"]');
      if (minimizeButton) {
        minimizeButton.title = tab.floatMinimized ? "Restore" : "Minimize";
        minimizeButton.setAttribute("aria-label", tab.floatMinimized ? "Restore" : "Minimize");
      }
    }
  }

  function clampFloatingTabsToContent() {
    const state = getState();
    let changed = false;
    for (const tab of state.tabs) {
      if (!isFloatingTab(tab)) continue;
      const next = clampFloatRect(tab.floatRect || defaultFloatRectFromPointer(window.innerWidth / 2, window.innerHeight / 2));
      const prev = tab.floatRect || {};
      if (prev.x !== next.x || prev.y !== next.y || prev.width !== next.width || prev.height !== next.height) {
        tab.floatRect = next;
        changed = true;
      }
    }
    if (changed) {
      renderFloatingWindows();
      saveState();
    }
  }

  return {
    applyDockedIframeLayout,
    clampFloatRect,
    clampFloatingTabsToContent,
    defaultFloatRectFromPointer,
    removeFloatPreview,
    renderFloatingWindows,
    updateFloatPreview,
  };
}
