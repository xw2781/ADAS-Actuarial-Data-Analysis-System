import { $, shell } from "./shell_context.js?v=20260430r";

let titlebarControlsWired = false;
let resizeHandleWired = false;

export function initTitlebarControls() {
  if (titlebarControlsWired) return;
  const api = shell.getHostApi?.();
  const minBtn = $("titlebarMinBtn");
  const maxBtn = $("titlebarMaxBtn");
  const closeBtn = $("titlebarCloseBtn");
  const titlebar = $("customTitlebar");
  if (!minBtn && !maxBtn && !closeBtn && !titlebar) return;
  titlebarControlsWired = true;
  let dragRestoreArmed = false;
  let dragStartX = 0;
  let dragStartY = 0;

  minBtn?.addEventListener("click", (e) => { e.stopPropagation(); api?.minimizeWindow?.(); });
  maxBtn?.addEventListener("click", (e) => {
    e.stopPropagation();
    if (!api?.isMaximized || !api?.restoreWindow) { api?.maximizeWindow?.(); return; }
    api.isMaximized().then((isMax) => { if (isMax) api.restoreWindow?.(); else api.maximizeWindow?.(); }).catch(() => api?.maximizeWindow?.());
  });
  closeBtn?.addEventListener("click", (e) => { e.stopPropagation(); shell.shutdownApplication?.(); });
  titlebar?.addEventListener("mousedown", (e) => {
    const target = e.target;
    if (target?.closest?.(".host-nodrag")) return;
    if (!api?.isMaximized) return;
    api.isMaximized().then((isMax) => { if (!isMax) return; dragRestoreArmed = true; dragStartX = e.clientX; dragStartY = e.clientY; }).catch(() => {});
  });
  window.addEventListener("mousemove", (e) => {
    if (!dragRestoreArmed) return;
    const dx = Math.abs(e.clientX - dragStartX);
    const dy = Math.abs(e.clientY - dragStartY);
    if (dx < 5 && dy < 5) return;
    dragRestoreArmed = false;
    api?.restoreWindow?.();
  });
  window.addEventListener("mouseup", () => { dragRestoreArmed = false; });
  titlebar?.addEventListener("dblclick", (e) => {
    if (!api?.isMaximized || !api?.restoreWindow) return;
    const target = e.target;
    if (target?.closest?.(".host-nodrag")) return;
    api.isMaximized().then((isMax) => { if (isMax) api.restoreWindow?.(); else api.maximizeWindow?.(); }).catch(() => api?.maximizeWindow?.());
  });
}

export function initResizeHandle() {
  if (resizeHandleWired) return;
  const api = shell.getHostApi?.();
  const handle = $("resizeHandle");
  if (!handle || !api?.getWindowSize || !api?.resizeWindow) return;
  resizeHandleWired = true;
  let dragging = false;
  let startX = 0;
  let startY = 0;
  let startW = 0;
  let startH = 0;
  let rafPending = false;
  let nextW = 0;
  let nextH = 0;
  let lastApplyTs = 0;
  const applyResize = () => {
    rafPending = false;
    const now = Date.now();
    if (now - lastApplyTs < 30) return;
    lastApplyTs = now;
    api.resizeWindow(Math.max(820, Math.round(nextW)), Math.max(620, Math.round(nextH)));
  };
  const onMove = (e) => {
    if (!dragging) return;
    nextW = startW + (e.clientX - startX);
    nextH = startH + (e.clientY - startY);
    if (!rafPending) { rafPending = true; requestAnimationFrame(applyResize); }
  };
  const onUp = () => {
    if (!dragging) return;
    dragging = false;
    document.body.classList.remove("is-resizing");
    document.removeEventListener("mousemove", onMove);
    document.removeEventListener("mouseup", onUp);
  };
  handle.addEventListener("mousedown", async (e) => {
    e.preventDefault();
    e.stopPropagation();
    try {
      const size = await api.getWindowSize();
      startW = Number(size?.width || size?.w || 0);
      startH = Number(size?.height || size?.h || 0);
    } catch { return; }
    if (!startW || !startH) return;
    dragging = true;
    startX = e.clientX;
    startY = e.clientY;
    document.body.classList.add("is-resizing");
    document.addEventListener("mousemove", onMove);
    document.addEventListener("mouseup", onUp);
  });
}
