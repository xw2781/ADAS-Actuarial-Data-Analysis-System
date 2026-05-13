import { $, getHostApi, shell } from "./shell_context.js?v=20260510a";

let assistantMessages = [];
let assistantActivities = [];
let assistantDebugLogs = [];
let currentSessionId = "";
let currentSessionTitle = "New ArcBot Chat";
let currentContext = null;
let currentUsage = null;
let currentRequestId = "";
let currentPendingMessageEl = null;
let currentRunStartedAt = 0;
let currentStepStartedAt = 0;
let currentWorkCardEl = null;
let latestSessionList = [];
let assistantMode = "edit";
const ASSISTANT_MODEL = "codex";
const ASSISTANT_LAUNCHER_VISIBLE_KEY = "arcrho_ai_assistant_launcher_visible";
const ASSISTANT_LAUNCHER_POSITION_KEY = "arcrho_ai_assistant_launcher_position";
const ASSISTANT_PANEL_SIZE_KEY = "arcrho_ai_assistant_panel_size";
let assistantReady = false;
let assistantBusy = false;
let assistantStatusChecked = false;
let suppressLauncherClick = false;

function setText(el, text) {
  if (el) el.textContent = text || "";
}

function setStatus(text, tone = "") {
  const el = $("aiAssistantStatus");
  if (!el) return;
  el.classList.toggle("error", tone === "error");
  setText(el, text);
}

function setSetup({ open = false, text = "", install = false, login = false } = {}) {
  const setup = $("aiAssistantSetup");
  const setupText = $("aiAssistantSetupText");
  const installBtn = $("aiAssistantSetupBtn");
  const loginBtn = $("aiAssistantLoginBtn");
  setup?.classList.toggle("open", !!open);
  setText(setupText, text);
  if (installBtn) installBtn.style.display = install ? "inline-block" : "none";
  if (loginBtn) loginBtn.style.display = login ? "inline-block" : "none";
}

function nowIso() {
  return new Date().toISOString();
}

function normalizeMessages(messages) {
  if (!Array.isArray(messages)) return [];
  return messages.map((message) => ({
    role: String(message?.role || "").toLowerCase() === "assistant" ? "assistant"
      : String(message?.role || "").toLowerCase() === "system" ? "system"
      : "user",
    content: String(message?.content || ""),
    timestamp: String(message?.timestamp || nowIso()),
  })).filter((message) => message.content.trim());
}

function normalizeActivities(activities) {
  if (!Array.isArray(activities)) return [];
  return activities.map((activity) => ({
    type: String(activity?.type || "info"),
    text: String(activity?.text || ""),
    elapsedMs: Number.isFinite(activity?.elapsedMs) ? Math.max(0, Math.round(activity.elapsedMs)) : null,
    timestamp: String(activity?.timestamp || nowIso()),
  })).filter((activity) => activity.text.trim()).slice(-120);
}

function normalizeDebugLogs(logs) {
  if (!Array.isArray(logs)) return [];
  return logs.map((entry) => ({
    type: String(entry?.type || "debug"),
    text: String(entry?.text || ""),
    timestamp: String(entry?.timestamp || nowIso()),
  })).filter((entry) => entry.text.trim()).slice(-300);
}

function getSessionPayload() {
  return {
    id: currentSessionId,
    title: currentSessionTitle,
    mode: assistantMode,
    model: ASSISTANT_MODEL,
    messages: assistantMessages,
    activities: assistantActivities,
    debugLogs: assistantDebugLogs,
    context: currentContext,
    usage: currentUsage,
  };
}

async function saveCurrentSession() {
  const host = getHostApi();
  if (!host?.codexAssistantSaveSession || !currentSessionId) return;
  try {
    const result = await host.codexAssistantSaveSession(getSessionPayload());
    if (result?.ok && result.session) {
      currentSessionId = result.session.id || currentSessionId;
      currentSessionTitle = result.session.title || currentSessionTitle;
      updateSessionSelectLabel();
    }
  } catch {
    // Session persistence failures should not block chat.
  }
}

function renderMessages() {
  const container = $("aiAssistantMessages");
  if (!container) return;
  container.textContent = "";
  currentWorkCardEl = null;
  for (const message of assistantMessages) {
    appendMessage(message.role, message.content, { save: false });
  }
  renderActivities();
  renderEmptyHint();
}

function formatElapsed(ms) {
  const value = Math.max(0, Number(ms || 0));
  if (value < 1000) return `${Math.round(value)}ms`;
  return `${(value / 1000).toFixed(value < 10000 ? 1 : 0)}s`;
}

function appendDebugLog(text, type = "debug") {
  const raw = String(text || "").trim();
  if (!raw) return;
  assistantDebugLogs.push({ type, text: raw, timestamp: nowIso() });
  assistantDebugLogs = assistantDebugLogs.slice(-300);
  renderDebugLog();
}

function renderDebugLog() {
  const log = $("aiAssistantDebugLog");
  if (!log) return;
  log.textContent = assistantDebugLogs
    .map((entry) => `[${entry.timestamp}] ${entry.type}: ${entry.text}`)
    .join("\n");
  log.scrollTop = log.scrollHeight;
}

function appendActivity(text, type = "activity", options = {}) {
  const now = performance.now();
  const elapsedMs = Number.isFinite(options.elapsedMs)
    ? options.elapsedMs
    : (currentStepStartedAt ? now - currentStepStartedAt : 0);
  if (currentRunStartedAt) currentStepStartedAt = now;
  const activity = {
    type,
    text: String(text || "").trim(),
    elapsedMs: Math.round(Math.max(0, elapsedMs)),
    timestamp: nowIso(),
  };
  if (!activity.text) return;
  assistantActivities.push(activity);
  assistantActivities = assistantActivities.slice(-120);
  renderActivities();
  if (options.save !== false) saveCurrentSession();
}

function renderActivities() {
  const legacyPanel = $("aiAssistantActivity");
  if (legacyPanel) legacyPanel.classList.remove("open");
  const container = $("aiAssistantMessages");
  if (!container) return;
  const visible = assistantActivities.slice(-8);
  if (!visible.length) {
    currentWorkCardEl?.remove();
    currentWorkCardEl = null;
    return;
  }

  if (!currentWorkCardEl || !currentWorkCardEl.isConnected) {
    currentWorkCardEl = document.createElement("details");
    currentWorkCardEl.className = "aiAssistantWorkCard";
    currentWorkCardEl.open = !!currentRunStartedAt;
    container.appendChild(currentWorkCardEl);
  }

  const totalMs = assistantActivities.reduce((sum, item) => (
    sum + (Number.isFinite(item.elapsedMs) ? Math.max(0, item.elapsedMs) : 0)
  ), 0);
  const isRunning = !!currentRunStartedAt;
  currentWorkCardEl.classList.toggle("running", isRunning);
  currentWorkCardEl.classList.toggle("complete", !isRunning);
  if (isRunning) currentWorkCardEl.open = true;
  else currentWorkCardEl.open = false;

  currentWorkCardEl.textContent = "";
  const summary = document.createElement("summary");
  const title = document.createElement("span");
  title.className = "aiAssistantWorkTitle";
  title.textContent = isRunning ? "Working" : "Worked";
  const meta = document.createElement("span");
  meta.className = "aiAssistantWorkMeta";
  meta.textContent = `${formatElapsed(totalMs)} · ${visible.length} step${visible.length === 1 ? "" : "s"}`;
  summary.append(title, meta);
  currentWorkCardEl.appendChild(summary);

  const list = document.createElement("div");
  list.className = "aiAssistantWorkSteps";
  for (const activity of visible) {
    const row = document.createElement("div");
    row.className = "aiAssistantActivityItem";
    const marker = document.createElement("span");
    marker.className = "aiAssistantActivityMarker";
    const text = document.createElement("span");
    text.className = "aiAssistantActivityText";
    text.textContent = activity.text;
    const time = document.createElement("span");
    time.className = "aiAssistantActivityTime";
    time.textContent = activity.elapsedMs == null ? "" : formatElapsed(activity.elapsedMs);
    row.append(marker, text, time);
    list.appendChild(row);
  }
  currentWorkCardEl.appendChild(list);
  scrollMessagesToBottom();
}

function updateContextPanel() {
  const panel = $("aiAssistantContextPanel");
  if (!panel) return;
  const context = currentContext || {};
  const usage = currentUsage || {};
  const rows = [
    ["Session", currentSessionTitle || currentSessionId || "New ArcBot Chat"],
    ["Tab", context.title || context.tabType || "No active tab context"],
    ["Type", context.tabType || "home"],
    ["JSON", context.targetPath || context.path || "No active JSON"],
    ["Context", usage.promptChars ? `${Number(usage.promptChars).toLocaleString()} chars, ~${Number(usage.estimatedTokens || 0).toLocaleString()} tokens` : "Not measured yet"],
    ["Included", usage.includedMessages != null ? `${usage.includedMessages} messages${usage.truncated ? ", truncated" : ""}` : "Not measured yet"],
  ];
  panel.textContent = "";
  const grid = document.createElement("div");
  grid.className = "aiAssistantContextGrid";
  for (const [label, value] of rows) {
    const labelEl = document.createElement("div");
    labelEl.className = "aiAssistantContextLabel";
    labelEl.textContent = label;
    const valueEl = document.createElement("div");
    valueEl.className = "aiAssistantContextValue";
    valueEl.title = String(value);
    valueEl.textContent = String(value);
    grid.append(labelEl, valueEl);
  }
  panel.appendChild(grid);
}

function updateSessionSelectLabel() {
  const select = $("aiAssistantSessionSelect");
  if (!select || !currentSessionId) return;
  let option = [...select.options].find((item) => item.value === currentSessionId);
  if (!option) {
    option = document.createElement("option");
    option.value = currentSessionId;
    select.prepend(option);
  }
  option.textContent = currentSessionTitle || "ArcBot Chat";
  select.value = currentSessionId;
  updateContextPanel();
}

async function refreshSessionList(selectedId = currentSessionId) {
  const host = getHostApi();
  const select = $("aiAssistantSessionSelect");
  if (!host?.codexAssistantListSessions || !select) return [];
  try {
    const result = await host.codexAssistantListSessions({ includeArchived: false });
    const sessions = result?.ok && Array.isArray(result.sessions) ? result.sessions : [];
    latestSessionList = sessions;
    select.textContent = "";
    for (const session of sessions) {
      const option = document.createElement("option");
      option.value = session.id;
      option.textContent = session.title || "ArcBot Chat";
      select.appendChild(option);
    }
    if (selectedId && [...select.options].some((option) => option.value === selectedId)) {
      select.value = selectedId;
    }
    return sessions;
  } catch {
    return [];
  }
}

async function loadAssistantSession(sessionId) {
  const host = getHostApi();
  if (!host?.codexAssistantLoadSession || !sessionId) return false;
  const result = await host.codexAssistantLoadSession(sessionId);
  if (!result?.ok || !result.session) return false;
  const session = result.session;
  currentSessionId = session.id || "";
  currentSessionTitle = session.title || "ArcBot Chat";
  assistantMode = session.mode === "review" ? "review" : "edit";
  assistantMessages = normalizeMessages(session.messages);
  assistantActivities = normalizeActivities(session.activities);
  assistantDebugLogs = normalizeDebugLogs(session.debugLogs);
  currentContext = session.context || null;
  currentUsage = session.usage || null;
  setAssistantMode(assistantMode, { save: false });
  renderMessages();
  renderActivities();
  renderDebugLog();
  await refreshSessionList(currentSessionId);
  updateSessionSelectLabel();
  return true;
}

async function createAssistantSession() {
  const host = getHostApi();
  if (!host?.codexAssistantCreateSession) return false;
  const result = await host.codexAssistantCreateSession({ mode: assistantMode, model: ASSISTANT_MODEL });
  if (!result?.ok || !result.session) return false;
  currentSessionId = result.session.id;
  currentSessionTitle = result.session.title || "New ArcBot Chat";
  assistantMessages = [];
  assistantActivities = [];
  assistantDebugLogs = [];
  currentContext = null;
  currentUsage = null;
  renderMessages();
  renderActivities();
  renderDebugLog();
  await refreshSessionList(currentSessionId);
  updateSessionSelectLabel();
  return true;
}

async function ensureAssistantSession() {
  const sessions = await refreshSessionList();
  if (sessions.length && await loadAssistantSession(sessions[0].id)) return;
  await createAssistantSession();
}

function formatSessionDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "";
  return date.toLocaleString([], {
    month: "short",
    day: "numeric",
    hour: "numeric",
    minute: "2-digit",
  });
}

async function refreshHistoryPage() {
  const host = getHostApi();
  const list = $("aiAssistantHistoryList");
  if (!host?.codexAssistantListSessions || !list) return;
  list.textContent = "";
  const result = await host.codexAssistantListSessions({ includeArchived: true });
  const sessions = result?.ok && Array.isArray(result.sessions) ? result.sessions : [];
  if (!sessions.length) {
    const empty = document.createElement("div");
    empty.className = "aiAssistantHistoryEmpty";
    empty.textContent = "No chat sessions yet.";
    list.appendChild(empty);
    return;
  }
  for (const session of sessions) {
    const row = document.createElement("div");
    row.className = `aiAssistantHistoryRow${session.archived ? " archived" : ""}`;
    const info = document.createElement("div");
    const name = document.createElement("div");
    name.className = "aiAssistantHistoryName";
    name.textContent = session.title || "ArcBot Chat";
    const meta = document.createElement("div");
    meta.className = "aiAssistantHistoryMeta";
    const updated = formatSessionDate(session.updatedAt);
    meta.textContent = `${session.messageCount || 0} messages${updated ? ` · ${updated}` : ""}${session.archived ? " · Archived" : ""}`;
    info.append(name, meta);

    const actions = document.createElement("div");
    actions.className = "aiAssistantHistoryActions";
    const openBtn = document.createElement("button");
    openBtn.className = "aiAssistantMiniBtn";
    openBtn.type = "button";
    openBtn.textContent = "Open";
    openBtn.addEventListener("click", async () => {
      if (session.archived && host.codexAssistantArchiveSession) {
        await host.codexAssistantArchiveSession(session.id, false);
      }
      await loadAssistantSession(session.id);
      await closeHistoryPage();
    });
    const archiveBtn = document.createElement("button");
    archiveBtn.className = "aiAssistantMiniBtn";
    archiveBtn.type = "button";
    archiveBtn.textContent = session.archived ? "Restore" : "Archive";
    archiveBtn.addEventListener("click", async () => {
      if (!host.codexAssistantArchiveSession) return;
      await host.codexAssistantArchiveSession(session.id, !session.archived);
      if (session.id === currentSessionId && !session.archived) {
        await createAssistantSession();
      }
      await refreshSessionList(currentSessionId);
      await refreshHistoryPage();
    });
    const deleteBtn = document.createElement("button");
    deleteBtn.className = "aiAssistantMiniBtn";
    deleteBtn.type = "button";
    deleteBtn.textContent = "Delete";
    deleteBtn.addEventListener("click", async () => {
      if (!host.codexAssistantDeleteSession) return;
      const confirmed = window.confirm(`Delete this ArcBot chat session?\n\n${session.title || "ArcBot Chat"}`);
      if (!confirmed) return;
      await host.codexAssistantDeleteSession(session.id);
      if (session.id === currentSessionId) {
        await createAssistantSession();
      }
      await refreshSessionList(currentSessionId);
      await refreshHistoryPage();
    });
    actions.append(openBtn, archiveBtn, deleteBtn);
    row.append(info, actions);
    list.appendChild(row);
  }
}

async function openHistoryPage() {
  const panel = $("aiAssistantPanel");
  panel?.classList.add("history-open");
  $("aiAssistantHistoryPage")?.classList.add("open");
  $("aiAssistantHistoryBtn")?.setAttribute("aria-expanded", "true");
  await refreshHistoryPage();
}

async function closeHistoryPage() {
  const panel = $("aiAssistantPanel");
  panel?.classList.remove("history-open");
  $("aiAssistantHistoryPage")?.classList.remove("open");
  $("aiAssistantHistoryBtn")?.setAttribute("aria-expanded", "false");
  await refreshSessionList(currentSessionId);
}

export function isAiAssistantLauncherVisible() {
  try {
    return localStorage.getItem(ASSISTANT_LAUNCHER_VISIBLE_KEY) !== "0";
  } catch {
    return true;
  }
}

export function setAiAssistantLauncherVisible(visible) {
  const show = !!visible;
  try {
    localStorage.setItem(ASSISTANT_LAUNCHER_VISIBLE_KEY, show ? "1" : "0");
  } catch {}
  const launcher = $("aiAssistantLauncher");
  const panel = $("aiAssistantPanel");
  if (launcher) launcher.style.display = show ? "" : "none";
  if (show && launcher) applyLauncherPosition(launcher, loadLauncherPosition(launcher));
  if (!show) {
    launcher?.classList.remove("assistant-open");
    panel?.classList.remove("open");
  } else if (panel?.classList.contains("open")) {
    launcher?.classList.add("assistant-open");
  }
  shell.updateViewMenuState?.();
  return show;
}

export function toggleAiAssistantLauncherVisible() {
  return setAiAssistantLauncherVisible(!isAiAssistantLauncherVisible());
}

function getModeLabel() {
  return assistantMode === "review" ? "Review Mode" : "Edit Mode";
}

function setAssistantMode(mode, options = {}) {
  assistantMode = mode === "review" ? "review" : "edit";
  setText($("aiAssistantModeLabel"), getModeLabel());
  $("aiAssistantReviewModeOption")?.classList.toggle("active", assistantMode === "review");
  $("aiAssistantEditModeOption")?.classList.toggle("active", assistantMode === "edit");
  setStatus(assistantReady ? `Codex ready. ${getModeLabel()}.` : `${getModeLabel()} selected.`);
  if (!assistantMessages.length) renderMessages();
  if (options.save !== false) saveCurrentSession();
}

function setComposerEnabled(enabled) {
  const input = $("aiAssistantInput");
  const sendBtn = $("aiAssistantSendBtn");
  if (input) input.disabled = false;
  if (sendBtn) sendBtn.disabled = !enabled;
}

function autoGrowAssistantInput() {
  const input = $("aiAssistantInput");
  if (!input) return;
  input.style.height = "auto";
  const nextHeight = Math.min(300, Math.max(38, input.scrollHeight));
  input.style.height = `${nextHeight}px`;
  input.style.overflowY = input.scrollHeight > 300 ? "auto" : "hidden";
}

function appendMessage(role, text) {
  const container = $("aiAssistantMessages");
  if (!container) return null;
  const el = document.createElement("div");
  el.className = `aiAssistantMessage ${role}`;
  el.textContent = text || "";
  container.appendChild(el);
  container.scrollTop = container.scrollHeight;
  return el;
}

function scrollMessagesToBottom() {
  const container = $("aiAssistantMessages");
  if (container) container.scrollTop = container.scrollHeight;
}

function renderEmptyHint() {
  const container = $("aiAssistantMessages");
  if (!container || container.children.length) return;
  appendMessage("system", assistantMode === "edit"
    ? "ArcBot is in Edit Mode for JSON files inside the configured Server Connection root."
    : "ArcBot is in Review Mode and cannot edit files.");
}

function applyStatus(status) {
  assistantReady = !!status?.installed && !!status?.authenticated;
  if (!status?.installed) {
    setStatus("Codex CLI is not installed.", "error");
    setSetup({
      open: true,
      install: true,
      login: false,
      text: "Install will run: npm install -g @openai/codex.",
    });
    setComposerEnabled(false);
    return;
  }
  if (!status?.authenticated) {
    setStatus("Codex CLI is installed but not signed in.", "error");
    setSetup({
      open: true,
      install: false,
      login: true,
      text: "Sign in to link this computer to your Codex account.",
    });
    setComposerEnabled(false);
    return;
  }
  setStatus(`Codex ready (${status.version || "installed"}). ${getModeLabel()}.`);
  setSetup({ open: false });
  setComposerEnabled(!assistantBusy);
}

function requestActivePageContext() {
  const activeTab = shell.state?.tabs?.find?.((tab) => tab.id === shell.state.activeId) || null;
  const baseContext = {
    available: false,
    tabId: activeTab?.id || "",
    tabType: activeTab?.type || "home",
    title: activeTab?.title || "",
  };
  const iframe = activeTab?.iframe || null;
  if (!iframe?.contentWindow) return Promise.resolve(baseContext);

  return new Promise((resolve) => {
    const requestId = `arcbot_context_${Date.now()}_${Math.random().toString(36).slice(2)}`;
    let done = false;
    const finish = (value) => {
      if (done) return;
      done = true;
      window.removeEventListener("message", onMessage);
      resolve({ ...baseContext, ...(value || {}), available: !!value?.available });
    };
    const onMessage = (event) => {
      if (event.source !== iframe.contentWindow) return;
      const msg = event.data || {};
      if (msg.type !== "arcrho:assistant-context-result" || msg.requestId !== requestId) return;
      finish(msg.context || {});
    };
    window.addEventListener("message", onMessage);
    try {
      iframe.contentWindow.postMessage({ type: "arcrho:assistant-context-request", requestId }, "*");
    } catch {
      finish(baseContext);
      return;
    }
    setTimeout(() => finish(baseContext), 900);
  });
}

function notifyActivePageJsonUpdated(result) {
  const activeTab = shell.state?.tabs?.find?.((tab) => tab.id === shell.state.activeId) || null;
  const iframe = activeTab?.iframe || null;
  if (!iframe?.contentWindow) return;
  try {
    iframe.contentWindow.postMessage({
      type: "arcrho:assistant-json-updated",
      path: result?.targetPath || "",
    }, "*");
  } catch {
    // ignore stale iframe messaging
  }
}

async function refreshAssistantStatus() {
  const host = getHostApi();
  if (!host?.codexAssistantStatus) {
    setStatus("ArcBot is available in the desktop app only.", "error");
    setSetup({ open: false });
    setComposerEnabled(false);
    return;
  }
  assistantStatusChecked = true;
  setStatus("Checking Codex CLI...");
  setComposerEnabled(false);
  try {
    const status = await host.codexAssistantStatus();
    applyStatus(status);
  } catch (err) {
    assistantReady = false;
    setStatus(String(err?.message || err || "Codex status check failed."), "error");
    setComposerEnabled(false);
  }
}

function openAssistant() {
  $("aiAssistantLauncher")?.classList.add("assistant-open");
  $("aiAssistantPanel")?.classList.add("open");
  renderEmptyHint();
  if (!assistantStatusChecked) refreshAssistantStatus();
  setTimeout(() => $("aiAssistantInput")?.focus(), 0);
}

function closeAssistant() {
  if (isAiAssistantLauncherVisible()) $("aiAssistantLauncher")?.classList.remove("assistant-open");
  $("aiAssistantPanel")?.classList.remove("open");
}

function closeModeMenu() {
  const button = $("aiAssistantModeButton");
  $("aiAssistantModeMenu")?.classList.remove("open");
  button?.setAttribute("aria-expanded", "false");
}

function closeModelMenu() {
  const button = $("aiAssistantModelButton");
  $("aiAssistantModelMenu")?.classList.remove("open");
  button?.setAttribute("aria-expanded", "false");
}

function closeSelectMenus() {
  closeModeMenu();
  closeModelMenu();
}

function toggleModeMenu(forceOpen) {
  const button = $("aiAssistantModeButton");
  const menu = $("aiAssistantModeMenu");
  if (!button || !menu) return;
  const shouldOpen = typeof forceOpen === "boolean" ? forceOpen : !menu.classList.contains("open");
  closeModelMenu();
  menu.classList.toggle("open", shouldOpen);
  button.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
}

function toggleModelMenu(forceOpen) {
  const button = $("aiAssistantModelButton");
  const menu = $("aiAssistantModelMenu");
  if (!button || !menu) return;
  const shouldOpen = typeof forceOpen === "boolean" ? forceOpen : !menu.classList.contains("open");
  closeModeMenu();
  menu.classList.toggle("open", shouldOpen);
  button.setAttribute("aria-expanded", shouldOpen ? "true" : "false");
}

function showUnavailableModel(name) {
  closeModelMenu();
  window.alert(`${name} is not currently available. ArcBot will continue using Codex.`);
}

function getLauncherDefaultPosition(launcher) {
  const rect = launcher.getBoundingClientRect();
  const width = rect.width || 42;
  const height = rect.height || 42;
  const statusbarHeight = Number(shell.getStatusBarHeight?.() || 0);
  return {
    left: window.innerWidth - width - 18,
    top: window.innerHeight - statusbarHeight - height - 18,
  };
}

function readLauncherPosition() {
  try {
    const parsed = JSON.parse(localStorage.getItem(ASSISTANT_LAUNCHER_POSITION_KEY) || "null");
    if (parsed && Number.isFinite(parsed.left) && Number.isFinite(parsed.top)) return parsed;
  } catch {}
  return null;
}

function loadLauncherPosition(launcher) {
  const parsed = readLauncherPosition();
  if (parsed) return adaptLauncherPositionToWindow(launcher, parsed);
  return getLauncherDefaultPosition(launcher);
}

function saveLauncherPosition(launcher, left, top) {
  const tucked = getLauncherTuckedEdges(launcher, left, top);
  const anchor = getLauncherResizeAnchor(launcher, left, top, tucked);
  const payload = {
    left: Math.round(left),
    top: Math.round(top),
    tuckedX: tucked.x,
    tuckedY: tucked.y,
  };
  if (anchor) {
    payload.anchorCornerX = anchor.cornerX;
    payload.anchorCornerY = anchor.cornerY;
    payload.anchorOffsetX = anchor.offsetX;
    payload.anchorOffsetY = anchor.offsetY;
  }
  try {
    localStorage.setItem(ASSISTANT_LAUNCHER_POSITION_KEY, JSON.stringify(payload));
  } catch {}
}

function getLauncherMetrics(launcher) {
  const rect = launcher.getBoundingClientRect();
  const width = rect.width || 42;
  const height = rect.height || 42;
  const statusbarHeight = Number(shell.getStatusBarHeight?.() || 0);
  return {
    width,
    height,
    halfWidth: Math.round(width / 2),
    halfHeight: Math.round(height / 2),
    viewportWidth: Math.max(0, window.innerWidth),
    viewportHeight: Math.max(0, window.innerHeight - statusbarHeight),
  };
}

function getLauncherTuckedEdges(launcher, left, top) {
  const metrics = getLauncherMetrics(launcher);
  return {
    x: left < 0 ? "left" : (left + metrics.width > metrics.viewportWidth ? "right" : ""),
    y: top < 0 ? "top" : (top + metrics.height > metrics.viewportHeight ? "bottom" : ""),
  };
}

function getLauncherResizeAnchor(launcher, left, top, tucked) {
  if (!tucked?.x && !tucked?.y) return null;
  const metrics = getLauncherMetrics(launcher);
  const centerX = left + metrics.halfWidth;
  const centerY = top + metrics.halfHeight;
  const cornerX = tucked.x || (centerX <= metrics.viewportWidth / 2 ? "left" : "right");
  const cornerY = tucked.y || (centerY <= metrics.viewportHeight / 2 ? "top" : "bottom");
  return {
    cornerX,
    cornerY,
    offsetX: Math.round(Math.max(0, cornerX === "right" ? metrics.viewportWidth - centerX : centerX)),
    offsetY: Math.round(Math.max(0, cornerY === "bottom" ? metrics.viewportHeight - centerY : centerY)),
  };
}

function adaptLauncherPositionToWindow(launcher, position) {
  const metrics = getLauncherMetrics(launcher);
  let left = Number(position?.left || 0);
  let top = Number(position?.top || 0);
  const hasAnchor = ["left", "right"].includes(position?.anchorCornerX)
    && ["top", "bottom"].includes(position?.anchorCornerY)
    && Number.isFinite(position?.anchorOffsetX)
    && Number.isFinite(position?.anchorOffsetY);
  if (hasAnchor && (position?.tuckedX || position?.tuckedY)) {
    let centerX = position.anchorCornerX === "right"
      ? metrics.viewportWidth - Math.max(0, Number(position.anchorOffsetX))
      : Math.max(0, Number(position.anchorOffsetX));
    let centerY = position.anchorCornerY === "bottom"
      ? metrics.viewportHeight - Math.max(0, Number(position.anchorOffsetY))
      : Math.max(0, Number(position.anchorOffsetY));
    if (position.tuckedX === "left") centerX = 0;
    else if (position.tuckedX === "right") centerX = metrics.viewportWidth;
    if (position.tuckedY === "top") centerY = 0;
    else if (position.tuckedY === "bottom") centerY = metrics.viewportHeight;
    left = centerX - metrics.halfWidth;
    top = centerY - metrics.halfHeight;
  } else {
    if (position?.tuckedX === "left") left = -metrics.halfWidth;
    else if (position?.tuckedX === "right") left = metrics.viewportWidth - metrics.halfWidth;
    if (position?.tuckedY === "top") top = -metrics.halfHeight;
    else if (position?.tuckedY === "bottom") top = metrics.viewportHeight - metrics.halfHeight;
  }
  return { left, top };
}

function clampLauncherPosition(launcher, left, top, options = {}) {
  const { snap = true } = options;
  const metrics = getLauncherMetrics(launcher);
  const fullMaxLeft = Math.max(0, metrics.viewportWidth - metrics.width);
  const fullMaxTop = Math.max(0, metrics.viewportHeight - metrics.height);
  let nextLeft = Math.min(Math.max(left, -metrics.halfWidth), metrics.viewportWidth - metrics.halfWidth);
  let nextTop = Math.min(Math.max(top, -metrics.halfHeight), metrics.viewportHeight - metrics.halfHeight);
  if (snap) {
    if (nextLeft < 0) nextLeft = -metrics.halfWidth;
    else if (nextLeft > fullMaxLeft) nextLeft = metrics.viewportWidth - metrics.halfWidth;
    if (nextTop < 0) nextTop = -metrics.halfHeight;
    else if (nextTop > fullMaxTop) nextTop = metrics.viewportHeight - metrics.halfHeight;
  }
  return {
    left: nextLeft,
    top: nextTop,
  };
}

function updateLauncherTuckedState(launcher, left, top) {
  const tucked = getLauncherTuckedEdges(launcher, left, top);
  const nearLeft = tucked.x === "left";
  const nearRight = tucked.x === "right";
  const nearTop = tucked.y === "top";
  const nearBottom = tucked.y === "bottom";
  launcher.classList.toggle("tucked", nearLeft || nearRight || nearTop || nearBottom);
  launcher.classList.toggle("tucked-left", nearLeft);
  launcher.classList.toggle("tucked-right", nearRight);
  launcher.classList.toggle("tucked-top", nearTop);
  launcher.classList.toggle("tucked-bottom", nearBottom);
}

function applyLauncherPosition(launcher, position, options = {}) {
  const next = clampLauncherPosition(launcher, Number(position?.left || 0), Number(position?.top || 0), options);
  launcher.style.left = `${Math.round(next.left)}px`;
  launcher.style.top = `${Math.round(next.top)}px`;
  launcher.style.right = "auto";
  launcher.style.bottom = "auto";
  updateLauncherTuckedState(launcher, next.left, next.top);
  return next;
}

function initAssistantLauncherDrag(launcher) {
  let dragState = null;
  launcher.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const rect = launcher.getBoundingClientRect();
    dragState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      left: rect.left,
      top: rect.top,
      moved: false,
    };
    try { launcher.setPointerCapture(event.pointerId); } catch {}
  });

  launcher.addEventListener("pointermove", (event) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    const dx = event.clientX - dragState.startX;
    const dy = event.clientY - dragState.startY;
    if (!dragState.moved && Math.hypot(dx, dy) < 4) return;
    dragState.moved = true;
    launcher.classList.add("dragging");
    applyLauncherPosition(launcher, {
      left: dragState.left + dx,
      top: dragState.top + dy,
    }, { snap: false });
    event.preventDefault();
  });

  const endDrag = (event) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    try { launcher.releasePointerCapture(event.pointerId); } catch {}
    launcher.classList.remove("dragging");
    if (dragState.moved) {
      const rect = launcher.getBoundingClientRect();
      const next = applyLauncherPosition(launcher, { left: rect.left, top: rect.top });
      saveLauncherPosition(launcher, next.left, next.top);
      suppressLauncherClick = true;
      setTimeout(() => { suppressLauncherClick = false; }, 150);
    }
    dragState = null;
  };

  launcher.addEventListener("pointerup", endDrag);
  launcher.addEventListener("pointercancel", endDrag);
  window.addEventListener("resize", () => {
    if (!isAiAssistantLauncherVisible()) return;
    const next = applyLauncherPosition(launcher, loadLauncherPosition(launcher));
    saveLauncherPosition(launcher, next.left, next.top);
  });
}

function clampPanelPosition(panel, left, top) {
  const margin = 8;
  const rect = panel.getBoundingClientRect();
  const width = rect.width || 420;
  const height = rect.height || 420;
  const maxLeft = Math.max(margin, window.innerWidth - width - margin);
  const maxTop = Math.max(margin, window.innerHeight - height - margin);
  return {
    left: Math.min(Math.max(margin, left), maxLeft),
    top: Math.min(Math.max(margin, top), maxTop),
  };
}

function readPanelSize() {
  try {
    const parsed = JSON.parse(localStorage.getItem(ASSISTANT_PANEL_SIZE_KEY) || "null");
    if (parsed && Number.isFinite(parsed.width) && Number.isFinite(parsed.height)) return parsed;
  } catch {}
  return null;
}

function savePanelSize(panel) {
  if (!panel) return;
  const rect = panel.getBoundingClientRect();
  try {
    localStorage.setItem(ASSISTANT_PANEL_SIZE_KEY, JSON.stringify({
      width: Math.round(rect.width),
      height: Math.round(rect.height),
    }));
  } catch {}
}

function clampPanelSize(width, height) {
  const statusbarHeight = Number(shell.getStatusBarHeight?.() || 0);
  const maxWidth = Math.max(360, window.innerWidth - 16);
  const maxHeight = Math.max(340, window.innerHeight - statusbarHeight - 16);
  return {
    width: Math.min(Math.max(360, Number(width) || 480), maxWidth),
    height: Math.min(Math.max(340, Number(height) || 640), maxHeight),
  };
}

function applyPanelSize(panel, size = readPanelSize()) {
  if (!panel || !size) return;
  const next = clampPanelSize(size.width, size.height);
  panel.style.width = `${Math.round(next.width)}px`;
  panel.style.height = `${Math.round(next.height)}px`;
}

function applyPanelPosition(panel, left, top) {
  const next = clampPanelPosition(panel, left, top);
  panel.style.left = `${Math.round(next.left)}px`;
  panel.style.top = `${Math.round(next.top)}px`;
  panel.style.right = "auto";
  panel.style.bottom = "auto";
}

function initAssistantResize(panel) {
  const handle = $("aiAssistantResizeHandle");
  if (!handle) return;
  applyPanelSize(panel);
  let resizeState = null;

  handle.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    const rect = panel.getBoundingClientRect();
    resizeState = {
      pointerId: event.pointerId,
      startX: event.clientX,
      startY: event.clientY,
      width: rect.width,
      height: rect.height,
    };
    try { handle.setPointerCapture(event.pointerId); } catch {}
    event.preventDefault();
  });

  handle.addEventListener("pointermove", (event) => {
    if (!resizeState || resizeState.pointerId !== event.pointerId) return;
    const next = clampPanelSize(
      resizeState.width + event.clientX - resizeState.startX,
      resizeState.height + event.clientY - resizeState.startY,
    );
    applyPanelSize(panel, next);
    const rect = panel.getBoundingClientRect();
    applyPanelPosition(panel, rect.left, rect.top);
  });

  const stopResize = (event) => {
    if (!resizeState || resizeState.pointerId !== event.pointerId) return;
    try { handle.releasePointerCapture(event.pointerId); } catch {}
    savePanelSize(panel);
    resizeState = null;
  };

  handle.addEventListener("pointerup", stopResize);
  handle.addEventListener("pointercancel", stopResize);
}

function initAssistantDrag(panel) {
  const header = $("aiAssistantHeader");
  if (!header) return;
  let dragState = null;

  header.addEventListener("pointerdown", (event) => {
    if (event.button !== 0) return;
    if (event.target?.closest?.("button")) return;
    const rect = panel.getBoundingClientRect();
    dragState = {
      pointerId: event.pointerId,
      offsetX: event.clientX - rect.left,
      offsetY: event.clientY - rect.top,
    };
    try { header.setPointerCapture(event.pointerId); } catch {}
    event.preventDefault();
  });

  header.addEventListener("pointermove", (event) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    applyPanelPosition(panel, event.clientX - dragState.offsetX, event.clientY - dragState.offsetY);
  });

  const stopDrag = (event) => {
    if (!dragState || dragState.pointerId !== event.pointerId) return;
    try { header.releasePointerCapture(event.pointerId); } catch {}
    dragState = null;
  };

  header.addEventListener("pointerup", stopDrag);
  header.addEventListener("pointercancel", stopDrag);
  window.addEventListener("resize", () => {
    if (!panel.classList.contains("open")) return;
    applyPanelSize(panel, {
      width: panel.getBoundingClientRect().width,
      height: panel.getBoundingClientRect().height,
    });
    const rect = panel.getBoundingClientRect();
    applyPanelPosition(panel, rect.left, rect.top);
  });
}

async function installCodexCli() {
  const host = getHostApi();
  if (!host?.codexAssistantInstall) return;
  const confirmed = window.confirm(
    "Install Codex CLI now?\n\nArcRho will run: npm install -g @openai/codex"
  );
  if (!confirmed) return;
  assistantBusy = true;
  setStatus("Installing Codex CLI...");
  setSetup({ open: false });
  setComposerEnabled(false);
  try {
    const result = await host.codexAssistantInstall();
    if (!result?.ok) {
      setStatus(result?.error || "Codex CLI installation failed.", "error");
      setSetup({
        open: true,
        install: true,
        login: false,
        text: "Install will run: npm install -g @openai/codex.",
      });
      return;
    }
    shell.updateStatusBar?.("Codex CLI installed.");
    await refreshAssistantStatus();
  } catch (err) {
    setStatus(String(err?.message || err || "Codex CLI installation failed."), "error");
  } finally {
    assistantBusy = false;
    setComposerEnabled(assistantReady);
  }
}

async function loginCodexCli() {
  const host = getHostApi();
  if (!host?.codexAssistantLogin) return;
  const confirmed = window.confirm(
    "Open Codex sign-in now?\n\nA terminal window will run: codex login"
  );
  if (!confirmed) return;
  try {
    const result = await host.codexAssistantLogin();
    if (!result?.ok) {
      setStatus(result?.error || "Could not start Codex sign-in.", "error");
      return;
    }
    setStatus("Complete Codex sign-in, then refresh status.");
  } catch (err) {
    setStatus(String(err?.message || err || "Could not start Codex sign-in."), "error");
  }
}

async function sendAssistantMessage() {
  if (assistantBusy) return;
  const host = getHostApi();
  const input = $("aiAssistantInput");
  const text = String(input?.value || "").trim();
  if (!text || !host?.codexAssistantSend) return;
  if (!currentSessionId) await ensureAssistantSession();
  if (!assistantReady) {
    setStatus("Install Codex CLI or sign in before sending.", "error");
    return;
  }

  assistantMessages.push({ role: "user", content: text, timestamp: nowIso() });
  appendMessage("user", text);
  currentSessionTitle = assistantMessages.find((message) => message.role === "user")?.content?.slice(0, 42) || currentSessionTitle;
  if (input) input.value = "";
  autoGrowAssistantInput();
  currentRequestId = `arcbot_${Date.now()}_${Math.random().toString(36).slice(2)}`;
  currentRunStartedAt = performance.now();
  currentStepStartedAt = currentRunStartedAt;
  assistantActivities = [];
  assistantDebugLogs = [];
  renderActivities();
  renderDebugLog();
  appendActivity("Request received", "activity", { elapsedMs: 0 });
  const pending = appendMessage("assistant", "...");
  currentPendingMessageEl = pending;
  await saveCurrentSession();

  assistantBusy = true;
  setComposerEnabled(false);
  setStatus("ArcBot is checking the active page context...");
  try {
    const activeContext = await requestActivePageContext();
    currentContext = {
      available: !!activeContext?.available,
      tabType: activeContext?.tabType || "home",
      title: activeContext?.title || "",
      targetPath: activeContext?.targetPath || activeContext?.path || "",
    };
    updateContextPanel();
    setStatus(`Codex is responding in ${getModeLabel()}...`);
    const result = await host.codexAssistantSend({
      requestId: currentRequestId,
      sessionId: currentSessionId,
      mode: assistantMode,
      model: ASSISTANT_MODEL,
      messages: assistantMessages,
      activeContext,
    });
    currentUsage = result?.usage || currentUsage;
    updateContextPanel();
    if (!result?.ok) {
      const message = result?.error || "Codex request failed.";
      if (pending) pending.textContent = message;
      assistantMessages.push({ role: "assistant", content: message, timestamp: nowIso() });
      if (result?.needsAuth) {
        assistantReady = false;
        setStatus("Codex CLI needs sign-in.", "error");
        setSetup({
          open: true,
          install: false,
          login: true,
          text: "Sign in to link this computer to your Codex account.",
        });
      } else {
        setStatus(message, "error");
      }
      appendActivity("Request failed", "error");
      return;
    }
    const reply = String(result?.text || "").trim() || "No response.";
    if (pending) pending.textContent = reply;
    assistantMessages.push({ role: "assistant", content: reply, timestamp: nowIso() });
    if (result?.editApplied) notifyActivePageJsonUpdated(result);
    appendActivity(result?.editApplied ? "Applied JSON edit with host validation." : "Response completed.", "activity");
    setStatus(result?.editApplied ? "ArcBot applied a JSON edit." : `Codex ready. ${getModeLabel()}.`);
  } catch (err) {
    const message = String(err?.message || err || "Codex request failed.");
    if (pending) pending.textContent = message;
    assistantMessages.push({ role: "assistant", content: message, timestamp: nowIso() });
    appendActivity("Request failed", "error");
    setStatus(message, "error");
  } finally {
    currentRequestId = "";
    currentPendingMessageEl = null;
    currentRunStartedAt = 0;
    currentStepStartedAt = 0;
    renderActivities();
    await saveCurrentSession();
    await refreshSessionList(currentSessionId);
    assistantBusy = false;
    setComposerEnabled(assistantReady);
  }
}

function handleAssistantEvent(event) {
  if (!event || event.requestId !== currentRequestId) return;
  if (event.type === "stdout") {
    appendDebugLog(event.text, "stdout");
    return;
  }
  if (event.type === "stderr") {
    appendDebugLog(event.text, "stderr");
    return;
  }
  if (event.type === "usage") {
    currentUsage = event.usage || currentUsage;
    updateContextPanel();
    appendDebugLog(event.text, "usage");
    return;
  }
  if (event.type === "context" && event.context) {
    currentContext = { ...(currentContext || {}), ...event.context };
    updateContextPanel();
    appendActivity("Read active context", "activity");
    appendDebugLog(`${event.text}\n${JSON.stringify(event.context, null, 2)}`, "context");
    return;
  }
  const text = String(event.text || "").trim();
  if (!text) return;
  appendDebugLog(text, event.type || "activity");
  const lower = text.toLowerCase();
  if (lower.includes("checking active json access")) appendActivity("Read active JSON", "activity");
  else if (lower.includes("creating editable local json copy")) appendActivity("Edit started", "activity");
  else if (lower.includes("cleaned explanatory text")) appendActivity("Cleaned edited JSON", "activity");
  else if (lower.includes("validating and applying")) appendActivity("Edit completed", "activity");
  else if (lower.includes("codex response received")) appendActivity("Response received", "activity");
  else if (lower.includes("checking latest arcbot edit history")) appendActivity("Revert started", "activity");
  else if (lower.includes("starting codex cli")) appendActivity("Codex started", "activity");
  else if (lower.includes("resolving arcbot project")) appendActivity("Session prepared", "activity");
}

export function initAiAssistant() {
  const launcher = $("aiAssistantLauncher");
  const panel = $("aiAssistantPanel");
  const composer = $("aiAssistantComposer");
  if (!launcher || !panel || !composer) return;
  const host = getHostApi();
  if (!host) {
    launcher.style.display = "none";
    return;
  }
  setAiAssistantLauncherVisible(isAiAssistantLauncherVisible());
  ensureAssistantSession();
  host.onCodexAssistantEvent?.(handleAssistantEvent);

  launcher.addEventListener("click", (event) => {
    if (suppressLauncherClick) {
      event.preventDefault();
      return;
    }
    if (panel.classList.contains("open")) closeAssistant();
    else openAssistant();
  });
  $("aiAssistantCloseBtn")?.addEventListener("click", closeAssistant);
  $("aiAssistantRefreshBtn")?.addEventListener("click", refreshAssistantStatus);
  $("aiAssistantSetupBtn")?.addEventListener("click", installCodexCli);
  $("aiAssistantLoginBtn")?.addEventListener("click", loginCodexCli);
  $("aiAssistantNewSessionBtn")?.addEventListener("click", () => {
    createAssistantSession();
  });
  $("aiAssistantHistoryBtn")?.addEventListener("click", () => {
    const page = $("aiAssistantHistoryPage");
    if (page?.classList.contains("open")) closeHistoryPage();
    else openHistoryPage();
  });
  $("aiAssistantHistoryCloseBtn")?.addEventListener("click", () => {
    closeHistoryPage();
  });
  $("aiAssistantSessionSelect")?.addEventListener("change", (event) => {
    const sessionId = String(event.target?.value || "");
    if (sessionId && sessionId !== currentSessionId) loadAssistantSession(sessionId);
  });
  $("aiAssistantContextBtn")?.addEventListener("click", () => {
    const panel = $("aiAssistantContextPanel");
    const open = !panel?.classList.contains("open");
    panel?.classList.toggle("open", open);
    $("aiAssistantContextBtn")?.setAttribute("aria-expanded", open ? "true" : "false");
    updateContextPanel();
  });
  $("aiAssistantDebugBtn")?.addEventListener("click", () => {
    const panel = $("aiAssistantDebugPanel");
    const open = !panel?.classList.contains("open");
    panel?.classList.toggle("open", open);
    $("aiAssistantDebugBtn")?.setAttribute("aria-expanded", open ? "true" : "false");
    renderDebugLog();
  });
  $("aiAssistantCopyDebugBtn")?.addEventListener("click", async () => {
    const text = $("aiAssistantDebugLog")?.textContent || "";
    try {
      await navigator.clipboard.writeText(text);
      setStatus("ArcBot debug log copied.");
    } catch {
      setStatus("Could not copy ArcBot debug log.", "error");
    }
  });
  $("aiAssistantModeButton")?.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleModeMenu();
  });
  $("aiAssistantReviewModeOption")?.addEventListener("click", () => {
    closeModeMenu();
    setAssistantMode("review");
  });
  $("aiAssistantEditModeOption")?.addEventListener("click", (event) => {
    event.preventDefault();
    closeModeMenu();
    setAssistantMode("edit");
  });
  $("aiAssistantModelButton")?.addEventListener("click", (event) => {
    event.stopPropagation();
    toggleModelMenu();
  });
  $("aiAssistantCodexModelOption")?.addEventListener("click", () => {
    closeModelMenu();
    setStatus(assistantReady ? `Codex ready. ${getModeLabel()}.` : "Codex selected.");
  });
  $("aiAssistantClaudeModelOption")?.addEventListener("click", () => showUnavailableModel("Claude"));
  $("aiAssistantCopilotModelOption")?.addEventListener("click", () => showUnavailableModel("Copilot"));
  composer.addEventListener("submit", (event) => {
    event.preventDefault();
    sendAssistantMessage();
  });
  $("aiAssistantInput")?.addEventListener("keydown", (event) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      sendAssistantMessage();
    }
  });
  $("aiAssistantInput")?.addEventListener("input", autoGrowAssistantInput);
  autoGrowAssistantInput();

  document.addEventListener("pointerdown", (event) => {
    if (event.target?.closest?.(".aiAssistantSelectWrap")) return;
    closeSelectMenus();
  }, true);
  window.addEventListener("keydown", (event) => {
    if (event.key === "Escape") closeSelectMenus();
  }, true);
  initAssistantDrag(panel);
  initAssistantResize(panel);
  initAssistantLauncherDrag(launcher);
  setComposerEnabled(false);
}
