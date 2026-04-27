export function createReservingClassTypesFeature(deps = {}) {
  const {
    reservingClassTypesBody = null,
    reservingClassTypesStatus = null,
    reservingClassTypesRowContextMenu = null,
    reservingClassTypeEditor = null,
    reservingClassTypeEditorHeader = null,
    reservingClassTypeEditorTitle = null,
    rctEditName = null,
    rctEditLevel = null,
    rctEditFormula = null,
    rctEditEexFormula = null,
    initTableColumnResizing = () => {},
    normalizeProjectKey = (name) => String(name || "").trim().toLowerCase(),
    fetchImpl = fetch,
    setStatus = () => {},
    loadAuditLog = async () => {},
    hideContextMenu = () => {},
    hideFolderContextMenu = () => {},
    hideTreeContextMenu = () => {},
    hideDatasetTypesRowContextMenu = () => {},
    scheduleReservingClassTypesAutoSave = () => {},
    positionContextMenu = (menu, x, y) => { menu.style.left = `${x}px`; menu.style.top = `${y}px`; menu.classList.add("show"); },
  } = deps;

  const RESERVING_CLASS_TYPES_COLUMNS = ["Name", "Level", "Formula", "EEX Formula"];
  const RESERVING_CLASS_TYPES_SORTABLE_COLS = new Set(["Name", "Formula", "EEX Formula"]);
  const RESERVING_CLASS_TYPES_SORT_STORAGE_KEY = "adas_ps_rct_sort";
  const RESERVING_CLASS_TYPES_DEFAULT_SORT = { colLabel: "Formula", dir: "asc", explicit: false };
  const reservingClassTypesByProject = new Map();
  const loadedReservingClassTypesByProject = new Set();
  const reservingClassTypesSourceNamesByProject = new Map();
  const reservingClassTypesCollapsedLevelsByProject = new Map();

  let reservingClassTypesContextProject = "";
  let reservingClassTypesContextRowIndex = -1;
  let reservingClassTypesContextCellText = "";
  let rctEditorProject = "";
  let rctEditorRowIndex = -1;
  let rctEditorMode = "edit";
  let rctEditorInsertAfterIndex = -1;
  let rctEditorDragState = null;
  let reservingClassTypesLoadSeq = 0;
  let reservingClassTypesSortState = loadReservingClassTypesSortState();
  let reservingClassTypesValidationTooltip = null;
  let reservingClassTypesValidationTooltipWired = false;

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, "&amp;")
      .replace(/</g, "&lt;")
      .replace(/>/g, "&gt;")
      .replace(/"/g, "&quot;");
  }

  async function copyTextToClipboard(rawText, doc = window.document) {
    const text = String(rawText ?? "");

    try {
      const nav = doc?.defaultView?.navigator || window.navigator;
      if (nav?.clipboard && typeof nav.clipboard.writeText === "function") {
        await nav.clipboard.writeText(text);
        return true;
      }
    } catch {
      // fallback below
    }

    try {
      const ta = doc.createElement("textarea");
      ta.value = text;
      ta.setAttribute("readonly", "readonly");
      ta.style.position = "fixed";
      ta.style.opacity = "0";
      ta.style.pointerEvents = "none";
      ta.style.left = "-9999px";
      ta.style.top = "0";
      doc.body.appendChild(ta);
      ta.focus();
      ta.select();
      const ok = typeof doc.execCommand === "function" ? doc.execCommand("copy") : false;
      if (ta.parentNode) ta.parentNode.removeChild(ta);
      return !!ok;
    } catch {
      return false;
    }
  }

  function setReservingClassTypesStatus(msg, isError = false) {
    if (!reservingClassTypesStatus) return;
    reservingClassTypesStatus.textContent = msg || "";
    reservingClassTypesStatus.classList.toggle("error", !!isError);
  }

  function getProjectReservingClassTypesState(projectName) {
    const key = normalizeProjectKey(projectName);
    if (!reservingClassTypesByProject.has(key)) {
      reservingClassTypesByProject.set(key, {
        columns: [...RESERVING_CLASS_TYPES_COLUMNS],
        rows: [],
      });
    }
    return reservingClassTypesByProject.get(key);
  }

  function getReservingClassTypesSourceSet(projectName) {
    const key = normalizeProjectKey(projectName);
    if (!reservingClassTypesSourceNamesByProject.has(key)) {
      reservingClassTypesSourceNamesByProject.set(key, new Set());
    }
    return reservingClassTypesSourceNamesByProject.get(key);
  }

  function setReservingClassTypesSourceNames(projectName, names) {
    const set = getReservingClassTypesSourceSet(projectName);
    set.clear();
    for (const raw of names || []) {
      const v = String(raw || "").trim();
      if (!v) continue;
      set.add(v.toLowerCase());
    }
  }

  function getReservingClassTypesCollapsedSet(projectName) {
    const key = normalizeProjectKey(projectName);
    if (!reservingClassTypesCollapsedLevelsByProject.has(key)) {
      reservingClassTypesCollapsedLevelsByProject.set(key, new Set());
    }
    return reservingClassTypesCollapsedLevelsByProject.get(key);
  }

  function sanitizeReservingClassTypesSortState(raw) {
    const explicit = !!raw?.explicit;
    const colLabel = RESERVING_CLASS_TYPES_SORTABLE_COLS.has(String(raw?.colLabel || "").trim())
      ? String(raw.colLabel).trim()
      : "";
    const dir = String(raw?.dir || "").trim().toLowerCase();
    const dirNorm = dir === "asc" || dir === "desc" ? dir : "";
    if (colLabel && dirNorm) return { colLabel, dir: dirNorm, explicit: true };
    if (explicit) return { colLabel: "", dir: "", explicit: true };
    return { ...RESERVING_CLASS_TYPES_DEFAULT_SORT };
  }

  function loadReservingClassTypesSortState() {
    try {
      const raw = localStorage.getItem(RESERVING_CLASS_TYPES_SORT_STORAGE_KEY);
      if (!raw) return { ...RESERVING_CLASS_TYPES_DEFAULT_SORT };
      const parsed = JSON.parse(raw);
      return sanitizeReservingClassTypesSortState(parsed);
    } catch {
      return { ...RESERVING_CLASS_TYPES_DEFAULT_SORT };
    }
  }

  function persistReservingClassTypesSortState() {
    try {
      localStorage.setItem(RESERVING_CLASS_TYPES_SORT_STORAGE_KEY, JSON.stringify({
        colLabel: reservingClassTypesSortState.colLabel,
        dir: reservingClassTypesSortState.dir,
        explicit: true,
      }));
    } catch {
      // ignore storage errors
    }
  }

  function getReservingClassTypesSortState() {
    return reservingClassTypesSortState;
  }

  function toggleReservingClassTypesSort(colLabel) {
    const state = reservingClassTypesSortState;
    const target = String(colLabel || "").trim();
    if (!target) return;
    if (state.colLabel !== target) {
      state.colLabel = target;
      state.dir = "asc";
      persistReservingClassTypesSortState();
      return;
    }
    if (state.dir === "asc") {
      state.dir = "desc";
      persistReservingClassTypesSortState();
      return;
    }
    if (state.dir === "desc") {
      state.colLabel = "";
      state.dir = "";
      persistReservingClassTypesSortState();
      return;
    }
    state.dir = "asc";
    persistReservingClassTypesSortState();
  }

  function compareReservingClassTypeSortValues(a, b, dir) {
    const av = String(a ?? "").trim();
    const bv = String(b ?? "").trim();
    const an = Number(av);
    const bn = Number(bv);
    const aNum = av !== "" && Number.isFinite(an);
    const bNum = bv !== "" && Number.isFinite(bn);
    let cmp = 0;
    if (aNum && bNum) cmp = an - bn;
    else cmp = av.localeCompare(bv, undefined, { sensitivity: "base", numeric: true });
    return dir === "desc" ? -cmp : cmp;
  }

  function getReservingClassLevelKey(value) {
    return String(value ?? "").trim();
  }

  function getReservingClassTypeColIndexes(columns) {
    const safe = Array.isArray(columns) && columns.length ? columns : [...RESERVING_CLASS_TYPES_COLUMNS];
    return {
      name: Math.max(0, safe.findIndex((c) => String(c || "").trim().toLowerCase() === "name")),
      level: safe.findIndex((c) => String(c || "").trim().toLowerCase() === "level"),
      formula: safe.findIndex((c) => String(c || "").trim().toLowerCase() === "formula"),
      eexFormula: safe.findIndex((c) => String(c || "").trim().toLowerCase() === "eex formula"),
    };
  }

  function normalizeOperatorSpacing(value) {
    const text = String(value ?? "").trim();
    if (!text) return "";
    const segments = text.match(/"[^"]*"|[^"]+/g) || [];
    let normalizedText = "";
    segments.forEach((segment) => {
      if (segment.startsWith("\"") && segment.endsWith("\"")) {
        normalizedText += segment;
        return;
      }
      let normalized = segment
        .replace(/\s*([+\-*/])\s*/g, " $1 ")
        .replace(/\s+/g, " ");
      if (!normalizedText) normalized = normalized.replace(/^\s+/, "");
      normalizedText += normalized;
    });
    return normalizedText.trim();
  }

  function canonReservingClassTypeName(value) {
    return String(value ?? "").trim().replace(/\s+/g, " ").toLowerCase();
  }

  function sanitizeFileStem(value) {
    return String(value || "").trim().replace(/[\\/:*?"<>|]/g, "_");
  }

  function joinWinPath(...parts) {
    const cleaned = parts
      .map((part) => String(part || "").trim().replace(/[\\/]+/g, "\\"))
      .filter(Boolean);
    return cleaned.join("\\");
  }

  function sanitizeReservingClassTypesRow(rowLike) {
    const out = createEmptyReservingClassTypesRow();
    out[0] = String(rowLike?.[0] ?? "").trim();
    out[1] = String(rowLike?.[1] ?? "").trim();
    out[2] = normalizeOperatorSpacing(rowLike?.[2] ?? "");
    out[3] = normalizeOperatorSpacing(rowLike?.[3] ?? "");
    return out;
  }

  function buildPersistableReservingClassTypesRows(rowList) {
    return (Array.isArray(rowList) ? rowList : [])
      .map((row) => sanitizeReservingClassTypesRow(row))
      .filter((row) => row[0] !== "" || row[1] !== "" || row[2] !== "" || row[3] !== "");
  }

  function ensureReservingClassTypesValidationTooltip() {
    if (reservingClassTypesValidationTooltip) return reservingClassTypesValidationTooltip;
    const el = document.createElement("div");
    el.className = "rct-validation-tooltip";
    el.setAttribute("role", "tooltip");
    el.setAttribute("aria-hidden", "true");
    document.body.appendChild(el);
    reservingClassTypesValidationTooltip = el;
    return el;
  }

  function hideReservingClassTypesValidationTooltip() {
    const tooltip = reservingClassTypesValidationTooltip;
    if (!tooltip) return;
    tooltip.classList.remove("show");
    tooltip.setAttribute("aria-hidden", "true");
    tooltip.textContent = "";
    delete tooltip.dataset.fieldLabel;
  }

  function showReservingClassTypesValidationTooltip(anchor, messageText) {
    if (!anchor || typeof anchor.getBoundingClientRect !== "function") return;
    const tooltip = ensureReservingClassTypesValidationTooltip();
    const text = String(messageText || "").trim();
    if (!text) return;
    tooltip.textContent = text;
    tooltip.classList.add("show");
    tooltip.setAttribute("aria-hidden", "false");

    const rect = anchor.getBoundingClientRect();
    const margin = 8;
    const maxWidth = Math.min(360, Math.max(220, window.innerWidth - 24));
    tooltip.style.maxWidth = `${maxWidth}px`;
    tooltip.style.left = "0px";
    tooltip.style.top = "0px";

    const tipRect = tooltip.getBoundingClientRect();
    let left = rect.left;
    if (left + tipRect.width > window.innerWidth - 12) {
      left = window.innerWidth - 12 - tipRect.width;
    }
    left = Math.max(12, left);

    let top = rect.top - tipRect.height - margin;
    if (top < 12) {
      top = rect.bottom + margin;
    }
    tooltip.style.left = `${left}px`;
    tooltip.style.top = `${top}px`;
  }

  function ensureReservingClassTypesValidationTooltipWiring() {
    if (reservingClassTypesValidationTooltipWired) return;
    reservingClassTypesValidationTooltipWired = true;
    rctEditFormula?.addEventListener("input", () => hideReservingClassTypesValidationTooltip());
    rctEditEexFormula?.addEventListener("input", () => hideReservingClassTypesValidationTooltip());
  }

  function uniqueReservingClassTypeNames(names) {
    const out = [];
    const seen = new Set();
    for (const raw of Array.isArray(names) ? names : []) {
      const value = String(raw ?? "").trim();
      const key = canonReservingClassTypeName(value);
      if (!value || !key || seen.has(key)) continue;
      seen.add(key);
      out.push(value);
    }
    return out;
  }

  function uniqueExactStrings(values) {
    const out = [];
    const seen = new Set();
    for (const raw of Array.isArray(values) ? values : []) {
      const value = String(raw ?? "");
      if (!value || seen.has(value)) continue;
      seen.add(value);
      out.push(value);
    }
    return out;
  }

  function escapeRegExp(value) {
    return String(value || "").replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
  }

  function reservingFormulaNameRequiresQuotes(value) {
    return /[+\-*/]/.test(String(value ?? ""));
  }

  function extractFormulaComponentRefs(formula, knownNames = []) {
    const text = String(formula ?? "").trim();
    if (!text) return [];

    const out = [];
    const quotedSeen = new Set();
    const used = [];
    for (const match of Array.from(text.matchAll(/"([^"]*)"/g))) {
      const value = String(match?.[1] ?? "");
      const start = Number(match?.index ?? -1);
      const end = start >= 0 ? start + String(match?.[0] ?? "").length : -1;
      if (start >= 0 && end >= start) used.push({ start, end });
      if (!value || quotedSeen.has(value)) continue;
      quotedSeen.add(value);
      out.push({ name: value, quoted: true });
    }

    const uniqueNames = uniqueReservingClassTypeNames(knownNames).sort((a, b) => b.length - a.length);
    const matches = [];
    for (const name of uniqueNames) {
      const pattern = new RegExp(`(?<![A-Za-z0-9_])${escapeRegExp(name)}(?![A-Za-z0-9_])`, "gi");
      let match;
      while ((match = pattern.exec(text)) !== null) {
        matches.push({ start: match.index, end: match.index + match[0].length, name });
      }
    }

    const seenKeys = new Set();
    let residual = text;
    if (used.length || matches.length) {
      const chars = Array.from(text);
      for (const entry of used) {
        for (let i = entry.start; i < entry.end && i < chars.length; i += 1) chars[i] = " ";
      }
      matches.sort((a, b) => (a.start - b.start) || ((b.end - b.start) - (a.end - a.start)));
      for (const match of matches) {
        const overlap = used.some((entry) => match.start < entry.end && match.end > entry.start);
        if (overlap) continue;
        used.push({ start: match.start, end: match.end });
        const key = canonReservingClassTypeName(match.name);
        if (!key || seenKeys.has(key)) continue;
        seenKeys.add(key);
        out.push({ name: match.name, quoted: false });
        for (let i = match.start; i < match.end && i < chars.length; i += 1) chars[i] = " ";
      }
      residual = chars.join("");
    }

    const tokenParts = residual
      .replace(/[()]/g, " ")
      .split(/[+\-*/]/)
      .map((part) => String(part ?? "").trim())
      .filter(Boolean)
      .filter((part) => !/^\d+(\.\d+)?$/.test(part));
    for (const token of tokenParts) {
      const key = canonReservingClassTypeName(token);
      if (!key || seenKeys.has(key)) continue;
      seenKeys.add(key);
      out.push({ name: token, quoted: false });
    }
    return out;
  }

  function collectInvalidFormulaReferences(projectName, rows, columns, options = {}) {
    const idx = getReservingClassTypeColIndexes(columns);
    const nameIdx = idx.name >= 0 ? idx.name : 0;
    const formulaIdx = idx.formula >= 0 ? idx.formula : 2;
    const eexIdx = idx.eexFormula >= 0 ? idx.eexFormula : 3;
    const allKnownNames = (Array.isArray(rows) ? rows : [])
      .map((row) => String(row?.[nameIdx] ?? "").trim())
      .filter(Boolean);
    const knownNames = uniqueReservingClassTypeNames(allKnownNames);
    const knownExactNames = new Set(allKnownNames);
    const knownKeys = new Set(knownNames.map((name) => canonReservingClassTypeName(name)).filter(Boolean));
    const targetRowIndexes = Array.isArray(options?.rowIndexes)
      ? new Set(options.rowIndexes.filter((value) => Number.isInteger(value) && value >= 0))
      : null;
    const issues = [];

    for (let rowIndex = 0; rowIndex < (Array.isArray(rows) ? rows.length : 0); rowIndex += 1) {
      if (targetRowIndexes && !targetRowIndexes.has(rowIndex)) continue;
      const row = rows[rowIndex];
      const rowName = String(row?.[nameIdx] ?? "").trim() || `Row ${rowIndex + 1}`;
      for (const field of [
        { label: "Formula", idx: formulaIdx },
        { label: "EEX Formula", idx: eexIdx },
      ]) {
        const formula = normalizeOperatorSpacing(row?.[field.idx] ?? "");
        if (!formula) continue;
        const components = extractFormulaComponentRefs(formula, knownNames);
        const invalid = [];
        for (const component of components) {
          const value = String(component?.name ?? "");
          if (!value) continue;
          if (!component?.quoted && reservingFormulaNameRequiresQuotes(value)) {
            invalid.push(value);
            continue;
          }
          const isValid = component?.quoted
            ? knownExactNames.has(value)
            : knownKeys.has(canonReservingClassTypeName(value));
          if (isValid) continue;
          invalid.push(value);
        }
        if (!invalid.length) continue;
        issues.push({
          rowIndex,
          rowName,
          fieldLabel: field.label,
          invalidComponents: uniqueExactStrings(invalid),
        });
      }
    }

    return issues;
  }

  function buildInvalidFormulaWarningMessage(projectName, issues) {
    const issue = Array.isArray(issues) && issues.length ? issues[0] : null;
    if (!issue) return "Invalid component(s) in formula.";
    const fieldLabel = String(issue.fieldLabel || "Formula").trim().toLowerCase();
    return `Invalid component(s) in ${fieldLabel}: ${issue.invalidComponents.join(", ")}. Check spacing and spelling. Names containing +, -, *, or / must be quoted.`;
  }

  function validateReservingClassTypesRows(projectName, rows, columns, options = {}) {
    const issues = collectInvalidFormulaReferences(projectName, rows, columns, options);
    if (!issues.length) return { ok: true, issues: [] };
    setReservingClassTypesStatus("Formula can only reference existing reserving class types; names containing +, -, *, or / must be quoted.", true);
    if (options.showTooltip !== false) {
      const firstIssue = issues[0] || null;
      const anchor = firstIssue?.fieldLabel === "EEX Formula" ? options.eexAnchorElement : options.formulaAnchorElement;
      showReservingClassTypesValidationTooltip(anchor, buildInvalidFormulaWarningMessage(projectName, issues));
    }
    return { ok: false, issues };
  }

  function normalizeLocalReservingClassTypesPayload(raw) {
    const fallback = { columns: [...RESERVING_CLASS_TYPES_COLUMNS], rows: [] };
    if (!raw || typeof raw !== "object") return fallback;

    const rawColumns = Array.isArray(raw.columns) ? raw.columns : [];
    const columnIndexByName = new Map();
    for (let i = 0; i < rawColumns.length; i += 1) {
      const key = String(rawColumns[i] ?? "").trim().toLowerCase();
      if (!key || columnIndexByName.has(key)) continue;
      columnIndexByName.set(key, i);
    }

    const rawRows = Array.isArray(raw.rows) ? raw.rows : [];
    const rows = [];
    for (const rawRow of rawRows) {
      if (!Array.isArray(rawRow)) continue;
      const pickValue = (name, fallbackIndex) => {
        const idx = columnIndexByName.has(name) ? columnIndexByName.get(name) : fallbackIndex;
        return String(rawRow?.[idx] ?? "").trim();
      };
      const row = sanitizeReservingClassTypesRow([
        pickValue("name", 0),
        pickValue("level", 1),
        pickValue("formula", 2),
        pickValue("eex formula", 3),
      ]);
      if (row[0] !== "" || row[1] !== "" || row[2] !== "" || row[3] !== "") {
        rows.push(row);
      }
    }

    return { columns: [...RESERVING_CLASS_TYPES_COLUMNS], rows };
  }

  function closeReservingClassTypeEditor() {
    hideReservingClassTypesValidationTooltip();
    if (!reservingClassTypeEditor) return;
    reservingClassTypeEditor.classList.remove("show");
    reservingClassTypeEditor.style.left = "";
    reservingClassTypeEditor.style.top = "";
    reservingClassTypeEditor.style.transform = "translateX(-50%)";
    rctEditorProject = "";
    rctEditorRowIndex = -1;
    rctEditorMode = "edit";
    rctEditorInsertAfterIndex = -1;
    rctEditorDragState = null;
  }

  function openReservingClassTypeEditor(projectName, rowIndex, options = {}) {
    hideReservingClassTypesValidationTooltip();
    ensureReservingClassTypesValidationTooltipWiring();
    if (!reservingClassTypeEditor || !projectName) return;
    const mode = String(options?.mode || "edit").toLowerCase() === "add" ? "add" : "edit";
    const state = getProjectReservingClassTypesState(projectName);
    const columns = Array.isArray(state.columns) && state.columns.length ? state.columns : [...RESERVING_CLASS_TYPES_COLUMNS];
    if (!Array.isArray(state.rows)) return;
    if (mode === "edit" && (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= state.rows.length)) return;
    const idx = getReservingClassTypeColIndexes(columns);
    const row = mode === "edit" ? state.rows[rowIndex] : null;
    const rowName = mode === "edit" ? String(row?.[idx.name] ?? "").trim().toLowerCase() : "";
    const sourceSet = getReservingClassTypesSourceSet(projectName);
    const isSourceDerived = mode === "edit" ? (!!rowName && sourceSet.has(rowName)) : false;
    const defaultLevel = (() => {
      if (options && Object.prototype.hasOwnProperty.call(options, "seedLevel")) {
        return String(options.seedLevel ?? "").trim();
      }
      if (mode === "edit") {
        const levelIdx = idx.level >= 0 ? idx.level : 1;
        return String(row?.[levelIdx] ?? "");
      }
      if (Number.isInteger(rowIndex) && rowIndex >= 0 && rowIndex < state.rows.length) {
        const levelIdx = idx.level >= 0 ? idx.level : 1;
        return String(state.rows[rowIndex]?.[levelIdx] ?? "");
      }
      return "";
    })();

    if (reservingClassTypeEditorTitle) {
      reservingClassTypeEditorTitle.textContent = mode === "add"
        ? "Add User Defined Reserving Class Type"
        : `Edit Reserving Class Type (Row ${rowIndex + 1})`;
    }
    if (rctEditName) {
      rctEditName.value = mode === "add" ? "" : String(row?.[idx.name] ?? "");
    }
    if (rctEditLevel) {
      rctEditLevel.value = defaultLevel;
    }
    if (rctEditFormula) {
      const formulaIdx = idx.formula >= 0 ? idx.formula : 2;
      rctEditFormula.value = mode === "add" ? "" : String(row?.[formulaIdx] ?? "");
      rctEditFormula.disabled = isSourceDerived;
    }
    if (rctEditEexFormula) {
      const eexIdx = idx.eexFormula >= 0 ? idx.eexFormula : 3;
      rctEditEexFormula.value = mode === "add" ? "" : String(row?.[eexIdx] ?? "");
      rctEditEexFormula.disabled = isSourceDerived;
    }

    rctEditorProject = projectName;
    rctEditorMode = mode;
    rctEditorRowIndex = mode === "edit" ? rowIndex : -1;
    rctEditorInsertAfterIndex = Number.isInteger(options?.insertAfterIndex)
      ? options.insertAfterIndex
      : (Number.isInteger(rowIndex) ? rowIndex : -1);
    reservingClassTypeEditor.style.transform = "translateX(-50%)";
    reservingClassTypeEditor.style.left = "50%";
    reservingClassTypeEditor.style.top = "140px";
    reservingClassTypeEditor.classList.add("show");
    setTimeout(() => {
      if (rctEditName && !rctEditName.disabled) rctEditName.focus();
      else if (rctEditLevel && !rctEditLevel.disabled) rctEditLevel.focus();
    }, 0);
  }

  function applyReservingClassTypeEditor() {
    const projectName = rctEditorProject;
    const mode = rctEditorMode;
    const rowIndex = rctEditorRowIndex;
    if (!projectName) return;

    const state = getProjectReservingClassTypesState(projectName);
    const columns = Array.isArray(state.columns) && state.columns.length ? state.columns : [...RESERVING_CLASS_TYPES_COLUMNS];
    const idx = getReservingClassTypeColIndexes(columns);
    if (!Array.isArray(state.rows)) return;

    const nameValue = String(rctEditName?.value ?? "").trim();
    const levelValue = String(rctEditLevel?.value ?? "").trim();
    const formulaValue = normalizeOperatorSpacing(rctEditFormula?.value ?? "");
    const eexValue = normalizeOperatorSpacing(rctEditEexFormula?.value ?? "");
    hideReservingClassTypesValidationTooltip();
    if (!nameValue) {
      setReservingClassTypesStatus("Name is required.", true);
      if (rctEditName) rctEditName.focus();
      return;
    }
    if (mode === "add" && !formulaValue) {
      setReservingClassTypesStatus("Formula is required for Add (user defined type).", true);
      if (rctEditFormula) rctEditFormula.focus();
      return;
    }

    let targetRowIndex = rowIndex;
    let target = null;
    if (mode === "edit") {
      if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= state.rows.length) return;
      const row = state.rows[rowIndex];
      if (!Array.isArray(row)) return;
      if (row.length < columns.length) {
        const fallback = createEmptyReservingClassTypesRow(columns.length);
        for (let i = 0; i < row.length && i < columns.length; i++) fallback[i] = row[i];
        state.rows[rowIndex] = fallback;
      }
      target = state.rows[rowIndex];
    } else {
      const key = canonReservingClassTypeName(nameValue);
      const existingIdx = state.rows.findIndex((r) => {
        if (!Array.isArray(r)) return false;
        const n = String(r[idx.name] ?? "").trim();
        return canonReservingClassTypeName(n) === key;
      });
      if (existingIdx >= 0) {
        targetRowIndex = existingIdx;
        if (state.rows[existingIdx].length < columns.length) {
          const fallback = createEmptyReservingClassTypesRow(columns.length);
          for (let i = 0; i < state.rows[existingIdx].length && i < columns.length; i++) fallback[i] = state.rows[existingIdx][i];
          state.rows[existingIdx] = fallback;
        }
        target = state.rows[existingIdx];
      } else {
        const newRow = createEmptyReservingClassTypesRow(columns.length);
        const after = Number.isInteger(rctEditorInsertAfterIndex) ? rctEditorInsertAfterIndex : -1;
        const insertPos = Math.max(0, Math.min(state.rows.length, after + 1));
        state.rows.splice(insertPos, 0, newRow);
        targetRowIndex = insertPos;
        target = state.rows[insertPos];
      }
    }
    if (!Array.isArray(target)) return;

    const formulaIdx = idx.formula >= 0 ? idx.formula : 2;
    const eexIdx = idx.eexFormula >= 0 ? idx.eexFormula : 3;
    const nextRow = target.slice();
    nextRow[idx.name] = nameValue;
    if (idx.level >= 0) {
      nextRow[idx.level] = levelValue;
    } else if (nextRow.length > 1) {
      nextRow[1] = levelValue;
    }
    if (!(rctEditFormula?.disabled)) nextRow[formulaIdx] = formulaValue;
    if (!(rctEditEexFormula?.disabled)) nextRow[eexIdx] = eexValue;

    const nextRows = state.rows.map((row, currentIndex) => (currentIndex === targetRowIndex ? nextRow : row));
    const validation = validateReservingClassTypesRows(projectName, nextRows, columns, {
      rowIndexes: [targetRowIndex],
      formulaAnchorElement: rctEditFormula,
      eexAnchorElement: rctEditEexFormula,
    });
    if (!validation.ok) {
      const firstIssue = validation.issues[0] || null;
      if (firstIssue?.fieldLabel === "EEX Formula") rctEditEexFormula?.focus();
      else rctEditFormula?.focus();
      if (mode === "add" && targetRowIndex >= 0 && targetRowIndex < state.rows.length && target === state.rows[targetRowIndex]) {
        state.rows.splice(targetRowIndex, 1);
      }
      return;
    }

    for (let i = 0; i < nextRow.length; i += 1) target[i] = nextRow[i];
    if (!(rctEditFormula?.disabled) && rctEditFormula) rctEditFormula.value = formulaValue;
    if (!(rctEditEexFormula?.disabled) && rctEditEexFormula) rctEditEexFormula.value = eexValue;

    renderReservingClassTypesTable(projectName);
    setReservingClassTypesStatus("");
    scheduleReservingClassTypesAutoSave(projectName);
    closeReservingClassTypeEditor();
  }

  function createEmptyReservingClassTypesRow(columnCount = RESERVING_CLASS_TYPES_COLUMNS.length) {
    return Array.from({ length: Math.max(1, Number(columnCount) || RESERVING_CLASS_TYPES_COLUMNS.length) }, () => "");
  }

  function normalizeReservingClassTypesPayload(payload) {
    const fallback = {
      columns: [...RESERVING_CLASS_TYPES_COLUMNS],
      rows: [],
    };
    if (!payload || typeof payload !== "object") return fallback;

    const rawColumns = Array.isArray(payload.columns) ? payload.columns : RESERVING_CLASS_TYPES_COLUMNS;
    const columns = rawColumns.map((v) => String(v ?? "").trim()).filter(Boolean);
    const effectiveColumns = columns.length ? columns : [...RESERVING_CLASS_TYPES_COLUMNS];

    const rawRows = Array.isArray(payload.rows) ? payload.rows : [];
    const rows = [];
    for (const rawRow of rawRows) {
      if (!Array.isArray(rawRow)) continue;
      const row = createEmptyReservingClassTypesRow(effectiveColumns.length);
      for (let i = 0; i < effectiveColumns.length; i++) {
        row[i] = String(rawRow[i] ?? "");
      }
      if (row.some((v) => String(v).trim() !== "")) {
        rows.push(row);
      }
    }
    return { columns: effectiveColumns, rows };
  }

  function renderReservingClassTypesEmpty(message, colspan = 3) {
    if (!reservingClassTypesBody) return;
    const span = Math.max(1, Number(colspan) || 3);
    reservingClassTypesBody.innerHTML = `
      <tr>
        <td colspan="${span}" class="dataset-types-empty">${escapeHtml(message || "No rows found.")}</td>
      </tr>
    `;
  }

  async function ensureReservingClassTypesLoaded(projectName, options = {}) {
    const force = !!options?.force;
    const key = normalizeProjectKey(projectName);
    if (!key) return false;
    if (!force && loadedReservingClassTypesByProject.has(key)) return true;
    if (force) loadedReservingClassTypesByProject.delete(key);

    const state = getProjectReservingClassTypesState(projectName);
    try {
      const res = await fetchImpl(`/reserving_class_types?project_name=${encodeURIComponent(projectName)}`);
      if (!res.ok) {
        const text = await res.text();
        throw new Error(`HTTP ${res.status}: ${text}`);
      }
      const out = await res.json();
      const parsed = normalizeReservingClassTypesPayload(out?.data);
      state.columns = [...parsed.columns];
      state.rows = parsed.rows.map((r) => r.map((v) => String(v ?? "")));
      setReservingClassTypesSourceNames(projectName, Array.isArray(out?.source_derived_names) ? out.source_derived_names : []);
      loadedReservingClassTypesByProject.add(key);
      return true;
    } catch (err) {
      setReservingClassTypesStatus(`Load error: ${err.message}`, true);
      state.columns = [...RESERVING_CLASS_TYPES_COLUMNS];
      state.rows = [];
      setReservingClassTypesSourceNames(projectName, []);
      return false;
    }
  }

  function showReservingClassTypesRowContextMenu(x, y, projectName, rowIndex, cellText = "") {
    if (!reservingClassTypesRowContextMenu) return;
    reservingClassTypesContextProject = projectName || "";
    reservingClassTypesContextRowIndex = rowIndex;
    reservingClassTypesContextCellText = String(cellText ?? "");
    positionContextMenu(reservingClassTypesRowContextMenu, x, y);
  }

  function hideReservingClassTypesRowContextMenu() {
    if (!reservingClassTypesRowContextMenu) return;
    reservingClassTypesRowContextMenu.classList.remove("show");
    reservingClassTypesContextProject = "";
    reservingClassTypesContextRowIndex = -1;
    reservingClassTypesContextCellText = "";
  }

  function deleteReservingClassTypesRow(projectName, rowIndex) {
    const state = getProjectReservingClassTypesState(projectName);
    if (!state.rows.length) return;
    if (!Number.isInteger(rowIndex) || rowIndex < 0 || rowIndex >= state.rows.length) return;
    state.rows.splice(rowIndex, 1);
    if (!state.rows.length) {
      state.rows.push(createEmptyReservingClassTypesRow((state.columns || []).length || RESERVING_CLASS_TYPES_COLUMNS.length));
    }
    renderReservingClassTypesTable(projectName);
    scheduleReservingClassTypesAutoSave(projectName);
  }

  function renderReservingClassTypesTable(projectName) {
    if (!reservingClassTypesBody) return;
    if (!projectName) {
      renderReservingClassTypesEmpty("Select a project to load reserving class types.");
      return;
    }

    const table = document.getElementById("reservingClassTypesTable");
    const state = getProjectReservingClassTypesState(projectName);
    const sourceSet = getReservingClassTypesSourceSet(projectName);
    const collapsedSet = getReservingClassTypesCollapsedSet(projectName);
    const sortState = getReservingClassTypesSortState();
    const columns = Array.isArray(state.columns) && state.columns.length ? state.columns : [...RESERVING_CLASS_TYPES_COLUMNS];
    state.columns = [...columns];
    const idx = getReservingClassTypeColIndexes(columns);
    const nameColIdx = idx.name;
    const levelColIdx = idx.level;
    const formulaColIdx = idx.formula;
    const eexFormulaColIdx = idx.eexFormula;
    const visibleCols = [
      {
        label: "Name",
        idx: nameColIdx >= 0 ? nameColIdx : 0,
        width: "300px",
        minWidth: 130,
      },
      {
        label: "Formula",
        idx: formulaColIdx >= 0 ? formulaColIdx : 2,
        width: "420px",
        minWidth: 130,
      },
      {
        label: "EEX Formula",
        idx: eexFormulaColIdx >= 0 ? eexFormulaColIdx : 3,
        width: "420px",
        minWidth: 130,
      },
    ];

    if (!Array.isArray(state.rows) || state.rows.length === 0) {
      state.rows = [createEmptyReservingClassTypesRow(columns.length)];
    }

    if (table) {
      const colgroup = table.querySelector("colgroup");
      if (colgroup) {
        colgroup.innerHTML = "";
        visibleCols.forEach((c) => {
          const col = document.createElement("col");
          col.style.width = c.width;
          colgroup.appendChild(col);
        });
      }

      const thead = table.querySelector("thead");
      if (thead) {
        thead.innerHTML = "";
        const tr = document.createElement("tr");
        visibleCols.forEach((c) => {
          const th = document.createElement("th");
          const isActiveSort = sortState.colLabel === c.label && (sortState.dir === "asc" || sortState.dir === "desc");
          const icon = isActiveSort ? (sortState.dir === "asc" ? " \u25B2" : " \u25BC") : "";
          th.textContent = `${c.label}${icon}`;
          th.title = isActiveSort
            ? `${c.label}: ${sortState.dir === "asc" ? "ascending" : "descending"} (click to toggle)`
            : `${c.label}: unsorted (click to sort ascending)`;
          th.style.cursor = "pointer";
          th.addEventListener("click", () => {
            toggleReservingClassTypesSort(c.label);
            renderReservingClassTypesTable(projectName);
          });
          tr.appendChild(th);
        });
        thead.appendChild(tr);
      }
    }

    for (let i = 0; i < state.rows.length; i++) {
      if (!Array.isArray(state.rows[i])) {
        state.rows[i] = createEmptyReservingClassTypesRow(columns.length);
        continue;
      }
      if (state.rows[i].length < columns.length) {
        const fallback = createEmptyReservingClassTypesRow(columns.length);
        for (let j = 0; j < state.rows[i].length && j < columns.length; j++) {
          fallback[j] = state.rows[i][j];
        }
        state.rows[i] = fallback;
      }
    }

    const groups = new Map();
    state.rows.forEach((row, rowIndex) => {
      const levelKey = getReservingClassLevelKey(levelColIdx >= 0 ? row[levelColIdx] : "");
      if (!groups.has(levelKey)) {
        groups.set(levelKey, {
          key: levelKey,
          label: levelKey ? `Level: ${levelKey}` : "Level: (blank)",
          rows: [],
        });
      }
      groups.get(levelKey).rows.push({ row, rowIndex });
    });

    const sortedGroups = Array.from(groups.values()).sort((a, b) => {
      const aKey = String(a.key || "");
      const bKey = String(b.key || "");
      const aNum = Number(aKey);
      const bNum = Number(bKey);
      const aIsNum = Number.isFinite(aNum) && aKey !== "";
      const bIsNum = Number.isFinite(bNum) && bKey !== "";
      if (aIsNum && bIsNum) return aNum - bNum;
      if (aIsNum && !bIsNum) return -1;
      if (!aIsNum && bIsNum) return 1;
      if (!aKey && bKey) return 1;
      if (aKey && !bKey) return -1;
      return aKey.localeCompare(bKey);
    });

    reservingClassTypesBody.innerHTML = "";
    for (const group of sortedGroups) {
      const groupTr = document.createElement("tr");
      groupTr.className = "rct-group-row";
      const groupTd = document.createElement("td");
      groupTd.colSpan = visibleCols.length;

      const toggleBtn = document.createElement("button");
      toggleBtn.type = "button";
      const collapsed = collapsedSet.has(group.key);
      toggleBtn.className = "rct-group-toggle" + (collapsed ? "" : " expanded");
      toggleBtn.textContent = collapsed ? "+" : "-";
      toggleBtn.title = collapsed ? "Expand level" : "Collapse level";
      toggleBtn.addEventListener("click", (e) => {
        e.preventDefault();
        e.stopPropagation();
        if (collapsedSet.has(group.key)) collapsedSet.delete(group.key);
        else collapsedSet.add(group.key);
        renderReservingClassTypesTable(projectName);
      });

      const label = document.createElement("span");
      label.className = "rct-group-label";
      label.textContent = group.label;

      groupTd.appendChild(toggleBtn);
      groupTd.appendChild(label);
      groupTr.appendChild(groupTd);
      reservingClassTypesBody.appendChild(groupTr);

      if (collapsed) continue;

      const sortCol = visibleCols.find((c) => c.label === sortState.colLabel);
      const sortIdx = sortCol ? sortCol.idx : -1;
      const sortDir = sortState.dir === "desc" ? "desc" : "asc";
      const rowsToRender = group.rows.slice();
      if (sortIdx >= 0 && sortState.dir) {
        rowsToRender.sort((a, b) => {
          const cmp = compareReservingClassTypeSortValues(
            a?.row?.[sortIdx],
            b?.row?.[sortIdx],
            sortDir,
          );
          if (cmp !== 0) return cmp;
          return (a?.rowIndex ?? 0) - (b?.rowIndex ?? 0);
        });
      }

      for (const item of rowsToRender) {
        const { row, rowIndex } = item;
        const tr = document.createElement("tr");
        tr.dataset.rowIndex = String(rowIndex);

        const rowName = String(row[nameColIdx >= 0 ? nameColIdx : 0] ?? "").trim().toLowerCase();
        const isSourceDerived = !!rowName && sourceSet.has(rowName);

        visibleCols.forEach((c) => {
          const td = document.createElement("td");
          td.dataset.col = c.label;
          const text = document.createElement("div");
          text.className = "rct-cell-text";
          text.textContent = String(row[c.idx] ?? "");
          if (isSourceDerived && (c.idx === formulaColIdx || c.idx === eexFormulaColIdx)) {
            td.title = "Source-derived rows lock formula fields.";
          }
          td.appendChild(text);
          tr.appendChild(td);
        });

        tr.addEventListener("contextmenu", (e) => {
          e.preventDefault();
          e.stopPropagation();
          hideContextMenu();
          hideFolderContextMenu();
          hideTreeContextMenu();
          hideDatasetTypesRowContextMenu();
          const cell = e.target?.closest?.("td");
          const cellText = cell ? String(cell.textContent ?? "") : "";
          showReservingClassTypesRowContextMenu(e.clientX, e.clientY, projectName, rowIndex, cellText);
        });

        reservingClassTypesBody.appendChild(tr);
      }
    }

    initTableColumnResizing("reservingClassTypesTable", visibleCols.map((c) => c.minWidth));
  }

  async function loadReservingClassTypes(projectName, options = {}) {
    const requestSeq = ++reservingClassTypesLoadSeq;
    if (!projectName) {
      renderReservingClassTypesEmpty("Select a project to load reserving class types.");
      setReservingClassTypesStatus("");
      return;
    }

    setReservingClassTypesStatus("Loading reserving class types...");
    const loadedOk = await ensureReservingClassTypesLoaded(projectName, options);
    if (requestSeq !== reservingClassTypesLoadSeq) return;
    renderReservingClassTypesTable(projectName);
    if (loadedOk) setReservingClassTypesStatus("");
  }

  async function saveReservingClassTypes(projectName) {
    if (!projectName) return false;
    const state = getProjectReservingClassTypesState(projectName);
    const columns = Array.isArray(state.columns) && state.columns.length ? [...state.columns] : [...RESERVING_CLASS_TYPES_COLUMNS];
    const rows = (state.rows || [])
      .map((raw) => {
        const row = createEmptyReservingClassTypesRow(columns.length);
        for (let i = 0; i < columns.length; i++) {
          row[i] = String(raw?.[i] ?? "").trim();
        }
        return row;
      })
      .filter((row) => row.some((v) => v !== ""));

    setReservingClassTypesStatus("Saving reserving class types...");
    try {
      const res = await fetchImpl("/reserving_class_types", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          project_name: projectName,
          columns,
          rows,
        }),
      });
      if (!res.ok) {
        let detail = "";
        try {
          const body = await res.json();
          detail = String(body?.detail || "").trim();
        } catch {
          const text = await res.text();
          detail = String(text || "").trim();
        }
        throw new Error(detail || `HTTP ${res.status}`);
      }

      const out = await res.json();
      const parsed = normalizeReservingClassTypesPayload(out?.data);
      state.columns = [...parsed.columns];
      state.rows = parsed.rows.map((r) => r.map((v) => String(v ?? "")));
      setReservingClassTypesSourceNames(projectName, Array.isArray(out?.source_derived_names) ? out.source_derived_names : []);
      renderReservingClassTypesTable(projectName);
      setReservingClassTypesStatus(`Saved reserving class types to ${out.path}`);
      setStatus(`Saved reserving class types: ${projectName}`);
      await loadAuditLog(projectName, true);
      return true;
    } catch (err) {
      setReservingClassTypesStatus(`Save error: ${err.message}`, true);
      setStatus(`Reserving class types save error: ${err.message}`);
      return false;
    }
  }

  async function saveReservingClassTypesToLocalFile(projectName) {
    const name = String(projectName || "").trim();
    if (!name) {
      setReservingClassTypesStatus("Select a project first.", true);
      return;
    }

    const hostApi = window.ADAHost || window.parent?.ADAHost || window.top?.ADAHost;
    if (!hostApi?.saveJsonFile) {
      setReservingClassTypesStatus("Local save is available in the desktop app only.", true);
      return;
    }

    const state = getProjectReservingClassTypesState(name);
    const columns = Array.isArray(state.columns) && state.columns.length ? [...state.columns] : [...RESERVING_CLASS_TYPES_COLUMNS];
    const rows = buildPersistableReservingClassTypesRows(state.rows || []);
    const safeName = sanitizeFileStem(name) || "project";
    const suggestedName = `${safeName}_reserving_class_types.json`;
    let startDir = "";
    try {
      const docsPath = String((await hostApi.getDocumentsPath?.()) || "").trim();
      if (docsPath) {
        startDir = joinWinPath(docsPath, "ArcRho", "templates");
      }
    } catch {
      startDir = "";
    }

    setReservingClassTypesStatus("Saving reserving class types to local file...");
    try {
      const result = await hostApi.saveJsonFile({
        data: {
          project_name: name,
          columns: [...RESERVING_CLASS_TYPES_COLUMNS],
          rows,
        },
        suggestedName,
        startDir,
        filters: [{ name: "JSON", extensions: ["json"] }],
      });
      if (result?.canceled) {
        setReservingClassTypesStatus("Local save canceled.");
        return;
      }
      if (result?.error) {
        throw new Error(String(result.error || "Unable to save local file."));
      }
      const outPath = String(result?.path || "").trim();
      setReservingClassTypesStatus(outPath ? `Saved local reserving class types: ${outPath}` : "Saved local reserving class types.");
      setStatus(outPath ? `Saved Reserving Class Types local file: ${outPath}` : "Saved Reserving Class Types local file.");
    } catch (err) {
      const msg = String(err?.message || err || "Unable to save local file.");
      setReservingClassTypesStatus(`Local save failed: ${msg}`, true);
      setStatus(`Reserving class types local save error: ${msg}`);
    }
  }

  async function loadReservingClassTypesFromLocalFile(projectName) {
    const name = String(projectName || "").trim();
    if (!name) {
      setReservingClassTypesStatus("Select a project first.", true);
      return;
    }

    const hostApi = window.ADAHost || window.parent?.ADAHost || window.top?.ADAHost;
    if (!hostApi?.pickOpenFile) {
      setReservingClassTypesStatus("Local load is available in the desktop app only.", true);
      return;
    }

    let startDir = "";
    try {
      const docsPath = String((await hostApi.getDocumentsPath?.()) || "").trim();
      if (docsPath) {
        startDir = joinWinPath(docsPath, "ArcRho", "templates");
      }
    } catch {
      startDir = "";
    }

    const pickedPath = String(
      (await hostApi.pickOpenFile({
        startDir,
        filters: [
          { name: "Reserving Class Types", extensions: ["json", "xlsx"] },
          { name: "JSON", extensions: ["json"] },
          { name: "Excel", extensions: ["xlsx"] },
        ],
      })) || "",
    ).trim();
    if (!pickedPath) {
      setReservingClassTypesStatus("Local load canceled.");
      return;
    }

    setReservingClassTypesStatus("Loading reserving class types from local file...");
    try {
      const parseRes = await fetchImpl("/reserving_class_types/import_local_file", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ file_path: pickedPath }),
      });
      if (!parseRes.ok) {
        let detail = "";
        try {
          const body = await parseRes.json();
          detail = String(body?.detail || "").trim();
        } catch {
          const text = await parseRes.text();
          detail = String(text || "").trim();
        }
        throw new Error(detail || `HTTP ${parseRes.status}`);
      }
      const parseOut = await parseRes.json();
      const parsed = normalizeLocalReservingClassTypesPayload(parseOut?.data);
      const state = getProjectReservingClassTypesState(name);
      state.columns = [...parsed.columns];
      state.rows = parsed.rows.length > 0 ? parsed.rows : [createEmptyReservingClassTypesRow(parsed.columns.length)];
      renderReservingClassTypesTable(name);
      scheduleReservingClassTypesAutoSave(name);

      const fileFormat = String(parseOut?.format || "").trim().toUpperCase();
      const formatText = fileFormat ? ` (${fileFormat})` : "";
      const finalMsg = `Loaded local reserving class types from ${pickedPath}${formatText}.`;
      setReservingClassTypesStatus(finalMsg);
      setStatus(finalMsg);
    } catch (err) {
      const msg = String(err?.message || err || "Unable to load local reserving class types.");
      setReservingClassTypesStatus(`Local load failed: ${msg}`, true);
      setStatus(`Reserving class types local load error: ${msg}`);
    }
  }

  async function handleReservingClassTypesRowContextAction(action) {
    if (!action || !reservingClassTypesContextProject) return;
    const projectName = reservingClassTypesContextProject;
    const rowIndex = reservingClassTypesContextRowIndex;
    const cellText = reservingClassTypesContextCellText;
    hideReservingClassTypesRowContextMenu();
    if (action === "copy-cell") {
      const ok = await copyTextToClipboard(cellText);
      if (ok) {
        setReservingClassTypesStatus("Cell value copied.");
      } else {
        setReservingClassTypesStatus("Unable to copy cell value.", true);
      }
    } else if (action === "edit-row") {
      openReservingClassTypeEditor(projectName, rowIndex);
    } else if (action === "add-type") {
      openReservingClassTypeEditor(projectName, rowIndex, {
        mode: "add",
        insertAfterIndex: Number.isInteger(rowIndex) ? rowIndex : -1,
      });
    } else if (action === "delete-row") {
      deleteReservingClassTypesRow(projectName, rowIndex);
    }
  }

  function onEditorHeaderMouseDown(e) {
    if (!reservingClassTypeEditor || e.button !== 0) return;
    const rect = reservingClassTypeEditor.getBoundingClientRect();
    // Convert to absolute pixel position before removing transform to avoid jump
    reservingClassTypeEditor.style.left = `${rect.left}px`;
    reservingClassTypeEditor.style.top = `${rect.top}px`;
    reservingClassTypeEditor.style.transform = "none";
    rctEditorDragState = {
      offsetX: e.clientX - rect.left,
      offsetY: e.clientY - rect.top,
    };
    e.preventDefault();
  }

  function onEditorMouseMove(e) {
    if (!reservingClassTypeEditor || !rctEditorDragState) return;
    const left = Math.max(8, Math.min(window.innerWidth - reservingClassTypeEditor.offsetWidth - 8, e.clientX - rctEditorDragState.offsetX));
    const top = Math.max(8, Math.min(window.innerHeight - reservingClassTypeEditor.offsetHeight - 8, e.clientY - rctEditorDragState.offsetY));
    reservingClassTypeEditor.style.left = `${left}px`;
    reservingClassTypeEditor.style.top = `${top}px`;
  }

  function onEditorMouseUp() {
    rctEditorDragState = null;
  }

  return {
    setReservingClassTypesStatus,
    closeReservingClassTypeEditor,
    openReservingClassTypeEditor,
    applyReservingClassTypeEditor,
    showReservingClassTypesRowContextMenu,
    hideReservingClassTypesRowContextMenu,
    deleteReservingClassTypesRow,
    renderReservingClassTypesTable,
    loadReservingClassTypes,
    saveReservingClassTypes,
    saveReservingClassTypesToLocalFile,
    loadReservingClassTypesFromLocalFile,
    handleReservingClassTypesRowContextAction,
    onEditorHeaderMouseDown,
    onEditorMouseMove,
    onEditorMouseUp,
  };
}
