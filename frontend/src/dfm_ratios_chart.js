/*
===============================================================================
DFM Ratios Chart - popup chart modal rendering and interactions
===============================================================================
*/
import {
  state,
  calcRatio, formatRatio, computeAverageForColumn,
  ratioStrikeSet, selectedSummaryByCol,
  ratioChartThresholdByCol, ratioChartLowerThresholdByCol,
  summaryRowMap,
  getRatioChartCol, setRatioChartCol,
  getRatioChartRaf, setRatioChartRaf,
  getRatioChartWired, setRatioChartWired,
  getRatioChartPoints, setRatioChartPoints,
  getRatioChartScale, setRatioChartScale,
  getRatioChartDragActive, setRatioChartDragActive,
  getRatioChartDragMoved, setRatioChartDragMoved,
  getRatioChartHoverLine, setRatioChartHoverLine,
  getRatioChartDragTarget, setRatioChartDragTarget,
  getRatioChartHoverTimer, setRatioChartHoverTimer,
  getRatioChartHoverKey, setRatioChartHoverKey,
  getRatioChartTooltipVisible, setRatioChartTooltipVisible,
  getEffectiveDevLabelsForModel, getRatioHeaderLabels,
  getOriginLabelTextForRatio, buildSummaryRows, getDfmDecimalPlaces,
  buildExcludedSetForColumn,
} from "./dfm_state.js";
import {
  isUserEntryConfig,
  getUserEntryValueForCol,
  scheduleRatioSummaryUpdate,
} from "./dfm_ratios_summary_table.js";

let _onRatioStateMutated = () => {};

export function setRatioChartCallbacks({ onRatioStateMutated } = {}) {
  if (typeof onRatioStateMutated === "function") _onRatioStateMutated = onRatioStateMutated;
}

// =============================================================================
// Ratio Chart Modal
// =============================================================================
function getRatioChartModalEl() {
  return document.getElementById("dfmRatioChartModal");
}

function getRatioChartCanvas() {
  return document.getElementById("dfmRatioChartCanvas");
}

export function isRatioChartOpen() {
  const modal = getRatioChartModalEl();
  return !!modal && modal.classList.contains("open");
}

function getRatioColumnLabel(col) {
  const model = state.model;
  const devs = getEffectiveDevLabelsForModel(model || {});
  const ratioLabels = getRatioHeaderLabels(devs);
  const label = ratioLabels[col] || "";
  return label ? `(${col + 1}) ${label}` : `Column ${col + 1}`;
}

function getSelectedSummaryConfigForCol(col) {
  const rows = buildSummaryRows();
  const defaultRowId = rows[0]?.id || "";
  const rowId = selectedSummaryByCol.get(col) || defaultRowId;
  return rowId ? summaryRowMap.get(rowId) : null;
}

function resolveUserEntryToSourceConfig(cfg, col) {
  // If cfg is a user entry whose formula references a known summary row,
  // return that row's config so the chart can show which points it uses.
  if (!cfg || !isUserEntryConfig(cfg)) return null;
  const inputs = cfg.inputs ?? cfg.formulas;
  if (!Array.isArray(inputs)) return null;
  const raw = String(inputs[col] ?? "").trim();
  if (!raw) return null;
  // Find referenced summary row labels in the formula
  for (const [rowId, rowCfg] of summaryRowMap) {
    if (isUserEntryConfig(rowCfg)) continue;
    const label = String(rowCfg.label || "").trim();
    if (!label) continue;
    const escaped = label.replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
    if (new RegExp(`["']${escaped}["']`, "i").test(raw) || new RegExp(escaped, "i").test(raw)) {
      return rowCfg;
    }
  }
  return null;
}

function buildUsedRowSetForColumn(model, col, cfg, excludedSet) {
  const used = new Set();
  if (!cfg) return used;
  if (isUserEntryConfig(cfg)) {
    // Resolve to the underlying summary method if formula references one
    const sourceCfg = resolveUserEntryToSourceConfig(cfg, col);
    if (sourceCfg) return buildUsedRowSetForColumn(model, col, sourceCfg, excludedSet);
    return used;
  }
  if (!model || !Array.isArray(model.values) || !Array.isArray(model.mask)) return used;

  const vals = model.values;
  const mask = model.mask;
  const rowCount = Array.isArray(model.origin_labels) ? model.origin_labels.length : vals.length;
  const periodsRaw = cfg?.periods ?? "all";
  const periods = typeof periodsRaw === "string" && periodsRaw.toLowerCase() === "all"
    ? "all"
    : Number(periodsRaw);
  const lookback = Number.isFinite(periods) && periods > 0 ? Math.floor(periods) : null;

  const includeRow = (r) => {
    const hasA = !!(mask[r] && mask[r][col]);
    const hasB = !!(mask[r] && mask[r][col + 1]);
    if (!hasA || !hasB) return null;
    const ratio = calcRatio(vals?.[r]?.[col], vals?.[r]?.[col + 1]);
    if (!Number.isFinite(ratio)) return null;
    return ratio;
  };

  if (lookback) {
    let picked = 0;
    for (let r = rowCount - 1; r >= 0; r--) {
      if (picked >= lookback) break;
      const ratio = includeRow(r);
      if (!Number.isFinite(ratio)) continue;
      if (excludedSet && excludedSet.has(`${r},${col}`)) continue;
      picked += 1;
      used.add(r);
    }
  } else {
    for (let r = 0; r < rowCount; r++) {
      const ratio = includeRow(r);
      if (!Number.isFinite(ratio)) continue;
      if (excludedSet && excludedSet.has(`${r},${col}`)) continue;
      used.add(r);
    }
  }

  return used;
}

function getThresholdValueForCol(col, fallback) {
  const raw = ratioChartThresholdByCol.get(col);
  if (Number.isFinite(raw)) return raw;
  const next = Number.isFinite(fallback) ? fallback : null;
  if (next != null) ratioChartThresholdByCol.set(col, next);
  return next;
}

function setThresholdValueForCol(col, value) {
  if (!Number.isFinite(value)) return;
  ratioChartThresholdByCol.set(col, value);
}

function getLowerThresholdValueForCol(col, fallback) {
  const raw = ratioChartLowerThresholdByCol.get(col);
  if (Number.isFinite(raw)) return raw;
  const next = Number.isFinite(fallback) ? fallback : null;
  if (next != null) ratioChartLowerThresholdByCol.set(col, next);
  return next;
}

function setLowerThresholdValueForCol(col, value) {
  if (!Number.isFinite(value)) return;
  ratioChartLowerThresholdByCol.set(col, value);
}

export function resetRatioChartThresholds() {
  ratioChartThresholdByCol.clear();
  ratioChartLowerThresholdByCol.clear();
  setRatioChartHoverLine(null);
  setRatioChartDragTarget(null);
  setRatioChartDragActive(false);
  setRatioChartDragMoved(false);
  clearRatioChartHover();
  if (isRatioChartOpen()) scheduleRatioChartRender();
}

function applyThresholdBandExcludes(col, upperValue, lowerValue) {
  const model = state.model;
  if (!model || !Array.isArray(model.values) || !Array.isArray(model.mask)) return;
  const devs = getEffectiveDevLabelsForModel(model);
  if (col < 0 || col >= devs.length - 1) return;
  const vals = model.values;
  const mask = model.mask;
  const rowCount = Array.isArray(model.origin_labels) ? model.origin_labels.length : vals.length;
  let upper = upperValue;
  let lower = lowerValue;
  if (Number.isFinite(upper) && Number.isFinite(lower) && upper < lower) {
    const tmp = upper;
    upper = lower;
    lower = tmp;
  }

  for (let r = 0; r < rowCount; r++) {
    const key = `${r},${col}`;
    const hasA = !!(mask[r] && mask[r][col]);
    const hasB = !!(mask[r] && mask[r][col + 1]);
    if (!hasA || !hasB) {
      ratioStrikeSet.delete(key);
      continue;
    }
    const ratio = calcRatio(vals?.[r]?.[col], vals?.[r]?.[col + 1]);
    if (!Number.isFinite(ratio)) {
      ratioStrikeSet.delete(key);
      continue;
    }
    if ((Number.isFinite(upper) && ratio > upper) || (Number.isFinite(lower) && ratio < lower)) {
      ratioStrikeSet.add(key);
    } else {
      ratioStrikeSet.delete(key);
    }
  }

  const cells = document.querySelectorAll(
    `#ratioWrap table.ratioMainTable td.ratioCell[data-col="${col}"][data-r]`
  );
  cells.forEach((cell) => {
    const r = cell.dataset.r;
    if (r == null) return;
    const key = `${r},${col}`;
    cell.classList.toggle("strike", ratioStrikeSet.has(key));
  });

  scheduleRatioSummaryUpdate();
  _onRatioStateMutated();
  scheduleRatioChartRender();
}

function buildRatioColumnSeries(col) {
  const model = state.model;
  if (!model || !Array.isArray(model.values) || !Array.isArray(model.mask)) {
    return { labels: [], values: [], status: [] };
  }
  const devs = getEffectiveDevLabelsForModel(model);
  if (col < 0 || col >= devs.length - 1) {
    return { labels: [], values: [], status: [] };
  }
  const origins = model.origin_labels || [];
  const vals = model.values;
  const mask = model.mask;
  const labels = [];
  const values = [];
  const status = [];
  const cfg = getSelectedSummaryConfigForCol(col);
  const excludedSet = buildExcludedSetForColumn(model, col, cfg, ratioStrikeSet);
  const usedSet = buildUsedRowSetForColumn(model, col, cfg, excludedSet);

  for (let r = 0; r < origins.length; r++) {
    labels.push(String(origins[r] ?? ""));
    const hasA = !!(mask[r] && mask[r][col]);
    const hasB = !!(mask[r] && mask[r][col + 1]);
    if (!hasA || !hasB) {
      values.push(null);
      status.push("none");
      continue;
    }
    const ratio = calcRatio(vals?.[r]?.[col], vals?.[r]?.[col + 1]);
    if (!Number.isFinite(ratio)) {
      values.push(null);
      status.push("none");
      continue;
    }
    values.push(ratio);
    if (excludedSet.has(`${r},${col}`)) {
      status.push("excluded");
    } else if (usedSet.has(r)) {
      status.push("selected");
    } else {
      status.push("not-used");
    }
  }

  return { labels, values, status };
}

function resizeCanvasToCSS(canvas) {
  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  const w = Math.max(1, Math.floor(rect.width * dpr));
  const h = Math.max(1, Math.floor(rect.height * dpr));
  if (canvas.width !== w || canvas.height !== h) {
    canvas.width = w;
    canvas.height = h;
  }
}

function getRatioChartTooltipEl() {
  return document.getElementById("dfmRatioChartTooltip");
}

function hideRatioChartTooltip() {
  const tooltip = getRatioChartTooltipEl();
  if (!tooltip) return;
  tooltip.style.display = "none";
  setRatioChartTooltipVisible(false);
}

function clearRatioChartHover() {
  const timer = getRatioChartHoverTimer();
  if (timer) {
    clearTimeout(timer);
    setRatioChartHoverTimer(null);
  }
  setRatioChartHoverKey(null);
  hideRatioChartTooltip();
}

function showRatioChartTooltip(point, canvas) {
  const tooltip = getRatioChartTooltipEl();
  if (!tooltip || !canvas || !point) return;
  const wrap = canvas.closest(".dfmRatioChartCanvasWrap");
  if (!wrap) return;
  const rect = canvas.getBoundingClientRect();
  const wrapRect = wrap.getBoundingClientRect();
  const label = String(point.label ?? "");
  const value =
    Number.isFinite(point.value) ? formatRatio(point.value, getDfmDecimalPlaces()) : "";
  let statusLabel = "Not Used";
  if (point.status === "excluded") statusLabel = "Excluded";
  if (point.status === "selected") statusLabel = "Selected";

  let inner = `<div class="dfmTooltipLabel">${label}</div>`;
  if (value) {
    inner += `<div class="dfmTooltipMeta">${value} | ${statusLabel}</div>`;
  } else {
    inner += `<div class="dfmTooltipMeta">${statusLabel}</div>`;
  }
  tooltip.innerHTML = inner;
  tooltip.style.display = "block";
  tooltip.style.visibility = "hidden";

  const x = point.x + (rect.left - wrapRect.left);
  const y = point.y + (rect.top - wrapRect.top);
  const tipW = tooltip.offsetWidth || 0;
  const tipH = tooltip.offsetHeight || 0;
  const pad = 6;
  let left = x - tipW / 2;
  let top = y - tipH - 10;
  const maxLeft = Math.max(pad, wrapRect.width - tipW - pad);
  const maxTop = Math.max(pad, wrapRect.height - tipH - pad);
  left = Math.min(maxLeft, Math.max(pad, left));
  top = Math.min(maxTop, Math.max(pad, top));
  tooltip.style.left = `${left}px`;
  tooltip.style.top = `${top}px`;
  tooltip.style.visibility = "visible";
  setRatioChartTooltipVisible(true);
}

function renderRatioColumnChart(canvas, labels, values, status) {
  if (!canvas) return;
  const ctx = canvas.getContext("2d");
  resizeCanvasToCSS(canvas);
  const dpr = window.devicePixelRatio || 1;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  const W_css = canvas.width / dpr;
  const H_css = canvas.height / dpr;
  ctx.clearRect(0, 0, W_css, H_css);

  setRatioChartPoints([]);
  const valid = values
    .map((v, i) => (Number.isFinite(v) ? { v, i } : null))
    .filter(Boolean);
  if (!valid.length) {
    ctx.font = "12px Arial";
    ctx.fillStyle = "#555";
    ctx.fillText("No data to plot.", 10, 20);
    return;
  }

  let yMin = Infinity;
  let yMax = -Infinity;
  valid.forEach((pt) => {
    yMin = Math.min(yMin, pt.v);
    yMax = Math.max(yMax, pt.v);
  });
  // Save raw data range for clamping threshold drag
  const dataMin = yMin;
  const dataMax = yMax;
  // Expand y-axis to include the selected summary value so it's not clipped
  const chartColForRange = getRatioChartCol();
  if (chartColForRange != null) {
    const cfgForRange = getSelectedSummaryConfigForCol(chartColForRange);
    if (cfgForRange) {
      let selVal = null;
      if (isUserEntryConfig(cfgForRange)) {
        selVal = getUserEntryValueForCol(cfgForRange, chartColForRange);
      } else {
        const model2 = state.model;
        if (model2 && Array.isArray(model2.values) && Array.isArray(model2.mask)) {
          const exc = buildExcludedSetForColumn(model2, chartColForRange, cfgForRange, ratioStrikeSet);
          const sum = computeAverageForColumn(model2, chartColForRange, exc, cfgForRange);
          const isVol = String(cfgForRange.base || "volume").toLowerCase() === "volume";
          const hasVal = sum.value !== null && (isVol ? sum.sumA : sum.totalIncluded > 0);
          selVal = hasVal ? sum.value : null;
        }
      }
      if (selVal != null && Number.isFinite(selVal)) {
        yMin = Math.min(yMin, selVal);
        yMax = Math.max(yMax, selVal);
      }
    }
  }
  // Expand y-axis to include threshold lines so they're always visible
  if (chartColForRange != null) {
    const upperTh = getThresholdValueForCol(chartColForRange);
    const lowerTh = getLowerThresholdValueForCol(chartColForRange);
    if (Number.isFinite(upperTh)) {
      yMin = Math.min(yMin, upperTh);
      yMax = Math.max(yMax, upperTh);
    }
    if (Number.isFinite(lowerTh)) {
      yMin = Math.min(yMin, lowerTh);
      yMax = Math.max(yMax, lowerTh);
    }
  }
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  // Add a small padding so edge values aren't clipped against the axis boundary
  const yPad = (yMax - yMin) * 0.04;
  yMin -= yPad;
  yMax += yPad;

  const W = W_css;
  const H = H_css;
  const padT = 12;
  const maxLabelLen = labels.reduce((m, v) => Math.max(m, String(v ?? "").length), 0);
  const denseLabels = labels.length > 8 || maxLabelLen > 4;
  const rotate90 = labels.length > 15;
  const labelPad = (denseLabels || rotate90) ? Math.min(48, Math.max(16, Math.ceil(maxLabelLen * 3))) : 0;
  const padL = 44 + labelPad;
  const padR = 12 + (denseLabels ? 8 : 0);
  const extraBottomPad = 16;
  const padB = 40 + labelPad + extraBottomPad;
  const x0 = padL;
  const y0 = padT;
  const x1 = W - padR;
  const y1 = H - padB;
  const span = Math.max(1, labels.length - 1);
  const getX = (i) => {
    if (labels.length <= 1) return x0 + (x1 - x0) / 2;
    return x0 + (i / span) * (x1 - x0);
  };
  setRatioChartScale({ x0, x1, y0, y1, yMin, yMax, dataMin, dataMax, width: W, height: H });

  ctx.strokeStyle = "#9ca3af";
  ctx.lineWidth = 1;
  ctx.beginPath();
  ctx.moveTo(x0, y0);
  ctx.lineTo(x0, y1);
  ctx.lineTo(x1, y1);
  ctx.stroke();

  const yTicks = 4;
  ctx.font = "11px Arial";
  ctx.textAlign = "left";
  for (let i = 0; i <= yTicks; i++) {
    const t = i / yTicks;
    const y = y1 - t * (y1 - y0);
    const v = yMin + t * (yMax - yMin);
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath();
    ctx.moveTo(x0, y);
    ctx.lineTo(x1, y);
    ctx.stroke();
    ctx.fillStyle = "#374151";
    ctx.fillText(formatRatio(v, getDfmDecimalPlaces()), 6, y + 4);
  }

  ctx.fillStyle = "#374151";
  ctx.font = denseLabels ? "10px Arial" : "11px Arial";
  ctx.textAlign = rotate90 ? "center" : (denseLabels ? "right" : "center");
  const labelY = H - extraBottomPad - (denseLabels ? 8 : 6);
  for (let i = 0; i < labels.length; i++) {
    let x = getX(i);
    ctx.strokeStyle = "#eef2f7";
    ctx.beginPath();
    ctx.moveTo(x, y0);
    ctx.lineTo(x, y1);
    ctx.stroke();
    const label = String(labels[i] ?? "");
    if (rotate90) {
      ctx.save();
      ctx.translate(x, labelY);
      ctx.rotate(-Math.PI / 2);
      ctx.fillText(label, 0, 0);
      ctx.restore();
    } else if (denseLabels) {
      if (i === 0) x += 8;
      ctx.save();
      ctx.translate(x, labelY);
      ctx.rotate(-Math.PI / 4);
      ctx.fillText(label, 0, 0);
      ctx.restore();
    } else {
      ctx.fillText(label, x, labelY);
    }
  }

  // Interactive threshold lines (upper/lower)
  const chartCol = getRatioChartCol();
  const upperVal = getThresholdValueForCol(chartCol, yMax);
  const lowerVal = getLowerThresholdValueForCol(chartCol, yMin);
  const scale = getRatioChartScale();
  // Dim regions outside the threshold band
  if (Number.isFinite(upperVal)) {
    const t = (upperVal - yMin) / (yMax - yMin);
    const upperY = y1 - t * (y1 - y0);
    ctx.save();
    ctx.fillStyle = "rgba(15, 23, 42, 0.08)";
    ctx.fillRect(x0, y0, Math.max(0, x1 - x0), Math.max(0, upperY - y0));
    ctx.restore();
  }
  if (Number.isFinite(lowerVal)) {
    const t = (lowerVal - yMin) / (yMax - yMin);
    const lowerY = y1 - t * (y1 - y0);
    ctx.save();
    ctx.fillStyle = "rgba(15, 23, 42, 0.08)";
    ctx.fillRect(x0, lowerY, Math.max(0, x1 - x0), Math.max(0, y1 - lowerY));
    ctx.restore();
  }

  if (Number.isFinite(upperVal)) {
    const t = (upperVal - yMin) / (yMax - yMin);
    const thresholdY = y1 - t * (y1 - y0);
    scale.upperY = thresholdY;
    const active = getRatioChartHoverLine() === "upper" || (getRatioChartDragActive() && getRatioChartDragTarget() === "upper");
    ctx.save();
    ctx.strokeStyle = active ? "#f97316" : "#f59e0b";
    ctx.lineWidth = active ? 2.2 : 1.4;
    ctx.setLineDash(active ? [] : [6, 4]);
    ctx.beginPath();
    ctx.moveTo(x0, thresholdY);
    ctx.lineTo(x1, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = active ? "#f97316" : "#f59e0b";
    ctx.beginPath();
    ctx.arc(x1 + 6, thresholdY, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }
  if (Number.isFinite(lowerVal)) {
    const t = (lowerVal - yMin) / (yMax - yMin);
    const thresholdY = y1 - t * (y1 - y0);
    scale.lowerY = thresholdY;
    const active = getRatioChartHoverLine() === "lower" || (getRatioChartDragActive() && getRatioChartDragTarget() === "lower");
    ctx.save();
    ctx.strokeStyle = active ? "#0284c7" : "#0ea5e9";
    ctx.lineWidth = active ? 2.2 : 1.4;
    ctx.setLineDash(active ? [] : [6, 4]);
    ctx.beginPath();
    ctx.moveTo(x0, thresholdY);
    ctx.lineTo(x1, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = active ? "#0284c7" : "#0ea5e9";
    ctx.beginPath();
    ctx.arc(x1 + 14, thresholdY, 3, 0, Math.PI * 2);
    ctx.fill();
    ctx.restore();
  }

  ctx.strokeStyle = "#2563eb";
  ctx.lineWidth = 2;
  ctx.beginPath();
  let started = false;
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) {
      started = false;
      continue;
    }
    const x = getX(i);
    const y = y1 - ((v - yMin) / (yMax - yMin)) * (y1 - y0);
    if (!started) {
      ctx.moveTo(x, y);
      started = true;
    } else {
      ctx.lineTo(x, y);
    }
  }
  ctx.stroke();

  const points = [];
  for (let i = 0; i < values.length; i++) {
    const v = values[i];
    if (!Number.isFinite(v)) continue;
    const x = getX(i);
    const y = y1 - ((v - yMin) / (yMax - yMin)) * (y1 - y0);
    const pointStatus = status && status[i];
    points.push({
      x,
      y,
      rowIndex: i,
      label: labels?.[i],
      value: v,
      status: pointStatus,
    });
    const xSize = 5;
    if (pointStatus === "excluded") {
      ctx.strokeStyle = "#dc2626";
      ctx.lineWidth = 2;
      ctx.beginPath();
      ctx.moveTo(x - xSize, y - xSize);
      ctx.lineTo(x + xSize, y + xSize);
      ctx.moveTo(x + xSize, y - xSize);
      ctx.lineTo(x - xSize, y + xSize);
      ctx.stroke();
    } else {
      const isSelected = pointStatus === "selected";
      ctx.beginPath();
      ctx.arc(x, y, 3.2, 0, Math.PI * 2);
      ctx.fillStyle = isSelected ? "#2563eb" : "#ffffff";
      ctx.strokeStyle = isSelected ? "#2563eb" : "#94a3b8";
      ctx.lineWidth = 1.5;
      ctx.fill();
      ctx.stroke();
      if (isSelected) {
        ctx.beginPath();
        ctx.arc(x, y, 6, 0, Math.PI * 2);
        ctx.strokeStyle = "#16a34a";
        ctx.lineWidth = 1.6;
        ctx.stroke();
      }
    }
  }
  setRatioChartPoints(points);

  // Draw "Selected" summary value line + update HTML legend value
  const selValEl = document.getElementById("dfmRatioChartSelectedValue");
  if (selValEl) {
    const dashedIcon = selValEl.querySelector(".dfmLegendDashedLine");
    selValEl.textContent = "";
    if (dashedIcon) selValEl.appendChild(dashedIcon);
  }
  if (chartCol != null) {
    const cfg = getSelectedSummaryConfigForCol(chartCol);
    if (cfg) {
      const model = state.model;
      if (model && Array.isArray(model.values) && Array.isArray(model.mask)) {
        let selectedValue = null;
        if (isUserEntryConfig(cfg)) {
          selectedValue = getUserEntryValueForCol(cfg, chartCol);
        } else {
          const excluded = buildExcludedSetForColumn(model, chartCol, cfg, ratioStrikeSet);
          const summary = computeAverageForColumn(model, chartCol, excluded, cfg);
          const isVolume = String(cfg.base || "volume").toLowerCase() === "volume";
          const hasValue = summary.value !== null && (isVolume ? summary.sumA : summary.totalIncluded > 0);
          selectedValue = hasValue ? summary.value : null;
        }
        if (selectedValue != null && Number.isFinite(selectedValue)) {
          const t = (selectedValue - yMin) / (yMax - yMin);
          const rawY = y1 - t * (y1 - y0);
          const selectedY = Math.max(y0, Math.min(y1, rawY));
          scale.selectedY = selectedY;
          ctx.save();
          ctx.strokeStyle = "#16a34a";
          ctx.lineWidth = 1.4;
          ctx.setLineDash([6, 4]);
          ctx.beginPath();
          ctx.moveTo(x0, selectedY);
          ctx.lineTo(x1, selectedY);
          ctx.stroke();
          ctx.setLineDash([]);
          ctx.restore();
          // Update HTML legend selected value
          if (selValEl) {
            const dashedIcon = selValEl.querySelector(".dfmLegendDashedLine");
            selValEl.textContent = "";
            if (dashedIcon) selValEl.appendChild(dashedIcon);
            selValEl.appendChild(document.createTextNode(` Selected (${formatRatio(selectedValue, getDfmDecimalPlaces())})`));
          }
        }
      }
    }
  }
}

export function scheduleRatioChartRender() {
  if (getRatioChartRaf()) return;
  setRatioChartRaf(requestAnimationFrame(() => {
    setRatioChartRaf(null);
    renderRatioChartNow();
  }));
}

function renderRatioChartNow() {
  const col = getRatioChartCol();
  if (col == null) return;
  if (!isRatioChartOpen()) return;
  const canvas = getRatioChartCanvas();
  if (!canvas) return;
  const { labels, values, status } = buildRatioColumnSeries(col);
  renderRatioColumnChart(canvas, labels, values, status);
}

export function showRatioColumnChart(col) {
  const model = state.model;
  if (!model || !Array.isArray(model.values) || !Array.isArray(model.mask)) return;
  const devs = getEffectiveDevLabelsForModel(model);
  if (col < 0 || col >= devs.length - 1) return;

  setRatioChartCol(col);
  clearRatioChartHover();
  const modal = getRatioChartModalEl();
  if (!modal) return;
  const titleEl = document.getElementById("dfmRatioChartTitle");
  const metaEl = document.getElementById("dfmRatioChartMeta");
  const cfg = getSelectedSummaryConfigForCol(col);
  if (titleEl) titleEl.textContent = `Ratios - ${getRatioColumnLabel(col)}`;
  if (metaEl) {
    const rowLabel = getOriginLabelTextForRatio();
    const formulaLabel = cfg?.label || cfg?.id || "Selected";
    metaEl.textContent = `Selected: ${formulaLabel} - ${rowLabel}`;
  }
  modal.querySelector(".dfmModalCard")?._resetDrag?.();
  modal.classList.add("open");
  scheduleRatioChartRender();
}

function hideRatioColumnChart() {
  const modal = getRatioChartModalEl();
  if (modal) modal.classList.remove("open");
  setRatioChartCol(null);
  clearRatioChartHover();
}

export function wireRatioChartModal() {
  if (getRatioChartWired()) return;
  const modal = getRatioChartModalEl();
  if (!modal) return;
  setRatioChartWired(true);
  modal.querySelector(".dfmModalBackdrop")?.addEventListener("click", () => hideRatioColumnChart());
  document.getElementById("dfmRatioChartClose")?.addEventListener("click", () => hideRatioColumnChart());

  /* draggable window via header */
  const header = modal.querySelector(".dfmRatioChartHeader");
  const card = modal.querySelector(".dfmModalCard");
  if (header && card) {
    let dx = 0, dy = 0, sx = 0, sy = 0;
    header.addEventListener("pointerdown", (e) => {
      if (e.target.closest("button")) return;
      e.preventDefault();
      sx = e.clientX; sy = e.clientY;
      const onMove = (ev) => {
        dx += ev.clientX - sx; dy += ev.clientY - sy;
        sx = ev.clientX; sy = ev.clientY;
        card.style.transform = `translate(${dx}px,${dy}px)`;
      };
      const onUp = () => { window.removeEventListener("pointermove", onMove); window.removeEventListener("pointerup", onUp); };
      window.addEventListener("pointermove", onMove);
      window.addEventListener("pointerup", onUp);
    });
    card._resetDrag = () => { dx = dy = 0; card.style.transform = ""; };
  }
  const canvas = getRatioChartCanvas();
  canvas?.addEventListener("pointerdown", (e) => {
    clearRatioChartHover();
    const chartCol = getRatioChartCol();
    const scale = getRatioChartScale();
    if (chartCol == null || !scale) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { x0, x1, y0, y1, upperY, lowerY } = scale;
    if (cx < x0 || cx > x1 || cy < y0 || cy > y1) return;
    const hitUpper = Number.isFinite(upperY) && Math.abs(cy - upperY) <= 6;
    const hitLower = Number.isFinite(lowerY) && Math.abs(cy - lowerY) <= 6;
    if (!hitUpper && !hitLower) return;
    setRatioChartDragActive(true);
    setRatioChartDragMoved(false);
    if (hitUpper && hitLower) {
      setRatioChartDragTarget(
        Math.abs(cy - upperY) <= Math.abs(cy - lowerY) ? "upper" : "lower"
      );
    } else {
      setRatioChartDragTarget(hitUpper ? "upper" : "lower");
    }
    setRatioChartHoverLine(getRatioChartDragTarget());
    e.preventDefault();
  });

  canvas?.addEventListener("pointermove", (e) => {
    const chartCol = getRatioChartCol();
    const scale = getRatioChartScale();
    if (getRatioChartDragActive() || chartCol == null || !scale) {
      clearRatioChartHover();
      return;
    }
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    const { x0, x1, y0, y1, upperY, lowerY } = scale;
    let hit = null;
    if (cx >= x0 && cx <= x1 && cy >= y0 && cy <= y1) {
      const hitUpper = Number.isFinite(upperY) && Math.abs(cy - upperY) <= 6;
      const hitLower = Number.isFinite(lowerY) && Math.abs(cy - lowerY) <= 6;
      if (hitUpper && hitLower) {
        hit = Math.abs(cy - upperY) <= Math.abs(cy - lowerY) ? "upper" : "lower";
      } else if (hitUpper) {
        hit = "upper";
      } else if (hitLower) {
        hit = "lower";
      }
    }
    if (hit !== getRatioChartHoverLine()) {
      setRatioChartHoverLine(hit);
      scheduleRatioChartRender();
    }
    if (canvas) canvas.style.cursor = hit ? "ns-resize" : "default";
    if (hit) {
      clearRatioChartHover();
      return;
    }
    const chartPoints = getRatioChartPoints();
    if (!chartPoints.length) {
      clearRatioChartHover();
      return;
    }
    let best = null;
    let bestDist = Infinity;
    for (const pt of chartPoints) {
      const ddx = pt.x - cx;
      const ddy = pt.y - cy;
      const dist = ddx * ddx + ddy * ddy;
      if (dist < bestDist) {
        bestDist = dist;
        best = pt;
      }
    }
    const hitRadius = 8;
    if (!best || bestDist > hitRadius * hitRadius) {
      clearRatioChartHover();
      return;
    }
    const key = `${best.rowIndex},${chartCol}`;
    if (key !== getRatioChartHoverKey()) {
      clearRatioChartHover();
      setRatioChartHoverKey(key);
    }
    if (getRatioChartTooltipVisible() || getRatioChartHoverTimer()) return;
    setRatioChartHoverTimer(setTimeout(() => {
      setRatioChartHoverTimer(null);
      if (getRatioChartHoverKey() !== key) return;
      if (!isRatioChartOpen()) return;
      showRatioChartTooltip(best, canvas);
    }, 500));
  });
  canvas?.addEventListener("pointerleave", () => {
    clearRatioChartHover();
  });

  window.addEventListener("pointermove", (e) => {
    const chartCol = getRatioChartCol();
    const scale = getRatioChartScale();
    if (!getRatioChartDragActive() || chartCol == null || !scale) return;
    const rect = canvas.getBoundingClientRect();
    const cy = e.clientY - rect.top;
    const { y0, y1, yMin, yMax, dataMin, dataMax } = scale;
    const clampedY = Math.max(y0, Math.min(y1, cy));
    const t = (y1 - clampedY) / (y1 - y0);
    let value = yMin + t * (yMax - yMin);
    // Clamp threshold to within 20% of data range beyond actual points
    const dataSpan = (dataMax - dataMin) || 1;
    const margin = dataSpan * 0.2;
    value = Math.max(dataMin - margin, Math.min(dataMax + margin, value));
    if (getRatioChartDragTarget() === "upper") {
      const lower = getLowerThresholdValueForCol(chartCol);
      const next = Number.isFinite(lower) ? Math.max(value, lower) : value;
      setThresholdValueForCol(chartCol, next);
    } else if (getRatioChartDragTarget() === "lower") {
      const upper = getThresholdValueForCol(chartCol);
      const next = Number.isFinite(upper) ? Math.min(value, upper) : value;
      setLowerThresholdValueForCol(chartCol, next);
    }
    setRatioChartDragMoved(true);
    scheduleRatioChartRender();
  });

  window.addEventListener("pointerup", () => {
    if (!getRatioChartDragActive()) return;
    setRatioChartDragActive(false);
    const chartCol = getRatioChartCol();
    if (getRatioChartDragMoved() && chartCol != null) {
      const upper = getThresholdValueForCol(chartCol);
      const lower = getLowerThresholdValueForCol(chartCol);
      if (Number.isFinite(upper) || Number.isFinite(lower)) {
        applyThresholdBandExcludes(chartCol, upper, lower);
      }
    }
    setRatioChartHoverLine(null);
    setRatioChartDragTarget(null);
    if (canvas) canvas.style.cursor = "default";
    scheduleRatioChartRender();
  });

  canvas?.addEventListener("click", (e) => {
    if (getRatioChartDragMoved()) {
      setRatioChartDragMoved(false);
      return;
    }
    clearRatioChartHover();
    const chartCol = getRatioChartCol();
    const chartPoints = getRatioChartPoints();
    if (chartCol == null || !chartPoints.length) return;
    const rect = canvas.getBoundingClientRect();
    const cx = e.clientX - rect.left;
    const cy = e.clientY - rect.top;
    let best = null;
    let bestDist = Infinity;
    for (const pt of chartPoints) {
      const ddx = pt.x - cx;
      const ddy = pt.y - cy;
      const dist = ddx * ddx + ddy * ddy;
      if (dist < bestDist) {
        bestDist = dist;
        best = pt;
      }
    }
    const hitRadius = 8;
    if (!best || bestDist > hitRadius * hitRadius) return;
    const key = `${best.rowIndex},${chartCol}`;
    if (ratioStrikeSet.has(key)) {
      ratioStrikeSet.delete(key);
    } else {
      ratioStrikeSet.add(key);
    }
    const cell = document.querySelector(
      `#ratioWrap td.ratioCell[data-r="${best.rowIndex}"][data-col="${chartCol}"]`
    );
    if (cell) cell.classList.toggle("strike", ratioStrikeSet.has(key));
    scheduleRatioSummaryUpdate();
    _onRatioStateMutated();
    scheduleRatioChartRender();
  });
  window.addEventListener("keydown", (e) => {
    if (e.key !== "Escape") return;
    if (isRatioChartOpen()) hideRatioColumnChart();
  });
  window.addEventListener("resize", () => {
    if (isRatioChartOpen()) scheduleRatioChartRender();
  });
}
