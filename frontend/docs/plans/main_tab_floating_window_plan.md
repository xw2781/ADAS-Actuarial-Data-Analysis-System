# Main Tab Floating Window Plan

Version: v0.1
Updated: 2026-04-29

---

## 1. Feature Goal

Allow a user to drag a top-level shell tab downward out of the main tab strip and convert that tab into a movable floating window inside the ArcRho app shell.

The floating window is not an Electron/OS pop-out window. It stays inside the current shell DOM, preserves the tab's existing iframe/session, and continues to use the same shell-level save, close, dirty-state, zoom, menu, and `arcrho:*` message contracts.

Core goals:
1. Drag a non-Home main tab downward from the tab strip to float it over the shell content area.
2. Preserve the tab iframe instance; do not reload feature pages during float/dock transitions.
3. Keep dirty-state prompts and close behavior identical to docked tabs.
4. Keep exactly one active tab across the whole app, whether that tab is docked or floating.
5. Remove floated tabs from the main tab strip and show each floated tab in its own shell-owned title bar using the tab name.
6. Let users move, resize, focus, close, and dock the floating window back to the tab strip.
7. Keep the implementation in `ui/shell/ui_shell.js` as the shell orchestration layer.

---

## 2. Current Shell Context

Relevant files:
1. `ui/index.html`
   - Defines the top-level `#tabs` strip, `#content` host, tab context menu, and shell CSS.
2. `ui/shell/ui_shell.js`
   - Owns `state.tabs`, `state.activeId`, tab creation, close confirmation, render flow, iframe creation, and pointer-based horizontal tab reorder.
3. `docs/frontend/shell.md`
   - Documents shell behavior and currently states detached tab pop-out windows are not supported.

Current behavior to preserve:
1. `Home` is fixed and cannot be closed or reordered like feature tabs.
2. Feature pages live in persistent iframes; switching tabs toggles visibility instead of recreating iframes.
3. Dirty tabs use `tab.isDirty` and close confirmation through `closeTab(...)` / close-other flows.
4. Dataset, DFM, workflow, project settings, browsing history, and scripting tabs all depend on shell `arcrho:*` messages.
5. Horizontal drag in `renderTabs()` reorders tabs using pointer events, placeholder UI, and `commitOrderFromDom()`.

Important constraint:
- Do not revive the removed detached OS-window behavior. This feature is an in-shell floating panel/window only.

---

## 3. Product Behavior

### 3.1 Float Gesture

1. A drag that starts from a non-Home top-level tab keeps the current horizontal reorder behavior while the pointer stays near the tab strip.
2. If vertical downward movement exceeds a float threshold, the drag mode switches from reorder to float preview.
3. On pointer release in float mode, the tab becomes a floating window inside `#content`.
4. The new floating window opens at approximately 80% of the current app content/window size, clamped to usable bounds and minimum dimensions.
5. If the pointer is released before the float threshold, existing reorder behavior remains unchanged.

Suggested thresholds:
1. Reorder threshold: keep the current horizontal threshold (`__DRAG_THRESHOLD_PX`).
2. Float threshold: use a separate vertical threshold, for example `dy >= 36px` and `dy > Math.abs(dx) * 0.75`.
3. Cancellation: `Escape` during drag cancels without changing order or float state.

### 3.2 Floating Window Chrome

Each floating tab renders with a shell-owned frame:
1. A dedicated title bar using the tab title and dirty indicator.
2. Dock button to return the tab to the normal tab host.
3. Close button that calls the same close logic as the tab strip.
4. Resize handle at the lower-right corner.
5. Drag handle on the title bar for moving the floating window.

The iframe/page content is embedded inside the floating frame body. Feature pages should not need changes for the MVP.

Floating tabs are removed from the main tab strip while floating. The floating window title bar becomes the visible tab identity while the tab is floated.

### 3.3 Activation and Focus

1. There must always be exactly one active top-level tab across the whole shell.
2. A floating tab and a docked tab cannot both be active at the same time.
3. Clicking a floating window makes its tab active (`state.activeId = tab.id`) and brings the window to front.
4. Shell menus and hotkeys continue to dispatch to `state.activeId`.
5. When a floating tab becomes active, send the same tab activation message already used by docked tabs:
   - Dataset and browsing history: `arcrho:tab-activated`
   - DFM: `arcrho:dfm-tab-activated`
6. If the active tab is floating, the floating window title bar marks that tab as active. Any visible docked iframe behind it is background content and must not receive active-tab menu/hotkey dispatch.
7. If the currently active floating tab closes or docks, use the same fallback selection rules as `closeTab(...)` so the shell immediately has one active tab.

### 3.4 Docking Back

1. Dock button immediately converts the floating tab back to docked mode.
2. Double-clicking the floating window title bar immediately docks the tab back into the normal tab host.
3. Dragging a floating window title bar back over the tab strip can be a Phase 2 enhancement.
4. Docking preserves iframe identity, dirty state, tab title, internal feature state, and local storage instance IDs.
5. Once docked, the tab reappears in the main tab strip in its saved order.

### 3.5 Persistence

Persist floating layout in the shell state:
1. `tab.layout`: `"docked"` or `"floating"`.
2. `tab.floatRect`: `{ x, y, width, height }` in shell content coordinates.
3. `tab.floatZ`: stacking order.
4. Optional future fields: `minimized`, `maximized`.

Use a new shell storage key version because persisted tab shape changes, for example `arcrho_ui_shell_state_v4`.

---

## 4. Technical Design

### 4.1 State Model

Extend each tab object with runtime and persisted layout fields:

```js
{
  id,
  title,
  type,
  iframe,
  isDirty,
  layout: "docked",
  floatRect: null,
  floatZ: 0
}
```

Migration rules:
1. Missing `layout` means `"docked"`.
2. `Home` is always docked.
3. Invalid or offscreen `floatRect` values are clamped to the visible `#content` bounds on load and resize.

### 4.2 DOM Structure

Add a shell-managed floating host under `#content`:

```html
<div id="floatingHost"></div>
```

The host can be created from `ensureContentContainers()` next to `homeView` and `iframeHost` to avoid changing static markup heavily.

For a floating tab, render:

```html
<section class="floatingTabWindow" data-tab-id="...">
  <header class="floatingTabTitlebar">
    <span class="floatingTabTitle">...</span>
    <button data-action="dock">Dock</button>
    <button data-action="close">Close</button>
  </header>
  <div class="floatingTabBody"></div>
  <div class="floatingTabResizeHandle"></div>
</section>
```

Move the tab's existing iframe into `.floatingTabBody` when floating. Move it back to `iframeHost` when docked.

Initial float sizing:
1. Compute the first `floatRect` from the current shell content bounds.
2. Use approximately 80% of the available width and height.
3. Center the window near the drop point where possible, then clamp it so the titlebar and resize handle remain reachable.
4. Enforce minimum dimensions after the 80% calculation.

### 4.3 Render Flow Changes

Split shell content rendering into three coordinated responsibilities:
1. `renderTabs()`
   - Render only docked main tabs in the tab strip.
   - Preserve floating tabs in `state.tabs` for ordering, close fallback, persistence, and docking.
   - When a floating tab docks, render it back into the tab strip at its saved order position.
2. `renderContent()`
   - Continue showing docked active tab content in `iframeHost`.
   - Skip floating iframes when hiding/showing docked iframe content.
   - Ensure `floatingHost` remains visible above content.
3. New `renderFloatingWindows()`
   - Creates/removes floating frames.
   - Moves iframes between `iframeHost` and floating bodies as needed.
   - Applies `floatRect`, z-index, dirty marker, and active state.

Do not replace iframe nodes during this flow. Reparent the same iframe element.

### 4.4 Drag Mode Integration

The current tab drag code should gain an explicit drag mode:

```js
let __tabDragMode = null; // "pending" | "reorder" | "float"
```

Pointer move rules:
1. `pending`: choose `reorder` when horizontal threshold wins; choose `float` when downward threshold wins.
2. `reorder`: keep existing placeholder/indicator behavior.
3. `float`: hide reorder placeholder, show a floating-window preview, and track pointer position in content coordinates.

Pointer up rules:
1. `reorder`: call `commitOrderFromDom()` and save state.
2. `float`: call `floatTab(tabId, initialRect)` and save state.
3. `pending`: treat as a click.

Cleanup must restore pointer capture, tab host layout, cursor, placeholders, and preview UI for both drag paths.

### 4.5 Floating Window Move and Resize

Add pointer handlers on shell-owned floating chrome:
1. Titlebar pointer drag updates `tab.floatRect.x/y`.
2. Resize handle pointer drag updates `tab.floatRect.width/height`.
3. Clamp minimum size, for example `360x240`.
4. Clamp final position so the titlebar remains reachable.
5. Save layout after pointer up, not on every pointer move.
6. Double-click on the titlebar calls the same `dockTab(tab.id)` path as the Dock button.

Use requestAnimationFrame or direct style updates during drag, then write state once at the end.

### 4.6 Menu and Hotkey Dispatch

Since shell menus already dispatch based on `state.activeId`, the main rule is:
1. Clicking/focusing a floating window must update `state.activeId`.
2. Closing, saving, printing, refreshing, and feature-specific menu actions should continue to use existing active-tab dispatch helpers.
3. Refresh should reload only the active tab's iframe regardless of docked/floating layout.

Review these shell functions during implementation:
1. `setActive(...)`
2. `renderContent()`
3. `printActiveTab()`
4. `refreshActiveTab()`
5. menu update helpers
6. active-tab postMessage helpers

---

## 5. Implementation Phases

### Phase 0: Locked Semantics

Locked decisions:
1. Background behind an active floating tab: keep the last docked tab visible as inactive background.
2. Floating tabs in main tab strip: remove them from the main tab strip while floating; each floating window has its own title bar based on the tab name.
3. Multiple floating windows: allow multiple floating tabs at once.
4. Floating-tab entry behavior: because floating tabs are removed from the main tab strip, there is no tab-strip entry to click while floated. If a future floating-window list/menu is added, selecting its entry should focus and raise the floating window rather than dock it.
5. Persistence: persist floating/docked state, position, and size.
6. Initial size: use approximately 80% of the `#content` area.
7. Drag-to-dock: not MVP; use a Dock button and titlebar double-click first.

### Phase 1: Shell State and Rendering

1. Bump shell storage key to a new version.
2. Load/save `layout`, `floatRect`, and `floatZ`.
3. Create `floatingHost`.
4. Add `renderFloatingWindows()`.
5. Reparent existing iframes without reload.
6. Update `renderTabs()` so floating tabs are omitted from the main tab strip.
7. Add CSS for floating frames, active title bars, dirty marker, and resize handle.

### Phase 2: Tab Drag Down to Float

1. Add explicit tab drag modes.
2. Preserve current horizontal reorder behavior.
3. Add downward float threshold and preview.
4. Convert tab to floating on pointer release with an initial rect sized to about 80% of the current app window/content area.
5. Ensure cleanup works for pointer cancel, `Escape`, fast clicks, and close button clicks.

### Phase 3: Floating Window Controls

1. Implement focus/raise on click.
2. Implement titlebar move.
3. Implement resize handle.
4. Implement Dock button.
5. Implement titlebar double-click to dock.
6. Implement Close button through existing `closeTab(...)`.
7. Clamp windows on app resize.

### Phase 4: Docs and Release Notes

1. Update `docs/frontend/shell.md` MANUAL sections:
   - Shell purpose/behavior
   - External interfaces if activation behavior changes
   - Data/state/caches for floating layout persistence
   - Common change tasks and risks
2. Add an unreleased release fragment under `changes/unreleased/`.
3. Run:
   - `python tools/docs_index_builder.py --write`
   - `python tools/docs_index_builder.py --check`

---

## 6. Acceptance Criteria

Functional:
1. Dragging a non-Home main tab downward creates an in-shell floating window.
2. The initial floating window is sized to roughly 80% of the current app shell/content area.
3. The floated tab is removed from the main tab strip and represented by its floating window title bar.
4. Horizontal tab drag reorder still works for docked tabs.
5. The floated tab's iframe does not reload during float/dock.
6. Exactly one top-level tab is active at all times across docked and floating tabs.
7. Shell menu actions target the active floating tab after the window is clicked.
8. Dirty indicators and close confirmation behave the same for docked and floating tabs.
9. Docking through the Dock button or titlebar double-click returns the same tab/iframe to normal docked rendering and re-adds the tab to the main tab strip.
10. Floating window position and size persist across shell reload.

Regression:
1. Home cannot float.
2. Close, close others, and close all remove floating iframes cleanly.
3. Dataset, DFM, workflow, project settings, browsing history, and scripting tabs can all float.
4. Dataset and DFM activation messages still fire when their floating windows are focused.
5. Refresh and print work for active floating tabs.
6. Multiple floating windows maintain predictable z-order.

Manual QA:
1. Open Dataset, DFM, Workflow, Project Explorer, Browsing History, and Scripting tabs.
2. Reorder tabs horizontally and confirm no regressions.
3. Float each tab type and verify content remains live.
4. Confirm each newly floated tab starts at approximately 80% of the app shell/content area.
5. Confirm floated tabs disappear from the main tab strip and show their tab names in floating title bars.
6. Click between docked and floating tabs and confirm only one tab/window is marked active.
7. Double-click a floating titlebar and confirm the tab docks back without reload.
8. Mark Workflow and DFM tabs dirty, float them, close them, and confirm prompts.
9. Dock floated tabs and confirm content/session state is preserved.
10. Reload the app and confirm floating layout restores within bounds.

---

## 7. Risks and Mitigations

Risk 1: iframe reload or state loss during reparenting.
- Mitigation: move the existing iframe node; never recreate it as part of float/dock.

Risk 2: existing render flow hides floating iframes.
- Mitigation: make `renderContent()` distinguish docked and floating iframes before applying `display: none`.

Risk 3: pointer drag conflicts between reorder and float gestures.
- Mitigation: add explicit drag modes and commit to one mode after threshold detection.

Risk 4: dirty-state prompts diverge for floating tabs.
- Mitigation: route floating close actions through `closeTab(...)` and keep `tab.isDirty` as the single shell dirty flag.

Risk 5: shell menus dispatch to the wrong tab.
- Mitigation: focus/raise floating windows through `setActive(tab.id)` before menu dispatch.

Risk 6: windows become unreachable after resize or app reload.
- Mitigation: clamp `floatRect` on load, float, move, resize, and window resize.

Risk 7: docked background content looks active while a floating window is active.
- Mitigation: keep a single `state.activeId`, style only the active docked tab or floating title bar as active, and treat any visible docked iframe behind a floating active tab as inactive background.

---

## 8. Out of Scope for MVP

1. OS-level detached Electron windows.
2. Dragging floating windows between monitors or outside the app shell.
3. Minimize/maximize window controls.
4. Split panes, tiling, docking zones, or layout presets.
5. Floating individual internal feature sub-tabs, such as DFM internal tabs.
6. Cross-window keyboard focus management beyond the active shell tab model.

---

## 9. Open Questions

1. Should a floated tab's original tab-strip order be preserved exactly on dock, or should dock always append it to the end?
   - Recommendation: preserve original `state.tabs` order so docking feels reversible.
2. Should the floating title bar use the same dirty dot location as the tab strip close button?
   - Recommendation: yes, reuse the same dirty semantics visually where possible.
3. Should a floating window be allowed to cover the whole content area after manual resize?
   - Recommendation: yes, as long as its title bar remains reachable and it stays inside shell bounds.
