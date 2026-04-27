# Frontend: Shell

## Purpose
<!-- MANUAL:BEGIN -->
Shell-level tab/iframe host for all feature pages.
Home view includes a Browsing History entry card that opens a dedicated Browsing History tab.
Top menubar supports mixed-scope actions: global actions are always shown, shared actions remain visible but dispatch based on active tab, and page-exclusive actions are hidden when their target page type is not active (for example, `View -> Hide/Show Navigation Panel` only on Workflow and `View -> Toggle Line Numbers` only on Scripting). Page-exclusive visibility is declarative via `data-page-scopes="<tabType>[,<tabType>...]"` on menu items in `index.html`, and shell-side filtering in `ui_shell.js` applies the scopes on every render/open so DFM/workflow/scripting-only actions do not leak into other page types. `Edit` and `View` dropdowns are marked with `data-requires-page-scope`, so newly added items in those menus stay hidden unless they explicitly declare `data-page-scopes` (use `data-page-scopes="*"` for always-visible entries). When any top menu dropdown is open, clicking in shell or inside an iframe page, or pressing `Esc`, closes all shell dropdown menus.
When Project Settings is active and its `Dataset Types` ribbon is selected, `File -> Save` / `Save As...` are relabeled and dispatched as `Save Dataset Types` / `Load Dataset Types` to the Project Settings iframe while keeping the same keyboard shortcuts. When Project Settings is active and its `Reserving Class Types` ribbon is selected, those same File actions are relabeled/dispatched as `Save Reserving Class Types As...` / `Load Reserving Class Types From...`.
When DFM is active and the internal DFM tab is `Details`, `File -> Save As...` is relabeled to `Save as Template` and dispatches DFM template save; on other DFM internal tabs, `Save As...` keeps normal DFM method Save-As behavior.
Dataset and DFM iframe URLs include a shell `v` token so reopening tabs picks up the latest static page assets instead of stale cached HTML/CSS.
Electron host bridge exposes local path open support; shell relays iframe `adas:open-path` requests to host and replies with `adas:open-path-result` so Dataset Notes click-to-open works from iframe pages.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN frontend.shell.entry_points -->
- `ui/index.html`: external scripts `/ui/shell/ui_shell.js?v=20260302`; inline imports _none_.

Detected `fetch(...)` targets in key JS files:
- `/`
- `/app/restart`
- `/app/restart_electron`
- `/app/shutdown`
- `/restart`
- `/ui_config`
- `/workflow/default_dir`
- `/workflow/load`

Detected `adas:*` message types in key JS files:
- `adas:autosave-toggle`
- `adas:browsing-history-updated`
- `adas:close-active-tab`
- `adas:dfm-tab-activated`
- `adas:force-rebuild-toggle`
- `adas:hotkey`
- `adas:open-path-result`
- `adas:set-app-font`
- `adas:set-zoom`
- `adas:tab-activated`
- `adas:workflow-load`
- `adas:zoom`
- `adas:zoom-reset`
- `adas:zoom-step`
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN frontend.shell.key_files -->
- [`ui/index.html`](../../ui/index.html) - Main desktop shell page and menu structure.
- [`ui/shell/ui_shell.js`](../../ui/shell/ui_shell.js) - Tab orchestration, iframe lifecycle, menus, and hotkeys.
- [`electron_preload.js`](../../electron_preload.js) - Renderer-safe host bridge APIs.
- [`electron_main.js`](../../electron_main.js) - Window lifecycle and shell-to-host wiring.
- [`ui/shell/popout_bridge.js`](../../ui/shell/popout_bridge.js) - BroadcastChannel helper for pop-out tabs.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Communicates with child iframes via `adas:*` postMessage events.
- Invokes backend for workflow import helpers and configuration endpoints.
- Uses Electron host bridge for shutdown/clear-cache actions; backend startup is host-managed with retry on transient launch failures.
- Consumes dataset-page browsing updates (`adas:dataset-settings-changed`, `adas:browsing-history-updated`) and forwards updates to any open Browsing History tab.
- Receives `adas:open-dataset-from-history` from Browsing History tab to open dataset tabs with selected inputs.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Persists tab state, zoom, and toggles in `localStorage`.
- Tracks popped-out tabs via `BroadcastChannel`.
- Persists dataset browsing history entries (latest 15) via `browsing_history.js`.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add a new tab type: update tab creation + iframe source logic in `ui_shell.js`.
2. Add shell menu action: wire menu item + action handler + hotkey map; for `Edit`/`View` items, always set `data-page-scopes` (`*` for global visibility) because those dropdowns require explicit scope declarations.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- DOM replacement in shell can invalidate iframe references.
- Unsaved-state handling must stay consistent for close/close-all flows.
- Host/backend startup races can surface as blank shell or startup timeout if lifecycle flags/process teardown are not coordinated.
<!-- MANUAL:END -->
