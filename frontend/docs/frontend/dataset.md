# Frontend: Dataset

## Purpose
<!-- MANUAL:BEGIN -->
Dataset editing/analysis page used inside shell tabs.
Dataset tab strip includes `Details`, `Data`, `Chart`, `Notes`, and `Audit Log`; default active tab remains `Data`.
Dataset runtime internals are split into controller/modules (`dataset_input_controller.js`, `dataset_run_controller.js`, `dataset_dependency_guard.js`, `dataset_headers_service.js`, `dataset_grid_interactions.js`, `dataset_host_bridge.js`, `dataset_notes_editor.js`, `notes_editor_interactions.js`, `dataset_name_picker.js`, `dataset_types_source.js`, `dataset_types_view_model.js`) while keeping existing UI behavior and `arcrho:*`/HTTP contracts unchanged; `notes_editor_interactions.js` contains reusable Notes-editor interaction behavior intended for Dataset and DFM Notes UIs to share while persistence logic remains page-specific, `dataset_name_picker.js` provides a reusable dataset-selection window with caller-supplied filtering options (used by Dataset and DFM pickers) plus header-level `Name Search`, `Data Format`, `Category`, and `Calculated` filter popups, and shared dataset-types data normalization/filter/sort helpers are being extracted into `dataset_types_source.js` and `dataset_types_view_model.js` for reuse across picker and project-settings table features. In the shared picker, long `Name` values wrap within the first column instead of truncating, header column widths reserve space for sort/filter controls so labels remain readable, sort direction indicators use filled SVG triangles to match Project Settings table semantics, and internal column boundaries can be drag-resized from the header grid lines (the dragged left column changes width while the last column auto-compensates to keep total table width constant). The `Calculated` column renders disabled checkboxes in rows (while its filter options remain `Yes` / `No`). Titlebar tools now include a category-group collapse/expand toggle and `Clear all filters` (placed left of picker preferences). The picker opens at an initial height of about two-thirds of the host app viewport height. The picker window's resize max bounds are refreshed when the host app window is resized so enlarging the app window also expands the picker's available resize range; vertical resize max is based on host-window height (not the picker's current top position). During dragging, the picker may move beyond the host window's left/right/bottom edges (top edge remains constrained so the title bar stays reachable). Picker `Data Format`/`Category`/`Calculated` filter popups use `(All)` as the explicit no-filter state (default open state: only `(All)` checked); choosing a specific option clears `(All)`, clicking `(All)` clears specific checks, and right-clicking an option row inverts all non-`(All)` selections.
Dataset `Details` tab uses a two-column form layout with right-aligned labels and left-aligned input controls, and includes a read-only `Formula` field that reflects the current Dataset Type `Formula` value loaded from project `dataset_types`.
Project Name, Reserving Class, and top-row Dataset Type inputs enforce valid-value lists before running ADASTri.
Reserving Class validation uses `reserving_class_types` Name values: a path is valid when each `\`-separated segment exists in the project's reserving class type names, while literal `/` characters inside a class name are preserved and are not treated as path delimiters.
Data tab table overflow is handled by the in-panel `#tableWrap` viewport so oversized tables scroll inside the tab (horizontal and vertical).
Dataset Data-tab sizes `#tableWrap` from viewport-relative values without fixed hard upper caps (still keeping minimum size clamps), so the viewport can grow with larger app windows while preserving in-wrap scrolling.
Dataset and DFM Data tabs share the same `#tableWrap` scrollbar styling from shared/global rules, and `#tableWrap` uses square corners.
Dataset `Notes` tab provides a real notes `textarea` (`#dsNotesInput`) with a raised/3D input frame style, a top formatting toolbar (whole-note display style controls: font family, font size, text color, bold/italic/underline/strikethrough), bottom-right drag resize (`resize: both`) for user-adjustable size, and a bottom action toolbar (left conditional save-state text + right `Save Notes` button) below the textbox (no separate left Notes label); toolbars are width-bound to the resizable notes editor so they stay aligned while dragging. Browser spellcheck is disabled for this field so file paths are not marked by red squiggles. Custom keyboard handling keeps editing flow where `Esc` exits Notes editing (blur), `Tab` inserts 4 spaces, and `Shift+Tab` performs one-level outdent, and click-to-open support for detected Windows absolute file paths and UNC network paths via desktop host bridge (direct call when available, otherwise shell message relay). Detected file paths are always rendered in deep-blue underlined non-bold text, and hover shows an in-place highlight plus a click hint tooltip; when the notes textarea is in active editing state (focused), the editor switches to plain-text rendering (rich/path highlight layer hidden), toolbar interactions do not end editing mode, and clicking a path does not open it, then rich rendering returns after focus leaves both the textarea and toolbar. Notes persistence is manual via `Save Notes`, saved under each project `data` folder as `ADASTriNotes@<reserving class path with "\" -> "^">@<dataset name>.json` (filesystem-unsafe filename characters are replaced with `^`), keyed by raw `reserving class + dataset name` within that project folder (ignoring triangle size parameters), auto-loaded on each dataset load, and unsaved key-switch prompts save-or-discard.
`Clear Cache and Reload` for ADASTri now also clears Dataset header-label caches (browser localStorage and project ADASHeaders CSV cache via `/adas/headers/cache/clear`) and force-refreshes year/development labels before table reload to avoid stale label reuse. Header refresh consumes `/adas/headers` response labels from `labels` (legacy fallbacks: `headers`, `origin_labels`).
Data tab `#tableWrap` frame uses a native square `border` stroke (no outline/pseudo overlay) so frame corners remain stable and the border does not paint over grid cells after table render at non-integer zoom/scaling; the reload button uses the same native border approach.
Data tab `#tableWrap` keeps native smooth scrolling behavior; after scrolling goes idle briefly, the viewport auto-snaps to the nearest row/column grid boundary so sticky headers land on full-cell boundaries instead of partial half-cover offsets.
Data tab `#tableWrap` overlay adds light, square directional arrow buttons (minimal SVG icon style) at scrollbar ends; each click scrolls exactly one grid unit (one row vertically, one data column horizontally), and button enabled/disabled state follows current scroll position.
Data tab column rendering always covers the full `values` matrix width; when development headers are fewer than data columns, missing headers are rendered as blank cells instead of truncating tail data columns.
Data tab table renders a bottom `Total` row that sums each development column; numeric cells are right-aligned in deep-blue regular-weight text, the `Total` label is black regular-weight text, and the body-to-total boundary uses a single thin grid line (no doubled/thick separator). Before rendering `Total`, the page checks current Dataset Type `Formula` from `dataset_types`; when `Formula` contains `*` or `/`, the `Total` row is not rendered, and the last body row keeps its bottom grid border so the bottom grid remains visible.
Dataset table now shows full grid lines by default (blank cells are visible grid cells unless toggled), and masked/blank cells in the triangle body remain clickable/selectable like normal data cells.
When Project Name switches to a different valid project, the page reuses the existing dataset loading popup/spinner and shows `Validating Reserving Class` while refreshing and validating project-scoped reserving class inputs/options.
Reserving Class tree picker highlights only one active node at a time: the deepest visible node on the active path (if deeper levels are collapsed, highlight falls back to the nearest visible ancestor), and uses the selected path text as the picker window title (falls back to `Reserving Class` when no path is selected).
`Link Period Length` is enabled by default on initial open; stored tab preferences still override this default when present.
Top-row `Dataset Type` and Details-tab `Dataset Type` are bi-directionally linked; switching Dataset Type auto-copies Details `Name` to the same value, and initial page load also auto-fills `Name` from the resolved Dataset Type when `Name` is empty (users can still edit `Name` afterward until the next Dataset Type switch).
Dataset Type picker `Picker Preferences` (`Double Click to Select`, `Close Window after Selection`) are user-global preferences saved in APPDATA via `/scripting/preferences` and apply across all projects (not project-scoped).
After each Dataset Type input/switch (including run-time strict validation), the UI first validates dependency resolvability using project `dataset_types` formulas plus direct Dataset mappings from `field_mapping`; if that fails, it runs `/adas/tri/precheck` with the same request-shape/path rules as normal ADASTri requests and allows refresh only when a matching local CSV already exists, otherwise it blocks generation with a missing-dependency popup. In this dependency-fallback path, any clear-cache run request is forcibly downgraded to local-CSV-only load (never sends `/adas/tri/refresh`).
Status text no longer appends Dataset Type `Formula`; after dataset render succeeds, status is `Ready`.
`Origin Length` and `Development Length` use custom dropdown popups styled to match the Dataset Type dropdown while keeping hidden native `select` controls as the value/event source for existing logic.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN frontend.dataset.entry_points -->
- `ui/dataset/dataset_viewer.html`: external scripts _none_; inline imports `/ui/dataset/dataset_main.js?v=2026022002`, `/ui/dataset/dataset_shared.js`.

Detected `fetch(...)` targets in key JS files:
- `${config.API_BASE}/dataset/${dsId}/patch`
- `${config.API_BASE}/dataset/${dsId}?start_year=${encodeURIComponent(startYear)}`
- `${config.API_BASE}/dataset/notes/load`
- `${config.API_BASE}/dataset/notes/save`
- `${config.API_BASE}/excel/active_selection`
- `${config.API_BASE}/excel/open_workbook`
- `${config.API_BASE}/excel/read_cell`
- `${config.API_BASE}/excel/read_cells_batch`
- `${config.API_BASE}/excel/wait_for_enter`
- `/adas/tri/precheck`

Detected `arcrho:*` message types in key JS files:
- `arcrho:browsing-history-updated`
- `arcrho:close-active-tab`
- `arcrho:close-shell-menus`
- `arcrho:dataset-settings-changed`
- `arcrho:hotkey`
- `arcrho:status`
- `arcrho:update-active-tab-title`
- `arcrho:zoom`
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN frontend.dataset.key_files -->
- [`ui/dataset/dataset_viewer.html`](../../ui/dataset/dataset_viewer.html) - Dataset page HTML entrypoint.
- [`ui/dataset/dataset_main.js`](../../ui/dataset/dataset_main.js) - Dataset grid, calculations, and API calls.
- [`ui/dataset/dataset_shared.js`](../../ui/dataset/dataset_shared.js) - Shared dataset markup helpers.
- [`ui/dataset/dataset_shared.css`](../../ui/dataset/dataset_shared.css) - Shared dataset/DFM visual styles.
- [`ui/shared/api.js`](../../ui/shared/api.js) - Client wrappers for dataset endpoints.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Calls app-server dataset/adas endpoints plus valid-value list endpoints (`/dataset_types`, `/reserving_class_*`, `/adas/projects`).
- Uses `/scripting/preferences` to persist and restore the last resolved Project + Reserving Class pair in APPDATA.
- Sends status/hotkey/close signals to parent shell.
- Publishes dataset input updates and browsing-history updates to shell via `arcrho:dataset-settings-changed` and `arcrho:browsing-history-updated`.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Uses in-page mutable state for active dataset and selection.
- Reads and caches valid value lists via `valid_value_list_provider.js` for:
  - project names from project map
  - dataset names by project
  - reserving class paths by project
- Reserving-class path normalization preserves literal `/` characters inside a class name; only `\` is treated as the segment delimiter for validation/history keys.
- Dataset-side reserving path list loading does not auto-crawl `/reserving_class_path_tree/children`; child-path hydration is opt-in to avoid background request storms.
- Caches reserving-class type names from `/reserving_class_types` and validates input paths by segment membership in the Name column.
- Reserving-class tree view toggle preferences (auto-expand/auto-close/double-click) are shared globally across projects.
- On project switch, current Reserving Class input is revalidated against the new project's reserving-class type names; valid paths are retained and invalid paths are cleared.
- Stores last resolved Project + Reserving Class defaults in APPDATA and reuses them as fallback when no scoped/query/workflow values exist.
- Persists last-viewed dataset inputs globally and restores them when opening a new Dataset tab.
- Stores latest browsing history entries via `browsing_history.js` (project + reserving class + dataset).
- Rejects invalid typed values on change/Enter and blocks ADASTri requests until all 3 inputs are valid.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add a new app-server call: update fetch call and API wrappers.
2. Change table behavior: update `dataset_main.js` render + patch flow together.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Formula or patch changes can cause silent data drift.
- Endpoint mismatches break runtime flows without compile-time safety.
<!-- MANUAL:END -->
