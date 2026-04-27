# Frontend Index

## Purpose
<!-- MANUAL:BEGIN -->
Frontend module map for page entrypoints, shell orchestration, and feature-specific scripts.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN frontend.index.entry_points -->
| HTML Entrypoint | External Scripts | Inline Imports |
| --- | --- | --- |
| `ui/index.html` | `/ui/shell/ui_shell.js?v=20260302` | - |
| `ui/dataset/dataset_viewer.html` | - | `/ui/dataset/dataset_main.js?v=2026022002`, `/ui/dataset/dataset_shared.js` |
| `ui/dfm/dfm.html` | - | `/ui/dataset/dataset_main.js?v=2026022002`, `/ui/dfm/dfm_main.js?v=20260129165558` |
| `ui/workflow/workflow.html` | `/ui/workflow/workflow_main.js?v=20260227` | - |
| `ui/project_settings/project_settings.html` | `/ui/project_settings/project_settings.js?v=2026040309` | - |
| `ui/shell/popout_shell.html` | - | `/ui/shell/popout_bridge.js` |
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN frontend.index.key_files -->
- [`docs/frontend/shell.md`](shell.md) - Shell tab host index.
- [`docs/frontend/dataset.md`](dataset.md) - Dataset feature index.
- [`docs/frontend/dfm.md`](dfm.md) - DFM feature index.
- [`docs/frontend/workflow.md`](workflow.md) - Workflow feature index.
- [`docs/frontend/project_settings.md`](project_settings.md) - Project settings feature index.
- [`docs/frontend/popout.md`](popout.md) - Pop-out window feature index.
<!-- AUTO-GEN:END -->

## Non-Negotiable Contracts
<!-- MANUAL:BEGIN -->
Mandatory before frontend behavior changes:
1. [`../contracts/frontend_behavior_contract.md`](../contracts/frontend_behavior_contract.md)
2. [`../contracts/business_logic_contract.md`](../contracts/business_logic_contract.md)
3. [`../architecture/architecture_guardrails.md`](../architecture/architecture_guardrails.md)

High-risk files that must follow contracts:
- `ui/shell/ui_shell.js`
- `ui/workflow/workflow_main.js`
- `ui/dataset/dataset_main.js`
- `ui/dfm/dfm.html` and `ui/dfm/dfm_*.js`
- `ui/project_settings/project_settings.js`
- `ui/shell/popout_shell.html`
<!-- MANUAL:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- App-server HTTP interface via `fetch(...)` calls.
- Cross-iframe messaging via `window.postMessage` (`adas:*` message types).
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Shell/tab state persisted in browser storage (`localStorage`, IndexedDB handles DB).
- Per-page state lives in each iframe module.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Shell tab lifecycle change -> [`shell.md`](shell.md).
2. Dataset behavior change -> [`dataset.md`](dataset.md).
3. DFM behavior change -> [`dfm.md`](dfm.md).
4. Workflow editor change -> [`workflow.md`](workflow.md).
5. Project settings flow change -> [`project_settings.md`](project_settings.md).
6. Pop-out/dock behavior change -> [`popout.md`](popout.md).
7. Scripting console shortcut/cell-run behavior change -> `ui/scripting_console/scripting_console_core.js`, `ui/scripting_console/scripting_console_cells.js`, `ui/scripting_console/scripting_console_execution.js`, `ui/scripting_console/scripting_console_shortcuts.js`, `ui/scripting_console/scripting_console_panels.js`, `ui/scripting_console/scripting_console_notebook_io.js`, `ui/scripting_console/scripting_console.js` (bootstrap), and `ui/scripting_console/scripting_console.html` (`Shift+Enter` run+advance, `Ctrl+Enter` and `Ctrl+Space` run in place by default, run shortcuts target the currently selected cell in both edit mode and command mode, and keyboard parsing falls back from `event.key` to `event.code` to avoid editor/input-method conflicts (including `Ctrl+Space` cases); for IME/composition scenarios, Ctrl/Meta-modified shortcuts are still parsed while plain composing keystrokes remain ignored; in command mode, `Z` undoes notebook cell content/order state and `Shift+Z` redoes (undo/redo history keeps up to 10 steps), `A` adds a new code cell before current, `B` adds a new code cell after current, `C` copies current cell, `V` pastes a buffered cell after current, `X` cuts current cell (ready for paste), `D` then `D` within 1s deletes current cell when not typing, and `Ctrl+Shift+L` toggles line numbers for code cells; users can customize bindings via the `Keyboard Shortcuts` floating window, and in Electron these bindings persist to `%APPDATA%/ArcRho/WebUI/prefs/scripting_shortcuts.json` (with localStorage as fallback), with host bridge discovery across current frame plus parent/top frames for iframe contexts, automatic APPDATA file bootstrap when missing, and explicit status feedback if APPDATA save fails; new cell type is chosen from toolbar dropdown `Code/Markdown/Raw`, cells are persisted with type metadata, markdown cells render on run like notebook markdown preview and collapse input (double-click rendered markdown to reopen edit mode), and newly added cells stay in command mode (do not auto-enter edit mode); at most one cell can be in edit mode at a time, and edit mode exits with `Esc` or clicking outside a cell editor (pressing `Esc` in markdown edit mode renders markdown immediately before exiting); each console window now sends a session header and app-server execution state is isolated per window session to avoid cross-window variable pollution; notebook Save/Open uses `.ipynb` format and import shows clear "unsupported rich output" notes for types not yet rendered in the UI, e.g., image outputs; the sidebar defaults to the left side, starts wider (290px) with a 210px minimum width, and is split into top/bottom panes (TOC and Variables) with a draggable horizontal splitter, per-pane collapse/expand toggles in each panel header (collapsed keeps the same header bar style as expanded and keeps the +/- control right-aligned), and header right-click actions to move a pane to Top or Bottom; inside Variables, the Variables/API Reference split has its own draggable horizontal splitter, and API Reference has a +/- collapse toggle in its header (collapsed keeps title bar only); default scripting notebook save directory is `~/Documents/ArcRho/scripts`; when the active tab is scripting, File -> Save / Save As routes to notebook save commands, first Save auto-opens Save As prompt, later Save overwrites current notebook directly, save outcomes are mirrored to shell status bar via `adas:status`, code-cell stdout/stderr streams live during execution via `/scripting/run-stream`, while a code cell is running its run button switches to a larger, slightly thicker circular spinner loading icon with explicit 1:1 sizing and continuous rotation, Stop interrupt cancels active cells including `time.sleep(...)` imports, `View -> Toggle Line Numbers` is shown only for active scripting tabs and dispatches `adas:scripting-toggle-line-numbers`; `Edit -> Render All Markdown Cells` is shown only for active scripting tabs and dispatches `adas:scripting-render-all-markdown`; opening a `.ipynb` notebook auto-renders all markdown cells once, and Restart session auto-renders all markdown cells; the in-page Save button is removed, and the toolbar title reflects current notebook name or `Untitled Notebook`; top menubar `Scripting` entry is removed, and Home card `Scripting` (description: "Write code in a notebook.") opens a new scripting tab with fresh notebook content (does not restore another notebook's last unsaved cells), while per-tab draft restoration uses tab-scoped storage keys).
8. Scripting console cell ordering behavior change -> `ui/scripting_console/scripting_console_core.js`, `ui/scripting_console/scripting_console_cells.js`, `ui/scripting_console/scripting_console_execution.js`, `ui/scripting_console/scripting_console_shortcuts.js`, `ui/scripting_console/scripting_console_panels.js`, `ui/scripting_console/scripting_console_notebook_io.js`, `ui/scripting_console/scripting_console.js` (bootstrap), and `ui/scripting_console/scripting_console.html` (cell top titlebar is removed; each cell uses a compact left-side panel next to the editor line-number area showing execution label `[...]` anchored at the bottom and horizontally centered; for code cells, content is split into a top input row (left side panel + code editor) and a bottom output row (left spacer placeholder matching side-panel width + output content, with shared width token and box-sizing alignment), while markdown/raw cells keep the original single-row shell layout (left side panel + right body with input/output stacked), and drag/reorder still treats the cell as one unit; markdown and raw cells no longer show type text labels in the side panel; markdown cells hide side-panel `[...]` label, and raw cells hide both run button and execution label in the side panel; for code cells, empty execution label `[ ]` is rendered with a smaller size; markdown cells also hide run button after markdown has been rendered, and rendered markdown enforces a minimum visible row height based on one H1 line plus output padding, with an additional side-panel minimum when fold controls/badge are present to prevent overlap; the prior 4-dot grip icon and side-panel `X` button are removed, run button is at the top of the side panel, run button is shown on cell hover and always visible for the active cell, and clicking a runnable cell's run button while another cell is executing queues that clicked cell to run next; toolbar cell-type dropdown is bound to the active cell, and selecting `Code/Markdown/Raw` in the dropdown immediately switches the active cell type; command-mode mouse multi-selection is supported (`Ctrl+Click` toggles individual cells and `Shift+Click` selects the ordered range between anchor and clicked cell), and a plain click on any selected cell collapses multi-selection back to only the clicked cell; when multiple cells are selected, clicking a selected cell run button queues selected runnable cells from that clicked cell through the last selected cell in notebook order, while pressing `Ctrl+Space` in command mode queues all selected runnable cells in notebook order; queued cells show `[*]` while waiting, and after the queue completes the pre-run multi-cell selection/focus range is restored; when a collapsed markdown heading cell is selected, its hidden descendant cells are auto-included in selection; multi-cell drag from the left panel is supported for selected cells only when the selection forms one continuous block in notebook order (non-contiguous selection blocks dragging); dragging remains reorder-by-insertion-point only (before first / between cells / after last), and the old "Drag to move cell" tooltip is removed; cell chrome now reserves left accent-strip width up front (transparent by default) so the error accent strip does not shift content horizontally when execution state changes; the error-state left accent color is softened (lower contrast); block in-cell drop/paste effects while dragging; focused command-mode cell highlight uses a stronger blue border without glow (editing mode keeps glow) for clearer selection state; rendered markdown heading cells now show larger SVG triangle fold icons in the side panel (expanded points down, collapsed points right), and collapsing an n-level heading hides all descendant cell types (including deeper n+1/n+2 headings and their content) until the next heading at level <= n, with a circular numeric badge showing how many code cells are hidden by that collapsed section and a badge tooltip in the format "<n> cells hidden"; TOC headers keep label-click navigation only (no collapse action on label text), while the fold control beside each label toggles the same section-collapse state and collapses/expands descendant TOC headers in sync with workspace content; when a cell is running, TOC highlights its mapped header row and shows a right-aligned spinner indicator; active code-line border is shown only while that code cell is in edit mode, and Monaco suggest/autocomplete popups are no longer clipped by cell borders during editing; when a code cell exits edit mode, editor text selection/occurrence highlights are cleared/disabled until edit mode is entered again).
9. Scripting console visual theme polish -> `ui/scripting_console/scripting_console.html`, `ui/scripting_console/scripting_console.js`, and `ui/scripting_console/scripting_console_cells.js` (cells workspace background panel behind cell cards can be themed via `--sc-cells-panel-bg`; current notebook workspace uses a deeper gray base panel while keeping individual cell cards styled independently; the toolbar cell-type selector is custom-rendered instead of native OS dropdown so menu corner radius and option spacing are style-controlled, while existing active-cell type binding behavior remains intact; selector/menu width is slightly narrower, popup corner radius is set to `3px`, and selector text/caret alignment is explicit with flex layout: label left-aligned and caret right-aligned).
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Shell/iframe messaging changes can break hotkeys and dirty-state sync.
- Endpoint path changes in JS can silently break page-level features.
<!-- MANUAL:END -->

















