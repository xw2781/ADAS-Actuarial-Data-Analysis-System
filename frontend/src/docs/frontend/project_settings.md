# Frontend: Project Settings

## Purpose
<!-- MANUAL:BEGIN -->
Project settings workspace (folders, mappings, dataset types, reserving class types).
Source Data tab derives origin/development date boundary inputs from table summary + field mapping.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN frontend.project_settings.entry_points -->
- `project_settings.html`: external scripts `project_settings.js?v=2026040309`; inline imports _none_.

Detected `fetch(...)` targets in key JS files:
- `/adas/headers/cache/clear`
- `/field_mapping?project_name=${encodeURIComponent(name)}`
- `/general_settings`
- `/general_settings?project_name=${encodeURIComponent(name)}`
- `/project_settings/${DEFAULT_SOURCE}`
- `/project_settings/${DEFAULT_SOURCE}/create_project_folder`
- `/project_settings/${DEFAULT_SOURCE}/delete_project_folder`
- `/project_settings/${DEFAULT_SOURCE}/duplicate_project_folder`
- `/project_settings/${DEFAULT_SOURCE}/folders`
- `/project_settings/${DEFAULT_SOURCE}/open_project_folder`
- `/project_settings/${DEFAULT_SOURCE}/rename_project_folder`
- `/project_settings/${sourceKey}`
- `/project_settings/${sourceKey}/folders`
- `/table_summary/refresh`
- `/table_summary?${q.toString()}`

Detected `adas:*` message types in key JS files:
- `adas:close-active-tab`
- `adas:close-shell-menus`
- `adas:hotkey`
- `adas:open-project`
- `adas:open-workbook`
- `adas:project-settings-ribbon-changed`
- `adas:status`
- `adas:zoom`
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN frontend.project_settings.key_files -->
- [`project_settings.html`](../../project_settings.html) - Project settings workspace and panels.
- [`project_settings.js`](../../project_settings.js) - Project settings coordinator and API calls.
- [`project_settings_field_mapping.js`](../../project_settings_field_mapping.js) - Field mapping feature module.
- [`project_settings_dataset_types.js`](../../project_settings_dataset_types.js) - Dataset types feature module.
- [`project_settings_reserving_class_types.js`](../../project_settings_reserving_class_types.js) - Reserving class types feature module.
- [`project_settings_audit.js`](../../project_settings_audit.js) - Audit log UI helper.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Calls `/project_settings/*`, `/table_summary*`, `/field_mapping`, `/general_settings`, `/adas/headers/cache/clear`, and related endpoints.
- Uses `POST /project_settings/{source}/open_project_folder` from the detail header action to open the selected project's folder in the OS file explorer.
- Folder tree "Create New Project" action calls `POST /project_settings/{source}/create_project_folder` before saving folder structure + settings JSON.
- Dataset Types pane persists changes through `POST /dataset_types` (debounced auto-save).
- Posts title/status events to shell.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Reads/writes settings payloads and folder structures.
- `Project Settings` ribbon page includes an `Open Folder` action button with folder icon styling and disabled-state feedback while the request is in flight.
- Coordinates feature modules for mapping/type editors.
- Dataset Types row mutations (add/edit/delete) update in-memory state and schedule per-project debounced save.
- Dataset Types pane now reuses shared dataset-types helpers (`dataset_types_source.js`, `dataset_types_view_model.js`) for `/dataset_types` payload normalization, Name-search token/match semantics, `Data Format`/`Category`/`Calculated` filter option-building, shared filter label/key generation, and active-filter state checks, to stay aligned with the reusable dataset picker while preserving the existing Project Settings UI behavior.
- Field Mapping `Dataset Type` cells use a modern floating suggestion dropdown and typed entry; typing filters suggestions, deleting all text is allowed, empty input shows all available options, dropdown arrow click forces full-option view regardless of current text, any input text change re-applies filtering, the dropdown always opens below the active input, and only dataset types with empty `Formula` are available for selection.
- Field Mapping `Level` cells auto-fill when `Significances` changes to `Reserving Class` using `max(other Level values) + 1`; level cells also support mouse-wheel +/- integer adjustment with a minimum of `1`.
- After Field Mapping save succeeds, Project Settings automatically saves Dataset Types for the same project so `dataset_types.json` `Source` values are re-synced from rows where Field Mapping `Significance = Dataset` (`Field Name` mapped into each Dataset Type source chain).
- Dataset Types editor blocks enabling `Calculated` when the dataset type name is already used by a field in Field Mapping for the same project.
- When the `Dataset Types` ribbon is active, shell `File` actions map to local export/import (`Save Dataset Types` / `Load Dataset Types`) using default folder `Documents\\ArcRho\\templates`; local load accepts both `.json` and `.xlsx` files.
- When the `Reserving Class Types` ribbon is active, shell `File` actions replace default Save/Save As with local export/import (`Save Reserving Class Types As...` / `Load Reserving Class Types From...`) using default folder `Documents\\ArcRho\\templates`; local load accepts both `.json` and `.xlsx` files and then schedules normal project auto-save.
- Local Dataset Types import uses an in-page custom dialog to choose merge vs overwrite (instead of browser `confirm`), except when imported rows are exactly identical to current UI rows (no prompt, no-op with status message). Merge keeps current behavior (update/add imported rows while preserving mapped-name rows). Overwrite removes existing rows not used by Field Mapping, then loads imported rows, and keeps mapped-name rows only when missing from imported content. For `.xlsx` local load, backend converts workbook data to the same JSON payload structure before merge/overwrite is applied.
- Dataset Types save validation allows calculated formulas to save when their formula components are Dataset Type `Name` values, even if field-mapping source resolution is incomplete.
- Dataset Types table renders grouped by `Category`, supports Name-header keyword search (space-delimited terms) against the `Name` column, supports multi-select dropdown filtering on `Data Format`, `Category`, and `Calculated` headers (with `Calculated` options shown as `Yes`/`No`), and with no options selected a column filter is treated as not applied (all rows remain visible for that column), does not auto-resize column widths during filter changes, keeps header filter icons right-aligned near each column's right border, uses single-triangle sort indicators (`U+25B2`/`U+25BC`) in headers to match Reserving Class Types, allows per-column sort toggles that apply within each category group, and provides left-side subgroup header buttons to collapse/expand each category.
- Dataset Types error status shows an underlined `see more` action that opens a floating details window; details are formatted one error per line.
- Dataset Types table right-click menu includes `Copy` before `Edit`; `Copy` copies the clicked cell's displayed value to the clipboard, including `TRUE` / `FALSE` for the `Calculated` checkbox column, and uses compact row spacing like the reserving class types menu.
- Reserving Class Types save writes `reserving_class_types.json` and a same-folder mirror workbook `reserving_class_types.xlsx`.
- Reserving Class Types source-derived rows are now generated per `(Name, Level)` pair, so duplicate names across levels (for example `PA` at level 2 and level 3) appear in each corresponding level group.
- Reserving Class Types `Source` output always quotes each component separately (for example `"All States"` for one component, `"A" + "B"` for multi-component formulas), and quoted `Formula` / `EEX Formula` tokens remain atomic so operator auto-formatting does not insert spaces inside quoted names such as `"eSales - Teachers"` or `"Affinity/Referral Partners"`.
- Reserving Class Types formula validation is scoped to the row currently being edited: `Apply` checks only that row's `Formula` / `EEX Formula`, and project save/autosave still proceeds even if untouched legacy rows contain invalid references; formulas may reference any existing reserving class type name in the current table, including user-defined rows, but any referenced name containing `+`, `-`, `*`, or `/` must be wrapped in double quotes. When the edited row references a missing name or leaves an operator-bearing name unquoted, the editor shows a floating tooltip above the relevant formula input with a check-spacing/check-spelling hint instead of opening a separate popup window.
- Reserving Class Types table right-click menu includes `Copy` before `Edit`; `Copy` copies the exact displayed text from the clicked cell to the clipboard.
- Project Settings page uses the same global WebKit scrollbar visual style as `dataset_shared.css`.
- Dataset Types header auto-fit measures full header content (label plus sort/filter controls); all Dataset Types headers are kept single-line.
- Source Data date inputs are editable and saved per-project to `general_settings.json`.
- Source Data date inputs are normalized to plain integer strings (no commas, no trailing `.0`/`.00`) in UI and persisted JSON.
- Source Data date inputs display as `MMM YYYY` in UI, while persisted values remain canonical `YYYYMM`.
- Each Source Data date input has inline up/down month steppers to increment/decrement by one month.
- Single-click on `MMM` or `YYYY` highlights that segment; steppers and mouse wheel only adjust the highlighted segment (`MMM` = +/- month, `YYYY` = +/- year).
- Source Data date row uses compact label+input grouping so each label stays visually attached to its date control.
- `general_settings.json` stores `auto_generated`; derived writes set it `true`, user edits set it `false`.
- When `auto_generated` is `false`, table reload will not overwrite the 3 date values unless `project_name` in JSON mismatches the project folder name (stale duplicated settings).
- Source Data date inputs auto-derive from table summary + field mapping when values are missing, stale mismatch is detected, or reload is requested while `auto_generated=true`.
- Source Data table reload clears project-level `ADASHeaders*.csv` cache files under the project `data` folder before refreshing table summary.
- Folder-node context menu supports `Create New Project`, prompts for a project name, creates an empty project folder (with `data` subfolder) via backend, then persists folder-tree mapping + blank project row with rollback on intermediate failures.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Add settings source behavior: update source key logic + endpoint calls.
2. Update one feature pane: modify corresponding `project_settings_*` module.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Folder rename/duplicate/delete/create-project flows have rollback branches.
- Large settings payload edits can impact response timing.
- Reserving class type formula auto-formatting preserves quoted text verbatim; only operators outside quotes are normalized, so quoted names can safely contain `/`, `+`, `-`, `*`, `/`, and repeated spaces. Validation is strict for quoted components: the text inside `"..."` must match an existing reserving class type name exactly, including repeated spaces.
<!-- MANUAL:END -->


