# Frontend: Workflow

## Purpose
<!-- MANUAL:BEGIN -->
Workflow editor page and save/load orchestration.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN frontend.workflow.entry_points -->
- `ui/workflow/workflow.html`: external scripts `/ui/workflow/workflow_main.js?v=20260227`; inline imports _none_.

Detected `fetch(...)` targets in key JS files:
- `/adas/projects`
- `/reserving_class_combinations?project_name=${encodeURIComponent(projectName)}`
- `/reserving_class_filter_spec`
- `/reserving_class_filter_spec?project_name=${encodeURIComponent(projectName)}`
- `/reserving_class_hidden_paths`
- `/reserving_class_hidden_paths?project_name=${encodeURIComponent(projectName)}`
- `/reserving_class_types?project_name=${encodeURIComponent(projectName)}`
- `/workflow/save`
- `/workflow/save_as`

Detected `arcrho:*` message types in key JS files:
- `arcrho:close-active-tab`
- `arcrho:close-shell-menus`
- `arcrho:dfm-save`
- `arcrho:get-dataset-settings`
- `arcrho:get-dfm-settings`
- `arcrho:hotkey`
- `arcrho:set-app-font`
- `arcrho:set-zoom`
- `arcrho:tooltip`
- `arcrho:update-workflow-tab-title`
- `arcrho:workflow-dirty`
- `arcrho:workflow-global-changed`
- `arcrho:workflow-import`
- `arcrho:workflow-saved`
- `arcrho:zoom`
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN frontend.workflow.key_files -->
- [`ui/workflow/workflow.html`](../../ui/workflow/workflow.html) - Workflow page layout and containers.
- [`ui/workflow/workflow_main.js`](../../ui/workflow/workflow_main.js) - Workflow editing logic, save/load events.
- [`ui/shared/menu_utils.js`](../../ui/shared/menu_utils.js) - Context menu helper utilities.
- [`ui/shared/reserving_class_lazy_picker.js`](../../ui/shared/reserving_class_lazy_picker.js) - Shared reserving-class tree selector.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Calls `/workflow/*` app-server routes.
- Coordinates with shell and embedded dataset/DFM iframes via message bridge.
- For DFM embeds, preserves optional `outputType` in step settings and forwards it as `output_type` URL param.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Persists workflow tab state using per-instance storage keys.
- Uses imported/exported `.arcwf` payloads.
- Enforces a single `global_control` step per workflow; duplicate instances are blocked or normalized to `picker`.
- Stores DFM step settings snapshots (including optional `outputType`) via `arcrho:get-dfm-settings` / `arcrho:dfm-settings`.
- Reserving-class tree view toggle preferences (auto-expand/auto-close/double-click) are shared globally across projects.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Extend workflow payload: update `workflow_main.js`, app-server schema/service, and save/load compatibility.
2. Add sidebar behavior: update `workflow.html` + resize/collapse handlers.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Save/load compatibility regressions across older workflow files.
- Dirty-state propagation to shell can become inconsistent.
<!-- MANUAL:END -->
