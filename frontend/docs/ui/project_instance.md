# Frontend: Project Instance

## Purpose
<!-- MANUAL:BEGIN -->
Project instance workspace for browsing one project's reserving-class paths and dataset types.
<!-- MANUAL:END -->

## Entry Points
<!-- AUTO-GEN:BEGIN frontend.project_instance.entry_points -->
- `ui/project_instance/project_instance.html`: external scripts `/ui/project_instance/project_instance.js?v=20260517a`; inline imports _none_.
<!-- AUTO-GEN:END -->

## Key Files
<!-- AUTO-GEN:BEGIN frontend.project_instance.key_files -->
- [`ui/project_instance/project_instance.html`](../../ui/project_instance/project_instance.html) - Project instance tab layout.
- [`ui/project_instance/project_instance.js`](../../ui/project_instance/project_instance.js) - Project instance path selector, dataset table, and in-tab dataset viewer windows.
- [`ui/dataset/dataset_viewer.html`](../../ui/dataset/dataset_viewer.html) - Reused dataset viewer page for floating dataset windows.
- [`ui/dataset/dataset_types_source.js`](../../ui/dataset/dataset_types_source.js) - Shared dataset type payload loader and normalizer.
- [`ui/shared/path_tree_picker.js`](../../ui/shared/path_tree_picker.js) - Shared path tree data builder used for reserving-class hierarchy rendering.
<!-- AUTO-GEN:END -->

## External Interfaces
<!-- MANUAL:BEGIN -->
- Opened by shell as a `project_instance` iframe tab after Project Settings posts `arcrho:open-project-instance`.
- Calls shared dataset-types and reserving-class path endpoints through existing frontend helpers.
- Loads the reserving-class path selector from `/reserving_class_path_tree` so server-provided hierarchy, level labels, and cached tree structure are preserved.
- Embeds the existing Dataset Viewer page in draggable in-tab windows.
<!-- MANUAL:END -->

## Data/State/Caches
<!-- MANUAL:BEGIN -->
- Uses the shell-persisted project name/folder/table path as tab inputs.
- Keeps the selected reserving-class path in page memory and passes it into new dataset viewer windows.
- Left and right panels own their scroll areas so overflowing path trees and dataset tables scroll inside the project instance tab frame.
<!-- MANUAL:END -->

## Common Change Tasks
<!-- MANUAL:BEGIN -->
1. Change project instance launch behavior: update Project Settings sender and shell message/tab routing together.
2. Change dataset-window behavior: update `project_instance.js` while preserving the reused Dataset Viewer page contract.
<!-- MANUAL:END -->

## Known Risks
<!-- MANUAL:BEGIN -->
- Nested dataset iframes post messages to the project instance page before reaching the shell.
- Dataset viewer query parameters must remain compatible with normal top-level dataset tabs.
<!-- MANUAL:END -->
